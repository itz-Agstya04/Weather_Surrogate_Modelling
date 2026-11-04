from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pino_data_pipeline import PINONormalizer, get_rollout_loader, get_single_step_loader, load_full_split_tensor
from models.residual import ResidualStep
from train.train_baselines import CoordinateFNO2d, fine_tune_rollout, step_loss, train_one_epoch, validate
from analysis.diagnose_persistence import LABELS, metrics
import xarray as xr
from data.store import open_project_zarr


def load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def raw_predictions(model, loader, normalizer, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inp, tgt in loader:
            predictions.append(normalizer.denormalize(model(inp.to(device)).float()).cpu().numpy())
            targets.append(normalizer.denormalize(tgt.float()).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main():
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = json.loads((ROOT / "DATASET" / "norm_stats.json").read_text(encoding="utf-8"))
    normalizer = PINONormalizer(stats)
    train_path = str(ROOT / "DATASET" / "train_2018_2019.zarr")
    val_path = str(ROOT / "DATASET" / "val_2020.zarr")
    test_path = str(ROOT / "DATASET" / "test_2021_2022.zarr")
    train_loader = get_single_step_loader(train_path, normalizer, batch_size=32, shuffle=True)
    val_loader = get_single_step_loader(val_path, normalizer, batch_size=32, shuffle=False)
    test_loader = get_single_step_loader(test_path, normalizer, batch_size=32, shuffle=False)
    rollout_loader = get_rollout_loader(train_path, normalizer, sequence_length=4, batch_size=8, shuffle=True)
    step_std = torch.ones(13, device=device, dtype=torch.float32)
    model = ResidualStep(CoordinateFNO2d(in_dim=13, out_dim=13, width=32), step_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    best_state = None
    best_val = float("inf")
    no_improve = 0
    epochs_run = 0
    for epoch in range(1, 31):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, max_grad_norm=1.0, step_std=step_std)[0]
        val_loss = validate(model, val_loader, device, step_std=step_std)[0]
        scheduler.step()
        epochs_run = epoch
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f"epoch={epoch},train={train_loss:.9f},val={val_loss:.9f},lr={scheduler.get_last_lr()[0]:.9g},best={int(improved)}", flush=True)
        if no_improve >= 6:
            break
    model.load_state_dict(best_state)
    fine_tune_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    fine_tune_rollout(model, rollout_loader, fine_tune_optimizer, device, epochs=3, step_std=step_std)
    checkpoint = ROOT / "DATASET" / "checkpoints" / "v2" / "fno_residual_hp.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
    residual_pred, truth = raw_predictions(model, test_loader, normalizer, device)
    test_raw = load_full_split_tensor(open_project_zarr(test_path, consolidated=True)).numpy().astype(np.float64)
    train_raw = load_full_split_tensor(open_project_zarr(train_path, consolidated=True)).numpy().astype(np.float64)
    truth = test_raw[1:]
    test_ds = open_project_zarr(test_path, consolidated=True)
    climatology = train_raw.mean(axis=0)
    means = np.asarray(normalizer.channel_means[:, 0, 0], dtype=np.float64)
    stds = np.asarray(normalizer.channel_stds[:, 0, 0], dtype=np.float64)
    old = CoordinateFNO2d(in_dim=13, out_dim=13, width=32).to(device)
    old.load_state_dict(load_state(ROOT / "DATASET" / "checkpoints" / "fno.pt", device), strict=False)
    old_pred, _ = raw_predictions(old, test_loader, normalizer, device)
    table = {
        "Persistence": metrics(test_raw[:-1], truth, climatology, stds, test_ds.latitude.values),
        "Climatology": metrics(np.broadcast_to(climatology, truth.shape), truth, climatology, stds, test_ds.latitude.values),
        "Old FNO": metrics(old_pred, truth, climatology, stds, test_ds.latitude.values),
        "FNO ResidualStep": metrics(residual_pred, truth, climatology, stds, test_ds.latitude.values),
    }
    serial = {}
    for model_name, values in table.items():
        serial[model_name] = {metric_name: [float(x) for x in array] for metric_name, array in values.items()}
    serial["metadata"] = {"device": str(device), "epochs_run": epochs_run, "best_val_loss": best_val, "checkpoint": str(checkpoint)}
    result_path = ROOT / "DATASET" / "phase1b_results.json"
    result_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"results={result_path}", flush=True)
    for model_name, values in table.items():
        print(f"{model_name},aggregate_rmse={values['rmse'].mean():.9f},aggregate_nrmse={values['nrmse'].mean():.9f},aggregate_acc={values['acc'].mean():.9f}", flush=True)
    for model_name, values in table.items():
        for index, label in enumerate(LABELS):
            print(f"{model_name},{label},{values['bias'][index]:.9f},{values['rmse'][index]:.9f},{values['pattern_rmse'][index]:.9f},{values['nrmse'][index]:.9f},{values['acc'][index]:.9f}", flush=True)


if __name__ == "__main__":
    main()
