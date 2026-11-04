import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pino_data_pipeline import (
    PINONormalizer,
    get_single_step_loader,
    get_rollout_loader,
    load_full_split_tensor,
    CHANNEL_NAMES as STATE_CHANNEL_NAMES
)
from models.proposed_model import STATE_CHANNELS

from data.store import open_project_zarr
from train.train_baselines import (
    CoordinateFNO2d,
    calculate_metrics,
    train_one_epoch,
    validate,
    fine_tune_rollout,
    save_checkpoint,
    compute_divergence_loss,
    compute_hydrostatic_loss
)
from models.proposed_model import ProposedModel, IterativeCorrection, load_terrain_artifact


class Refined2DFNO(nn.Module):
    def __init__(self, in_dim=len(STATE_CHANNEL_NAMES), out_dim=len(STATE_CHANNEL_NAMES), width=32, refinement_steps=3, refinement_alpha=0.5):
        super().__init__()
        self.backbone = CoordinateFNO2d(in_dim=in_dim, out_dim=out_dim, width=width)
        self.refinement_steps = refinement_steps
        self.refinement_alpha = refinement_alpha
        self.corrector = IterativeCorrection(channels=out_dim, width=width) if refinement_steps > 1 else nn.Identity()

    def forward(self, x):
        output = self.backbone(x)
        if self.refinement_steps == 1:
            return output
        for _ in range(self.refinement_steps - 1):
            correction = self.corrector(x, output)
            output = output + self.refinement_alpha * correction
        return output


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure_latency_and_memory(model, sample_input, device, warmup=25, runs=100):
    model.eval()
    sample = sample_input.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(sample)
        torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(sample)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
        median_lat = float(np.median(times))
        std_lat = float(np.std(times))
    else:
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(sample)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(sample)
            times.append((time.perf_counter() - t0) * 1000.0)
        peak_mem = 0.0
        median_lat = float(np.median(times))
        std_lat = float(np.std(times))
    return median_lat, std_lat, peak_mem


@torch.no_grad()
def evaluate_detailed(model, test_loader, normalizer, device, climatology, steep_indices, flat_indices):
    model.eval()
    predictions = []
    targets = []
    for inp, tgt in test_loader:
        inp = inp.to(device)
        pred = model(inp)
        predictions.append(normalizer.denormalize(pred.float()).cpu())
        targets.append(normalizer.denormalize(tgt.float()).cpu())

    preds = torch.cat(predictions)
    tgts = torch.cat(targets)
    lats = np.linspace(33.5, 30.0, 15)

    overall = calculate_metrics(preds, tgts, normalizer, lats, climatology=climatology)

    b, c, h, w = preds.shape
    p_flat = preds.view(b, c, -1)
    t_flat = tgts.view(b, c, -1)

    stds = torch.from_numpy(normalizer.channel_stds).view(1, -1, 1)

    sq_steep = (p_flat[:, :, steep_indices] - t_flat[:, :, steep_indices]) ** 2
    sq_flat = (p_flat[:, :, flat_indices] - t_flat[:, :, flat_indices]) ** 2

    steep_nrmse = float((torch.sqrt(sq_steep.mean(dim=(0, 2))) / stds.squeeze()).mean())
    flat_nrmse = float((torch.sqrt(sq_flat.mean(dim=(0, 2))) / stds.squeeze()).mean())
    delta_steep_minus_flat = steep_nrmse - flat_nrmse

    steep_t2m_rmse = float(torch.sqrt(sq_steep[:, 0].mean()))
    flat_t2m_rmse = float(torch.sqrt(sq_flat[:, 0].mean()))
    delta_t2m = steep_t2m_rmse - flat_t2m_rmse

    return {
        "overall": overall,
        "steep_nrmse": steep_nrmse,
        "flat_nrmse": flat_nrmse,
        "delta_nrmse": delta_steep_minus_flat,
        "steep_t2m": steep_t2m_rmse,
        "flat_t2m": flat_t2m_rmse,
        "delta_t2m": delta_t2m
    }


@torch.no_grad()
def evaluate_rollouts(model, test_rollout_loader, normalizer, device, climatology, max_horizon=12):
    model.eval()
    horizons = [6, 12, 24, 48, 72]
    step_indices = [h // 6 - 1 for h in horizons]
    predictions = {h: [] for h in horizons}
    targets = {h: [] for h in horizons}

    for seq in test_rollout_loader:
        seq = seq.to(device)
        current = seq[:, 0]
        for step in range(max_horizon):
            current = model(current)
            h = (step + 1) * 6
            if h in horizons:
                predictions[h].append(normalizer.denormalize(current.float()).cpu())
                targets[h].append(normalizer.denormalize(seq[:, step + 1].float()).cpu())

    lats = np.linspace(33.5, 30.0, 15)
    rollout_metrics = {}
    for h in horizons:
        p = torch.cat(predictions[h])
        t = torch.cat(targets[h])
        m = calculate_metrics(p, t, normalizer, lats, climatology=climatology)
        rollout_metrics[f"{h}h"] = {
            "nrmse": m["aggregate_nrmse"],
            "acc": m["aggregate_acc"],
            "t2m_rmse": m["rmse"]["T2m"],
            "z500_rmse": m["rmse"]["Z500"]
        }
    return rollout_metrics


def train_and_eval_model(model, train_loader, val_loader, test_loader, train_rollout, test_rollout,
                         normalizer, climatology, steep_idx, flat_idx, device, epochs=15, rollout_epochs=2,
                         use_physics=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        tr_loss, _, _, _, _ = train_one_epoch(
            model, train_loader, optimizer, device,
            use_physics_loss=use_physics, lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer
        )
        vl_loss, _ = validate(
            model, val_loader, device,
            use_physics_loss=use_physics, lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer
        )
        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if rollout_epochs > 0:
        ft_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        fine_tune_rollout(model, train_rollout, ft_opt, device, epochs=rollout_epochs)

    sample = next(iter(test_loader))[0]
    lat_med, lat_std, peak_mem = measure_latency_and_memory(model, sample, device)
    eval_res = evaluate_detailed(model, test_loader, normalizer, device, climatology, steep_idx, flat_idx)
    roll_res = evaluate_rollouts(model, test_rollout, normalizer, device, climatology)

    return {
        "eval": eval_res,
        "rollout": roll_res,
        "latency_ms": lat_med,
        "latency_std_ms": lat_std,
        "peak_mem_mb": peak_mem,
        "param_count": sum(p.numel() for p in model.parameters())
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open("DATASET/norm_stats.json") as f:
        stats = json.load(f)
    normalizer = PINONormalizer(stats)

    train_loader = get_single_step_loader("DATASET/train_2018_2019.zarr", normalizer, 32, True)
    val_loader = get_single_step_loader("DATASET/val_2020.zarr", normalizer, 32, False)
    test_loader = get_single_step_loader("DATASET/test_2021_2022.zarr", normalizer, 32, False)

    train_rollout = get_rollout_loader("DATASET/train_2018_2019.zarr", normalizer, 4, 8, True)
    test_rollout = get_rollout_loader("DATASET/test_2021_2022.zarr", normalizer, 13, 8, False)

    climatology = normalizer.denormalize(train_loader.dataset.data.float()).mean(dim=0)

    terrain = np.load("DATASET/terrain_hp.npz")
    steep_idx = terrain["node_flat_indices"]
    all_idx = np.arange(15 * 17)
    flat_idx = np.setdiff1d(all_idx, steep_idx)

    seeds = [42, 43, 44]

    configs = {
        "2D_FNO_R1": {
            "creator": lambda: CoordinateFNO2d(width=32),
            "use_physics": False
        },
        "2D_FNO_R3": {
            "creator": lambda: Refined2DFNO(width=32, refinement_steps=3),
            "use_physics": False
        },
        "Standard_PINO_R1": {
            "creator": lambda: CoordinateFNO2d(width=32),
            "use_physics": True
        },
        "Standard_PINO_R3": {
            "creator": lambda: Refined2DFNO(width=32, refinement_steps=3),
            "use_physics": True
        },
        "FactorizedFNO_NoTerrain_R3": {
            "creator": lambda: ProposedModel(terrain_path="DATASET/terrain_hp.npz", width=32, physics_mode="none", use_gnn_residual=False, refinement_steps=3),
            "use_physics": False,
            "zero_terrain": True
        },
        "FactorizedFNO_WithTerrain_GNNOff_R3": {
            "creator": lambda: ProposedModel(terrain_path="DATASET/terrain_hp.npz", width=32, physics_mode="none", use_gnn_residual=False, refinement_steps=3),
            "use_physics": False
        },
        "FactorizedFNO_SparseGNN_R3": {
            "creator": lambda: ProposedModel(terrain_path="DATASET/terrain_hp.npz", width=32, physics_mode="none", use_gnn_residual=True, gnn_dense=False, refinement_steps=3),
            "use_physics": False
        },
        "FactorizedFNO_DenseGNN_R3": {
            "creator": lambda: ProposedModel(terrain_path="DATASET/terrain_hp.npz", width=32, physics_mode="none", use_gnn_residual=True, gnn_dense=True, refinement_steps=3),
            "use_physics": False
        }
    }

    results = {}

    for cname, cinfo in configs.items():
        results[cname] = []
        for seed in seeds:
            print(f"Running config={cname} seed={seed}", flush=True)
            set_seed(seed)
            model = cinfo["creator"]().to(device)
            if cinfo.get("zero_terrain", False):
                model.terrain_features.zero_()
            res = train_and_eval_model(
                model, train_loader, val_loader, test_loader, train_rollout, test_rollout,
                normalizer, climatology, steep_idx, flat_idx, device,
                epochs=15, rollout_epochs=2, use_physics=cinfo["use_physics"]
            )
            results[cname].append(res)
            print(f"  done seed={seed}: 6h nRMSE={res['eval']['overall']['aggregate_nrmse']:.4f} ACC={res['eval']['overall']['aggregate_acc']:.4f} 72h nRMSE={res['rollout']['72h']['nrmse']:.4f}", flush=True)

    summary = {}
    for cname, seed_results in results.items():
        nrmses = [r["eval"]["overall"]["aggregate_nrmse"] for r in seed_results]
        accs = [r["eval"]["overall"]["aggregate_acc"] for r in seed_results]
        t2ms = [r["eval"]["overall"]["rmse"]["T2m"] for r in seed_results]
        u10s = [r["eval"]["overall"]["rmse"]["U10"] for r in seed_results]
        v10s = [r["eval"]["overall"]["rmse"]["V10"] for r in seed_results]
        z500s = [r["eval"]["overall"]["rmse"]["Z500"] for r in seed_results]
        t500s = [r["eval"]["overall"]["rmse"]["T500"] for r in seed_results]

        r24s = [r["rollout"]["24h"]["nrmse"] for r in seed_results]
        r48s = [r["rollout"]["48h"]["nrmse"] for r in seed_results]
        r72s = [r["rollout"]["72h"]["nrmse"] for r in seed_results]

        steeps = [r["eval"]["steep_nrmse"] for r in seed_results]
        flats = [r["eval"]["flat_nrmse"] for r in seed_results]
        deltas = [r["eval"]["delta_nrmse"] for r in seed_results]

        latencies = [r["latency_ms"] for r in seed_results]
        mems = [r["peak_mem_mb"] for r in seed_results]
        params = seed_results[0]["param_count"]

        summary[cname] = {
            "params": params,
            "latency_median_ms": float(np.mean(latencies)),
            "peak_mem_mb": float(np.mean(mems)),
            "6h_nrmse_mean": float(np.mean(nrmses)),
            "6h_nrmse_std": float(np.std(nrmses)),
            "6h_acc_mean": float(np.mean(accs)),
            "6h_acc_std": float(np.std(accs)),
            "T2m_rmse_mean": float(np.mean(t2ms)),
            "U10_rmse_mean": float(np.mean(u10s)),
            "V10_rmse_mean": float(np.mean(v10s)),
            "Z500_rmse_mean": float(np.mean(z500s)),
            "T500_rmse_mean": float(np.mean(t500s)),
            "24h_nrmse_mean": float(np.mean(r24s)),
            "48h_nrmse_mean": float(np.mean(r48s)),
            "72h_nrmse_mean": float(np.mean(r72s)),
            "steep_nrmse_mean": float(np.mean(steeps)),
            "flat_nrmse_mean": float(np.mean(flats)),
            "delta_steep_minus_flat_mean": float(np.mean(deltas)),
            "delta_steep_minus_flat_std": float(np.std(deltas))
        }

    out_file = "DATASET/rigorous_evaluation_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll experiments complete. Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
