from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from data.store import open_project_zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pino_data_pipeline import (
    SURFACE_VARS,
    TARGET_LEVELS,
    UPPER_AIR_TEMPERATURE_VAR,
    UPPER_AIR_VAR,
    load_full_split_tensor,
)

LABELS = ["T2m", "U10", "V10"] + [f"Z{x}" for x in TARGET_LEVELS] + [f"T{x}" for x in TARGET_LEVELS]
UNITS = ["K", "m/s", "m/s"] + ["m2/s2"] * 5 + ["K"] * 5
REMOTE = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"


def stats_vectors(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    mean = [data[x]["mean"] for x in SURFACE_VARS]
    std = [data[x]["std"] for x in SURFACE_VARS]
    for level in TARGET_LEVELS:
        mean.append(data[UPPER_AIR_VAR][str(level)]["mean"])
        std.append(data[UPPER_AIR_VAR][str(level)]["std"])
    for level in TARGET_LEVELS:
        mean.append(data[UPPER_AIR_TEMPERATURE_VAR][str(level)]["mean"])
        std.append(data[UPPER_AIR_TEMPERATURE_VAR][str(level)]["std"])
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def metrics(pred, truth, climatology, std, latitude):
    weights = np.cos(np.deg2rad(latitude)).astype(np.float64)[None, None, :, None]
    denom = weights.sum() * pred.shape[0] * pred.shape[3]
    error = pred - truth
    rmse = np.sqrt((error ** 2 * weights).sum(axis=(0, 2, 3)) / denom)
    bias = (error * weights).sum(axis=(0, 2, 3)) / denom
    pattern_error = error - error.mean(axis=0, keepdims=True)
    pattern_rmse = np.sqrt((pattern_error ** 2 * weights).sum(axis=(0, 2, 3)) / denom)
    nrmse = rmse / std
    pred_anom = pred - climatology[None]
    truth_anom = truth - climatology[None]
    cov = (pred_anom * truth_anom * weights).sum(axis=(0, 2, 3))
    pred_var = (pred_anom ** 2 * weights).sum(axis=(0, 2, 3))
    truth_var = (truth_anom ** 2 * weights).sum(axis=(0, 2, 3))
    acc = cov / (np.sqrt(pred_var * truth_var) + 1e-12)
    return {"bias": bias, "rmse": rmse, "pattern_rmse": pattern_rmse, "nrmse": nrmse, "acc": acc}


def load_model_predictions(test, means, stds, checkpoint, kind):
    import torch
    from train.train_baselines import CoordinateFNO2d, load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = ((test[:-1] - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)
    if kind == "proposed":
        from models.proposed_model import ProposedModel
        model = ProposedModel(terrain_path=str(ROOT / "DATASET" / "terrain_hp.npz"), physics_mode="soft", use_gnn_residual=False, refinement_steps=3, refinement_alpha=0.5).to(device)
    else:
        model = CoordinateFNO2d(in_dim=13, out_dim=13, width=32).to(device)
    try:
        state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(inputs), 32):
            result.append(model(torch.from_numpy(inputs[start:start + 32]).to(device)).float().cpu().numpy())
    return np.concatenate(result, axis=0) * stds[None, :, None, None] + means[None, :, None, None]


def print_metric_table(table):
    print("model,channel,unit,bias,RMSE,pattern_RMSE,nRMSE,ACC")
    for model, entry in table.items():
        for i, label in enumerate(LABELS):
            print(f"{model},{label},{UNITS[i]},{entry['bias'][i]:.9f},{entry['rmse'][i]:.9f},{entry['pattern_rmse'][i]:.9f},{entry['nrmse'][i]:.9f},{entry['acc'][i]:.9f}")
        print(f"{model},AGGREGATE,,{entry['bias'].mean():.9f},{entry['rmse'].mean():.9f},{entry['pattern_rmse'].mean():.9f},{entry['nrmse'].mean():.9f},{entry['acc'].mean():.9f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--skip-remote", action="store_true")
    args = parser.parse_args()
    train_ds = open_project_zarr(ROOT / "DATASET" / "train_2018_2019.zarr", consolidated=True)
    test_ds = open_project_zarr(ROOT / "DATASET" / "test_2021_2022.zarr", consolidated=True)
    means, stds = stats_vectors(ROOT / "DATASET" / "norm_stats.json")
    train = load_full_split_tensor(train_ds).numpy().astype(np.float64)
    test = load_full_split_tensor(test_ds).numpy().astype(np.float64)
    truth = test[1:]
    climatology = train.mean(axis=0)
    table = {
        "Persistence": metrics(test[:-1], truth, climatology, stds, test_ds.latitude.values),
        "Climatology": metrics(np.broadcast_to(climatology, truth.shape), truth, climatology, stds, test_ds.latitude.values),
    }
    checkpoint_dir = ROOT / "DATASET" / "checkpoints"
    for name, filename, kind in [("FNO", "fno.pt", "fno"), ("PINO", "pino.pt", "pino"), ("Proposed", "FactorizedFNO_GNNOff_soft_R3.pt", "proposed")]:
        path = checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        table[name] = metrics(load_model_predictions(test, means, stds, path, kind), truth, climatology, stds, test_ds.latitude.values)
    print("PHASE 0B PERSISTENCE DIAGNOSTIC")
    print(f"train={train_ds.time.values[0]}..{train_ds.time.values[-1]} samples={len(train)}")
    print(f"test={test_ds.time.values[0]}..{test_ds.time.values[-1]} samples={len(test)}")
    print("test_step_hours=" + str(np.unique(np.diff(test_ds.time.values).astype("timedelta64[h]").astype(int)).tolist()))
    print("\nTABLE 1: 6-hour metrics")
    print_metric_table(table)
    train_norm = (train - means[None, :, None, None]) / stds[None, :, None, None]
    test_norm = (test - means[None, :, None, None]) / stds[None, :, None, None]
    print("\nTABLE 2: normalized train and test mean/std")
    print("channel,train_norm_mean,train_norm_std,test_norm_mean,test_norm_std")
    for i, label in enumerate(LABELS):
        print(f"{label},{train_norm[:, i].mean():.9f},{train_norm[:, i].std():.9f},{test_norm[:, i].mean():.9f},{test_norm[:, i].std():.9f}")
    print("\nTABLE 3: largest absolute 6-hour jump")
    print("channel,max_abs_jump,from_timestamp,to_timestamp")
    for i, label in enumerate(LABELS):
        delta = np.abs(np.diff(test[:, i], axis=0))
        flat = int(np.nanargmax(delta))
        step, row, col = np.unravel_index(flat, delta.shape)
        print(f"{label},{delta[step, row, col]:.9f},{test_ds.time.values[step]},{test_ds.time.values[step + 1]}")
    print("\nTABLE 4: data integrity")
    print(f"NaNs_train={int(np.isnan(train).sum())},Infs_train={int(np.isinf(train).sum())},NaNs_test={int(np.isnan(test).sum())},Infs_test={int(np.isinf(test).sum())}")
    if args.skip_remote:
        raise RuntimeError("remote surface_pressure check was explicitly skipped")
    remote = xr.open_zarr(args.remote, consolidated=True, storage_options={"token": "anon"})
    selected = remote.sel(time=test_ds.time.values, method="nearest").sel(latitude=test_ds.latitude.values, longitude=test_ds.longitude.values, method="nearest")
    pressure_parts = []
    for start in range(0, len(test_ds.time), 64):
        pressure_parts.append(np.asarray(selected["surface_pressure"].isel(time=slice(start, start + 64)).values, dtype=np.float64))
    pressure = np.concatenate(pressure_parts, axis=0)
    if pressure.ndim == 4:
        pressure = pressure[:, 0]
    print("\nTABLE 5: below-ground fraction from surface_pressure")
    print("level_hPa,below_ground_cells_timesteps,all_cells_timesteps,below_ground_fraction")
    for level in TARGET_LEVELS:
        below = pressure < level * 100.0
        print(f"{level},{int(below.sum())},{below.size},{below.mean():.9f}")
    print("\nTABLE 6: Persistence RMSE by pressure level and ground mask")
    print("channel,above_ground_RMSE,below_ground_RMSE,above_ground_count,below_ground_count")
    error = test[:-1] - truth
    target_pressure = pressure[1:]
    for offset, level in enumerate(TARGET_LEVELS):
        above = target_pressure >= level * 100.0
        below = ~above
        for index in [3 + offset, 8 + offset]:
            above_rmse = np.sqrt(np.mean(error[:, index][above] ** 2)) if above.any() else np.nan
            below_rmse = np.sqrt(np.mean(error[:, index][below] ** 2)) if below.any() else np.nan
            print(f"{LABELS[index]},{above_rmse:.9f},{below_rmse:.9f},{int(above.sum())},{int(below.sum())}")
    print("\nTABLE 7: direct WeatherBench2 pressure selection")
    print(f"remote_time_samples={pressure.shape[0]},remote_lat_samples={pressure.shape[1]},remote_lon_samples={pressure.shape[2]}")


if __name__ == "__main__":
    main()
