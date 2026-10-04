"""Standalone autoregressive inference for trained weather-surrogate models."""

import argparse
import json
import os
import uuid

import numpy as np
import torch
import xarray as xr

from data.pino_data_pipeline import (
    CHANNEL_NAMES,
    PINONormalizer,
    TARGET_LEVELS,
    UPPER_AIR_TEMPERATURE_VAR,
    UPPER_AIR_VAR,
    load_full_split_tensor,
)
from train.proposed_model import ProposedModel
from train.train_baselines import CoordinateFNO2d, load_checkpoint


SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
DATASET_DIR = "DATASET"


def parse_proposed_config(value):
    """Accept canonical checkpoint names and the older ``phys_*`` names."""
    if not value:
        raise ValueError("--config is required for --model proposed")
    if value.startswith("FactorizedFNO_GNN"):
        tail = value.removeprefix("FactorizedFNO_GNN")
        for variant in ("Dense", "Sparse", "Off"):
            if tail.startswith(f"{variant}_"):
                physics = tail[len(variant) + 1:]
                refinement_steps = 1
                if "_R" in physics:
                    physics, refinement = physics.rsplit("_R", 1)
                    refinement_steps = int(refinement)
                return {
                    "physics_mode": physics.lower(),
                    "use_gnn_residual": variant != "Off",
                    "gnn_dense": variant == "Dense",
                    "refinement_steps": refinement_steps,
                    "refinement_alpha": 0.5,
                }
    if value.startswith("phys_"):
        parts = value.split("_")
        return {
            "physics_mode": parts[1],
            "use_gnn_residual": "gnn" in parts,
            "gnn_dense": "dense" in parts,
            "refinement_steps": 1,
            "refinement_alpha": 0.5,
        }
    raise ValueError(f"Unknown proposed configuration name: {value}")


def load_initial_state(args):
    if bool(args.init_time) == bool(args.init_npz):
        raise ValueError("Provide exactly one of --init-time or --init-npz")
    if args.init_npz:
        loaded = np.load(args.init_npz)
        try:
            if isinstance(loaded, np.ndarray):
                array = loaded
            else:
                key = "state" if "state" in loaded.files else loaded.files[0]
                array = loaded[key]
        finally:
            if hasattr(loaded, "close"):
                loaded.close()
        state = torch.as_tensor(array, dtype=torch.float32)
        if tuple(state.shape) != (13, 15, 17):
            raise ValueError(f"--init-npz must contain shape (13, 15, 17), got {tuple(state.shape)}")
        return state, None, None

    store = os.path.join(DATASET_DIR, "test_2021_2022.zarr")
    ds = xr.open_zarr(store)
    timestamp = np.datetime64(args.init_time)
    selected = ds.sel(time=timestamp).expand_dims(time=[timestamp])
    return load_full_split_tensor(selected).squeeze(0), ds, timestamp


def build_model(model_name, config, device):
    if model_name == "persistence":
        return None
    if model_name in {"fno", "pino"}:
        return CoordinateFNO2d(in_dim=13, out_dim=13).to(device)
    if model_name == "proposed":
        return ProposedModel(
            terrain_path=os.path.join(DATASET_DIR, "terrain_hp.npz"),
            **parse_proposed_config(config),
        ).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def metric_summary(prediction, truth):
    result = {}
    for index, name in enumerate(CHANNEL_NAMES):
        pred = prediction[:, index]
        target = truth[:, index]
        error = pred - target
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        acc = (pred_centered * target_centered).sum() / (
            torch.sqrt(pred_centered.square().sum() * target_centered.square().sum()) + 1e-8
        )
        result[name] = {
            "rmse": float(torch.sqrt(error.square().mean())),
            "acc": float(acc),
        }
    return result


def build_xarray(states, times, latitude, longitude):
    surface = {
        name: (("time", "latitude", "longitude"), states[:, index].numpy())
        for index, name in enumerate(SURFACE_VARS)
    }
    geopotential = states[:, 3:8].numpy()
    temperature = states[:, 8:13].numpy()
    return xr.Dataset(
        {
            **surface,
            UPPER_AIR_VAR: (("time", "level", "latitude", "longitude"), geopotential),
            UPPER_AIR_TEMPERATURE_VAR: (("time", "level", "latitude", "longitude"), temperature),
        },
        coords={"time": times, "level": TARGET_LEVELS, "latitude": latitude, "longitude": longitude},
    )


def run(args):
    os.makedirs(os.path.join(DATASET_DIR, "inference"), exist_ok=True)
    with open(os.path.join(DATASET_DIR, "norm_stats.json"), encoding="utf-8") as handle:
        normalizer = PINONormalizer(json.load(handle))

    initial, truth_store, initial_time = load_initial_state(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.config, device)
    if model is not None:
        checkpoint_name = args.config if args.model == "proposed" else args.model
        checkpoint = os.path.join(DATASET_DIR, "checkpoints", f"{checkpoint_name}.pt")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        load_checkpoint(model, checkpoint, device)
        model.eval()

    forecast = []
    current = initial
    with torch.no_grad():
        for _ in range(args.horizon_steps):
            if model is None:
                next_state = current.clone()
            else:
                normalized = normalizer.normalize(current.unsqueeze(0).to(device))
                prediction = model(normalized)
                next_state = normalizer.denormalize(prediction.cpu()).squeeze(0)
            forecast.append(next_state.cpu())
            current = next_state.cpu()
    forecast = torch.stack(forecast)

    if initial_time is None:
        times = np.datetime64("1970-01-01T00") + np.arange(1, args.horizon_steps + 1, dtype="timedelta64[h]") * 6
        latitude = np.linspace(33.5, 30.0, 15)
        longitude = np.linspace(75.5, 79.5, 17)
    else:
        times = initial_time + np.arange(1, args.horizon_steps + 1, dtype="timedelta64[h]") * 6
        latitude = truth_store.latitude.values
        longitude = truth_store.longitude.values

    run_id = args.run_id or f"{args.model}_{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(DATASET_DIR, "inference", run_id)
    os.makedirs(output_dir, exist_ok=True)
    build_xarray(forecast, times, latitude, longitude).to_netcdf(os.path.join(output_dir, "forecast.nc"))

    summary = {
        "model": args.model,
        "config": args.config,
        "horizon_steps": args.horizon_steps,
        "variables": {
            name: {
                "min": float(forecast[:, index].min()),
                "max": float(forecast[:, index].max()),
                "mean": float(forecast[:, index].mean()),
            }
            for index, name in enumerate(CHANNEL_NAMES)
        },
    }
    if truth_store is not None:
        truth = load_full_split_tensor(truth_store.sel(time=times))
        summary["metrics"] = metric_summary(forecast, truth)
        error_map = (forecast - truth).square().mean(dim=1).sqrt().numpy()
        xr.Dataset(
            {"rmse": (("time", "latitude", "longitude"), error_map)},
            coords={"time": times, "latitude": latitude, "longitude": longitude},
        ).to_netcdf(os.path.join(output_dir, "error_map.nc"))
    with open(os.path.join(output_dir, "forecast_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[DONE] Forecast written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("persistence", "fno", "pino", "proposed"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--init-time")
    parser.add_argument("--init-npz")
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--run-id")
    run(parser.parse_args())
