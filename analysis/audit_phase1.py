import os
import sys
import json
from pathlib import Path
import numpy as np
import xarray as xr
from data.store import open_project_zarr
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pino_data_pipeline import (
    SURFACE_VARS,
    TARGET_LEVELS,
    UPPER_AIR_VAR,
    UPPER_AIR_TEMPERATURE_VAR,
    CHANNEL_NAMES,
    PINONormalizer,
    load_full_split_tensor,
)
from models.proposed_model import ProposedModel, FactorizedFNO2d
from train.train_baselines import CoordinateFNO2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_leakage():
    print("=== CHECK 1: DATA SPLITS & LEAKAGE ===")
    train_ds = open_project_zarr(ROOT / "DATASET" / "train_2018_2019.zarr", consolidated=True)
    val_ds = open_project_zarr(ROOT / "DATASET" / "val_2020.zarr", consolidated=True)
    test_ds = open_project_zarr(ROOT / "DATASET" / "test_2021_2022.zarr", consolidated=True)

    t_train = train_ds.time.values
    t_val = val_ds.time.values
    t_test = test_ds.time.values

    print(f"Train times: {t_train[0]} to {t_train[-1]} (count={len(t_train)})")
    print(f"Val times:   {t_val[0]} to {t_val[-1]} (count={len(t_val)})")
    print(f"Test times:  {t_test[0]} to {t_test[-1]} (count={len(t_test)})")

    overlap_train_val = np.intersect1d(t_train, t_val)
    overlap_val_test = np.intersect1d(t_val, t_test)
    overlap_train_test = np.intersect1d(t_train, t_test)

    print(f"Overlap train-val: count={len(overlap_train_val)}")
    print(f"Overlap val-test:  count={len(overlap_val_test)}")
    print(f"Overlap train-test: count={len(overlap_train_test)}")

    # Check normalization stats source
    norm_path = ROOT / "DATASET" / "norm_stats.json"
    with open(norm_path) as f:
        stats = json.load(f)
    print(f"norm_stats.json variables: {list(stats.keys())}")
    
    # Verify stats match train set exactly
    train_tensor = load_full_split_tensor(train_ds).numpy().astype(np.float64)
    normalizer = PINONormalizer(stats)
    calc_mean = train_tensor.mean(axis=(0, 2, 3))
    calc_std = train_tensor.std(axis=(0, 2, 3))
    stored_mean = normalizer.channel_means.flatten()
    stored_std = normalizer.channel_stds.flatten()

    mean_diff = np.max(np.abs(calc_mean - stored_mean))
    std_diff = np.max(np.abs(calc_std - stored_std))
    print(f"Max abs diff between train data and stored norm_stats mean: {mean_diff:.6e}")
    print(f"Max abs diff between train data and stored norm_stats std:  {std_diff:.6e}")

    # Check search/tuning configs to confirm test set was untouched
    search_file = ROOT / "ablations" / "search_proposed.py"
    with open(search_file) as f:
        search_code = f.read()
    test_in_search = "test_2021_2022" in search_code
    print(f"Search code references test_2021_2022: {test_in_search}")

    leakage_passed = (
        len(overlap_train_val) == 0 and
        len(overlap_val_test) == 0 and
        len(overlap_train_test) == 0 and
        mean_diff < 1e-4 and
        std_diff < 1e-4 and
        not test_in_search
    )
    print(f"LEAKAGE CHECK PASSED: {leakage_passed}")
    return train_ds, val_ds, test_ds, normalizer, train_tensor

def check_fourier_modes():
    print("\n=== CHECK 5: FOURIER MODES VS GRID SIZE ===")
    H, W = 15, 17
    # For rfft2 along axis -2 (lat, size 15) and axis -1 (lon, size 17)
    # Along lat (size 15): complex modes are 15 total, Nyquist max frequency is floor(15/2) = 7
    # Along lon (size 17): rfft length is 17 // 2 + 1 = 9
    max_usable_lat = H // 2  # 7
    max_usable_lon = W // 2 + 1  # 9
    print(f"Grid dimensions: lat={H}, lon={W}")
    print(f"Maximum usable positive/negative Fourier modes along lat (dim={H}): {max_usable_lat}")
    print(f"Maximum usable Fourier modes along lon (rfft dim={H}x{max_usable_lon}): {max_usable_lon}")

    # Configured modes in CoordinateFNO2d and FactorizedFNOBlock
    # FactorizedFNOBlock: modes_lat=8, modes_lon=6
    cfg_modes_lat = 8
    cfg_modes_lon = 6
    print(f"Configured modes in models: modes_lat={cfg_modes_lat}, modes_lon={cfg_modes_lon}")
    modes_exceed = cfg_modes_lat > max_usable_lat
    print(f"Does configured modes_lat exceed maximum usable modes? {modes_exceed} ({cfg_modes_lat} > {max_usable_lat})")
    print(f"Modes clamped in SpectralConv2d forward: m1 = min(modes1, height // 2) = min({cfg_modes_lat}, {max_usable_lat}) = {min(cfg_modes_lat, max_usable_lat)}")

def check_vertical_branch():
    print("\n=== CHECK 6: VERTICAL FACTORIZATION ACTIVE CHECK ===")
    ckpt_path = ROOT / "DATASET" / "checkpoints" / "FactorizedFNO_GNNOff_soft_R3.pt"
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    
    # Check weights in vertical branch
    vert_weights = {}
    for k, v in state.items():
        if "vertical" in k:
            vert_weights[k] = (v.shape, float(v.abs().mean()), float(v.abs().max()))
            print(f"  {k}: shape={v.shape}, mean_abs={vert_weights[k][1]:.6f}, max_abs={vert_weights[k][2]:.6f}")

    # Check forward pass gradient and output of vertical branch
    model = ProposedModel(
        terrain_path=str(ROOT / "DATASET" / "terrain_hp.npz"),
        physics_mode="soft",
        use_gnn_residual=False,
        refinement_steps=3,
        refinement_alpha=0.5
    ).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy_input = torch.randn(2, 13, 15, 17, device=DEVICE, requires_grad=True)
    # Hook on block 0 vertical output
    vert_outputs = []
    def hook_fn(module, input, output):
        vert_outputs.append(output[1]) # vertical is 2nd element
    
    hook = model.backbone.blocks[0].register_forward_hook(hook_fn)
    out = model(dummy_input)
    hook.remove()

    v_out = vert_outputs[0]
    print(f"Block 0 vertical output shape: {v_out.shape}")
    print(f"Block 0 vertical output mean abs: {float(v_out.detach().abs().mean()):.6f}")
    print(f"Block 0 vertical output std: {float(v_out.detach().std()):.6f}")

def compute_all_metrics(train_ds, test_ds, normalizer, train_tensor):
    print("\n=== CHECKS 2, 3, 4: BASELINES & METRICS COMPUTATION ===")
    test_tensor = load_full_split_tensor(test_ds).numpy().astype(np.float64)
    lats = test_ds.latitude.values # (15,)
    weights = np.cos(np.deg2rad(lats)).astype(np.float64)[None, None, :, None]
    denom = weights.sum() * (len(test_tensor) - 1) * test_tensor.shape[3]

    truth = test_tensor[1:]  # (2904, 13, 15, 17)
    test_inputs = test_tensor[:-1]

    # 1. Trivial Baseline: Persistence
    pred_persistence = test_inputs

    # 2. Trivial Baseline: Train-set Climatology
    clim_mean = train_tensor.mean(axis=0)  # (13, 15, 17)
    pred_climatology = np.broadcast_to(clim_mean[None], truth.shape)

    # 3. Trivial Baseline: Seasonal / Diurnal Climatology
    # Train is 6-hourly. Let's group train by (month, hour)
    import pandas as pd
    train_times = pd.to_datetime(train_ds.time.values)
    test_times = pd.to_datetime(test_ds.time.values[1:])

    # Seasonal (month) + Diurnal (hour) climatology from train set
    train_months = train_times.month
    train_hours = train_times.hour

    diurnal_seasonal_clim = {}
    for m in range(1, 13):
        for h in [0, 6, 12, 18]:
            mask = (train_months == m) & (train_hours == h)
            if mask.sum() > 0:
                diurnal_seasonal_clim[(m, h)] = train_tensor[mask].mean(axis=0)
            else:
                # fallback to hour
                mask_h = (train_hours == h)
                diurnal_seasonal_clim[(m, h)] = train_tensor[mask_h].mean(axis=0)

    pred_seasonal_diurnal = np.zeros_like(truth)
    for i, t in enumerate(test_times):
        pred_seasonal_diurnal[i] = diurnal_seasonal_clim[(t.month, t.hour)]

    # Model predictions
    means = normalizer.channel_means.flatten().astype(np.float64)
    stds = normalizer.channel_stds.flatten().astype(np.float64)

    def run_model(model_cls, ckpt_name, **kwargs):
        path = ROOT / "DATASET" / "checkpoints" / ckpt_name
        if not path.exists():
            return None
        model = model_cls(**kwargs).to(DEVICE)
        st = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(st, strict=False)
        model.eval()
        
        inp_norm = ((test_inputs - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)
        preds = []
        with torch.no_grad():
            for b in range(0, len(inp_norm), 32):
                bx = torch.from_numpy(inp_norm[b:b+32]).to(DEVICE)
                preds.append(model(bx).cpu().numpy())
        pred_norm = np.concatenate(preds, axis=0).astype(np.float64)
        return pred_norm * stds[None, :, None, None] + means[None, :, None, None]

    pred_fno = run_model(CoordinateFNO2d, "fno.pt", in_dim=13, out_dim=13, width=32)
    pred_pino = run_model(CoordinateFNO2d, "pino.pt", in_dim=13, out_dim=13, width=32)
    pred_proposed = run_model(
        ProposedModel, "FactorizedFNO_GNNOff_soft_R3.pt",
        terrain_path=str(ROOT / "DATASET" / "terrain_hp.npz"),
        physics_mode="soft", use_gnn_residual=False, refinement_steps=3, refinement_alpha=0.5
    )

    models_dict = {
        "Persistence": pred_persistence,
        "Train Climatology": pred_climatology,
        "Seasonal-Diurnal Climatology": pred_seasonal_diurnal,
        "FNO": pred_fno,
        "PINO": pred_pino,
        "Proposed": pred_proposed,
    }

    # Compute Climatology RMSE for each channel as the denominator for nRMSE_clim
    err_clim = pred_climatology - truth
    clim_rmse = np.sqrt((err_clim ** 2 * weights).sum(axis=(0, 2, 3)) / denom)

    # Anomaly reference is train climatology
    truth_anom = truth - clim_mean[None]
    truth_var = (truth_anom ** 2 * weights).sum(axis=(0, 2, 3))

    results = {}
    print("\n--- DETAILED METRICS BY MODEL ---")
    for name, pred in models_dict.items():
        if pred is None:
            continue
        err = pred - truth
        rmse = np.sqrt((err ** 2 * weights).sum(axis=(0, 2, 3)) / denom)
        
        # Check 2: nRMSE definitions
        nrmse_std = rmse / stds
        nrmse_clim = rmse / clim_rmse

        # Check 4: ACC using anomalies relative to train climatology
        pred_anom = pred - clim_mean[None]
        pred_var = (pred_anom ** 2 * weights).sum(axis=(0, 2, 3))
        cov = (pred_anom * truth_anom * weights).sum(axis=(0, 2, 3))
        acc = cov / (np.sqrt(pred_var * truth_var) + 1e-12)

        results[name] = {
            "rmse": rmse,
            "nrmse_std": nrmse_std,
            "nrmse_clim": nrmse_clim,
            "acc": acc,
            "agg_rmse": float(rmse.mean()),
            "agg_nrmse_std": float(nrmse_std.mean()),
            "agg_nrmse_clim": float(nrmse_clim.mean()),
            "agg_acc": float(acc.mean()),
        }

    # Print comparison table
    print(f"\n{'Model':<30} | {'Agg RMSE':<10} | {'nRMSE (train-std)':<18} | {'nRMSE (clim-RMSE)':<18} | {'Agg ACC':<10}")
    print("-" * 95)
    for name, res in results.items():
        print(f"{name:<30} | {res['agg_rmse']:<10.4f} | {res['agg_nrmse_std']:<18.4f} | {res['agg_nrmse_clim']:<18.4f} | {res['agg_acc']:<10.4f}")

    # Key variables: T2m (idx 0), U10 (idx 1), V10 (idx 2), Z500 (idx 6)
    key_vars = [("T2m", 0), ("U10", 1), ("V10", 2), ("Z500", 6)]
    print("\n--- KEY VARIABLES LATITUDE-WEIGHTED RMSE ---")
    header = f"{'Model':<30} | " + " | ".join([f"{kv[0]} RMSE" for kv in key_vars])
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        vals = " | ".join([f"{res['rmse'][kv[1]]:<10.4f}" for kv in key_vars])
        print(f"{name:<30} | {vals}")

    print("\n--- KEY VARIABLES LATITUDE-WEIGHTED ACC ---")
    header = f"{'Model':<30} | " + " | ".join([f"{kv[0]} ACC" for kv in key_vars])
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        vals = " | ".join([f"{res['acc'][kv[1]]:<10.4f}" for kv in key_vars])
        print(f"{name:<30} | {vals}")

    # Per-level results for Geopotential (Z1000..Z300) and Temperature (T1000..T300)
    print("\n--- PER-LEVEL GEOPOTENTIAL RMSE (m2/s2) ---")
    levels = [1000, 850, 700, 500, 300]
    header = f"{'Model':<30} | " + " | ".join([f"Z{lvl}" for lvl in levels])
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        vals = " | ".join([f"{res['rmse'][3+i]:<10.2f}" for i, lvl in enumerate(levels)])
        print(f"{name:<30} | {vals}")

    print("\n--- PER-LEVEL TEMPERATURE RMSE (K) ---")
    header = f"{'Model':<30} | " + " | ".join([f"T{lvl}" for lvl in levels])
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        vals = " | ".join([f"{res['rmse'][8+i]:<10.4f}" for i, lvl in enumerate(levels)])
        print(f"{name:<30} | {vals}")

if __name__ == "__main__":
    train_ds, val_ds, test_ds, normalizer, train_tensor = check_leakage()
    check_fourier_modes()
    check_vertical_branch()
    compute_all_metrics(train_ds, test_ds, normalizer, train_tensor)
