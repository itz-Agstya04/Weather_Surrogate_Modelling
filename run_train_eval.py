import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '.')
from data.pino_data_pipeline import (
    PINONormalizer,
    get_single_step_loader,
    get_rollout_loader,
    load_full_split_tensor,
    CHANNEL_NAMES,
)
from train.train_baselines import (
    CoordinateFNO2d,
    train_one_epoch,
    validate,
    METRIC_NAMES,
)
from models.proposed_model import ProposedModel

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_boundary_forcing(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    h, w = pred.shape[-2:]
    mask_outer = torch.zeros((h, w), dtype=torch.bool, device=pred.device)
    mask_outer[:2, :] = True
    mask_outer[-2:, :] = True
    mask_outer[:, :2] = True
    mask_outer[:, -2:] = True

    weight_blend = torch.zeros((h, w), dtype=pred.dtype, device=pred.device)
    weight_blend[mask_outer] = 1.0

    r2 = torch.zeros((h, w), dtype=torch.bool, device=pred.device)
    r2[2:-2, 2:-2] = True
    r3 = torch.zeros((h, w), dtype=torch.bool, device=pred.device)
    r3[3:-3, 3:-3] = True
    r4 = torch.zeros((h, w), dtype=torch.bool, device=pred.device)
    r4[4:-4, 4:-4] = True

    cell_2 = r2 & (~r3)
    cell_3 = r3 & (~r4)

    weight_blend[cell_2] = 0.75
    weight_blend[cell_3] = 0.50
    weight_blend[r4 & (~torch.zeros((h, w), dtype=torch.bool, device=pred.device))] = 0.25

    wb = weight_blend.view(1, 1, h, w)
    return pred * (1.0 - wb) + truth * wb

def fine_tune_rollout_with_boundaries(model: nn.Module, rollout_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int = 3):
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for seq in rollout_loader:
            seq = seq.to(device)
            b, t, c, h, w = seq.shape
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=True):
                curr = seq[:, 0]
                loss = 0.0
                for step in range(1, t):
                    truth = seq[:, step]
                    pred = model(curr)
                    loss = loss + F.mse_loss(pred, truth)
                    curr = apply_boundary_forcing(pred, truth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

def compute_spherical_divergence_metrics(u: torch.Tensor, v: torch.Tensor):
    lat = torch.linspace(33.5, 30.0, 15, dtype=torch.float32, device=u.device)
    dlat_rad = torch.abs(lat[1] - lat[0]) * (3.141592653589793 / 180.0)
    dlon_rad = (4.0 / (17 - 1)) * (3.141592653589793 / 180.0)
    earth_radius = 6_371_000.0
    phi = lat * (3.141592653589793 / 180.0)
    cos_phi = torch.cos(phi).view(1, 15, 1)

    du_dlam = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dlon_rad)
    v_cos = v * cos_phi
    d_vcos_dphi = (v_cos[:, :-2, :] - v_cos[:, 2:, :]) / (2.0 * dlat_rad)

    du_dlam_int = du_dlam[:, 1:-1, :]
    d_vcos_dphi_int = d_vcos_dphi[:, :, 1:-1]
    div = (du_dlam_int + d_vcos_dphi_int) / (earth_radius * cos_phi[:, 1:-1, :])

    div_interior = div[:, 1:-1, 1:-1]
    u_grad = du_dlam_int[:, 1:-1, 1:-1] / (earth_radius * cos_phi[:, 2:-2, :])
    v_grad = d_vcos_dphi_int[:, 1:-1, 1:-1] / earth_radius
    grad_mag = torch.mean(torch.sqrt(u_grad**2 + v_grad**2 + 1e-12))
    norm_div = torch.mean(torch.abs(div_interior)) / grad_mag

    return float(torch.mean(torch.abs(div_interior)).item()), float(norm_div.item())

def calculate_metrics_with_climatology(preds_raw: torch.Tensor, tgts_raw: torch.Tensor, clim_raw: torch.Tensor, normalizer: PINONormalizer, lats: np.ndarray) -> dict:
    weights = torch.from_numpy(np.cos(np.radians(lats)).astype(np.float32)).reshape(1, 1, -1, 1)
    sq_err = (preds_raw - tgts_raw) ** 2
    denom = weights.sum() * preds_raw.size(0) * preds_raw.size(3)
    rmse = torch.sqrt((sq_err * weights).sum(dim=(0, 2, 3)) / denom)

    stds = torch.from_numpy(normalizer.channel_stds.reshape(1, -1, 1, 1))
    nrmse = rmse / stds.flatten()

    pred_anom = preds_raw - clim_raw
    tgt_anom = tgts_raw - clim_raw
    cov = (weights * pred_anom * tgt_anom).sum(dim=(0, 2, 3))
    pred_var = (weights * pred_anom.square()).sum(dim=(0, 2, 3))
    tgt_var = (weights * tgt_anom.square()).sum(dim=(0, 2, 3))
    acc = cov / (torch.sqrt(pred_var * tgt_var) + 1e-8)

    return {
        'rmse': {name: float(value) for name, value in zip(METRIC_NAMES, rmse)},
        'nrmse': {name: float(value) for name, value in zip(METRIC_NAMES, nrmse)},
        'acc': {name: float(value) for name, value in zip(METRIC_NAMES, acc)},
        'aggregate_rmse': float(rmse.mean()),
        'aggregate_nrmse': float(nrmse.mean()),
        'aggregate_acc': float(acc.mean()),
    }

def benchmark_inference(model: nn.Module, sample_x: torch.Tensor, device: torch.device):
    model.eval()
    x = sample_x.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=True):
            for _ in range(25):
                _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=True):
                _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0.0
    return float(np.median(timings)), float(vram_mb)

def run_evaluation_suite(model: nn.Module, model_name: str, test_rollout_loader: DataLoader, normalizer: PINONormalizer, clim_table: np.ndarray, test_times: pd.DatetimeIndex, device: torch.device, lats: np.ndarray):
    model.eval()
    max_steps = 12
    step_targets = [[] for _ in range(max_steps)]
    step_preds = [[] for _ in range(max_steps)]
    step_clims = [[] for _ in range(max_steps)]

    lead_map = {1: '6h', 4: '24h', 8: '48h', 12: '72h'}
    batch_offset = 0

    wind_u_list = []
    wind_v_list = []
    res_wind_u_list = []
    res_wind_v_list = []

    with torch.no_grad():
        for seq in test_rollout_loader:
            seq = seq.to(device)
            b = seq.shape[0]
            curr = seq[:, 0]
            for step in range(1, max_steps + 1):
                truth = seq[:, step]
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    pred = model(curr)
                if step == 1 and hasattr(model, 'gnn') and model.gnn is not None:
                    res_wind = model.gnn(curr, return_residual_wind_only=True)
                    res_wind_u_list.append(res_wind[:, 0].cpu())
                    res_wind_v_list.append(res_wind[:, 1].cpu())
                if step == 1:
                    wind_u_list.append(pred[:, 1].cpu())
                    wind_v_list.append(pred[:, 2].cpu())

                step_preds[step - 1].append(normalizer.denormalize(pred.float()).cpu())
                step_targets[step - 1].append(normalizer.denormalize(truth.float()).cpu())

                clim_batch = np.zeros((b, 13, 15, 17), dtype=np.float32)
                for idx_b in range(b):
                    t = test_times[batch_offset + idx_b + step]
                    clim_batch[idx_b] = clim_table[t.dayofyear, t.hour]
                step_clims[step - 1].append(torch.from_numpy(clim_batch))

                curr = apply_boundary_forcing(pred, truth)
            batch_offset += b

    metrics_by_lead = {}
    for step_idx, lead_str in lead_map.items():
        all_p = torch.cat(step_preds[step_idx - 1], dim=0)
        all_t = torch.cat(step_targets[step_idx - 1], dim=0)
        all_c = torch.cat(step_clims[step_idx - 1], dim=0)
        metrics_by_lead[lead_str] = calculate_metrics_with_climatology(all_p, all_t, all_c, normalizer, lats)

    all_u = torch.cat(wind_u_list, dim=0)
    all_v = torch.cat(wind_v_list, dim=0)
    div_mean, norm_div = compute_spherical_divergence_metrics(all_u, all_v)

    if res_wind_u_list:
        all_ru = torch.cat(res_wind_u_list, dim=0)
        all_rv = torch.cat(res_wind_v_list, dim=0)
        r_div_mean, r_norm_div = compute_spherical_divergence_metrics(all_ru, all_rv)
    else:
        r_div_mean, r_norm_div = 0.0, 0.0

    return {
        'leads': metrics_by_lead,
        'lat_weighted_rmse_6h': metrics_by_lead['6h']['aggregate_rmse'],
        'lat_weighted_acc_6h': metrics_by_lead['6h']['aggregate_acc'],
        'nrmse_72h': metrics_by_lead['72h']['aggregate_nrmse'],
        'wind_div_mean': div_mean,
        'wind_norm_div': norm_div,
        'res_wind_div_mean': r_div_mean,
        'res_wind_norm_div': r_norm_div,
    }

def evaluate_non_learned_baselines(test_rollout_loader: DataLoader, normalizer: PINONormalizer, clim_table: np.ndarray, test_times: pd.DatetimeIndex, lats: np.ndarray):
    lead_map = {1: '6h', 4: '24h', 8: '48h', 12: '72h'}
    pers_preds = {k: [] for k in lead_map}
    clim_preds = {k: [] for k in lead_map}
    tgts = {k: [] for k in lead_map}
    clims = {k: [] for k in lead_map}

    batch_offset = 0
    for seq in test_rollout_loader:
        b = seq.shape[0]
        initial = normalizer.denormalize(seq[:, 0].float()).cpu()
        for step_idx in lead_map:
            pers_preds[step_idx].append(initial)
            truth = normalizer.denormalize(seq[:, step_idx].float()).cpu()
            tgts[step_idx].append(truth)
            clim_batch = np.zeros((b, 13, 15, 17), dtype=np.float32)
            for idx_b in range(b):
                t = test_times[batch_offset + idx_b + step_idx]
                clim_batch[idx_b] = clim_table[t.dayofyear, t.hour]
            clims[step_idx].append(torch.from_numpy(clim_batch))
            clim_preds[step_idx].append(torch.from_numpy(clim_batch))
        batch_offset += b

    pers_leads = {}
    clim_leads = {}
    for step_idx, lead_str in lead_map.items():
        all_t = torch.cat(tgts[step_idx], dim=0)
        all_c = torch.cat(clims[step_idx], dim=0)
        pers_leads[lead_str] = calculate_metrics_with_climatology(torch.cat(pers_preds[step_idx], dim=0), all_t, all_c, normalizer, lats)
        clim_leads[lead_str] = calculate_metrics_with_climatology(torch.cat(clim_preds[step_idx], dim=0), all_t, all_c, normalizer, lats)

    return {
        'persistence': {
            'leads': pers_leads,
            'lat_weighted_rmse_6h': pers_leads['6h']['aggregate_rmse'],
            'lat_weighted_acc_6h': pers_leads['6h']['aggregate_acc'],
            'nrmse_72h': pers_leads['72h']['aggregate_nrmse'],
        },
        'climatology': {
            'leads': clim_leads,
            'lat_weighted_rmse_6h': clim_leads['6h']['aggregate_rmse'],
            'lat_weighted_acc_6h': clim_leads['6h']['aggregate_acc'],
            'nrmse_72h': clim_leads['72h']['aggregate_nrmse'],
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seeds', type=str, default='0,1,2')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running pipeline on {device} (smoke={args.smoke}, seeds={seeds})", flush=True)

    normalizer = PINONormalizer(json.load(open('DATASET/norm_stats.json')))
    clim_table = np.load('DATASET/train_climatology_smoothed.npy')
    ds_test = xr.open_zarr('DATASET/test_2021_2022.zarr')
    test_times = pd.to_datetime(ds_test.time.values)
    lats = ds_test.latitude.values

    train_zarr = 'DATASET/train_2018_2019.zarr'
    val_zarr = 'DATASET/val_2020.zarr'
    test_zarr = 'DATASET/test_2021_2022.zarr'

    warmup_epochs = 1 if args.smoke else 20
    rollout_epochs = 1 if args.smoke else 3

    model_factories = {
        'dense_fno': lambda: CoordinateFNO2d(in_dim=13, out_dim=13, width=32),
        'dense_fno_matched': lambda: CoordinateFNO2d(in_dim=13, out_dim=13, width=36),
        'pino_soft': lambda: CoordinateFNO2d(in_dim=13, out_dim=13, width=32),
        'proposed': lambda: ProposedModel(terrain_path='DATASET/terrain_hp.npz', width=32, physics_mode='none', use_gnn_residual=True, gnn_dense=False, refinement_steps=3),
    }

    results_by_model = {m: [] for m in model_factories}
    sample_x = None

    for seed in seeds:
        set_seed(seed)
        train_loader = get_single_step_loader(train_zarr, normalizer, batch_size=32, shuffle=True)
        val_loader = get_single_step_loader(val_zarr, normalizer, batch_size=32, shuffle=False)
        train_rollout_loader = get_rollout_loader(train_zarr, normalizer, sequence_length=4, batch_size=8, shuffle=True)
        test_rollout_loader = get_rollout_loader(test_zarr, normalizer, sequence_length=13, batch_size=8, shuffle=False)

        if args.smoke:
            sub_tr = max(2, int(len(train_loader.dataset) * 0.02))
            train_loader = DataLoader(Subset(train_loader.dataset, range(sub_tr)), batch_size=4, shuffle=True)
            val_loader = DataLoader(Subset(val_loader.dataset, range(max(2, int(len(val_loader.dataset) * 0.02)))), batch_size=4, shuffle=False)
            train_rollout_loader = DataLoader(Subset(train_rollout_loader.dataset, range(2)), batch_size=2, shuffle=True)
            test_rollout_loader = DataLoader(Subset(test_rollout_loader.dataset, range(2)), batch_size=2, shuffle=False)

        if sample_x is None:
            sample_x = next(iter(val_loader))[0][:1]

        for model_name, factory in model_factories.items():
            print(f"\n--- Model: {model_name} | Seed: {seed} ---", flush=True)
            model = factory().to(device)
            use_physics = (model_name == 'pino_soft')
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warmup_epochs, eta_min=1e-5)

            best_val = float('inf')
            best_state = None
            for ep in range(1, warmup_epochs + 1):
                tr_res = train_one_epoch(model, train_loader, opt, device, use_physics_loss=use_physics, lambda_div=0.01, lambda_hydro=0.1, max_grad_norm=1.0, normalizer=normalizer)
                vl_res = validate(model, val_loader, device, use_physics_loss=use_physics, lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer)
                sched.step()
                vl_loss = vl_res[0] if isinstance(vl_res, tuple) else vl_res
                if vl_loss < best_val:
                    best_val = vl_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  Warmup Ep {ep:2d}/{warmup_epochs} | Val: {vl_loss:.5f}", flush=True)

            if best_state is not None:
                model.load_state_dict(best_state)

            rollout_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            fine_tune_rollout_with_boundaries(model, train_rollout_loader, rollout_opt, device, epochs=rollout_epochs)

            lat_ms, peak_vram = benchmark_inference(model, sample_x, device)
            eval_dict = run_evaluation_suite(model, model_name, test_rollout_loader, normalizer, clim_table, test_times, device, lats)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            eval_dict['param_count'] = param_count
            eval_dict['latency_ms'] = lat_ms
            eval_dict['peak_vram_mb'] = peak_vram
            results_by_model[model_name].append(eval_dict)
            print(f"  Result: RMSE={eval_dict['lat_weighted_rmse_6h']:.4f}, ACC={eval_dict['lat_weighted_acc_6h']:.4f}, 72h nRMSE={eval_dict['nrmse_72h']:.4f}", flush=True)

    non_learned = evaluate_non_learned_baselines(test_rollout_loader, normalizer, clim_table, test_times, lats)

    final_output = {
        'persistence': non_learned['persistence'],
        'climatology': non_learned['climatology'],
        'models': {}
    }

    for model_name, runs in results_by_model.items():
        summary = {
            'param_count': runs[0]['param_count'],
            'latency_ms_mean': float(np.mean([r['latency_ms'] for r in runs])),
            'latency_ms_std': float(np.std([r['latency_ms'] for r in runs])),
            'peak_vram_mb_mean': float(np.mean([r['peak_vram_mb'] for r in runs])),
            'lat_weighted_rmse_6h_mean': float(np.mean([r['lat_weighted_rmse_6h'] for r in runs])),
            'lat_weighted_rmse_6h_std': float(np.std([r['lat_weighted_rmse_6h'] for r in runs])),
            'lat_weighted_acc_6h_mean': float(np.mean([r['lat_weighted_acc_6h'] for r in runs])),
            'lat_weighted_acc_6h_std': float(np.std([r['lat_weighted_acc_6h'] for r in runs])),
            'nrmse_72h_mean': float(np.mean([r['nrmse_72h'] for r in runs])),
            'nrmse_72h_std': float(np.std([r['nrmse_72h'] for r in runs])),
            'wind_norm_div_mean': float(np.mean([r['wind_norm_div'] for r in runs])),
            'res_wind_norm_div_mean': float(np.mean([r['res_wind_norm_div'] for r in runs])),
            'runs': runs,
        }
        final_output['models'][model_name] = summary

    if not args.smoke:
        os.makedirs('results', exist_ok=True)
        with open('results/final_metrics.json', 'w') as f:
            json.dump(final_output, f, indent=2)
        print("Wrote results to results/final_metrics.json", flush=True)

    return final_output

if __name__ == '__main__':
    main()
