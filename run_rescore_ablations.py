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
    CHANNEL_NAMES,
)
from train.train_baselines import (
    CoordinateFNO2d,
    train_one_epoch,
    validate,
    METRIC_NAMES,
)
from models.proposed_model import (
    ProposedModel,
    FactorizedFNO2d,
    TerrainGNNResidual,
    IterativeCorrection,
    load_terrain_artifact,
    STATE_CHANNELS,
    TERRAIN_CHANNELS,
)

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
    norm_div = torch.mean(torch.abs(div_interior)) / (grad_mag + 1e-12)

    return float(torch.mean(torch.abs(div_interior)).item()), float(norm_div.item())

def calculate_metrics_masked(preds_raw: torch.Tensor, tgts_raw: torch.Tensor, clim_raw: torch.Tensor, normalizer: PINONormalizer, lats: np.ndarray, mask: torch.Tensor) -> dict:
    w_lat = torch.from_numpy(np.cos(np.radians(lats)).astype(np.float32)).view(1, 1, -1, 1)
    mask_4d = mask.view(1, 1, 15, 17).float()
    w_effective = w_lat * mask_4d
    denom = w_effective.sum() * preds_raw.size(0)

    sq_err = (preds_raw - tgts_raw) ** 2
    rmse = torch.sqrt((sq_err * w_effective).sum(dim=(0, 2, 3)) / (denom + 1e-12))

    stds = torch.from_numpy(normalizer.channel_stds.reshape(1, -1, 1, 1))
    nrmse = rmse / (stds.flatten() + 1e-12)

    pred_anom = preds_raw - clim_raw
    tgt_anom = tgts_raw - clim_raw
    cov = (w_effective * pred_anom * tgt_anom).sum(dim=(0, 2, 3))
    pred_var = (w_effective * pred_anom.square()).sum(dim=(0, 2, 3))
    tgt_var = (w_effective * tgt_anom.square()).sum(dim=(0, 2, 3))
    acc = cov / (torch.sqrt(pred_var * tgt_var) + 1e-8)

    return {
        'rmse': {name: float(value) for name, value in zip(METRIC_NAMES, rmse)},
        'nrmse': {name: float(value) for name, value in zip(METRIC_NAMES, nrmse)},
        'acc': {name: float(value) for name, value in zip(METRIC_NAMES, acc)},
        'aggregate_rmse': float(rmse.mean()),
        'aggregate_nrmse': float(nrmse.mean()),
        'aggregate_acc': float(acc.mean()),
    }

class FFNOOnlyModel(nn.Module):
    def __init__(self, terrain_path: str = "DATASET/terrain_hp.npz", width: int = 35, refinement_steps: int = 3, refinement_alpha: float = 0.5):
        super().__init__()
        terrain_data = load_terrain_artifact(terrain_path)
        self.backbone = FactorizedFNO2d(width=width)
        self.register_buffer("terrain_features", terrain_data["terrain_features"])
        self.corrector = IterativeCorrection(width=width)
        self.refinement_steps = refinement_steps
        self.refinement_alpha = refinement_alpha

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.backbone(state, self.terrain_features)
        for _ in range(self.refinement_steps - 1):
            correction = self.corrector(state, output)
            output = output + self.refinement_alpha * correction
        return output

class FFNORandomSparseGNNModel(nn.Module):
    def __init__(self, terrain_path: str = "DATASET/terrain_hp.npz", width: int = 32, refinement_steps: int = 3, refinement_alpha: float = 0.5):
        super().__init__()
        terrain_data = load_terrain_artifact(terrain_path)
        self.backbone = FactorizedFNO2d(width=width)
        self.register_buffer("terrain_features", terrain_data["terrain_features"])
        self.corrector = IterativeCorrection(width=width)
        self.refinement_steps = refinement_steps
        self.refinement_alpha = refinement_alpha

        # Build random sparse graph (128 nodes chosen uniformly at random, fixed seed 42)
        rng = np.random.RandomState(42)
        rand_nodes = rng.choice(255, size=128, replace=False)
        rand_nodes.sort()
        rand_nodes_t = torch.from_numpy(rand_nodes).long()

        dense_edge_index = terrain_data["dense_edge_index"]
        dense_edge_attr = terrain_data["dense_edge_attr"]
        node_set = set(rand_nodes)
        mask = [(u.item() in node_set and v.item() in node_set) for u, v in dense_edge_index.t()]
        mask_t = torch.tensor(mask, dtype=torch.bool)
        sub_edges = dense_edge_index[:, mask_t]
        sub_attr = dense_edge_attr[mask_t]

        node_to_idx = {n: i for i, n in enumerate(rand_nodes)}
        src_remapped = torch.tensor([node_to_idx[u.item()] for u in sub_edges[0]], dtype=torch.long)
        dst_remapped = torch.tensor([node_to_idx[v.item()] for v in sub_edges[1]], dtype=torch.long)
        remapped_edge_index = torch.stack([src_remapped, dst_remapped], dim=0)

        custom_terrain_data = {
            "terrain_features": terrain_data["terrain_features"],
            "node_flat_indices": rand_nodes_t,
            "edge_index": remapped_edge_index,
            "edge_attr": sub_attr,
            "dense_edge_index": terrain_data["dense_edge_index"],
            "dense_edge_attr": terrain_data["dense_edge_attr"],
        }
        self.gnn = TerrainGNNResidual(terrain_path, gnn_dense=False, terrain_data=custom_terrain_data)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.backbone(state, self.terrain_features) + self.gnn(state)
        for _ in range(self.refinement_steps - 1):
            correction = self.corrector(state, output)
            output = output + self.refinement_alpha * correction
        return output

class FFNOGatedUnconstrainedModel(nn.Module):
    def __init__(self, terrain_path: str = "DATASET/terrain_hp.npz", width: int = 32, refinement_steps: int = 3, refinement_alpha: float = 0.5):
        super().__init__()
        terrain_data = load_terrain_artifact(terrain_path)
        self.backbone = FactorizedFNO2d(width=width)
        self.register_buffer("terrain_features", terrain_data["terrain_features"])
        self.corrector = IterativeCorrection(width=width)
        self.refinement_steps = refinement_steps
        self.refinement_alpha = refinement_alpha

        # GNN predicting directly without stream function
        self.gnn = TerrainGNNResidual(terrain_path, gnn_dense=False, terrain_data=terrain_data)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b, channels, height, width = state.shape
        flat = state.flatten(2).transpose(1, 2)
        terrain_flat = self.gnn.terrain_features.flatten(1).transpose(0, 1)
        indices = self.gnn.node_flat_indices
        nodes = torch.cat((flat[:, indices], terrain_flat[indices].unsqueeze(0).expand(b, -1, -1)), dim=-1)
        hidden = self.gnn.node_encoder(nodes)
        src, dst = self.gnn.edge_index
        for edge_mlp, update in zip(self.gnn.edge_mlps, self.gnn.node_updates):
            messages = edge_mlp(torch.cat((hidden[:, src], hidden[:, dst], self.gnn.edge_attr.unsqueeze(0).expand(b, -1, -1)), dim=-1))
            aggregate = torch.zeros_like(hidden).index_add_(1, dst, messages)
            hidden = hidden + update(torch.cat((hidden, aggregate), dim=-1))
        residual = self.gnn.gnn_scale * self.gnn.output(hidden)
        full = torch.zeros(b, channels, height * width, device=state.device, dtype=state.dtype)
        full[:, :, indices] = residual.transpose(1, 2).to(state.dtype)
        gnn_out = full.reshape(b, channels, height, width)

        output = self.backbone(state, self.terrain_features) + gnn_out
        for _ in range(self.refinement_steps - 1):
            correction = self.corrector(state, output)
            output = output + self.refinement_alpha * correction
        return output

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

def run_evaluation_suite_masked(model: nn.Module, test_rollout_loader: DataLoader, normalizer: PINONormalizer, clim_table: np.ndarray, test_times: pd.DatetimeIndex, device: torch.device, lats: np.ndarray, masks: dict):
    model.eval()
    max_steps = 12
    step_targets = [[] for _ in range(max_steps)]
    step_preds = [[] for _ in range(max_steps)]
    step_clims = [[] for _ in range(max_steps)]
    lead_map = {1: '6h', 4: '24h', 8: '48h', 12: '72h'}

    wind_u_list = []
    wind_v_list = []
    res_wind_u_list = []
    res_wind_v_list = []

    batch_offset = 0
    with torch.no_grad():
        for seq in test_rollout_loader:
            seq = seq.to(device)
            b = seq.shape[0]
            curr = seq[:, 0]
            for step in range(1, max_steps + 1):
                truth = seq[:, step]
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    pred = model(curr)

                if step == 1 and hasattr(model, 'gnn') and model.gnn is not None and hasattr(model.gnn, 'forward'):
                    try:
                        res_wind = model.gnn(curr, return_residual_wind_only=True)
                        res_wind_u_list.append(res_wind[:, 0].cpu())
                        res_wind_v_list.append(res_wind[:, 1].cpu())
                    except Exception:
                        pass
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

    all_preds_by_lead = {}
    all_tgts_by_lead = {}
    all_clims_by_lead = {}
    for step_idx, lead_str in lead_map.items():
        all_preds_by_lead[lead_str] = torch.cat(step_preds[step_idx - 1], dim=0)
        all_tgts_by_lead[lead_str] = torch.cat(step_targets[step_idx - 1], dim=0)
        all_clims_by_lead[lead_str] = torch.cat(step_clims[step_idx - 1], dim=0)

    results_by_mask = {}
    for mask_name, mask_tensor in masks.items():
        mask_leads = {}
        for lead_str in ['6h', '24h', '48h', '72h']:
            mask_leads[lead_str] = calculate_metrics_masked(
                all_preds_by_lead[lead_str],
                all_tgts_by_lead[lead_str],
                all_clims_by_lead[lead_str],
                normalizer, lats, mask_tensor
            )
        results_by_mask[mask_name] = {
            'leads': mask_leads,
            'lat_weighted_rmse_6h': mask_leads['6h']['aggregate_rmse'],
            'lat_weighted_acc_6h': mask_leads['6h']['aggregate_acc'],
            'nrmse_72h': mask_leads['72h']['aggregate_nrmse'],
        }

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
        'masks': results_by_mask,
        'wind_div_mean': div_mean,
        'wind_norm_div': norm_div,
        'res_wind_div_mean': r_div_mean,
        'res_wind_norm_div': r_norm_div,
    }

def evaluate_non_learned_baselines_masked(test_rollout_loader: DataLoader, normalizer: PINONormalizer, clim_table: np.ndarray, test_times: pd.DatetimeIndex, lats: np.ndarray, masks: dict):
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

    pers_by_mask = {}
    clim_by_mask = {}
    for mask_name, mask_tensor in masks.items():
        pers_leads = {}
        clim_leads = {}
        for step_idx, lead_str in lead_map.items():
            all_t = torch.cat(tgts[step_idx], dim=0)
            all_c = torch.cat(clims[step_idx], dim=0)
            pers_leads[lead_str] = calculate_metrics_masked(torch.cat(pers_preds[step_idx], dim=0), all_t, all_c, normalizer, lats, mask_tensor)
            clim_leads[lead_str] = calculate_metrics_masked(torch.cat(clim_preds[step_idx], dim=0), all_t, all_c, normalizer, lats, mask_tensor)
        pers_by_mask[mask_name] = {
            'leads': pers_leads,
            'lat_weighted_rmse_6h': pers_leads['6h']['aggregate_rmse'],
            'lat_weighted_acc_6h': pers_leads['6h']['aggregate_acc'],
            'nrmse_72h': pers_leads['72h']['aggregate_nrmse'],
        }
        clim_by_mask[mask_name] = {
            'leads': clim_leads,
            'lat_weighted_rmse_6h': clim_leads['6h']['aggregate_rmse'],
            'lat_weighted_acc_6h': clim_leads['6h']['aggregate_acc'],
            'nrmse_72h': clim_leads['72h']['aggregate_nrmse'],
        }
    return {'persistence': pers_by_mask, 'climatology': clim_by_mask}


def main_part_a(smoke=False):
    seeds = [0] if smoke else [0, 1, 2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Part A on {device} (smoke={smoke}, seeds={seeds})", flush=True)

    normalizer = PINONormalizer(json.load(open('DATASET/norm_stats.json')))
    clim_table = np.load('DATASET/train_climatology_smoothed.npy')
    ds_test = xr.open_zarr('DATASET/test_2021_2022.zarr')
    test_times = pd.to_datetime(ds_test.time.values)
    lats = ds_test.latitude.values

    # Masks
    m_full = torch.ones((15, 17), dtype=torch.bool)
    m_ring = torch.zeros((15, 17), dtype=torch.bool)
    m_ring[2:-2, 2:-2] = True
    m_free = torch.zeros((15, 17), dtype=torch.bool)
    m_free[5:-5, 5:-5] = True
    masks = {'full': m_full, 'outside_ring': m_ring, 'free_interior': m_free}

    train_zarr = 'DATASET/train_2018_2019.zarr'
    val_zarr = 'DATASET/val_2020.zarr'
    test_zarr = 'DATASET/test_2021_2022.zarr'

    warmup_epochs = 1 if smoke else 20
    rollout_epochs = 1 if smoke else 3

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

        if smoke:
            train_loader = DataLoader(Subset(train_loader.dataset, range(max(2, int(len(train_loader.dataset) * 0.02)))), batch_size=4, shuffle=True)
            val_loader = DataLoader(Subset(val_loader.dataset, range(max(2, int(len(val_loader.dataset) * 0.02)))), batch_size=4, shuffle=False)
            train_rollout_loader = DataLoader(Subset(train_rollout_loader.dataset, range(2)), batch_size=2, shuffle=True)
            test_rollout_loader = DataLoader(Subset(test_rollout_loader.dataset, range(2)), batch_size=2, shuffle=False)

        if sample_x is None:
            sample_x = next(iter(val_loader))[0][:1]

        for model_name, factory in model_factories.items():
            print(f"--- Training {model_name} | Seed {seed} ---", flush=True)
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

            if best_state is not None:
                model.load_state_dict(best_state)

            rollout_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            fine_tune_rollout_with_boundaries(model, train_rollout_loader, rollout_opt, device, epochs=rollout_epochs)

            lat_ms, peak_vram = benchmark_inference(model, sample_x, device)
            eval_dict = run_evaluation_suite_masked(model, test_rollout_loader, normalizer, clim_table, test_times, device, lats, masks)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            eval_dict['param_count'] = param_count
            eval_dict['latency_ms'] = lat_ms
            eval_dict['peak_vram_mb'] = peak_vram
            results_by_model[model_name].append(eval_dict)
            print(f"  {model_name} free_interior: RMSE={eval_dict['masks']['free_interior']['lat_weighted_rmse_6h']:.4f}, ACC={eval_dict['masks']['free_interior']['lat_weighted_acc_6h']:.4f}", flush=True)

    non_learned = evaluate_non_learned_baselines_masked(test_rollout_loader, normalizer, clim_table, test_times, lats, masks)

    part_a_summary = {'non_learned': non_learned, 'models': {}}
    for m_name, runs in results_by_model.items():
        summary = {
            'param_count': runs[0]['param_count'],
            'latency_ms_mean': float(np.mean([r['latency_ms'] for r in runs])),
            'peak_vram_mb_mean': float(np.mean([r['peak_vram_mb'] for r in runs])),
            'wind_norm_div_mean': float(np.mean([r['wind_norm_div'] for r in runs])),
            'res_wind_norm_div_mean': float(np.mean([r['res_wind_norm_div'] for r in runs])),
            'masks': {},
            'runs': runs,
        }
        for mask_name in ['full', 'outside_ring', 'free_interior']:
            rmse_vals = [r['masks'][mask_name]['lat_weighted_rmse_6h'] for r in runs]
            acc_vals = [r['masks'][mask_name]['lat_weighted_acc_6h'] for r in runs]
            nrmse_vals = [r['masks'][mask_name]['nrmse_72h'] for r in runs]
            summary['masks'][mask_name] = {
                'rmse_6h_mean': float(np.mean(rmse_vals)),
                'rmse_6h_std': float(np.std(rmse_vals)),
                'acc_6h_mean': float(np.mean(acc_vals)),
                'acc_6h_std': float(np.std(acc_vals)),
                'nrmse_72h_mean': float(np.mean(nrmse_vals)),
                'nrmse_72h_std': float(np.std(nrmse_vals)),
            }
        part_a_summary['models'][m_name] = summary

    if not smoke:
        with open('results/part_a_metrics.json', 'w') as f:
            json.dump(part_a_summary, f, indent=2)
        print("Wrote results/part_a_metrics.json", flush=True)

    return part_a_summary

if __name__ == '__main__':
    is_smoke = '--smoke' in sys.argv
    main_part_a(smoke=is_smoke)



def main_part_b(smoke=False):
    seeds = [0] if smoke else [0, 1, 2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Part B on {device} (smoke={smoke}, seeds={seeds})", flush=True)

    normalizer = PINONormalizer(json.load(open('DATASET/norm_stats.json')))
    clim_table = np.load('DATASET/train_climatology_smoothed.npy')
    ds_test = xr.open_zarr('DATASET/test_2021_2022.zarr')
    test_times = pd.to_datetime(ds_test.time.values)
    lats = ds_test.latitude.values

    m_full = torch.ones((15, 17), dtype=torch.bool)
    m_ring = torch.zeros((15, 17), dtype=torch.bool)
    m_ring[2:-2, 2:-2] = True
    m_free = torch.zeros((15, 17), dtype=torch.bool)
    m_free[5:-5, 5:-5] = True
    masks = {'full': m_full, 'outside_ring': m_ring, 'free_interior': m_free}

    train_zarr = 'DATASET/train_2018_2019.zarr'
    val_zarr = 'DATASET/val_2020.zarr'
    test_zarr = 'DATASET/test_2021_2022.zarr'

    warmup_epochs = 1 if smoke else 20
    rollout_epochs = 1 if smoke else 3

    variant_factories = {
        'v1_ffno_only': lambda: FFNOOnlyModel(width=35),
        'v2_ffno_dense_gnn': lambda: ProposedModel(terrain_path='DATASET/terrain_hp.npz', width=32, physics_mode='none', use_gnn_residual=True, gnn_dense=True, refinement_steps=3),
        'v3_ffno_random_sparse_gnn': lambda: FFNORandomSparseGNNModel(width=32),
        'v4_ffno_gated_unconstrained': lambda: FFNOGatedUnconstrainedModel(width=32),
    }

    results_by_variant = {v: [] for v in variant_factories}
    sample_x = None

    for seed in seeds:
        set_seed(seed)
        train_loader = get_single_step_loader(train_zarr, normalizer, batch_size=32, shuffle=True)
        val_loader = get_single_step_loader(val_zarr, normalizer, batch_size=32, shuffle=False)
        train_rollout_loader = get_rollout_loader(train_zarr, normalizer, sequence_length=4, batch_size=8, shuffle=True)
        test_rollout_loader = get_rollout_loader(test_zarr, normalizer, sequence_length=13, batch_size=8, shuffle=False)

        if smoke:
            train_loader = DataLoader(Subset(train_loader.dataset, range(max(2, int(len(train_loader.dataset) * 0.02)))), batch_size=4, shuffle=True)
            val_loader = DataLoader(Subset(val_loader.dataset, range(max(2, int(len(val_loader.dataset) * 0.02)))), batch_size=4, shuffle=False)
            train_rollout_loader = DataLoader(Subset(train_rollout_loader.dataset, range(2)), batch_size=2, shuffle=True)
            test_rollout_loader = DataLoader(Subset(test_rollout_loader.dataset, range(2)), batch_size=2, shuffle=False)

        if sample_x is None:
            sample_x = next(iter(val_loader))[0][:1]

        for v_name, factory in variant_factories.items():
            print(f"--- Training {v_name} | Seed {seed} ---", flush=True)
            model = factory().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warmup_epochs, eta_min=1e-5)

            best_val = float('inf')
            best_state = None
            for ep in range(1, warmup_epochs + 1):
                tr_res = train_one_epoch(model, train_loader, opt, device, use_physics_loss=False, max_grad_norm=1.0, normalizer=normalizer)
                vl_res = validate(model, val_loader, device, use_physics_loss=False, normalizer=normalizer)
                sched.step()
                vl_loss = vl_res[0] if isinstance(vl_res, tuple) else vl_res
                if vl_loss < best_val:
                    best_val = vl_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            rollout_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            fine_tune_rollout_with_boundaries(model, train_rollout_loader, rollout_opt, device, epochs=rollout_epochs)

            lat_ms, peak_vram = benchmark_inference(model, sample_x, device)
            eval_dict = run_evaluation_suite_masked(model, test_rollout_loader, normalizer, clim_table, test_times, device, lats, masks)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            eval_dict['param_count'] = param_count
            eval_dict['latency_ms'] = lat_ms
            eval_dict['peak_vram_mb'] = peak_vram
            results_by_variant[v_name].append(eval_dict)
            print(f"  {v_name} free_interior: RMSE={eval_dict['masks']['free_interior']['lat_weighted_rmse_6h']:.4f}, ACC={eval_dict['masks']['free_interior']['lat_weighted_acc_6h']:.4f}", flush=True)

    part_b_summary = {'variants': {}}
    for v_name, runs in results_by_variant.items():
        summary = {
            'param_count': runs[0]['param_count'],
            'latency_ms_mean': float(np.mean([r['latency_ms'] for r in runs])),
            'peak_vram_mb_mean': float(np.mean([r['peak_vram_mb'] for r in runs])),
            'wind_norm_div_mean': float(np.mean([r['wind_norm_div'] for r in runs])),
            'res_wind_norm_div_mean': float(np.mean([r['res_wind_norm_div'] for r in runs])),
            'masks': {},
            'runs': runs,
        }
        for mask_name in ['full', 'outside_ring', 'free_interior']:
            rmse_vals = [r['masks'][mask_name]['lat_weighted_rmse_6h'] for r in runs]
            acc_vals = [r['masks'][mask_name]['lat_weighted_acc_6h'] for r in runs]
            nrmse_vals = [r['masks'][mask_name]['nrmse_72h'] for r in runs]
            summary['masks'][mask_name] = {
                'rmse_6h_mean': float(np.mean(rmse_vals)),
                'rmse_6h_std': float(np.std(rmse_vals)),
                'acc_6h_mean': float(np.mean(acc_vals)),
                'acc_6h_std': float(np.std(acc_vals)),
                'nrmse_72h_mean': float(np.mean(nrmse_vals)),
                'nrmse_72h_std': float(np.std(nrmse_vals)),
            }
        part_b_summary['variants'][v_name] = summary

    if not smoke:
        with open('results/part_b_metrics.json', 'w') as f:
            json.dump(part_b_summary, f, indent=2)
        print("Wrote results/part_b_metrics.json", flush=True)

    return part_b_summary
