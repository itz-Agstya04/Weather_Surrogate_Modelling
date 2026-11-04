import os, sys, json, time, random
import numpy as np, pandas as pd, xarray as xr
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '.')
from data.pino_data_pipeline import PINONormalizer, get_single_step_loader, get_rollout_loader, CHANNEL_NAMES
from train.train_baselines import CoordinateFNO2d, train_one_epoch, validate, METRIC_NAMES
from models.proposed_model import ProposedModel, FactorizedFNO2d, TerrainGNNResidual, IterativeCorrection, load_terrain_artifact, STATE_CHANNELS, TERRAIN_CHANNELS
from run_rescore_ablations import (
    set_seed, apply_boundary_forcing, fine_tune_rollout_with_boundaries,
    benchmark_inference, run_evaluation_suite_masked, evaluate_non_learned_baselines_masked,
    FFNOOnlyModel, FFNORandomSparseGNNModel, FFNOGatedUnconstrainedModel,
    calculate_metrics_masked, compute_spherical_divergence_metrics
)


class ProposedZeroInit(nn.Module):
    def __init__(self, terrain_path='DATASET/terrain_hp.npz', width=32, refinement_steps=3, refinement_alpha=0.5, l2_weight=0.0):
        super().__init__()
        terrain_data = load_terrain_artifact(terrain_path)
        self.backbone = FactorizedFNO2d(width=width)
        self.register_buffer('terrain_features', terrain_data['terrain_features'])
        self.gnn = TerrainGNNResidual(terrain_path, gnn_dense=False, terrain_data=terrain_data)
        # zero-init output layer and gate
        nn.init.zeros_(self.gnn.output.weight)
        nn.init.zeros_(self.gnn.output.bias)
        self.gnn.gnn_scale = nn.Parameter(torch.zeros(1))
        self.corrector = IterativeCorrection(width=width)
        self.refinement_steps = refinement_steps
        self.refinement_alpha = refinement_alpha
        self.l2_weight = l2_weight

    def forward(self, state):
        gnn_out = self.gnn(state)
        output = self.backbone(state, self.terrain_features) + gnn_out
        for _ in range(self.refinement_steps - 1):
            correction = self.corrector(state, output)
            output = output + self.refinement_alpha * correction
        return output, gnn_out

    def forward_eval(self, state):
        out, _ = self.forward(state)
        return out


def train_one_epoch_t2(model, loader, opt, device, l2_weight=0.0, max_grad_norm=1.0):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=True):
            pred, gnn_out = model(x)
            loss = F.mse_loss(pred, y)
            if l2_weight > 0:
                loss = loss + l2_weight * gnn_out.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def validate_t2(model, loader, device):
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=True):
                pred, _ = model(x)
                loss = F.mse_loss(pred, y)
            total += loss.item(); n += 1
    return total / max(n, 1)


def fine_tune_rollout_t2(model, rollout_loader, optimizer, device, epochs=3):
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
                    pred, _ = model(curr)
                    loss = loss + F.mse_loss(pred, truth)
                    curr = apply_boundary_forcing(pred, truth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def eval_suite_t2(model, test_rollout_loader, normalizer, clim_table, test_times, device, lats, masks):
    class _Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.forward_eval(x)
        def __getattr__(self, k):
            if k == 'm': return super().__getattr__(k)
            return getattr(self.m, k)
    w = _Wrapper(model).to(device)
    return run_evaluation_suite_masked(w, test_rollout_loader, normalizer, clim_table, test_times, device, lats, masks)


def select_l2_weight(val_zarr, train_zarr, normalizer, clim_table, val_times, lats, masks, device):
    set_seed(0)
    train_loader = get_single_step_loader(train_zarr, normalizer, batch_size=32, shuffle=True)
    val_loader = get_single_step_loader(val_zarr, normalizer, batch_size=32, shuffle=False)
    sub_tr = max(2, int(len(train_loader.dataset) * 0.02))
    train_loader = DataLoader(Subset(train_loader.dataset, range(sub_tr)), batch_size=4, shuffle=True)
    val_loader = DataLoader(Subset(val_loader.dataset, range(max(2, int(len(val_loader.dataset) * 0.02)))), batch_size=4, shuffle=False)

    m_free = masks['free_interior']
    best_l2, best_val = None, float('inf')
    for l2w in [1e-3, 1e-2]:
        set_seed(0)
        model = ProposedZeroInit(l2_weight=l2w).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        train_one_epoch_t2(model, train_loader, opt, device, l2_weight=l2w)
        vl = validate_t2(model, val_loader, device)
        print(f'  L2={l2w} val_loss={vl:.5f}', flush=True)
        if vl < best_val:
            best_val, best_l2 = vl, l2w
    print(f'Selected L2 weight: {best_l2}', flush=True)
    return best_l2


def main_part_b_repair(smoke=False):
    seeds = [0] if smoke else [0, 1, 2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Part B Repair on {device} (smoke={smoke}, seeds={seeds})', flush=True)

    normalizer = PINONormalizer(json.load(open('DATASET/norm_stats.json')))
    clim_table = np.load('DATASET/train_climatology_smoothed.npy')
    ds_test = xr.open_zarr('DATASET/test_2021_2022.zarr')
    ds_val = xr.open_zarr('DATASET/val_2020.zarr')
    test_times = pd.to_datetime(ds_test.time.values)
    val_times = pd.to_datetime(ds_val.time.values)
    lats = ds_test.latitude.values

    m_full = torch.ones((15, 17), dtype=torch.bool)
    m_ring = torch.zeros((15, 17), dtype=torch.bool); m_ring[2:-2, 2:-2] = True
    m_free = torch.zeros((15, 17), dtype=torch.bool); m_free[5:-5, 5:-5] = True
    masks = {'full': m_full, 'outside_ring': m_ring, 'free_interior': m_free}

    train_zarr = 'DATASET/train_2018_2019.zarr'
    val_zarr = 'DATASET/val_2020.zarr'
    test_zarr = 'DATASET/test_2021_2022.zarr'

    warmup_epochs = 1 if smoke else 20
    rollout_epochs = 1 if smoke else 3

    # Select L2 weight
    print('Selecting L2 weight...', flush=True)
    l2_weight = select_l2_weight(val_zarr, train_zarr, normalizer, clim_table, val_times, lats, masks, device)

    # t2, t3, t4 factories
    variant_factories = {
        't2_proposed_zero_init': lambda: ProposedZeroInit(l2_weight=0.0),
        't3_proposed_zero_init_l2': lambda: ProposedZeroInit(l2_weight=l2_weight),
        't4_ffno_terrain_gnn': lambda: ProposedZeroInit(l2_weight=0.0),
    }

    results_by_variant = {v: [] for v in variant_factories}
    sample_x = None

    for seed in seeds:
        set_seed(seed)
        train_loader = get_single_step_loader(train_zarr, normalizer, batch_size=32, shuffle=True)
        val_loader_s = get_single_step_loader(val_zarr, normalizer, batch_size=32, shuffle=False)
        val_rollout = get_rollout_loader(val_zarr, normalizer, sequence_length=13, batch_size=8, shuffle=False)
        train_rollout_loader = get_rollout_loader(train_zarr, normalizer, sequence_length=4, batch_size=8, shuffle=True)
        test_rollout_loader = get_rollout_loader(test_zarr, normalizer, sequence_length=13, batch_size=8, shuffle=False)

        if smoke:
            sub_tr = max(2, int(len(train_loader.dataset) * 0.02))
            train_loader = DataLoader(Subset(train_loader.dataset, range(sub_tr)), batch_size=4, shuffle=True)
            val_loader_s = DataLoader(Subset(val_loader_s.dataset, range(max(2, int(len(val_loader_s.dataset) * 0.02)))), batch_size=4, shuffle=False)
            train_rollout_loader = DataLoader(Subset(train_rollout_loader.dataset, range(2)), batch_size=2, shuffle=True)
            test_rollout_loader = DataLoader(Subset(test_rollout_loader.dataset, range(2)), batch_size=2, shuffle=False)
            val_rollout = DataLoader(Subset(val_rollout.dataset, range(2)), batch_size=2, shuffle=False)

        if sample_x is None:
            sample_x = next(iter(val_loader_s))[0][:1]

        for v_name, factory in variant_factories.items():
            print(f'--- Training {v_name} | Seed {seed} ---', flush=True)
            l2w = l2_weight if v_name == 't3_proposed_zero_init_l2' else 0.0
            model = factory().to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warmup_epochs, eta_min=1e-5)

            best_val = float('inf'); best_state = None
            for ep in range(1, warmup_epochs + 1):
                train_one_epoch_t2(model, train_loader, opt, device, l2_weight=l2w)
                vl = validate_t2(model, val_loader_s, device)
                sched.step()
                if vl < best_val:
                    best_val = vl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_state:
                model.load_state_dict(best_state)

            rollout_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            fine_tune_rollout_t2(model, train_rollout_loader, rollout_opt, device, epochs=rollout_epochs)

            class _BenchWrap(nn.Module):
                def __init__(self2, m): super().__init__(); self2._m = m
                def forward(self2, x): return self2._m.forward_eval(x)
            lat_ms, peak_vram = benchmark_inference(_BenchWrap(model).to(device), sample_x, device)

            val_eval = eval_suite_t2(model, val_rollout, normalizer, clim_table, val_times, device, lats, masks)
            test_eval = eval_suite_t2(model, test_rollout_loader, normalizer, clim_table, test_times, device, lats, masks)

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            test_eval['param_count'] = param_count
            test_eval['latency_ms'] = lat_ms
            test_eval['peak_vram_mb'] = peak_vram
            test_eval['val_free_interior_rmse'] = val_eval['masks']['free_interior']['lat_weighted_rmse_6h']
            test_eval['val_free_interior_acc'] = val_eval['masks']['free_interior']['lat_weighted_acc_6h']

            results_by_variant[v_name].append(test_eval)
            print(f'  {v_name} val fi: RMSE={val_eval["masks"]["free_interior"]["lat_weighted_rmse_6h"]:.4f} ACC={val_eval["masks"]["free_interior"]["lat_weighted_acc_6h"]:.4f} | test fi: RMSE={test_eval["masks"]["free_interior"]["lat_weighted_rmse_6h"]:.4f} ACC={test_eval["masks"]["free_interior"]["lat_weighted_acc_6h"]:.4f}', flush=True)

    part_t_summary = {'variants': {}, 'l2_weight_selected': l2_weight}
    for v_name, runs in results_by_variant.items():
        summary = {
            'param_count': runs[0]['param_count'],
            'latency_ms_mean': float(np.mean([r['latency_ms'] for r in runs])),
            'peak_vram_mb_mean': float(np.mean([r['peak_vram_mb'] for r in runs])),
            'wind_norm_div_mean': float(np.mean([r['wind_norm_div'] for r in runs])),
            'res_wind_norm_div_mean': float(np.mean([r['res_wind_norm_div'] for r in runs])),
            'val_free_interior_rmse_mean': float(np.mean([r['val_free_interior_rmse'] for r in runs])),
            'val_free_interior_acc_mean': float(np.mean([r['val_free_interior_acc'] for r in runs])),
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
        part_t_summary['variants'][v_name] = summary

    if not smoke:
        os.makedirs('results', exist_ok=True)
        with open('results/part_t_metrics.json', 'w') as f:
            json.dump(part_t_summary, f, indent=2)
        print('Wrote results/part_t_metrics.json', flush=True)

    return part_t_summary


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    main_part_b_repair(smoke=smoke)
