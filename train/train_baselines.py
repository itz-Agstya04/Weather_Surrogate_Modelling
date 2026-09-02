"""
Baseline 1 (Dense 3D FNO) and Baseline 2 (Standard PINO) Implementation, Training,
Rollout Fine-Tuning, and Test Evaluation Suite.

Himachal Pradesh Microclimates PINO Project.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import pipeline components
# Allow both `python train/train_baselines.py` and
# `python -m train.train_baselines` from the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pino_data_pipeline import (
    PINONormalizer,
    HPERA5SingleStepDataset,
    HPERA5RolloutDataset,
    get_single_step_loader,
    get_rollout_loader,
    SURFACE_VARS,
    UPPER_AIR_VAR,
    TARGET_LEVELS
)


# =====================================================================
# 1. DENSE 3D FOURIER NEURAL OPERATOR (BASELINE 1 & 2 ARCHITECTURE)
# =====================================================================

class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution Layer over (Depth=8, Latitude=15, Longitude=17).
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Depth modes (max 4)
        self.modes2 = modes2  # Lat modes (max 8)
        self.modes3 = modes3  # Lon modes (max 9)

        scale = (1.0 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, d, h, w), (in_channel, out_channel, d, h, w) -> (batch, out_channel, d, h, w)
        return torch.einsum("bixyz,ioxyz->boxyz", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        dtype = x.dtype
        # Cast to float32 for cuFFT precision support on non-power-of-2 spatial grids [8, 15, 17]
        x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1, m2, m3 = self.modes1, self.modes2, self.modes3

        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4)

        x_out = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x_out.to(dtype=dtype)


class Dense3DFNO(nn.Module):
    """
    Dense 3D FNO Model.
    Reshapes 8-channel 2D state (B, 8, Lat, Lon) into 3D volume (B, 1, 8, Lat, Lon),
    applies 3D spectral convolution blocks, and projects back to (B, 8, Lat, Lon).
    """
    def __init__(self, in_dim: int = 8, out_dim: int = 8, width: int = 32, modes1: int = 4, modes2: int = 6, modes3: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width

        # Uplift: 1 channel 3D volume -> width channels
        self.p = nn.Conv3d(1, width, kernel_size=1)

        # 4 Spectral 3D Convolution Blocks
        self.conv0 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = SpectralConv3d(width, width, modes1, modes2, modes3)

        self.w0 = nn.Conv3d(width, width, kernel_size=1)
        self.w1 = nn.Conv3d(width, width, kernel_size=1)
        self.w2 = nn.Conv3d(width, width, kernel_size=1)
        self.w3 = nn.Conv3d(width, width, kernel_size=1)

        # Projection back to 1 channel 3D volume
        self.q1 = nn.Conv3d(width, width // 2, kernel_size=1)
        self.q2 = nn.Conv3d(width // 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C=8, H=15, W=17)
        B, C, H, W = x.shape
        x_3d = x.unsqueeze(1)  # (B, 1, D=8, H=15, W=17)

        x_feat = self.p(x_3d)

        # Block 0
        x1 = self.conv0(x_feat) + self.w0(x_feat)
        x_feat = F.gelu(x1)

        # Block 1
        x2 = self.conv1(x_feat) + self.w1(x_feat)
        x_feat = F.gelu(x2)

        # Block 2
        x3 = self.conv2(x_feat) + self.w2(x_feat)
        x_feat = F.gelu(x3)

        # Block 3
        x4 = self.conv3(x_feat) + self.w3(x_feat)
        x_feat = F.gelu(x4)

        # Output projection
        out_3d = self.q1(x_feat)
        out_3d = F.gelu(out_3d)
        out_3d = self.q2(out_3d)  # (B, 1, D=8, H=15, W=17)

        out = out_3d.squeeze(1)  # (B, 8, H=15, W=17)
        return out


# =====================================================================
# 2. STEP 3: PHYSICS LOSS TERMS FOR BASELINE 2 (STANDARD PINO)
# =====================================================================

def compute_divergence_loss(pred_state: torch.Tensor) -> torch.Tensor:
    """
    Mass / continuity conservation loss: penalizes horizontal wind divergence (10m_u, 10m_v).
    pred_state: (B, C=8, H=15, W=17)
    10m_u is index 1, 10m_v is index 2.

    NOTE: We use dimensionless grid-unit differences (no physical dx/dy scaling).
    Physical scaling by 1/(2*25000) makes gradients ~5e-8, causing div²~5e-16 -> numerically zero.
    Dimensionless differences are O(0.1) in normalized space, giving a useful gradient signal.
    """
    u = pred_state[:, 1, :, :]  # (B, H, W)
    v = pred_state[:, 2, :, :]  # (B, H, W)

    # Central finite differences in grid-unit space
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / 2.0  # (B, H, W-2)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / 2.0  # (B, H-2, W)

    # Align spatial dimensions (H-2, W-2)
    du_dx_crop = du_dx[:, 1:-1, :]
    dv_dy_crop = dv_dy[:, :, 1:-1]

    div = du_dx_crop + dv_dy_crop
    return torch.mean(div ** 2)


def compute_hydrostatic_loss(pred_state: torch.Tensor) -> torch.Tensor:
    """
    Hydrostatic balance loss: penalizes geopotential inversions across pressure levels.
    Geopotential levels [1000, 850, 700, 500, 300] hPa are indices 3, 4, 5, 6, 7.

    NOTE: pred_state is in normalized space. Since normalization is per-level with
    different means/stds, the ordering of normalized values is not monotone by design.
    We therefore penalize only inconsistency in the *sign* of differences: in raw space,
    phi increases with altitude (lower pressure), so we soft-penalize negative differences
    in a relative sense (diff < -1 in normalized units signals a clear inversion).
    """
    phi = pred_state[:, 3:8, :, :]         # (B, 5, H, W) — normalized geopotential levels
    diffs = phi[:, 1:, :, :] - phi[:, :-1, :, :]  # (B, 4, H, W)

    # Only penalize clear inversions (diffs < -1.0 in normalized units)
    inversion_penalty = torch.relu(-1.0 - diffs)
    return torch.mean(inversion_penalty ** 2)


# =====================================================================
# 3. STEP 1 & SHARED TRAINING LOOP
# =====================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_physics_loss: bool = False,
    lambda_div: float = 0.01,
    lambda_hydro: float = 0.01,
    max_grad_norm: float = 1.0
):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_div_loss = 0.0
    total_hydro_loss = 0.0

    use_cuda = device.type == 'cuda'

    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            pred = model(inp)
            data_loss = F.mse_loss(pred, tgt)

            if use_physics_loss:
                div_loss = compute_divergence_loss(pred)
                hydro_loss = compute_hydrostatic_loss(pred)
                loss = data_loss + lambda_div * div_loss + lambda_hydro * hydro_loss
            else:
                div_loss = torch.tensor(0.0, device=device)
                hydro_loss = torch.tensor(0.0, device=device)
                loss = data_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inp.size(0)
        total_data_loss += data_loss.item() * inp.size(0)
        total_div_loss += div_loss.item() * inp.size(0)
        total_hydro_loss += hydro_loss.item() * inp.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_data_loss / n, total_div_loss / n, total_hydro_loss / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_physics_loss: bool = False,
    lambda_div: float = 0.01,
    lambda_hydro: float = 0.01
):
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0

    use_cuda = device.type == 'cuda'

    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            pred = model(inp)
            data_loss = F.mse_loss(pred, tgt)

            if use_physics_loss:
                div_loss = compute_divergence_loss(pred)
                hydro_loss = compute_hydrostatic_loss(pred)
                loss = data_loss + lambda_div * div_loss + lambda_hydro * hydro_loss
            else:
                loss = data_loss

        total_loss += loss.item() * inp.size(0)
        total_data_loss += data_loss.item() * inp.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_data_loss / n


# =====================================================================
# 4. STEP 5: ROLLOUT AUTOREGRESSIVE FINE-TUNING
# =====================================================================

def fine_tune_rollout(
    model: nn.Module,
    rollout_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 3
):
    model.train()
    use_cuda = device.type == 'cuda'

    print(f"\n--- Starting Autoregressive Rollout Fine-Tuning ({epochs} epochs) ---", flush=True)

    for epoch in range(1, epochs + 1):
        total_rollout_loss = 0.0
        n_samples = 0

        for seq_batch in rollout_loader:
            # seq_batch shape: (B, T, C=8, Lat=15, Lon=17)
            seq_batch = seq_batch.to(device)
            B, T, C, H, W = seq_batch.shape

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                current_state = seq_batch[:, 0, :, :, :]  # Initial state t=0
                loss = 0.0

                for t_step in range(1, T):
                    target_state = seq_batch[:, t_step, :, :, :]
                    pred_state = model(current_state)

                    loss = loss + F.mse_loss(pred_state, target_state)
                    current_state = pred_state  # Autoregressive feed forward

                loss = loss / (T - 1)

            loss.backward()
            optimizer.step()

            total_rollout_loss += loss.item() * B
            n_samples += B

        avg_loss = total_rollout_loss / n_samples
        print(f"  Rollout Epoch {epoch}/{epochs} -> Loss: {avg_loss:.6f}", flush=True)


# =====================================================================
# 5. STEP 6: EVALUATION METRICS (LAT-WEIGHTED RMSE & ACC)
# =====================================================================

@torch.no_grad()
def evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    normalizer: PINONormalizer,
    device: torch.device,
    lats: np.ndarray = np.linspace(33.5, 30.0, 15)
):
    model.eval()
    use_cuda = device.type == 'cuda'

    # Compute latitude weights cos(lat * pi / 180)
    w_lat = np.cos(np.radians(lats)).astype(np.float32)
    w_lat = torch.from_numpy(w_lat).reshape(1, 1, -1, 1).to(device)  # (1, 1, Lat, 1)

    all_preds_raw = []
    all_tgts_raw = []

    # Reset peak GPU memory tracking
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.perf_counter()
    n_batches = 0

    for inp, tgt in test_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            pred = model(inp)

        # Denormalize predictions and targets to raw physical units
        # Cast to float32 first: autocast produces float16 tensors; geopotential std * values
        # can exceed float16 max (65504) causing overflow -> inf RMSE
        pred_raw = normalizer.denormalize(pred.float())
        tgt_raw = normalizer.denormalize(tgt.float())

        all_preds_raw.append(pred_raw.cpu())
        all_tgts_raw.append(tgt_raw.cpu())
        n_batches += 1

    t_end = time.perf_counter()
    inference_latency_ms = ((t_end - t_start) / max(1, n_batches)) * 1000.0

    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if use_cuda else 0.0

    preds_cat = torch.cat(all_preds_raw, dim=0)  # (N, 8, 15, 17) in raw units
    tgts_cat = torch.cat(all_tgts_raw, dim=0)

    w_lat_cpu = w_lat.cpu()

    # 1. Latitude-Weighted RMSE in raw units
    sq_err = (preds_cat - tgts_cat) ** 2  # (N, 8, 15, 17)
    weighted_sq_err = sq_err * w_lat_cpu
    lat_rmse_raw = torch.sqrt(
        torch.sum(weighted_sq_err) / (torch.sum(w_lat_cpu) * preds_cat.size(0) * preds_cat.size(1) * preds_cat.size(3))
    ).item()

    # 1b. Normalized RMSE: divide each channel's MSE by its train-split std^2
    channel_stds = torch.from_numpy(normalizer.channel_stds.reshape(1, -1, 1, 1))  # (1, 8, 1, 1)
    sq_err_norm = sq_err / (channel_stds ** 2 + 1e-8)
    weighted_sq_err_norm = sq_err_norm * w_lat_cpu
    lat_rmse_norm = torch.sqrt(
        torch.sum(weighted_sq_err_norm) / (torch.sum(w_lat_cpu) * preds_cat.size(0) * preds_cat.size(1) * preds_cat.size(3))
    ).item()

    # 2. Latitude-Weighted ACC (Anomaly Correlation)
    tgt_mean = torch.mean(tgts_cat, dim=0, keepdim=True)
    pred_ano = preds_cat - tgt_mean
    tgt_ano = tgts_cat - tgt_mean

    cov = torch.sum(w_lat_cpu * pred_ano * tgt_ano)
    var_pred = torch.sum(w_lat_cpu * (pred_ano ** 2))
    var_tgt = torch.sum(w_lat_cpu * (tgt_ano ** 2))

    lat_acc = (cov / (torch.sqrt(var_pred * var_tgt) + 1e-8)).item()

    return {
        'lat_weighted_rmse': lat_rmse_raw,
        'lat_weighted_rmse_norm': lat_rmse_norm,
        'lat_weighted_acc': lat_acc,
        'peak_gpu_mem_mb': peak_gpu_mem_mb,
        'inference_latency_ms': inference_latency_ms
    }


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

def run_baseline_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================", flush=True)
    print(f"   RUNNING BASELINE MODEL EXPERIMENTS ON DEVICE: {device}   ", flush=True)
    print(f"=======================================================", flush=True)

    train_zarr = os.path.join("DATASET", "train_2018_2019.zarr")
    val_zarr = os.path.join("DATASET", "val_2020.zarr")
    test_zarr = os.path.join("DATASET", "test_2021_2022.zarr")
    stats_file = os.path.join("DATASET", "norm_stats.json")

    with open(stats_file, 'r') as f:
        stats_dict = json.load(f)
    normalizer = PINONormalizer(stats_dict)

    # Loaders
    train_loader = get_single_step_loader(train_zarr, normalizer, batch_size=32, shuffle=True)
    val_loader = get_single_step_loader(val_zarr, normalizer, batch_size=32, shuffle=False)
    test_loader = get_single_step_loader(test_zarr, normalizer, batch_size=32, shuffle=False)

    # Fine-tune only on train windows.  Validation remains untouched for
    # early stopping and model selection.
    train_rollout_loader = get_rollout_loader(train_zarr, normalizer, sequence_length=4, batch_size=8, shuffle=True)

    results_summary = {}

    epochs = 30
    patience = 6          # early stopping patience
    max_grad_norm = 1.0   # gradient clipping

    # -----------------------------------------------------------------
    # BASELINE 1: DENSE 3D FNO (DATA LOSS ONLY)
    # -----------------------------------------------------------------
    print("\n-------------------------------------------------------", flush=True)
    print("      STEP 2: TRAINING BASELINE 1 (DENSE 3D FNO)      ", flush=True)
    print("-------------------------------------------------------", flush=True)

    b1_model = Dense3DFNO(in_dim=8, out_dim=8, width=32).to(device)
    b1_optimizer = torch.optim.AdamW(b1_model.parameters(), lr=3e-4, weight_decay=1e-4)
    b1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(b1_optimizer, T_max=epochs, eta_min=1e-5)

    b1_train_losses = []
    b1_val_losses = []
    b1_best_val = float('inf')
    b1_no_improve = 0
    b1_best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_data, _, _ = train_one_epoch(
            b1_model, train_loader, b1_optimizer, device,
            use_physics_loss=False, max_grad_norm=max_grad_norm
        )
        vl_loss, vl_data = validate(b1_model, val_loader, device, use_physics_loss=False)
        b1_scheduler.step()

        b1_train_losses.append(tr_loss)
        b1_val_losses.append(vl_loss)

        improved = vl_loss < b1_best_val
        if improved:
            b1_best_val = vl_loss
            b1_best_state = {k: v.cpu().clone() for k, v in b1_model.state_dict().items()}
            b1_no_improve = 0
        else:
            b1_no_improve += 1

        marker = " *" if improved else ""
        print(f"  B1 Epoch {epoch:2d}/{epochs} -> Train: {tr_loss:.5f} | Val: {vl_loss:.5f} | LR: {b1_scheduler.get_last_lr()[0]:.2e}{marker}", flush=True)
        assert not np.isnan(tr_loss) and not np.isnan(vl_loss), "Baseline 1 Loss is NaN!"

        if b1_no_improve >= patience:
            print(f"  [Early Stop] No val improvement for {patience} epochs. Best val: {b1_best_val:.5f}", flush=True)
            break

    # Restore best weights
    if b1_best_state is not None:
        b1_model.load_state_dict({k: v.to(device) for k, v in b1_best_state.items()})
        print(f"  [Restored] Best B1 weights (val={b1_best_val:.5f})", flush=True)

    # Rollout Fine-Tuning for Baseline 1
    print("\n--- Step 5: Rollout Fine-Tuning Baseline 1 ---", flush=True)
    b1_ft_optimizer = torch.optim.AdamW(b1_model.parameters(), lr=1e-4, weight_decay=1e-4)
    fine_tune_rollout(b1_model, train_rollout_loader, b1_ft_optimizer, device, epochs=3)

    # Evaluate Baseline 1 on Test Split
    print("\n--- Step 6: Evaluating Baseline 1 on Test Split ---", flush=True)
    b1_metrics = evaluate_model_on_test(b1_model, test_loader, normalizer, device)
    results_summary['Baseline 1 (Dense 3D FNO)'] = b1_metrics
    print(f"  [OK] B1 Test RMSE: {b1_metrics['lat_weighted_rmse']:.2f} | Norm-RMSE: {b1_metrics['lat_weighted_rmse_norm']:.4f} | ACC: {b1_metrics['lat_weighted_acc']:.4f} | Mem: {b1_metrics['peak_gpu_mem_mb']:.1f}MB | Lat: {b1_metrics['inference_latency_ms']:.2f}ms")

    # -----------------------------------------------------------------
    # BASELINE 2: STANDARD PINO (WITH SOFT PHYSICS CONSTRAINTS)
    # -----------------------------------------------------------------
    print("\n-------------------------------------------------------", flush=True)
    print("      STEP 4: TRAINING BASELINE 2 (STANDARD PINO)     ", flush=True)
    print("-------------------------------------------------------", flush=True)

    b2_model = Dense3DFNO(in_dim=8, out_dim=8, width=32).to(device)
    b2_optimizer = torch.optim.AdamW(b2_model.parameters(), lr=3e-4, weight_decay=1e-4)
    b2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(b2_optimizer, T_max=epochs, eta_min=1e-5)

    b2_train_losses = []
    b2_val_losses = []
    b2_best_val = float('inf')
    b2_no_improve = 0
    b2_best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_data, tr_div, tr_hydro = train_one_epoch(
            b2_model, train_loader, b2_optimizer, device,
            use_physics_loss=True, lambda_div=0.01, lambda_hydro=0.1,
            max_grad_norm=max_grad_norm
        )
        vl_loss, vl_data = validate(b2_model, val_loader, device, use_physics_loss=True, lambda_div=0.01, lambda_hydro=0.1)
        b2_scheduler.step()

        b2_train_losses.append(tr_loss)
        b2_val_losses.append(vl_loss)

        improved = vl_loss < b2_best_val
        if improved:
            b2_best_val = vl_loss
            b2_best_state = {k: v.cpu().clone() for k, v in b2_model.state_dict().items()}
            b2_no_improve = 0
        else:
            b2_no_improve += 1

        marker = " *" if improved else ""
        print(f"  B2 Epoch {epoch:2d}/{epochs} -> Total: {tr_loss:.5f} (Data: {tr_data:.5f} Div: {tr_div:.7f} Hydro: {tr_hydro:.7f}) | Val: {vl_loss:.5f}{marker}", flush=True)
        assert not np.isnan(tr_loss) and not np.isnan(vl_loss), "Baseline 2 Loss is NaN!"

        if b2_no_improve >= patience:
            print(f"  [Early Stop] No val improvement for {patience} epochs. Best val: {b2_best_val:.5f}", flush=True)
            break

    # Restore best weights
    if b2_best_state is not None:
        b2_model.load_state_dict({k: v.to(device) for k, v in b2_best_state.items()})
        print(f"  [Restored] Best B2 weights (val={b2_best_val:.5f})", flush=True)

    # Rollout Fine-Tuning for Baseline 2
    print("\n--- Step 5: Rollout Fine-Tuning Baseline 2 ---", flush=True)
    b2_ft_optimizer = torch.optim.AdamW(b2_model.parameters(), lr=1e-4, weight_decay=1e-4)
    fine_tune_rollout(b2_model, train_rollout_loader, b2_ft_optimizer, device, epochs=3)

    # Evaluate Baseline 2 on Test Split
    print("\n--- Step 6: Evaluating Baseline 2 on Test Split ---", flush=True)
    b2_metrics = evaluate_model_on_test(b2_model, test_loader, normalizer, device)
    results_summary['Baseline 2 (Standard PINO)'] = b2_metrics
    print(f"  [OK] B2 Test RMSE: {b2_metrics['lat_weighted_rmse']:.2f} | Norm-RMSE: {b2_metrics['lat_weighted_rmse_norm']:.4f} | ACC: {b2_metrics['lat_weighted_acc']:.4f} | Mem: {b2_metrics['peak_gpu_mem_mb']:.1f}MB | Lat: {b2_metrics['inference_latency_ms']:.2f}ms")

    # Save Results Table JSON
    results_file = os.path.join("DATASET", "baseline_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n=======================================================", flush=True)
    print("            FINAL BASELINES COMPARISON TABLE           ", flush=True)
    print("=======================================================", flush=True)
    print(f"{'Model':<28s} | {'Raw RMSE':>10s} | {'Norm RMSE':>9s} | {'Lat ACC':>8s} | {'GPU MB':>8s} | {'Lat ms':>8s}")
    print("-" * 85)
    for model_name, metrics in results_summary.items():
        rmse_str  = f"{metrics['lat_weighted_rmse']:.2f}"      if np.isfinite(metrics['lat_weighted_rmse'])      else "inf"
        nrmse_str = f"{metrics['lat_weighted_rmse_norm']:.4f}" if np.isfinite(metrics['lat_weighted_rmse_norm']) else "inf"
        acc_str   = f"{metrics['lat_weighted_acc']:.4f}"       if np.isfinite(metrics['lat_weighted_acc'])       else "nan"
        print(f"{model_name:<28s} | {rmse_str:>10s} | {nrmse_str:>9s} | {acc_str:>8s} | {metrics['peak_gpu_mem_mb']:>8.1f} | {metrics['inference_latency_ms']:>8.2f}")
    print("=======================================================", flush=True)


if __name__ == "__main__":
    run_baseline_experiments()
