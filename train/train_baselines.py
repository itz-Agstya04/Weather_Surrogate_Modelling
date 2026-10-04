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
    UPPER_AIR_TEMPERATURE_VAR,
    TARGET_LEVELS,
    CHANNEL_NAMES as STATE_CHANNEL_NAMES
)


def save_checkpoint(state_dict: dict, name: str, directory: str = "DATASET/checkpoints") -> str:
    """Persist a CPU state dict and return its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.pt")
    torch.save(state_dict, path)
    return path


def load_checkpoint(model: nn.Module, path: str, device: torch.device = None) -> nn.Module:
    """Load a saved state dict into ``model`` for inference or evaluation."""
    try:
        state_dict = torch.load(path, map_location=device or torch.device("cpu"), weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        state_dict = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model


# =====================================================================
# 1. 2D FOURIER NEURAL OPERATOR (BASELINE ARCHITECTURE)
# =====================================================================

class SpectralConv2d(nn.Module):
    """
    2D spectral convolution over the latitude/longitude grid.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (1.0 / (in_channels * out_channels))
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        dtype = x.dtype
        height, width = x.shape[-2:]
        x_ft = torch.fft.rfft2(x.float(), dim=(-2, -1))

        out_ft = torch.zeros(
            batchsize, self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, height // 2 + 1)
        m2 = min(self.modes2, width // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        out_ft[:, :, :m1, -m2:] = self.compl_mul2d(x_ft[:, :, :m1, -m2:], self.weights3[:, :, :m1, :m2])
        out_ft[:, :, -m1:, -m2:] = self.compl_mul2d(x_ft[:, :, -m1:, -m2:], self.weights4[:, :, :m1, :m2])

        x_out = torch.fft.irfft2(out_ft, s=(height, width))
        return x_out.to(dtype=dtype)


class CoordinateFNO2d(nn.Module):
    """
    2D FNO with four fixed geographic coordinate channels.  The physical
    state remains (B, 13, latitude, longitude); coordinates are auxiliary input.
    """
    def __init__(self, in_dim: int = len(STATE_CHANNEL_NAMES), out_dim: int = len(STATE_CHANNEL_NAMES), width: int = 32, modes1: int = 8, modes2: int = 6):
        super().__init__()
        self.in_dim = in_dim + 4
        self.out_dim = out_dim
        self.width = width

        self.p = nn.Conv2d(self.in_dim, width, kernel_size=1)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, kernel_size=1)
        self.w1 = nn.Conv2d(width, width, kernel_size=1)
        self.w2 = nn.Conv2d(width, width, kernel_size=1)
        self.w3 = nn.Conv2d(width, width, kernel_size=1)

        self.q1 = nn.Conv2d(width, width // 2, kernel_size=1)
        self.q2 = nn.Conv2d(width // 2, out_dim, kernel_size=1)

    @staticmethod
    def coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lat = torch.linspace(33.5, 30.0, height, device=device, dtype=dtype) * (np.pi / 180.0)
        lon = torch.linspace(75.5, 79.5, width, device=device, dtype=dtype) * (np.pi / 180.0)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        return torch.stack((torch.sin(lat_grid), torch.cos(lat_grid),
                            torch.sin(lon_grid), torch.cos(lon_grid)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        coords = self.coordinates(height, width, x.device, x.dtype).unsqueeze(0).expand(x.size(0), -1, -1, -1)
        x_feat = self.p(torch.cat((x, coords), dim=1))

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
        out = self.q2(F.gelu(self.q1(x_feat)))
        return out


# =====================================================================
# 2. STEP 3: PHYSICS LOSS TERMS FOR BASELINE 2 (STANDARD PINO)
# =====================================================================

def compute_divergence_loss(pred_state: torch.Tensor, normalizer: PINONormalizer = None) -> torch.Tensor:
    """
    Mass / continuity conservation loss: penalizes horizontal wind divergence (10m_u, 10m_v).
    pred_state: (B, C=13, H=15, W=17)
    10m_u is index 1, 10m_v is index 2.

    NOTE: We use dimensionless grid-unit differences (no physical dx/dy scaling).
    Physical scaling by 1/(2*25000) makes gradients ~5e-8, causing div²~5e-16 -> numerically zero.
    Dimensionless differences are O(0.1) in normalized space, giving a useful gradient signal.
    """
    if normalizer is not None:
        u = pred_state[:, 1].float() * float(normalizer.channel_stds[1].item()) + float(normalizer.channel_means[1].item())
        v = pred_state[:, 2].float() * float(normalizer.channel_stds[2].item()) + float(normalizer.channel_means[2].item())
    else:
        u, v = pred_state[:, 1].float(), pred_state[:, 2].float()

    height, width = u.shape[-2:]
    lat = torch.linspace(33.5, 30.0, height, device=pred_state.device, dtype=torch.float32)
    dlat = torch.abs(lat[1] - lat[0]) * (np.pi / 180.0)
    dlon = (4.0 / (width - 1)) * (np.pi / 180.0)
    earth_radius = 6_371_000.0
    dy = earth_radius * dlat
    dx = earth_radius * torch.cos(lat * (np.pi / 180.0)) * dlon

    # Central finite differences in grid-unit space
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dx.view(1, -1, 1))
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2.0 * dy)

    # Align spatial dimensions (H-2, W-2)
    du_dx_crop = du_dx[:, 1:-1, :]
    dv_dy_crop = dv_dy[:, :, 1:-1]

    div = du_dx_crop + dv_dy_crop
    # Divergence is in s^-1. Normalize by a fixed characteristic value so
    # the loss is numerically trainable without changing the physical units
    # of the residual itself.
    return torch.mean((div / 1e-5) ** 2)


def compute_hydrostatic_loss(
    pred_state: torch.Tensor,
    normalizer: PINONormalizer = None,
) -> torch.Tensor:
    """
    Hydrostatic balance loss: penalizes geopotential inversions across pressure levels.
    Geopotential levels [1000, 850, 700, 500, 300] hPa are indices 3, 4, 5, 6, 7.

    pred_state is normalized by the data pipeline. If a normalizer is supplied,
    geopotential is converted back to physical units before checking that it
    increases as pressure decreases.
    """
    # Keep this calculation in float32: raw 300 hPa geopotential is ~90,000,
    # which can overflow float16 under CUDA autocast.
    phi = pred_state[:, 3:8, :, :].float()  # (B, 5, H, W)
    temperature = pred_state[:, 8:13, :, :].float()
    if normalizer is not None:
        phi_means = torch.as_tensor(normalizer.channel_means[3:8], dtype=torch.float32, device=pred_state.device)
        phi_stds = torch.as_tensor(normalizer.channel_stds[3:8], dtype=torch.float32, device=pred_state.device)
        temp_means = torch.as_tensor(normalizer.channel_means[8:13], dtype=torch.float32, device=pred_state.device)
        temp_stds = torch.as_tensor(normalizer.channel_stds[8:13], dtype=torch.float32, device=pred_state.device)
        phi = phi.float() * phi_stds + phi_means
        temperature = temperature * temp_stds + temp_means

    pressure = torch.tensor([1000, 850, 700, 500, 300], dtype=torch.float32, device=pred_state.device) * 100.0
    dp = pressure[1:] - pressure[:-1]
    phi_dp = (phi[:, 1:] - phi[:, :-1]) / dp.view(1, -1, 1, 1)
    p_mid = ((pressure[1:] + pressure[:-1]) / 2.0).view(1, -1, 1, 1)
    temp_mid = (temperature[:, 1:] + temperature[:, :-1]) / 2.0
    expected = -287.05 * temp_mid / p_mid
    return torch.mean((phi_dp - expected).square())


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
    max_grad_norm: float = 1.0,
    normalizer: PINONormalizer = None,
):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_div_loss = 0.0
    total_hydro_loss = 0.0
    total_physics_grad_norm = 0.0

    use_cuda = device.type == 'cuda'

    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            pred = model(inp)
            data_loss = F.mse_loss(pred, tgt)

            if use_physics_loss:
                div_loss = compute_divergence_loss(pred, normalizer=normalizer)
                hydro_loss = compute_hydrostatic_loss(pred, normalizer=normalizer)
                loss = data_loss + lambda_div * div_loss + lambda_hydro * hydro_loss
            else:
                div_loss = torch.tensor(0.0, device=device)
                hydro_loss = torch.tensor(0.0, device=device)
                loss = data_loss

        if use_physics_loss:
            physics_term = lambda_div * div_loss + lambda_hydro * hydro_loss
            physics_grads = torch.autograd.grad(
                physics_term, tuple(model.parameters()), retain_graph=True, allow_unused=True
            )
            physics_grad_norm = torch.sqrt(sum(
                grad.detach().float().square().sum() for grad in physics_grads if grad is not None
            )).item()
        else:
            physics_grad_norm = 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inp.size(0)
        total_data_loss += data_loss.item() * inp.size(0)
        total_div_loss += div_loss.item() * inp.size(0)
        total_hydro_loss += hydro_loss.item() * inp.size(0)
        total_physics_grad_norm += physics_grad_norm * inp.size(0)

    n = len(loader.dataset)
    return (total_loss / n, total_data_loss / n, total_div_loss / n,
            total_hydro_loss / n, total_physics_grad_norm / n)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_physics_loss: bool = False,
    lambda_div: float = 0.01,
    lambda_hydro: float = 0.01,
    normalizer: PINONormalizer = None,
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
                div_loss = compute_divergence_loss(pred, normalizer=normalizer)
                hydro_loss = compute_hydrostatic_loss(pred, normalizer=normalizer)
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
            # seq_batch shape: (B, T, C=13, Lat, Lon)
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

METRIC_NAMES = (
    ['T2m', 'U10', 'V10']
    + [f'Z{level}' for level in TARGET_LEVELS]
    + [f'T{level}' for level in TARGET_LEVELS]
)


def calculate_metrics(preds_raw: torch.Tensor, tgts_raw: torch.Tensor,
                      normalizer: PINONormalizer, lats: np.ndarray,
                      climatology: torch.Tensor = None) -> dict:
    """Return per-channel latitude-weighted RMSE and anomaly ACC."""
    weights = torch.from_numpy(np.cos(np.radians(lats)).astype(np.float32)).reshape(1, 1, -1, 1)
    sq_err = (preds_raw - tgts_raw) ** 2
    denom = weights.sum() * preds_raw.size(0) * preds_raw.size(3)
    rmse = torch.sqrt((sq_err * weights).sum(dim=(0, 2, 3)) / denom)

    stds = torch.from_numpy(normalizer.channel_stds.reshape(1, -1, 1, 1))
    nrmse = rmse / stds.flatten()

    # ACC uses the training climatology when supplied. This avoids using test
    # targets to define the anomaly reference.
    if climatology is None:
        clim = ((tgts_raw * weights).sum(dim=(0, 2, 3)) / denom).reshape(1, -1, 1, 1)
    elif climatology.ndim == 1:
        clim = climatology.reshape(1, -1, 1, 1)
    elif climatology.ndim == 3:
        clim = climatology.unsqueeze(0)
    else:
        clim = climatology
    pred_anom = preds_raw - clim
    tgt_anom = tgts_raw - clim
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


@torch.no_grad()
def evaluate_model_on_test(model: nn.Module, test_loader: DataLoader,
                           normalizer: PINONormalizer, device: torch.device,
                           lats: np.ndarray = np.linspace(33.5, 30.0, 15),
                           climatology: torch.Tensor = None):
    model.eval()
    use_cuda = device.type == 'cuda'
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    predictions, targets = [], []
    start = time.perf_counter()
    batches = 0
    for inp, tgt in test_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            pred = model(inp)
        predictions.append(normalizer.denormalize(pred.float()).cpu())
        targets.append(normalizer.denormalize(tgt.float()).cpu())
        batches += 1
    metrics = calculate_metrics(torch.cat(predictions), torch.cat(targets), normalizer, lats, climatology)
    metrics['peak_gpu_mem_mb'] = (torch.cuda.max_memory_allocated(device) / 1e6) if use_cuda else 0.0
    metrics['inference_latency_ms'] = ((time.perf_counter() - start) / max(1, batches)) * 1000.0
    return metrics


@torch.no_grad()
def evaluate_persistence_on_test(test_loader: DataLoader, normalizer: PINONormalizer,
                                 lats: np.ndarray = np.linspace(33.5, 30.0, 15),
                                 climatology: torch.Tensor = None):
    predictions, targets = [], []
    for inp, tgt in test_loader:
        predictions.append(normalizer.denormalize(inp.float()).cpu())
        targets.append(normalizer.denormalize(tgt.float()).cpu())
    metrics = calculate_metrics(torch.cat(predictions), torch.cat(targets), normalizer, lats, climatology)
    metrics['peak_gpu_mem_mb'] = 0.0
    metrics['inference_latency_ms'] = 0.0
    return metrics


@torch.no_grad()
def evaluate_rollout_horizons(model: nn.Module, rollout_loader: DataLoader,
                              normalizer: PINONormalizer, device: torch.device,
                              max_horizon: int = 12,
                              lats: np.ndarray = np.linspace(33.5, 30.0, 15),
                              climatology: torch.Tensor = None):
    model.eval()
    model_predictions = [[] for _ in range(max_horizon)]
    persistence_predictions = [[] for _ in range(max_horizon)]
    targets = [[] for _ in range(max_horizon)]
    for sequence in rollout_loader:
        sequence = sequence.to(device)
        current = sequence[:, 0]
        initial = current
        for horizon in range(max_horizon):
            target = sequence[:, horizon + 1]
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                current = model(current)
            model_predictions[horizon].append(normalizer.denormalize(current.float()).cpu())
            persistence_predictions[horizon].append(normalizer.denormalize(initial.float()).cpu())
            targets[horizon].append(normalizer.denormalize(target.float()).cpu())
    output = {}
    for horizon in range(max_horizon):
        target = torch.cat(targets[horizon])
        output[f'{(horizon + 1) * 6}h'] = {
            'FNO': calculate_metrics(torch.cat(model_predictions[horizon]), target, normalizer, lats, climatology),
            'Persistence': calculate_metrics(torch.cat(persistence_predictions[horizon]), target, normalizer, lats, climatology),
        }
    return output


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
    test_rollout_loader = get_rollout_loader(test_zarr, normalizer, sequence_length=13, batch_size=8, shuffle=False)
    training_climatology = normalizer.denormalize(train_loader.dataset.data.float()).mean(dim=0)

    results_summary = {}

    print("\n--- Persistence Baseline (1-step) ---", flush=True)
    persistence_metrics = evaluate_persistence_on_test(test_loader, normalizer, climatology=training_climatology)
    results_summary['Persistence'] = persistence_metrics

    epochs = 30
    patience = 6          # early stopping patience
    max_grad_norm = 1.0   # gradient clipping

    # -----------------------------------------------------------------
    # BASELINE 1: 2D FNO WITH GEOGRAPHIC COORDINATES (DATA LOSS ONLY)
    # -----------------------------------------------------------------
    print("\n-------------------------------------------------------", flush=True)
    print("      STEP 2: TRAINING BASELINE 1 (2D COORDINATE FNO)   ", flush=True)
    print("-------------------------------------------------------", flush=True)

    b1_model = CoordinateFNO2d(in_dim=len(STATE_CHANNEL_NAMES), out_dim=len(STATE_CHANNEL_NAMES), width=32).to(device)
    b1_optimizer = torch.optim.AdamW(b1_model.parameters(), lr=3e-4, weight_decay=1e-4)
    b1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(b1_optimizer, T_max=epochs, eta_min=1e-5)

    b1_train_losses = []
    b1_val_losses = []
    b1_best_val = float('inf')
    b1_no_improve = 0
    b1_best_state = None
    fno_ckpt = os.path.join("DATASET", "checkpoints", "fno.pt")

    if os.path.exists(fno_ckpt):
        print(f"  [Found Checkpoint] Loading existing B1 weights from {fno_ckpt}", flush=True)
        try:
            b1_best_state = torch.load(fno_ckpt, map_location=device, weights_only=True)
        except TypeError:
            b1_best_state = torch.load(fno_ckpt, map_location=device)
    else:
        for epoch in range(1, epochs + 1):
            tr_loss, tr_data, _, _, _ = train_one_epoch(
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
        print(f"  [Checkpoint] {save_checkpoint(b1_best_state, 'fno')}", flush=True)

    # Rollout Fine-Tuning for Baseline 1
    print("\n--- Step 5: Rollout Fine-Tuning Baseline 1 ---", flush=True)
    b1_ft_optimizer = torch.optim.AdamW(b1_model.parameters(), lr=1e-4, weight_decay=1e-4)
    fine_tune_rollout(b1_model, train_rollout_loader, b1_ft_optimizer, device, epochs=3)

    # Evaluate Baseline 1 on Test Split
    print("\n--- Step 6: Evaluating Baseline 1 on Test Split ---", flush=True)
    b1_metrics = evaluate_model_on_test(b1_model, test_loader, normalizer, device, climatology=training_climatology)
    results_summary['FNO (2D + coordinates)'] = b1_metrics
    print(f"  [OK] B1 aggregate nRMSE: {b1_metrics['aggregate_nrmse']:.4f} | aggregate ACC: {b1_metrics['aggregate_acc']:.4f}")

    # -----------------------------------------------------------------
    # BASELINE 2: STANDARD PINO (WITH SOFT PHYSICS CONSTRAINTS)
    # -----------------------------------------------------------------
    print("\n-------------------------------------------------------", flush=True)
    print("      STEP 4: TRAINING BASELINE 2 (STANDARD PINO)     ", flush=True)
    print("-------------------------------------------------------", flush=True)

    b2_model = CoordinateFNO2d(in_dim=len(STATE_CHANNEL_NAMES), out_dim=len(STATE_CHANNEL_NAMES), width=32).to(device)
    b2_optimizer = torch.optim.AdamW(b2_model.parameters(), lr=3e-4, weight_decay=1e-4)
    b2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(b2_optimizer, T_max=epochs, eta_min=1e-5)

    b2_train_losses = []
    b2_val_losses = []
    b2_best_val = float('inf')
    b2_no_improve = 0
    b2_best_state = None
    pino_ckpt = os.path.join("DATASET", "checkpoints", "pino.pt")

    if os.path.exists(pino_ckpt):
        print(f"  [Found Checkpoint] Loading existing B2 weights from {pino_ckpt}", flush=True)
        try:
            b2_best_state = torch.load(pino_ckpt, map_location=device, weights_only=True)
        except TypeError:
            b2_best_state = torch.load(pino_ckpt, map_location=device)
    else:
        for epoch in range(1, epochs + 1):
            tr_loss, tr_data, tr_div, tr_hydro, tr_physics_grad = train_one_epoch(
                b2_model, train_loader, b2_optimizer, device,
                use_physics_loss=True, lambda_div=0.01, lambda_hydro=0.1,
                max_grad_norm=max_grad_norm, normalizer=normalizer
            )
            vl_loss, vl_data = validate(
                b2_model, val_loader, device, use_physics_loss=True,
                lambda_div=0.01, lambda_hydro=0.1, normalizer=normalizer
            )
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
            print(f"  B2 Epoch {epoch:2d}/{epochs} -> Total: {tr_loss:.5f} (Data: {tr_data:.5f} Div: {tr_div:.7f} Hydro: {tr_hydro:.7f} PhysicsGrad: {tr_physics_grad:.3e}) | Val: {vl_loss:.5f}{marker}", flush=True)
            assert not np.isnan(tr_loss) and not np.isnan(vl_loss), "Baseline 2 Loss is NaN!"

            if b2_no_improve >= patience:
                print(f"  [Early Stop] No val improvement for {patience} epochs. Best val: {b2_best_val:.5f}", flush=True)
                break

    # Restore best weights
    if b2_best_state is not None:
        b2_model.load_state_dict({k: v.to(device) for k, v in b2_best_state.items()})
        print(f"  [Restored] Best B2 weights (val={b2_best_val:.5f})", flush=True)
        print(f"  [Checkpoint] {save_checkpoint(b2_best_state, 'pino')}", flush=True)

    # Rollout Fine-Tuning for Baseline 2
    print("\n--- Step 5: Rollout Fine-Tuning Baseline 2 ---", flush=True)
    b2_ft_optimizer = torch.optim.AdamW(b2_model.parameters(), lr=1e-4, weight_decay=1e-4)
    fine_tune_rollout(b2_model, train_rollout_loader, b2_ft_optimizer, device, epochs=3)

    # Evaluate Baseline 2 on Test Split
    print("\n--- Step 6: Evaluating Baseline 2 on Test Split ---", flush=True)
    b2_metrics = evaluate_model_on_test(b2_model, test_loader, normalizer, device, climatology=training_climatology)
    results_summary['PINO (provisional physics)'] = b2_metrics
    print(f"  [OK] B2 aggregate nRMSE: {b2_metrics['aggregate_nrmse']:.4f} | aggregate ACC: {b2_metrics['aggregate_acc']:.4f}")

    print("\n--- 6h to 72h Rollout Benchmark ---", flush=True)
    rollout_results = {
        'FNO (2D + coordinates)': evaluate_rollout_horizons(
            b1_model, test_rollout_loader, normalizer, device, climatology=training_climatology
        ),
        'PINO (provisional physics)': evaluate_rollout_horizons(
            b2_model, test_rollout_loader, normalizer, device, climatology=training_climatology
        ),
    }
    results_summary['rollout_benchmark'] = rollout_results
    print(f"{'Model':<28s} | {'Horizon':>7s} | {'Agg nRMSE':>10s} | {'Agg ACC':>9s}")
    print("-" * 64)
    for model_name, horizons in rollout_results.items():
        for horizon, metrics_entry in horizons.items():
            metrics = metrics_entry.get('FNO', metrics_entry)
            print(f"{model_name:<28s} | {horizon:>7s} | {metrics['aggregate_nrmse']:>10.4f} | {metrics['aggregate_acc']:>9.4f}")

    # Save Results Table JSON
    results_file = os.path.join("DATASET", "baseline_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n=======================================================", flush=True)
    print("            FINAL BASELINES COMPARISON TABLE           ", flush=True)
    print("=======================================================", flush=True)
    print(f"{'Model':<32s} | {'Agg nRMSE':>10s} | {'Agg ACC':>9s}")
    print("-" * 58)
    for model_name, metrics in results_summary.items():
        if model_name == 'rollout_benchmark':
            continue
        print(f"{model_name:<32s} | {metrics['aggregate_nrmse']:>10.4f} | {metrics['aggregate_acc']:>9.4f}")
    print("=======================================================", flush=True)


if __name__ == "__main__":
    run_baseline_experiments()
