"""Factorized spectral + terrain residual surrogate model.

The model consumes the 13-channel state produced by ``pino_data_pipeline``.
Terrain data is intentionally loaded only from the documented NPZ artifact;
this module does not fabricate terrain features when that artifact is absent.
"""

import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from train.train_baselines import (
    CoordinateFNO2d,
    SpectralConv2d,
    compute_divergence_loss,
    compute_hydrostatic_loss,
)


STATE_CHANNELS = 13
TERRAIN_CHANNELS = 3


def load_terrain_artifact(terrain_path: str) -> dict:
    """Load the terrain artifact once and require both sparse and dense graphs."""
    if not os.path.exists(terrain_path):
        raise FileNotFoundError(
            f"Terrain artifact not found: {terrain_path}. Run data/terrain_pipeline.py first."
        )
    with np.load(terrain_path) as terrain:
        required = {
            "terrain_features", "node_flat_indices", "edge_index", "edge_attr",
            "dense_edge_index", "dense_edge_attr",
        }
        missing = sorted(required.difference(terrain.files))
        if missing:
            raise KeyError(
                f"Terrain artifact is missing required graph keys: {missing}. "
                "Regenerate it with the updated terrain pipeline."
            )
        return {
            "terrain_features": torch.from_numpy(terrain["terrain_features"].astype(np.float32)),
            "node_flat_indices": torch.from_numpy(terrain["node_flat_indices"].astype(np.int64)),
            "edge_index": torch.from_numpy(terrain["edge_index"].astype(np.int64)),
            "edge_attr": torch.from_numpy(terrain["edge_attr"].astype(np.float32)),
            "dense_edge_index": torch.from_numpy(terrain["dense_edge_index"].astype(np.int64)),
            "dense_edge_attr": torch.from_numpy(terrain["dense_edge_attr"].astype(np.float32)),
        }


class VerticalSpectralConv1d(nn.Module):
    """Fourier convolution along the five pressure levels."""

    def __init__(self, in_channels: int, out_channels: int, modes: int = 2):
        super().__init__()
        self.modes = min(modes, 2)
        scale = 1.0 / (in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, levels)
        length = x.size(-1)
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        output = torch.zeros(
            x.size(0), self.weights_pos.size(1), length // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        modes = min(self.modes, spectrum.size(-1))
        output[:, :, :modes] = torch.einsum(
            "bcl,col->bol", spectrum[:, :, :modes], self.weights_pos[:, :, :modes]
        )
        return torch.fft.irfft(output, n=length, dim=-1).to(dtype=x.dtype)


class FactorizedFNOBlock(nn.Module):
    def __init__(self, width: int, modes_lat: int = 8, modes_lon: int = 6, modes_vertical: int = 2):
        super().__init__()
        self.horizontal = SpectralConv2d(width, width, modes_lat, modes_lon)
        self.horizontal_skip = nn.Conv2d(width, width, 1)
        self.vertical = VerticalSpectralConv1d(width, width, modes_vertical)
        self.vertical_skip = nn.Conv1d(width, width, 1)

    def forward(self, horizontal: torch.Tensor, vertical: torch.Tensor):
        b, c, levels, height, width = vertical.shape
        vertical_flat = vertical.permute(0, 3, 4, 1, 2).reshape(-1, c, levels)
        vertical_out = self.vertical(vertical_flat) + self.vertical_skip(vertical_flat)
        vertical_out = vertical_out.reshape(b, height, width, c, levels).permute(0, 3, 4, 1, 2)
        vertical_2d = vertical_out.mean(dim=2)
        horizontal_out = self.horizontal(horizontal) + self.horizontal_skip(horizontal)
        return torch.nn.functional.gelu(horizontal_out + vertical_2d), torch.nn.functional.gelu(vertical_out)


class FactorizedFNO2d(nn.Module):
    def __init__(self, in_dim: int = STATE_CHANNELS, out_dim: int = STATE_CHANNELS, width: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.p = nn.Conv2d(in_dim + 4 + TERRAIN_CHANNELS, width, 1)
        self.vertical_lift = nn.Conv1d(2, width, 1)
        self.blocks = nn.ModuleList([FactorizedFNOBlock(width) for _ in range(4)])
        self.q1 = nn.Conv2d(width, width // 2, 1)
        self.q2 = nn.Conv2d(width // 2, out_dim, 1)

    def forward(self, state: torch.Tensor, terrain_features: torch.Tensor = None) -> torch.Tensor:
        b, _, height, width = state.shape
        coords = CoordinateFNO2d.coordinates(height, width, state.device, state.dtype)
        if terrain_features is None:
            terrain_features = torch.zeros(3, height, width, device=state.device, dtype=state.dtype)
        terrain = terrain_features.to(device=state.device, dtype=state.dtype).unsqueeze(0).expand(b, -1, -1, -1)
        horizontal = self.p(torch.cat((state, coords.unsqueeze(0).expand(b, -1, -1, -1), terrain), dim=1))
        vertical_raw = state[:, 3:13].reshape(b, 2, 5, height, width)
        vertical_flat = vertical_raw.permute(0, 3, 4, 1, 2).reshape(-1, 2, 5)
        vertical = self.vertical_lift(vertical_flat).reshape(b, height, width, self.width, 5).permute(0, 3, 4, 1, 2)
        for block in self.blocks:
            horizontal, vertical = block(horizontal, vertical)
        return self.q2(torch.nn.functional.gelu(self.q1(horizontal)))


class TerrainGNNResidual(nn.Module):
    def __init__(self, terrain_path: str = "DATASET/terrain_hp.npz", channels: int = STATE_CHANNELS,
                 hidden: int = 64, layers: int = 3, gnn_dense: bool = False,
                 terrain_data: dict = None):
        super().__init__()
        if terrain_data is None:
            terrain_data = load_terrain_artifact(terrain_path)
        self.register_buffer("terrain_features", terrain_data["terrain_features"])
        self.register_buffer("node_flat_indices", terrain_data["node_flat_indices"])
        self.register_buffer("edge_index", terrain_data["edge_index"])
        self.register_buffer("edge_attr", terrain_data["edge_attr"])
        self.register_buffer("dense_edge_index", terrain_data["dense_edge_index"])
        self.register_buffer("dense_edge_attr", terrain_data["dense_edge_attr"])
        self.register_buffer("full_flat_indices", torch.arange(self.terrain_features.shape[1] * self.terrain_features.shape[2], dtype=torch.long))
        self.channels = channels
        self.gnn_dense = gnn_dense
        node_input = channels + TERRAIN_CHANNELS
        self.node_encoder = nn.Linear(node_input, hidden)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2 + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        ])
        self.node_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        ])
        self.output = nn.Linear(hidden, channels)
        # Learnable gate: starts near-zero so the untrained GNN doesn't corrupt
        # backbone predictions on steep cells.  The gate opens only if the GNN
        # earns it through gradient descent — a falsifiable test of terrain signal.
        self.gnn_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b, channels, height, width = state.shape
        flat = state.flatten(2).transpose(1, 2)
        terrain_flat = self.terrain_features.flatten(1).transpose(0, 1)
        indices = self.full_flat_indices if self.gnn_dense else self.node_flat_indices
        nodes = torch.cat((flat[:, indices], terrain_flat[indices].unsqueeze(0).expand(b, -1, -1)), dim=-1)
        hidden = self.node_encoder(nodes)
        edge_index = self.dense_edge_index if self.gnn_dense else self.edge_index
        edge_attr = self.dense_edge_attr if self.gnn_dense else self.edge_attr
        src, dst = edge_index
        for edge_mlp, update in zip(self.edge_mlps, self.node_updates):
            messages = edge_mlp(torch.cat((hidden[:, src], hidden[:, dst], edge_attr.unsqueeze(0).expand(b, -1, -1)), dim=-1))
            aggregate = torch.zeros_like(hidden).index_add_(1, dst, messages)
            hidden = hidden + update(torch.cat((hidden, aggregate), dim=-1))
        # Scale output by learnable gate (initialized to 0.01)
        residual = self.gnn_scale * self.output(hidden)
        full = torch.zeros(b, channels, height * width, device=state.device, dtype=state.dtype)
        full[:, :, indices] = residual.transpose(1, 2).to(state.dtype)
        return full.reshape(b, channels, height, width)


class LerayProjection2d(nn.Module):
    """Differentiable spectral projection of u/v onto divergence-free fields."""
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        u, v = state[:, 1], state[:, 2]
        height, width = u.shape[-2:]
        u_hat = torch.fft.rfft2(u.float())
        v_hat = torch.fft.rfft2(v.float())
        ky = torch.fft.fftfreq(height, device=state.device).reshape(-1, 1)
        kx = torch.fft.rfftfreq(width, device=state.device).reshape(1, -1)
        denom = kx.square() + ky.square()
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        parallel = (kx * u_hat + ky * v_hat) / denom
        u_projected = u_hat - kx * parallel
        v_projected = v_hat - ky * parallel
        output = state.clone()
        output[:, 1] = torch.fft.irfft2(u_projected, s=(height, width)).to(state.dtype)
        output[:, 2] = torch.fft.irfft2(v_projected, s=(height, width)).to(state.dtype)
        return output


class SoftPhysicsPenalty:
    def __call__(self, state: torch.Tensor, normalizer) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_divergence_loss(state, normalizer), compute_hydrostatic_loss(state, normalizer)


class IterativeCorrection(nn.Module):
    """Small learned corrector applied repeatedly to the same forecast step."""

    def __init__(self, channels: int = STATE_CHANNELS, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, channels, 1),
        )

    def forward(self, initial: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((initial, candidate, candidate - initial), dim=1))


class ProposedModel(nn.Module):
    def __init__(self, terrain_path: str = "DATASET/terrain_hp.npz", width: int = 32,
                 physics_mode: Literal["soft", "hard", "none"] = "none",
                 use_gnn_residual: bool = True, gnn_dense: bool = False,
                 refinement_steps: int = 1, refinement_alpha: float = 0.5):
        super().__init__()
        if physics_mode not in {"soft", "hard", "none"}:
            raise ValueError("physics_mode must be 'soft', 'hard', or 'none'")
        if refinement_steps < 1:
            raise ValueError("refinement_steps must be at least 1")
        if not 0.0 < refinement_alpha <= 1.0:
            raise ValueError("refinement_alpha must be in (0, 1]")
        self.physics_mode = physics_mode
        self.refinement_steps = refinement_steps
        self.refinement_alpha = refinement_alpha
        terrain_data = load_terrain_artifact(terrain_path)
        self.backbone = FactorizedFNO2d(width=width)
        self.register_buffer("terrain_features", terrain_data["terrain_features"])
        self.use_gnn_residual = use_gnn_residual
        self.gnn = TerrainGNNResidual(terrain_path, gnn_dense=gnn_dense, terrain_data=terrain_data) if use_gnn_residual else None
        self.corrector = (
            IterativeCorrection(width=width)
            if refinement_steps > 1 else nn.Identity()
        )
        self.projection = LerayProjection2d() if physics_mode == "hard" else nn.Identity()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.backbone(state, self.terrain_features)
        if self.gnn is not None:
            output = output + self.gnn(state)
        output = self.projection(output)
        if self.refinement_steps == 1:
            return output

        # Every pass targets the same next state.  This is a learned fixed-point
        # refinement, not another time step, so the temporal rollout remains in
        # train_baselines.fine_tune_rollout and is not duplicated here.
        for _ in range(self.refinement_steps - 1):
            correction = self.corrector(state, output)
            output = self.projection(output + self.refinement_alpha * correction)
        return output
