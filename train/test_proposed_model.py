"""Focused unit tests for the proposed model components."""

import os

import pytest
import torch

from train.proposed_model import FactorizedFNO2d, LerayProjection2d, TerrainGNNResidual
from train.train_baselines import CoordinateFNO2d


def test_factorized_forward_and_parameter_count():
    baseline = CoordinateFNO2d(in_dim=13, out_dim=13, width=16)
    model = FactorizedFNO2d(in_dim=13, out_dim=13, width=16)
    output = model(torch.randn(2, 13, 15, 17))
    assert output.shape == (2, 13, 15, 17)
    assert sum(p.numel() for p in model.parameters()) > 0
    print("CoordinateFNO2d parameters:", sum(p.numel() for p in baseline.parameters()))
    print("FactorizedFNO2d parameters:", sum(p.numel() for p in model.parameters()))


def test_terrain_gnn_shape_and_sparse_scatter():
    path = "DATASET/terrain_hp.npz"
    if not os.path.exists(path):
        pytest.skip("DATASET/terrain_hp.npz is not available")
    model = TerrainGNNResidual(path)
    state = torch.randn(2, 13, 15, 17)
    residual = model(state)
    assert residual.shape == state.shape
    mask = torch.zeros(15 * 17, dtype=torch.bool)
    mask[model.node_flat_indices.cpu()] = True
    assert torch.all(residual.flatten(2)[:, :, ~mask] == 0)


def test_leray_projection_is_divergence_free():
    model = LerayProjection2d()
    state = torch.randn(2, 13, 15, 17)
    projected = model(state)
    u_hat = torch.fft.rfft2(projected[:, 1].float())
    v_hat = torch.fft.rfft2(projected[:, 2].float())
    ky = torch.fft.fftfreq(15).reshape(-1, 1)
    kx = torch.fft.rfftfreq(17).reshape(1, -1)
    divergence = kx * u_hat + ky * v_hat
    assert divergence.abs().max().item() < 1e-4
    assert torch.equal(projected[:, 0], state[:, 0])
    assert torch.equal(projected[:, 3:], state[:, 3:])
