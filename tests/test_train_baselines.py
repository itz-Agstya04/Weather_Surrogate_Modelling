import numpy as np
import torch

from train.train_baselines import CoordinateFNO2d, SpectralConv2d, compute_divergence_loss, compute_hydrostatic_loss


def test_spectral_conv_preserves_shape_even_and_odd_modes():
    for modes1, modes2 in [(8, 6), (31, 29)]:
        layer = SpectralConv2d(3, 5, modes1, modes2)
        assert layer(torch.randn(2, 3, 15, 17)).shape == (2, 5, 15, 17)


def test_coordinates_are_unit_norm_pairs():
    coords = CoordinateFNO2d.coordinates(15, 17, torch.device("cpu"), torch.float32)
    assert torch.allclose(coords[0].square() + coords[1].square(), torch.ones(15, 17), atol=1e-6)
    assert torch.allclose(coords[2].square() + coords[3].square(), torch.ones(15, 17), atol=1e-6)


def test_divergence_loss_constant_is_zero_and_divergent_field_positive():
    constant = torch.zeros(1, 13, 15, 17)
    assert compute_divergence_loss(constant).item() < 1e-10
    divergent = constant.clone()
    divergent[:, 1] = torch.linspace(0.0, 1.0, 17).view(1, 1, 17)
    assert compute_divergence_loss(divergent).item() > 0


def test_hydrostatic_loss_exact_profile_is_zero():
    pressures = torch.tensor([1000., 850., 700., 500., 300.]) * 100.0
    temperature = torch.full((5, 1, 1), 280.0)
    phi = torch.zeros(5, 1, 1)
    for index in range(4):
        dp = pressures[index + 1] - pressures[index]
        phi[index + 1] = phi[index] - 287.05 * 280.0 / ((pressures[index + 1] + pressures[index]) / 2.0) * dp
    state = torch.zeros(1, 13, 15, 17)
    state[:, 3:8] = phi.view(1, 5, 1, 1)
    state[:, 8:13] = temperature.view(1, 5, 1, 1)
    assert compute_hydrostatic_loss(state).item() < 1e-10
    state[:, 7] += 100.0
    assert compute_hydrostatic_loss(state).item() > 0
