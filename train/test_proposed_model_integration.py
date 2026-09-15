"""Integration coverage for every proposed-model ablation combination."""

import os

import pytest
import torch

from train.proposed_model import ProposedModel


@pytest.mark.parametrize("physics_mode", ["none", "soft", "hard"])
@pytest.mark.parametrize("use_gnn_residual,gnn_dense", [(False, False), (True, False), (True, True)])
def test_all_proposed_ablation_combinations(physics_mode, use_gnn_residual, gnn_dense):
    if not os.path.exists("DATASET/terrain_hp.npz"):
        pytest.skip("DATASET/terrain_hp.npz is not available")
    model = ProposedModel(
        physics_mode=physics_mode,
        use_gnn_residual=use_gnn_residual,
        gnn_dense=gnn_dense,
    )
    output = model(torch.randn(2, 13, 15, 17))
    assert output.shape == (2, 13, 15, 17)
    assert torch.isfinite(output).all(), f"NaN/Inf output for {physics_mode}, {use_gnn_residual}, {gnn_dense}"
