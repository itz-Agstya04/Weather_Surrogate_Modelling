import os

import numpy as np
import pytest

from data.terrain_pipeline import build_grid_graph


def test_build_grid_graph_3x3_closed_form():
    coords = [(row, col) for row in range(3) for col in range(3)]
    edge_index, edge_attr = build_grid_graph(coords, H=3, W=3)
    assert edge_index.shape == (2, 40)
    assert edge_attr.shape == (40, 2)
    assert np.all(np.isin(edge_attr[:, 0] * 3, [-1, 0, 1]))
    assert np.all(np.isin(edge_attr[:, 1] * 3, [-1, 0, 1]))


def test_real_terrain_sparse_dense_relationship():
    path = "DATASET/terrain_hp.npz"
    if not os.path.exists(path):
        pytest.skip("terrain artifact is not present")
    with np.load(path) as artifact:
        sparse_nodes = artifact["node_flat_indices"].size
        dense_nodes = artifact["terrain_features"].shape[1] * artifact["terrain_features"].shape[2]
        assert sparse_nodes <= dense_nodes
        assert artifact["edge_index"].shape[1] <= artifact["dense_edge_index"].shape[1]
