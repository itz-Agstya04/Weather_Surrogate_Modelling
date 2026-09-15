import json
import os

import numpy as np
import pytest
import torch
import xarray as xr

from data.pino_data_pipeline import (
    CHANNEL_NAMES,
    HPERA5RolloutDataset,
    HPERA5SingleStepDataset,
    PINONormalizer,
    SURFACE_VARS,
    TARGET_LEVELS,
    UPPER_AIR_TEMPERATURE_VAR,
    UPPER_AIR_VAR,
)


def stats():
    result = {name: {"mean": float(i), "std": 1.0 + i / 10} for i, name in enumerate(SURFACE_VARS)}
    result[UPPER_AIR_VAR] = {str(level): {"mean": 10.0 + i, "std": 2.0} for i, level in enumerate(TARGET_LEVELS)}
    result[UPPER_AIR_TEMPERATURE_VAR] = {str(level): {"mean": 20.0 + i, "std": 3.0} for i, level in enumerate(TARGET_LEVELS)}
    return result


def test_normalizer_round_trip_13_channels():
    normalizer = PINONormalizer(stats())
    raw = torch.randn(13, 15, 17)
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(raw)), raw, atol=1e-4)


@pytest.mark.parametrize("shape", [(13, 15, 17), (4, 13, 15, 17), (2, 5, 13, 15, 17)])
def test_normalizer_supported_shapes(shape):
    normalizer = PINONormalizer(stats())
    tensor = torch.randn(*shape)
    assert normalizer.normalize(tensor).shape == tensor.shape


def test_channel_contract():
    assert len(CHANNEL_NAMES) == 13
    assert CHANNEL_NAMES[3:8] == [f"geopotential_{level}" for level in TARGET_LEVELS]
    assert CHANNEL_NAMES[8:] == [f"temperature_{level}" for level in TARGET_LEVELS]


def test_single_step_dataset_real_store():
    path = "DATASET/train_2018_2019.zarr"
    if not os.path.exists(path):
        pytest.skip("training Zarr store is not present")
    ds = HPERA5SingleStepDataset(path, transform=False)
    assert len(ds) == ds.ds.sizes["time"] - 1
    assert ds[0][0].shape == (13, 15, 17)


def test_rollout_dataset_real_store_and_noncontiguous(monkeypatch):
    path = "DATASET/train_2018_2019.zarr"
    if not os.path.exists(path):
        pytest.skip("training Zarr store is not present")
    ds = HPERA5RolloutDataset(path, sequence_length=4, transform=False)
    assert ds[0].shape == (4, 13, 15, 17)

    times = np.array(["2020-01-01T00", "2020-01-01T06", "2020-01-01T18"], dtype="datetime64[h]")
    data = np.zeros((3, 15, 17), dtype=np.float32)
    mock = xr.Dataset({name: (("time", "latitude", "longitude"), data) for name in SURFACE_VARS})
    mock[UPPER_AIR_VAR] = (("time", "level", "latitude", "longitude"), np.zeros((3, 5, 15, 17), dtype=np.float32))
    mock[UPPER_AIR_TEMPERATURE_VAR] = mock[UPPER_AIR_VAR].copy()
    mock = mock.assign_coords(time=times, level=TARGET_LEVELS, latitude=np.arange(15), longitude=np.arange(17))
    monkeypatch.setattr("data.pino_data_pipeline.xr.open_zarr", lambda _: mock)
    with pytest.raises(AssertionError, match="non-6h time gaps"):
        HPERA5RolloutDataset("mock.zarr", transform=False)
