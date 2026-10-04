"""Compatibility entry point for the canonical all-splits temperature updater."""

from data.update_splits_temperature import add_temperature_to_zarr


if __name__ == "__main__":
    for split in ("train_2018_2019.zarr", "val_2020.zarr", "test_2021_2022.zarr"):
        add_temperature_to_zarr(split)
