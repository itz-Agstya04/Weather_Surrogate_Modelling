"""
Download 3D temperature at target pressure levels for val_2020.zarr and test_2021_2022.zarr
from WeatherBench2 and append to local Zarr stores so all splits have all 13 channels.
"""

import argparse
import os
import dask
import xarray as xr
import numpy as np

ERA5_ZARR_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
TARGET_LEVELS = [1000, 850, 700, 500, 300]
LAT_SLICE = slice(33.5, 30.0)
LON_SLICE = slice(75.5, 79.5)


def add_temperature_to_splits(split_names: list[str], output_dir: str = "DATASET"):
    splits_to_process = []
    for split_name in split_names:
        zarr_path = os.path.join(output_dir, split_name)
        if not os.path.exists(zarr_path):
            print(f"Split path {zarr_path} does not exist. Skipping.", flush=True)
            continue
        local_ds = xr.open_zarr(zarr_path)
        if "temperature" in local_ds.data_vars:
            print(f"  'temperature' already exists in {split_name}. Skipping.", flush=True)
        else:
            splits_to_process.append((split_name, zarr_path, local_ds))

    if not splits_to_process:
        print("\nAll requested splits already contain 'temperature'. Done.", flush=True)
        return

    print(f"\nOpening WeatherBench2 ERA5 store lazily...", flush=True)
    cloud_ds = xr.open_zarr(
        ERA5_ZARR_URL,
        chunks={"time": 48},
        consolidated=True,
        storage_options={"token": "anon"},
    )

    for split_name, zarr_path, local_ds in splits_to_process:
        start_time = str(local_ds.time.values[0])
        end_time = str(local_ds.time.values[-1])
        print(f"\nProcessing '{split_name}'...", flush=True)
        print(f"  Time range: {start_time} to {end_time} ({local_ds.time.size} steps)", flush=True)
        print("  Fetching temperature from WeatherBench2...", flush=True)

        with dask.config.set(scheduler="threads", num_workers=4):
            temp_sub = cloud_ds["temperature"].sel(
                time=slice(start_time, end_time),
                latitude=LAT_SLICE,
                longitude=LON_SLICE,
                level=TARGET_LEVELS,
            ).load()

        print(f"  Fetched temperature with shape {temp_sub.shape}. Preparing zarr append...", flush=True)
        temp_sub.encoding.clear()
        temp_ds = temp_sub.to_dataset(name="temperature")
        for var in temp_ds.variables:
            temp_ds[var].encoding.clear()

        num_times = temp_ds.time.size
        temp_ds = temp_ds.chunk(
            {
                "time": min(1460, num_times),
                "latitude": -1,
                "longitude": -1,
                "level": -1,
            }
        )

        print(f"  Appending 'temperature' variable to {zarr_path}...", flush=True)
        temp_ds.to_zarr(zarr_path, mode="a")
        print(f"  [OK] Successfully added 'temperature' to {zarr_path}!", flush=True)

        check_ds = xr.open_zarr(zarr_path)
        assert "temperature" in check_ds.data_vars, f"Failed to verify temperature in {zarr_path}"
        print(f"  Verified data variables: {list(check_ds.data_vars)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add upper-air temperature to local split stores.")
    parser.add_argument("splits", nargs="*", help="Split filenames; defaults to all project splits.")
    parser.add_argument("--output-dir", default="DATASET")
    args = parser.parse_args()
    split_names = args.splits or [
        "train_2018_2019.zarr", "val_2020.zarr", "test_2021_2022.zarr"
    ]
    add_temperature_to_splits(split_names, args.output_dir)
    print("\nAll requested splits updated with temperature!", flush=True)
