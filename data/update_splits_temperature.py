"""
Download 3D temperature at target pressure levels for val_2020.zarr and test_2021_2022.zarr
from WeatherBench2 and append to local Zarr stores so all splits have all 13 channels.
"""

import argparse
import os
import xarray as xr
import numpy as np

ERA5_ZARR_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
TARGET_LEVELS = [1000, 850, 700, 500, 300]
LAT_SLICE = slice(33.5, 30.0)
LON_SLICE = slice(75.5, 79.5)


def add_temperature_to_zarr(split_name: str, output_dir: str = "DATASET"):
    zarr_path = os.path.join(output_dir, split_name)
    print(f"\nProcessing '{zarr_path}'...", flush=True)
    local_ds = xr.open_zarr(zarr_path)
    
    if "temperature" in local_ds.data_vars:
        print(f"  'temperature' already exists in {split_name}. Skipping.", flush=True)
        return
        
    start_time = str(local_ds.time.values[0])
    end_time = str(local_ds.time.values[-1])
    print(f"  Time range: {start_time} to {end_time} ({local_ds.time.size} steps)", flush=True)
    
    # Open WB2 cloud store
    cloud_ds = xr.open_zarr(
        ERA5_ZARR_URL,
        chunks={"time": 48},
        consolidated=True,
        storage_options={"token": "anon"},
    )
    
    print("  Fetching temperature from WeatherBench2...", flush=True)
    temp_sub = cloud_ds["temperature"].sel(
        time=local_ds.time,
        latitude=local_ds.latitude,
        longitude=local_ds.longitude,
        level=local_ds.level,
    ).load()
    
    # Clear encoding to prevent zarr encoding conflicts
    temp_sub.encoding.clear()
    
    # Convert to dataset
    temp_ds = temp_sub.to_dataset(name="temperature")
    for var in temp_ds.variables:
        temp_ds[var].encoding.clear()
        
    print(f"  Appending 'temperature' variable to {zarr_path}...", flush=True)
    temp_ds.to_zarr(zarr_path, mode="a")
    print(f"  [OK] Successfully added 'temperature' to {zarr_path}!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add upper-air temperature to local split stores.")
    parser.add_argument("splits", nargs="*", help="Split filenames; defaults to all project splits.")
    parser.add_argument("--output-dir", default="DATASET")
    args = parser.parse_args()
    split_names = args.splits or [
        "train_2018_2019.zarr", "val_2020.zarr", "test_2021_2022.zarr"
    ]
    for split in split_names:
        add_temperature_to_zarr(split, args.output_dir)
    print("\nAll requested splits updated with temperature!", flush=True)
