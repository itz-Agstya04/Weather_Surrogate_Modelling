"""
Prepare train, validation, and test ERA5 data for Himachal Pradesh microclimates.

Reads ERA5 directly from the WeatherBench2 cloud Zarr store, crops it to the
Himachal Pradesh bounding box, selects required variables and pressure levels,
creates chronological splits, writes local Zarr files, and validates them.
"""

import asyncio
import os
import time

import dask
import numpy as np
import xarray as xr


ERA5_ZARR_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
DOWNLOAD_WORKERS = 4
MAX_DOWNLOAD_ATTEMPTS = 4


def get_input_dataset() -> xr.Dataset:
    """Open the public WeatherBench2 ERA5 Zarr store via GCS."""
    try:
        import gcsfs  # noqa: F401  # Registers the ``gs://`` fsspec backend.
    except ImportError as exc:
        raise RuntimeError(
            "Cloud ERA5 access requires gcsfs. Install it with:\n"
            ".\\.venv\\Scripts\\pip.exe install gcsfs"
        ) from exc

    print(f"Opening public ERA5 Zarr store from:\n{ERA5_ZARR_URL}", flush=True)
    return xr.open_zarr(
        ERA5_ZARR_URL,
        chunks={"time": 48},
        consolidated=True,
        storage_options={"token": "anon"},
    )


def load_split_with_retries(
    ds: xr.Dataset,
    start_time: str,
    end_time: str,
) -> xr.Dataset:
    """Materialize one split, retrying transient cloud read failures."""
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            with dask.config.set(scheduler="threads", num_workers=DOWNLOAD_WORKERS):
                return ds.sel(time=slice(start_time, end_time)).load()
        except (asyncio.TimeoutError, OSError) as exc:
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise

            wait_seconds = 15 * attempt
            print(
                f"[WARN] Cloud read failed ({type(exc).__name__}). "
                f"Retrying in {wait_seconds}s "
                f"({attempt}/{MAX_DOWNLOAD_ATTEMPTS - 1})...",
                flush=True,
            )
            time.sleep(wait_seconds)


def process_hp_era5(output_dir: str = "DATASET") -> None:
    """Create and validate Himachal Pradesh ERA5 train/val/test Zarr datasets."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 0: Open cloud store lazily
    ds = get_input_dataset()

    # Step 1: Spatial subset
    # WeatherBench2 latitude is descending, so use high-to-low latitude slice.
    lat_slice = slice(33.5, 30.0)
    lon_slice = slice(75.5, 79.5)

    print("\n--- Step 1: Spatial Subsetting ---", flush=True)
    ds_hp = ds.sel(latitude=lat_slice, longitude=lon_slice)

    print(f"Cropped latitude grid size: {ds_hp.latitude.size}", flush=True)
    print(f"Cropped longitude grid size: {ds_hp.longitude.size}", flush=True)

    assert ds_hp.latitude.size > 0, (
        "Latitude slice is empty. Use descending order: slice(33.5, 30.0)."
    )
    assert ds_hp.longitude.size > 0, "Longitude slice is empty."
    print("[OK] Spatial slice is non-empty.", flush=True)

    # Step 2: Variable and pressure-level subset
    print("\n--- Step 2: Variable and Pressure Level Selection ---", flush=True)

    target_levels = [1000, 850, 700, 500, 300]
    surface_vars = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    upper_air_vars = ["geopotential"]
    target_vars = surface_vars + upper_air_vars

    available_levels = list(ds_hp.level.values)
    print(f"Available pressure levels (hPa): {available_levels}", flush=True)

    for level in target_levels:
        assert level in available_levels, (
            f"Target pressure level {level} hPa is not available."
        )

    for variable in target_vars:
        assert variable in ds_hp.data_vars, (
            f"Variable '{variable}' was not found in the ERA5 dataset."
        )

    ds_sub = ds_hp[target_vars].sel(level=target_levels)
    print(f"Selected variables: {list(ds_sub.data_vars)}", flush=True)

    # Step 3: Chronological splits
    # Each split ends three days before the nominal year boundary to avoid
    # autoregressive rollouts crossing into the next split.
    splits = {
        "train_2018_2019.zarr": (
            "2018-01-01T00:00:00",
            "2019-12-28T18:00:00",
        ),
        "val_2020.zarr": (
            "2020-01-01T00:00:00",
            "2020-12-28T18:00:00",
        ),
        "test_2021_2022.zarr": (
            "2021-01-01T00:00:00",
            "2022-12-28T18:00:00",
        ),
    }

    created_paths = {}

    # Step 4: Write local output Zarr stores
    print("\n--- Step 3: Writing Local Zarr Stores ---", flush=True)

    for split_name, (start_time, end_time) in splits.items():
        out_path = os.path.join(output_dir, split_name)

        print(
            f"\nProcessing '{split_name}' from {start_time} to {end_time}...",
            flush=True,
        )

        ds_split = load_split_with_retries(ds_sub, start_time, end_time)

        num_times = ds_split.time.size
        assert num_times > 0, (
            f"No time steps found between {start_time} and {end_time}."
        )

        print(f"Time steps: {num_times}", flush=True)

        ds_split = ds_split.chunk(
            {
                "time": min(1460, num_times),
                "latitude": -1,
                "longitude": -1,
                "level": -1,
            }
        )

        # Avoid inherited encoding conflicts while writing Zarr v2.
        for variable in ds_split.variables:
            ds_split[variable].encoding.clear()

        print(f"Writing Zarr store to: {out_path}", flush=True)
        ds_split.to_zarr(out_path, mode="w", zarr_format=2)

        print(f"[OK] Saved: {out_path}", flush=True)
        created_paths[split_name] = out_path

    # Step 5: Validate the written datasets
    print("\n--- Step 4: Validating Created Zarr Stores ---", flush=True)

    for split_name, out_path in created_paths.items():
        print(f"\nValidating: {split_name}", flush=True)
        ds_val = xr.open_zarr(out_path)

        lat_vals = ds_val.latitude.values
        lon_vals = ds_val.longitude.values

        assert np.isclose(lat_vals[0], 33.5)
        assert np.isclose(lat_vals[-1], 30.0)
        assert np.isclose(lon_vals[0], 75.5)
        assert np.isclose(lon_vals[-1], 79.5)
        print("[OK] Bounding box verified.", flush=True)

        levels_val = list(ds_val.level.values)
        assert levels_val == target_levels, (
            f"Pressure levels mismatch: {levels_val} vs {target_levels}"
        )
        print("[OK] Pressure levels verified.", flush=True)

        for variable in target_vars:
            values = ds_val[variable].values
            assert not np.isnan(values).all(), (
                f"Variable '{variable}' contains only NaN values."
            )
            print(f"[OK] '{variable}' contains valid data.", flush=True)

        times = ds_val.time.values
        time_diffs = np.diff(times).astype("timedelta64[h]").astype(int)

        assert np.all(time_diffs == 6), (
            f"Expected 6-hour intervals; found: {np.unique(time_diffs)}"
        )
        print("[OK] Time coordinate is contiguous at 6-hour intervals.", flush=True)

    print("\n=======================================================", flush=True)
    print("ALL ZARR STORES VALIDATED AND READY FOR THE PINO LOADER", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    process_hp_era5()
