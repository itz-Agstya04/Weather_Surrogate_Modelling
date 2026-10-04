"""
Prepare train, validation, and test ERA5 data for Himachal Pradesh microclimates.

Reads ERA5 directly from the WeatherBench2 cloud Zarr store, crops it to the
Himachal Pradesh bounding box, selects required variables and pressure levels,
creates chronological splits, writes local Zarr files incrementally (one
month at a time, with a progress bar and resume support), and validates them.
"""

import asyncio
import os
import shutil
import time

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


ERA5_ZARR_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
DOWNLOAD_WORKERS = 12  # anon GCS reads parallelize well; raise if you have bandwidth
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


def print_chunk_diagnostics(ds_sub: xr.Dataset, target_vars: list) -> None:
    """
    Print the underlying dask chunk shape for each variable. If the lat/lon
    chunk sizes here match the FULL source grid (721 lat x 1440 lon) rather
    than something close to your cropped region, every read pulls the whole
    global tile over the network before discarding almost all of it locally
    -- this is the most common cause of multi-hour stalls on a tiny crop.
    """
    print("\n--- Chunk diagnostics (why downloads may be slow) ---", flush=True)
    for var in target_vars:
        chunks = ds_sub[var].data.chunksize
        dims = ds_sub[var].dims
        print(f"{var}: dims={dims} chunk_shape={chunks}", flush=True)
    print(
        "If lat/lon chunk sizes above are near the full source grid "
        "(721, 1440), each read fetches full global tiles over the network.",
        flush=True,
    )


def month_starts(start_time: str, end_time: str):
    """Yield (chunk_start, chunk_end) timestamp pairs, one calendar month at a time."""
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    cur = start
    while cur <= end:
        month_end = (cur + pd.offsets.MonthEnd(0)).replace(
            hour=23, minute=59, second=59
        )
        chunk_end = min(month_end, end)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()


def load_chunk_with_retries(
    ds: xr.Dataset,
    start_time: str,
    end_time: str,
) -> xr.Dataset:
    """Materialize one time chunk, retrying transient cloud read failures."""
    last_exc = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            with dask.config.set(scheduler="threads", num_workers=DOWNLOAD_WORKERS):
                with ProgressBar():
                    return ds.sel(time=slice(start_time, end_time)).load()
        except (asyncio.TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise
            wait_seconds = 15 * attempt
            print(
                f"[WARN] Cloud read failed ({type(exc).__name__}: {exc}). "
                f"Retrying in {wait_seconds}s "
                f"({attempt}/{MAX_DOWNLOAD_ATTEMPTS - 1})...",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise last_exc  # pragma: no cover - defensive


def write_split_incrementally(
    ds_sub: xr.Dataset,
    out_path: str,
    start_time: str,
    end_time: str,
    resume: bool = True,
) -> None:
    """
    Download and write one split one calendar month at a time, appending to
    a local Zarr store as each month completes. This bounds memory, shows
    real progress, and lets you resume after a crash instead of restarting
    a multi-hour call from scratch.
    """
    store_exists = os.path.exists(out_path)

    if store_exists and resume:
        existing = xr.open_zarr(out_path)
        last_time = pd.Timestamp(existing.time.values[-1])
        existing.close()
        print(
            f"[RESUME] Found existing store with data through {last_time}. "
            f"Skipping months already written.",
            flush=True,
        )
    else:
        last_time = None
        if store_exists:
            print(f"[INFO] Overwriting existing store at {out_path}", flush=True)
            shutil.rmtree(out_path)

    chunks = list(month_starts(start_time, end_time))
    first_write = not (store_exists and resume and last_time is not None)

    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        if last_time is not None and pd.Timestamp(chunk_end) <= last_time:
            print(f"  [{i}/{len(chunks)}] Skipping {chunk_start[:7]} (already written)", flush=True)
            continue

        # Avoid re-fetching timestamps already covered by a partially written month.
        effective_start = chunk_start
        if last_time is not None and pd.Timestamp(chunk_start) <= last_time:
            effective_start = (last_time + pd.Timedelta(hours=6)).isoformat()

        print(
            f"  [{i}/{len(chunks)}] Downloading {effective_start} -> {chunk_end} ...",
            flush=True,
        )
        ds_chunk = load_chunk_with_retries(ds_sub, effective_start, chunk_end)

        if ds_chunk.time.size == 0:
            print(f"  [{i}/{len(chunks)}] No timesteps in this range, skipping.", flush=True)
            continue

        ds_chunk = ds_chunk.chunk(
            {
                "time": ds_chunk.time.size,
                "latitude": -1,
                "longitude": -1,
                "level": -1,
            }
        )
        for variable in ds_chunk.variables:
            ds_chunk[variable].encoding.clear()

        if first_write:
            ds_chunk.to_zarr(out_path, mode="w", zarr_format=2)
            first_write = False
        else:
            ds_chunk.to_zarr(out_path, mode="a", append_dim="time", zarr_format=2)

        print(
            f"  [{i}/{len(chunks)}] Wrote {ds_chunk.time.size} timesteps to {out_path}",
            flush=True,
        )


def process_hp_era5(output_dir: str = "DATASET", resume: bool = True) -> None:
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
    upper_air_vars = ["geopotential", "temperature"]
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

    print_chunk_diagnostics(ds_sub, target_vars)

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

    # Step 4: Write local output Zarr stores, one month at a time
    print("\n--- Step 3: Writing Local Zarr Stores (chunked by month) ---", flush=True)

    for split_name, (start_time, end_time) in splits.items():
        out_path = os.path.join(output_dir, split_name)
        print(f"\nProcessing '{split_name}' from {start_time} to {end_time}...", flush=True)
        write_split_incrementally(ds_sub, out_path, start_time, end_time, resume=resume)
        print(f"[OK] Finished: {out_path}", flush=True)
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