"""Fetch and verify cropped ERA5 data from the public WeatherBench2 Zarr store."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.store import open_project_zarr, stamp_store

SOURCE = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
DATASET = ROOT / "DATASET"
# Keep newly acquired stores separate from any existing or quarantined stores.
# Override with WEATHER_OUTPUT_DIR when running an acquisition.
OUTPUT_DATASET = Path(os.environ.get("WEATHER_OUTPUT_DIR", str(DATASET)))
TRAIN = DATASET / "train_2018_2019.zarr"
MARKERS = OUTPUT_DATASET / "acquisition_markers"
TARGET_LEVELS = [1000, 850, 700, 500, 300]
VARIABLES = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential", "temperature", "surface_pressure", "geopotential_at_surface"]
CORE_VARIABLES = VARIABLES[:-1]
SPLITS = {"val_2020.zarr": ("2020-01-01T00:00:00", "2020-12-31T18:00:00"), "test_2021_2022.zarr": ("2021-01-01T00:00:00", "2022-12-31T18:00:00")}
DOWNLOAD_WORKERS = 12
MAX_ATTEMPTS = 4


def load_source() -> xr.Dataset:
    try:
        import gcsfs
    except ImportError as exc:
        raise RuntimeError("WeatherBench acquisition failed: gcsfs is required") from exc
    print(f"source={SOURCE}", flush=True)
    return xr.open_zarr(SOURCE, chunks={"time": 48}, consolidated=True, storage_options={"token": "anon"})


def target_dataset(source: xr.Dataset) -> xr.Dataset:
    missing = [name for name in VARIABLES if name not in source.data_vars]
    if missing:
        raise RuntimeError(f"WeatherBench acquisition failed: missing source variables {missing}")
    available = set(np.asarray(source.level.values).tolist())
    if not set(TARGET_LEVELS).issubset(available):
        raise RuntimeError(f"WeatherBench acquisition failed: missing pressure levels; available={sorted(available)}")
    latitude = source.latitude.values
    lat_slice = slice(33.5, 30.0) if latitude[0] > latitude[-1] else slice(30.0, 33.5)
    longitude = source.longitude.values
    if longitude.max() > 180:
        lon_slice = slice(75.5, 79.5)
    else:
        lon_slice = slice(75.5, 79.5)
    result = source[VARIABLES].sel(latitude=lat_slice, longitude=lon_slice, level=TARGET_LEVELS)
    if result.sizes.get("latitude") != 15 or result.sizes.get("longitude") != 17:
        raise RuntimeError(f"WeatherBench acquisition failed: cropped grid is {dict(result.sizes)}, expected 15x17")
    train = open_project_zarr(TRAIN, consolidated=True)
    try:
        if not np.allclose(result.latitude.values, train.latitude.values) or not np.allclose(result.longitude.values, train.longitude.values) or not np.array_equal(result.level.values, train.level.values):
            raise RuntimeError("WeatherBench acquisition failed: cropped coordinates do not match train")
    finally:
        train.close()
    return result


def month_ranges(start: str, end: str):
    current = pd.Timestamp(start).normalize()
    finish = pd.Timestamp(end)
    while current <= finish:
        next_month = current + pd.offsets.MonthBegin(1)
        month_end = min(next_month - pd.Timedelta(hours=6), finish)
        yield current, month_end
        current = next_month


def read_month(ds: xr.Dataset, start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"read_month={start.isoformat()}..{end.isoformat()} attempt={attempt}", flush=True)
            with dask.config.set(scheduler="threads", num_workers=DOWNLOAD_WORKERS):
                return ds.sel(time=slice(start, end)).load()
        except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
            last = exc
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"WeatherBench month read failed: {start.isoformat()}..{end.isoformat()}: {exc}") from exc
            wait = 15 * attempt
            print(f"read_retry_after={wait} s error={type(exc).__name__}:{exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"WeatherBench month read failed: {last}")


def marker_path(split: str, month: pd.Timestamp) -> Path:
    return MARKERS / split / f"{month.year:04d}-{month.month:02d}.json"


def write_month(output: Path, month_data: xr.Dataset, first: bool) -> None:
    for variable in month_data.variables:
        month_data[variable].encoding.clear()
    month_data = month_data.chunk({"time": month_data.sizes["time"], "latitude": -1, "longitude": -1, "level": -1})
    month_data.to_zarr(output, mode="w" if first else "a", append_dim=None if first else "time", zarr_format=2)


def verify_2019(ds: xr.Dataset) -> None:
    train = open_project_zarr(TRAIN, consolidated=True)
    try:
        common = np.intersect1d(ds.time.values, train.time.values)
        if not len(common):
            raise RuntimeError("2019 verification failed: no overlapping timesteps with train")
        failures = []
        for variable in CORE_VARIABLES:
            left = np.asarray(ds[variable].sel(time=common).values, dtype=np.float64)
            right = np.asarray(train[variable].sel(time=common).values, dtype=np.float64)
            maximum = float(np.max(np.abs(left - right)))
            rms = float(np.sqrt(np.mean((left - right) ** 2)))
            print(f"verify_2019 variable={variable} max_abs={maximum} rms={rms}", flush=True)
            if not np.allclose(left, right, rtol=1e-5, atol=1e-4):
                failures.append(variable)
        if failures:
            raise RuntimeError(f"2019 verification failed for channels: {failures}")
    finally:
        train.close()


def acquire_split(source: xr.Dataset, split: str, start: str, end: str) -> None:
    output = OUTPUT_DATASET / split
    split_marker_root = MARKERS / split
    split_marker_root.mkdir(parents=True, exist_ok=True)
    first = not output.exists()
    for month_start, month_end in month_ranges(start, end):
        marker = marker_path(split, month_start)
        if marker.exists():
            print(f"resume_marker={marker}", flush=True)
            first = False
            continue
        month_data = read_month(source, month_start, month_end)
        if month_data.sizes.get("time", 0) == 0:
            raise RuntimeError(f"WeatherBench acquisition failed: empty month {month_start:%Y-%m}")
        write_month(output, month_data, first)
        first = False
        marker.write_text(json.dumps({"month": month_start.strftime("%Y-%m"), "timesteps": int(month_data.sizes["time"]), "source": SOURCE}), encoding="utf-8")
        print(f"written_month={month_start:%Y-%m} timesteps={month_data.sizes['time']}", flush=True)
    stamp_store(output, source=SOURCE, request_parameters={"latitude": [33.5, 30.0], "longitude": [75.5, 79.5], "levels_hpa": TARGET_LEVELS, "variables": VARIABLES, "start": start, "end": end}, download_date=pd.Timestamp.utcnow().date().isoformat())
    print(f"stamped_store={output}", flush=True)


def main() -> None:
    OUTPUT_DATASET.mkdir(parents=True, exist_ok=True)
    source = load_source()
    cropped = target_dataset(source)
    try:
        for month_start, month_end in month_ranges("2019-01-01", "2019-12-31T18:00:00"):
            verify_2019(read_month(cropped, month_start, month_end))
        print("verify_2019=PASS", flush=True)
        for split, (start, end) in SPLITS.items():
            acquire_split(cropped, split, start, end)
    finally:
        source.close()


if __name__ == "__main__":
    main()
