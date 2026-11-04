from __future__ import annotations

import json
import shutil
from pathlib import Path

import dask
import numcodecs
import numpy as np
import xarray as xr
import zarr
from data.store import open_project_zarr, stamp_store

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
SOURCE_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential", "temperature"]
LEVELS = [1000, 850, 700, 500, 300]
SPLITS = {
    "train_2018_2019.zarr": ("2018-01-01T00:00:00", "2019-12-31T23:59:59"),
    "val_2020.zarr": ("2020-01-01T00:00:00", "2020-12-31T23:59:59"),
    "test_2021_2022.zarr": ("2021-01-01T00:00:00", "2022-12-31T23:59:59"),
}
BATCH_SIZE = 512


def fetch(remote, times, latitude, longitude):
    queries = []
    for variable in SOURCE_VARS:
        query = remote[variable].sel(time=times, method="nearest").sel(latitude=latitude, longitude=longitude, method="nearest")
        if variable in ["geopotential", "temperature"]:
            query = query.sel(level=LEVELS)
        queries.append(query)
    for attempt in range(1, 5):
        try:
            values = dask.compute(*queries, scheduler="threads", num_workers=16)
            return {variable: value.astype(np.float32) for variable, value in zip(SOURCE_VARS, values)}
        except Exception as exc:
            print(f"fetch_retry={attempt},error={type(exc).__name__}:{exc}", flush=True)
    raise RuntimeError("WB2 fetch failed after four attempts")


def build_dataset(remote, times, latitude, longitude):
    arrays = fetch(remote, times, latitude, longitude)
    data = {
        "2m_temperature": (("time", "latitude", "longitude"), arrays["2m_temperature"].values),
        "10m_u_component_of_wind": (("time", "latitude", "longitude"), arrays["10m_u_component_of_wind"].values),
        "10m_v_component_of_wind": (("time", "latitude", "longitude"), arrays["10m_v_component_of_wind"].values),
        "geopotential": (("time", "level", "latitude", "longitude"), arrays["geopotential"].values),
        "temperature": (("time", "level", "latitude", "longitude"), arrays["temperature"].values),
    }
    return xr.Dataset(data, coords={"time": times, "level": np.asarray(LEVELS, dtype=np.int64), "latitude": latitude.astype(np.float32), "longitude": longitude.astype(np.float32)})


def main():
    local_reference = open_project_zarr(ROOT / "DATASET" / "train_2018_2019.zarr", consolidated=True)
    latitude = local_reference.latitude.values
    longitude = local_reference.longitude.values
    remote = xr.open_zarr(REMOTE, consolidated=True, storage_options={"token": "anon"})
    clean_root = ROOT / "DATASET" / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    for filename, (start, end) in SPLITS.items():
        local = open_project_zarr(ROOT / "DATASET" / filename, consolidated=True)
        times = local.time.values
        output = clean_root / filename
        if output.exists():
            shutil.rmtree(output)
        for index in range(0, len(times), BATCH_SIZE):
            batch_times = times[index:index + BATCH_SIZE]
            batch = build_dataset(remote, batch_times, latitude, longitude)
            encoding = {name: {"dtype": "float32", "compressor": compressor, "chunks": tuple(min(BATCH_SIZE, size) if axis == 0 else size for axis, size in enumerate(batch[name].shape))} for name in batch.data_vars}
            if index == 0:
                batch.to_zarr(output, mode="w", encoding=encoding, zarr_format=2)
            else:
                batch.to_zarr(output, mode="a", append_dim="time", encoding=encoding, zarr_format=2)
            print(f"{filename},samples_written={min(index + BATCH_SIZE, len(times))},total={len(times)}", flush=True)
        zarr.consolidate_metadata(str(output))
        stamp_store(
            output,
            source=REMOTE,
            request_parameters={"variables": SOURCE_VARS, "levels_hpa": LEVELS, "time_start": start, "time_end": end},
            download_date=__import__("datetime").date.today().isoformat(),
        )
        print(f"completed={output}", flush=True)


if __name__ == "__main__":
    main()
