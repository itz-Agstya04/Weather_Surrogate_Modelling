"""Probe public anonymous ERA5 Zarr sources before acquisition."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dask
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "DATASET" / "train_2018_2019.zarr"
SOURCES = {
    "weatherbench2": "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr",
    "arco": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
}
CHANNELS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential", "temperature"]
EXTRA = "surface_pressure"
LEVELS = [1000, 850, 700, 500, 300]
START = np.datetime64("2019-01-01T00:00:00")
N_TIMESTEPS = 20
TOTAL_TIMESTEPS = 5844


def coordinate_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    raise RuntimeError(f"Coordinate not found; tried {candidates}, available={list(ds.coords)}")


def crop(ds: xr.Dataset, reference: xr.Dataset) -> xr.Dataset:
    lat_name = coordinate_name(ds, ("latitude", "lat"))
    lon_name = coordinate_name(ds, ("longitude", "lon"))
    ref_lat = np.asarray(reference.latitude.values)
    ref_lon = np.asarray(reference.longitude.values)
    lat = np.asarray(ds[lat_name].values)
    lon = np.asarray(ds[lon_name].values)
    if lon.max() > 180 and ref_lon.min() < 0:
        target_lon = np.mod(ref_lon, 360)
    elif lon.max() <= 180 and ref_lon.max() > 180:
        target_lon = ((ref_lon + 180) % 360) - 180
    else:
        target_lon = ref_lon
    lat_slice = slice(float(ref_lat.max()), float(ref_lat.min())) if lat[0] > lat[-1] else slice(float(ref_lat.min()), float(ref_lat.max()))
    lon_slice = slice(float(target_lon.min()), float(target_lon.max()))
    selected = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    if selected.sizes.get(lat_name) != 15 or selected.sizes.get(lon_name) != 17:
        raise RuntimeError(f"Grid crop failed: got {dict(selected.sizes)}, expected 15x17")
    if not np.allclose(selected[lat_name].values, ref_lat) or not np.allclose(selected[lon_name].values, target_lon):
        raise RuntimeError("Grid coordinate mismatch against existing train store")
    return selected


def selected_variables(ds: xr.Dataset) -> list[str]:
    missing = [name for name in CHANNELS + [EXTRA] if name not in ds.data_vars]
    if missing:
        raise RuntimeError(f"Required variables missing: {missing}")
    if "level" not in ds.coords or not set(LEVELS).issubset(set(np.asarray(ds.level.values).tolist())):
        raise RuntimeError(f"Required pressure levels missing; available={np.asarray(ds.level.values).tolist() if 'level' in ds.coords else []}")
    return CHANNELS + [EXTRA]


def time_name(ds: xr.Dataset) -> str:
    return coordinate_name(ds, ("time", "valid_time"))


def chunk_summary(data: object) -> object:
    chunks = getattr(data, "chunks", None)
    if chunks is None:
        return None
    return tuple((len(axis), axis[0], axis[-1]) for axis in chunks)


def probe(name: str, url: str, reference: xr.Dataset) -> dict:
    print(f"\nSOURCE {name} {url}")
    ds = xr.open_zarr(url, consolidated=True, storage_options={"token": "anon"})
    try:
        print(f"dims={dict(ds.sizes)}")
        for variable in CHANNELS + [EXTRA, "geopotential_at_surface"]:
            if variable in ds:
                data = ds[variable]
                print(f"variable={variable} dtype={data.dtype} dims={data.dims} chunk_summary={chunk_summary(data.data)}")
            else:
                print(f"variable={variable} missing")
        selected_variables(ds)
        cropped = crop(ds, reference)
        tname = time_name(cropped)
        all_times = np.asarray(cropped[tname].values)
        if name == "arco":
            requested = all_times[(all_times.astype('datetime64[h]').astype(int) % 6) == 0]
            requested = requested[(requested >= START) & (requested < START + np.timedelta64(120, 'h'))][:N_TIMESTEPS]
        else:
            requested = all_times[(all_times >= START) & (all_times < START + np.timedelta64(6 * N_TIMESTEPS, 'h'))][:N_TIMESTEPS]
        if len(requested) != N_TIMESTEPS:
            raise RuntimeError(f"Could not select {N_TIMESTEPS} consecutive six-hour timesteps; found {len(requested)}")
        selected = cropped[CHANNELS + [EXTRA]].sel({tname: requested})
        start = time.perf_counter()
        with dask.config.set(scheduler="threads"):
            materialized = selected.load()
        elapsed = time.perf_counter() - start
        payload_bytes = int(sum(np.asarray(materialized[var].values).nbytes for var in materialized.data_vars))
        per_step_bytes = payload_bytes / N_TIMESTEPS
        seconds_per_step = elapsed / N_TIMESTEPS
        estimate_bytes = per_step_bytes * TOTAL_TIMESTEPS
        estimate_seconds = seconds_per_step * TOTAL_TIMESTEPS
        print(f"read_timesteps={N_TIMESTEPS} count")
        print(f"read_seconds={elapsed:.6f} s")
        print(f"payload_bytes_per_timestep={per_step_bytes:.3f} bytes")
        print(f"estimated_transfer_2019_2022={estimate_bytes:.3f} bytes")
        print(f"estimated_wall_2019_2022={estimate_seconds:.3f} s")
        return {"name": name, "url": url, "seconds": elapsed, "payload_bytes": payload_bytes, "bytes_per_timestep": per_step_bytes, "estimated_bytes": estimate_bytes, "estimated_seconds": estimate_seconds}
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "DATASET" / "stage_a_probe.json")
    args = parser.parse_args()
    reference = xr.open_zarr(REFERENCE, consolidated=True)
    try:
        results = [probe(name, url, reference) for name, url in SOURCES.items()]
    finally:
        reference.close()
    results.sort(key=lambda item: item["estimated_seconds"])
    print(f"\nCHEAPER_SOURCE={results[0]['name']}")
    print(f"CHEAPER_ESTIMATED_TRANSFER={results[0]['estimated_bytes']:.3f} bytes")
    print(f"CHEAPER_ESTIMATED_WALL={results[0]['estimated_seconds']:.3f} s")
    if results[0]["estimated_bytes"] > 100 * 1024**3:
        raise RuntimeError("Gate A failed: cheaper source exceeds 100 GB")
    if results[0]["estimated_seconds"] > 12 * 3600:
        raise RuntimeError("Gate A failed: cheaper source exceeds 12 hours")
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("GATE A PASSED")


if __name__ == "__main__":
    main()
