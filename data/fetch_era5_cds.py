"""Acquire and assemble real ERA5 data through the Copernicus CDS API."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from data.store import open_project_zarr, stamp_store


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "DATASET"
DOWNLOADS = DATASET / "cds_downloads"
PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
SURFACE_DATASET = "reanalysis-era5-single-levels"
PRESSURE_LEVELS = ["1000", "850", "700", "500", "300"]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
AREA = [33.5, 75.5, 30.0, 79.5]
VARIABLES = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential", "temperature"]
SPLITS = {"train_2000_2019.zarr": (2000, 2019), "val_2020.zarr": (2020, 2020), "test_2021_2022.zarr": (2021, 2022)}


def require_cds() -> object:
    credentials = Path.home() / ".cdsapirc"
    if not credentials.exists():
        raise RuntimeError("CDS setup failed: ~/.cdsapirc is missing. Create it from the Copernicus Climate Data Store profile page with your url and key, then rerun.")
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("CDS setup failed: cdsapi is not installed. Install it with .\\.venv\\Scripts\\pip.exe install cdsapi, then rerun.") from exc
    try:
        return cdsapi.Client()
    except Exception as exc:
        raise RuntimeError(f"CDS setup failed: credentials were rejected or the CDS client could not start: {exc}") from exc


def request(client: object, dataset: str, request_parameters: dict, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"verified-file-candidate={target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, 5):
        try:
            print(f"download dataset={dataset} target={target} attempt={attempt}", flush=True)
            client.retrieve(dataset, request_parameters, str(target))
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(f"CDS returned an empty file: {target}")
            return
        except Exception as exc:
            last = exc
            if attempt == 4:
                raise RuntimeError(f"CDS download failed after four attempts for {target}: {exc}") from exc
            time.sleep(15 * attempt)
    raise RuntimeError(f"CDS download failed: {last}")


def yearly_requests(client: object, years: range) -> None:
    months = [f"{month:02d}" for month in range(1, 13)]
    days = [f"{day:02d}" for day in range(1, 32)]
    for year in years:
        common = {"product_type": "reanalysis", "year": str(year), "month": months, "day": days, "time": TIMES, "area": AREA, "format": "netcdf"}
        request(client, PRESSURE_DATASET, {**common, "variable": ["geopotential", "temperature"], "pressure_level": PRESSURE_LEVELS}, DOWNLOADS / f"pressure_{year}.nc")
        request(client, SURFACE_DATASET, {**common, "variable": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential"]}, DOWNLOADS / f"surface_{year}.nc")


def open_years(years: range) -> xr.Dataset:
    pressure = xr.concat([xr.open_dataset(DOWNLOADS / f"pressure_{year}.nc") for year in years], dim="valid_time")
    surface = xr.concat([xr.open_dataset(DOWNLOADS / f"surface_{year}.nc") for year in years], dim="valid_time")
    pressure = pressure.rename({"valid_time": "time"})
    surface = surface.rename({"valid_time": "time"})
    pressure = pressure.sortby("time")
    surface = surface.sortby("time")
    result = xr.Dataset({
        "2m_temperature": surface["t2m"],
        "10m_u_component_of_wind": surface["u10"],
        "10m_v_component_of_wind": surface["v10"],
        "geopotential": pressure["z"],
        "temperature": pressure["t"],
    })
    result = result.sel(latitude=slice(33.5, 30.0), longitude=slice(75.5, 79.5), level=PRESSURE_LEVELS)
    return result


def verify_grid(ds: xr.Dataset, reference: xr.Dataset) -> None:
    if ds.sizes.get("latitude") != 15 or ds.sizes.get("longitude") != 17:
        raise RuntimeError(f"Grid verification failed: got {dict(ds.sizes)} instead of 15x17")
    for coordinate in ("latitude", "longitude", "level"):
        if not np.array_equal(ds[coordinate].values, reference[coordinate].values):
            raise RuntimeError(f"Grid verification failed: coordinate mismatch in {coordinate}")


def compare_overlap(new_ds: xr.Dataset, old_path: Path) -> None:
    old = open_project_zarr(old_path, consolidated=True)
    try:
        common = np.intersect1d(new_ds.time.values, old.time.values)
        if not common.size:
            raise RuntimeError("Overlap verification failed: no shared 2018-2019 timesteps")
        failures = []
        for variable in VARIABLES:
            left = np.asarray(new_ds[variable].sel(time=common).values, dtype=np.float64)
            right = np.asarray(old[variable].sel(time=common).values, dtype=np.float64)
            maximum = float(np.max(np.abs(left - right)))
            rms = float(np.sqrt(np.mean((left - right) ** 2)))
            print(f"overlap variable={variable} max_abs={maximum} rms={rms}")
            if not np.allclose(left, right, rtol=1e-5, atol=1e-4):
                failures.append(variable)
        if failures:
            raise RuntimeError(f"Overlap verification failed for channels: {', '.join(failures)}")
    finally:
        old.close()


def assemble(reference_path: Path) -> None:
    reference = open_project_zarr(reference_path, consolidated=True)
    try:
        for filename, (start, end) in SPLITS.items():
            years = range(start, end + 1)
            dataset = open_years(years)
            verify_grid(dataset, reference)
            if start <= 2018 <= end:
                compare_overlap(dataset, reference_path)
            output = DATASET / filename
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite existing store: {output}")
            dataset.to_zarr(output, mode="w", zarr_format=2)
            stamp_store(output, source="Copernicus Climate Data Store", request_parameters={"pressure_dataset": PRESSURE_DATASET, "surface_dataset": SURFACE_DATASET, "years": [start, end], "area": AREA, "times": TIMES, "levels": PRESSURE_LEVELS, "variables": VARIABLES}, download_date=date.today().isoformat())
            print(f"assembled={output}")
    finally:
        reference.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DATASET / "train_2018_2019.zarr")
    args = parser.parse_args()
    client = require_cds()
    yearly_requests(client, range(2000, 2023))
    assemble(args.reference)


if __name__ == "__main__":
    main()
