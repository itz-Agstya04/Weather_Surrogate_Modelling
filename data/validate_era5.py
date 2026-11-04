"""Strict ERA5 split validation and Gate 1 reporting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.store import open_project_zarr

DEFAULTS = {"train": ROOT / "DATASET" / "train_2018_2019.zarr", "val": ROOT / "DATASET" / "val_2020.zarr", "test": ROOT / "DATASET" / "test_2021_2022.zarr"}
SURFACE = "2m_temperature"
Z500 = ("geopotential", 500)
T500 = ("temperature", 500)
LEADS = (6, 12, 24, 48, 72)
G0 = 9.80665


def field(ds: xr.Dataset, spec: tuple[str, int] | str) -> np.ndarray:
    name, level = spec if isinstance(spec, tuple) else (spec, None)
    values = np.asarray(ds[name].sel(level=level).values if level is not None else ds[name].values, dtype=np.float64)
    return values / G0 if name == "geopotential" else values


def flat_time_mean(values: np.ndarray) -> np.ndarray:
    return values.reshape(values.shape[0], -1).mean(axis=1)


def autocorrelation(values: np.ndarray, lag: int = 1) -> float:
    series = flat_time_mean(values)
    return float(np.corrcoef(series[:-lag], series[lag:])[0, 1])


def persistence_rmse(values: np.ndarray, lead_steps: int) -> float:
    return float(np.sqrt(np.mean((values[:-lead_steps] - values[lead_steps:]) ** 2)))


def annual_harmonic_explained_variance(values: np.ndarray, times: xr.DataArray) -> float:
    series = flat_time_mean(values)
    phase = (np.asarray(times.dt.dayofyear) - 1 + np.asarray(times.dt.hour) / 24.0) * 2 * np.pi / 365.2425
    design = np.column_stack((np.ones(series.size), np.sin(phase), np.cos(phase)))
    prediction = design @ np.linalg.lstsq(design, series, rcond=None)[0]
    return float(1.0 - np.sum((series - prediction) ** 2) / np.sum((series - series.mean()) ** 2))


def neighbor_correlation(values: np.ndarray) -> float:
    grid = values.reshape(values.shape[0], values.shape[-2], values.shape[-1])
    correlations = []
    for left, right in ((grid[:, :, :-1], grid[:, :, 1:]), (grid[:, :-1, :], grid[:, 1:, :])):
        for index in np.ndindex(left.shape[1:]):
            correlations.append(np.corrcoef(left[(slice(None),) + index], right[(slice(None),) + index])[0, 1])
    return float(np.nanmean(correlations))


def seasonal_climatology(ds: xr.Dataset) -> dict:
    result = {}
    for name in (SURFACE, "geopotential", "temperature"):
        arr = ds[name]
        levels = ds.level.values if "level" in arr.dims else [None]
        for level in levels:
            current = arr if level is None else arr.sel(level=level)
            group = np.asarray(current.time.dt.dayofyear) * 24 + np.asarray(current.time.dt.hour)
            values = np.asarray(current.values, dtype=np.float64)
            key = (name, int(level)) if level is not None else name
            result[key] = {int(g): values[group == g].mean(axis=0) for g in np.unique(group)}
    return result


def seasonal_stats(ds: xr.Dataset, climatology: dict) -> dict:
    result = {}
    for name in (SURFACE, "geopotential", "temperature"):
        arr = ds[name]
        levels = ds.level.values if "level" in arr.dims else [None]
        residuals = []
        for level in levels:
            current = arr if level is None else arr.sel(level=level)
            group = np.asarray(current.time.dt.dayofyear) * 24 + np.asarray(current.time.dt.hour)
            values = np.asarray(current.values, dtype=np.float64)
            key = (name, int(level)) if level is not None else name
            residuals.append(values - np.asarray([climatology[key][int(g)] for g in group]))
        combined = np.concatenate([value.reshape(value.shape[0], -1) for value in residuals], axis=1)
        result[name] = (float(combined.mean()), float(combined.std()))
    return result


def check_single_split(name: str, ds: xr.Dataset) -> list[tuple[str, object, str, bool]]:
    checks = []
    missing = sorted({SURFACE, "geopotential", "temperature"} - set(ds.data_vars))
    checks.append(("required variables", ",".join(missing) if missing else "present", "all present", not missing))
    for var in ds.data_vars:
        values = np.asarray(ds[var].values)
        finite = bool(np.isfinite(values).all())
        checks.append((f"{var} finite", "yes" if finite else "no", "no NaNs/inf", finite))
        if "time" in ds[var].dims:
            axis = ds[var].get_axis_num("time")
            temporal = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
            constant = bool(np.any(np.std(temporal, axis=0) == 0))
            checks.append((f"{var} constant-in-time cells", int(constant), "0", not constant))
    times = np.asarray(ds.time.values).astype("datetime64[ns]")
    duplicate = len(np.unique(times)) != len(times)
    diffs = np.diff(times).astype("timedelta64[h]").astype(np.int64)
    continuous = bool(len(diffs) and np.all(diffs == 6))
    checks.extend([("duplicate timesteps", int(duplicate), "0", not duplicate), ("6-hourly axis", "6 h" if continuous else "not 6 h", "6 h throughout", continuous)])
    z500, t500, t2m = field(ds, Z500), field(ds, T500), field(ds, SURFACE)
    z_lag, t_lag = autocorrelation(z500), autocorrelation(t500)
    t2m_ac = [autocorrelation(t2m, lag) for lag in range(1, 13)]
    z_rmse = {h: persistence_rmse(z500, h // 6) for h in LEADS}
    t_rmse = {h: persistence_rmse(t500, h // 6) for h in LEADS}
    t2m_rmse = {h: persistence_rmse(t2m, h // 6) for h in LEADS}
    annual = annual_harmonic_explained_variance(t2m, ds.time)
    z_neighbor, t_neighbor = neighbor_correlation(z500), neighbor_correlation(t500)
    checks.extend([
        ("Z500 lag-1 autocorrelation", z_lag, "> 0.95", z_lag > 0.95),
        ("T500 lag-1 autocorrelation", t_lag, "> 0.95", t_lag > 0.95),
        ("T2m diurnal peak lag", int(np.argmax(t2m_ac) + 1), "4", int(np.argmax(t2m_ac) + 1) == 4),
        ("T2m annual harmonic explained variance", annual, "> 0.7", annual > 0.7),
        ("Z500 neighbor-cell correlation", z_neighbor, "> 0.9", z_neighbor > 0.9),
        ("T500 neighbor-cell correlation", t_neighbor, "> 0.9", t_neighbor > 0.9),
        ("Z500 persistence RMSE 6 h", z_rmse[6], "10-30 m", 10 <= z_rmse[6] <= 30),
        ("Z500 persistence RMSE 24<48<72 h", f"{z_rmse[24]:.6g} < {z_rmse[48]:.6g} < {z_rmse[72]:.6g}", "strictly increasing", z_rmse[24] < z_rmse[48] < z_rmse[72]),
        ("Z500 persistence RMSE 72 h / 6 h", z_rmse[72] / z_rmse[6], ">= 2", z_rmse[72] >= 2 * z_rmse[6]),
        ("Z500 persistence RMSE 6->12 h dip", z_rmse[12] - z_rmse[6], "reported; allowed", True),
        ("T500 persistence RMSE 24<48<72 h", f"{t_rmse[24]:.6g} < {t_rmse[48]:.6g} < {t_rmse[72]:.6g}", "strictly increasing", t_rmse[24] < t_rmse[48] < t_rmse[72]),
        ("T2m persistence RMSE 24 h < 12 h", f"{t2m_rmse[24]:.6g} < {t2m_rmse[12]:.6g}", "strictly less", t2m_rmse[24] < t2m_rmse[12]),
    ])
    return checks


def print_checks(split: str, checks: list[tuple[str, object, str, bool]]) -> bool:
    passed = True
    print(f"\n[{split}]")
    for label, value, threshold, ok in checks:
        if "autocorrelation" in label or "explained variance" in label or "correlation" in label or " / " in label:
            unit = "dimensionless"
        elif "RMSE" in label or "dip" in label:
            unit = "m" if "Z500" in label else "K"
        elif "constant" in label:
            unit = "cells"
        elif "duplicate" in label or "overlap" in label:
            unit = "timesteps"
        elif "peak lag" in label:
            unit = "6-hour lags"
        else:
            unit = ""
        value_text = str(value) if unit and unit in str(value) else f"{value} {unit}" if unit else str(value)
        threshold_text = str(threshold) if unit and unit in str(threshold) else f"{threshold} {unit}" if unit else str(threshold)
        print(f"{'PASS' if ok else 'FAIL'} | {label} | value={value_text} | threshold={threshold_text}")
        passed &= ok
    return passed


def validate(paths: dict[str, Path], train_only: bool = False) -> bool:
    stores = {}
    passed = True
    try:
        for name, path in paths.items():
            stores[name] = open_project_zarr(path, consolidated=True)
            print(f"\nPROVENANCE {name}: {dict(stores[name].attrs)}")
            passed &= print_checks(name, check_single_split(name, stores[name]))
        if not train_only:
            for left, right in (("train", "val"), ("val", "test"), ("train", "test")):
                overlap = np.intersect1d(stores[left].time.values, stores[right].time.values).size
                ok = overlap == 0
                print(f"{'PASS' if ok else 'FAIL'} | split overlap {left}/{right} | value={overlap} timesteps | threshold=0")
                passed &= ok
            train_stats = seasonal_stats(stores["train"], seasonal_climatology(stores["train"]))
            for split in ("val", "test"):
                stats = seasonal_stats(stores[split], seasonal_climatology(stores["train"]))
                for variable in (SURFACE, "geopotential", "temperature"):
                    train_mean, train_std = train_stats[variable]
                    mean, std = stats[variable]
                    mean_ok = abs(mean - train_mean) <= 0.15 * max(abs(train_mean), train_std, 1e-12)
                    std_ok = abs(std - train_std) <= 0.15 * max(train_std, 1e-12)
                    print(f"{'PASS' if mean_ok else 'FAIL'} | {split} {variable} seasonal-residual mean | value={mean} | threshold=within 15 percent of {train_mean}")
                    print(f"{'PASS' if std_ok else 'FAIL'} | {split} {variable} seasonal-residual std | value={std} | threshold=within 15 percent of {train_std}")
                    passed &= mean_ok and std_ok
            tmeans = [float(stores[name][SURFACE].mean()) for name in ("train", "val", "test")]
            tmean_ok = max(tmeans) - min(tmeans) < 3.0
            print(f"{'PASS' if tmean_ok else 'FAIL'} | T2m split mean spread | value={tmeans} K | threshold=< 3 K")
            passed &= tmean_ok
    finally:
        for ds in stores.values():
            ds.close()
    if not passed:
        raise RuntimeError("ERA5 Gate 1 FAILED")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULTS["train"])
    parser.add_argument("--val", type=Path, default=DEFAULTS["val"])
    parser.add_argument("--test", type=Path, default=DEFAULTS["test"])
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    paths = {"train": args.train, "val": args.val, "test": args.test}
    if args.train_only:
        paths = {"train": args.train}
    validate(paths, train_only=args.train_only)
    print("\nERA5 GATE 1 PASSED")


if __name__ == "__main__":
    main()
