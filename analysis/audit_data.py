from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from data.store import open_project_zarr

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
SPLITS = {
    "train": ROOT / "DATASET" / "train_2018_2019.zarr",
    "val_2020": ROOT / "DATASET" / "val_2020.zarr",
    "test": ROOT / "DATASET" / "test_2021_2022.zarr",
}
LEVELS = [1000, 850, 700, 500, 300]
CHANNELS = [
    ("T2m", "2m_temperature", None),
    ("U10", "10m_u_component_of_wind", None),
    ("V10", "10m_v_component_of_wind", None),
    *[(f"Z{x}", "geopotential", x) for x in LEVELS],
    *[(f"T{x}", "temperature", x) for x in LEVELS],
]
SOURCE_FILES = ["data/prepare_hp_era5.py", "data/update_splits_temperature.py", "data/append_temperature.py"]
SEARCH_TERMS = ["random", "normal", "randn", "noise", "jitter", "augment", "quantize", "round", "float16", "int8", "int16"]


class Capture:
    def __init__(self):
        self.lines = []

    def write(self, value=""):
        text = str(value)
        self.lines.extend(text.splitlines())
        print(text, end="")

    def line(self, value=""):
        self.write(value + "\n")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def array_for(ds, variable, level):
    result = ds[variable]
    if level is not None:
        result = result.sel(level=level)
    return result


def coordinate_text(ds, name):
    values = np.asarray(ds[name].values)
    step = np.diff(values)
    unique = np.unique(step)
    step_text = str(clean(unique[0])) if len(unique) == 1 else str(clean(unique.tolist()))
    return f"first={clean(values[0])},last={clean(values[-1])},step={step_text},order={'ascending' if values[-1] > values[0] else 'descending'},values={clean(values.tolist())}"


def metadata(capture, stores):
    capture.line("ITEM 1: LOCAL ZARR METADATA")
    for split, path in SPLITS.items():
        ds = stores[split]
        capture.line(f"STORE {split} path={path}")
        for label, variable, level in CHANNELS:
            array = array_for(ds, variable, level)
            encoding = clean(array.encoding)
            selected = {key: encoding.get(key) for key in ["scale_factor", "add_offset", "_FillValue"]}
            capture.line(f"CHANNEL {label} source={variable} level={level}")
            capture.line(f"dims={array.dims} shape={array.shape} dtype={array.dtype} chunks={array.encoding.get('chunks')} compressor={encoding.get('compressor', encoding.get('compressors'))} filters={encoding.get('filters')}")
            capture.line(f"attrs={json.dumps(clean(array.attrs), sort_keys=True, default=str)}")
            capture.line(f"encoding={json.dumps(encoding, sort_keys=True, default=str)}")
            capture.line(f"encoding_scale_add_fill={json.dumps(selected, sort_keys=True, default=str)}")
        for coordinate in ["latitude", "longitude", "level"]:
            capture.line(f"COORDINATE {coordinate} {coordinate_text(ds, coordinate)}")
    capture.line("COORDINATE IDENTITY")
    for coordinate in ["latitude", "longitude", "level"]:
        reference = stores["train"][coordinate].values
        parts = []
        for split in SPLITS:
            current = stores[split][coordinate].values
            parts.append(f"{split}=identical:{bool(np.array_equal(reference, current))},max_abs_diff:{float(np.max(np.abs(reference.astype(float) - current.astype(float))))}")
        capture.line(f"{coordinate}: " + "; ".join(parts))


def remote_values(remote, variable, level, times, latitude, longitude):
    chunks = []
    for start in range(0, len(times), len(times)):
        query = remote[variable].sel(time=times[start:start + len(times)], method="nearest")
        query = query.sel(latitude=latitude, longitude=longitude, method="nearest")
        if level is not None:
            query = query.sel(level=level)
        chunks.append(np.asarray(query.compute(scheduler="threads", num_workers=8).values, dtype=np.float64))
    return np.concatenate(chunks, axis=0)


def item2(capture, stores):
    capture.line("ITEM 2: SEEDED LOCAL VERSUS WEATHERBENCH2")
    seed = 20261005
    rng = np.random.default_rng(seed)
    remote = xr.open_zarr(REMOTE, consolidated=True, storage_options={"token": "anon"})
    capture.line(f"remote={REMOTE} seed={seed} samples_per_split=40 selection=nearest_latitude_longitude_and_nearest_time")
    for split, ds in stores.items():
        indices = np.sort(rng.choice(len(ds.time), size=40, replace=False))
        times = ds.time.values[indices]
        capture.line(f"SPLIT {split} local_indices={indices.tolist()} local_times={times.tolist()}")
        remote_cache = {}
        for variable in ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential", "temperature"]:
            remote_cache[variable] = remote_values(remote, variable, None, times, ds.latitude.values, ds.longitude.values)
        for label, variable, level in CHANNELS:
            local = np.asarray(array_for(ds, variable, level).isel(time=indices).values, dtype=np.float64)
            wb2 = remote_cache[variable]
            if level is not None:
                wb2 = wb2[:, LEVELS.index(level)]
            diff = local - wb2
            capture.line(f"{split},{label},max_abs_diff={np.max(np.abs(diff)):.12g},mean_diff={np.mean(diff):.12g},RMS_diff={np.sqrt(np.mean(diff ** 2)):.12g},diff_std={np.std(diff):.12g}")
        sample_index = int(indices[0])
        sample_time = ds.time.values[sample_index:sample_index + 1]
        local_z = np.asarray(array_for(ds, "geopotential", 500).isel(time=sample_index).values)
        local_t = np.asarray(array_for(ds, "temperature", 500).isel(time=sample_index).values)
        wb_z = remote_values(remote, "geopotential", 500, sample_time, ds.latitude.values, ds.longitude.values)[0]
        wb_t = remote_values(remote, "temperature", 500, sample_time, ds.latitude.values, ds.longitude.values)[0]
        capture.line(f"{split} ONE_TIMESTAMP time={sample_time[0]}")
        capture.line("LOCAL_Z500_FIRST3_ROWS_FIRST5_COLS=" + repr(local_z[:3, :5].tolist()))
        capture.line("WB2_Z500_FIRST3_ROWS_FIRST5_COLS=" + repr(wb_z[:3, :5].tolist()))
        capture.line("LOCAL_T500_FIRST3_ROWS_FIRST5_COLS=" + repr(local_t[:3, :5].tolist()))
        capture.line("WB2_T500_FIRST3_ROWS_FIRST5_COLS=" + repr(wb_t[:3, :5].tolist()))


def item3(capture, stores):
    capture.line("ITEM 3: LOCAL SPATIAL AND TEMPORAL DIFFERENCE AUDIT")
    capture.line("split,channel,zonal_adjacent_std,meridional_adjacent_std,temporal_6h_diff_std,raw_space_time_std,flag")
    for split, ds in stores.items():
        for label, variable, level in CHANNELS:
            values = np.asarray(array_for(ds, variable, level).values, dtype=np.float64)
            zonal = np.diff(values, axis=2)
            meridional = np.diff(values, axis=1)
            temporal = np.diff(values, axis=0)
            raw_std = np.std(values)
            threshold = 100.0 if label.startswith("Z") else 2.0 if label.startswith("T") and label != "T2m" else 10.0
            flag = "FAR_ABOVE" if max(np.std(zonal), np.std(meridional)) > threshold else "within_requested_reference"
            capture.line(f"{split},{label},{np.std(zonal):.9f},{np.std(meridional):.9f},{np.std(temporal):.9f},{raw_std:.9f},{flag}")


def item4(capture):
    capture.line("ITEM 4: REPOSITORY SEARCH")
    capture.line("terms=" + ",".join(SEARCH_TERMS))
    pattern = "|".join(SEARCH_TERMS)
    command = ["rg", "-n", "-i", "--hidden", "--glob", "!.git/**", "--glob", "!*.pt", "--glob", "!*.zip", pattern, str(ROOT)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    capture.write(result.stdout)
    capture.line("SEARCH_EXIT_CODE=" + str(result.returncode))
    capture.line("EXACT_SOURCE_CREATION_CODE")
    for relative in SOURCE_FILES:
        path = ROOT / relative
        capture.line(f"FILE {relative}")
        capture.write(path.read_text(encoding="utf-8"))


def month_table(capture, stores):
    capture.line("ITEM 5: MONTHLY SPATIAL MEANS")
    capture.line("split,month,T2m,T500,Z500,U10,V10")
    for split, ds in stores.items():
        values = {
            "T2m": np.asarray(array_for(ds, "2m_temperature", None).values, dtype=np.float64),
            "T500": np.asarray(array_for(ds, "temperature", 500).values, dtype=np.float64),
            "Z500": np.asarray(array_for(ds, "geopotential", 500).values, dtype=np.float64),
            "U10": np.asarray(array_for(ds, "10m_u_component_of_wind", None).values, dtype=np.float64),
            "V10": np.asarray(array_for(ds, "10m_v_component_of_wind", None).values, dtype=np.float64),
        }
        times = ds.time.values.astype("datetime64[M]")
        for month in np.unique(times):
            mask = times == month
            means = [values[name][mask].mean() for name in ["T2m", "T500", "Z500", "U10", "V10"]]
            capture.line(f"{split},{str(month)},{','.join(f'{value:.9f}' for value in means)}")


def item6(capture, stores):
    capture.line("ITEM 6: CENTRAL CELL EIGHT CONSECUTIVE TIMESTAMPS")
    for split, ds in stores.items():
        row = len(ds.latitude) // 2
        col = len(ds.longitude) // 2
        capture.line(f"SPLIT {split} latitude={float(ds.latitude.values[row])} longitude={float(ds.longitude.values[col])}")
        for index in range(8):
            values = [
                array_for(ds, "2m_temperature", None).isel(time=index, latitude=row, longitude=col).values.item(),
                array_for(ds, "10m_u_component_of_wind", None).isel(time=index, latitude=row, longitude=col).values.item(),
                array_for(ds, "geopotential", 500).isel(time=index, latitude=row, longitude=col).values.item(),
                array_for(ds, "temperature", 500).isel(time=index, latitude=row, longitude=col).values.item(),
            ]
            capture.line(f"{ds.time.values[index]},{','.join(f'{value:.9f}' for value in values)}")


def item7(capture, stores):
    capture.line("ITEM 7: COMPUTED CONCLUSIONS")
    groups = [("surface", ["T2m", "U10", "V10"]), ("geopotential", [f"Z{x}" for x in LEVELS]), ("temperature_aloft", [f"T{x}" for x in LEVELS])]
    for split, ds in stores.items():
        for group, labels in groups:
            rows = []
            for label in labels:
                _, variable, level = next(item for item in CHANNELS if item[0] == label)
                values = np.asarray(array_for(ds, variable, level).values, dtype=np.float64)
                zonal = np.std(np.diff(values, axis=2))
                meridional = np.std(np.diff(values, axis=1))
                temporal = np.std(np.diff(values, axis=0))
                raw = np.std(values)
                rows.append((label, zonal, meridional, temporal, raw))
            max_adj = max(max(row[1], row[2]) for row in rows)
            raw_values = [row[4] for row in rows]
            if group == "geopotential":
                state = "noisy" if max_adj > 100.0 or np.mean(raw_values) > 450.0 else "clean"
                support = f"max adjacent-cell std={max_adj:.9f} m2/s2, mean raw std={np.mean(raw_values):.9f} m2/s2"
            elif group == "temperature_aloft":
                state = "noisy" if max_adj > 2.0 or np.mean(raw_values) > 10.0 else "clean"
                support = f"max adjacent-cell std={max_adj:.9f} K, mean raw std={np.mean(raw_values):.9f} K"
            else:
                state = "clean" if max_adj <= 10.0 else "noisy"
                support = f"max adjacent-cell std={max_adj:.9f}, mean raw std={np.mean(raw_values):.9f}"
            capture.line(f"{split},{group},{state},{support}")
        t2 = np.asarray(array_for(ds, "2m_temperature", None).values, dtype=np.float64)
        t500 = np.asarray(array_for(ds, "temperature", 500).values, dtype=np.float64)
        z500 = np.asarray(array_for(ds, "geopotential", 500).values, dtype=np.float64)
        capture.line(f"{split},shift_support,spatial_mean_T2m={t2.mean():.9f},spatial_mean_T500={t500.mean():.9f},spatial_mean_Z500={z500.mean():.9f}")


def main():
    capture = Capture()
    stores = {name: open_project_zarr(path, consolidated=True) for name, path in SPLITS.items()}
    capture.line("PHASE 0C DATA INTEGRITY AUDIT")
    metadata(capture, stores)
    item2(capture, stores)
    item3(capture, stores)
    item4(capture)
    month_table(capture, stores)
    item6(capture, stores)
    item7(capture, stores)
    report = "# Phase 0C data integrity audit\n\n" + "\n".join(capture.lines) + "\n"
    (ROOT / "REPORT_phase0c.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
