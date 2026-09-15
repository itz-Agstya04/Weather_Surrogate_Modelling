"""Reproducible project pipeline with explicit artifact and timing checks."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "DATASET")
SPLITS = ["train_2018_2019.zarr", "val_2020.zarr", "test_2021_2022.zarr"]
REQUIRED_VARS = {"2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "geopotential", "temperature"}


def exists(path):
    return os.path.exists(os.path.join(ROOT, path))


def zarr_has_required(path):
    import xarray as xr
    try:
        ds = xr.open_zarr(os.path.join(ROOT, path))
        return REQUIRED_VARS.issubset(ds.data_vars)
    except Exception:
        return False


def require_artifacts(paths, step):
    missing = [path for path in paths if not exists(path)]
    if missing:
        raise FileNotFoundError(f"Step {step} requires missing upstream artifact(s): {', '.join(missing)}")


def run_command(step, command, log):
    start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"\n=== STEP {step}: {' '.join(command)} ===", flush=True)
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    print(output, end="", flush=True)
    entry = {
        "step": step, "command": command, "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - start, 3),
        "returncode": proc.returncode, "log": output,
    }
    log.append(entry)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command, output=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse artifacts when their required contents exist.")
    args = parser.parse_args()
    os.makedirs(DATASET, exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    log = []

    def step(number, command, outputs=(), prerequisites=(), skip=False):
        if number < args.from_step:
            return
        require_artifacts(prerequisites, number)
        if args.skip_existing and outputs and all(exists(path) for path in outputs):
            print(f"=== STEP {number}: skipped; outputs already exist ===", flush=True)
            return
        run_command(number, command, log)

    step(1, [sys.executable, "-m", "data.terrain_pipeline"], ["DATASET/terrain_hp.npz"])
    step(2, [sys.executable, "-m", "data.prepare_hp_era5"], [f"DATASET/{name}" for name in SPLITS])
    if args.from_step <= 3:
        missing = [name for name in SPLITS if not zarr_has_required(f"DATASET/{name}")]
        if missing:
            run_command(3, [sys.executable, "-m", "data.update_splits_temperature", *missing], log)
        else:
            print("=== STEP 3: skipped; all splits already contain temperature ===", flush=True)
    step(4, [sys.executable, "-m", "data.pino_data_pipeline"], ["DATASET/norm_stats.json"], [f"DATASET/{name}" for name in SPLITS])
    step(5, [sys.executable, "-m", "train.train_baselines"], ["DATASET/baseline_results.json"], ["DATASET/norm_stats.json", "DATASET/train_2018_2019.zarr", "DATASET/val_2020.zarr", "DATASET/test_2021_2022.zarr"])
    step(6, [sys.executable, "-m", "train.train_proposed"], ["DATASET/proposed_results.json"], ["DATASET/terrain_hp.npz", "DATASET/norm_stats.json"])
    step(7, [sys.executable, "-m", "analysis.spatial_error_analysis"], ["DATASET/spatial_error_analysis.csv", "DATASET/spatial_error_summary.json"], ["DATASET/baseline_results.json", "DATASET/proposed_results.json"])
    step(8, [sys.executable, "-m", "analysis.build_comparison_table"], ["DATASET/comparison_table.csv"], ["DATASET/baseline_results.json", "DATASET/proposed_results.json"])

    try:
        runtime = subprocess.check_output([
            sys.executable, "-c",
            "import json,torch; print(json.dumps({'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'cuda': torch.version.cuda, 'torch': torch.__version__}))",
        ], text=True).strip()
        runtime = json.loads(runtime)
    except Exception:
        runtime = {"gpu": "unknown", "cuda": "unknown", "torch": "unknown"}
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_hash = "unknown"
    metadata = {"run_start": run_started, "run_end": datetime.now(timezone.utc).isoformat(), "git_commit": git_hash, "gpu": runtime["gpu"], "cuda_version": runtime["cuda"], "pytorch_build": runtime["torch"], "steps": log}
    with open(os.path.join(DATASET, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("\nPipeline completed. Metadata saved to DATASET/run_metadata.json", flush=True)


if __name__ == "__main__":
    main()
