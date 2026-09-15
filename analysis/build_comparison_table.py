"""Combine baseline and proposed-model JSON results for slide tables."""

import csv
import json
import os


def build_comparison_table(baseline_path="DATASET/baseline_results.json",
                           proposed_path="DATASET/proposed_results.json",
                           output_path="DATASET/comparison_table.csv"):
    with open(baseline_path, "r") as handle:
        baseline = json.load(handle)
    with open(proposed_path, "r") as handle:
        proposed = json.load(handle)

    rows = []
    for name, metrics in baseline.items():
        if "aggregate_nrmse" not in metrics:
            continue
        rows.append({
            "model": name,
            "aggregate_nrmse_1step": metrics["aggregate_nrmse"],
            "aggregate_acc_1step": metrics["aggregate_acc"],
            "peak_gpu_mem_mb": metrics.get("peak_gpu_mem_mb", ""),
            "inference_latency_ms": metrics.get("inference_latency_ms", ""),
            "parameter_count": metrics.get("parameter_count", ""),
        })

    for name, result in proposed.items():
        one_step = result["one_step"]
        rollout = result.get("rollout", {})
        row = {
            "model": name,
            "aggregate_nrmse_1step": one_step["aggregate_nrmse"],
            "aggregate_acc_1step": one_step["aggregate_acc"],
            "peak_gpu_mem_mb": one_step.get("peak_gpu_mem_mb", ""),
            "inference_latency_ms": one_step.get("inference_latency_ms", ""),
            "parameter_count": result.get("parameter_count", ""),
        }
        for horizon in ("6h", "72h"):
            if horizon in rollout:
                row[f"aggregate_nrmse_{horizon}"] = rollout[horizon]["FNO"]["aggregate_nrmse"]
                row[f"aggregate_acc_{horizon}"] = rollout[horizon]["FNO"]["aggregate_acc"]
        rows.append(row)

    columns = ["model", "aggregate_nrmse_1step", "aggregate_acc_1step",
               "aggregate_nrmse_6h", "aggregate_acc_6h",
               "aggregate_nrmse_72h", "aggregate_acc_72h",
               "peak_gpu_mem_mb", "inference_latency_ms", "parameter_count"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("\n--- Step 3: Comparison Table ---", flush=True)
    for row in rows:
        print(" | ".join(str(row.get(column, "")) for column in columns), flush=True)
    return rows


if __name__ == "__main__":
    build_comparison_table()
