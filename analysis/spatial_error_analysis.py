"""
Spatial Error Analysis: Himachal Pradesh Microclimates.

Computes per-grid-cell error across Himachal Pradesh and overlays it against
`steep_mask` from `DATASET/terrain_hp.npz`.

Evaluates whether prediction error concentrates in complex terrain (steep cells)
versus flat valley cells for:
- Persistence Baseline
- Baseline 1 (FNO 2D + coordinates)
- Baseline 2 (Standard PINO)
- Best Proposed Model (Factorized FNO + GNN + Leray/Hard/Soft)

Outputs:
- DATASET/spatial_error_analysis.csv
- DATASET/spatial_error_summary.json
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pino_data_pipeline import PINONormalizer, get_single_step_loader, CHANNEL_NAMES
from train.train_baselines import CoordinateFNO2d
from train.proposed_model import ProposedModel


@torch.no_grad()
def compute_spatial_error(model: nn.Module, test_loader, normalizer: PINONormalizer, device: torch.device):
    """
    Computes per-cell RMSE of shape (15, 17) in normalized units.
    """
    if model is not None:
        model.eval()
        
    all_sq_err = []
    for inp, tgt in test_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        if model is None:
            # Persistence: prediction is input state at t
            pred = inp
        else:
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(inp)
                
        # Compute squared error in normalized units: (B, C=13, H=15, W=17)
        sq_err = (pred - tgt) ** 2
        # Mean across channels: (B, H, W)
        sq_err_cell = sq_err.mean(dim=1)
        all_sq_err.append(sq_err_cell.cpu())
        
    all_sq_err = torch.cat(all_sq_err, dim=0)  # (N, H, W)
    # Mean MSE per cell over all test time steps
    mean_cell_mse = torch.mean(all_sq_err, dim=0)  # (H, W)
    cell_rmse = torch.sqrt(mean_cell_mse).numpy()  # (15, 17)
    return cell_rmse


def run_spatial_error_analysis(
    baseline_path: str = "DATASET/baseline_results.json",
    proposed_path: str = "DATASET/proposed_results.json",
    terrain_path: str = "DATASET/terrain_hp.npz",
    test_zarr: str = "DATASET/test_2021_2022.zarr",
    stats_file: str = "DATASET/norm_stats.json",
    output_csv: str = "DATASET/spatial_error_analysis.csv",
    output_json: str = "DATASET/spatial_error_summary.json",
    models_dict: dict = None,
):
    print("\n=======================================================", flush=True)
    print("      PHASE 5: SPATIAL ERROR ANALYSIS (STEEP VS FLAT)  ", flush=True)
    print("=======================================================", flush=True)

    # 1. Load terrain artifact and steep mask
    terrain = np.load(terrain_path)
    steep_mask = terrain["steep_mask"]  # (15, 17) bool
    flat_mask = ~steep_mask
    n_steep = int(np.sum(steep_mask))
    n_flat = int(np.sum(flat_mask))
    print(f"Terrain grid: {steep_mask.shape[0]}x{steep_mask.shape[1]} | Steep cells: {n_steep} | Flat cells: {n_flat}", flush=True)

    # 2. Data pipeline
    with open(stats_file, "r") as f:
        normalizer = PINONormalizer(json.load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_single_step_loader(test_zarr, normalizer, batch_size=32, shuffle=False)

    results = {}
    rows = []

    # If explicit model instances provided, evaluate them
    if models_dict is None:
        models_dict = {"Persistence": None}

        fno_path = "DATASET/checkpoints/fno.pt"
        if os.path.exists(fno_path):
            fno = CoordinateFNO2d(in_dim=13, out_dim=13, width=32).to(device)
            fno.load_state_dict(torch.load(fno_path, map_location=device, weights_only=True))
            models_dict["Baseline 1 (FNO)"] = fno

        pino_path = "DATASET/checkpoints/pino.pt"
        if os.path.exists(pino_path):
            pino = CoordinateFNO2d(in_dim=13, out_dim=13, width=32).to(device)
            pino.load_state_dict(torch.load(pino_path, map_location=device, weights_only=True))
            models_dict["Baseline 2 (PINO)"] = pino

        proposed_ckpt = "DATASET/checkpoints/FactorizedFNO_GNNOff_soft_R3.pt"
        if os.path.exists(proposed_ckpt):
            proposed = ProposedModel(
                terrain_path=terrain_path,
                physics_mode="soft",
                use_gnn_residual=False,
                gnn_dense=False,
                refinement_steps=3,
                refinement_alpha=0.5,
            ).to(device)
            proposed.load_state_dict(torch.load(proposed_ckpt, map_location=device, weights_only=True))
            models_dict["Proposed (Factorized FNO + Soft Phys R3)"] = proposed

    for model_name, model_inst in models_dict.items():
        cell_rmse = compute_spatial_error(model_inst, test_loader, normalizer, device)
        
        steep_rmse = float(np.mean(cell_rmse[steep_mask]))
        flat_rmse = float(np.mean(cell_rmse[flat_mask]))
        overall_rmse = float(np.mean(cell_rmse))
        ratio = float(steep_rmse / (flat_rmse + 1e-8))

        results[model_name] = {
            "overall_rmse": overall_rmse,
            "steep_cells_rmse": steep_rmse,
            "flat_cells_rmse": flat_rmse,
            "steep_to_flat_error_ratio": ratio,
            "per_cell_rmse": cell_rmse.tolist(),
        }

        rows.append({
            "model": model_name,
            "overall_rmse": f"{overall_rmse:.5f}",
            "steep_cells_rmse": f"{steep_rmse:.5f}",
            "flat_cells_rmse": f"{flat_rmse:.5f}",
            "steep_to_flat_error_ratio": f"{ratio:.3f}",
        })

    # Save CSV
    columns = ["model", "overall_rmse", "steep_cells_rmse", "flat_cells_rmse", "steep_to_flat_error_ratio"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON summary
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Model':<35s} | {'Overall RMSE':>12s} | {'Steep RMSE':>10s} | {'Flat RMSE':>10s} | {'Steep/Flat':>10s}")
    print("-" * 87)
    for r in rows:
        print(f"{r['model']:<35s} | {r['overall_rmse']:>12s} | {r['steep_cells_rmse']:>10s} | {r['flat_cells_rmse']:>10s} | {r['steep_to_flat_error_ratio']:>10s}")
    print("=======================================================", flush=True)
    print(f"[SUCCESS] Spatial error analysis saved to {output_csv} and {output_json}!\n", flush=True)
    return results


if __name__ == "__main__":
    run_spatial_error_analysis()
