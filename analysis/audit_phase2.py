import os
import sys
import json
from pathlib import Path
import numpy as np
import xarray as xr
from data.store import open_project_zarr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pino_data_pipeline import (
    SURFACE_VARS,
    TARGET_LEVELS,
    UPPER_AIR_VAR,
    UPPER_AIR_TEMPERATURE_VAR,
    CHANNEL_NAMES,
    PINONormalizer,
    load_full_split_tensor,
)
from models.proposed_model import ProposedModel, LerayProjection2d
from train.train_baselines import (
    CoordinateFNO2d,
    compute_divergence_loss,
    compute_hydrostatic_loss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_phase2():
    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "DATASET" / "norm_stats.json") as f:
        stats = json.load(f)
    normalizer = PINONormalizer(stats)
    means = normalizer.channel_means.flatten().astype(np.float64)
    stds = normalizer.channel_stds.flatten().astype(np.float64)

    train_ds = open_project_zarr(ROOT / "DATASET" / "train_2018_2019.zarr", consolidated=True)
    test_ds = open_project_zarr(ROOT / "DATASET" / "test_2021_2022.zarr", consolidated=True)

    train_tensor = load_full_split_tensor(train_ds).numpy().astype(np.float64)
    test_tensor = load_full_split_tensor(test_ds).numpy().astype(np.float64)
    lats = test_ds.latitude.values
    weights = np.cos(np.deg2rad(lats)).astype(np.float64)[None, None, :, None]
    weights_2d = np.cos(np.deg2rad(lats)).astype(np.float64)[:, None]

    clim_mean = train_tensor.mean(axis=0)

    # Load models
    ckpt_dir = ROOT / "DATASET" / "checkpoints"
    
    # 1. FNO
    fno = CoordinateFNO2d(in_dim=13, out_dim=13, width=32).to(DEVICE)
    fno.load_state_dict(torch.load(ckpt_dir / "fno.pt", map_location=DEVICE, weights_only=True), strict=False)
    fno.eval()

    # 2. PINO
    pino = CoordinateFNO2d(in_dim=13, out_dim=13, width=32).to(DEVICE)
    pino.load_state_dict(torch.load(ckpt_dir / "pino.pt", map_location=DEVICE, weights_only=True), strict=False)
    pino.eval()

    # 3. Proposed (GNN Off, soft)
    proposed_soft = ProposedModel(
        terrain_path=str(ROOT / "DATASET" / "terrain_hp.npz"),
        physics_mode="soft", use_gnn_residual=False, refinement_steps=3, refinement_alpha=0.5
    ).to(DEVICE)
    proposed_soft.load_state_dict(torch.load(ckpt_dir / "FactorizedFNO_GNNOff_soft_R3.pt", map_location=DEVICE, weights_only=True), strict=False)
    proposed_soft.eval()

    # 4. Gated GNN model
    proposed_gnn = ProposedModel(
        terrain_path=str(ROOT / "DATASET" / "terrain_hp.npz"),
        physics_mode="soft", use_gnn_residual=True, gnn_dense=False, refinement_steps=1, refinement_alpha=0.5
    ).to(DEVICE)
    if (ckpt_dir / "GNNSparse_soft_gated.pt").exists():
        proposed_gnn.load_state_dict(torch.load(ckpt_dir / "GNNSparse_soft_gated.pt", map_location=DEVICE, weights_only=True), strict=False)
        proposed_gnn.eval()
        print(f"GNN learnable gate value: {float(proposed_gnn.gnn.gnn_scale.item()):.6f}")

    # 5. Proposed Hard (Leray Projection)
    proposed_hard = ProposedModel(
        terrain_path=str(ROOT / "DATASET" / "terrain_hp.npz"),
        physics_mode="hard", use_gnn_residual=False, refinement_steps=3, refinement_alpha=0.5
    ).to(DEVICE)
    proposed_hard.load_state_dict(torch.load(ckpt_dir / "FactorizedFNO_GNNOff_soft_R3.pt", map_location=DEVICE, weights_only=True), strict=False)
    proposed_hard.eval()

    print("\n=== ITEM 7: ERROR AS A FUNCTION OF LEAD TIME (6h to 72h) ===")
    lead_steps = 12 # 12 * 6h = 72h
    horizons_h = [step * 6 for step in range(1, lead_steps + 1)]

    # To evaluate efficiently and without OOM, sample sequence starts from test set
    # Test set has 2905 samples. Sequences of length 13 (t0 .. t12)
    # Stride of 24 hours (every 4 steps) to cover all seasons efficiently
    stride = 4
    seq_starts = list(range(0, len(test_tensor) - lead_steps, stride))
    print(f"Evaluating {len(seq_starts)} rollout trajectories of length 72h across test set...")

    rollout_nrmse = {
        "Persistence": np.zeros(lead_steps),
        "Climatology": np.zeros(lead_steps),
        "FNO": np.zeros(lead_steps),
        "PINO": np.zeros(lead_steps),
        "Proposed": np.zeros(lead_steps),
    }

    # Extract sequences: shape (N, 13, 13, 15, 17)
    test_seqs = np.array([test_tensor[s : s + lead_steps + 1] for s in seq_starts]) # (N, 13, 13, 15, 17)
    N = len(test_seqs)

    # Initial states
    init_states = test_seqs[:, 0] # (N, 13, 15, 17)
    denom_rollout = weights.sum() * N * 17

    # Persistence predictions: state at t0 repeated for t1..t12
    for step in range(1, lead_steps + 1):
        target_step = test_seqs[:, step]
        err_persist = init_states - target_step
        rmse_p = np.sqrt((err_persist ** 2 * weights).sum(axis=(0, 2, 3)) / denom_rollout)
        rollout_nrmse["Persistence"][step - 1] = float((rmse_p / stds).mean())

        err_clim = clim_mean[None] - target_step
        rmse_c = np.sqrt((err_clim ** 2 * weights).sum(axis=(0, 2, 3)) / denom_rollout)
        rollout_nrmse["Climatology"][step - 1] = float((rmse_c / stds).mean())

    # Autoregressive rollouts for neural models
    def rollout_model(model):
        curr_norm = ((init_states - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)
        nrmse_by_step = []
        with torch.no_grad():
            for step in range(1, lead_steps + 1):
                target_step = test_seqs[:, step]
                next_norm = []
                for b in range(0, N, 32):
                    bx = torch.from_numpy(curr_norm[b:b+32]).to(DEVICE)
                    next_norm.append(model(bx).cpu().numpy())
                curr_norm = np.concatenate(next_norm, axis=0)
                pred_raw = curr_norm.astype(np.float64) * stds[None, :, None, None] + means[None, :, None, None]
                err = pred_raw - target_step
                rmse = np.sqrt((err ** 2 * weights).sum(axis=(0, 2, 3)) / denom_rollout)
                nrmse_by_step.append(float((rmse / stds).mean()))
        return np.array(nrmse_by_step)

    print("Running autoregressive rollouts for FNO...")
    rollout_nrmse["FNO"] = rollout_model(fno)
    print("Running autoregressive rollouts for PINO...")
    rollout_nrmse["PINO"] = rollout_model(pino)
    print("Running autoregressive rollouts for Proposed...")
    rollout_nrmse["Proposed"] = rollout_model(proposed_soft)

    print(f"\n{'Lead Time':<10} | {'Persistence':<12} | {'Climatology':<12} | {'FNO':<12} | {'PINO':<12} | {'Proposed':<12}")
    print("-" * 75)
    for i, h in enumerate(horizons_h):
        print(f"{h:2d}h{'':<7} | {rollout_nrmse['Persistence'][i]:<12.4f} | {rollout_nrmse['Climatology'][i]:<12.4f} | {rollout_nrmse['FNO'][i]:<12.4f} | {rollout_nrmse['PINO'][i]:<12.4f} | {rollout_nrmse['Proposed'][i]:<12.4f}")

    # Plot lead-time error curve
    plt.figure(figsize=(10, 6))
    for name, curve in rollout_nrmse.items():
        style = "--" if "Clim" in name or "Persist" in name else "-"
        plt.plot(horizons_h, curve, label=name, linestyle=style, marker="o" if style=="-" else "x")
    plt.xlabel("Lead Time (Hours)")
    plt.ylabel("Latitude-Weighted Aggregate nRMSE")
    plt.title("Forecast Error vs Lead Time (6h to 72h)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = ROOT / "results" / "lead_time_error.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot to: {plot_path}")

    print("\n=== ITEM 8: TERRAIN-GATED GNN CONTRIBUTION (STEEP VS FLAT) ===")
    terrain_data = np.load(ROOT / "DATASET" / "terrain_hp.npz")
    steep_indices = terrain_data["node_flat_indices"] # 1D flat indices in 15*17=255
    total_nodes = 15 * 17
    fraction_gated = len(steep_indices) / total_nodes
    print(f"Total grid nodes: {total_nodes}")
    print(f"Steep (gated) nodes: {len(steep_indices)}")
    print(f"Fraction of points gated: {fraction_gated:.4f} ({len(steep_indices)}/{total_nodes})")

    # Mask for steep vs flat
    mask_steep_2d = np.zeros((15, 17), dtype=bool)
    mask_steep_2d.ravel()[steep_indices] = True
    mask_flat_2d = ~mask_steep_2d

    # Evaluate single-step test predictions for:
    # 1. Proposed WITHOUT GNN (FactorizedFNO_GNNOff_soft_R3)
    # 2. Proposed WITH GNN (GNNSparse_soft_gated)
    test_in = test_tensor[:-1]
    test_tgt = test_tensor[1:]
    inp_norm = ((test_in - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)

    def get_preds(m):
        preds = []
        with torch.no_grad():
            for b in range(0, len(inp_norm), 32):
                preds.append(m(torch.from_numpy(inp_norm[b:b+32]).to(DEVICE)).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float64) * stds[None, :, None, None] + means[None, :, None, None]

    pred_no_gnn = get_preds(proposed_soft)
    pred_with_gnn = get_preds(proposed_gnn)

    def calc_masked_rmse(preds, mask):
        # mask: (15, 17)
        w = (weights_2d * mask)[None, None, :, :]
        d = w.sum() * len(preds)
        err = preds - test_tgt
        rmse = np.sqrt((err ** 2 * w).sum(axis=(0, 2, 3)) / d)
        nrmse = rmse / stds
        return float(nrmse.mean())

    nrmse_no_gnn_steep = calc_masked_rmse(pred_no_gnn, mask_steep_2d)
    nrmse_no_gnn_flat = calc_masked_rmse(pred_no_gnn, mask_flat_2d)
    nrmse_with_gnn_steep = calc_masked_rmse(pred_with_gnn, mask_steep_2d)
    nrmse_with_gnn_flat = calc_masked_rmse(pred_with_gnn, mask_flat_2d)

    print(f"GNN Residual Status | Steep Points nRMSE | Flat Points nRMSE | Steep/Flat Ratio")
    print("-" * 75)
    print(f"WITHOUT GNN (Off)   | {nrmse_no_gnn_steep:<18.4f} | {nrmse_no_gnn_flat:<17.4f} | {nrmse_no_gnn_steep/nrmse_no_gnn_flat:<16.4f}")
    print(f"WITH GNN (Gated)    | {nrmse_with_gnn_steep:<18.4f} | {nrmse_with_gnn_flat:<17.4f} | {nrmse_with_gnn_steep/nrmse_with_gnn_flat:<16.4f}")

    print("\n=== ITEM 9: PHYSICS RESIDUALS ON TEST DATA ===")
    # Evaluate divergence norm and hydrostatic residual
    # Compute on normalized predictions converted through compute_divergence_loss and compute_hydrostatic_loss
    def eval_physics(preds_raw):
        # preds_raw: (N, 13, 15, 17)
        # Convert to tensor normalized
        norm_t = torch.from_numpy(((preds_raw - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)).to(DEVICE)
        
        # Batch evaluation
        div_losses = []
        hydro_losses = []
        div_norms_phys = [] # physical divergence ||du/dx + dv/dy|| in s^-1
        
        # Grid parameters for physical divergence
        height, width = 15, 17
        lat = torch.linspace(33.5, 30.0, height, device=DEVICE, dtype=torch.float32)
        dlat = torch.abs(lat[1] - lat[0]) * (np.pi / 180.0)
        dlon = (4.0 / (width - 1)) * (np.pi / 180.0)
        earth_radius = 6_371_000.0
        dy = earth_radius * dlat
        dx = earth_radius * torch.cos(lat * (np.pi / 180.0)) * dlon

        with torch.no_grad():
            for b in range(0, len(norm_t), 64):
                bx = norm_t[b:b+64]
                d_loss = compute_divergence_loss(bx, normalizer)
                h_loss = compute_hydrostatic_loss(bx, normalizer)
                div_losses.append(d_loss.item())
                hydro_losses.append(h_loss.item())

                # Physical divergence in s^-1
                u = bx[:, 1] * float(normalizer.channel_stds[1].item()) + float(normalizer.channel_means[1].item())
                v = bx[:, 2] * float(normalizer.channel_stds[2].item()) + float(normalizer.channel_means[2].item())
                du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dx.view(1, -1, 1))
                dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2.0 * dy)
                div_phys = du_dx[:, 1:-1, :] + dv_dy[:, :, 1:-1]
                div_norms_phys.append(float(torch.sqrt((div_phys ** 2).mean()).item()))

        return np.mean(div_norms_phys), np.mean(div_losses), np.mean(hydro_losses)

    print("Evaluating physics residuals across models...")
    models_to_test = {
        "Ground Truth (ERA5)": test_tgt,
        "FNO": get_preds(fno),
        "PINO": get_preds(pino),
        "Proposed (Soft)": pred_no_gnn,
        "Proposed (Hard / Leray)": get_preds(proposed_hard),
    }

    print(f"\n{'Model':<25} | {'Divergence Norm (s^-1)':<24} | {'Divergence Loss':<18} | {'Hydrostatic Residual'}")
    print("-" * 85)
    for mname, mpreds in models_to_test.items():
        div_norm, div_loss, hydro_loss = eval_physics(mpreds)
        print(f"{mname:<25} | {div_norm:<24.6e} | {div_loss:<18.6f} | {hydro_loss:<20.6e}")

if __name__ == "__main__":
    run_phase2()

