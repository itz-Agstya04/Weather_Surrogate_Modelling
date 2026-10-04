"""
GNN Residual Diagnostic Script.

Answers three concrete questions:
  Q1. Is the GNN residual output magnitude nonzero or near-zero compared to backbone?
  Q2. Does GNN output have any spatial pattern correlated with terrain (steep vs flat)?
  Q3. What is the gradient norm of GNN parameters during a single training step?
      If gradients are nearly zero, the loss surface gives no signal to the GNN.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pino_data_pipeline import PINONormalizer, get_single_step_loader
from train.proposed_model import ProposedModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TERRAIN_PATH = "DATASET/terrain_hp.npz"
NORM_STATS = "DATASET/norm_stats.json"
TEST_ZARR = "DATASET/test_2021_2022.zarr"


def run_gnn_diagnosis():
    print("=" * 65)
    print("  GNN RESIDUAL DIAGNOSTIC")
    print("=" * 65)

    normalizer = PINONormalizer(json.load(open(NORM_STATS)))
    test_loader = get_single_step_loader(TEST_ZARR, normalizer, batch_size=16, shuffle=False)

    terrain = np.load(TERRAIN_PATH)
    steep_mask_flat = np.zeros(15 * 17, dtype=bool)
    steep_mask_flat[terrain["node_flat_indices"]] = True
    steep_mask = torch.from_numpy(steep_mask_flat.reshape(15, 17))

    # --- Test a GNN-sparse-soft model (freshly initialized = worst case) ---
    model_fresh = ProposedModel(
        terrain_path=TERRAIN_PATH,
        physics_mode="soft",
        use_gnn_residual=True,
        gnn_dense=False,
        refinement_steps=1,
    ).to(DEVICE)

    # --- Check if the best proposed checkpoint (GNNOff) exists, compare ---
    ckpt_path = "DATASET/checkpoints/FactorizedFNO_GNNOff_soft_R3.pt"
    model_off = ProposedModel(
        terrain_path=TERRAIN_PATH,
        physics_mode="soft",
        use_gnn_residual=False,
        gnn_dense=False,
        refinement_steps=3,
        refinement_alpha=0.5,
    ).to(DEVICE)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model_off.load_state_dict(state)
        print(f"Loaded trained GNNOff checkpoint: {ckpt_path}")

    # ============================================================
    # DIAGNOSTIC 1: GNN output norm vs backbone norm
    # ============================================================
    print("\n--- Q1: Output norm: backbone vs GNN residual ---")
    backbone_norms = []
    gnn_norms = []
    backbone_norms_per_cell_steep = []
    backbone_norms_per_cell_flat = []
    gnn_norms_per_cell_steep = []
    gnn_norms_per_cell_flat = []

    model_fresh.eval()
    n_batches = 0
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp = inp.to(DEVICE)
            # Manually separate backbone and GNN outputs
            backbone_out = model_fresh.backbone(inp, model_fresh.terrain_features)
            gnn_out = model_fresh.gnn(inp)

            # Whole-tensor L2 norms
            backbone_norms.append(backbone_out.norm(dim=1).mean().item())
            gnn_norms.append(gnn_out.norm(dim=1).mean().item())

            # Per-cell norms (H x W), averaged over batch and channels
            b_cell = backbone_out.norm(dim=1)  # (B, H, W)
            g_cell = gnn_out.norm(dim=1)       # (B, H, W)

            b_steep = b_cell[:, steep_mask].mean().item()
            b_flat  = b_cell[:, ~steep_mask].mean().item()
            g_steep = g_cell[:, steep_mask].mean().item()
            g_flat  = g_cell[:, ~steep_mask].mean().item()

            backbone_norms_per_cell_steep.append(b_steep)
            backbone_norms_per_cell_flat.append(b_flat)
            gnn_norms_per_cell_steep.append(g_steep)
            gnn_norms_per_cell_flat.append(g_flat)

            n_batches += 1
            if n_batches >= 20:
                break

    avg_backbone = np.mean(backbone_norms)
    avg_gnn = np.mean(gnn_norms)
    ratio = avg_gnn / (avg_backbone + 1e-10)

    print(f"  Backbone output L2 norm (avg over test): {avg_backbone:.4f}")
    print(f"  GNN residual L2 norm   (avg over test): {avg_gnn:.6f}")
    print(f"  Ratio GNN/backbone:                     {ratio:.6f}")
    print()
    print(f"  Backbone norm: steep={np.mean(backbone_norms_per_cell_steep):.4f}  flat={np.mean(backbone_norms_per_cell_flat):.4f}")
    print(f"  GNN norm:      steep={np.mean(gnn_norms_per_cell_steep):.6f}  flat={np.mean(gnn_norms_per_cell_flat):.6f}")

    if ratio < 0.01:
        verdict_q1 = "NEAR-ZERO: GNN residual is effectively inactive (<1% of backbone norm)."
    elif ratio < 0.10:
        verdict_q1 = "SMALL: GNN residual exists but is suppressed (~1-10% of backbone)."
    else:
        verdict_q1 = "ACTIVE: GNN residual is contributing meaningfully."
    print(f"\n  [VERDICT Q1] {verdict_q1}")

    # ============================================================
    # DIAGNOSTIC 2: Gradient norm of GNN params on one training batch
    # ============================================================
    print("\n--- Q2: GNN parameter gradient norms (one train step) ---")
    model_fresh.train()
    inp, tgt = next(iter(test_loader))
    inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

    optimizer = torch.optim.AdamW(model_fresh.parameters(), lr=3e-4)
    optimizer.zero_grad()

    pred = model_fresh(inp)
    loss = F.mse_loss(pred, tgt)
    loss.backward()

    gnn_grad_norms = []
    non_gnn_grad_norms = []
    for name, param in model_fresh.named_parameters():
        if param.grad is not None:
            g = param.grad.norm().item()
            if "gnn" in name:
                gnn_grad_norms.append((name, g))
            else:
                non_gnn_grad_norms.append(g)

    print(f"  GNN parameters: {len(gnn_grad_norms)} groups")
    for name, g in gnn_grad_norms:
        print(f"    {name}: grad norm = {g:.6f}")

    avg_backbone_grad = np.mean(non_gnn_grad_norms) if non_gnn_grad_norms else 0.0
    avg_gnn_grad = np.mean([g for _, g in gnn_grad_norms]) if gnn_grad_norms else 0.0
    print(f"\n  Avg backbone param grad norm: {avg_backbone_grad:.6f}")
    print(f"  Avg GNN param grad norm:      {avg_gnn_grad:.6f}")

    if avg_gnn_grad < 1e-6:
        verdict_q2 = "DEAD GRADIENT: GNN is getting no training signal from MSE loss."
    elif avg_gnn_grad < avg_backbone_grad * 0.05:
        verdict_q2 = "SUPPRESSED GRADIENT: GNN gets 20x less signal than backbone — vanishing gradient through sparse scatter."
    else:
        verdict_q2 = "ACTIVE GRADIENT: GNN is receiving meaningful gradient signal."
    print(f"\n  [VERDICT Q2] {verdict_q2}")

    # ============================================================
    # DIAGNOSTIC 3: Does GNN residual have spatial pattern vs terrain?
    # ============================================================
    print("\n--- Q3: GNN residual spatial pattern vs terrain slope ---")
    terrain_slope = terrain["terrain_features"][1].flatten()  # (255,)
    gnn_spatial = []
    model_fresh.eval()
    n_batches = 0
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp = inp.to(DEVICE)
            gnn_out = model_fresh.gnn(inp)
            # Per-cell norm: (B, H, W) -> average over batch
            g_cell = gnn_out.norm(dim=1).mean(dim=0).cpu().numpy().flatten()  # (255,)
            gnn_spatial.append(g_cell)
            n_batches += 1
            if n_batches >= 10:
                break
    gnn_spatial_mean = np.mean(gnn_spatial, axis=0)  # (255,)
    corr = np.corrcoef(terrain_slope, gnn_spatial_mean)[0, 1]
    print(f"  Pearson correlation(terrain slope, GNN output norm): {corr:.4f}")
    if abs(corr) > 0.3:
        verdict_q3 = "SPATIALLY STRUCTURED: GNN output correlates with terrain. Good."
    elif abs(corr) > 0.1:
        verdict_q3 = "WEAKLY STRUCTURED: Some slope-output correlation but noisy."
    else:
        verdict_q3 = "UNSTRUCTURED: GNN output shows no correlation with terrain slope."
    print(f"  [VERDICT Q3] {verdict_q3}")

    # ============================================================
    # DIAGNOSTIC 4: How separable are the steep/flat ERA5 signals?
    # ============================================================
    print("\n--- Q4: Steep vs flat ERA5 temporal variance (is signal separable?) ---")
    import xarray as xr
    ds_test = xr.open_zarr(TEST_ZARR)
    t2m = ds_test["2m_temperature"].values  # (N, H, W)
    steep_2d = steep_mask.numpy()
    temporal_var_steep = t2m[:, steep_2d].var(axis=0).mean()
    temporal_var_flat = t2m[:, ~steep_2d].var(axis=0).mean()
    spatial_var_steep = t2m[:, steep_2d].mean(axis=0).var()  # spatial heterogeneity
    spatial_var_flat = t2m[:, ~steep_2d].mean(axis=0).var()
    print(f"  2m_temperature temporal variance: steep={temporal_var_steep:.2f}  flat={temporal_var_flat:.2f}")
    print(f"  2m_temperature spatial variance (climatological):  steep={spatial_var_steep:.4f}  flat={spatial_var_flat:.4f}")

    steep_vs_flat_var_ratio = float(spatial_var_steep) / (float(spatial_var_flat) + 1e-8)
    print(f"  Spatial heterogeneity ratio (steep/flat): {steep_vs_flat_var_ratio:.3f}")
    if steep_vs_flat_var_ratio > 2.0:
        verdict_q4 = "HETEROGENEOUS: Steep cells have distinct climatological spatial patterns the GNN could learn."
    elif steep_vs_flat_var_ratio > 1.2:
        verdict_q4 = "MILDLY HETEROGENEOUS: Slight spatial differentiation in steep terrain."
    else:
        verdict_q4 = "HOMOGENEOUS at 0.25-deg resolution: steep and flat cells have similar spatial variance — GNN has nothing to anchor on."
    print(f"  [VERDICT Q4] {verdict_q4}")

    print("\n" + "=" * 65)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 65)
    print(f"  Q1 (Output magnitude):  {verdict_q1}")
    print(f"  Q2 (Gradient signal):   {verdict_q2}")
    print(f"  Q3 (Spatial pattern):   {verdict_q3}")
    print(f"  Q4 (Signal separability): {verdict_q4}")
    print()

    results = {
        "backbone_norm": float(avg_backbone),
        "gnn_norm": float(avg_gnn),
        "gnn_to_backbone_ratio": float(ratio),
        "backbone_steep_norm": float(np.mean(backbone_norms_per_cell_steep)),
        "backbone_flat_norm": float(np.mean(backbone_norms_per_cell_flat)),
        "gnn_steep_norm": float(np.mean(gnn_norms_per_cell_steep)),
        "gnn_flat_norm": float(np.mean(gnn_norms_per_cell_flat)),
        "avg_backbone_grad": float(avg_backbone_grad),
        "avg_gnn_grad": float(avg_gnn_grad),
        "slope_gnn_correlation": float(corr),
        "t2m_spatial_var_steep": float(spatial_var_steep),
        "t2m_spatial_var_flat": float(spatial_var_flat),
        "steep_flat_var_ratio": float(steep_vs_flat_var_ratio),
        "verdicts": {
            "Q1": verdict_q1,
            "Q2": verdict_q2,
            "Q3": verdict_q3,
            "Q4": verdict_q4,
        },
    }
    import json as _json
    out_path = "DATASET/gnn_diagnosis.json"
    with open(out_path, "w") as f:
        _json.dump(results, f, indent=2)
    print(f"  [Saved] {out_path}")


if __name__ == "__main__":
    run_gnn_diagnosis()
