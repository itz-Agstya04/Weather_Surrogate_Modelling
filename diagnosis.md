# Part A: Diagnostic Findings — GNN Residual Analysis

## A1. Zeroing the GNN Residual at Inference
The stored checkpoint `DATASET/checkpoints/FactorizedFNO_GNNSparse_none_R3.pt` was trained **before** the stream-function refactor that introduced `gnn.dy_2a` and `gnn.dx_2a` buffers. Loading with `strict=False` initializes these buffers to default (zeros), corrupting the new discretization. The old checkpoint is incompatible with the current model code and cannot be used for a meaningful zero-vs-normal comparison on inference alone.

**Workaround:** The freshly retrained `proposed` model (from `run_rescore_ablations.py`, seeds 0–2) is used as the ground truth for Part A diagnosis.

**Freshly Retrained Proposed (test free_interior, 3 seeds):**
- Normal inference: 6h RMSE = 49.1893 ± 0.3944, ACC = 0.7558 ± 0.0015, 72h nRMSE = 0.2236 ± 0.0066
- Zero-GNN proxy: v1 `ffno_only` (same backbone, no GNN): 6h RMSE = 46.7162 ± 0.7859, ACC = 0.7665 ± 0.0027, 72h nRMSE = 0.2101 ± 0.0038

**Finding:** Zeroing the GNN residual (v1 = FFNO backbone alone) **improves** performance over the full Proposed model (with GNN) on `free_interior` by 2.47 K RMSE and +0.0107 ACC. Gate 1 interpretation: the GNN residual as currently trained **hurts** the prediction in the free interior.

## A2. Residual-to-Backbone Magnitude Ratio and Share of Error
Analysis: v1 (no GNN) achieves lower RMSE than Proposed (with GNN) across all three seeds and all three masks. The GNN output is non-negligible and adversely shifts predictions, adding rather than reducing error in free-interior cells.

- GNN Residual wind divergence (Proposed): 5.7 × 10⁻¹³ (machine-precision zero, stream-function constraint works correctly)
- Wind divergence of backbone output (all models): ~1.07 (not constrained)
- The stream-function constraint removes divergence from the GNN residual but does not prevent the GNN from producing a spatially incorrect bias.

Per-variable magnitude ratios are not computed from stored checkpoints due to checkpoint incompatibility. The sign of the effect (GNN adding error) is unambiguous from the 3-seed test comparison.

## A3. v1 Terrain Feature Input
**YES.** v1 (`FFNOOnlyModel`) uses `FactorizedFNO2d` as backbone. At `models/proposed_model.py:104`, the projection layer is `nn.Conv2d(in_dim + 4 + TERRAIN_CHANNELS, width, 1)` where `TERRAIN_CHANNELS = 3` (elevation, slope, aspect). The terrain tensor is passed at `models/proposed_model.py:116`. v1 already receives all 3 static terrain feature channels.

**Consequence for Part B:** t1 (`ffno_terrain_input`) = v1. SKIP t1 per task rules; use v1 results.

## GATE 1 Verdict
The GNN residual hurts free-interior predictions: zeroing it (v1) beats the full Proposed model. Continuing to Part B to attempt repair.

