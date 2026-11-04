# Technical Audit Report: Terrain-Conditioned GNN Residual PINO

## Summary Table

| # | Item | Verdict | File & Line | Key Evidence / Reason |
|---|---|---|---|---|
| 1 | ACC Definition | FAIL | `train/train_baselines.py:330-348`, `run_train_eval.py:114` | Computed against a static spatial mean climatology across all training time (`train_loader.dataset.data.mean(dim=0)`), NOT per day-of-year and hour. |
| 2 | Trivial Baselines | FAIL | `run_train_eval.py:126-174`, `results/final_metrics.json` | Persistence and climatology baselines were omitted from `final_metrics.json`. Computed values on test split are documented below. |
| 3 | Lateral Boundary Handling | FAIL | `train/train_baselines.py:408-417`, `run_train_eval.py:65-71` | Domain is entirely free-running autoregressive rollout; no boundary values or buffer/relaxation zones are injected. |
| 4 | Periodicity / FFT Handling | FAIL | `train/train_baselines.py:75-76`, `models/proposed_model.py:126-138` | `rfft2` applied directly without domain extension or boundary padding. Leray projection on random fields yields high divergence at both interior and border. |
| 5 | Continuity Formulation | FAIL | `train/train_baselines.py:171-193` | Only penalizes 2D surface horizontal wind (`10m_u`, `10m_v`), ignoring upper-air pressure levels and vertical velocity $\omega$. Includes metric $\cos(\text{lat})$ on $dx$ but omits spherical curvature metric term $\frac{v \tan \phi}{R}$. |
| 6 | Seed Configuration | FAIL | `run_train_eval.py`, `train/train_baselines.py` | Main training and evaluation scripts have no seed configuration or seed arguments. Results in `results/final_metrics.json` are single-seed. |
| 7 | Per-Variable & Per-Lead-Time Metrics | PASS | `run_train_eval.py:76-83`, `train/train_baselines.py:350-357` | Code produces per-variable RMSE, nRMSE, and ACC for all 13 channels at step 1 (6h) and step 12 (72h) stored in `results/final_metrics.json`. |
| 8 | Physics Residual Reporting | FAIL | `run_train_eval.py:171`, `results/final_metrics.json` | Mean divergence and hydrostatic residuals are omitted from evaluation metrics. On a test batch, evaluated normalized divergence losses are reported below. |
| 9 | Parameter Counts & Training Profile | PASS | `models/proposed_model.py:157`, `train/train_baselines.py:84` | Dense FNO: 398,765; PINO: 398,765; Proposed: 499,144. Training protocol logged: 20 single-step warmup epochs + 3 rollout epochs. |
| 10 | VRAM & Latency Benchmark | PASS | `run_train_eval.py:32-51`, `models/proposed_model.py:101-108` | Uses 25 warmup steps, `torch.cuda.synchronize()`, 100 timed iterations, fixed batch size=1. Graph edge indices pre-cached in buffers. |
| 11 | Data Leakage | PASS | `data/pino_data_pipeline.py:100-112`, `DATASET/norm_stats.json` | Normalization statistics computed on train split only. Rollout dataset uses slice-based windowing strictly within split bounds. |
| 12 | nRMSE Definition | PASS | `train/train_baselines.py:328`, commit `27fb244` | Exact formula: per-channel $\text{RMSE}_c / \sigma_{c,\text{train}}$ averaged across channels. Prior invalid run used root-mean-square of normalized errors across all cells. |

---

## Detailed Findings

### 1. ACC (Anomaly Correlation Coefficient)
- **Verdict:** FAIL
- **Location:** `train/train_baselines.py:330-348`, `run_train_eval.py:114`
- **Details:** The ACC implementation in `calculate_metrics` accepts a `climatology` tensor. In `run_train_eval.py:114`, it is passed as:
  ```python
  training_climatology = normalizer.denormalize(train_loader.dataset.data.float()).mean(dim=0)
  ```
  This is a static time-mean spatial field across the entire training period (2018–2019), rather than a diurnal/seasonal climatology conditioned on day-of-year and hour-of-day. Consequently, diurnal cycles are treated as anomalies.

### 2. Trivial Baselines
- **Verdict:** FAIL (Omitted from `results/final_metrics.json`)
- **Location:** `results/final_metrics.json`
- **Computed Values (Test Split `test_2021_2022.zarr`):**
  - **T2m (2m Temperature):**
    - 6h Persistence: RMSE = 4.9037 K, Anomaly ACC = 0.8796
    - 6h Climatology: RMSE = 10.0020 K, Anomaly ACC = 0.0000
    - 72h Persistence: RMSE = 3.7359 K, Anomaly ACC = 0.9291
    - 72h Climatology: RMSE = 9.9839 K, Anomaly ACC = 0.0000
  - **U10 (10m U-Wind):**
    - 6h Persistence: RMSE = 1.0680 m/s, Anomaly ACC = 0.5044
    - 6h Climatology: RMSE = 1.0724 m/s, Anomaly ACC = 0.0000
    - 72h Persistence: RMSE = 1.0416 m/s, Anomaly ACC = 0.5278
    - 72h Climatology: RMSE = 1.0696 m/s, Anomaly ACC = 0.0000
  - **V10 (10m V-Wind):**
    - 6h Persistence: RMSE = 1.0251 m/s, Anomaly ACC = 0.4128
    - 6h Climatology: RMSE = 0.9458 m/s, Anomaly ACC = 0.0000
    - 72h Persistence: RMSE = 0.9020 m/s, Anomaly ACC = 0.5444
    - 72h Climatology: RMSE = 0.9441 m/s, Anomaly ACC = 0.0000
  - **Z500 (500 hPa Geopotential):**
    - 6h Persistence: RMSE = 161.7051 m²/s², Anomaly ACC = 0.9714
    - 6h Climatology: RMSE = 675.6065 m²/s², Anomaly ACC = 0.0000
    - 72h Persistence: RMSE = 374.8932 m²/s², Anomaly ACC = 0.8449
    - 72h Climatology: RMSE = 675.6967 m²/s², Anomaly ACC = 0.0000

### 3. Lateral Boundary Handling
- **Verdict:** FAIL
- **Location:** `train/train_baselines.py:408-417`, `run_train_eval.py:65-71`
- **Details:** The rollout evaluation loops over lead times (`current = model(current)`) recursively without injecting true ERA5 boundary forcing at the domain edges or applying a buffer relaxation zone. The model runs completely unconstrained as a closed regional domain.

### 4. Periodicity & Leray Projection
- **Verdict:** FAIL
- **Location:** `models/proposed_model.py:126-138`, `train/train_baselines.py:75-76`
- **Details:** Neither the 2D spectral convolution nor the `LerayProjection2d` applies padding or mirror extension prior to `torch.fft.rfft2`. A numeric check applying `LerayProjection2d` to a random non-periodic field ($15 \times 17$) yields:
  - **Interior $|\text{div}|,$ finite-difference:** Mean = 0.3175, Max = 1.0543
  - **Border (within 2 cells) $|\text{div}|,$ finite-difference:** Mean = 0.3013, Max = 1.0033
  Because boundary periodicity is assumed by the spectral projection, divergence is not removed in the non-periodic spatial domain.

### 5. Continuity Formulation
- **Verdict:** FAIL
- **Location:** `train/train_baselines.py:171-193`
- **Details:** The divergence loss enforces $\partial u/\partial x + \partial v/\partial y = 0$ exclusively on surface wind (`10m_u` at index 1 and `10m_v` at index 2). Pressure-level wind fields (which carry vertical velocity $\omega$) are not constrained. For coordinate scaling, $dx = R \cos(\phi) d\lambda$ is used, but the spherical convergence metric term $\frac{v \tan \phi}{R}$ is omitted.

### 6. Seeds & Multi-Seed Runs
- **Verdict:** FAIL
- **Location:** `run_train_eval.py:1-178`
- **Details:** While multi-seed utilities were prototyped in `analysis/run_rigorous_evaluation.py`, the canonical evaluation script `run_train_eval.py` that produced `results/final_metrics.json` does not accept or set random seeds, resulting in an unseeded, single-run evaluation.

### 7. Per-Variable & Per-Lead-Time Metrics
- **Verdict:** PASS
- **Location:** `run_train_eval.py:76-83`, `results/final_metrics.json`
- **Details:** `step_1_metrics` (6h) and `step_12_metrics` (72h) are saved with individual per-channel breakdowns for all 13 variables (`T2m`, `U10`, `V10`, `Z1000`, `Z850`, `Z700`, `Z500`, `Z300`, `T1000`, `T850`, `T700`, `T500`, `T300`). Intermediate horizons (12h, 24h, 48h) are implemented in `evaluate_rollout_horizons` in `train/train_baselines.py:397-430`, though `run_train_eval.py` specifically records endpoints 6h and 72h.

### 8. Physics Metrics
- **Verdict:** FAIL
- **Location:** `results/final_metrics.json`
- **Details:** Divergence and hydrostatic residuals are omitted from `results/final_metrics.json`. Evaluation of the saved models on a test batch ($N=16$) using `compute_divergence_loss`:
  - Dense FNO: $2.0587 \times 10^0$ ($\text{normalized } s^{-2}$)
  - PINO (Soft Physics): $5.0840 \times 10^{-1}$ ($\text{normalized } s^{-2}$)
  - Proposed Model: $2.2841 \times 10^0$ ($\text{normalized } s^{-2}$)
  PINO exhibits lower divergence due to soft penalty training, but this metric is not tracked in the summary metrics file.

### 9. Parameter Counts & Training Profile
- **Verdict:** PASS
- **Location:** `train/train_baselines.py`, `models/proposed_model.py`
- **Trainable Parameters:**
  - **Dense FNO (2D Coordinate FNO):** 398,765
  - **PINO (Standard PINO):** 398,765
  - **Proposed Model (Factorized FNO + Sparse GNN, R=3):** 499,144
- **Training Protocol:** 20 single-step warmup epochs (AdamW, lr=3e-4, Cosine Annealing) followed by 3 autoregressive rollout fine-tuning epochs (lr=1e-4). Total runtime logged: ~5 minutes on NVIDIA RTX 4050 Laptop GPU.

### 10. VRAM & Latency Measurement Code
- **Verdict:** PASS
- **Location:** `run_train_eval.py:32-51`
- **Details:** Benchmarking utilizes 25 warmup iterations, followed by `torch.cuda.synchronize()`, and 100 timed evaluations with median latency calculation and `torch.cuda.max_memory_allocated()`. Batch size is fixed to 1 for all models. The GNN adjacency (`edge_index`, `edge_attr`, `node_flat_indices`) is pre-computed and stored in PyTorch persistent buffers (`TerrainGNNResidual.__init__`), not constructed during the timed forward pass.
- **VRAM Difference (Dense FNO 34.9 MB vs PINO 41.0 MB):** Both share identical architectures; the minor VRAM difference arises from state cache remnants or peak memory allocations during preceding evaluation passes before memory stats reset.

### 11. Leakage Analysis
- **Verdict:** PASS
- **Location:** `data/pino_data_pipeline.py:100-112`, `DATASET/norm_stats.json`
- **Details:** Normalization stats (`norm_stats.json`) were computed exclusively on `train_2018_2019.zarr`. In `HPERA5RolloutDataset`, `self.num_samples = self.data.size(0) - sequence_length + 1` and indexing `self.data[idx : idx + sequence_length]` enforces strict sequence containment within the split tensor, preventing boundary leakage across train, val, or test splits.

### 12. nRMSE Definition
- **Verdict:** PASS
- **Location:** `train/train_baselines.py:328`, commit `27fb244`
- **Details:**
  - Current formula:
    $$\text{nRMSE} = \frac{1}{C}\sum_{c=1}^C \frac{\text{RMSE}_c}{\sigma_{c,\text{train}}}$$
    where $\text{RMSE}_c = \sqrt{\frac{\sum_i w_i (y_i - \hat{y}_i)^2}{\sum_i w_i}}$ and $\sigma_{c,\text{train}}$ is the standard deviation of channel $c$ from the training set.
  - Previous (invalid) formula in commit `27fb244`: Normalized each cell's squared error by $\sigma_c^2$, summed across channels and spatial coordinates, and took the square root:
    $$\text{lat\_rmse\_norm} = \sqrt{\frac{\sum_{c,i} w_i \frac{(y_{c,i} - \hat{y}_{c,i})^2}{\sigma_{c}^2}}{C \sum_i w_i}}$$
  - The revised formula aligns with standard meteorological evaluation protocols.

