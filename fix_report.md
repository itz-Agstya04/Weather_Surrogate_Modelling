# Fix & Evaluation Report: Terrain-Conditioned GNN Residual PINO

## Pre-Training Verification Checks (a to d)

| Check | Description | Target / Criterion | Observed Result | Status |
|---|---|---|---|---|
| **a** | Residual Wind Divergence | Normalized interior divergence of $(u_{\text{res}}, v_{\text{res}})$ on random $\psi < 10^{-3}$ | $8.3699 \times 10^{-14}$ | **PASS** |
| **b** | Spectral Conv Padding | Forward pass on $15 \times 17$ outputs matching shape, no NaN | Identical $(B, 13, 15, 17)$, zero NaN | **PASS** |
| **c** | Boundary Forcing | Outer 2-cell ring equals ERA5 truth exactly after step | Maximum absolute difference $= 0.0$ | **PASS** |
| **d** | Climatology Anomaly ACC | ACC(truth, truth) $= 1.0$; ACC(climatology, truth) $= 0.0$ | Truth: $1.000000$, Climatology: $0.000000$ | **PASS** |

---

## Canonical Multi-Seed Evaluation Summary (Seeds 0, 1, 2)

All models evaluated with exact boundary forcing, diurnal/seasonal smoothed training climatology anomalies, and reflect-padded spectral convolutions. Values are reported as mean $\pm$ standard deviation across 3 seeds.

| Model | Params | Lat-weighted RMSE (6h) | Anomaly ACC (6h) | 72h nRMSE | Peak VRAM (MB) | Latency (ms) | Wind Norm Div | GNN Res Wind Norm Div |
|---|---|---|---|---|---|---|---|---|
| **Persistence** | — | 75.7902 | 0.6949 | 0.4805 | — | — | — | — |
| **Climatology** | — | 110.8151 | 0.0000 | 0.3689 | — | — | — | — |
| **Dense FNO (width=32)** | 398,765 | 59.0121 ± 0.5388 | 0.7157 ± 0.0078 | 0.2514 ± 0.0027 | 35.1 | 3.89 | 1.0584 | — |
| **Dense FNO Matched (width=36)** | 504,553 | 56.0371 ± 1.3202 | 0.7277 ± 0.0089 | 0.2440 ± 0.0054 | 40.0 | 3.94 | 1.0651 | — |
| **PINO (Soft Physics)** | 398,765 | 58.6452 ± 1.1806 | 0.7018 ± 0.0037 | 0.2571 ± 0.0026 | 35.1 | 4.06 | 1.0969 | — |
| **Proposed Model** | 499,144 | **52.0168 ± 0.4873** | **0.7812 ± 0.0017** | **0.2302 ± 0.0036** | 38.1 | 10.16 | 1.0699 | **5.6948e-13** |

---

## Lead-Time Metrics Progression

| Lead Time | Persistence (RMSE / ACC) | Dense FNO Matched (RMSE / ACC) | Proposed Model (RMSE / ACC) |
|---|---|---|---|
| **6h** | 75.7902 / 0.6949 | 56.0371 / 0.7277 | **52.0168 / 0.7812** |
| **24h** | 89.9701 / 0.5401 | 58.2699 / 0.7099 | **53.7712 / 0.7694** |
| **48h** | 99.4120 / 0.4578 | 62.7788 / 0.6654 | **57.7301 / 0.7381** |
| **72h** | 107.0345 / 0.3982 | 67.8920 / 0.6189 | **63.4501 / 0.6965** |

