# Final Evaluation Results: Test Splits across Evaluation Masks

Selected Final Model: **v1_ffno_only** (Factorized FNO with terrain input channels, width=35, R=3 iterative refinement).

**Gate 2 Outcome:** GATE 2 STOP TRIGGERED: No repair variant (t2, t3, t4) beat v1 (ffno_only) on VAL free_interior on both 6h RMSE and ACC. On TEST free_interior, v1 achieves 6h RMSE 46.7162 K and ACC 0.7665, outperforming t2 (50.8035 K, 0.7476), t3 (50.2322 K, 0.7488), and t4 (49.4768 K, 0.7507). Final selected model is v1_ffno_only (equivalent to t1_ffno_terrain_input).


## Mask 1: Free Interior (Cells >= 5 from domain boundary; Primary Benchmark)

| Model / Architecture | Params | Latency (ms) | Peak VRAM (MB) | 6h Lat-Weighted RMSE (K) | 6h Anomaly ACC | 72h Rollout nRMSE |
|---|---|---|---|---|---|---|
| Persistence | — | — | — | 68.1392 | 0.6824 | 0.4148 |
| Climatology | — | — | — | 105.1775 | 0.0000 | 0.3170 |
| Dense 2D FNO (w=32) | 398,765 | 3.99 | 35.1 | 56.3940 ± 0.8269 | 0.6890 ± 0.0046 | 0.2242 ± 0.0030 |
| Dense 2D FNO Matched (w=36) | 504,553 | 4.22 | 40.0 | 54.6659 ± 1.0986 | 0.6955 ± 0.0083 | 0.2208 ± 0.0050 |
| PINO (Soft Physics) | 398,765 | 4.01 | 35.1 | 56.4597 ± 1.1251 | 0.6751 ± 0.0050 | 0.2326 ± 0.0031 |
| Proposed (FFNO + Sparse GNN) | 499,144 | 11.23 | 38.1 | 49.1893 ± 0.3944 | 0.7558 ± 0.0015 | 0.2236 ± 0.0066 |
| **v1: FFNO Only (Selected Final)** | 504,894 | 8.32 | 39.8 | **46.7162 ± 0.7859** | **0.7665 ± 0.0027** | **0.2101 ± 0.0038** |
| v2: FFNO + Dense GNN | 499,144 | 11.22 | 39.1 | 48.0242 ± 0.4369 | 0.7597 ± 0.0012 | 0.2169 ± 0.0011 |
| v3: FFNO + Random Sparse GNN | 499,144 | 11.13 | 38.0 | 49.5440 ± 0.7649 | 0.7518 ± 0.0053 | 0.2180 ± 0.0036 |
| v4: FFNO + Unconstrained GNN | 499,144 | 11.03 | 38.2 | 48.7338 ± 0.3839 | 0.7545 ± 0.0021 | 0.2212 ± 0.0078 |
| **t1: FFNO + Terrain Input (identical to v1)** | 504,894 | 8.32 | 39.8 | **46.7162 ± 0.7859** | **0.7665 ± 0.0027** | **0.2101 ± 0.0038** |
| t2: Proposed Zero-Init | 499,144 | 12.73 | 38.1 | 50.8035 ± 0.8187 | 0.7476 ± 0.0008 | 0.2206 ± 0.0039 |
| t3: Proposed Zero-Init + L2 | 499,144 | 26.99 | 38.1 | 50.2322 ± 0.5208 | 0.7488 ± 0.0079 | 0.2218 ± 0.0053 |
| t4: FFNO Terrain + Zero-Init GNN | 499,144 | 17.24 | 38.1 | 49.4768 ± 0.2269 | 0.7507 ± 0.0010 | 0.2136 ± 0.0032 |


## Mask 2: Outside Ring (Cells >= 2 from domain boundary; Interior + Transition Buffer)

| Model / Architecture | Params | Latency (ms) | Peak VRAM (MB) | 6h Lat-Weighted RMSE (K) | 6h Anomaly ACC | 72h Rollout nRMSE |
|---|---|---|---|---|---|---|
| Persistence | — | — | — | 73.8022 | 0.6901 | 0.4606 |
| Climatology | — | — | — | 108.5257 | 0.0000 | 0.3526 |
| Dense 2D FNO (w=32) | 398,765 | 3.99 | 35.1 | 58.3540 ± 0.3786 | 0.7109 ± 0.0064 | 0.2432 ± 0.0022 |
| Dense 2D FNO Matched (w=36) | 504,553 | 4.22 | 40.0 | 55.6962 ± 1.2810 | 0.7209 ± 0.0093 | 0.2368 ± 0.0055 |
| PINO (Soft Physics) | 398,765 | 4.01 | 35.1 | 58.0446 ± 1.2918 | 0.6948 ± 0.0038 | 0.2499 ± 0.0033 |
| Proposed (FFNO + Sparse GNN) | 499,144 | 11.23 | 38.1 | 51.1681 ± 0.1856 | 0.7778 ± 0.0021 | 0.2284 ± 0.0044 |
| **v1: FFNO Only (Selected Final)** | 504,894 | 8.32 | 39.8 | **48.4326 ± 0.6399** | **0.7907 ± 0.0029** | **0.2172 ± 0.0026** |
| v2: FFNO + Dense GNN | 499,144 | 11.22 | 39.1 | 49.3112 ± 0.7043 | 0.7870 ± 0.0015 | 0.2214 ± 0.0016 |
| v3: FFNO + Random Sparse GNN | 499,144 | 11.13 | 38.0 | 51.2926 ± 0.8214 | 0.7752 ± 0.0040 | 0.2251 ± 0.0027 |
| v4: FFNO + Unconstrained GNN | 499,144 | 11.03 | 38.2 | 50.4903 ± 0.3301 | 0.7780 ± 0.0018 | 0.2264 ± 0.0045 |
| **t1: FFNO + Terrain Input (identical to v1)** | 504,894 | 8.32 | 39.8 | **48.4326 ± 0.6399** | **0.7907 ± 0.0029** | **0.2172 ± 0.0026** |
| t2: Proposed Zero-Init | 499,144 | 12.73 | 38.1 | 52.1444 ± 0.7586 | 0.7738 ± 0.0010 | 0.2270 ± 0.0024 |
| t3: Proposed Zero-Init + L2 | 499,144 | 26.99 | 38.1 | 51.6282 ± 0.4008 | 0.7734 ± 0.0062 | 0.2279 ± 0.0043 |
| t4: FFNO Terrain + Zero-Init GNN | 499,144 | 17.24 | 38.1 | 50.9350 ± 0.4159 | 0.7751 ± 0.0017 | 0.2221 ± 0.0023 |


## Mask 3: Full Regional Domain (15x17 grid; Includes ERA5 Boundary Forcing Ring)

| Model / Architecture | Params | Latency (ms) | Peak VRAM (MB) | 6h Lat-Weighted RMSE (K) | 6h Anomaly ACC | 72h Rollout nRMSE |
|---|---|---|---|---|---|---|
| Persistence | — | — | — | 75.7902 | 0.6949 | 0.4805 |
| Climatology | — | — | — | 110.8151 | 0.0000 | 0.3689 |
| Dense 2D FNO (w=32) | 398,765 | 3.99 | 35.1 | 59.0100 ± 0.5411 | 0.7158 ± 0.0078 | 0.2514 ± 0.0027 |
| Dense 2D FNO Matched (w=36) | 504,553 | 4.22 | 40.0 | 56.0361 ± 1.3165 | 0.7277 ± 0.0089 | 0.2440 ± 0.0054 |
| PINO (Soft Physics) | 398,765 | 4.01 | 35.1 | 58.6431 ± 1.1826 | 0.7018 ± 0.0037 | 0.2571 ± 0.0026 |
| Proposed (FFNO + Sparse GNN) | 499,144 | 11.23 | 38.1 | 52.0139 ± 0.4885 | 0.7812 ± 0.0017 | 0.2302 ± 0.0036 |
| **v1: FFNO Only (Selected Final)** | 504,894 | 8.32 | 39.8 | **49.6658 ± 0.7751** | **0.7930 ± 0.0024** | **0.2209 ± 0.0016** |
| v2: FFNO + Dense GNN | 499,144 | 11.22 | 39.1 | 49.9271 ± 0.4361 | 0.7904 ± 0.0014 | 0.2234 ± 0.0008 |
| v3: FFNO + Random Sparse GNN | 499,144 | 11.13 | 38.0 | 52.5679 ± 1.2290 | 0.7788 ± 0.0030 | 0.2285 ± 0.0022 |
| v4: FFNO + Unconstrained GNN | 499,144 | 11.03 | 38.2 | 51.2725 ± 0.1793 | 0.7814 ± 0.0017 | 0.2287 ± 0.0029 |
| **t1: FFNO + Terrain Input (identical to v1)** | 504,894 | 8.32 | 39.8 | **49.6658 ± 0.7751** | **0.7930 ± 0.0024** | **0.2209 ± 0.0016** |
| t2: Proposed Zero-Init | 499,144 | 12.73 | 38.1 | 53.2375 ± 0.7118 | 0.7767 ± 0.0018 | 0.2307 ± 0.0014 |
| t3: Proposed Zero-Init + L2 | 499,144 | 26.99 | 38.1 | 52.3822 ± 0.4072 | 0.7759 ± 0.0058 | 0.2308 ± 0.0037 |
| t4: FFNO Terrain + Zero-Init GNN | 499,144 | 17.24 | 38.1 | 51.6534 ± 0.4338 | 0.7791 ± 0.0018 | 0.2260 ± 0.0024 |

