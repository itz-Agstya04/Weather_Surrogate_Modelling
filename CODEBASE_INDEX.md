# Codebase index

| Path | Purpose | Inputs | Outputs |
|---|---|---|---|
| `data/store.py` | Provenance, content hashing, trusted Zarr loading | Local Zarr store | Trusted `xarray.Dataset` |
| `data/validate_era5.py` | ERA5 physical, temporal, and split validation | Train, val, test Zarr stores | Gate report or failure |
| `data/fetch_weatherbench_cropped.py` | Public WeatherBench2 monthly acquisition | Anonymous public Zarr source | Real val/test Zarr stores and resume markers |
| `data/fetch_era5_cds.py` | CDS acquisition alternative | CDS credentials | Real ERA5 stores |
| `data/pino_data_pipeline.py` | Normalization and PyTorch datasets | Trusted Zarr and train statistics | Loaders and tensors |
| `data/terrain_pipeline.py` | Terrain feature and graph construction | Surface geopotential | Terrain NPZ artifact |
| `models/proposed_model.py` | Factorized spectral model, terrain residual, projection | State tensor and terrain artifact | Predicted state |
| `train/train_baselines.py` | Dense FNO, PINO losses, rollout training, evaluation utilities | Loaders, normalizer, model config | Checkpoints and metrics |
| `ablations/search_proposed.py` | Proposed-model configuration screening | Train/val loaders and terrain | Validation-selected configurations |
| `analysis/build_comparison_table.py` | Comparison-table construction | Evaluation result files | Comparison CSV |
| `run_pipeline.py` | Ordered pipeline entry point and validation gate | Project data and configs | Pipeline outputs and metadata |
| `archive_invalid_synthetic/` | Immutable quarantine | Invalid historical artifacts | Audit-only archive |
