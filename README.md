# Terrain-Aware Weather Surrogate Modelling

A regional, physics-informed surrogate model for short-range weather forecasting over Himachal Pradesh and the Western Himalayan region.

This project uses Fourier Neural Operators (FNOs) to learn the evolution of atmospheric fields on a latitude-longitude grid. It augments the spectral backbone with terrain features and an optional message-passing graph neural network (GNN) residual, while supporting differentiable physical constraints for wind and vertical atmospheric structure.

## Project highlights

- Forecasts **13 atmospheric state variables** at 6-hour intervals:
  - 2 m temperature
  - 10 m east–west and north–south wind components
  - Geopotential at 1000, 850, 700, 500, and 300 hPa
  - Temperature at 1000, 850, 700, 500, and 300 hPa
- Operates on a regional **15 x 17 latitude-longitude grid** covering Himachal Pradesh.
- Uses a factorized spectral model with separate horizontal and vertical Fourier processing.
- Represents local terrain using normalized elevation, slope, curvature, and 8-connected grid graphs.
- Supports soft divergence and hydrostatic losses, hard spectral wind projection, and iterative forecast refinement.
- Includes single-step evaluation, autoregressive rollout evaluation up to 72 hours, spatial error analysis, ablation search, and standalone inference.

## Model architecture

The proposed model is configurable through four components:

1. **Factorized FNO backbone** - applies 2D spectral convolutions across the geographic grid and 1D spectral convolutions across the five pressure levels.
2. **Terrain conditioning** - injects elevation-derived features into the backbone.
3. **Terrain GNN residual** - optionally propagates information over sparse steep-terrain cells or the full grid using learned edge messages.
4. **Physics and refinement modules** - applies optional Leray projection and iterative learned corrections to the predicted next state.

The repository also contains coordinate-aware FNO and provisional PINO baselines for controlled comparison.

## Results

The saved proposed-model experiment (`FactorizedFNO_GNNOff_soft_R3`) reports the following test-set results:

| Metric | One-step / 6 h | 72 h rollout |
|---|---:|---:|
| Aggregate nRMSE | **1.048** | **1.280** |
| Aggregate ACC | **0.679** | **0.381** |
| Inference latency | **31.4 ms** | — |
| Parameter count | **815,546** | — |

The one-step aggregate nRMSE compares with 1.118 for the coordinate FNO baseline, 1.112 for the provisional PINO baseline, and 1.211 for persistence.

The model search evaluates terrain-GNN variants, physics modes, and refinement settings. The best saved checkpoint uses the FNO backbone with soft physics and three refinement passes; the selected checkpoint has the GNN residual disabled. This distinction is preserved in the experiment name and results files rather than being hidden behind a generic “hybrid model” label.

## Data and preprocessing

The pipeline is designed around ERA5 reanalysis data accessed through the WeatherBench2 dataset layout. It uses:

- **Xarray** for labeled multidimensional data handling
- **Zarr** for chunked dataset storage
- **Dask** for parallel and lazy data processing
- Train split: `2018–2019`
- Validation split: `2020`
- Test split: `2021–2022`

The data preparation code extracts the Himachal Pradesh bounding box, validates the required variables, creates normalized tensors, and stores reusable artifacts under `DATASET/`.

## Repository structure

```text
.
├── ablations/                  Budget-aware proposed-model search
├── analysis/                   Evaluation, comparison, and spatial diagnostics
├── data/                       ERA5, terrain, and PINO data pipelines
├── models/                     Proposed FNO/GNN/physics model components
├── train/                      Baselines, training, and rollout fine-tuning
├── tests/                      Unit and integration tests
├── DATASET/                    Generated datasets, checkpoints, and metrics
├── run_pipeline.py             End-to-end reproducible pipeline
└── run_inference.py            Standalone autoregressive inference
```

## Installation

Create a Python environment and install the project dependencies. A CUDA-enabled PyTorch installation is recommended for training.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install torch numpy pandas xarray zarr dask scipy scikit-learn netCDF4 pytest
```

The exact PyTorch build should match the available CUDA toolkit when GPU training is required.

## Reproducing the pipeline

Run the complete pipeline from the repository root:

```bash
python run_pipeline.py
```

To reuse existing artifacts and checkpoints where available:

```bash
python run_pipeline.py --skip-existing
```

The pipeline performs the following stages:

1. Generate the terrain artifact and validate its graph tensors.
2. Prepare regional ERA5/WeatherBench2 train, validation, and test stores.
3. Add or validate the required temperature fields.
4. Compute train-only normalization statistics.
5. Train and evaluate persistence, FNO, and PINO baselines.
6. Search and fully train proposed-model configurations.
7. Compute spatial error diagnostics.
8. Build a combined comparison table.

Generated metrics and metadata are written to `DATASET/`, including:

- `baseline_results.json`
- `proposed_results.json`
- `comparison_table.csv`
- `spatial_error_summary.json`
- `run_metadata.json`

Individual stages can also be run directly, for example:

```bash
python -m data.terrain_pipeline
python -m data.prepare_hp_era5
python -m train.train_baselines
python -m ablations.search_proposed --budget-hours 1 --top-k 1
```

## Inference

After generating a checkpoint, run an autoregressive forecast from a test-set timestamp:

```bash
python run_inference.py \
  --model proposed \
  --config FactorizedFNO_GNNOff_soft_R3 \
  --init-time 2021-01-02T00:00:00 \
  --horizon-steps 12
```

Each step represents six hours, so `--horizon-steps 12` produces a 72-hour forecast. Forecasts are written as NetCDF files under `DATASET/inference/`.

The inference script also supports the baseline models:

```bash
python run_inference.py --model fno --init-time 2021-01-02T00:00:00 --horizon-steps 12
python run_inference.py --model pino --init-time 2021-01-02T00:00:00 --horizon-steps 12
python run_inference.py --model persistence --init-time 2021-01-02T00:00:00 --horizon-steps 12
```

## Testing

Run the test suite from the repository root:

```bash
pytest -q
```

The tests cover the terrain artifact, model components, physics losses, data loading, rollout contiguity, and training integration paths.

## Limitations and research scope

- The current experiments are regional and use a relatively coarse 0.25-degree-scale grid.
- Reported metrics depend on the saved dataset splits, checkpoint, and training configuration.
- Terrain-GNN variants are implemented and evaluated, but the best saved checkpoint is the GNN-off configuration.
- The project is a research prototype; broader validation across seasons, regions, resolutions, and independent weather datasets is required before operational use.

## License and attribution

This repository incorporates or follows components from the WeatherBench2 ecosystem. Refer to the `weatherbench2/` directory for its license and upstream project information. Add a project-specific license and citation file before public release if this repository is intended for external distribution.
