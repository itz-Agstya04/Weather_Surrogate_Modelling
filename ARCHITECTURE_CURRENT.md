# Current architecture

## Data

`data/pino_data_pipeline.py` loads the five data variables in the existing 13-channel order: surface temperature and wind, followed by geopotential at five pressure levels and temperature at five pressure levels. Normalization statistics are computed from the train store only. `data/store.py` requires source, request parameters, download date, content hash, and a matching content hash before opening a local store.

## Dense FNO and PINO

`train/train_baselines.py` defines `CoordinateFNO2d` with coordinate channels, spectral convolution over the horizontal grid, pointwise projections, and 13-channel output. The standard PINO path uses the same backbone and adds the existing divergence and hydrostatic penalties. The divergence penalty is horizontal 10 m wind divergence; it is not full three-dimensional mass continuity. The hydrostatic penalty compares geopotential and temperature relationships across pressure levels.

## Proposed model

`models/proposed_model.py` defines a factorized horizontal spectral backbone and a vertical spectral branch, followed by an output projection. `TerrainGNNResidual` uses terrain features and a learned gate on steep cells; the residual can be sparse or dense. `LerayProjection2d` applies the current spectral wind projection for the hard-constraint mode. The current hard projection is periodic and does not yet provide a boundary-aware treatment for the non-periodic domain.

## Current gaps

The current training paths predict absolute normalized states rather than a uniformly configured normalized 6-hour increment for every model. Time-of-day and annual phase channels, pressure-level surface masking, per-variable loss weights, boundary-aware projection, harmonic climatology, and the complete real-data Gate 2 protocol are not yet implemented. Existing historical model outputs are invalid and remain quarantined.
