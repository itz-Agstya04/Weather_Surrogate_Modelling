# Decisions

## 2026-10-11

- Retain `train_2018_2019.zarr` as the training split because its revised Gate 1 validation passes.
- Keep `archive_invalid_synthetic/` untouched and exclude it from all loaders and acquisition logic.
- Use the public WeatherBench2 anonymous Zarr source for real validation and test acquisition because no CDS account is configured and the user explicitly approved the cropped WeatherBench route.
- Acquire 2019 only for timestep-by-timestep verification against train; assemble only `val_2020.zarr` and `test_2021_2022.zarr` as replacement stores.
- Write monthly resume markers and provenance hashes before any replacement store can be trusted by project loaders.
- Do not start Stage 2 model or evaluation work until the replacement stores pass `data/validate_era5.py`.
