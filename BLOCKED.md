# Blocked items

## Current

The public WeatherBench2 source is globally chunked. The bounded 20-timestep probe did not complete, and the raw chunk lower-bound estimate for 2019-2022 was 678.09 GiB, above the original Gate A threshold. The user explicitly authorized proceeding with the cropped WeatherBench acquisition anyway.

The detached WeatherBench attempt was stopped after the first January 2019 read had consumed about 1 GB without producing a month marker. Its log is retained in `logs/fetch_weatherbench_cropped_20261011_144510.log`. No synthetic fallback was used. The alternative path is to provide a source with spatially addressable chunks or create and accept the Copernicus CDS account licences, create `C:\Users\agast\.cdsapirc`, and rerun `data/fetch_era5_cds.py`.
