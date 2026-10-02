# Urban shading and the solar potential of vehicle-integrated photovoltaics

Data-processing and analysis code for on-board irradiance measurements
collected by sensors carried by vehicles in Lisbon (standard campaign
2021–2023, tuk-tuk campaign 2026).

> **Status:** repository under construction. Sections marked TODO are being
> completed script by script.

## Repository layout

```
config.py              all input/output paths (the only file to edit)
code/
  common/              shared modules (land-cover mappings, canyon model)
  1_data_preparation/  cleaning, spatial/temporal filters, tuk-tuk decoding
  2_data_enrichment/   joins: CAMS, FCUL, CLC, COS2023, building height, street width
  3_irradiance/        GHI, clear-sky and diffuse-fraction analyses, FCUL/CAMS comparison
  4_mobility_and_sampling/  parked/moving status, spatial distribution of measurements
  5_land_cover/        CLC and COS2023 breakdowns and maps
  6_urban_geometry/    building height, street width, H/W ratio
  7_solar_autonomy/    solar autonomy and shading-loss estimates
  tuktuk_exploration/  raw tuk-tuk telemetry exploration
data/                  datasets — not versioned, see data/README.md
outputs/               figures and tables written by the scripts — not versioned
docs/                  internship report and supporting documents
```

## Getting started

1. Install Python ≥ 3.10 and the dependencies:
   `pip install -r requirements.txt`
2. Obtain the datasets and place them under `data/` as described in
   [`data/README.md`](data/README.md) (or point `DATA_ROOT` in `config.py`
   to their location).
3. Run the scripts in folder order (1 → 7). Each script is configured by the
   constants at its top (campaign, file variant, figure display) and writes
   its results to `outputs/<theme>/<script>/`.

Preparation and enrichment scripts (folders 1 and 2) write new data files
next to their input in `data/`, appending a suffix that records each
processing step (e.g. `_sd_cleaned_hq`). See `docs/pipeline.md`.

## Scripts

TODO — one line per script: the question it answers and what it produces.

## Authors, licence, citation

See [`AUTHORS`](AUTHORS) and [`CITATION.cff`](CITATION.cff).
Licence: TODO.
