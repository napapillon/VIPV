# Urban shading and the solar potential of vehicle-integrated photovoltaics

Data-processing and analysis code for on-board irradiance measurements
collected by Solar Jinie sensors carried by vehicles in Lisbon: a standard
campaign (sensors J30-J33, 2021-2023) and a tuk-tuk campaign (2026). The
code quantifies how much solar energy a vehicle roof loses to urban shading,
and relates these losses to land cover, building height and street width.

Context and results are described in the internship report,
[`docs/Internship_Report_Papillon.pdf`](docs/Internship_Report_Papillon.pdf).

## Repository layout

```
config.py              all input/output paths (the only file to edit)
code/
  common/              shared modules (land-cover mappings, canyon model, joins)
  1_data_preparation/  tuk-tuk decoding, cleaning, spatial and temporal filters
  2_data_enrichment/   joins: CAMS, FCUL, CLC, COS2023, building height, street width
  3_irradiance/        GHI, clear-sky and diffuse-fraction analyses, FCUL/CAMS comparison
  4_mobility_and_sampling/  parked/moving status, spatial distribution of measurements
  5_land_cover/        CLC and COS2023 breakdowns and maps
  6_urban_geometry/    building height, street width, H/W ratio
  7_solar_autonomy/    solar autonomy and shading-loss estimates
  tuktuk_exploration/  raw tuk-tuk telemetry exploration
data/                  datasets, not versioned (see data/README.md)
docs/                  report and nomenclature tables (see docs/README.md)
outputs/               figures and tables written by the scripts, not versioned
cache/                 CAMS and OpenStreetMap downloads, not versioned
```

## Getting started

1. Install Python 3.12 or later and the dependencies:
   `pip install -r requirements.txt`
2. Obtain the datasets and place them under `data/` as described in
   [`data/README.md`](data/README.md), or point `DATA_ROOT` in `config.py`
   (or the `VIPV_DATA_ROOT` environment variable) to their location.
3. Run a script directly, from any working directory, e.g.
   `python code/7_solar_autonomy/solar_autonomy_losses.py`.
   Each script is configured by the constants in the `CONFIGURATION` block
   at its top (campaign, input variant, figure display); nothing else needs
   editing. Figures and tables go to `outputs/<folder>/<script output dir>/`.

Every script starts with a docstring giving its purpose, inputs, outputs and
parameters. The shared modules of `code/common/` run a self-test when
executed directly (`python code/common/clc_mapping.py`).

## Preparation chains and file names

Preparation and enrichment scripts (folders 1 and 2) write a new file next
to their input in `data/<campaign>/`, appending one suffix per processing
step. A file name therefore records how the file was produced.

Standard campaign:

```
j3x_satellite_landuse_v2.csv
  join_cams_standard.py        -> _cams    (CAMS irradiance and solar position)
  join_cos_standard.py         -> _cos
  join_clc_standard.py         -> _clc
  join_clc_raster_standard.py  -> _clcr
  join_bh_standard.py          -> _bh
  join_sw_standard.py          -> _sw
  clean_data.py                -> j3x_satellite_landuse_v2_sd_cleaned[_hq][_clearsky].csv
```

Tuk-tuk campaign:

```
Jinie33_full_mission_tuktuk.csv (raw export)
  decode_tuktuk.py             -> j33_tuktuk.csv
  join_ghi_fcul_tuktuk.py      -> _ghi
  join_cams_tuktuk.py          -> _cams
  join_cos23_tuktuk.py         -> _cos
  join_clc_tuktuk.py           -> _clc
  join_bh_tuktuk.py            -> _bh
  join_sw_tuktuk.py            -> _sw
  normalize_tuktuk.py          -> _norm
  clean_data.py                -> j33_tuktuk_ghi_cams_cos_clc_bh_sw_norm_sd_cleaned[_hq][_clearsky].csv
```

Both campaigns, optional selections on a cleaned file:
`filter_lisbon.py` (`_lisbon`) or `filter_region.py` (`_region`), then
`select_time_range.py` (`_<label>`).

Suffixes: `_sd` start-date filter (excludes the rooftop calibration phase),
`_cleaned` valid rows, `_hq` high-quality radiometric and geometric filter,
`_clearsky` clear-sky minutes (GHI/GHI_clear >= 0.95). The analysis scripts
select their input with a variant string such as `sd_cleaned_hq`. Appendix C
of the report describes the chains and the naming system in full.

## Scripts

### `code/common/` — shared modules

| Module | Role |
|---|---|
| `clc_mapping.py` | CORINE Land Cover 2018 classes (44) and urban/rural grouping |
| `cos23_mapping.py` | COS2023 level-4 classes (93), English labels |
| `bh_mapping.py` | Buffer join between measurement points and building heights |
| `sw_mapping.py` | Nearest-street join with OpenStreetMap street widths |
| `brito_svf_model.py` | Urban-canyon irradiance-loss model of Brito et al. (2024) |

### `code/1_data_preparation/`

| Script | Purpose |
|---|---|
| `decode_tuktuk.py` | Decodes the raw tuk-tuk database export into one CSV per sensor |
| `normalize_tuktuk.py` | Renames the joined tuk-tuk columns to the common sensor-prefixed schema |
| `clean_data.py` | Cleaning pipeline of both campaigns: start-date, validity, high-quality and clear-sky variants |
| `filter_lisbon.py` | Keeps the measurements inside the municipality of Lisbon (GADM) |
| `filter_region.py` | Keeps the measurements inside the Lisbon-Cascais bounding box |
| `select_time_range.py` | Extracts a date or date-time range of a cleaned variant |
| `sensor_deployment_diagnostic.py` | Detects the pre-deployment (rooftop) phase of each sensor, to set the start date |

### `code/2_data_enrichment/`

| Script | Purpose |
|---|---|
| `join_cams_standard.py` | CAMS irradiance and solar position, standard campaign (future exports) |
| `join_cams_tuktuk.py` | CAMS irradiance and solar position, tuk-tuk campaign |
| `join_ghi_fcul_tuktuk.py` | FCUL rooftop pyranometer GHI, tuk-tuk campaign |
| `join_cos_standard.py`, `join_cos23_tuktuk.py` | COS2023 land-use class |
| `join_clc_standard.py`, `join_clc_tuktuk.py` | CLC 2018 class (vector) |
| `join_clc_raster_standard.py` | CLC 2018 class (100 m raster), for comparison with the vector join |
| `join_bh_standard.py`, `join_bh_tuktuk.py` | Height statistics of the surrounding buildings |
| `join_sw_standard.py`, `join_sw_tuktuk.py` | Width of the nearest OpenStreetMap street |

### `code/3_irradiance/`

| Script | Purpose |
|---|---|
| `ghi_overview.py` | All-sky, clear-sky and extraterrestrial CAMS GHI over time |
| `daily_irradiance_bells.py` | Daily irradiance curves of the tuk-tuk campaign |
| `ghi_cams_vs_sensor_scatter.py` | CAMS GHI vs on-board sensor, by distance ring around FCUL |
| `ghi_fcul_vs_cams_scatter.py` | FCUL pyranometer vs CAMS GHI, by distance ring |
| `ghi_fcul_vs_cams_statistics.py` | Error statistics of CAMS GHI against the FCUL pyranometer |
| `kd_kt_analysis.py` | Diffuse fraction vs clearness index, monthly clearness index |
| `diffuse_ratio_impact_analysis.py` | Effect of the diffuse fraction on autonomy and irradiance losses |
| `clearsky_threshold_diagnostic.py` | Supports the choice of the clear-sky threshold (0.95) |

### `code/4_mobility_and_sampling/`

| Script | Purpose |
|---|---|
| `vehicle_movement_analysis.py` | Parked vs moving classification of each measurement |
| `movement_distribution_map.py` | Maps of moving and parked points |
| `measurement_distribution_map.py` | Map of all measurements over mainland Portugal |
| `measurement_distribution_map_bh_sw.py` | Same map restricted to points with building height and street width |
| `map_distance_rings.py` | Measurements in Lisbon with distance rings around FCUL |
| `measurement_rings_bar_chart.py` | Number of measurements per distance ring |
| `measurement_rings_stacked_by_sensor.py` | Same, stacked by sensor |
| `monthly_events_distance_chart.py` | Monthly events and travelled distance, city centre vs full dataset |
| `sensor_measurement_grid_intersection.py` | Measurements on a regular 0.03° grid over Portugal |
| `export_clusters_google_maps.py` | Exports stationary clusters to Excel for Google My Maps |
| `tuktuk_daily_distance_analysis.py` | Daily distance travelled by each tuk-tuk sensor |

### `code/5_land_cover/`

| Script | Purpose |
|---|---|
| `clc_urban_rural.py` | Urban/rural distribution of measurements (CLC), hourly and monthly |
| `cos23_urban_breakdown.py` | Measurements, relative loss and diffuse ratio by COS2023 category |
| `cos23_category_diagnostic_map.py` | Map of the measurements of selected COS2023 categories |
| `cos23_osm_polygone_map.py` | COS2023 polygons over an OpenStreetMap basemap |
| `landuse_classification_inspector.py` | Interactive map comparing four land-use classifications |

### `code/6_urban_geometry/`

| Script | Purpose |
|---|---|
| `building_height_vs_losses.py` | Autonomy losses vs building height, canyon-model overlay and confound diagnostics |
| `hw_ratio_impact_analysis.py` | Theoretical H/W sweep with the Brito et al. (2024) canyon model |
| `hw_ratio_real_analysis.py` | Measured losses vs real H/W ratio (building height / street width) |
| `building_height_quality_diagnostic.py` | Quality and coverage of the building-height data |
| `street_width_quality_diagnostic.py` | Quality and coverage of the OpenStreetMap street widths |
| `diagnose_building_coverage.py` | Rate of points without nearby buildings, by COS2023 category |
| `diagnose_bh_map.py` | Interactive map of the building-height join, all sensors |
| `bh_street_inspector.py` | Interactive map of the building-height join, one sensor |

### `code/7_solar_autonomy/`

| Script | Purpose |
|---|---|
| `solar_autonomy_losses.py` | **Main analysis**: daily solar autonomy and shading losses, Method A (pooled ratio of sums, retained) |
| `solar_autonomy_losses_method_b.py` | Same analysis, Method B (mean of minute ratios) |
| `solar_autonomy_losses_method_c.py` | Same analysis, Method C (mean of daily ratios) |
| `solar_autonomy_losses_building_height.py` | Method A with building-height bins in place of the urban/rural split |
| `solar_autonomy_analysis_clc.py` | Monthly summary and statistics by CLC class |
| `solar_autonomy_analysis_cos23.py` | Monthly summary and statistics by COS2023 class |
| `sal_sensitivity_analysis.py` | Comparison of the three aggregation methods |
| `sal_kc_elevation_diagnostic.py` | Clear-sky index x solar elevation, explains the A/B divergence |
| `sal_method_stability_diagnostic.py` | Daily-ratio distributions and leave-one-day-out test |
| `sal_weather_imbalance_diagnostic.py` | Weather-mix (clear-sky index) imbalance between compared conditions |
| `sal_threshold_calibration_diagnostic.py` | Calibration of the reliability thresholds (minimum minutes, sensor-days, parked/moving distance) |

### `code/tuktuk_exploration/`

| Script | Purpose |
|---|---|
| `tuktuk_telemetry_analysis.py` | Temperature, battery voltage and state of charge of the tuk-tuk sensor |
| `unknown_fields_analysis.py` | Exploration of the two undocumented Solar Jinie fields |
| `cams_grid_map.py` | CAMS discretisation grid over the tuk-tuk measurements |
| `visualize_cams_nodes.py` | CAMS time series downloaded for each grid node |

## Authors, licence, citation

See [`AUTHORS`](AUTHORS) and [`CITATION.cff`](CITATION.cff).
Released under the MIT licence, see [`LICENSE`](LICENSE).
