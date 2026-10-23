# Data

The datasets are **not** stored in this repository: some are too large for
GitHub, and the sensor measurements are GPS tracks of real vehicles, which
are not published. Every script reads its inputs from the folder defined by
`DATA_ROOT` in [`config.py`](../config.py), which is this `data/` folder by
default. The contents of `data/` are ignored by git (see `.gitignore`), so
files placed here stay local.

Either place the files below in this folder, or keep them elsewhere and set
`DATA_ROOT` in `config.py` (or the `VIPV_DATA_ROOT` environment variable) to
a folder with the same layout. File names must match exactly: they are the
names the scripts look for.

```
data/
  campaign_standard/      standard campaign, sensors J30-J33
    j30_satellite_landuse_v2.csv ... j33_satellite_landuse_v2.csv
  campaign_tuktuk/        tuk-tuk campaign
    Jinie33_full_mission_tuktuk.csv
  fcul_pyranometer/       FCUL rooftop pyranometer
    GHI2021.xlsx  GHI2022_part2.xlsx  GHI2023.xlsx
    solar_data_for_Jinie33_full_mission_tuktuk.xlsx
  land_cover/
    clc2018/              U2018_CLC2018_V2020_20u1.gpkg
                          U2018_CLC2018_V2020_20u1.tif
    cos2023/              COS2023v1-S2.gpkg
  buildings/              lisbon_metro_buildings_height.geojson
  boundaries/             gadm41_PRT_0.shp ... gadm41_PRT_3.shp
                          (+ .dbf .shx .prj .cpg of each)
```

The preparation and enrichment scripts (`code/1_data_preparation`,
`code/2_data_enrichment`) write their outputs next to their inputs, in the
same campaign folder, with a suffix per processing step (see the main
README). Two further folders are created at the repository root and are not
versioned either: `outputs/` (figures and tables) and `cache/` (CAMS and
OpenStreetMap downloads).

---

## 1. Sensor measurements (not published)

**Source.** Solar Jinie VIPV irradiance sensors
([Enerjin](https://www.enerjin.com/vipv)) carried by vehicles in Lisbon and
operated by the VIPV research group of Instituto Dom Luiz (IDL), Faculdade de
Ciências da Universidade de Lisboa (FCUL).

**Access.** The files contain the GPS positions of real vehicles and are not
distributed with this repository. Researchers who obtain them from the group
can place them in `campaign_standard/` and `campaign_tuktuk/` as described
here; nothing else needs to change.

**Standard campaign (2021-2023).** One CSV per sensor,
`j3x_satellite_landuse_v2.csv`, 1-minute resolution, with the columns

```
Unnamed: 0, Human readable date (Europe/London), J3x lat, J3x lon,
J3x Rad raw, J3x Rad calib, ghi, ghi_clear, ghi_extra, bhi, dhi, dni,
sun_elev, sun_azim, land_use, land_use_str
```

These exports already carry the CAMS irradiance and the solar position, so
`join_cams_standard.py` was **not** run on them: it is kept for future
exports that lack these columns.

**Tuk-tuk campaign (2026).** Raw database export of sensor J33,
`Jinie33_full_mission_tuktuk.csv`, one JSON payload per transmitted packet.
`decode_tuktuk.py` turns it into `j33_tuktuk.csv`; the file names expected
for other sensors are listed in `INPUT_FILES` at the top of that script.

---

## 2. FCUL rooftop pyranometer (on request)

**Source.** Eko MS-802 pyranometer on the roof of FCUL (Lisbon), 1-minute
global horizontal irradiance, operated by IDL/FCUL.

**Access.** On request to the group; not published here.

**Format.** Excel files, 1-minute averages, timestamps in UTC:

| File | Sheet | Used by |
|---|---|---|
| `GHI2021.xlsx` | `DATA_2021` | `3_irradiance/ghi_fcul_vs_cams_*.py` (standard campaign) |
| `GHI2022_part2.xlsx` | `DATA_2022` | idem (January-May 2022 absent from the provider's data) |
| `GHI2023.xlsx` | `DATA_2023` | idem |
| `solar_data_for_Jinie33_full_mission_tuktuk.xlsx` | first sheet | `2_data_enrichment/join_ghi_fcul_tuktuk.py` |

Columns: `GMT TIME` and `Avg_GHI` for the yearly files; `TIMESTAMP, GMT` and
`Avg_GHI`, with a second row of units, for the tuk-tuk file.

---

## 3. CORINE Land Cover 2018 (public)

**Source.** Copernicus Land Monitoring Service (EEA), CLC2018, version
V2020_20u1. Product page and download:
<https://land.copernicus.eu/en/products/corine-land-cover/clc2018>
(free EU Login account required).

**Files.**
- Vector, GeoPackage: `U2018_CLC2018_V2020_20u1.gpkg`
  (used by `join_clc_standard.py`, `join_clc_tuktuk.py`).
- Raster, 100 m GeoTIFF: `U2018_CLC2018_V2020_20u1.tif`
  (used by `join_clc_raster_standard.py`).

The portal lets you restrict the download to an area of interest; a
Portugal extract is enough. Keep the original file names.

**Nomenclature.** The 44 classes and the urban/rural grouping used in this
work are listed in `docs/nomenclatures/CLC_Classification.xlsx` and
implemented in `code/common/clc_mapping.py`. Official legend:
<https://www.eea.europa.eu/en/analysis/publications/cor0-part2>.

---

## 4. COS2023 — Carta de Uso e Ocupação do Solo 2023 (public)

**Source.** Direção-Geral do Território (DGT), Portugal. COS2023, version 1,
Série 2, minimum mapping unit 1 ha, 93 level-4 classes.

**Download.** GeoPackage `COS2023v1-S2.gpkg` (about 1.5 GB), from the DGT
item on ArcGIS Online:
<https://www.arcgis.com/home/item.html?id=007e7eb1e01c47a29783f46a2d583cb5>,
or from the national catalogue, SNIG:
<https://snig.dgterritorio.gov.pt/>.

Note: since 2025 the DGT viewer (<https://smos.dgterritorio.gov.pt/vi-smos/>)
shows COS2025 (102 classes, 0.5 ha). The code expects the **COS2023**
93-class nomenclature; a newer edition needs an updated
`code/common/cos23_mapping.py`.

**Nomenclature.** The DGT nomenclature is published in Portuguese. The
English labels used in this work are a translation made for the project:
`docs/nomenclatures/COS_Classification_EN.xlsx`, with the Portuguese
original in `COS_Classification_PT.xlsx`.

---

## 5. Building heights — GlobalBuildingAtlas (public, derived file)

**Source.** GlobalBuildingAtlas (GBA), Zhu et al. (2025), *Earth System
Science Data* 17, 6647, <https://doi.org/10.5194/essd-17-6647-2025>;
code and data: <https://github.com/zhu-xlab/GlobalBuildingAtlas>.

**File.** `lisbon_metro_buildings_height.geojson`: one point per building
(footprint centroid) over the Lisbon metropolitan area, WGS 84, with the
properties `source, id, height, var, est_floors, footprint_m2,
total_floor_area_m2, volume_m3`. Only `height` (and `footprint_m2`, when
present) is used, through `code/common/bh_mapping.py`.

**How it was obtained.** The file comes from the U-Shift IMPT-data project
(<https://github.com/U-Shift/IMPT-data>, release 3.0), whose script
`code/buildings_height.R` extracts the GBA LoD1 buildings of the Lisbon tile
(`w010_n40_w005_n35`) from the public copy on Source Cooperative
(<https://source.coop/tge-labs/globalbuildingatlas-lod1>) and converts each
footprint to its centroid. To rebuild it, either download it from that
release, or run that script with the Lisbon bounding box.

---

## 6. Street widths — OpenStreetMap (downloaded by the code)

No file to provide. `code/common/sw_mapping.py` downloads the drivable
street network around the measurements from OpenStreetMap through `osmnx`
(Overpass API) each time a street-width join runs; an internet connection
is required. `6_urban_geometry/street_width_quality_diagnostic.py` caches
its network in `cache/osm_network/`. Data © OpenStreetMap contributors,
ODbL.

---

## 7. CAMS satellite irradiance (downloaded by the code)

No file to provide. `join_cams_tuktuk.py` and `join_cams_standard.py`
download the *CAMS solar radiation time-series* from the Copernicus
Atmosphere Data Store (ADS):
<https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries>
(documentation:
<https://confluence.ecmwf.int/spaces/CKB/pages/266592908/CAMS+solar+radiation+time-series+data+documentation>).

Requirements:
1. a free ADS account, and acceptance of the dataset licence on its page;
2. the `cdsapi` package and a `~/.cdsapirc` file with your ADS URL and key
   (see the ADS "How to use the API" page);
3. an internet connection. The number of requests per day is capped by the
   service; downloads are cached in `cache/cams/` and reused.

---

## 8. Administrative boundaries — GADM 4.1 (public)

**Source.** GADM version 4.1, Portugal, shapefile:
<https://gadm.org/download_country.html> (select Portugal, "Shapefile").
Unzip `gadm41_PRT_shp.zip` into `boundaries/`.

Levels used: 0 (country; `sensor_measurement_grid_intersection.py`, which
falls back to a coarser boundary if absent), 1 (districts) and 2
(municipalities; required by `filter_lisbon.py`, used for map outlines
elsewhere), 3 (parishes; optional, one figure of
`building_height_quality_diagnostic.py`). GADM data are free for academic
and non-commercial use (see the GADM licence).
