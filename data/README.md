# Data

Datasets are **not** stored in this repository (size and licensing).
Place them here with the layout below, or set `DATA_ROOT` in `config.py`
(or the `VIPV_DATA_ROOT` environment variable) to another folder with the
same layout.

```
data/
  campaign_standard/   Solar Jinie sensor exports, j3x_satellite_landuse_v2.csv
  campaign_tuktuk/     tuk-tuk sensor J33 export
  fcul_pyranometer/    FCUL rooftop pyranometer GHI (Excel, one file per year)
  land_cover/
    clc2018/           U2018_CLC2018_V2020_20u1.gpkg and .tif
    cos2023/           COS2023v1-S2.gpkg
  buildings/           lisbon_metro_buildings_height.geojson
  boundaries/          gadm41_PRT_*.shp (+ .dbf .shx .prj .cpg)
```

| Dataset | Source | Access |
|---|---|---|
| Solar Jinie sensors (standard and tuk-tuk campaigns) | IDL / FCUL research group | TODO — on request to the group |
| FCUL rooftop pyranometer | IDL / FCUL | TODO — on request to the group |
| CORINE Land Cover 2018 (vector + 100 m raster) | Copernicus Land Monitoring Service (EEA) | TODO — download link |
| COS2023 | Direção-Geral do Território (DGT), Portugal | TODO — download link |
| Building heights, Lisbon metropolitan area | GlobalBuildingAtlas (Zhu et al., 2025), Lisbon clip | TODO |
| Administrative boundaries | GADM 4.1, Portugal | TODO — download link |
| Satellite irradiance | CAMS radiation service (ADS API) | downloaded by the scripts; requires an ADS key (`.cdsapirc`) |
