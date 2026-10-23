"""
================================================================================
COS2023 Land-Use Join — Standard Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Assigns a COS2023 land-use class (Carta de Uso e Ocupação do Solo, DGT
Portugal) to every GPS point of the standard Solar Jinie campaign (J30–J33)
by point-in-polygon join against the official national GeoPackage.

The join runs once, on the raw sensor files, before clean_data.py: the new
columns then pass unchanged into every cleaned variant.

--------------------------------------------------------------------------------
Columns added
--------------------------------------------------------------------------------
    cos23_n4_code   : official COS2023 level-4 code (e.g. '1.1.1.1')
    cos23_n4_label  : English level-4 label (cos23_mapping.py)
    cos23_n1_label  : English level-1 branch label (9 branches)

Rows failing the valid-row filter (missing lat/lon/rad_calib, rad_calib = 0,
GPS outside mainland Portugal) and points outside every polygon receive
'Unknown / missing'. Row count and existing columns are unchanged.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_satellite_landuse_v2_cams.csv
    output : <sensor_id>_satellite_landuse_v2_cams_cos.csv

Standard campaign join chain:
    join_cams_standard.py        ->  _cams
    join_cos_standard.py         ->  _cams_cos                 <- this script
    join_clc_standard.py         ->  _cams_cos_clc
    join_clc_raster_standard.py  ->  _cams_cos_clc_clcr
    join_bh_standard.py          ->  _cams_cos_clc_clcr_bh
    join_sw_standard.py          ->  _cams_cos_clc_clcr_bh_sw  ->  clean_data.py

--------------------------------------------------------------------------------
Data source
--------------------------------------------------------------------------------
COS2023v1, Série 2, mainland Portugal — Direção-Geral do Território (DGT):
    https://geo2.dgterritorio.gov.pt/cos/S2/COS2023/COS2023v1-S2-gpkg.zip
Unzip and place the .gpkg at the location given by COS23_PATH in config.py.

Requirements
------------
    pip install pandas geopandas shapely

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.0
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import STANDARD_DIR, COS23_PATH  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ── VIPV shared modules ───────────────────────────────────────────────────────
import cos23_mapping as cos23  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = STANDARD_DIR

# --- COS2023 GeoPackage -------------------------------------------------------
COS_GPKG_FILE: Path = COS23_PATH

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j30", "j31", "j32", "j33"]

# --- Input suffix (output of join_cams_standard.py; "" for the raw file) ------
INPUT_SUFFIX: str = "_cams"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_cos"

# Mainland Portugal sanity bounds (lat_min, lat_max, lon_min, lon_max).
# Catches GPS sentinel values such as lat = lon = 0.0, which notna() does not.
MAINLAND_PT_BOUNDS: tuple[float, float, float, float] = (36.8, 42.2, -9.6, -6.1)

# Candidate names of the level-4 code field, tried in order:
# 'COS23n4_C' (DGT technical specification) and 'COS23_n4_C' (spelling found
# in the distributed GeoPackage and the DGT ArcGIS service).
CODE_FIELD_CANDIDATES: list[str] = ["COS23n4_C", "COS23_n4_C"]

# ==============================================================================
# SPATIAL JOIN
# ==============================================================================

def _detect_code_field(columns: list[str]) -> str:
    """
    Return whichever candidate level-4 code field name is present in the
    GeoPackage, trying CODE_FIELD_CANDIDATES in order.

    Raises
    ------
    KeyError
        If none of the candidates are found, listing all available columns
        so the correct name can be added to CODE_FIELD_CANDIDATES.
    """
    for candidate in CODE_FIELD_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(
        f"None of the expected level-4 code fields {CODE_FIELD_CANDIDATES} "
        f"were found in the GeoPackage.\nAvailable columns: {columns}\n"
        f"Add the correct field name to CODE_FIELD_CANDIDATES."
    )


def load_polygons(gpkg_path: Path) -> tuple[gpd.GeoDataFrame, str]:
    """
    Load the national COS2023 GeoPackage into a GeoDataFrame in EPSG:4326.

    Parameters
    ----------
    gpkg_path : Path
        Path to the .gpkg file downloaded from DGT.

    Returns
    -------
    tuple[gpd.GeoDataFrame, str]
        (polygon layer in EPSG:4326, name of the detected level-4 code field)
    """
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    code_field = _detect_code_field(gdf.columns.tolist())
    return gdf, code_field


def join_cos23_subset(
    df_valid: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    polygons_gdf: gpd.GeoDataFrame,
    code_field: str,
) -> pd.DataFrame:
    """
    Spatially join COS2023 land-use codes onto a DataFrame of GPS points.

    Every row of df_valid is assumed to already have non-null lat/lon
    (the valid-row pre-filter is applied by the caller). Uses a
    point-in-polygon predicate; if a point falls inside overlapping
    polygons, only the first match is kept. Points outside all polygons
    receive 'Unknown / missing'.

    Parameters
    ----------
    df_valid : pd.DataFrame
        Pre-filtered rows with non-null lat_col / lon_col.
    lat_col, lon_col : str
        Column names (e.g. 'J31 lat', 'J31 lon').
    polygons_gdf : gpd.GeoDataFrame
        COS2023 polygon layer loaded by load_polygons().
    code_field : str
        Name of the level-4 code column in polygons_gdf (auto-detected).

    Returns
    -------
    pd.DataFrame
        df_valid with three new columns appended:
        cos23_n4_code, cos23_n4_label, cos23_n1_label.
    """
    result = df_valid.copy()

    points_gdf = gpd.GeoDataFrame(
        result[[lat_col, lon_col]].copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(result[lat_col], result[lon_col])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points_gdf,
        polygons_gdf[[code_field, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    result["cos23_n4_code"]      = joined[code_field]
    result["cos23_n4_label"]     = result["cos23_n4_code"].apply(cos23.get_label)
    result["cos23_n1_label"] = result["cos23_n4_code"].apply(
        lambda c: cos23.get_level1_label(c) if pd.notna(c) else "Unknown / missing"
    )
    return result


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(
    sensor_id: str,
    polygons_gdf: gpd.GeoDataFrame,
    code_field: str,
) -> None:
    """
    Join COS2023 codes onto one sensor's RAW dataset and write the output CSV.

    All original rows are preserved (same row count as input). Only rows
    passing the valid-row filter (non-null lat/lon, non-null and non-zero
    rad_calib) are actually spatially joined; all other rows receive
    'Unknown / missing' directly.
    """
    input_file  = DATA_DIR / f"{sensor_id}_satellite_landuse_v2{INPUT_SUFFIX}.csv"
    output_file = DATA_DIR / f"{sensor_id}_satellite_landuse_v2{INPUT_SUFFIX}{OUTPUT_SUFFIX}.csv"

    lat_col  = f"{sensor_id.upper()} lat"
    lon_col  = f"{sensor_id.upper()} lon"
    rad_col  = f"{sensor_id.upper()} Rad calib"

    print(f"\n{'-' * 68}")
    print(f"  Sensor : {sensor_id.upper()}")
    print(f"  Input  : {input_file.name}")
    print(f"  Output : {output_file.name}")
    print(f"{'-' * 68}")

    if not input_file.exists():
        print("  [SKIP] Input file not found.")
        return

    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows loaded")

    for col in (lat_col, lon_col, rad_col):
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' not found in {input_file.name}.\n"
                f"Available: {df.columns.tolist()}"
            )

    # ── Valid-row pre-filter (performance only — does not alter row count) ────
    lat_min_pt, lat_max_pt, lon_min_pt, lon_max_pt = MAINLAND_PT_BOUNDS
    valid_mask = (
        df[lat_col].notna()
        & df[lon_col].notna()
        & df[rad_col].notna()
        & (df[rad_col] != 0.0)
        & df[lat_col].between(lat_min_pt, lat_max_pt)
        & df[lon_col].between(lon_min_pt, lon_max_pt)
    )
    n_valid = valid_mask.sum()
    n_invalid = len(df) - n_valid
    print(f"  Valid rows (lat/lon/rad_calib present, rad_calib != 0, "
          f"within mainland Portugal): {n_valid:,} ({n_valid / len(df) * 100:.1f}%)")
    if n_invalid:
        print(f"  Skipping spatial join on {n_invalid:,} invalid rows "
              f"(GPS sentinel errors such as lat=lon=0.0, or will be dropped "
              f"by clean_data.py anyway) -> 'Unknown / missing'")

    # ── Initialise output columns for ALL rows ────────────────────────────────
    result = df.copy()
    result["cos23_n4_code"]      = pd.NA
    result["cos23_n4_label"]     = "Unknown / missing"
    result["cos23_n1_label"] = "Unknown / missing"

    # ── Spatial join on the valid subset only ─────────────────────────────────
    if n_valid > 0:
        joined_valid = join_cos23_subset(
            df.loc[valid_mask], lat_col, lon_col, polygons_gdf, code_field
        )
        result.loc[valid_mask, "cos23_n4_code"]      = joined_valid["cos23_n4_code"]
        result.loc[valid_mask, "cos23_n4_label"]     = joined_valid["cos23_n4_label"]
        result.loc[valid_mask, "cos23_n1_label"] = joined_valid["cos23_n1_label"]

    # ── Summary (computed over the valid subset, the only one that matters) ──
    valid_result = result.loc[valid_mask]
    n_matched   = (valid_result["cos23_n4_label"] != "Unknown / missing").sum()
    n_unmatched = n_valid - n_matched
    match_rate  = n_matched / n_valid * 100.0 if n_valid > 0 else 0.0
    print(f"\n  Matched (of valid rows)   : {n_matched:,}  ({match_rate:.1f}%)")
    print(f"  Unmatched (of valid rows) : {n_unmatched:,}  "
          f"(open water, coastline drift, or CAOP-boundary edge effects)")

    if n_valid > 0 and match_rate < 95.0:
        print(
            f"  [warn] Match rate below 95% ({match_rate:.1f}%). With the "
            "full national layer this usually means genuine GPS points over "
            "water/outside CAOP limits, not a coverage gap - but worth a "
            "spot check if the rate seems too low for your data."
        )

    if n_valid > 0:
        cos23.validate_against_nomenclature(valid_result, code_col="cos23_n4_code")

        print("\n  Distribution by COS2023 level-4 label (valid rows only):")
        for label, count in valid_result["cos23_n4_label"].value_counts().items():
            print(f"    {label:<60s}  {count:>5,}  ({count/n_valid*100:.1f}%)")

    result.to_csv(output_file, index=False)
    print(f"\n  Written -> {output_file.name}  ({len(result):,} rows, "
          f"{len(result.columns)} columns)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  COS2023 SPATIAL JOIN — standard campaign (J30-J33) — RAW stage")
    print(f"  COS GeoPackage : {COS_GPKG_FILE}")
    print(f"  Sensors        : {SENSOR_IDS}")
    print("=" * 68)

    if not COS_GPKG_FILE.exists():
        raise FileNotFoundError(
            f"COS2023 GeoPackage not found:\n  {COS_GPKG_FILE}\n"
            "Download it from:\n"
            "  https://geo2.dgterritorio.gov.pt/cos/S2/COS2023/COS2023v1-S2-gpkg.zip\n"
            "unzip it, and place the .gpkg at COS23_PATH (see config.py)."
        )

    print("\nLoading COS2023 national GeoPackage (this may take a moment) ...")
    polygons_gdf, code_field = load_polygons(COS_GPKG_FILE)
    print(f"  {len(polygons_gdf):,} polygons loaded  (CRS: {polygons_gdf.crs})")
    print(f"  detected level-4 code field: '{code_field}'")

    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, polygons_gdf, code_field)

    print(f"\n{'=' * 68}")
    print("  Done.")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
