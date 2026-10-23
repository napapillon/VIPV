"""
================================================================================
CLC 2018 Land-Cover Join (vector) — Standard Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Assigns a CORINE Land Cover 2018 class to every GPS point of the standard
Solar Jinie campaign (J30–J33) by point-in-polygon join against the CLC
vector layer. Runs on the raw sensor files, before clean_data.py.

land_use_clcv is the land-cover reference used by all downstream analyses.
It replaces the 'land_use' / 'land_use_str' columns delivered with the raw
files, which are kept untouched. A raster-based cross-check is available in
join_clc_raster_standard.py (98–99 % agreement with this vector join).

--------------------------------------------------------------------------------
Columns added
--------------------------------------------------------------------------------
    land_use_clcv      : CLC grid_code (1–44). -999 outside every polygon or
                         for rows failing the valid-row filter
    land_use_str_clcv  : CLC level-3 English label (clc_mapping.py);
                         'Unknown / missing' where land_use_clcv = -999

Urban / Rural is not stored: it is a lookup of land_use_clcv in
clc_mapping.CLC_URBAN_RURAL. Row count and existing columns are unchanged.

Valid rows: non-null lat/lon/rad_calib, rad_calib != 0, GPS inside
MAINLAND_PT_BOUNDS.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_satellite_landuse_v2<INPUT_SUFFIX>.csv
    output : <sensor_id>_satellite_landuse_v2<INPUT_SUFFIX>_clc.csv

Standard campaign join chain:
    join_cams_standard.py        ->  _cams
    join_cos_standard.py         ->  _cams_cos
    join_clc_standard.py         ->  _cams_cos_clc             <- this script
    join_clc_raster_standard.py  ->  _cams_cos_clc_clcr
    join_bh_standard.py          ->  _cams_cos_clc_clcr_bh
    join_sw_standard.py          ->  _cams_cos_clc_clcr_bh_sw  ->  clean_data.py

--------------------------------------------------------------------------------
Data source
--------------------------------------------------------------------------------
CORINE Land Cover 2018, vector GeoPackage, Portugal subset — Copernicus Land
Monitoring Service (EEA): https://land.copernicus.eu/en/products/corine-land-cover
Native CRS EPSG:3035, reprojected to EPSG:4326 for the join. Field 'Code_18'
(3-digit CLC code) is mapped to the 1–44 grid_code of clc_mapping.py.
Location: CLC_VECTOR_PATH in config.py.

CLC has a 25 ha minimum mapping unit: patches below it (small parks, narrow
green strips) are absorbed into the surrounding class by design.

Requirements
------------
    pip install pandas geopandas shapely

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.0
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import STANDARD_DIR, CLC_VECTOR_PATH  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ── VIPV shared modules ───────────────────────────────────────────────────────
from clc_mapping import CLC_MAPPING, CLC_LVL3_LABEL  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = STANDARD_DIR

# --- CLC 2018 vector GeoPackage -----------------------------------------------
CLC_FILE: Path = CLC_VECTOR_PATH

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j30", "j31", "j32", "j33"]

# --- Input suffix (default: after the CAMS / COS joins; "" for raw) -----------
INPUT_SUFFIX: str = "_cams_cos"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_clc"

# Mainland Portugal sanity bounds (lat_min, lat_max, lon_min, lon_max).
# Catches GPS sentinel values such as lat = lon = 0.0, which notna() does not.
MAINLAND_PT_BOUNDS: tuple[float, float, float, float] = (36.8, 42.2, -9.6, -6.1)

# ==============================================================================
# REVERSE LOOKUP: 3-digit CLC code -> land_use (1-44)
# ==============================================================================

CLC_CODE_TO_LAND_USE: dict[int, int] = {
    entry.clc_code: land_use
    for land_use, entry in CLC_MAPPING.items()
}


# ==============================================================================
# POLYGON LOADING
# ==============================================================================

def load_clc_polygons(gpkg_path: Path) -> gpd.GeoDataFrame:
    """
    Load the CLC 2018 GeoPackage and reproject to EPSG:4326.

    The EEA distributes CLC in EPSG:3035 (ETRS89 / LAEA Europe). Reprojecting
    to WGS-84 allows a direct point-in-polygon join with GPS coordinates.
    The 'Code_18' column is cast to integer for reliable matching with
    CLC_CODE_TO_LAND_USE.

    Parameters
    ----------
    gpkg_path : Path
        Path to the CLC 2018 GeoPackage file.

    Returns
    -------
    gpd.GeoDataFrame
        CLC polygons in EPSG:4326 with columns 'Code_18' and 'geometry'.
    """
    gdf = gpd.read_file(gpkg_path)
    print(f"  {len(gdf):,} polygons loaded  (source CRS: {gdf.crs})")

    if "Code_18" not in gdf.columns:
        candidates = [c for c in gdf.columns if "code" in c.lower() or "clc" in c.lower()]
        if not candidates:
            raise KeyError(
                f"Column 'Code_18' not found in the CLC GeoPackage.\n"
                f"Available columns: {gdf.columns.tolist()}\n"
                f"Update this script to match your file's column name."
            )
        actual_col = candidates[0]
        print(f"  [INFO] 'Code_18' not found — using '{actual_col}' instead.")
        gdf = gdf.rename(columns={actual_col: "Code_18"})

    gdf["Code_18"] = pd.to_numeric(gdf["Code_18"], errors="coerce").astype("Int64")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=3035)
    gdf = gdf.to_crs(epsg=4326)
    print("  Reprojected to EPSG:4326")

    return gdf[["Code_18", "geometry"]]


# ==============================================================================
# SPATIAL JOIN
# ==============================================================================

def join_clc_subset(
    df_valid: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    polygons_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatially join CLC 2018 land-use codes onto a DataFrame of GPS points.

    Every row of df_valid is assumed to already have non-null lat/lon (the
    valid-row pre-filter is applied by the caller). If a point falls inside
    overlapping polygons, only the first match is kept.

    Parameters
    ----------
    df_valid : pd.DataFrame
        Pre-filtered rows with non-null lat_col / lon_col.
    lat_col, lon_col : str
        Column names (e.g. 'J31 lat', 'J31 lon').
    polygons_gdf : gpd.GeoDataFrame
        CLC polygon layer from load_clc_polygons().

    Returns
    -------
    pd.DataFrame
        df_valid with land_use_clcv and land_use_str_clcv appended.
        -999 / 'Unknown / missing' for points outside all polygons.
    """
    result = df_valid.copy()

    points_gdf = gpd.GeoDataFrame(
        result[[lat_col, lon_col]].copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(result[lat_col], result[lon_col])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(points_gdf, polygons_gdf, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    result["land_use_clcv"] = joined["Code_18"].map(CLC_CODE_TO_LAND_USE)
    result["land_use_str_clcv"] = result["land_use_clcv"].map(CLC_LVL3_LABEL)

    result["land_use_clcv"] = result["land_use_clcv"].fillna(-999).astype(int)
    result["land_use_str_clcv"] = result["land_use_str_clcv"].fillna("Unknown / missing")

    return result


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(sensor_id: str, polygons_gdf: gpd.GeoDataFrame) -> None:
    """
    Join CLC land-use codes onto one sensor's dataset and write the output
    CSV. All original rows and columns are preserved unmodified; only
    land_use_clcv and land_use_str_clcv are appended.
    """
    input_file  = DATA_DIR / f"{sensor_id}_satellite_landuse_v2{INPUT_SUFFIX}.csv"
    output_file = DATA_DIR / f"{sensor_id}_satellite_landuse_v2{INPUT_SUFFIX}{OUTPUT_SUFFIX}.csv"

    lat_col = f"{sensor_id.upper()} lat"
    lon_col = f"{sensor_id.upper()} lon"
    rad_col = f"{sensor_id.upper()} Rad calib"

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

    # ── Valid-row pre-filter ───────────────────────────────────────────────────
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
    result["land_use_clcv"]     = -999
    result["land_use_str_clcv"] = "Unknown / missing"

    # ── Spatial join on the valid subset only ─────────────────────────────────
    if n_valid > 0:
        joined_valid = join_clc_subset(df.loc[valid_mask], lat_col, lon_col, polygons_gdf)
        result.loc[valid_mask, "land_use_clcv"]     = joined_valid["land_use_clcv"]
        result.loc[valid_mask, "land_use_str_clcv"] = joined_valid["land_use_str_clcv"]

    # ── Summary (over the valid subset only) ──────────────────────────────────
    valid_result = result.loc[valid_mask]
    n_matched   = (valid_result["land_use_clcv"] != -999).sum()
    n_unmatched = n_valid - n_matched
    match_rate  = n_matched / n_valid * 100.0 if n_valid > 0 else 0.0
    print(f"\n  CLC polygon match (of valid rows) : {n_matched:,}  ({match_rate:.1f}%)")
    print(f"  Unmatched (outside all CLC polygons) : {n_unmatched:,}")

    if n_valid > 0 and match_rate < 90.0:
        print(
            f"  [warn] Match rate below 90% ({match_rate:.1f}%). "
            "Verify the CLC file covers the full campaign area."
        )

    if n_valid > 0:
        print("\n  Distribution by land_use_str_clcv:")
        for label, count in valid_result["land_use_str_clcv"].value_counts().items():
            print(f"    {label:<55s}  {count:>5,}  ({count/n_valid*100:.1f}%)")

    result.to_csv(output_file, index=False)
    print(f"\n  Written -> {output_file.name}  ({len(result):,} rows, "
          f"{len(result.columns)} columns)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  CLC 2018 VECTOR SPATIAL JOIN — standard campaign (J30-J33)")
    print(f"  INPUT_SUFFIX : {INPUT_SUFFIX!r}")
    print(f"  Sensors      : {SENSOR_IDS}")
    print(f"  CLC file     : {CLC_FILE}")
    print("=" * 68)

    if not CLC_FILE.exists():
        raise FileNotFoundError(
            f"CLC GeoPackage not found:\n  {CLC_FILE}\n"
            f"Download CLC 2018 (vector, GeoPackage, Portugal) from:\n"
            f"  https://land.copernicus.eu/en/products/corine-land-cover\n"
            f"and place it at CLC_VECTOR_PATH (see config.py)."
        )

    print("\nLoading CLC 2018 polygons ...")
    polygons_gdf = load_clc_polygons(CLC_FILE)

    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, polygons_gdf)

    print(f"\n{'=' * 68}")
    print("  Done.")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
