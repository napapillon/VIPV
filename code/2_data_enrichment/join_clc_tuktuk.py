"""
================================================================================
CLC 2018 Land-Cover Join (vector) — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Assigns a CORINE Land Cover 2018 class to every GPS point of the tuk-tuk
campaign by point-in-polygon join against the CLC vector layer. Produces the
same land_use_clcv / land_use_str_clcv columns as join_clc_standard.py, so
both campaigns share the same land-cover reference downstream.

--------------------------------------------------------------------------------
Columns added
--------------------------------------------------------------------------------
    land_use_clcv      : CLC grid_code (1–44). -999 outside every polygon or
                         for rows with missing GPS
    land_use_str_clcv  : CLC level-3 English label (clc_mapping.py)
    clc_urban_rural    : 'Urban' or 'Rural' (clc_mapping.py)

Rows with missing lat/lon receive -999 / 'Unknown / missing'. Row count and
existing columns are unchanged.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_tuktuk<INPUT_SUFFIX>.csv
    output : <sensor_id>_tuktuk<INPUT_SUFFIX>_clc.csv

Tuk-tuk join chain:
    decode_tuktuk.py         ->  <sensor_id>_tuktuk.csv
    join_ghi_fcul_tuktuk.py  ->  _ghi
    join_cams_tuktuk.py      ->  _ghi_cams
    join_cos23_tuktuk.py     ->  _ghi_cams_cos
    join_clc_tuktuk.py       ->  _ghi_cams_cos_clc             <- this script
    join_bh_tuktuk.py        ->  _ghi_cams_cos_clc_bh
    join_sw_tuktuk.py        ->  _ghi_cams_cos_clc_bh_sw
    normalize_tuktuk.py      ->  _ghi_cams_cos_clc_bh_sw_norm   ->  clean_data.py

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
from config import TUKTUK_DIR, CLC_VECTOR_PATH  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ── VIPV shared modules ───────────────────────────────────────────────────────
from clc_mapping import CLC_MAPPING, CLC_LVL3_LABEL, CLC_URBAN_RURAL  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = TUKTUK_DIR

# --- CLC 2018 vector GeoPackage -----------------------------------------------
CLC_FILE: Path = CLC_VECTOR_PATH

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j29", "j30", "j31", "j32", "j33"]

# --- Input suffix (pipeline stage of the input files) -------------------------
#   "_ghi_cams_cos" -> after join_cos23_tuktuk.py
INPUT_SUFFIX: str = "_ghi_cams_cos"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_clc"

# --- Column names (decode_tuktuk.py schema, before normalize_tuktuk.py) -------
LAT_COL: str = "lat"
LON_COL: str = "lon"

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

    # Reproject to WGS-84 for direct GPS join
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=3035)
    gdf = gdf.to_crs(epsg=4326)
    print("  Reprojected to EPSG:4326")

    return gdf[["Code_18", "geometry"]]


# ==============================================================================
# SPATIAL JOIN
# ==============================================================================

def join_clc(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    polygons_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatially join CLC 2018 land-use codes onto a DataFrame of GPS points.

    Three columns are added:
        land_use_clcv     : integer 1–44 (CLC grid_code). -999 for
                            unmatched points.
        land_use_str_clcv : CLC level-3 label from clc_mapping.py.
        clc_urban_rural   : 'Urban' or 'Rural' from clc_mapping.py.

    Rows with missing GPS receive land_use_clcv = -999 and
    'Unknown / missing'. If a point falls in overlapping polygons, the
    first match is kept.

    Parameters
    ----------
    df : pd.DataFrame
        Tuk-tuk measurements with lat_col and lon_col columns.
    lat_col, lon_col : str
        Names of the latitude and longitude columns.
    polygons_gdf : gpd.GeoDataFrame
        CLC polygon layer from load_clc_polygons().

    Returns
    -------
    pd.DataFrame
        Copy of df with land_use_clcv, land_use_str_clcv, clc_urban_rural
        appended.
    """
    result = df.copy()

    # Initialise output columns with sentinel values
    result["land_use_clcv"]     = -999
    result["land_use_str_clcv"] = "Unknown / missing"
    result["clc_urban_rural"]   = "Unknown / missing"

    valid_mask = result[lat_col].notna() & result[lon_col].notna()

    points_gdf = gpd.GeoDataFrame(
        result.loc[valid_mask, [lat_col, lon_col]].copy(),
        geometry=[
            Point(lon, lat)
            for lat, lon in zip(
                result.loc[valid_mask, lat_col],
                result.loc[valid_mask, lon_col],
            )
        ],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points_gdf,
        polygons_gdf,
        how="left",
        predicate="within",
    )

    # Keep only first match for overlapping polygons
    joined = joined[~joined.index.duplicated(keep="first")]

    # Map 3-digit CLC code → land_use_clcv (1–44)
    joined["land_use_clcv"] = joined["Code_18"].map(CLC_CODE_TO_LAND_USE)

    # Derive labels from clc_mapping.py
    joined["land_use_str_clcv"] = joined["land_use_clcv"].map(CLC_LVL3_LABEL)
    joined["clc_urban_rural"]   = joined["land_use_clcv"].map(CLC_URBAN_RURAL)

    # Fill NaN (unmatched points) with sentinel values
    joined["land_use_clcv"]     = joined["land_use_clcv"].fillna(-999).astype(int)
    joined["land_use_str_clcv"] = joined["land_use_str_clcv"].fillna("Unknown / missing")
    joined["clc_urban_rural"]   = joined["clc_urban_rural"].fillna("Unknown / missing")

    # Write back into result, preserving row order
    result.loc[valid_mask, "land_use_clcv"]     = joined["land_use_clcv"]
    result.loc[valid_mask, "land_use_str_clcv"] = joined["land_use_str_clcv"]
    result.loc[valid_mask, "clc_urban_rural"]   = joined["clc_urban_rural"]

    return result


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(sensor_id: str, polygons_gdf: gpd.GeoDataFrame) -> None:
    """
    Join CLC land-use codes onto one sensor's tuk-tuk dataset and write
    the output CSV.

    Parameters
    ----------
    sensor_id : str
        Sensor identifier (e.g. 'j33').
    polygons_gdf : gpd.GeoDataFrame
        Pre-loaded CLC polygon layer (shared across all sensors).
    """
    input_file  = DATA_DIR / f"{sensor_id}_tuktuk{INPUT_SUFFIX}.csv"
    output_file = DATA_DIR / f"{sensor_id}_tuktuk{INPUT_SUFFIX}{OUTPUT_SUFFIX}.csv"

    print(f"\n{'─' * 68}")
    print(f"  Sensor : {sensor_id.upper()}")
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")
    print(f"{'─' * 68}")

    if not input_file.exists():
        print("  [SKIP] Input file not found.")
        return

    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows loaded")

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise KeyError(
            f"Expected columns '{LAT_COL}' and '{LON_COL}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    n_missing = df[LAT_COL].isna().sum() + df[LON_COL].isna().sum()
    if n_missing > 0:
        print(f"  [warn] {n_missing} rows with missing GPS → land_use_clcv = -999")

    result = join_clc(df, LAT_COL, LON_COL, polygons_gdf)

    # Summary
    n_matched   = (result["land_use_clcv"] != -999).sum()
    n_unmatched = (result["land_use_clcv"] == -999).sum()
    match_rate  = n_matched / len(result) * 100.0 if len(result) > 0 else 0.0
    print(f"  Matched   : {n_matched:,}  ({match_rate:.1f}%)")
    print(f"  Unmatched : {n_unmatched:,}  (land_use_clcv = -999)")

    if match_rate < 80.0:
        print(
            f"  [warn] Low match rate ({match_rate:.1f}%). "
            f"Verify that the CLC file covers the full campaign area."
        )

    # Distribution
    print("\n  Distribution by CLC level-3 label (land_use_str_clcv):")
    for label, count in result["land_use_str_clcv"].value_counts().items():
        pct = count / len(result) * 100.0
        print(f"    {label:<55s}  {count:>5,}  ({pct:.1f}%)")

    print("\n  Distribution by Urban / Rural:")
    for group, count in result["clc_urban_rural"].value_counts().items():
        pct = count / len(result) * 100.0
        print(f"    {group:<20s}  {count:>5,}  ({pct:.1f}%)")

    result.to_csv(output_file, index=False)
    print(f"\n  Written → {output_file}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  CLC 2018 SPATIAL JOIN — tuk-tuk campaign")
    print(f"  Sensors      : {SENSOR_IDS}")
    print(f"  Input suffix : {INPUT_SUFFIX!r}")
    print(f"  CLC file     : {CLC_FILE}")
    print("=" * 68)

    if not CLC_FILE.exists():
        raise FileNotFoundError(
            f"CLC GeoPackage not found:\n  {CLC_FILE}\n"
            f"Download CLC 2018 (vector, GeoPackage, Portugal) from:\n"
            f"  https://land.copernicus.eu/en/products/corine-land-cover\n"
            f"and place it at CLC_VECTOR_PATH (see config.py)."
        )

    # Load polygons once — shared across all sensors
    print("\nLoading CLC 2018 polygons ...")
    polygons_gdf = load_clc_polygons(CLC_FILE)

    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, polygons_gdf)

    print(f"\n{'=' * 68}")
    print("  Done.")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
