"""
================================================================================
Building-Height Join — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Appends the height statistics of the buildings surrounding each GPS point of
the tuk-tuk campaign. Same method and parameters as join_bh_standard.py,
applied to the tuk-tuk schema (generic 'lat' / 'lon' / 'rad_calib' columns,
one file per sensor).

--------------------------------------------------------------------------------
Columns added
--------------------------------------------------------------------------------
    mean_building_height_m  : mean height of buildings within the buffer [m]
    max_building_height_m   : height of the tallest building in the buffer [m]
    n_buildings_nearby      : number of building centroids within the buffer

Row count and existing columns are unchanged.

--------------------------------------------------------------------------------
Method
--------------------------------------------------------------------------------
Building centroids (GlobalBuildingAtlas, Lisbon metropolitan area clip, each
carrying a 'height' attribute) and valid GPS points are projected to
EPSG:3763 (Portugal TM06). Every centroid within BUFFER_RADIUS_M of a point is
captured (bh_mapping.join_building_height_buffer); the three columns are the
mean, max and count over that set. Only centroids inside the bbox of the
valid GPS points (plus BBOX_MARGIN_DEG) are loaded.

n_buildings_nearby = 0 is a genuine result (open space, wide road, park) and
is kept distinct from NaN (row failed the valid-row filter: missing
lat/lon/rad_calib, rad_calib = 0, or GPS outside mainland Portugal).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_tuktuk<INPUT_SUFFIX>.csv
    output : <sensor_id>_tuktuk<INPUT_SUFFIX>_bh.csv

Tuk-tuk join chain:
    decode_tuktuk.py         ->  <sensor_id>_tuktuk.csv
    join_ghi_fcul_tuktuk.py  ->  _ghi
    join_cams_tuktuk.py      ->  _ghi_cams
    join_cos23_tuktuk.py     ->  _ghi_cams_cos
    join_clc_tuktuk.py       ->  _ghi_cams_cos_clc
    join_bh_tuktuk.py        ->  _ghi_cams_cos_clc_bh          <- this script
    join_sw_tuktuk.py        ->  _ghi_cams_cos_clc_bh_sw
    normalize_tuktuk.py      ->  _ghi_cams_cos_clc_bh_sw_norm   ->  clean_data.py

--------------------------------------------------------------------------------
Data source
--------------------------------------------------------------------------------
GlobalBuildingAtlas (Zhu et al., 2025, ESSD), Lisbon metropolitan area clip,
as a GeoJSON of building centroids with heights — BUILDINGS_PATH in config.py.

Requirements
------------
    pip install pandas geopandas shapely pyproj

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
from config import TUKTUK_DIR, BUILDINGS_PATH  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── VIPV shared modules ───────────────────────────────────────────────────────
import bh_mapping as bhm  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = TUKTUK_DIR

# --- Building centroids (GeoJSON) ---------------------------------------------
BUILDINGS_FILE: Path = BUILDINGS_PATH

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j29", "j30", "j31", "j32", "j33"]

# --- Input suffix (pipeline stage of the input files) -------------------------
#   "_ghi_cams_cos_clc" -> after join_clc_tuktuk.py
INPUT_SUFFIX: str = "_ghi_cams_cos_clc"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_bh"

# --- Column names (decode_tuktuk.py schema, before normalize_tuktuk.py) -------
LAT_COL: str = "lat"
LON_COL: str = "lon"
RAD_CALIB_COL: str = "rad_calib"

# Mainland Portugal sanity bounds (lat_min, lat_max, lon_min, lon_max).
# Catches GPS sentinel values such as lat = lon = 0.0, which notna() does not
# and which would otherwise inflate the auto-detected bbox to near-global size.
MAINLAND_PT_BOUNDS: tuple[float, float, float, float] = (36.8, 42.2, -9.6, -6.1)

# --- Buffer radius [m] (same as the standard campaign, bh_mapping default) ----
BUFFER_RADIUS_M: float = bhm.BUFFER_RADIUS_M

# --- Margin around the GPS bbox when loading building centroids [deg] ---------
BBOX_MARGIN_DEG: float = 0.02

# ==============================================================================
# BBOX DETECTION
# ==============================================================================

def detect_bbox(
    sensor_ids: list[str],
    input_suffix: str,
    lat_col: str,
    lon_col: str,
    margin_deg: float,
) -> tuple[float, float, float, float]:
    """
    Detect the geographic bounding box (west, south, east, north) from all
    sensor files, with a margin added on each side.

    Only rows with a non-null, non-zero rad_calib and GPS coordinates within
    MAINLAND_PT_BOUNDS are used.

    Only files that actually exist are used; missing files are skipped with a
    warning. Returns (xmin, ymin, xmax, ymax) in EPSG:4326.
    """
    lat_min_pt, lat_max_pt, lon_min_pt, lon_max_pt = MAINLAND_PT_BOUNDS
    lats, lons = [], []
    for sensor_id in sensor_ids:
        path = DATA_DIR / f"{sensor_id}_tuktuk{input_suffix}.csv"
        if not path.exists():
            print(f"  [bbox] {sensor_id}: file not found, skipped.")
            continue
        df_tmp = pd.read_csv(path, usecols=[lat_col, lon_col, RAD_CALIB_COL])

        mask = (
            df_tmp[lat_col].notna() & df_tmp[lon_col].notna()
            & df_tmp[RAD_CALIB_COL].notna() & (df_tmp[RAD_CALIB_COL] != 0.0)
            & df_tmp[lat_col].between(lat_min_pt, lat_max_pt)
            & df_tmp[lon_col].between(lon_min_pt, lon_max_pt)
        )
        n_sentinel = (
            df_tmp[lat_col].notna() & df_tmp[lon_col].notna()
            & df_tmp[RAD_CALIB_COL].notna() & (df_tmp[RAD_CALIB_COL] != 0.0)
            & ~(df_tmp[lat_col].between(lat_min_pt, lat_max_pt)
                & df_tmp[lon_col].between(lon_min_pt, lon_max_pt))
        ).sum()
        if n_sentinel:
            print(f"  [bbox] {sensor_id}: dropped {n_sentinel:,} row(s) with "
                  f"GPS outside mainland Portugal bounds (sentinel/error values)")

        lats.extend(df_tmp.loc[mask, lat_col].tolist())
        lons.extend(df_tmp.loc[mask, lon_col].tolist())

    if not lats:
        raise RuntimeError("No valid GPS coordinates found across all sensor files.")

    return (
        min(lons) - margin_deg,
        min(lats) - margin_deg,
        max(lons) + margin_deg,
        max(lats) + margin_deg,
    )


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(
    sensor_id: str,
    buildings_gdf,
) -> None:
    """
    Join building heights onto one sensor's tuk-tuk dataset and write the
    output CSV.

    Parameters
    ----------
    sensor_id : str
        Sensor identifier (e.g. 'j33').
    buildings_gdf : gpd.GeoDataFrame
        Pre-loaded building footprints (shared across all sensors).
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
            f"Expected columns '{LAT_COL}' and '{LON_COL}' not found in "
            f"{input_file.name}.\nAvailable: {df.columns.tolist()}"
        )
    if RAD_CALIB_COL not in df.columns:
        raise KeyError(
            f"Expected column '{RAD_CALIB_COL}' not found in "
            f"{input_file.name}.\nAvailable: {df.columns.tolist()}"
        )

    # ── Valid-row pre-filter (same criterion as join_bh_standard.py) ─────────
    lat_min_pt, lat_max_pt, lon_min_pt, lon_max_pt = MAINLAND_PT_BOUNDS
    valid_mask = (
        df[LAT_COL].notna() & df[LON_COL].notna()
        & df[RAD_CALIB_COL].notna() & (df[RAD_CALIB_COL] != 0.0)
        & df[LAT_COL].between(lat_min_pt, lat_max_pt)
        & df[LON_COL].between(lon_min_pt, lon_max_pt)
    )
    n_valid = valid_mask.sum()
    n_invalid = len(df) - n_valid
    print(f"  Valid rows (lat/lon/rad_calib present, rad_calib != 0, "
          f"within mainland Portugal): {n_valid:,} ({n_valid / len(df) * 100:.1f}%)")
    if n_invalid:
        print(f"  Skipping buffer join on {n_invalid:,} invalid rows -> NaN "
              f"(includes GPS sentinel errors such as lat=lon=0.0)")

    # ── Initialise output columns for ALL rows (NaN by default) ──────────────
    result = df.copy()
    result["mean_building_height_m"] = pd.NA
    result["max_building_height_m"]  = pd.NA
    result["n_buildings_nearby"]     = pd.NA

    # ── Buffer join on the valid subset only ──────────────────────────────────
    if n_valid > 0:
        joined_valid = bhm.join_building_height_buffer(
            df.loc[valid_mask], LAT_COL, LON_COL, buildings_gdf,
            radius_m=BUFFER_RADIUS_M,
        )
        result.loc[valid_mask, "mean_building_height_m"] = joined_valid["mean_building_height_m"]
        result.loc[valid_mask, "max_building_height_m"]  = joined_valid["max_building_height_m"]
        result.loc[valid_mask, "n_buildings_nearby"]     = joined_valid["n_buildings_nearby"]

    # ── Summary (over the valid subset only) ──────────────────────────────────
    valid_result = result.loc[valid_mask]
    n_with_bldg   = (valid_result["n_buildings_nearby"] > 0).sum()
    n_no_bldg     = (valid_result["n_buildings_nearby"] == 0).sum()
    coverage_pct  = n_with_bldg / n_valid * 100.0 if n_valid > 0 else 0.0

    print("\n  Building-height join summary (valid rows only):")
    print(f"    rows with ≥1 building      : {n_with_bldg:,}  ({coverage_pct:.1f}%)")
    print(f"    rows with 0 buildings      : {n_no_bldg:,}  (open space / park / gap)")

    bh_valid = valid_result.loc[valid_result["n_buildings_nearby"] > 0, "mean_building_height_m"]
    if len(bh_valid) > 0:
        print("\n  mean_building_height_m (rows with ≥1 building):")
        print(f"    mean   = {bh_valid.mean():.2f} m")
        print(f"    median = {bh_valid.median():.2f} m")
        print(f"    p10    = {bh_valid.quantile(0.10):.2f} m")
        print(f"    p90    = {bh_valid.quantile(0.90):.2f} m")

    result.to_csv(output_file, index=False)
    print(f"\n  Written → {output_file}  ({len(result):,} rows)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("═" * 68)
    print("  BUILDING HEIGHT JOIN — tuk-tuk campaign")
    print(f"  Sensors        : {SENSOR_IDS}")
    print(f"  Input suffix   : {INPUT_SUFFIX!r}")
    print(f"  Output suffix  : {OUTPUT_SUFFIX!r}")
    print(f"  Buffer radius  : {BUFFER_RADIUS_M} m")
    print(f"  Buildings file : {BUILDINGS_FILE}")
    print("═" * 68)

    if not BUILDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Building footprints file not found:\n  {BUILDINGS_FILE}\n"
            "See data/README.md for how to obtain it."
        )

    # ── Detect bounding box from data ─────────────────────────────────────────
    print("\n[1/3] Detecting bounding box from sensor data ...")
    bbox = detect_bbox(SENSOR_IDS, INPUT_SUFFIX, LAT_COL, LON_COL, BBOX_MARGIN_DEG)
    print(f"  bbox (W, S, E, N) = {tuple(round(v, 4) for v in bbox)}")

    # ── Load building footprints once (shared across sensors) ─────────────────
    print("\n[2/3] Loading building footprints ...")
    buildings_gdf = bhm.load_building_heights(BUILDINGS_FILE, bbox=bbox)

    # ── Process each sensor ───────────────────────────────────────────────────
    print("\n[3/3] Joining building heights per sensor ...")
    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, buildings_gdf)

    print(f"\n{'═' * 68}")
    print("  Done.")
    print(f"{'═' * 68}\n")


if __name__ == "__main__":
    main()
