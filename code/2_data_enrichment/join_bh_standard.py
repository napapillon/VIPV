"""
================================================================================
Building-Height Join — Standard Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Appends the height statistics of the buildings surrounding each GPS point of
the standard Solar Jinie campaign (J30–J33). Runs on the raw sensor files,
before clean_data.py.

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
    input  : <sensor_id>_satellite_landuse_v2_cams_cos_clc_clcr.csv
    output : <sensor_id>_satellite_landuse_v2_cams_cos_clc_clcr_bh.csv

Standard campaign join chain:
    join_cams_standard.py        ->  _cams
    join_cos_standard.py         ->  _cams_cos
    join_clc_standard.py         ->  _cams_cos_clc
    join_clc_raster_standard.py  ->  _cams_cos_clc_clcr
    join_bh_standard.py          ->  _cams_cos_clc_clcr_bh     <- this script
    join_sw_standard.py          ->  _cams_cos_clc_clcr_bh_sw  ->  clean_data.py

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
from config import STANDARD_DIR, BUILDINGS_PATH  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── VIPV shared modules ───────────────────────────────────────────────────────
import bh_mapping as bhm  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = STANDARD_DIR

# --- Building centroids (GeoJSON) ---------------------------------------------
BUILDINGS_FILE: Path = BUILDINGS_PATH

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j30", "j31", "j32", "j33"]

# --- Input suffix (output of join_clc_raster_standard.py; "" for raw) --------
INPUT_SUFFIX: str = "_cams_cos_clc_clcr"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_bh"

# --- Buffer radius [m] (bh_mapping default: 50 m) -----------------------------
BUFFER_RADIUS_M: float = bhm.BUFFER_RADIUS_M

# --- Margin around the GPS bbox when loading building centroids [deg] ---------
BBOX_MARGIN_DEG: float = 0.02

# Mainland Portugal sanity bounds (lat_min, lat_max, lon_min, lon_max).
# Catches GPS sentinel values such as lat = lon = 0.0, which notna() does not
# and which would otherwise inflate the auto-detected bbox to near-global size.
MAINLAND_PT_BOUNDS: tuple[float, float, float, float] = (36.8, 42.2, -9.6, -6.1)

# ==============================================================================
# BBOX DETECTION  (computed on VALID rows only)
# ==============================================================================

def detect_bbox(
    sensor_ids: list[str],
    input_suffix: str,
    margin_deg: float,
) -> tuple[float, float, float, float]:
    """
    Detect the geographic bounding box from all available sensor files,
    restricted to valid rows (non-null lat/lon/rad_calib, rad_calib != 0).

    Returns
    -------
    tuple
        (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
    """
    lats, lons = [], []
    for sensor_id in sensor_ids:
        lat_col = f"{sensor_id.upper()} lat"
        lon_col = f"{sensor_id.upper()} lon"
        rad_col = f"{sensor_id.upper()} Rad calib"
        path = DATA_DIR / f"{sensor_id}_satellite_landuse_v2{input_suffix}.csv"
        if not path.exists():
            print(f"  [bbox] {sensor_id}: file not found, skipped.")
            continue
        df_tmp = pd.read_csv(path, usecols=[lat_col, lon_col, rad_col])
        lat_min_pt, lat_max_pt, lon_min_pt, lon_max_pt = MAINLAND_PT_BOUNDS
        mask = (
            df_tmp[lat_col].notna() & df_tmp[lon_col].notna()
            & df_tmp[rad_col].notna() & (df_tmp[rad_col] != 0.0)
            & df_tmp[lat_col].between(lat_min_pt, lat_max_pt)
            & df_tmp[lon_col].between(lon_min_pt, lon_max_pt)
        )
        n_sentinel = (
            df_tmp[lat_col].notna() & df_tmp[lon_col].notna()
            & df_tmp[rad_col].notna() & (df_tmp[rad_col] != 0.0)
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

def process_sensor(sensor_id: str, buildings_gdf) -> None:
    """
    Join building heights onto one sensor's dataset and write the output CSV.

    All original rows are preserved. Only rows passing the valid-row filter
    are sent through the buffer join; all other rows receive NaN in the
    three new columns.
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
        df[lat_col].notna() & df[lon_col].notna()
        & df[rad_col].notna() & (df[rad_col] != 0.0)
        & df[lat_col].between(lat_min_pt, lat_max_pt)
        & df[lon_col].between(lon_min_pt, lon_max_pt)
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
            df.loc[valid_mask], lat_col, lon_col, buildings_gdf,
            radius_m=BUFFER_RADIUS_M,
        )
        result.loc[valid_mask, "mean_building_height_m"] = joined_valid["mean_building_height_m"]
        result.loc[valid_mask, "max_building_height_m"]  = joined_valid["max_building_height_m"]
        result.loc[valid_mask, "n_buildings_nearby"]     = joined_valid["n_buildings_nearby"]

    # ── Summary (over the valid subset only) ──────────────────────────────────
    valid_result = result.loc[valid_mask]
    n_with_bldg = (valid_result["n_buildings_nearby"] > 0).sum()
    n_no_bldg   = (valid_result["n_buildings_nearby"] == 0).sum()
    coverage    = n_with_bldg / n_valid * 100.0 if n_valid > 0 else 0.0

    print("\n  Building-height join summary (valid rows only):")
    print(f"    rows with >=1 building : {n_with_bldg:,}  ({coverage:.1f}%)")
    print(f"    rows with 0 buildings  : {n_no_bldg:,}  (open space / gap)")

    bh_valid = valid_result.loc[valid_result["n_buildings_nearby"] > 0, "mean_building_height_m"]
    if len(bh_valid) > 0:
        print("\n  mean_building_height_m (rows with >=1 building):")
        print(f"    mean   = {bh_valid.mean():.2f} m")
        print(f"    median = {bh_valid.median():.2f} m")
        print(f"    p10    = {bh_valid.quantile(0.10):.2f} m")
        print(f"    p90    = {bh_valid.quantile(0.90):.2f} m")

    result.to_csv(output_file, index=False)
    print(f"\n  Written -> {output_file.name}  ({len(result):,} rows, "
          f"{len(result.columns)} columns)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  BUILDING HEIGHT JOIN — standard campaign (J30-J33) — raw stage")
    print(f"  INPUT_SUFFIX   : {INPUT_SUFFIX!r}")
    print(f"  Sensors        : {SENSOR_IDS}")
    print(f"  Buffer radius  : {BUFFER_RADIUS_M} m")
    print(f"  Buildings file : {BUILDINGS_FILE.name}")
    print("=" * 68)

    if not BUILDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Building footprints file not found:\n  {BUILDINGS_FILE}\n"
            "See data/README.md for how to obtain it."
        )

    print("\n[1/3] Detecting bounding box from valid sensor data ...")
    bbox = detect_bbox(SENSOR_IDS, INPUT_SUFFIX, BBOX_MARGIN_DEG)
    print(f"  bbox (W, S, E, N) = {tuple(round(v, 4) for v in bbox)}")

    print("\n[2/3] Loading building footprints (restricted to bbox) ...")
    buildings_gdf = bhm.load_building_heights(BUILDINGS_FILE, bbox=bbox)

    print("\n[3/3] Joining building heights per sensor ...")
    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, buildings_gdf)

    print(f"\n{'=' * 68}")
    print("  Done.")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
