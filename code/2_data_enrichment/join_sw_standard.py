"""
================================================================================
Street-Width Join — Standard Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Appends the OSM-derived width of the nearest street to each GPS point of the
standard Solar Jinie campaign (J30–J33). Runs on the raw sensor files, before
clean_data.py.

--------------------------------------------------------------------------------
Columns added
--------------------------------------------------------------------------------
    street_width_m     : estimated carriageway width of the nearest OSM edge [m]
    width_source       : provenance of the estimate
                           'osm_tag'           explicit OSM width= tag
                           'lanes_estimate'    lane count x lane width
                           'highway_default'   default width for the highway type
                           'unmatched_too_far' no edge within MAX_MATCH_DISTANCE_M
                           'invalid_gps'       row failed the valid-row filter
                           'no_bh_coverage'    valid row without building height
                                               (RESTRICT_TO_VALID_BH = True)
    edge_match_dist_m  : distance from the GPS point to the matched edge [m]

Row count and existing columns are unchanged.

--------------------------------------------------------------------------------
Method
--------------------------------------------------------------------------------
The OSM drivable network covering the valid GPS points (plus BBOX_MARGIN_DEG)
is downloaded once with osmnx and projected to EPSG:3763. Each valid point is
matched to its nearest edge; if that edge is within MAX_MATCH_DISTANCE_M, its
width is estimated by the 3-tier fallback of sw_mapping.estimate_edge_width_m.
The result is a carriageway-width proxy, not a facade-to-facade canyon width.

Valid rows: non-null lat/lon/rad_calib, rad_calib != 0, GPS inside
MAINLAND_PT_BOUNDS.

Requires internet access (Overpass API).

Requirements
------------
    pip install pandas geopandas osmnx networkx numpy

--------------------------------------------------------------------------------
Restriction to rows with a building height (RESTRICT_TO_VALID_BH)
--------------------------------------------------------------------------------
H/W requires both H and W. With RESTRICT_TO_VALID_BH = True, only rows with a
non-NaN mean_building_height_m (join_bh_standard.py) are matched. This also
limits the OSM download to the coverage of the building dataset (Lisbon area)
instead of every location visited in mainland Portugal.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_satellite_landuse_v2_cams_cos_clc_clcr_bh.csv
    output : <sensor_id>_satellite_landuse_v2_cams_cos_clc_clcr_bh_sw.csv

Standard campaign join chain:
    join_cams_standard.py        ->  _cams
    join_cos_standard.py         ->  _cams_cos
    join_clc_standard.py         ->  _cams_cos_clc
    join_clc_raster_standard.py  ->  _cams_cos_clc_clcr
    join_bh_standard.py          ->  _cams_cos_clc_clcr_bh
    join_sw_standard.py          ->  _cams_cos_clc_clcr_bh_sw  <- this script  ->  clean_data.py

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
from config import STANDARD_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── VIPV shared modules ───────────────────────────────────────────────────────
import sw_mapping as swm  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = STANDARD_DIR

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j30", "j31", "j32", "j33"]

# --- Input suffix (output of join_bh_standard.py) -----------------------------
INPUT_SUFFIX: str = "_cams_cos_clc_clcr_bh"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_sw"

# --- Maximum point-to-edge distance [m] (sw_mapping default: 30 m) ------------
MAX_MATCH_DISTANCE_M: float = swm.MAX_MATCH_DISTANCE_M

# --- Margin around the GPS bbox for the OSM download [deg] --------------------
BBOX_MARGIN_DEG: float = 0.02

# Mainland Portugal sanity bounds (lat_min, lat_max, lon_min, lon_max).
# Catches GPS sentinel values such as lat = lon = 0.0, which notna() does not
# and which would otherwise inflate the bbox of the OSM download.
MAINLAND_PT_BOUNDS: tuple[float, float, float, float] = (36.8, 42.2, -9.6, -6.1)

# --- Restrict to rows with a building height (see module docstring) -----------
# Requires 'mean_building_height_m' in the input (INPUT_SUFFIX includes "_bh").
RESTRICT_TO_VALID_BH: bool = True

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
    restricted to valid rows (non-null lat/lon/rad_calib, rad_calib != 0,
    within mainland Portugal bounds) and, if RESTRICT_TO_VALID_BH is True,
    further restricted to rows with a non-NaN mean_building_height_m.

    The second restriction limits the bbox, and thus the OSM download, to the
    coverage of the building dataset.
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

        usecols = [lat_col, lon_col, rad_col]
        if RESTRICT_TO_VALID_BH:
            usecols.append("mean_building_height_m")
        df_tmp = pd.read_csv(path, usecols=usecols)

        lat_min_pt, lat_max_pt, lon_min_pt, lon_max_pt = MAINLAND_PT_BOUNDS
        mask = (
            df_tmp[lat_col].notna() & df_tmp[lon_col].notna()
            & df_tmp[rad_col].notna() & (df_tmp[rad_col] != 0.0)
            & df_tmp[lat_col].between(lat_min_pt, lat_max_pt)
            & df_tmp[lon_col].between(lon_min_pt, lon_max_pt)
        )
        n_before_bh = mask.sum()
        n_sentinel = (
            df_tmp[lat_col].notna() & df_tmp[lon_col].notna()
            & df_tmp[rad_col].notna() & (df_tmp[rad_col] != 0.0)
            & ~(df_tmp[lat_col].between(lat_min_pt, lat_max_pt)
                & df_tmp[lon_col].between(lon_min_pt, lon_max_pt))
        ).sum()
        if n_sentinel:
            print(f"  [bbox] {sensor_id}: dropped {n_sentinel:,} row(s) with "
                  f"GPS outside mainland Portugal bounds (sentinel/error values)")

        if RESTRICT_TO_VALID_BH:
            mask = mask & df_tmp["mean_building_height_m"].notna()
            n_dropped_bh = n_before_bh - mask.sum()
            if n_dropped_bh:
                print(f"  [bbox] {sensor_id}: dropped {n_dropped_bh:,} row(s) "
                      f"with no valid building-height value "
                      f"(outside GBA coverage or n_buildings_nearby == 0)")

        lats.extend(df_tmp.loc[mask, lat_col].tolist())
        lons.extend(df_tmp.loc[mask, lon_col].tolist())

    if not lats:
        raise RuntimeError(
            "No valid GPS coordinates found across all sensor files"
            + (" with a usable building-height value." if RESTRICT_TO_VALID_BH else ".")
        )

    return (
        min(lons) - margin_deg,
        min(lats) - margin_deg,
        max(lons) + margin_deg,
        max(lats) + margin_deg,
    )


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(sensor_id: str, graph_proj, edges_gdf) -> None:
    """
    Join street width onto one sensor's dataset and write the output CSV.

    All original rows are preserved. Only rows passing the valid-row filter
    are matched against the street network; all other rows receive NaN and
    'invalid_gps' or 'no_bh_coverage'.
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

    if RESTRICT_TO_VALID_BH and "mean_building_height_m" not in df.columns:
        raise KeyError(
            "RESTRICT_TO_VALID_BH is True but 'mean_building_height_m' is not "
            f"in {input_file.name}.\nEither run join_bh_standard.py first "
            "(so INPUT_SUFFIX includes '_bh'), or set "
            "RESTRICT_TO_VALID_BH = False to process every valid GPS row."
        )

    # ── Valid-row pre-filter ───────────────────────────────────────────────────
    lat_min_pt, lat_max_pt, lon_min_pt, lon_max_pt = MAINLAND_PT_BOUNDS
    basic_valid_mask = (
        df[lat_col].notna() & df[lon_col].notna()
        & df[rad_col].notna() & (df[rad_col] != 0.0)
        & df[lat_col].between(lat_min_pt, lat_max_pt)
        & df[lon_col].between(lon_min_pt, lon_max_pt)
    )
    n_basic_valid = basic_valid_mask.sum()
    n_invalid = len(df) - n_basic_valid
    print(f"  Valid rows (lat/lon/rad_calib present, rad_calib != 0, "
          f"within mainland Portugal): {n_basic_valid:,} "
          f"({n_basic_valid / len(df) * 100:.1f}%)")
    if n_invalid:
        print(f"  Skipping street-width match on {n_invalid:,} invalid rows "
              f"-> 'invalid_gps' (includes GPS sentinel errors such as "
              f"lat=lon=0.0)")

    if RESTRICT_TO_VALID_BH:
        valid_mask = basic_valid_mask & df["mean_building_height_m"].notna()
        n_dropped_bh = n_basic_valid - valid_mask.sum()
        if n_dropped_bh:
            print(f"  Further restricting to rows with a valid building-height "
                  f"value: dropped {n_dropped_bh:,} row(s) without one "
                  f"({valid_mask.sum():,} rows remaining, "
                  f"{valid_mask.sum() / len(df) * 100:.1f}% of total)")
    else:
        valid_mask = basic_valid_mask
    n_valid = valid_mask.sum()

    # ── Initialise output columns for ALL rows ────────────────────────────────
    result = df.copy()
    result["street_width_m"]    = pd.NA
    result["width_source"]      = "invalid_gps"
    result["edge_match_dist_m"] = pd.NA
    if RESTRICT_TO_VALID_BH:
        result.loc[basic_valid_mask & ~valid_mask, "width_source"] = "no_bh_coverage"

    # ── Street-width match on the valid subset only ───────────────────────────
    if n_valid > 0:
        joined_valid = swm.join_street_width_nearest(
            df.loc[valid_mask], lat_col, lon_col, graph_proj, edges_gdf,
            max_match_dist_m=MAX_MATCH_DISTANCE_M,
        )
        result.loc[valid_mask, "street_width_m"]    = joined_valid["street_width_m"]
        result.loc[valid_mask, "width_source"]      = joined_valid["width_source"]
        result.loc[valid_mask, "edge_match_dist_m"] = joined_valid["edge_match_dist_m"]

    # ── Summary (over the processed subset only) ──────────────────────────────
    # Matched rows carry 'osm_tag', 'lanes_estimate' or 'highway_default'.
    valid_result = result.loc[valid_mask]
    n_unmatched = (valid_result["width_source"] == "unmatched_too_far").sum()
    n_osm_tag   = (valid_result["width_source"] == "osm_tag").sum()
    n_lanes     = (valid_result["width_source"] == "lanes_estimate").sum()
    n_default   = (valid_result["width_source"] == "highway_default").sum()
    match_rate  = (n_valid - n_unmatched) / n_valid * 100.0 if n_valid > 0 else 0.0

    print("\n  Street-width join summary (processed rows only):")
    print(f"    OSM explicit tag (width=)    : {n_osm_tag:,}")
    print(f"    Lane-count estimate          : {n_lanes:,}")
    print(f"    Highway-type default         : {n_default:,}")
    print(f"    Unmatched (>{MAX_MATCH_DISTANCE_M:.0f} m)          : {n_unmatched:,}")
    print(f"    Match rate                   : {match_rate:.1f}%")

    sw_valid = valid_result.loc[valid_result["width_source"] != "unmatched_too_far", "street_width_m"]
    if len(sw_valid) > 0:
        print("\n  street_width_m (matched rows):")
        print(f"    mean   = {sw_valid.mean():.2f} m")
        print(f"    median = {sw_valid.median():.2f} m")
        print(f"    p10    = {sw_valid.quantile(0.10):.2f} m")
        print(f"    p90    = {sw_valid.quantile(0.90):.2f} m")

    result.to_csv(output_file, index=False)
    print(f"\n  Written -> {output_file.name}  ({len(result):,} rows, "
          f"{len(result.columns)} columns)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  STREET WIDTH JOIN — standard campaign (J30-J33) — raw stage")
    print(f"  INPUT_SUFFIX         : {INPUT_SUFFIX!r}")
    print(f"  RESTRICT_TO_VALID_BH : {RESTRICT_TO_VALID_BH}")
    print(f"  Sensors              : {SENSOR_IDS}")
    print(f"  Max match distance   : {MAX_MATCH_DISTANCE_M} m")
    print(f"  Network type         : {swm.NETWORK_TYPE}")
    print(f"  Sidewalk allowance   : {swm.SIDEWALK_ALLOWANCE_M} m")
    print("=" * 68)

    print("\n[1/3] Detecting bounding box from valid sensor data ...")
    bbox = detect_bbox(SENSOR_IDS, INPUT_SUFFIX, BBOX_MARGIN_DEG)
    print(f"  bbox (W, S, E, N) = {tuple(round(v, 4) for v in bbox)}")

    print("\n[2/3] Downloading OSM street network ...")
    graph_proj, edges_gdf = swm.load_street_network(bbox)

    print("\n[3/3] Joining street width per sensor ...")
    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, graph_proj, edges_gdf)

    print(f"\n{'=' * 68}")
    print("  Done.")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
