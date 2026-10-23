"""
================================================================================
Street-Width Join — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Appends the OSM-derived width of the nearest street to each GPS point of the
tuk-tuk campaign. Same method and parameters as join_sw_standard.py, applied
to the tuk-tuk schema (generic 'lat' / 'lon' / 'rad_calib' columns, one file
per sensor).

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
    edge_match_dist_m  : distance from the GPS point to the matched edge [m]

Row count and existing columns are unchanged. An existing output file is
skipped unless FORCE_RECOMPUTE = True.

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
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_tuktuk<INPUT_SUFFIX>.csv
    output : <sensor_id>_tuktuk<INPUT_SUFFIX>_sw.csv

Tuk-tuk join chain:
    decode_tuktuk.py         ->  <sensor_id>_tuktuk.csv
    join_ghi_fcul_tuktuk.py  ->  _ghi
    join_cams_tuktuk.py      ->  _ghi_cams
    join_cos23_tuktuk.py     ->  _ghi_cams_cos
    join_clc_tuktuk.py       ->  _ghi_cams_cos_clc
    join_bh_tuktuk.py        ->  _ghi_cams_cos_clc_bh
    join_sw_tuktuk.py        ->  _ghi_cams_cos_clc_bh_sw       <- this script
    normalize_tuktuk.py      ->  _ghi_cams_cos_clc_bh_sw_norm   ->  clean_data.py

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
from config import TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

# ── VIPV shared modules ───────────────────────────────────────────────────────
import sw_mapping as swm  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = TUKTUK_DIR

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j29", "j30", "j31", "j32", "j33"]

# --- Input suffix (pipeline stage of the input files) -------------------------
#   "_ghi_cams_cos_clc_bh" -> after join_bh_tuktuk.py
INPUT_SUFFIX: str = "_ghi_cams_cos_clc_bh"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_sw"

# --- Column names (decode_tuktuk.py schema, before normalize_tuktuk.py) -------
LAT_COL: str = "lat"
LON_COL: str = "lon"
RAD_CALIB_COL: str = "rad_calib"

# Mainland Portugal sanity bounds (lat_min, lat_max, lon_min, lon_max).
# Catches GPS sentinel values such as lat = lon = 0.0, which notna() does not
# and which would otherwise inflate the bbox of the OSM download.
MAINLAND_PT_BOUNDS: tuple[float, float, float, float] = (36.8, 42.2, -9.6, -6.1)

# --- Maximum point-to-edge distance [m] (sw_mapping default: 30 m) ------------
MAX_MATCH_DISTANCE_M: float = swm.MAX_MATCH_DISTANCE_M

# --- Overwrite an existing output file? ---------------------------------------
FORCE_RECOMPUTE: bool = False

# --- Margin around the GPS bbox for the OSM download [deg] --------------------
BBOX_MARGIN_DEG: float = 0.02

# ==============================================================================
# BOUNDING BOX DETECTION
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
    available sensor files, with a margin added on each side.

    Only rows with a non-null, non-zero rad_calib and GPS coordinates within
    MAINLAND_PT_BOUNDS are used.

    Only files that actually exist are included; missing files are skipped
    with a printed warning. Returns (xmin, ymin, xmax, ymax) in EPSG:4326.
    """
    lat_min_pt, lat_max_pt, lon_min_pt, lon_max_pt = MAINLAND_PT_BOUNDS
    lats: list[float] = []
    lons: list[float] = []

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
        raise RuntimeError(
            "No valid GPS coordinates found in any sensor file. "
            "Check DATA_DIR and INPUT_SUFFIX."
        )

    return (
        min(lons) - margin_deg,  # west
        min(lats) - margin_deg,  # south
        max(lons) + margin_deg,  # east
        max(lats) + margin_deg,  # north
    )


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(
    sensor_id: str,
    graph_proj,
    edges_gdf,
) -> None:
    """
    Join OSM street widths onto one sensor's tuk-tuk dataset and write the
    output CSV. The OSM graph is shared across sensors (downloaded once in
    main()) to avoid redundant Overpass API calls.

    Parameters
    ----------
    sensor_id : str
        Sensor identifier (e.g. 'j33').
    graph_proj : networkx.MultiDiGraph
        OSM street network projected to EPSG:3763 (Portugal TM06).
    edges_gdf : geopandas.GeoDataFrame
        Edge GeoDataFrame with pre-computed 'street_width_m' / 'width_source'.
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

    # Skip recomputation if output already exists and FORCE_RECOMPUTE is False.
    if output_file.exists() and not FORCE_RECOMPUTE:
        print("  [SKIP] Output already exists — set FORCE_RECOMPUTE = True to overwrite.")
        return

    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows loaded")

    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise KeyError(
            f"Expected columns '{LAT_COL}' and '{LON_COL}' not found in "
            f"{input_file.name}.\nAvailable columns: {df.columns.tolist()}"
        )
    if RAD_CALIB_COL not in df.columns:
        raise KeyError(
            f"Expected column '{RAD_CALIB_COL}' not found in "
            f"{input_file.name}.\nAvailable columns: {df.columns.tolist()}"
        )

    # ── Valid-row pre-filter (same criterion as join_sw_standard.py) ─────────
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
        print(f"  Skipping street-width match on {n_invalid:,} invalid rows "
              f"-> NaN / 'invalid_gps' (includes GPS sentinel errors such as "
              f"lat=lon=0.0)")

    # ── Initialise output columns for ALL rows ────────────────────────────────
    result = df.copy()
    result["street_width_m"]    = pd.NA
    result["width_source"]      = "invalid_gps"
    result["edge_match_dist_m"] = pd.NA

    # ── Nearest-edge join on the valid subset only ────────────────────────────
    if n_valid > 0:
        joined_valid = swm.join_street_width_nearest(
            df.loc[valid_mask], LAT_COL, LON_COL, graph_proj, edges_gdf,
            max_match_dist_m=MAX_MATCH_DISTANCE_M,
        )
        result.loc[valid_mask, "street_width_m"]    = joined_valid["street_width_m"]
        result.loc[valid_mask, "width_source"]      = joined_valid["width_source"]
        result.loc[valid_mask, "edge_match_dist_m"] = joined_valid["edge_match_dist_m"]

    # ── Summary (over the processed subset only) ──────────────────────────────
    valid_result   = result.loc[valid_mask]
    n_unmatched    = (valid_result["width_source"] == "unmatched_too_far").sum()
    n_matched      = n_valid - n_unmatched
    matched_pct    = n_matched / n_valid * 100.0 if n_valid > 0 else 0.0

    print("\n  Street-width join summary (valid rows only):")
    print(f"    Matched (≤{MAX_MATCH_DISTANCE_M:.0f} m)      : "
          f"{n_matched:,}  ({matched_pct:.1f}%)")
    print(f"    Unmatched (too far)    : {n_unmatched:,}")

    print("\n  width_source distribution (valid rows only):")
    print(valid_result["width_source"].value_counts(dropna=False).to_string())

    matched = valid_result.loc[valid_result["width_source"] != "unmatched_too_far", "street_width_m"]
    if len(matched) > 0:
        print("\n  street_width_m (matched rows only):")
        print(f"    mean   = {matched.mean():.2f} m")
        print(f"    median = {matched.median():.2f} m")
        print(f"    p10    = {matched.quantile(0.10):.2f} m")
        print(f"    p90    = {matched.quantile(0.90):.2f} m")

    result.to_csv(output_file, index=False)
    print(f"\n  Written → {output_file}  ({len(result):,} rows)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("═" * 68)
    print("  STREET WIDTH JOIN (OSM nearest-edge) — tuk-tuk campaign")
    print(f"  Sensors           : {SENSOR_IDS}")
    print(f"  Input suffix      : {INPUT_SUFFIX!r}")
    print(f"  Output suffix     : {OUTPUT_SUFFIX!r}")
    print(f"  Network type      : {swm.NETWORK_TYPE}")
    print(f"  Max match dist    : {MAX_MATCH_DISTANCE_M} m")
    print(f"  Sidewalk allowance: {swm.SIDEWALK_ALLOWANCE_M} m "
          f"(0.0 = carriageway width only)")
    print(f"  Force recompute   : {FORCE_RECOMPUTE}")
    print("═" * 68)

    # ── Step 1: bounding box ──────────────────────────────────────────────────
    print("\n[1/3] Detecting bounding box from sensor data ...")
    bbox = detect_bbox(SENSOR_IDS, INPUT_SUFFIX, LAT_COL, LON_COL, BBOX_MARGIN_DEG)
    print(f"  bbox (W, S, E, N) = {tuple(round(v, 4) for v in bbox)}")

    # ── Step 2: download OSM network (single call for all sensors) ────────────
    print("\n[2/3] Downloading OSM street network ...")
    graph_proj, edges_gdf = swm.load_street_network(bbox)

    # ── Step 3: join each sensor ──────────────────────────────────────────────
    print("\n[3/3] Joining street widths per sensor ...")
    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, graph_proj, edges_gdf)

    print(f"\n{'═' * 68}")
    print("  Done.")
    print(f"{'═' * 68}\n")


if __name__ == "__main__":
    main()
