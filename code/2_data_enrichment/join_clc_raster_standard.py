"""
================================================================================
CLC 2018 Land-Cover Sampling (raster) — Standard Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Samples the CORINE Land Cover 2018 raster (100 m) at every valid GPS point of
the standard Solar Jinie campaign (J30–J33).

Cross-check only: kept in the join chain to compare with the vector join on
an interactive map, but not used as the land-cover reference. The vector join
(join_clc_standard.py, land_use_clcv) is the land-cover reference; this
raster sampling agreed with it at 98–99 % and is kept for re-verification.

--------------------------------------------------------------------------------
Columns added
--------------------------------------------------------------------------------
    land_use_clcr      : CLC grid_code (1–44). -999 for nodata, outside the
                         raster extent, or rows failing the valid-row filter
    land_use_str_clcr  : CLC level-3 English label (clc_mapping.py);
                         'Unknown / missing' where land_use_clcr = -999

Valid rows: non-null lat/lon/rad_calib, rad_calib != 0, GPS inside
MAINLAND_PT_BOUNDS. Row count and existing columns are unchanged.

--------------------------------------------------------------------------------
Pixel-value encoding
--------------------------------------------------------------------------------
Depending on the package, a CLC raster stores either the 1–44 grid_code
("grid_code") or the 3-digit CLC code ("clc_code"). With ENCODING_MODE =
"auto", a sample of pixels is read first and the convention matching the CLC
codebook best is selected.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_satellite_landuse_v2<INPUT_SUFFIX>.csv  (any stage)
    output : <sensor_id>_satellite_landuse_v2<INPUT_SUFFIX>_clcr.csv

Standard campaign join chain:
    join_cams_standard.py        ->  _cams
    join_cos_standard.py         ->  _cams_cos
    join_clc_standard.py         ->  _cams_cos_clc
    join_clc_raster_standard.py  ->  _cams_cos_clc_clcr        <- this script
    join_bh_standard.py          ->  _cams_cos_clc_clcr_bh
    join_sw_standard.py          ->  _cams_cos_clc_clcr_bh_sw  ->  clean_data.py

--------------------------------------------------------------------------------
Data source
--------------------------------------------------------------------------------
CORINE Land Cover 2018, raster 100 m, EPSG:3035 — Copernicus Land Monitoring
Service (EEA), DOI 10.2909/960998c1-1870-4e82-8051-6485205ebbac:
    https://land.copernicus.eu/en/products/corine-land-cover/clc2018
Location: CLC_RASTER_PATH in config.py.

Requirements
------------
    pip install pandas numpy rasterio pyproj

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
from config import STANDARD_DIR, CLC_RASTER_PATH  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

# ── VIPV shared modules ───────────────────────────────────────────────────────
from clc_mapping import CLC_MAPPING, CLC_LVL3_LABEL  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = STANDARD_DIR

# --- CLC 2018 raster (GeoTIFF, 100 m, EPSG:3035) ------------------------------
CLC_RASTER_FILE: Path = CLC_RASTER_PATH

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j30", "j31", "j32", "j33"]

# --- Input suffix (default: after the CAMS / COS / CLC joins; any stage) ------
INPUT_SUFFIX: str = "_cams_cos_clc"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_clcr"

# --- Pixel-value encoding: "auto" | "grid_code" | "clc_code" ------------------
ENCODING_MODE: str = "auto"

# --- Pixel values treated as nodata -------------------------------------------
RASTER_NODATA_VALUES: set[int] = {0, -128, 255}

# Mainland Portugal sanity bounds (lat_min, lat_max, lon_min, lon_max).
# Catches GPS sentinel values such as lat = lon = 0.0, which notna() does not.
MAINLAND_PT_BOUNDS: tuple[float, float, float, float] = (36.8, 42.2, -9.6, -6.1)

# --- Number of pixels sampled for encoding auto-detection ---------------------
ENCODING_DETECTION_SAMPLE_SIZE: int = 2000

# ==============================================================================
# REVERSE LOOKUP: 3-digit CLC code -> land_use (1-44)
# ==============================================================================

CLC_CODE_TO_LAND_USE: dict[int, int] = {
    entry.clc_code: land_use
    for land_use, entry in CLC_MAPPING.items()
}
VALID_GRID_CODES: set[int] = set(CLC_MAPPING.keys())          # {1, 2, ..., 44}
VALID_CLC_CODES: set[int] = set(CLC_CODE_TO_LAND_USE.keys())  # {111, 112, ..., 523}


# ==============================================================================
# RASTER SAMPLING
# ==============================================================================

def sample_raster(
    raster_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Sample raw pixel values from the CLC raster at the given WGS-84
    coordinates.

    Parameters
    ----------
    raster_path : Path
        Path to the CLC GeoTIFF.
    lons, lats : np.ndarray
        Longitude / latitude arrays in EPSG:4326.

    Returns
    -------
    np.ndarray
        Raw pixel values (int), one per input coordinate. Points falling
        outside the raster extent or in a masked/nodata cell return the
        raster's declared nodata value (or -999 if none is declared).
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata if src.nodata is not None else -999
        if src.crs is None:
            raise ValueError(
                f"Raster {raster_path.name} has no CRS defined — cannot "
                "reproject GPS coordinates for sampling."
            )
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)

        values = np.full(len(lons), nodata, dtype=np.float64)
        coords = list(zip(xs, ys))
        for i, val in enumerate(src.sample(coords)):
            values[i] = val[0]
    return values


def detect_raster_encoding(sample_values: np.ndarray) -> str:
    """
    Auto-detect whether raw raster pixel values are 1-44 grid_codes or
    3-digit CLC codes, by checking which convention the sampled values
    match better.

    Parameters
    ----------
    sample_values : np.ndarray
        A sample of raw pixel values from the raster (nodata already
        excluded by the caller).

    Returns
    -------
    str
        "grid_code" or "clc_code".
    """
    vals = set(int(v) for v in sample_values if not np.isnan(v))
    if not vals:
        print("  [warn] No valid sample values for encoding detection; "
              "defaulting to 'grid_code'.")
        return "grid_code"

    n_grid_match = len(vals & VALID_GRID_CODES)
    n_clc_match = len(vals & VALID_CLC_CODES)
    grid_rate = n_grid_match / len(vals)
    clc_rate = n_clc_match / len(vals)

    print(f"  Encoding detection: {len(vals)} distinct sampled values.")
    print(f"    Match rate as grid_code (1-44) : {grid_rate * 100:.1f}%")
    print(f"    Match rate as clc_code (3-digit): {clc_rate * 100:.1f}%")

    if grid_rate >= clc_rate:
        print("  -> Using 'grid_code' encoding (pixel value = land_use directly).")
        return "grid_code"
    print("  -> Using 'clc_code' encoding (pixel value = 3-digit CLC code).")
    return "clc_code"


def raw_values_to_land_use(raw_values: np.ndarray, encoding: str) -> np.ndarray:
    """
    Convert raw sampled pixel values to the 1-44 land_use grid_code,
    according to the detected/configured encoding. Unmapped or nodata
    values become -999.
    """
    land_use = np.full(len(raw_values), -999, dtype=np.int64)
    for i, v in enumerate(raw_values):
        if np.isnan(v):
            continue
        iv = int(round(v))
        if iv in RASTER_NODATA_VALUES:
            continue
        if encoding == "grid_code":
            if iv in VALID_GRID_CODES:
                land_use[i] = iv
        elif encoding == "clc_code":
            mapped = CLC_CODE_TO_LAND_USE.get(iv)
            if mapped is not None:
                land_use[i] = mapped
        else:
            raise ValueError(
                f"Unknown encoding {encoding!r}; expected 'grid_code' or 'clc_code'."
            )
    return land_use


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(sensor_id: str, encoding: str) -> None:
    """
    Sample the CLC raster onto one sensor's dataset and write the output CSV.
    All original rows and columns are preserved; only land_use_clcr and
    land_use_str_clcr are appended.
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

    df = pd.read_csv(input_file, low_memory=False)
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
        print(f"  Skipping raster sampling on {n_invalid:,} invalid rows -> sentinels")

    # ── Initialise output columns for ALL rows ────────────────────────────────
    result = df.copy()
    result["land_use_clcr"]     = -999
    result["land_use_str_clcr"] = "Unknown / missing"

    # ── Raster sampling on the valid subset only ──────────────────────────────
    if n_valid > 0:
        lons = df.loc[valid_mask, lon_col].to_numpy()
        lats = df.loc[valid_mask, lat_col].to_numpy()
        raw_values = sample_raster(CLC_RASTER_FILE, lons, lats)
        land_use_raster = raw_values_to_land_use(raw_values, encoding)

        result.loc[valid_mask, "land_use_clcr"] = land_use_raster
        result.loc[valid_mask, "land_use_str_clcr"] = pd.Series(
            land_use_raster, index=df.loc[valid_mask].index
        ).map(lambda lu: CLC_LVL3_LABEL.get(lu, "Unknown / missing") if lu != -999
              else "Unknown / missing")

    # ── Summary ────────────────────────────────────────────────────────────────
    valid_result = result.loc[valid_mask]
    n_matched = (valid_result["land_use_clcr"] != -999).sum()
    n_unmatched = n_valid - n_matched
    match_rate = n_matched / n_valid * 100.0 if n_valid > 0 else 0.0
    print(f"\n  Raster sampling match (of valid rows) : {n_matched:,}  ({match_rate:.1f}%)")
    print(f"  Unmatched (nodata / outside extent)   : {n_unmatched:,}")

    if n_valid > 0:
        print("\n  Distribution by land_use_str_clcr:")
        for label, count in valid_result["land_use_str_clcr"].value_counts().items():
            print(f"    {label:<55s}  {count:>5,}  ({count/n_valid*100:.1f}%)")

    result.to_csv(output_file, index=False)
    print(f"\n  Written -> {output_file.name}  ({len(result):,} rows, "
          f"{len(result.columns)} columns)")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  CLC 2018 RASTER SAMPLING — standard campaign (J30-J33)")
    print(f"  INPUT_SUFFIX  : {INPUT_SUFFIX!r}")
    print(f"  Sensors       : {SENSOR_IDS}")
    print(f"  Raster file   : {CLC_RASTER_FILE}")
    print(f"  Encoding mode : {ENCODING_MODE!r}")
    print("=" * 68)

    if not CLC_RASTER_FILE.exists():
        raise FileNotFoundError(
            f"CLC raster file not found:\n  {CLC_RASTER_FILE}\n"
            f"Download it from:\n"
            f"  https://land.copernicus.eu/en/products/corine-land-cover/clc2018\n"
            f"(Download tab -> pre-packaged raster collection) and place it "
            f"at CLC_RASTER_PATH (see config.py)."
        )

    with rasterio.open(CLC_RASTER_FILE) as src:
        print(f"\n  Raster CRS    : {src.crs}")
        print(f"  Raster shape  : {src.width} x {src.height} px")
        print(f"  Raster bounds : {src.bounds}")

    # ── Determine encoding ─────────────────────────────────────────────────────
    if ENCODING_MODE == "auto":
        print("\n[1/2] Auto-detecting raster pixel-value encoding ...")
        sample_lons, sample_lats = [], []
        for sensor_id in SENSOR_IDS:
            path = DATA_DIR / f"{sensor_id}_satellite_landuse_v2{INPUT_SUFFIX}.csv"
            if not path.exists():
                continue
            lat_col = f"{sensor_id.upper()} lat"
            lon_col = f"{sensor_id.upper()} lon"
            try:
                df_tmp = pd.read_csv(path, usecols=[lat_col, lon_col]).dropna()
            except ValueError:
                continue
            df_tmp = df_tmp.sample(
                n=min(ENCODING_DETECTION_SAMPLE_SIZE, len(df_tmp)), random_state=42
            )
            sample_lons.extend(df_tmp[lon_col].tolist())
            sample_lats.extend(df_tmp[lat_col].tolist())
            if len(sample_lons) >= ENCODING_DETECTION_SAMPLE_SIZE:
                break

        if not sample_lons:
            print("  [warn] No sensor data available for encoding detection; "
                  "defaulting to 'grid_code'.")
            encoding = "grid_code"
        else:
            raw_vals = sample_raster(
                CLC_RASTER_FILE,
                np.array(sample_lons[:ENCODING_DETECTION_SAMPLE_SIZE]),
                np.array(sample_lats[:ENCODING_DETECTION_SAMPLE_SIZE]),
            )
            raw_vals = raw_vals[~np.isin(raw_vals.astype(int), list(RASTER_NODATA_VALUES))]
            encoding = detect_raster_encoding(raw_vals)
    elif ENCODING_MODE in ("grid_code", "clc_code"):
        encoding = ENCODING_MODE
        print(f"\n[1/2] Using configured encoding: {encoding!r}")
    else:
        raise ValueError(
            f"Invalid ENCODING_MODE {ENCODING_MODE!r}; "
            "expected 'auto', 'grid_code' or 'clc_code'."
        )

    print("\n[2/2] Sampling raster per sensor ...")
    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, encoding)

    print(f"\n{'=' * 68}")
    print("  Done.")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
