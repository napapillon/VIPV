"""
================================================================================
Lisbon Metropolitan Region Spatial Filter
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Keeps only the measurements recorded inside a rectangular bounding box
(REGION_BBOX, WGS-84). The default box covers Lisbon, Cascais and the
surrounding municipalities; unlike filter_lisbon.py, no shapefile is
needed and the area is deliberately wider than the municipality. Works for
both campaigns (CAMPAIGN = 'standard' | 'tuktuk').

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <stem>_<INPUT_SUFFIX>.csv
    output : <stem>_<INPUT_SUFFIX>_<REGION_TAG>.csv

<stem> follows CAMPAIGN, as in clean_data.py:
    'standard' : <sensor_id>_satellite_landuse_v2
    'tuktuk'   : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm
Both campaigns share the column names used here ('<SENSOR> lat' / '<SENSOR> lon',
'Human readable date (Europe/London)') after normalize_tuktuk.py.

Outputs are written next to the inputs; downstream scripts select them with
FILE_SUFFIX = '<INPUT_SUFFIX>_<REGION_TAG>'. Rows with missing GPS are
dropped. Sensors whose input file is not found are skipped.

Presets (REGION_BBOX, REGION_TAG):
    Lisbon–Cascais region   (-9.52, 38.64, -9.04, 38.87)            'region'
    GBA building-data area  (-9.4975, 38.4001, -8.7000, 39.0999)    'zoneGBA'

Requirements
------------
    pip install pandas

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = 'standard'   # 'standard' | 'tuktuk'
if CAMPAIGN not in ('standard', 'tuktuk'):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input / output directory (one per campaign) ------------------------------
_DIRS = {
    'standard': STANDARD_DIR,
    'tuktuk':   TUKTUK_DIR,
}
DATA_DIR: Path = _DIRS[CAMPAIGN]
OUTPUT_DIR: Path = DATA_DIR

# --- File stem per campaign (stem of the clean_data.py outputs) ---------------
_STEMS = {
    'standard': '{sensor_id}_satellite_landuse_v2',
    'tuktuk':   '{sensor_id}_tuktuk_ghi_cams_cos_clc_bh_sw_norm',
}
FILE_STEM: str = _STEMS[CAMPAIGN]

# --- Input suffix (variant to filter; all four sensors are processed) --------
#   'sd_cleaned'              -> *_sd_cleaned_region.csv
#   'sd_cleaned_hq'           -> *_sd_cleaned_hq_region.csv
#   'sd_cleaned_clearsky'     -> *_sd_cleaned_clearsky_region.csv
#   'sd_cleaned_hq_clearsky'  -> *_sd_cleaned_hq_clearsky_region.csv
# Without the start-date filter in clean_data.py, drop the 'sd_' prefix.
INPUT_SUFFIX: str = 'sd_cleaned'

# --- Region bounding box (lon_min, lat_min, lon_max, lat_max), WGS-84 ---------
# A point is kept if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max.
REGION_BBOX: tuple[float, float, float, float] = (-9.52, 38.64, -9.04, 38.87)
# REGION_BBOX = (-9.4974996, 38.40008402, -8.70003556, 39.0999674)   # GBA area

# --- Tag appended to the output suffix ----------------------------------------
REGION_TAG: str = 'region'   # 'zoneGBA' for the GBA-area preset

# --- Name used in console output ----------------------------------------------
REGION_NAME: str = 'Lisbon Metropolitan Region'

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(data_dir: Path, suffix: str) -> dict[str, dict]:
    """
    Build a dict mapping sensor IDs to their input CSV path and GPS columns.

    The naming convention followed by the VIPV pipeline is:
        <stem>_<suffix>.csv   (FILE_STEM, set by CAMPAIGN)

    GPS column names in the CSV follow the pattern '<SENSOR_ID> lat' and
    '<SENSOR_ID> lon' (e.g. 'J31 lat', 'J31 lon').

    Parameters
    ----------
    data_dir : Path
        Directory containing the CSV files.
    suffix : str
        INPUT_SUFFIX value set in the configuration section.

    Returns
    -------
    dict[str, dict]
        {'J30': {'path': Path(...), 'lat': 'J30 lat', 'lon': 'J30 lon'}, ...}
    """
    result = {}
    for sid in ['J30', 'J31', 'J32', 'J33']:
        result[sid] = {
            'path': data_dir / f'{FILE_STEM.format(sensor_id=sid.lower())}_{suffix}.csv',
            'lat' : f'{sid} lat',
            'lon' : f'{sid} lon',
        }
    return result


def _build_output_path(output_dir: Path, sensor_id: str,
                       input_suffix: str) -> Path:
    """
    Build the output CSV path for one sensor.

    The output suffix is the input suffix with '_<REGION_TAG>' appended.

    Example (CAMPAIGN = 'standard', REGION_TAG = 'region')
    -----------------------------------------------------
    input_suffix='sd_cleaned_hq' → j31_satellite_landuse_v2_sd_cleaned_hq_region.csv

    Parameters
    ----------
    output_dir : Path
        Directory to write the output file into.
    sensor_id : str
        Sensor identifier, e.g. 'J31'.
    input_suffix : str
        INPUT_SUFFIX value set in the configuration section.

    Returns
    -------
    Path
        Full output file path.
    """
    out_suffix = f'{input_suffix}_{REGION_TAG}'
    return output_dir / f'{FILE_STEM.format(sensor_id=sensor_id.lower())}_{out_suffix}.csv'


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the region bounding-box spatial filter for all four sensors."""

    # ── Resolve sensor file dict from suffix ───────────────────────────────────
    sensor_files = _build_sensor_files(DATA_DIR, INPUT_SUFFIX)
    output_suffix = f'{INPUT_SUFFIX}_{REGION_TAG}'
    lon_min, lat_min, lon_max, lat_max = REGION_BBOX

    # ── Print configuration header ─────────────────────────────────────────────
    print('=' * 70)
    print('  Lisbon Metropolitan Region Spatial Filter  —  v1.0')
    print('=' * 70)
    print(f'  Campaign       : {CAMPAIGN}')
    print(f'  Input suffix   : {INPUT_SUFFIX}')
    print(f'  Output suffix  : {output_suffix}')
    print(f'  Data directory : {DATA_DIR}')
    print(f'  Region bbox    : lon [{lon_min}, {lon_max}]  '
          f'lat [{lat_min}, {lat_max}]  ({REGION_NAME})')
    print()

    # ── Filter each sensor CSV ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_in  = 0
    total_out = 0

    for sensor, cfg in sensor_files.items():
        csv_path = cfg['path']
        lat_col  = cfg['lat']
        lon_col  = cfg['lon']

        if not csv_path.exists():
            print(f'  [WARNING] File not found, skipping : {csv_path.name}')
            print()
            continue

        print(f'Processing sensor {sensor} : {csv_path.name}')
        df      = pd.read_csv(csv_path, low_memory=False)
        n_total = len(df)
        print(f'  → {n_total:,} rows loaded')

        # ── Validate GPS columns ──────────────────────────────────────────────
        missing_cols = [c for c in [lat_col, lon_col] if c not in df.columns]
        if missing_cols:
            print(f'  [WARNING] Columns {missing_cols} not found, skipping {sensor}')
            print()
            continue

        # ── Cast GPS to numeric, drop rows with missing coordinates ───────────
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df_valid    = df.dropna(subset=[lat_col, lon_col]).copy()

        n_missing_gps = n_total - len(df_valid)
        if n_missing_gps > 0:
            print(f'  → {n_missing_gps:,} rows dropped (missing GPS)')

        # ── Bounding-box test: keep only points inside REGION_BBOX ────────────
        inside_mask = (
            (df_valid[lon_col] >= lon_min) & (df_valid[lon_col] <= lon_max) &
            (df_valid[lat_col] >= lat_min) & (df_valid[lat_col] <= lat_max)
        )
        df_region = df_valid.loc[inside_mask].copy()

        n_inside = len(df_region)
        pct      = n_inside / n_total * 100 if n_total > 0 else 0.0
        print(f'  → {n_inside:,} rows inside {REGION_NAME} bbox '
              f'({pct:.1f}% of total)')

        # ── Write output ──────────────────────────────────────────────────────
        out_path = _build_output_path(OUTPUT_DIR, sensor, INPUT_SUFFIX)
        df_region.to_csv(out_path, index=False)
        print(f'  → Saved : {out_path.name}')

        total_in  += n_total
        total_out += n_inside
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print('─' * 70)
    print(f'  Total input rows  : {total_in:,}')
    if total_in > 0:
        print(f'  Total output rows : {total_out:,}  '
              f'({total_out / total_in * 100:.1f}% of input)')
    print(f'  Region bbox       : {REGION_BBOX}  ({REGION_NAME})')
    print(f'  Output suffix     : {output_suffix}')
    print('─' * 70)
    print()
    print('Done.')
    print()
    print('Next step — set FILE_SUFFIX in downstream scripts to:')
    print(f"    FILE_SUFFIX = '{output_suffix}'")


if __name__ == '__main__':
    main()
