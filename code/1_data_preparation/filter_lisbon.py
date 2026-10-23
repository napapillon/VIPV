"""
================================================================================
Lisbon Spatial Filter
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Keeps only the measurements recorded strictly inside the municipality of
Lisbon: point-in-polygon test (geopandas .within()) against the Lisboa
polygon of the GADM level-2 boundaries (gadm41_PRT_2.shp). Works for both
campaigns (CAMPAIGN = 'standard' | 'tuktuk').

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <stem>_<INPUT_SUFFIX>.csv
    output : <stem>_<INPUT_SUFFIX>_lisbon.csv

<stem> follows CAMPAIGN, as in clean_data.py:
    'standard' : <sensor_id>_satellite_landuse_v2
    'tuktuk'   : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm
Both campaigns share the column names used here ('<SENSOR> lat' / '<SENSOR> lon',
'Human readable date (Europe/London)') after normalize_tuktuk.py.

Outputs are written next to the inputs; downstream scripts select them with
FILE_SUFFIX = '<INPUT_SUFFIX>_lisbon'. Rows with missing GPS are dropped.
Sensors whose input file is not found are skipped.

    clean_data.py        ->  *_sd_cleaned[...].csv
    filter_lisbon.py     ->  *_<INPUT_SUFFIX>_lisbon.csv     <- this script
    select_time_range.py ->  *_<INPUT_SUFFIX>_<label>.csv   (optional)

--------------------------------------------------------------------------------
Data source
--------------------------------------------------------------------------------
GADM 4.1, Portugal, level 2 (https://gadm.org) — BOUNDARIES_DIR in config.py.

Requirements
------------
    pip install pandas geopandas shapely

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import STANDARD_DIR, TUKTUK_DIR, BOUNDARIES_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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
#   'sd_cleaned'              -> *_sd_cleaned_lisbon.csv
#   'sd_cleaned_hq'           -> *_sd_cleaned_hq_lisbon.csv
#   'sd_cleaned_clearsky'     -> *_sd_cleaned_clearsky_lisbon.csv
#   'sd_cleaned_hq_clearsky'  -> *_sd_cleaned_hq_clearsky_lisbon.csv
# Without the start-date filter in clean_data.py, drop the 'sd_' prefix.
INPUT_SUFFIX: str = 'sd_cleaned'

# --- GADM level-2 boundaries --------------------------------------------------
SHAPEFILE_PATH: Path = BOUNDARIES_DIR / 'gadm41_PRT_2.shp'

# --- Municipality name in the NAME_2 column -----------------------------------
LISBON_NAME: str = 'Lisboa'

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

    The output suffix is the input suffix with '_lisbon' appended, so the
    output integrates seamlessly into the downstream FILE_SUFFIX convention.

    Examples (CAMPAIGN = 'standard')
    --------------------------------
    input_suffix='cleaned'             → j31_satellite_landuse_v2_cleaned_lisbon.csv
    input_suffix='cleaned_hq'          → j31_satellite_landuse_v2_cleaned_hq_lisbon.csv
    input_suffix='cleaned_clearsky'    → j31_satellite_landuse_v2_cleaned_clearsky_lisbon.csv
    input_suffix='cleaned_hq_clearsky' → j31_satellite_landuse_v2_cleaned_hq_clearsky_lisbon.csv

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
    out_suffix = f'{input_suffix}_lisbon'
    return output_dir / f'{FILE_STEM.format(sensor_id=sensor_id.lower())}_{out_suffix}.csv'


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the Lisbon spatial filter for all four sensors."""

    # ── Resolve sensor file dict from suffix ───────────────────────────────────
    sensor_files = _build_sensor_files(DATA_DIR, INPUT_SUFFIX)
    output_suffix = f'{INPUT_SUFFIX}_lisbon'

    # ── Print configuration header ─────────────────────────────────────────────
    print('=' * 70)
    print('  Lisbon Spatial Filter  —  v2.0')
    print('=' * 70)
    print(f'  Campaign       : {CAMPAIGN}')
    print(f'  Input suffix   : {INPUT_SUFFIX}')
    print(f'  Output suffix  : {output_suffix}')
    print(f'  Data directory : {DATA_DIR}')
    print()

    # ── Load and validate the Lisbon polygon ───────────────────────────────────
    if not SHAPEFILE_PATH.exists():
        raise FileNotFoundError(
            f'Shapefile not found: {SHAPEFILE_PATH}\n'
            'Download GADM Portugal shapefiles from https://gadm.org/ '
            'and place them in BOUNDARIES_DIR (see config.py).'
        )

    print(f'Loading shapefile : {SHAPEFILE_PATH.name}')
    gdf_muni = gpd.read_file(SHAPEFILE_PATH)

    # Locate the NAME_2 column (consistent with measurement_distribution_map.py)
    name_col = 'NAME_2' if 'NAME_2' in gdf_muni.columns else gdf_muni.columns[2]

    lisbon_gdf = gdf_muni[gdf_muni[name_col] == LISBON_NAME].copy()
    if lisbon_gdf.empty:
        lisbon_gdf = gdf_muni[
            gdf_muni[name_col].str.lower() == LISBON_NAME.lower()
        ].copy()
    if lisbon_gdf.empty:
        raise ValueError(
            f"Municipality '{LISBON_NAME}' not found in column '{name_col}'.\n"
            f"Available (first 30): {sorted(gdf_muni[name_col].unique())[:30]}\n"
            'Adjust LISBON_NAME in the configuration section.'
        )

    # Dissolve multi-polygon into a single geometry
    try:
        lisbon_polygon = lisbon_gdf.geometry.union_all()
    except AttributeError:
        lisbon_polygon = lisbon_gdf.geometry.unary_union

    bounds = lisbon_gdf.total_bounds   # (minx, miny, maxx, maxy)
    print(f'  Lisboa polygon loaded : {len(lisbon_gdf)} part(s), '
          f'bounds = [{bounds[0]:.4f}, {bounds[1]:.4f}, '
          f'{bounds[2]:.4f}, {bounds[3]:.4f}]')
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

        # ── Spatial join: keep only points within the Lisbon polygon ──────────
        geometry = [
            Point(lon, lat)
            for lon, lat in zip(df_valid[lon_col].values,
                                df_valid[lat_col].values)
        ]
        gdf_pts     = gpd.GeoDataFrame(df_valid, geometry=geometry, crs='EPSG:4326')
        inside_mask = gdf_pts.geometry.within(lisbon_polygon)

        # Use boolean mask on df_valid's index to avoid geometry column in output
        df_lisbon = df_valid.loc[inside_mask.values].copy()

        n_inside = len(df_lisbon)
        pct      = n_inside / n_total * 100 if n_total > 0 else 0.0
        print(f'  → {n_inside:,} rows inside Lisboa municipality '
              f'({pct:.1f}% of total)')

        # ── Write output ──────────────────────────────────────────────────────
        out_path = _build_output_path(OUTPUT_DIR, sensor, INPUT_SUFFIX)
        df_lisbon.to_csv(out_path, index=False)
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
    print(f'  Lisbon polygon    : {LISBON_NAME}  '
          f'(source: {SHAPEFILE_PATH.name})')
    print(f'  Output suffix     : {output_suffix}')
    print('─' * 70)
    print()
    print('Done.')
    print()
    print('Next step — set FILE_SUFFIX in downstream scripts to:')
    print(f"    FILE_SUFFIX = '{output_suffix}'")


if __name__ == '__main__':
    main()
