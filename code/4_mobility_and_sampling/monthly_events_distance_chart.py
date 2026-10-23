"""
================================================================================
Monthly Frequency and Travelled Distance — City Centre vs. Full Dataset
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Monthly frequency of registered events within the city centre (wide grey
columns) and for the entire dataset (white columns), with the travelled
distance on the secondary axis, within the city centre (narrow dark blue
columns) and for the entire dataset (light blue).

Two figures are always produced at each run:

  Figure 1 — Total distance (medc_monthly_events_distance_total_*.png)
  Right axis: total monthly distance [thousand km].
  Direct sum of Haversine distances per month. Comparable across months
  only when the number of available data points is similar.
  Intended for use with a complete, balanced dataset.

  Figure 2 — Normalised distance (medc_monthly_events_distance_normalised_*.png)
  Right axis: mean daily distance per sensor [km / sensor-day].
  = dist_km / n_unique_sensor_days
  where n_unique_sensor_days is the count of unique (sensor, date) pairs.
  Same normalisation as the solar-autonomy scripts
  (solar_autonomy_losses*.py), directly comparable to their solar driving
  range figures.
  Months with fewer than MIN_SENSOR_DAYS unique sensor-days are left blank.

Distance calculation
--------------------
  The Haversine distance is computed between each row and the previous row,
  but ONLY when the time gap between them is ≤ MAX_GAP_MINUTES (default: 2).
  Gaps larger than this threshold indicate a recording interruption
  (overnight stop, underground parking, sensor off) and are set to zero.
  This prevents spurious large distances caused by jumps between the last
  GPS position of one recording session and the first of the next.
  This is consistent with vehicle_movement_analysis.py which uses the same
  consecutive-pair logic with a threshold for parked/moving classification.

Plausibility filter
-------------------
  After gap-corrected distances are computed, each (sensor, date) pair whose
  total daily distance falls below MIN_KM_PER_SENSOR_DAY (default: 3.0 km)
  is classified as non-vehicular and excluded from all aggregations.
  This removes days when the sensor was stationary (roof calibration, bench
  test, storage) but still recording valid irradiance data that would otherwise
  pass the upstream clean_data.py filters.
  Excluded sensor-days are reported in the console and counted in dedicated
  columns of the monthly CSV (n_excluded_sdays_city, n_excluded_sdays_full)
  for traceability.

Bar layout (identical in both figures)
---------------------------------------
  Wide bars  (event count)  : centred on each month tick, width ≈ 1 tick
                              spacing so adjacent months touch.
                              White bar  (full dataset) in background.
                              Grey bar   (city centre)  on top (foreground).
  Narrow bars (distance)    : centred on the same tick, overlaid on the
                              wide bars (higher zorder).
                              Light-blue bar (full dataset) in background.
                              Dark-blue bar  (city centre)  on top.

  Both y-axes are colour-coded:
    Left  axis (events)   — dark grey label + ticks
    Right axis (distance) — steel blue label + ticks

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
The script always compares TWO variants of the selected campaign: the full
dataset (FILE_SUFFIX_FULL / TUKTUK_FILE_SUFFIX_FULL, e.g. "sd_cleaned" from
clean_data.py) and the city centre (FILE_SUFFIX_CITY /
TUKTUK_FILE_SUFFIX_CITY, e.g. "sd_cleaned_lisbon" from filter_lisbon.py).

    input  : one file per sensor and variant
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <full variant tag> /
             medc_monthly_events_distance_total_<CAMPAIGN>_<tag>.png
             medc_monthly_events_distance_normalised_<CAMPAIGN>_<tag>.png
             medc_monthly_events_distance_summary_<CAMPAIGN>_<tag>.csv
                 monthly event counts, distances, sensor-days, excluded
                 sensor-days and city / full ratios
             medc_seasonal_events_distance_summary_<CAMPAIGN>_<tag>.csv
                 seasonal aggregation (DJF / MAM / JJA / SON)

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 4.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input suffixes: full dataset vs city centre (standard campaign) ---------
#   FULL : e.g. "sd_cleaned"         (clean_data.py)
#   CITY : e.g. "sd_cleaned_lisbon"  (filter_lisbon.py)
FILE_SUFFIX_FULL: str = "sd_cleaned"
FILE_SUFFIX_CITY: str = "sd_cleaned_lisbon"

# --- Input suffixes: full dataset vs city centre (tuk-tuk campaign) ----------
TUKTUK_FILE_SUFFIX_FULL: str = "sd_cleaned"
TUKTUK_FILE_SUFFIX_CITY: str = "sd_cleaned_lisbon"

# --- Data and output directories ----------------------------------------------
DATA_DIR:        Path = STANDARD_DIR     # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "4_mobility_and_sampling" / "medc_outputs"

SENSOR_COLS: dict[str, dict[str, str]] = {
    'J30': {'lat': 'J30 lat', 'lon': 'J30 lon'},
    'J31': {'lat': 'J31 lat', 'lon': 'J31 lon'},
    'J32': {'lat': 'J32 lat', 'lon': 'J32 lon'},
    'J33': {'lat': 'J33 lat', 'lon': 'J33 lon'},
}

DATE_COL     = 'Human readable date (Europe/London)'

# Maximum time gap between two consecutive records for a distance to be
# computed between them. Gaps larger than this value indicate a recording
# interruption (overnight, underground parking, sensor restart) and the
# distance is set to zero to avoid spurious jumps.
# Data resolution is 1 minute; 2 minutes allows for occasional missed records.
MAX_GAP_MINUTES: int = 2

# Minimum unique (sensor, date) pairs for a normalised bar to be shown.
# Same value as MIN_SENSOR_DAYS in solar_autonomy_losses*.py.
MIN_SENSOR_DAYS: int = 5

# Plausibility filter — minimum distance per sensor-day [km].
# A sensor-day whose total Haversine distance falls below this threshold is
# considered non-vehicular (sensor stored, bench test, roof calibration) and
# is excluded from all event counts, distance sums, and sensor-day counts.
# The threshold of 3 km is a conservative lower bound: legitimate urban driving
# sessions (even short intra-city trips) consistently exceed this value, while
# GPS noise on a stationary sensor typically produces < 0.5 km/day.
# Set to 0.0 to disable the filter entirely.
MIN_KM_PER_SENSOR_DAY: float = 3.0

R_EARTH_KM: float = 6371.0
LOCAL_TZ:   str   = 'Europe/Lisbon'

OUTPUT_DPI: int = 300

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# All paths are derived from DATA_DIR, OUTPUT_BASE_DIR, FILE_SUFFIX_FULL and
# FILE_SUFFIX_CITY.  Do NOT edit the variables below — change the suffixes above.

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build sensor CSV paths for a given variant suffix, for either campaign.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    sensors = ['J30', 'J31', 'J32', 'J33']
    if campaign == "standard":
        return {
            s: data_dir / f"{s.lower()}_satellite_landuse_v2_{suffix}.csv"
            for s in sensors
        }
    elif campaign == "tuktuk":
        sep = f"_{suffix}" if suffix else ""
        return {
            s: tuktuk_data_dir / f"{s.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv"
            for s in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag_full() -> str:
    """Return the active FULL-variant tag for the current campaign."""
    return FILE_SUFFIX_FULL if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX_FULL


def _variant_tag_city() -> str:
    """Return the active CITY-variant tag for the current campaign."""
    return FILE_SUFFIX_CITY if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX_CITY


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Dedicated sub-folder per campaign/variant so outputs never overwrite each other."""
    return base_dir / campaign / variant_tag


def _build_output_path(out_dir: Path, base_name: str,
                       campaign: str, variant_tag: str, extension: str) -> Path:
    """Embed the active campaign and variant tag in every output filename."""
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolve all paths at startup — change CAMPAIGN / the FULL / CITY suffixes above
_VARIANT_TAG_FULL: str = _variant_tag_full()
_VARIANT_TAG_CITY: str = _variant_tag_city()

SENSOR_FILES_FULL: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG_FULL
)
SENSOR_FILES_CITY: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG_CITY
)
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG_FULL)
OUTPUT_PNG_TOTAL = _build_output_path(
    OUTPUT_DIR, 'medc_monthly_events_distance_total',
    CAMPAIGN, _VARIANT_TAG_FULL, 'png')
OUTPUT_PNG_NORM  = _build_output_path(
    OUTPUT_DIR, 'medc_monthly_events_distance_normalised',
    CAMPAIGN, _VARIANT_TAG_FULL, 'png')
OUTPUT_CSV          = _build_output_path(
    OUTPUT_DIR, 'medc_monthly_events_distance_summary',
    CAMPAIGN, _VARIANT_TAG_FULL, 'csv')
OUTPUT_CSV_SEASONAL = _build_output_path(
    OUTPUT_DIR, 'medc_seasonal_events_distance_summary',
    CAMPAIGN, _VARIANT_TAG_FULL, 'csv')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# STYLE CONSTANTS
# ==============================================================================

COL_EVENTS_CITY = '#9E9E9E'
COL_EVENTS_FULL = '#FFFFFF'
COL_DIST_CITY   = '#1A237E'
COL_DIST_FULL   = '#90CAF9'
EDGE_COLOR      = '#333333'
EDGE_LW         = 0.5
COL_AX_EVENTS   = '#444444'
COL_AX_DIST     = '#1565C0'
FONT_SIZE_LABEL  = 10
FONT_SIZE_TICK   =  9
FONT_SIZE_LEGEND =  8.5

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def haversine_km(lat1, lon1, lat2, lon2, R=R_EARTH_KM):
    """
    Compute the great-circle distance in kilometres between consecutive GPS
    points using the Haversine formula.

    Parameters
    ----------
    lat1, lon1 : array-like  — origin coordinates (decimal degrees, WGS-84)
    lat2, lon2 : array-like  — destination coordinates (decimal degrees, WGS-84)
    R          : float       — mean Earth radius in km (default: 6371.0, IAU 2012)

    Returns
    -------
    numpy.ndarray of float — distances in kilometres
    """
    lat1_r = np.radians(np.asarray(lat1, dtype=float))
    lat2_r = np.radians(np.asarray(lat2, dtype=float))
    dlat   = lat2_r - lat1_r
    dlon   = np.radians(np.asarray(lon2, dtype=float)
                        - np.asarray(lon1, dtype=float))
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2)
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_sensor_data(sensor_name, file_path, lat_col, lon_col,
                     dataset_label=''):
    """
    Load and pre-process one sensor CSV file produced by the upstream pipeline.

    All quality filters (land_use, sun_elev, rad_calib, FCUL exclusion zone,
    Perez-Astudillo QC) are already applied by clean_data.py and, where relevant,
    filter_lisbon.py.  This function performs no additional filtering; it only:
      - strips anonymous 'Unnamed' index columns left by pd.to_csv(),
      - parses the timestamp and converts it to the local timezone,
      - adds 'month', 'date', and 'sensor' columns required for aggregation,
      - coerces GPS coordinates to float and drops rows with missing coordinates.

    Parameters
    ----------
    sensor_name   : str   — short identifier, e.g. 'J31'
    file_path     : Path  — path to the CSV file
    lat_col       : str   — latitude column name
    lon_col       : str   — longitude column name
    dataset_label : str   — display label for console messages

    Returns
    -------
    pd.DataFrame sorted by local timestamp, or None if file missing / empty.
    """
    if not file_path.is_file():
        print(f'    [WARNING] File not found — skipping {sensor_name} '
              f'({dataset_label}): {file_path.name}')
        return None

    df = pd.read_csv(file_path, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    df['datetime_local'] = (
        pd.to_datetime(df[DATE_COL], utc=True, errors='coerce')
        .dt.tz_convert(LOCAL_TZ)
    )
    df = df.dropna(subset=['datetime_local'])
    df['month']  = df['datetime_local'].dt.month
    df['date']   = df['datetime_local'].dt.date
    df['sensor'] = sensor_name

    for col in [lat_col, lon_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_raw = len(df)
    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    print(f'    {sensor_name} [{dataset_label}]: {n_raw:>8,} rows loaded'
          + (f' ({n_raw - len(df):,} dropped — missing GPS)'
             if n_raw != len(df) else ''))

    if len(df) == 0:
        print(f'    [WARNING] No valid data — skipping {sensor_name}.')
        return None

    df = df.sort_values('datetime_local').reset_index(drop=True)
    return df


def compute_distances_km(df, lat_col, lon_col):
    """
    Compute the Haversine distance (km) between each row and the previous row,
    setting the distance to zero whenever a segment boundary is detected.

    A segment boundary occurs when ANY of the following conditions is true
    for a pair of consecutive rows:

      1. Time gap > MAX_GAP_MINUTES : recording interruption (overnight stop,
         underground parking, sensor restart, missed records).
      2. Different sensor : the DataFrame must contain only one sensor's data
         (process_dataset calls this function once per sensor file), but this
         guard is kept as an explicit safety net against future refactoring
         that might concatenate multiple sensors before calling this function.
      3. Different local date : catches overnight continuity that might slip
         through condition 1 if MAX_GAP_MINUTES were ever set very large.
         Also explicitly prevents cross-day GPS jumps when a sensor records
         late into the evening and resumes early the next morning with a gap
         just above the threshold.

    In all three cases, no actual continuous travel occurred between the two
    rows and computing a Haversine distance would produce a spurious value.

    Parameters
    ----------
    df      : pd.DataFrame — single-sensor DataFrame, sorted by local time,
                             with columns 'datetime_local', 'date', 'sensor'
    lat_col : str          — latitude column name
    lon_col : str          — longitude column name

    Returns
    -------
    numpy.ndarray of float, shape (len(df),)
        Distances in kilometres. Values at segment boundaries are 0.
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=float)

    lat    = df[lat_col].values
    lon    = df[lon_col].values
    ts     = df['datetime_local'].values
    dates  = df['date'].values
    sensor = df['sensor'].values

    # Condition 1 — time gap too large
    gap_min = pd.Series(ts).diff().dt.total_seconds().fillna(0).values / 60.0
    break_time   = gap_min > MAX_GAP_MINUTES

    # Condition 2 — sensor change (safety net; should never fire in normal use)
    break_sensor = np.concatenate([[False], sensor[1:] != sensor[:-1]])

    # Condition 3 — date change
    break_date   = np.concatenate([[False], dates[1:] != dates[:-1]])

    segment_break = break_time | break_sensor | break_date

    # Haversine distance for all consecutive pairs
    raw_dist = np.zeros(len(df), dtype=float)
    raw_dist[1:] = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])

    # Zero out all segment boundaries
    raw_dist[segment_break] = 0.0

    # Console diagnostics (breakdown by cause)
    n_time   = int(break_time.sum())
    n_sensor = int(break_sensor.sum())
    n_date   = int(break_date.sum())
    n_total  = int(segment_break.sum())
    if n_total > 0:
        print(f'      → {n_total:,} segment breaks zeroed '
              f'(time gap: {n_time:,} | date change: {n_date:,} '
              f'| sensor change: {n_sensor:,})')

    return raw_dist


def process_dataset(sensor_files, label):
    """
    Load all sensors for one dataset, apply the plausibility filter, compute
    gap-corrected distances, and accumulate monthly event counts, distances,
    and unique sensor-day counts.

    Plausibility filter
    -------------------
    After computing gap-corrected Haversine distances, each (sensor, date) pair
    whose total daily distance is below MIN_KM_PER_SENSOR_DAY is excluded from
    all aggregations.  This removes days when the sensor was stationary (roof
    calibration, bench test, storage) but still recording irradiance.  The
    number of excluded sensor-days is tracked per month for CSV reporting.
    Set MIN_KM_PER_SENSOR_DAY = 0.0 in the configuration to disable.

    Parameters
    ----------
    sensor_files : dict[str, Path]  — sensor name → CSV file path
    label        : str              — 'full dataset' or 'city centre'

    Returns
    -------
    events        : dict[int, int]   — event count (minutes) per month,
                                       after plausibility filter
    dists         : dict[int, float] — total gap-corrected distance [km] per month
    sensor_days   : dict[int, int]   — valid (sensor, date) pairs per month
    excluded_days : dict[int, int]   — sensor-days removed by plausibility filter
    loaded        : list[str]        — sensors successfully loaded
    """
    MONTH_RANGE   = list(range(1, 13))
    events        = {m: 0   for m in MONTH_RANGE}
    dists         = {m: 0.0 for m in MONTH_RANGE}
    sday_sets:     dict[int, set] = {m: set() for m in MONTH_RANGE}
    excl_sets:     dict[int, set] = {m: set() for m in MONTH_RANGE}
    loaded        = []

    for sensor_name, file_path in sensor_files.items():
        lat_col = SENSOR_COLS[sensor_name]['lat']
        lon_col = SENSOR_COLS[sensor_name]['lon']

        df = load_sensor_data(sensor_name, file_path,
                              lat_col, lon_col, label)
        if df is None:
            continue

        df['dist_km'] = compute_distances_km(df, lat_col, lon_col)

        # ── Plausibility filter ────────────────────────────────────────────────
        # Compute total distance per (sensor, date) pair, then flag days that
        # fall below the minimum threshold as non-vehicular.
        daily_dist = (
            df.groupby('date')['dist_km']
            .sum()
            .rename('daily_dist_km')
        )
        df = df.join(daily_dist, on='date')
        plausible_mask = df['daily_dist_km'] >= MIN_KM_PER_SENSOR_DAY

        n_excl_rows = int((~plausible_mask).sum())
        n_excl_days = int(df.loc[~plausible_mask, 'date'].nunique())
        if n_excl_days > 0:
            print(f'      [plausibility] {sensor_name} [{label}]: '
                  f'{n_excl_days} sensor-day(s) excluded '
                  f'(< {MIN_KM_PER_SENSOR_DAY} km/day) — '
                  f'{n_excl_rows:,} rows removed')
            # Track excluded (sensor, date) pairs per month for CSV reporting
            excl_df = df.loc[~plausible_mask, ['month', 'date']].drop_duplicates()
            for _, row in excl_df.iterrows():
                excl_sets[row['month']].add((sensor_name, row['date']))

        df = df[plausible_mask].drop(columns='daily_dist_km').reset_index(drop=True)

        # ── Monthly aggregation (plausibility-filtered data only) ──────────────
        for month in MONTH_RANGE:
            mask = df['month'] == month
            events[month] += int(mask.sum())
            dists[month]  += float(df.loc[mask, 'dist_km'].sum())
            for d in df.loc[mask, 'date']:
                sday_sets[month].add((sensor_name, d))

        loaded.append(sensor_name)

    sensor_days   = {m: len(sday_sets[m]) for m in MONTH_RANGE}
    excluded_days = {m: len(excl_sets[m]) for m in MONTH_RANGE}

    return events, dists, sensor_days, excluded_days, loaded


def km_per_sensor_day(dist_km, n_sensor_days):
    """
    Compute mean daily distance per sensor [km / sensor-day].

    Returns NaN when there are fewer than MIN_SENSOR_DAYS unique sensor-days
    to avoid unreliable estimates, consistent with solar_autonomy_losses*.py.

    Parameters
    ----------
    dist_km       : float — total gap-corrected distance [km]
    n_sensor_days : int   — unique (sensor, date) pairs

    Returns
    -------
    float — km per sensor-day, or NaN if insufficient data
    """
    if n_sensor_days < MIN_SENSOR_DAYS:
        return float('nan')
    return dist_km / n_sensor_days


def draw_figure(ev_city_k, ev_full_k,
                rax_city, rax_full,
                rax_ylabel, legend_dist_city, legend_dist_full,
                output_path):
    """
    Build and save one bar-chart figure.

    Parameters
    ----------
    ev_city_k        : np.ndarray — event count city centre [thousands]
    ev_full_k        : np.ndarray — event count full dataset [thousands]
    rax_city         : np.ndarray — right-axis values, city centre
                                    (NaN = masked month → blank bar + dash)
    rax_full         : np.ndarray — right-axis values, full dataset
    rax_ylabel       : str        — right y-axis label
    legend_dist_city : str        — legend label for city distance bars
    legend_dist_full : str        — legend label for full distance bars
    output_path      : Path       — destination PNG file
    """
    MONTH_RANGE = list(range(1, 13))
    x        = np.arange(len(MONTH_RANGE))
    W_WIDE   = 1
    W_NARROW = 0.33

    rax_city_plot = np.where(np.isnan(rax_city), 0.0, rax_city)
    rax_full_plot = np.where(np.isnan(rax_full), 0.0, rax_full)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()

    ax1.bar(x, ev_full_k, width=W_WIDE,
            color=COL_EVENTS_FULL, edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
            zorder=2)
    ax1.bar(x, ev_city_k, width=W_WIDE,
            color=COL_EVENTS_CITY, edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
            zorder=3)
    ax2.bar(x, rax_full_plot, width=W_NARROW,
            color=COL_DIST_FULL, edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
            zorder=4)
    ax2.bar(x, rax_city_plot, width=W_NARROW,
            color=COL_DIST_CITY, edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
            zorder=5)

    ax2.set_ylim(bottom=0)
    ymax = ax2.get_ylim()[1] or 1.0
    for i in range(len(MONTH_RANGE)):
        if np.isnan(rax_city[i]) or np.isnan(rax_full[i]):
            ax2.text(i, ymax * 0.02, '—', ha='center', va='bottom',
                     fontsize=8, color='#AAAAAA')

    ax1.set_ylabel('Frequency of events [thousands]',
                   fontsize=FONT_SIZE_LABEL, color=COL_AX_EVENTS)
    ax1.tick_params(axis='y', labelsize=FONT_SIZE_TICK,
                    colors=COL_AX_EVENTS, which='both')
    ax1.spines['left'].set_edgecolor(COL_AX_EVENTS)
    ax1.set_ylim(bottom=0)

    ax2.set_ylabel(rax_ylabel,
                   fontsize=FONT_SIZE_LABEL, color=COL_AX_DIST)
    ax2.tick_params(axis='y', labelsize=FONT_SIZE_TICK,
                    colors=COL_AX_DIST, which='both')
    ax2.spines['right'].set_edgecolor(COL_AX_DIST)

    ax1.set_xticks(x)
    ax1.set_xticklabels(MONTH_NAMES, fontsize=FONT_SIZE_TICK)
    ax1.set_xlim(x[0] - W_WIDE / 2 - 0.02, x[-1] + W_WIDE / 2 + 0.02)
    ax1.grid(axis='y', which='major', color='#DDDDDD', linewidth=0.5, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=COL_EVENTS_CITY, edgecolor=EDGE_COLOR,
                       linewidth=EDGE_LW, label='Events — city centre'),
        mpatches.Patch(facecolor=COL_EVENTS_FULL, edgecolor=EDGE_COLOR,
                       linewidth=EDGE_LW, label='Events — full dataset'),
        mpatches.Patch(facecolor=COL_DIST_CITY,   edgecolor=EDGE_COLOR,
                       linewidth=EDGE_LW, label=legend_dist_city),
        mpatches.Patch(facecolor=COL_DIST_FULL,   edgecolor=EDGE_COLOR,
                       linewidth=EDGE_LW, label=legend_dist_full),
    ]
    ax1.legend(handles=legend_handles, loc='upper left',
               fontsize=FONT_SIZE_LEGEND, frameon=True,
               framealpha=0.92, edgecolor='#AAAAAA')

    # Embed the active CAMPAIGN + FULL variant tag in the figure title
    fig.suptitle(f'[{CAMPAIGN} / {_VARIANT_TAG_FULL}]', x=0.99, y=0.99,
                 ha='right', va='top',
                 fontsize=7, color='#AAAAAA', style='italic')
    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load both variants, aggregate per month and season, write all outputs."""
    print('=' * 70)
    print('MONTHLY EVENTS & DISTANCE — City Centre vs. Full Dataset')
    print(f'  CAMPAIGN     : {CAMPAIGN}')
    print(f'  Full variant : {_VARIANT_TAG_FULL}')
    print(f'  City variant : {_VARIANT_TAG_CITY}')
    print('=' * 70)
    print(f'\n  Output dir   : {OUTPUT_DIR}')

    MONTH_RANGE = list(range(1, 13))

    print(f'\n[1/5] Loading full dataset (*_{_VARIANT_TAG_FULL}.csv) ...')
    events_full, dist_full, sdays_full, excl_full, loaded_full = process_dataset(
        SENSOR_FILES_FULL, 'full dataset')

    print(f'\n[2/5] Loading city-centre dataset (*_{_VARIANT_TAG_CITY}.csv) ...')
    events_city, dist_city, sdays_city, excl_city, loaded_city = process_dataset(
        SENSOR_FILES_CITY, 'city centre')

    if not loaded_full:
        raise RuntimeError('No full-dataset files found.')
    if not loaded_city:
        raise RuntimeError('No city-centre files found. '
                           'Run filter_lisbon.py first.')

    print('\n[3/5] Monthly summary:')
    hdr = (f"  {'Month':<5} {'Ev_City':>9} {'Ev_Full':>9} "
           f"{'SDays_C':>8} {'SDays_F':>8} "
           f"{'Excl_C':>7} {'Excl_F':>7} "
           f"{'km/sday_C':>10} {'km/sday_F':>10}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for m in MONTH_RANGE:
        rc = km_per_sensor_day(dist_city[m], sdays_city[m])
        rf = km_per_sensor_day(dist_full[m], sdays_full[m])
        rc_s = f'{rc:.1f}' if not np.isnan(rc) else '  —'
        rf_s = f'{rf:.1f}' if not np.isnan(rf) else '  —'
        excl_c = excl_city[m]
        excl_f = excl_full[m]
        flag = '  ← excluded' if (excl_c > 0 or excl_f > 0) else ''
        print(f'  {MONTH_NAMES[m-1]:<5} '
              f'{events_city[m]:>9,} {events_full[m]:>9,} '
              f'{sdays_city[m]:>8,} {sdays_full[m]:>8,} '
              f'{excl_c:>7,} {excl_f:>7,} '
              f'{rc_s:>10} {rf_s:>10}{flag}')

    print('\n[4/5] Exporting CSV summaries ...')
    # ── Monthly summary CSV ────────────────────────────────────────────────────
    rows = []
    for m in MONTH_RANGE:
        ef  = events_full[m];  ec  = events_city[m]
        df_ = dist_full[m];    dc  = dist_city[m]
        sdf = sdays_full[m];   sdc = sdays_city[m]
        xf  = excl_full[m];    xc  = excl_city[m]
        rc  = km_per_sensor_day(dc,  sdc)
        rf  = km_per_sensor_day(df_, sdf)
        rows.append({
            'month':                        m,
            'month_name':                   MONTH_NAMES[m - 1],
            'events_city':                  ec,
            'events_full':                  ef,
            'events_city_frac':             round(ec / ef,   4) if ef   else None,
            'events_city_k':                round(ec  / 1_000, 3),
            'events_full_k':                round(ef  / 1_000, 3),
            'sensor_days_city':             sdc,
            'sensor_days_full':             sdf,
            # Sensor-days excluded by the plausibility filter (< MIN_KM_PER_SENSOR_DAY)
            'n_excluded_sdays_city':        xc,
            'n_excluded_sdays_full':        xf,
            'dist_city_km':                 round(dc,  1),
            'dist_full_km':                 round(df_, 1),
            'dist_city_frac':               round(dc / df_, 4) if df_  else None,
            'dist_city_k':                  round(dc  / 1_000, 3),
            'dist_full_k':                  round(df_ / 1_000, 3),
            'dist_city_km_per_sday':        round(rc, 2) if not np.isnan(rc) else None,
            'dist_full_km_per_sday':        round(rf, 2) if not np.isnan(rf) else None,
        })
    df_monthly = pd.DataFrame(rows)
    df_monthly.to_csv(OUTPUT_CSV, index=False)
    print(f'\n  CSV saved: {OUTPUT_CSV}')

    # ── Seasonal aggregation CSV ───────────────────────────────────────────────
    # Meteorological seasons (Southern Europe convention):
    #   DJF = Dec–Jan–Feb  |  MAM = Mar–Apr–May
    #   JJA = Jun–Jul–Aug  |  SON = Sep–Oct–Nov
    season_map = {
        1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON',
        12: 'DJF',
    }
    df_monthly['season'] = df_monthly['month'].map(season_map)
    season_order = ['DJF', 'MAM', 'JJA', 'SON']

    sea_rows = []
    for season in season_order:
        s = df_monthly[df_monthly['season'] == season]
        ec_s  = int(s['events_city'].sum())
        ef_s  = int(s['events_full'].sum())
        dc_s  = float(s['dist_city_km'].sum())
        df_s  = float(s['dist_full_km'].sum())
        sdc_s = int(s['sensor_days_city'].sum())
        sdf_s = int(s['sensor_days_full'].sum())
        rc_s  = km_per_sensor_day(dc_s, sdc_s)
        rf_s  = km_per_sensor_day(df_s, sdf_s)
        sea_rows.append({
            'season':                   season,
            'events_city':              ec_s,
            'events_full':              ef_s,
            'events_city_frac':         round(ec_s / ef_s, 4) if ef_s else None,
            'sensor_days_city':         sdc_s,
            'sensor_days_full':         sdf_s,
            'dist_city_km':             round(dc_s, 1),
            'dist_full_km':             round(df_s, 1),
            'dist_city_frac':           round(dc_s / df_s, 4) if df_s else None,
            'dist_city_km_per_sday':    round(rc_s, 2) if not np.isnan(rc_s) else None,
            'dist_full_km_per_sday':    round(rf_s, 2) if not np.isnan(rf_s) else None,
        })
    pd.DataFrame(sea_rows).to_csv(OUTPUT_CSV_SEASONAL, index=False)
    print(f'  CSV saved: {OUTPUT_CSV_SEASONAL}')

    ev_city_k = np.array([events_city[m] / 1_000 for m in MONTH_RANGE])
    ev_full_k = np.array([events_full[m] / 1_000 for m in MONTH_RANGE])

    rax_city_total = np.array([dist_city[m] / 1_000 for m in MONTH_RANGE])
    rax_full_total = np.array([dist_full[m] / 1_000 for m in MONTH_RANGE])

    rax_city_norm = np.array([km_per_sensor_day(dist_city[m], sdays_city[m])
                               for m in MONTH_RANGE])
    rax_full_norm = np.array([km_per_sensor_day(dist_full[m], sdays_full[m])
                               for m in MONTH_RANGE])

    print('\n[5/5] Drawing figures ...')
    # Note: figures show only plausibility-filtered data (non-vehicular
    # sensor-days have already been removed from events_* and dist_*).

    draw_figure(
        ev_city_k, ev_full_k,
        rax_city_total, rax_full_total,
        rax_ylabel       = 'Monthly km [thousands]',
        legend_dist_city = 'Distance — city centre',
        legend_dist_full = 'Distance — full dataset',
        output_path      = OUTPUT_PNG_TOTAL,
    )

    draw_figure(
        ev_city_k, ev_full_k,
        rax_city_norm, rax_full_norm,
        rax_ylabel       = 'Daily travelled distance [km / sensor-day]',
        legend_dist_city = 'Distance — city centre',
        legend_dist_full = 'Distance — full dataset',
        output_path      = OUTPUT_PNG_NORM,
    )

    print('=' * 70)


if __name__ == '__main__':
    main()