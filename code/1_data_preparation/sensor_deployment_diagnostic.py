"""
================================================================================
Sensor Deployment Diagnostic — Pre-deployment Phase Detection
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Analyses the daily behaviour of each sensor over calendar time to identify
the periods during which it was NOT mounted on a vehicle (rooftop
calibration, bench test, storage), in order to choose the deployment start
date used by the start-date filter of clean_data.py (START_DATE). Works for
both campaigns (CAMPAIGN = 'standard' | 'tuktuk').

--------------------------------------------------------------------------------
Daily indicators
--------------------------------------------------------------------------------

  1. Distance [km/day]
       Total Haversine distance with gap-correction (gaps > MAX_GAP_MIN zeroed).
       Expected on vehicle : highly variable, typically 5-100 km/day.
       Expected on rooftop : near-zero (GPS drift only, < 0.5 km/day).

  2. Fraction of time near FCUL [%]
       Share of rows where the GPS coordinate lies within EXCLUSION_RADIUS_KM
       of the FCUL reference point (rooftop installation site).
       Expected on vehicle : low outside campus trips, < 20 % on most days.
       Expected on rooftop : near 100 % (sensor never leaves the roof).

  3. Daily measurement duration [hours]
       Number of valid rows (minutes) divided by 60.
       Expected on vehicle : variable, reflects trip schedule (2-12 h/day).
       Expected in storage : 0 h (sensor off) or continuous 12-15 h (sensor
       always on outdoors).

  4. Mean speed while moving [km/h]
       Mean of per-minute speeds (dist_km × 60) for rows classified as moving
       (distance ≥ THRESHOLD_M metres in one minute).
       Expected on vehicle : 10-60 km/h depending on urban / highway mix.
       Expected on rooftop : near-zero (< 1 km/h — GPS noise only).

  5. Ratio rad_calib / ghi (daily median)
       Ratio of the sensor irradiance ('<SENSOR> Rad calib') to the CAMS GHI,
       over rows with ghi > 10 W/m² and rad_calib > 0 (at least 10 per day).
       Expected on vehicle in urban canyons : typically < 1 (shading effect).
       Expected on rooftop (no shading) : close to 1.0.
       Note: a ratio persistently at 1.0 is a strong signal for a fixed,
       unshaded installation.

  6. Fraction of time moving [%]
       Share of rows classified as moving (Haversine >= THRESHOLD_M).
       Expected on vehicle : 10-40 % (urban driving + parking).
       Expected on rooftop : near 0 % (no displacement).

A grey band on each panel marks the "normal vehicle-mounted" range
(REFERENCE_BANDS). Days outside the bands on several indicators at once are
candidates for exclusion; the daily CSV gives the exact values for choosing
the start date.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <stem>_<FILE_SUFFIX>.csv in the campaign data directory, with
             <stem> = <sensor_id>_satellite_landuse_v2 (standard) or
                      <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm (tuk-tuk)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <FILE_SUFFIX> /
                 sddiag_<CAMPAIGN>_<sensor_id>_<FILE_SUFFIX>.png
                 sddiag_<CAMPAIGN>_<sensor_id>_daily_<FILE_SUFFIX>.csv

Use the '_cleaned' files produced by clean_data.py with the start-date filter
DISABLED (ENABLE_START_DATE_FILTER = False): the '_sd' files and the spatial
subsets (filter_lisbon.py, filter_region.py) remove precisely the rooftop
phase this script is meant to detect.

Requirements
------------
    pip install numpy pandas matplotlib

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
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = 'standard'   # 'standard' | 'tuktuk'
if CAMPAIGN not in ('standard', 'tuktuk'):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directory (one per campaign) ---------------------------------------
_DIRS = {
    'standard': STANDARD_DIR,
    'tuktuk':   TUKTUK_DIR,
}
DATA_DIR: Path = _DIRS[CAMPAIGN]

# --- File stem per campaign (stem of the clean_data.py outputs) ---------------
_STEMS = {
    'standard': '{sensor_id}_satellite_landuse_v2',
    'tuktuk':   '{sensor_id}_tuktuk_ghi_cams_cos_clc_bh_sw_norm',
}
FILE_STEM: str = _STEMS[CAMPAIGN]

# --- Input suffix -------------------------------------------------------------
# '_cleaned' files of clean_data.py WITHOUT the start-date filter and before
# any spatial filter, so that the rooftop phase is still present.
FILE_SUFFIX: str = 'cleaned'

# --- Sensors (missing files are skipped) --------------------------------------
# Column names are the same in both campaigns after normalize_tuktuk.py.
SENSOR_IDS: list[str] = ['J30', 'J31', 'J32', 'J33']
SENSOR_COLS: dict[str, dict[str, str]] = {
    s: {'lat': f'{s} lat', 'lon': f'{s} lon', 'rad_calib': f'{s} Rad calib'}
    for s in SENSOR_IDS
}

DATE_COL = 'Human readable date (Europe/London)'
LOCAL_TZ  = 'Europe/Lisbon'

# --- FCUL reference point (rooftop installation site) and radius --------------
FCUL_LAT:            float = 38.7565
FCUL_LON:            float = -9.1564
EXCLUSION_RADIUS_KM: float = 0.200      # 200 m

# Movement threshold: a consecutive-row displacement >= this value [metres]
# classifies the transition as "moving".
THRESHOLD_M: float = 30.0

# Maximum time gap between consecutive rows for distance to be non-zero [min].
# Gaps above this are set to zero (parked / sensor off).
MAX_GAP_MIN: float = 2.0

# Reference bands — "normal vehicle-mounted" expected range per indicator.
# Each entry is (low, high).  Days falling outside these ranges across several
# indicators are likely non-vehicular.
REFERENCE_BANDS: dict[str, tuple] = {
    'dist_km':          (3.0,  500.0),   # km/day
    'fcul_frac_pct':    (0.0,   60.0),   # %
    'duration_h':       (1.0,   16.0),   # hours
    'mean_speed_kmh':   (5.0,   80.0),   # km/h (moving only)
    'rad_ratio':        (0.3,    0.98),  # rad_calib / ghi
    'moving_frac_pct':  (2.0,   60.0),   # %
}

# Rolling window for smoothing trend lines [days]
ROLLING_DAYS: int = 7

# Minimum rows per day for the day to be included (avoids single-row artefacts)
MIN_ROWS_PER_DAY: int = 10

# --- Outputs ------------------------------------------------------------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "1_data_preparation" / "sddiag_outputs"
OUTPUT_DPI = 200

# ==============================================================================
# PATH HELPERS
# ==============================================================================

def _build_sensor_files(data_dir: Path, suffix: str) -> dict[str, Path]:
    """Input CSV path per sensor: <FILE_STEM>_<suffix>.csv."""
    return {
        s: data_dir / f'{FILE_STEM.format(sensor_id=s.lower())}_{suffix}.csv'
        for s in SENSOR_IDS
    }


def _build_output_dir(base_dir: Path, campaign: str, suffix: str) -> Path:
    """Create and return OUTPUT_BASE_DIR / <campaign> / <suffix>."""
    out = base_dir / campaign / suffix
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_output_path(out_dir: Path, stem: str, suffix: str,
                       ext: str) -> Path:
    """Output file path: <stem>_<suffix>.<ext>."""
    return out_dir / f'{stem}_{suffix}.{ext}'


SENSOR_FILES = _build_sensor_files(DATA_DIR, FILE_SUFFIX)
OUTPUT_DIR   = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, FILE_SUFFIX)

# ==============================================================================
# EARTH / GEOMETRY HELPERS
# ==============================================================================

R_EARTH_KM = 6371.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine great-circle distance [km]."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R_EARTH_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_to_fcul_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance [km] from each GPS point to the FCUL reference."""
    return haversine_km(lat, lon,
                        np.full_like(lat, FCUL_LAT),
                        np.full_like(lon, FCUL_LON))


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sensor(sensor_name: str, file_path: Path) -> pd.DataFrame | None:
    """
    Load and minimally prepare one sensor CSV file.

    No additional filters are applied beyond what clean_data.py already did.
    GPS coordinates are coerced to float; rows with missing coordinates or
    timestamp are dropped. Generic working columns 'lat', 'lon' and
    'rad_calib' are added from the sensor-specific columns.

    Returns None if the file is missing or empty after loading.
    """
    if not file_path.is_file():
        print(f'  [SKIP] {sensor_name}: file not found — {file_path.name}')
        return None

    df = pd.read_csv(file_path, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    df['datetime_local'] = (
        pd.to_datetime(df[DATE_COL], utc=True, errors='coerce')
        .dt.tz_convert(LOCAL_TZ)
    )
    df = df.dropna(subset=['datetime_local'])

    lat_col = SENSOR_COLS[sensor_name]['lat']
    lon_col = SENSOR_COLS[sensor_name]['lon']

    for col in [lat_col, lon_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    df = df.sort_values('datetime_local').reset_index(drop=True)

    df['date']   = df['datetime_local'].dt.date
    df['hour']   = df['datetime_local'].dt.hour
    df['lat']    = df[lat_col]
    df['lon']    = df[lon_col]

    # Numeric irradiance columns (may be absent in some variants)
    rad_col = SENSOR_COLS[sensor_name]['rad_calib']
    if rad_col in df.columns:
        df['rad_calib'] = pd.to_numeric(df[rad_col], errors='coerce')
    if 'ghi' in df.columns:
        df['ghi'] = pd.to_numeric(df['ghi'], errors='coerce')

    print(f'  {sensor_name}: {len(df):>8,} rows  '
          f'({df["date"].min()} → {df["date"].max()})')
    return df


# ==============================================================================
# DAILY INDICATOR COMPUTATION
# ==============================================================================

def compute_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the six daily diagnostic indicators from a sensor DataFrame.

    Each day (date) produces one row in the output with columns:
      date, dist_km, fcul_frac_pct, duration_h, mean_speed_kmh,
      rad_ratio, moving_frac_pct, n_rows.

    Gap-corrected Haversine distances are computed here; gaps > MAX_GAP_MIN
    between consecutive rows are zeroed (same logic as
    vehicle_movement_analysis.py).
    """
    df = df.copy().sort_values('datetime_local').reset_index(drop=True)

    # ── Per-row gap-corrected Haversine distance ───────────────────────────────
    lat  = df['lat'].values
    lon  = df['lon'].values
    dt_s = df['datetime_local'].diff().dt.total_seconds().fillna(0).values

    dist_km = np.zeros(len(df))
    for i in range(1, len(df)):
        if 0 < dt_s[i] <= MAX_GAP_MIN * 60:
            dist_km[i] = haversine_km(lat[i-1], lon[i-1], lat[i], lon[i])
    df['dist_km']   = dist_km
    df['dist_m']    = dist_km * 1000

    # Speed [km/h]: distance covered in one minute × 60
    df['speed_kmh'] = df['dist_km'] * 60.0

    # Movement classification (same threshold as vehicle_movement_analysis.py)
    df['moving']    = df['dist_m'] >= THRESHOLD_M

    # Distance to FCUL
    df['d_fcul_km'] = distance_to_fcul_km(df['lat'].values, df['lon'].values)
    df['in_fcul']   = df['d_fcul_km'] <= EXCLUSION_RADIUS_KM

    # ── Daily aggregation ──────────────────────────────────────────────────────
    records = []
    for date, grp in df.groupby('date'):
        n = len(grp)
        if n < MIN_ROWS_PER_DAY:
            continue

        # 1. Daily distance [km]
        d_km = float(grp['dist_km'].sum())

        # 2. Fraction of time within FCUL zone [%]
        fcul_frac = float(grp['in_fcul'].mean() * 100)

        # 3. Measurement duration [hours]
        duration_h = n / 60.0

        # 4. Mean speed while moving [km/h]
        moving_rows = grp[grp['moving']]
        mean_spd = float(moving_rows['speed_kmh'].mean()) if len(moving_rows) > 0 else 0.0

        # 5. Median rad_calib / ghi ratio (only when both are positive)
        rad_ratio = np.nan
        if 'rad_calib' in grp.columns and 'ghi' in grp.columns:
            sub = grp[(grp['ghi'] > 10) & (grp['rad_calib'] > 0)]
            if len(sub) >= 10:
                rad_ratio = float((sub['rad_calib'] / sub['ghi']).median())

        # 6. Fraction of time moving [%]
        moving_frac = float(grp['moving'].mean() * 100)

        records.append({
            'date':            date,
            'dist_km':         d_km,
            'fcul_frac_pct':   fcul_frac,
            'duration_h':      duration_h,
            'mean_speed_kmh':  mean_spd,
            'rad_ratio':       rad_ratio,
            'moving_frac_pct': moving_frac,
            'n_rows':          n,
        })

    daily = pd.DataFrame(records)
    daily['date'] = pd.to_datetime(daily['date'])
    return daily.sort_values('date').reset_index(drop=True)


# ==============================================================================
# PLOTTING
# ==============================================================================

# Panel definitions: (column, ylabel, y-unit, colour, reference_band_key)
PANELS = [
    ('dist_km',         'Distance',         'km / day',       '#1565C0', 'dist_km'),
    ('fcul_frac_pct',   'FCUL proximity',   '% of time',      '#C0392B', 'fcul_frac_pct'),
    ('duration_h',      'Measurement\nduration', 'hours / day', '#2E7D32', 'duration_h'),
    ('mean_speed_kmh',  'Mean speed\n(moving)', 'km / h',     '#6A1B9A', 'mean_speed_kmh'),
    ('rad_ratio',       'rad_calib / GHI\n(daily median)', '—', '#E65100', 'rad_ratio'),
    ('moving_frac_pct', 'Moving\nfraction',  '% of time',      '#00838F', 'moving_frac_pct'),
]

BAND_COLOUR  = '#F5F5F5'
BAND_EDGE    = '#BDBDBD'
TREND_ALPHA  = 0.85
SCATTER_SIZE = 8
SCATTER_ALPHA= 0.45
FIG_BG       = 'white'


def plot_sensor(sensor_name: str, daily: pd.DataFrame, out_path: Path) -> None:
    """
    Produce the 6-panel diagnostic figure for one sensor.

    Each panel shows:
      - Grey band: "normal vehicle-mounted" reference range
      - Scatter dots: individual daily values
      - Solid line: ROLLING_DAYS rolling mean
      - Red triangles: days above / below the reference band
      - Dotted line at 1.0 on the rad_ratio panel, to spot the sustained
        near-1.0 plateau typical of a rooftop deployment
    """
    n_panels = len(PANELS)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(16, 2.6 * n_panels),
                             sharex=True,
                             facecolor=FIG_BG)
    fig.subplots_adjust(hspace=0.08, top=0.93, bottom=0.05,
                        left=0.10, right=0.97)

    fig.suptitle(
        f'Sensor {sensor_name} — Daily deployment diagnostic\n'
        f'[{CAMPAIGN} / {FILE_SUFFIX}]  '
        f'{daily["date"].min().strftime("%Y-%m-%d")} → '
        f'{daily["date"].max().strftime("%Y-%m-%d")}',
        fontsize=13, fontweight='bold', y=0.97
    )
    # Method parameters (figure footer)
    fig.text(0.01, 0.005,
             f'Movement threshold: {THRESHOLD_M:.0f} m  |  '
             f'Gap correction: {MAX_GAP_MIN:.0f} min  |  '
             f'Min rows/day: {MIN_ROWS_PER_DAY}  |  '
             f'Rolling mean: {ROLLING_DAYS} days',
             fontsize=7, color='grey', style='italic')

    for ax, (col, ylabel, unit, colour, band_key) in zip(axes, PANELS):
        ax.set_facecolor(FIG_BG)

        # Reference band
        lo, hi = REFERENCE_BANDS[band_key]
        ax.axhspan(lo, hi, color=BAND_COLOUR, linewidth=0)
        ax.axhline(lo, color=BAND_EDGE, linewidth=0.6, linestyle='--')
        ax.axhline(hi, color=BAND_EDGE, linewidth=0.6, linestyle='--')

        vals = daily[col].values.astype(float)
        valid = ~np.isnan(vals)

        # Scatter
        ax.scatter(daily['date'][valid], vals[valid],
                   s=SCATTER_SIZE, color=colour, alpha=SCATTER_ALPHA,
                   zorder=3, linewidths=0)

        # Rolling mean
        s = pd.Series(vals, index=daily['date'])
        roll = s.rolling(window=ROLLING_DAYS, min_periods=3, center=True).mean()
        ax.plot(roll.index, roll.values,
                color=colour, linewidth=1.6, alpha=TREND_ALPHA, zorder=4)

        # Panel annotation
        ylab_full = f'{ylabel}\n[{unit}]' if unit != '—' else ylabel
        ax.set_ylabel(ylab_full, fontsize=8, labelpad=4)
        ax.yaxis.set_label_position('left')
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(axis='y', linewidth=0.4, alpha=0.5)
        ax.grid(axis='x', linewidth=0.3, alpha=0.3)

        # Band label (right side)
        ax.text(1.002, (lo + hi) / 2,
                f'normal\n[{lo}–{hi}]',
                transform=ax.get_yaxis_transform(),
                fontsize=6, color='grey', va='center', ha='left')

        # Special: rad_ratio reference line at 1.0
        if col == 'rad_ratio':
            ax.axhline(1.0, color='#E65100', linewidth=0.8,
                       linestyle=':', alpha=0.7, label='ratio = 1.0')

        # Flag days above/below band
        above = vals > hi
        below = (vals < lo) & valid
        if above.any():
            ax.scatter(daily['date'][above], vals[above],
                       s=SCATTER_SIZE * 2.5, color='#C0392B',
                       marker='^', zorder=5, linewidths=0, alpha=0.7)
        if below.any():
            ax.scatter(daily['date'][below], vals[below],
                       s=SCATTER_SIZE * 2.5, color='#C0392B',
                       marker='v', zorder=5, linewidths=0, alpha=0.7)

    # x-axis formatting (bottom panel only)
    ax_bot = axes[-1]
    ax_bot.xaxis.set_major_locator(mdates.MonthLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax_bot.tick_params(axis='x', labelsize=7)

    # Legend for out-of-band markers
    patch_band  = mpatches.Patch(color=BAND_COLOUR, label='Normal range (vehicle-mounted)')
    patch_above = plt.Line2D([0], [0], marker='^', color='w',
                             markerfacecolor='#C0392B', markersize=6,
                             label='Above normal range')
    patch_below = plt.Line2D([0], [0], marker='v', color='w',
                             markerfacecolor='#C0392B', markersize=6,
                             label='Below normal range')
    axes[0].legend(handles=[patch_band, patch_above, patch_below],
                   fontsize=7, loc='upper right', framealpha=0.85)

    fig.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches='tight',
                facecolor=FIG_BG)
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print('=' * 72)
    print('SENSOR DEPLOYMENT DIAGNOSTIC')
    print(f'  Campaign : {CAMPAIGN}')
    print(f'  Suffix   : {FILE_SUFFIX}')
    print(f'  Input    : {DATA_DIR}')
    print(f'  Output   : {OUTPUT_DIR}')
    print('=' * 72)

    for sensor_name, file_path in SENSOR_FILES.items():
        print(f'\n[{sensor_name}] Loading ...')
        df = load_sensor(sensor_name, file_path)
        if df is None:
            continue

        print('  Computing daily indicators ...')
        daily = compute_daily_indicators(df)
        print(f'  {len(daily)} valid days  '
              f'({daily["date"].min().strftime("%Y-%m-%d")} → '
              f'{daily["date"].max().strftime("%Y-%m-%d")})')

        # ── Export CSV ────────────────────────────────────────────────────────
        csv_path = _build_output_path(
            OUTPUT_DIR,
            f'sddiag_{CAMPAIGN}_{sensor_name.lower()}_daily',
            FILE_SUFFIX, 'csv'
        )
        daily.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
        print(f'  CSV  : {csv_path.name}')

        # ── Export figure ─────────────────────────────────────────────────────
        png_path = _build_output_path(
            OUTPUT_DIR,
            f'sddiag_{CAMPAIGN}_{sensor_name.lower()}',
            FILE_SUFFIX, 'png'
        )
        plot_sensor(sensor_name, daily, png_path)

    print('\n' + '=' * 72)
    print('Done.')
    print('=' * 72)


if __name__ == '__main__':
    main()
