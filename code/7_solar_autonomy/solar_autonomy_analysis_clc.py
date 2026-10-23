"""
================================================================================
Solar Autonomy Analysis — Monthly Summary & Land-Use (CLC) Statistics
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Pools all sensors of the active campaign and produces:

  OUTPUT 1 — CSV: monthly table
      One row per calendar month with energy, time, autonomy and distance
      metrics.
  OUTPUT 2 — CSV: land-use (CLC) statistics table
      One row per CLC grid_code (1–44), all rows always shown.
  OUTPUT 3 — Four figures
      Fig 1 : Top-10 CLC zones by measurement count
      Fig 2 : Dual-axis bar + line chart of autonomy loss rate and mean
              captured-radiation fraction by CLC zone
      Fig 3 : Horizontal bar chart of relative autonomy losses (%) by CLC zone
      Fig 4 : 100 % stacked horizontal bars of Moving / Parked proportion per
              CLC zone, ordered by Moving % (descending), framed by an
              Urban / Rural edge strip; a dashed line marks the dataset-wide
              mean Moving %.
      Zones with fewer than MIN_FIG_ZONE_POINTS records are left out of
      Figs 2-3, and with fewer than MIN_ZONE_POINTS out of Fig 4.

Physical model
--------------
  E_solar   [kWh/kWp] = rad_calib [W/m²] / 1000 × Δt [h]
  Autonomy  [km/kWp]  = E_solar / CVE    (CVE = 16.6 kWh/100 km, EEA /
                                          Weiss et al., 2024 — EU average BEV)
  loss_irr  [kWh/kWp] = max(0, GHI − rad_calib) / 1000 × Δt
  loss_auto [km/kWp]  = loss_irr / CVE
  frac_captured [-]   = rad_calib / ghi   (0 when either term is zero)

Movement classification
-----------------------
  A GPS transition between consecutive points >= THRESHOLD_M metres is
  "Moving", otherwise "Parked" (pairs across a gap > MAX_GAP_MINUTES or a date
  change count as Parked).

Monthly table
-------------
  Time complement per active day: unrecorded_h = 24 − parked_h − moving_h
  (night, acquisition gaps and any other period without data), averaged over
  the active days of the month. Months with fewer than MIN_MEASUREMENTS
  records stay in the table with their count, but all computed columns are
  set to NaN. This threshold does not apply to the land-use table.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', '<SENSOR> Rad calib', 'ghi', 'sun_elev',
             'land_use_clcv'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             saa_monthly_table, saa_landuse_table
                 (_<CAMPAIGN>_<variant_tag>.csv)
             saa_fig1_top10_zones, saa_fig2_loss_frac, saa_fig3_loss_pct,
             saa_fig4_moving_parked_by_zone  (_<CAMPAIGN>_<variant_tag>.png)

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.7
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from clc_mapping import CLC_MAPPING

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR_STANDARD = STANDARD_DIR
DATA_DIR_TUKTUK   = TUKTUK_DIR
if CAMPAIGN == "standard":
    DATA_DIR = DATA_DIR_STANDARD
elif CAMPAIGN == "tuktuk":
    DATA_DIR = DATA_DIR_TUKTUK

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output directory ---------------------------------------------------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "saa_clc_outputs"

# --- Display figures during execution? ----------------------------------------
SHOW_FIGURES: bool = False

# --- Physical / vehicle parameters -------------------------------------------
# Average Battery Electric Vehicle energy consumption
# Source: European Environment Agency / Weiss et al., 2024 — EU average BEV
CVE_KWH_PER_100KM: float = 16.6
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0

# Measurement time resolution
DELTA_T_H: float = 1.0 / 60.0   # 1 minute in hours

# --- Movement threshold -------------------------------------------------------
# GPS transitions ≥ THRESHOLD_M metres → classified as Moving
THRESHOLD_M: float = 30.0        # metres
R_EARTH_KM:  float = 6371.0      # mean Earth radius [km]
MAX_GAP_MINUTES: int = 2
# --- Reliability threshold (monthly table only) -------------------------------
# Months with fewer than MIN_MEASUREMENTS filtered records have all computed
# columns set to NaN.  The row remains visible with the count shown.
# This threshold does NOT apply to the land-use table.
MIN_MEASUREMENTS: int = 600

# --- Reliability threshold for Fig 4 (Moving/Parked split by CLC zone) -------
MIN_ZONE_POINTS: int = 600

# --- Reliability threshold for Figs 2 & 3 ------------------------------------
# CLC zones with fewer than MIN_FIG_ZONE_POINTS records are excluded from the
# loss-rate and relative-loss figures.
MIN_FIG_ZONE_POINTS: int = 600

# ==============================================================================
# CLC LAND-USE MAPPING  (grid_code 1–44)
# Imported from clc_mapping.py (code/common/).
# land_use_clcv column = grid_code directly (1-based, range 1–44).
# CLC_MAPPING     : dict[int, CLCEntry]  land_use_clcv → named tuple
# CLC_LVL3_LABEL  : dict[int, str]       land_use_clcv → LVL3 label
# CLC_URBAN_RURAL : dict[int, str]       land_use_clcv → 'Urban' / 'Rural'
# ==============================================================================

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

SENSORS = ['J30', 'J31', 'J32', 'J33']


# ==============================================================================
# FILENAME RESOLUTION  (consistent with the rest of the VIPV pipeline)
# ==============================================================================

def _build_sensor_files(data_dir: Path, suffix: str) -> dict[str, Path]:
    """
    Return the dict mapping sensor names to their CSV paths.

    standard : j3x_satellite_landuse_v2_{suffix}.csv
    tuktuk   : j3x_tuktuk_ghi_cams_cos_clc_bh_sw_norm{suffix}.csv

    """
    sep = f"_{suffix}" if suffix else ""
    if CAMPAIGN == "standard":
        return {s: data_dir / f"{s.lower()}_satellite_landuse_v2{sep}.csv"
                for s in SENSORS}
    elif CAMPAIGN == "tuktuk":
        return {s: data_dir / f"{s.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv"
                for s in SENSORS}
    else:
        raise ValueError(f"Unknown CAMPAIGN '{CAMPAIGN}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output sub-directory, embedding campaign + variant tag so
    outputs from different campaigns/variants never overwrite each other."""
    return base_dir / campaign / (variant_tag if variant_tag else "raw")


def _build_output_path(out_dir: Path, base_name: str, campaign: str,
                       variant_tag: str, extension: str) -> Path:
    """Build a full output file path embedding CAMPAIGN + the active variant tag."""
    label = variant_tag if variant_tag else "raw"
    return out_dir / f"{base_name}_{campaign}_{label}.{extension}"


# Resolve all paths at module level
_VARIANT_TAG: str = _variant_tag()
SENSOR_FILES: dict[str, Path] = _build_sensor_files(DATA_DIR, _VARIANT_TAG)
OUTPUT_DIR:   Path             = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)

OUT_CSV_MONTHLY  = _build_output_path(OUTPUT_DIR, 'saa_monthly_table',   CAMPAIGN, _VARIANT_TAG, 'csv')
OUT_CSV_LANDUSE  = _build_output_path(OUTPUT_DIR, 'saa_landuse_table',   CAMPAIGN, _VARIANT_TAG, 'csv')
OUT_FIG1         = _build_output_path(OUTPUT_DIR, 'saa_fig1_top10_zones',CAMPAIGN, _VARIANT_TAG, 'png')
OUT_FIG2         = _build_output_path(OUTPUT_DIR, 'saa_fig2_loss_frac',  CAMPAIGN, _VARIANT_TAG, 'png')
OUT_FIG3         = _build_output_path(OUTPUT_DIR, 'saa_fig3_loss_pct',   CAMPAIGN, _VARIANT_TAG, 'png')
OUT_FIG4         = _build_output_path(OUTPUT_DIR, 'saa_fig4_moving_parked_by_zone', CAMPAIGN, _VARIANT_TAG, 'png')

# Column name mapping per sensor (sensor-specific prefixes in CSV files)
SENSOR_COLS: dict[str, dict[str, str]] = {
    s: {
        'date': 'Human readable date (Europe/London)',
        'lat':  f'{s} lat',
        'lon':  f'{s} lon',
        'rad':  f'{s} Rad calib',
        'ghi':  'ghi',
        'elev': 'sun_elev',
        'land': 'land_use_clcv',
    }
    for s in SENSORS
}


# ==============================================================================
# HELPER — HAVERSINE DISTANCE
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2, R=R_EARTH_KM,
                         timestamps=None, max_gap_minutes=MAX_GAP_MINUTES):
    """
    Compute the great-circle distance in metres between consecutive GPS points.

    Parameters
    ----------
    lat1, lon1 : array-like
        Coordinates of the departure point (decimal degrees, WGS-84).
        Typically df[lat_col].values[:-1].
    lat2, lon2 : array-like
        Coordinates of the arrival point (decimal degrees, WGS-84).
        Typically df[lat_col].values[1:].
    R : float
        Earth radius in kilometres (default: 6371.0 — IAU 2012).
    timestamps : array-like of datetime64, optional
        Full timestamp array of length N (i.e. df['datetime'].values).
        When provided, pairs that cross a segment boundary are zeroed:
          - time gap > max_gap_minutes  (overnight stop, sensor restart)
          - calendar-date change        (explicit cross-day guard)
        When None (default), all distances are returned as computed — safe
        for point-to-fixed-reference uses (distance rings, FCUL proximity).
    max_gap_minutes : float
        Maximum allowed gap in minutes between consecutive rows.
        Defaults to MAX_GAP_MINUTES (defined in the CONFIG section).

    Returns
    -------
    numpy.ndarray of float, shape (N-1,)
        Great-circle distances in metres.
        Boundary pairs are returned as 0.0 when timestamps is provided.
    """
    lat1 = np.asarray(lat1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    phi1    = np.radians(lat1)
    phi2    = np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(np.asarray(lon2, dtype=float)
                         - np.asarray(lon1, dtype=float))
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    distances = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * 1000.0

    if timestamps is not None:
        s         = pd.Series(np.asarray(timestamps))
        gap_s     = s.diff().dt.total_seconds().fillna(0).values
        dates     = s.dt.date.values
        # gap_s[i+1] = elapsed seconds between row i and row i+1;
        # distances[i] covers that same pair -> compare gap_s[1:]
        is_boundary = (
            (gap_s[1:] > max_gap_minutes * 60)
            | (dates[1:] != dates[:-1])
        )
        distances[is_boundary] = 0.0

    return distances


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sensor(sensor_name: str, file_path: Path) -> pd.DataFrame | None:
    """
    Load, rename, and enrich one sensor CSV file.

    Steps
    -----
    1. Verify existence; return None if missing.
    2. Read required columns only.
    3. Parse timestamps (UTC-aware).
    4. Rename sensor-specific columns to generic internal names.
    5. Coerce numeric types; drop rows with NaN in key columns.
    6. Compute Haversine distance to next row and classify Moving/Parked.
    7. Compute per-minute energy, loss, gain, fraction, and autonomy.
    8. Apply daytime and irradiance filters.

    Parameters
    ----------
    sensor_name : str   Sensor identifier, e.g. 'J31'.
    file_path   : Path  Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
    """
    if not file_path.exists():
        print(f"  [WARNING] File not found, skipping: {file_path}")
        return None

    cols = SENSOR_COLS[sensor_name]
    required = [cols['date'], cols['lat'], cols['lon'],
                cols['rad'],  cols['ghi'], cols['elev'], cols['land']]

    df = pd.read_csv(file_path, usecols=required, low_memory=False)

    # ── Rename to generic names ───────────────────────────────────────────────
    df = df.rename(columns={
        cols['date']: 'datetime',
        cols['lat']:  'lat',
        cols['lon']:  'lon',
        cols['rad']:  'rad_calib',
        cols['ghi']:  'ghi',
        cols['elev']: 'sun_elev',
        cols['land']: 'land_use_clcv',
    })

    # ── Parse timestamps ──────────────────────────────────────────────────────
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df = (df.dropna(subset=['datetime'])
            .sort_values('datetime')
            .reset_index(drop=True))

    # ── Numeric coercion ──────────────────────────────────────────────────────
    for col in ['lat', 'lon', 'rad_calib', 'ghi', 'sun_elev']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['land_use_clcv'] = pd.to_numeric(df['land_use_clcv'], errors='coerce').astype('Int64')

    df = df.dropna(subset=['lat', 'lon', 'rad_calib', 'ghi', 'sun_elev'])
    df['sensor'] = sensor_name

    # ── Calendar columns ──────────────────────────────────────────────────────
    # Convert to local time (Europe/Lisbon) for month/date grouping
    local = df['datetime'].dt.tz_convert('Europe/Lisbon')
    df['month'] = local.dt.month
    df['date']  = local.dt.date      # used for "active days" counting

    # ── Haversine distance to next consecutive record (per sensor) ────────────
    lats = df['lat'].values
    lons = df['lon'].values
    dist_m = np.full(len(df), np.nan)
    if len(df) > 1:
        dist_m[:-1] = haversine_vectorized(lats[:-1], lons[:-1],
                                            lats[1:],  lons[1:],
                                            timestamps=df['datetime'].values)
    df['dist_m'] = dist_m

    # ── Movement classification ───────────────────────────────────────────────
    # NaN (last row) treated as Parked.
    df['is_moving'] = (df['dist_m'] >= THRESHOLD_M).fillna(False)

    # ── Per-minute irradiance quantities ─────────────────────────────────────
    # Differential power: positive = sensor below reference (loss)
    diff_W = df['ghi'] - df['rad_calib']

    # Energy [kWh/kWp] per minute
    df['e_solar_kWh']  = df['rad_calib'] / 1000.0 * DELTA_T_H
    df['loss_irr_kWh'] = (diff_W / 1000.0 * DELTA_T_H).clip(lower=0.0)
    df['gain_irr_kWh'] = ((-diff_W) / 1000.0 * DELTA_T_H).clip(lower=0.0)

    # Autonomy [km/kWp] per minute
    df['auto_solar_km'] = df['e_solar_kWh'] / CVE_KWH_PER_KM
    df['loss_auto_km']  = df['loss_irr_kWh'] / CVE_KWH_PER_KM
    df['gain_auto_km']  = df['gain_irr_kWh'] / CVE_KWH_PER_KM

    # Captured-radiation fraction (0 when either term is zero)
    mask_frac = (df['ghi'] > 0) & (df['rad_calib'] > 0)
    df['frac_captured'] = 0.0
    df.loc[mask_frac, 'frac_captured'] = (
        df.loc[mask_frac, 'rad_calib'] / df.loc[mask_frac, 'ghi']
    )

    return df


# ==============================================================================
# OUTPUT 1 — MONTHLY TABLE
# ==============================================================================

def build_monthly_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-minute data into a monthly summary table.

    Columns
    -------
    month, month_name,
    n_measurements,
    energy_loss_kWh_m2,  energy_gain_kWh_m2,  energy_diff_kWh_m2,
    parked_h_per_day,    moving_h_per_day,     unrecorded_h_per_day,
    auto_solar_km_per_day, auto_loss_km_per_day, auto_diff_km_per_day,
    distance_total_km,
    distance_J30_km, distance_J31_km, distance_J32_km, distance_J33_km

    All computed columns are NaN for months with n_measurements < MIN_MEASUREMENTS.

    Time complement logic
    ---------------------
    For each active day d:
        parked_min_d   = count of Parked records on day d
        moving_min_d   = count of Moving records on day d
        unrecorded_h_d = 24 − (parked_min_d + moving_min_d) / 60
    Then average over all active days in the month.

    Distance
    --------
    Sum of Haversine distances ≥ THRESHOLD_M (i.e. only Moving transitions),
    reported as total and per sensor.

    Parameters
    ----------
    data : pd.DataFrame  Combined dataset from all sensors.

    Returns
    -------
    pd.DataFrame  12 rows (all calendar months represented).
    """
    rows = []

    for m in range(1, 13):
        sub = data[data['month'] == m]
        n   = len(sub)

        row: dict = {
            'month':       m,
            'month_name':  MONTH_NAMES[m - 1],
            'n_measurements': n,
        }

        if n < MIN_MEASUREMENTS:
            # Keep row visible but mask all computed columns
            for col in [
                'energy_loss_kWh_m2', 'energy_gain_kWh_m2', 'energy_diff_kWh_m2',
                'parked_h_per_day', 'moving_h_per_day', 'unrecorded_h_per_day',
                'auto_solar_km_per_day', 'auto_loss_km_per_day',
                'auto_diff_km_per_day',
                'distance_total_km',
            ] + [f'distance_{s}_km' for s in SENSORS]:
                row[col] = np.nan
            rows.append(row)
            continue

        # ── Active days ───────────────────────────────────────────────────────
        # A day is "active" when a given sensor has at least one filtered
        # record on it. Time complement is computed per (sensor, date) pair
        # then averaged — grouping by 'date' alone would pool minutes from
        # several simultaneously-active sensors into a single 24 h budget
        # (parked_min + moving_min could then exceed 1440 min/day, making
        # unrecorded_h negative). This matches the (sensor, date) grouping
        # already used for daily_auto below.
        daily_times = (
            sub.groupby(['sensor', 'date'], as_index=False)
               .agg(
                   parked_min  = ('is_moving', lambda x: (x == False).sum()),
                   moving_min  = ('is_moving', lambda x: (x == True ).sum()),
               )
        )
        daily_times['unrecorded_h'] = (
            24.0
            - daily_times['parked_min'] / 60.0
            - daily_times['moving_min'] / 60.0
        )
        n_active_days = len(daily_times)

        row['parked_h_per_day']     = (daily_times['parked_min'].sum() / 60.0
                                       / n_active_days)
        row['moving_h_per_day']     = (daily_times['moving_min'].sum() / 60.0
                                       / n_active_days)
        row['unrecorded_h_per_day'] = daily_times['unrecorded_h'].mean()

        # ── Energy metrics ────────────────────────────────────────────────────
        row['energy_loss_kWh_m2'] = sub['loss_irr_kWh'].sum()
        row['energy_gain_kWh_m2'] = sub['gain_irr_kWh'].sum()
        row['energy_diff_kWh_m2'] = (row['energy_gain_kWh_m2']
                                     - row['energy_loss_kWh_m2'])

        # ── Autonomy metrics (daily average) ─────────────────────────────────
        daily_auto = (
            sub.groupby(['sensor', 'date'], as_index=False)
               .agg(
                   auto_solar = ('auto_solar_km', 'sum'),
                   auto_loss  = ('loss_auto_km',  'sum'),
                   auto_gain  = ('gain_auto_km',  'sum'),
               )
        )
        row['auto_solar_km_per_day'] = daily_auto['auto_solar'].mean()
        row['auto_loss_km_per_day']  = daily_auto['auto_loss'].mean()
        row['auto_diff_km_per_day']  = (daily_auto['auto_gain'].mean()
                                        - daily_auto['auto_loss'].mean())

        # ── Distance (Moving transitions only) ───────────────────────────────
        moving_sub = sub[sub['is_moving']]
        row['distance_total_km'] = moving_sub['dist_m'].sum() / 1000.0
        for s in SENSORS:
            row[f'distance_{s}_km'] = (
                moving_sub.loc[moving_sub['sensor'] == s, 'dist_m'].sum() / 1000.0
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# OUTPUT 2 — LAND-USE TABLE
# ==============================================================================

def build_landuse_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-minute data by CLC land-use type.

    All 44 CLC grid_codes (1–44) are included in the output, even when no
    data is present for a given code.

    Columns
    -------
    grid_code, clc_code, land_use_clcv_label, clc_category, urban_rural,
    n_points,            (= number of filtered 1-minute records)
    duration_h,          (= n_points / 60)
    pct_of_total,        (% of all filtered records)
    auto_loss_rate_km_kWp_h,  (sum of per-minute losses / duration_h)
    mean_frac_captured_pct,   (mean rad_calib/ghi × 100)
    pct_moving,          (% of records classified as Moving)
    pct_parked           (% of records classified as Parked)

    Parameters
    ----------
    data : pd.DataFrame  Combined dataset from all sensors.

    Returns
    -------
    pd.DataFrame  44 rows (one per CLC class, sorted by loss rate desc).
    """
    total_n = len(data)
    rows    = []

    for land_use_clcv in range(1, 45):
        # land_use_clcv = grid_code directly (1-based, no offset needed)
        sub = data[data['land_use_clcv'] == land_use_clcv]
        n   = len(sub)

        entry    = CLC_MAPPING[land_use_clcv]
        clc_code = entry.clc_code
        label    = entry.lvl3_label
        category = entry.lvl2_label
        ur       = entry.urban_rural

        duration_h  = n / 60.0
        pct_total   = (n / total_n * 100.0) if total_n > 0 else 0.0

        if n == 0:
            rows.append({
                'grid_code':              land_use_clcv,
                'clc_code':               clc_code,
                'land_use_clcv_label':         label,
                'clc_category':           category,
                'urban_rural':            ur,
                'n_points':               0,
                'duration_h':             0.0,
                'pct_of_total':           0.0,
                'auto_loss_rate_km_kWp_h':np.nan,
                'auto_loss_pct':          np.nan,
                'mean_frac_captured_pct': np.nan,
                'pct_moving':             np.nan,
                'pct_parked':             np.nan,
            })
            continue

        loss_rate = (sub['loss_auto_km'].sum() / duration_h
                     if duration_h > 0 else np.nan)
        mean_frac = sub['frac_captured'].mean() * 100.0
        pct_mov   = sub['is_moving'].mean() * 100.0
        pct_par   = (1.0 - sub['is_moving'].mean()) * 100.0

        # Relative autonomy loss [%]:
        #   loss_pct = loss_auto / (auto_solar + loss_auto) × 100
        total_loss = sub['loss_auto_km'].sum()
        total_auto = sub['auto_solar_km'].sum()
        denom      = total_auto + total_loss
        auto_loss_pct = (total_loss / denom * 100.0
                         if denom > 0 else np.nan)

        rows.append({
            'grid_code':               land_use_clcv,
            'clc_code':                clc_code,
            'land_use_clcv_label':          label,
            'clc_category':            category,
            'urban_rural':             ur,
            'n_points':                n,
            'duration_h':              round(duration_h, 4),
            'pct_of_total':            round(pct_total,  4),
            'auto_loss_rate_km_kWp_h': round(loss_rate,  6),
            'auto_loss_pct':           round(auto_loss_pct, 4) if not np.isnan(auto_loss_pct) else np.nan,
            'mean_frac_captured_pct':  round(mean_frac,  4),
            'pct_moving':              round(pct_mov,    2),
            'pct_parked':              round(pct_par,    2),
        })

    df_lu = pd.DataFrame(rows)
    return df_lu.sort_values(
        'auto_loss_rate_km_kWp_h', ascending=False, na_position='last'
    ).reset_index(drop=True)


# ==============================================================================
# OUTPUT 3 — FIGURES
# ==============================================================================

def _save_and_show(fig: plt.Figure, path: Path) -> None:
    """Save figure to disk and optionally display it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved → {path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_fig1_top10_zones(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 1 — Top-10 CLC zones by measurement count.

    Displays a horizontal bar chart (more readable than a pie for 10 items)
    with each bar coloured by Urban / Rural classification.
    A secondary mini-pie chart shows the Urban/Rural split of the top-10
    as an inset for quick visual reference.

    Parameters
    ----------
    lu       : land-use table (output of build_landuse_table)
    out_path : destination PNG path
    """
    # Keep only zones with data, exclude -999
    plot_df = lu[
        (lu['grid_code'] != -999) & (lu['n_points'] > 0)
    ].nlargest(10, 'n_points').copy()

    if plot_df.empty:
        print("  [WARNING] Fig 1: no data to plot.")
        return

    # ── Project-wide color palette ────────────────────────────────────────────
    COLOR_URBAN = '#C0392B'
    COLOR_RURAL = '#2E7D32'
    colors      = [COLOR_URBAN if r == 'Urban' else COLOR_RURAL
                   for r in plot_df['urban_rural']]

    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    y_pos = np.arange(len(plot_df))
    ax_bar.barh(y_pos, plot_df['n_points'].values,
                color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    # Annotate bars with percentage of total
    for i, (n, pct) in enumerate(
        zip(plot_df['n_points'].values, plot_df['pct_of_total'].values)
    ):
        ax_bar.text(n * 1.005, y_pos[i],
                    f'{pct:.1f}%', va='center', ha='left',
                    fontsize=8, color='#333333')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(plot_df['land_use_clcv_label'].values, fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Number of 1-minute records', fontsize=10)
    ax_bar.set_title('Top-10 CLC Zones by Measurement Count', fontsize=11,
                     fontweight='bold')
    ax_bar.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.legend(
        handles=[mpatches.Patch(color=COLOR_URBAN, label='Urban'),
                 mpatches.Patch(color=COLOR_RURAL, label='Rural')],
        fontsize=8.5, loc='lower right'
    )

    # Mini pie: Urban vs Rural share within top-10
    n_urban = plot_df.loc[plot_df['urban_rural'] == 'Urban', 'n_points'].sum()
    n_rural = plot_df.loc[plot_df['urban_rural'] == 'Rural', 'n_points'].sum()
    if n_urban + n_rural > 0:
        ax_pie.pie(
            [n_urban, n_rural],
            labels=['Urban', 'Rural'],
            colors=[COLOR_URBAN, COLOR_RURAL],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 8.5},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}
        )
        ax_pie.set_title('Urban / Rural split\n(top-10 zones)',
                         fontsize=9, fontweight='bold')
    else:
        ax_pie.axis('off')

    fig.suptitle(
        f'[{CAMPAIGN} / {_VARIANT_TAG or "raw"}]',
        fontsize=8, y=1.01, color='#888888', style='italic'
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


def plot_fig2_loss_frac(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 2 — Dual-axis chart: autonomy loss rate and mean fraction captured.

    Left axis  (bars)  : auto_loss_rate_km_kWp_h  [km/kWp/h]
    Right axis (line)  : mean_frac_captured_pct   [%]

    Only zones with at least MIN_FIG_ZONE_POINTS records are shown to prevent
    artefacts from sparse zones.  In particular, sparse zones frequently show
    mean_frac_captured > 100 % — this is physically possible when the satellite
    GHI underestimates actual irradiance (e.g. localised clear-sky conditions,
    sensor geometry) but is unreliable with very few measurements.  Filtering
    by MIN_FIG_ZONE_POINTS removes these misleading data points.

    Parameters
    ----------
    lu       : land-use table
    out_path : destination PNG path
    """
    plot_df = lu[
        (lu['grid_code'] != -999) &
        (lu['n_points'] >= MIN_FIG_ZONE_POINTS) &
        lu['auto_loss_rate_km_kWp_h'].notna()
    ].copy()

    n_excluded = ((lu['grid_code'] != -999) &
                  (lu['n_points'] > 0) &
                  (lu['n_points'] < MIN_FIG_ZONE_POINTS)).sum()

    if plot_df.empty:
        print("  [WARNING] Fig 2: no data to plot.")
        return

    # ── Project-wide color palette ────────────────────────────────────────────
    COLOR_URBAN = '#C0392B'   # Urban CLC zones
    COLOR_RURAL = '#2E7D32'   # Rural CLC zones
    bar_colors  = [COLOR_URBAN if r == 'Urban' else COLOR_RURAL
                   for r in plot_df['urban_rural']]

    n   = len(plot_df)
    y   = np.arange(n)
    fig, ax1 = plt.subplots(figsize=(13, max(6, n * 0.38)))
    ax2 = ax1.twiny()   # second x-axis on top for the fraction line

    # Bars — loss rate (bottom x-axis)
    ax1.barh(y, plot_df['auto_loss_rate_km_kWp_h'].values,
             color=bar_colors, edgecolor='white', linewidth=0.4,
             height=0.65, label='Loss rate (km/kWp/h)', zorder=3)

    # Line — fraction captured (top x-axis, ax2)
    # Values > 100 % are physically possible (satellite GHI underestimation)
    # but only shown here for zones with sufficient data (≥ MIN_FIG_ZONE_POINTS)
    frac_vals = plot_df['mean_frac_captured_pct'].values
    ax2.plot(frac_vals, y, color='#1a6faf', marker='o',
             markersize=4, linewidth=1.6, label='Mean fraction captured (%)',
             zorder=4)

    # Reference line at 100 % fraction (= sensor matches GHI exactly)
    ax2.axvline(100.0, color='#1a6faf', linewidth=0.8, linestyle=':',
                alpha=0.6, zorder=2)

    # Axes formatting
    ax1.set_yticks(y)
    ax1.set_yticklabels(plot_df['land_use_clcv_label'].values, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Autonomy Loss Rate [km/kWp/h]', fontsize=10,
                   fontweight='bold', color='#333333')
    ax1.grid(True, axis='x', alpha=0.25, linestyle='--')
    ax1.spines[['top']].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.set_xlabel(
        'Mean Captured-Radiation Fraction [%]\n'
        '(> 100 % = sensor irradiance exceeds satellite GHI — physically possible)',
        fontsize=9, fontweight='bold', color='#1a6faf'
    )
    ax2.tick_params(axis='x', colors='#1a6faf')
    ax2.spines['top'].set_color('#1a6faf')
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Combined legend
    handles = [
        mpatches.Patch(color=COLOR_URBAN, label='Urban (loss rate)'),
        mpatches.Patch(color=COLOR_RURAL, label='Rural (loss rate)'),
        plt.Line2D([0], [0], color='#1a6faf', marker='o',
                   markersize=5, label='Mean fraction captured (%)'),
    ]
    ax1.legend(handles=handles, fontsize=8.5, loc='lower right')

    ax1.set_title(
        'Solar Autonomy Loss Rate & Mean Captured-Radiation Fraction\n'
        'by CLC Land-Use Type',
        fontsize=12, fontweight='bold', pad=12
    )

    # Discreet annotation: variant tag + filter note
    fig.text(
        0.5, -0.01,
        f'[{CAMPAIGN} / {_VARIANT_TAG or "raw"}]  —  zones with ≥ {MIN_FIG_ZONE_POINTS:,} records'
        f'  ({n_excluded} excluded)',
        ha='center', fontsize=7.5, color='#888888', style='italic',
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


def plot_fig3_loss_pct(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 3 — Relative autonomy losses (%) by CLC zone.

    Style:
      - Horizontal bar chart, ordered by relative loss descending
      - Coloured by Urban / Rural (project-wide palette)
      - Bars annotated with percentage of total measurement time

    Only zones with at least MIN_FIG_ZONE_POINTS records are shown.  Sparse
    zones can yield extreme auto_loss_pct values due to small-sample variability
    and are excluded to preserve readability and statistical reliability.

    Relative autonomy loss formula (per zone):
        loss_pct = sum(loss_auto) / (sum(auto_solar) + sum(loss_auto)) × 100

    The denominator is the theoretical autonomy under unobstructed GHI.
    Always ≥ 0 and bounded [0, 100 %].

    Parameters
    ----------
    lu       : land-use table (must contain column 'auto_loss_pct')
    out_path : destination PNG path
    """
    plot_df = lu[
        (lu['grid_code'] != -999) &
        (lu['n_points'] >= MIN_FIG_ZONE_POINTS) &
        lu['auto_loss_pct'].notna()
    ].copy()

    n_excluded = ((lu['grid_code'] != -999) &
                  (lu['n_points'] > 0) &
                  (lu['n_points'] < MIN_FIG_ZONE_POINTS)).sum()

    if plot_df.empty:
        print("  [WARNING] Fig 3: no data to plot.")
        return

    plot_df = plot_df.sort_values('auto_loss_pct', ascending=False)

    # ── Project-wide color palette ────────────────────────────────────────────
    COLOR_URBAN = '#C0392B'   # Urban CLC zones
    COLOR_RURAL = '#2E7D32'   # Rural CLC zones
    bar_colors  = [COLOR_URBAN if r == 'Urban' else COLOR_RURAL
                   for r in plot_df['urban_rural']]

    n   = len(plot_df)
    y   = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.38)))

    ax.barh(y, plot_df['auto_loss_pct'].values,
            color=bar_colors, edgecolor='white', linewidth=0.4,
            height=0.75, zorder=3)

    # Annotate each bar with the zone's share of total measurement time
    x_max = plot_df['auto_loss_pct'].max()
    for i, (lp, pct) in enumerate(
        zip(plot_df['auto_loss_pct'].values, plot_df['pct_of_total'].values)
    ):
        ax.text(lp + x_max * 0.008, y[i],
                f'{pct:.1f}%', va='center', ha='left',
                fontsize=7.5, color='#333333')

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['land_use_clcv_label'].values, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(
        r'Relative Autonomy Loss [%]  '
        r'$= \frac{\Sigma\,loss\_auto}{\Sigma\,auto\_solar + \Sigma\,loss\_auto} \times 100$',
        fontsize=10, fontweight='bold'
    )

    excl_note = f'zones with ≥ {MIN_FIG_ZONE_POINTS:,} records  ({n_excluded} excluded)'
    ax.set_title(
        'Relative Solar Autonomy Losses by CLC Land-Use Type',
        fontsize=12, fontweight='bold', pad=10
    )
    fig.text(
        0.5, -0.01,
        f'[{CAMPAIGN} / {_VARIANT_TAG or "raw"}]  —  {excl_note}  —  bar labels = % of total measurement time',
        ha='center', fontsize=7.5, color='#888888', style='italic',
    )
    ax.legend(
        handles=[mpatches.Patch(color=COLOR_URBAN, label='Urban'),
                 mpatches.Patch(color=COLOR_RURAL, label='Rural')],
        fontsize=9, loc='lower right'
    )
    ax.set_xlim(left=0)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# OUTPUT 3 — FIGURE 4  (Moving / Parked split by CLC zone)
# ==============================================================================

def plot_fig4_moving_parked_by_zone(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 4 — Moving / Parked proportion for each CLC zone with sufficient data.

    Each zone is represented by a horizontal 100 % stacked bar split into:
      - Moving (blue)  : percentage of 1-minute records classified as moving
      - Parked (orange): percentage of 1-minute records classified as parked

    Only zones with at least MIN_ZONE_POINTS records are displayed, ensuring
    that proportions are statistically meaningful.  Zones are ordered by their
    Moving percentage (descending), making it easy to identify which land-use
    types are associated with the most active vehicle behaviour.

    A vertical dashed line at the dataset-wide mean Moving % provides a
    reference for comparison across zones.

    Bar annotations
    ---------------
    Each bar is annotated with:
      - Moving %   (left segment, white text, hidden if < 6 % to avoid overlap)
      - Parked %   (right segment, white text, hidden if < 6 %)
      - N records  (right margin, grey text) — signals data volume per zone

    Parameters
    ----------
    lu       : land-use table (output of build_landuse_table).
               Must contain columns: grid_code, land_use_clcv_label, urban_rural,
               n_points, pct_moving, pct_parked.
    out_path : destination PNG path.
    """
    # ── Filter: zones with data, exclude -999 sentinel, apply MIN_ZONE_POINTS ─
    plot_df = lu[
        (lu['grid_code'] != -999) &
        (lu['n_points'] >= MIN_ZONE_POINTS) &
        lu['pct_moving'].notna() &
        lu['pct_parked'].notna()
    ].copy()

    if plot_df.empty:
        print(
            f"  [WARNING] Fig 4: no CLC zone has ≥ {MIN_ZONE_POINTS} records. "
            "Lower MIN_ZONE_POINTS or check the dataset."
        )
        return

    # ── Sort by Moving % descending ──────────────────────────────────────────
    plot_df = plot_df.sort_values('pct_moving', ascending=True).reset_index(drop=True)

    n = len(plot_df)
    y = np.arange(n)

    # ── Colours ──────────────────────────────────────────────────────────────
    COLOR_MOVING = '#1565C0'   # vehicle in motion
    COLOR_PARKED = '#E65100'   # vehicle stationary
    COLOR_URBAN_EDGE = '#C0392B'   # Urban
    COLOR_RURAL_EDGE = '#2E7D32'   # Rural

    # ── Dataset-wide weighted mean Moving % (weighted by n_points) ───────────
    total_n   = plot_df['n_points'].sum()
    mean_mov  = (plot_df['pct_moving'] * plot_df['n_points']).sum() / total_n

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_h = max(5, n * 0.42 + 1.8)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    # Draw stacked bars
    ax.barh(
        y, plot_df['pct_moving'].values,
        height=0.70,
        color=COLOR_MOVING,
        edgecolor='white', linewidth=0.5,
        label='Moving',
        zorder=3,
    )
    ax.barh(
        y, plot_df['pct_parked'].values,
        left=plot_df['pct_moving'].values,
        height=0.70,
        color=COLOR_PARKED,
        edgecolor='white', linewidth=0.5,
        label='Parked',
        zorder=3,
    )

    # Urban / Rural indicator: thin coloured strip on the left edge of each bar
    edge_colors = [
        COLOR_URBAN_EDGE if r == 'Urban' else COLOR_RURAL_EDGE
        for r in plot_df['urban_rural']
    ]
    for i, ec in enumerate(edge_colors):
        ax.barh(
            y[i], 0.6,           # 0.6 % wide strip
            height=0.72,
            color=ec,
            zorder=4,
        )

    # ── Bar annotations ───────────────────────────────────────────────────────
    MIN_LABEL_WIDTH = 6.0   # % — hide text if segment is too narrow

    for i, (mov, par, n_pts) in enumerate(
        zip(plot_df['pct_moving'].values,
            plot_df['pct_parked'].values,
            plot_df['n_points'].values)
    ):
        # Moving label (centre of the moving segment)
        if mov >= MIN_LABEL_WIDTH:
            ax.text(
                mov / 2.0, y[i],
                f'{mov:.1f}%',
                ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold',
                zorder=5,
            )
        # Parked label (centre of the parked segment)
        if par >= MIN_LABEL_WIDTH:
            ax.text(
                mov + par / 2.0, y[i],
                f'{par:.1f}%',
                ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold',
                zorder=5,
            )
        # N records — right margin
        ax.text(
            101.5, y[i],
            f'n = {n_pts:,}',
            ha='left', va='center',
            fontsize=7.0, color='#555555',
        )

    # ── Dataset-wide mean Moving % — vertical dashed reference line ──────────
    ax.axvline(
        mean_mov,
        color='#333333', linewidth=1.2, linestyle='--',
        zorder=5,
        label=f'Dataset mean Moving ({mean_mov:.1f}%)',
    )
    ax.text(
        mean_mov + 0.4, n - 0.1,
        f'Mean: {mean_mov:.1f}%',
        fontsize=8, color='#333333', va='top',
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['land_use_clcv_label'].values, fontsize=8.5)
    ax.set_xlim(0, 115)          # extra space for the "n = …" annotations
    ax.set_xlabel('Proportion of 1-minute records [%]', fontsize=10,
                  fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    ax.grid(True, axis='x', alpha=0.25, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLOR_MOVING,      label='Moving'),
        mpatches.Patch(color=COLOR_PARKED,      label='Parked'),
        mpatches.Patch(color=COLOR_URBAN_EDGE,  label='Urban zone'),
        mpatches.Patch(color=COLOR_RURAL_EDGE,  label='Rural zone'),
        plt.Line2D([0], [0], color='#333333', linewidth=1.2,
                   linestyle='--', label=f'Dataset mean Moving ({mean_mov:.1f}%)'),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=8.5,
        loc='lower right',
        framealpha=0.85,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    n_zones_excluded = int((lu['n_points'] >= MIN_ZONE_POINTS).sum() - len(plot_df))
    ax.set_title(
        'Moving / Parked Proportion by CLC Land-Use Zone',
        fontsize=12, fontweight='bold', pad=12,
    )
    fig.text(
        0.5, -0.01,
        f'[{CAMPAIGN} / {_VARIANT_TAG or "raw"}]  —  zones with ≥ {MIN_ZONE_POINTS:,} records'
        f'  ({n_zones_excluded} excluded for insufficient data)  —  ordered by Moving % ↑',
        ha='center', fontsize=7.5, color='#888888', style='italic',
    )

    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Full pipeline:
      1. Load and process all sensor CSV files.
      2. Build and save monthly summary table (Output 1).
      3. Build and save land-use statistics table (Output 2).
      4. Generate and save the four figures (Output 3).
    """
    print("=" * 70)
    print("  SOLAR AUTONOMY ANALYSIS — MONTHLY & LAND-USE STATISTICS")
    print("=" * 70)
    print(f"\n  CAMPAIGN           : {CAMPAIGN}")
    print(f"  Data variant       : {_VARIANT_TAG or '(raw)'}")
    print(f"  CVE                : {CVE_KWH_PER_100KM} kWh/100 km")
    print(f"  THRESHOLD_M        : {THRESHOLD_M:.0f} m  (parked/moving)")
    print(f"  MIN_MEASUREMENTS   : {MIN_MEASUREMENTS}  (monthly table mask)")
    print(f"  MIN_ZONE_POINTS    : {MIN_ZONE_POINTS}  (Fig 4 zone filter)")
    print(f"  MIN_FIG_ZONE_POINTS: {MIN_FIG_ZONE_POINTS}  (Figs 2 & 3 sparse-zone filter)")

    # ── 1. Load all sensors ───────────────────────────────────────────────────
    print("\n[1/4] Loading sensor CSV files ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for sensor_name, file_path in SENSOR_FILES.items():
        df = load_sensor(sensor_name, file_path)
        if df is not None:
            print(f"  {sensor_name}: {len(df):,} daytime records  "
                  f"(land_use_clcv range: {df['land_use_clcv'].min()}–{df['land_use_clcv'].max()})")
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Check DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX in the configuration section."
        )

    data = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(data):,} daytime records across all sensors")

    # ── 2. Monthly table ──────────────────────────────────────────────────────
    print("\n[2/4] Building monthly summary table ...")
    monthly = build_monthly_table(data)

    masked = monthly[monthly['energy_loss_kWh_m2'].isna()]['month_name'].tolist()
    if masked:
        print(f"  [INFO] Months with < {MIN_MEASUREMENTS} records (columns masked): "
              f"{', '.join(masked)}")

    monthly.to_csv(OUT_CSV_MONTHLY, index=False, float_format='%.6f')
    print(f"  CSV saved → {OUT_CSV_MONTHLY}")

    # Print summary
    print(f"\n  {'Month':<6} {'N':>8}  {'Loss (kWh)':>12}  "
          f"{'Auto (km/kWp/d)':>16}  {'Dist (km)':>10}")
    print(f"  {'-'*6} {'-'*8}  {'-'*12}  {'-'*16}  {'-'*10}")
    for _, r in monthly.iterrows():
        n    = f"{int(r['n_measurements']):>8,}"
        loss = f"{r['energy_loss_kWh_m2']:>12.4f}" if pd.notna(r['energy_loss_kWh_m2']) else '         N/A'
        auto = f"{r['auto_solar_km_per_day']:>16.3f}"  if pd.notna(r['auto_solar_km_per_day']) else '             N/A'
        dist = f"{r['distance_total_km']:>10.2f}"      if pd.notna(r['distance_total_km'])     else '       N/A'
        print(f"  {r['month_name']:<6} {n}  {loss}  {auto}  {dist}")

    # ── 3. Land-use table ─────────────────────────────────────────────────────
    print("\n[3/4] Building land-use statistics table ...")
    landuse = build_landuse_table(data)

    landuse.to_csv(OUT_CSV_LANDUSE, index=False, float_format='%.6f')
    print(f"  CSV saved → {OUT_CSV_LANDUSE}")

    zones_with_data = (landuse['n_points'] > 0).sum()
    print(f"  {zones_with_data} CLC zones represented (out of 44 rows total)")

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures ...")

    print("  Fig 1 — Top-10 CLC zones by measurement count ...")
    plot_fig1_top10_zones(landuse, OUT_FIG1)

    print("  Fig 2 — Dual-axis loss rate + fraction captured ...")
    plot_fig2_loss_frac(landuse, OUT_FIG2)

    print("  Fig 3 — Relative autonomy losses (%) by CLC zone ...")
    plot_fig3_loss_pct(landuse, OUT_FIG3)

    print("  Fig 4 — Moving / Parked proportion by CLC zone ...")
    plot_fig4_moving_parked_by_zone(landuse, OUT_FIG4)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()