"""
================================================================================
SAL Method A/B/C Stability Diagnostic — Daily-Ratio Distributions & Leave-One-Day-Out
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Method C (mean of daily ratios) diverges from Method B more strongly, and with
higher variance, than Method A in several bins of sal_sensitivity_analysis.py
(e.g. Rural x July). Two explanations are possible:

  (1) GENUINE SIGNAL — day-to-day conditions (weather, driving pattern) are
      truly heterogeneous, and Method C, by using the sensor-day as the unit
      of observation, reveals variability that Methods A and B smooth away
      by treating every 1-minute row as an independent sample (which
      understates the sampling uncertainty, since minutes of the same day
      are strongly autocorrelated).

  (2) SMALL-SAMPLE ARTEFACT — the bin is fed by very few sensor-days, and a
      single atypical day (partial sensor fault, unusual GPS trace, one-off
      weather event) dominates the daily-ratio average.

The script cannot decide automatically; it produces the diagnostics for a
human judgement:

  1. A sensor-day count map (condition x month), to flag thin bins.
  2. The full distribution of daily ratios for the TOP_N_BINS most divergent
     bins (|Method C - Method B|), as strip / box plots.
  3. A leave-one-day-out (jackknife) check: for each of these bins, Method C
     is recomputed without its most extreme daily ratio. A large swing means
     the mean was driven by one outlier day (artefact); a small swing means
     broad, genuine inter-day variability.
  4. The same leave-one-sensor-day-out check for Method A (pooled ratio of
     sums) and Method B (mean of minute ratios) on the same bins: a single
     day can also weigh on A (its minutes enter the global sums) and on B (if
     it is a large share of the bin's minutes). Comparing the three
     max-influences (MC-4) shows whether the fragility of Method C is
     specific to it or a property of the bin's data.

A sensor-day enters a bin only with at least MIN_MINUTES_PER_DAY minutes in
that bin.

Figures produced
----------------
  MC-1 : Heatmap — number of sensor-days per (condition x month) bin
  MC-2 : Strip / box plot — daily-ratio distributions of the most divergent
         monthly-by-condition bins
  MC-3 : Leave-one-day-out — Method C mean with all days vs without the most
         extreme day, same bins
  MC-4 : Grouped bars — leave-one-sensor-day-out max-influence (pp),
         Method A vs B vs C, same bins

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
             sal_method_c_diagnostic_<figure>_<CAMPAIGN>_<variant_tag>.png
             sal_method_c_daily_ratios_<CAMPAIGN>_<variant_tag>.csv
                 (every daily ratio: condition, month, sensor, date, ratio,
                 minutes that day)
             sal_method_c_stability_table_<CAMPAIGN>_<variant_tag>.csv
                 (per bin: n_days, mean, std, jackknife max-influence for
                 Methods A, B and C)
No filtering on sun_elev or rad_calib is applied here (clean_data.py does it
upstream): choose the input suffix accordingly (e.g. 'sd_cleaned_hq').

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.0
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

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from clc_mapping import CLC_URBAN_RURAL

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR = STANDARD_DIR                  # CAMPAIGN = "standard"
TUKTUK_DATA_DIR = TUKTUK_DIR             # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "salmeth_outputs"

SHOW_FIGURES: bool = False

DELTA_T_H: float = 1.0 / 60.0

# No filtering on land_use_clcv, sun_elev or rad_calib here: clean_data.py
# applies it upstream (choose the input suffix accordingly).
LOCAL_TZ: str = 'Europe/Lisbon'


THRESHOLD_M: float = 30.0
R_EARTH_KM:  float = 6371.0

MIN_SENSOR_DAYS: int = 5
MIN_MINUTES:     int = 600

# Minimum minutes required WITHIN a single sensor-day for that day's ratio
# to be counted as a valid observation. Days with fewer minutes than this
# in the relevant bin are excluded before computing the daily ratio, to
# avoid a ratio computed on e.g. 2 stray minutes.
MIN_MINUTES_PER_DAY: int = 15

# Number of most-divergent monthly-by-condition bins to inspect in detail
TOP_N_BINS: int = 8

MONTH_NAMES: list[str] = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

COLOR_A = '#1565C0'
COLOR_B = '#E05C5C'
COLOR_C = '#2E7D32'
COLOR_OUTLIER = '#F4A800'

# ==============================================================================
# FILE RESOLUTION
# ==============================================================================

def build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    sensors = ['J30', 'J31', 'J32', 'J33']
    if campaign == "standard":
        return {
            sensor: data_dir / f"{sensor.lower()}_satellite_landuse_v2_{suffix}.csv"
            for sensor in sensors
        }
    elif campaign == "tuktuk":
        return {
            sensor: tuktuk_data_dir / f"{sensor.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
            for sensor in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR = OUTPUT_BASE_DIR / CAMPAIGN / _VARIANT_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_FILES = build_sensor_files(CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG)

# Column mapping reflecting the actual CSV schema for each sensor file:
#   Unnamed: 0, Human readable date (Europe/London), J3x lat, J3x lon,
#   J3x Rad raw, J3x Rad calib, ghi, ghi_clear, ghi_extra, bhi, dhi, dni,
#   sun_elev, sun_azim, land_use, land_use_str
SENSOR_COLS: dict[str, dict[str, str]] = {
    sensor: {
        'date': 'Human readable date (Europe/London)',
        'rad':  f'{sensor} Rad calib',
        'ghi':  'ghi',
        'elev': 'sun_elev',
        'land': 'land_use_clcv',
        'lat':  f'{sensor} lat',
        'lon':  f'{sensor} lon',
    }
    for sensor in ['J30', 'J31', 'J32', 'J33']
}


def show_or_close(fig: plt.Figure) -> None:
    """Show the figure interactively if SHOW_FIGURES is True, otherwise close it."""
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure to OUTPUT_DIR using the project naming convention."""
    path = OUTPUT_DIR / f"sal_method_c_diagnostic_{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved : {path}')
    show_or_close(fig)


# ==============================================================================
# HAVERSINE -- movement classification
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2, R=R_EARTH_KM):
    """Compute the great-circle distance in metres between consecutive GPS points."""
    phi1    = np.radians(lat1)
    phi2    = np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a)) * 1000


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print('=' * 70)
print('  SAL Method C Stability Diagnostic')
print(f'  CAMPAIGN     : {CAMPAIGN}')
print(f'  Data variant : {_VARIANT_TAG}')
print('=' * 70)

all_frames: list[pd.DataFrame] = []

for sensor, csv_path in SENSOR_FILES.items():
    cols = SENSOR_COLS[sensor]
    if not csv_path.exists():
        print(f'  [WARNING] Not found, skipping : {csv_path}')
        continue

    df = pd.read_csv(
        csv_path, low_memory=False,
        usecols=[cols['date'], cols['rad'], cols['ghi'],
                 cols['elev'], cols['land'],
                 cols['lat'],  cols['lon']],
    )
    df = df.rename(columns={
        cols['date']: 'datetime',
        cols['rad']:  'rad_calib',
        cols['ghi']:  'ghi',
        cols['elev']: 'sun_elev',
        cols['land']: 'land_use_clcv',
        cols['lat']:  'lat',
        cols['lon']:  'lon',
    })

    df['datetime']  = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df['datetime']  = df['datetime'].dt.tz_convert(LOCAL_TZ)
    df['rad_calib'] = pd.to_numeric(df['rad_calib'], errors='coerce')
    df['ghi']       = pd.to_numeric(df['ghi'],       errors='coerce')
    df['sun_elev']  = pd.to_numeric(df['sun_elev'],  errors='coerce')
    df['land_use_clcv'] = pd.to_numeric(df['land_use_clcv'], errors='coerce').astype('Int64')

    df = df.dropna(subset=['datetime', 'rad_calib', 'ghi', 'sun_elev'])

    df['sensor'] = sensor
    df['date']   = df['datetime'].dt.date
    df['month']  = df['datetime'].dt.month
    df['hour']   = df['datetime'].dt.hour

    all_frames.append(df)

if not all_frames:
    raise RuntimeError('No valid data found. Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX.')

data = pd.concat(all_frames, ignore_index=True)
print(f'Total valid measurements : {len(data):,}')

# ==============================================================================
# 2. PER-MINUTE ENERGY COLUMNS
# ==============================================================================
data['E_solar_Wh']  = data['rad_calib'] * DELTA_T_H
data['loss_irr_Wh'] = np.maximum(0.0, data['ghi'] - data['rad_calib']) * DELTA_T_H

data['clc_group'] = data['land_use_clcv'].map(CLC_URBAN_RURAL).fillna('Unknown')

# ==============================================================================
# 3. MOVEMENT CLASSIFICATION
# ==============================================================================
move_frames: list[pd.DataFrame] = []
for sensor, csv_path in SENSOR_FILES.items():
    cols_s = SENSOR_COLS[sensor]
    if not csv_path.exists():
        continue
    df_s = pd.read_csv(
        csv_path, low_memory=False,
        usecols=[cols_s['date'], cols_s['lat'], cols_s['lon']],
    )
    df_s = df_s.rename(columns={
        cols_s['date']: 'datetime',
        cols_s['lat']:  'lat',
        cols_s['lon']:  'lon',
    })
    df_s['datetime'] = pd.to_datetime(df_s['datetime'], utc=True, errors='coerce')
    df_s['datetime'] = df_s['datetime'].dt.tz_convert(LOCAL_TZ)
    for col in ['lat', 'lon']:
        df_s[col] = pd.to_numeric(df_s[col], errors='coerce')
    df_s = df_s.dropna(subset=['datetime', 'lat', 'lon']).sort_values('datetime').reset_index(drop=True)

    dist = haversine_vectorized(
        df_s['lat'].values[:-1], df_s['lon'].values[:-1],
        df_s['lat'].values[1:],  df_s['lon'].values[1:],
    )
    df_s = df_s.iloc[:-1].copy()
    df_s['moving_status'] = np.where(dist >= THRESHOLD_M, 'Moving', 'Parked')
    df_s['sensor']        = sensor
    move_frames.append(df_s[['datetime', 'sensor', 'moving_status']])

move_df = pd.concat(move_frames, ignore_index=True)
data    = data.merge(move_df, on=['datetime', 'sensor'], how='left')

# ==============================================================================
# 4. VALID MONTHS
# ==============================================================================
daily_counts = (
    data.groupby(['sensor', 'date', 'month'], as_index=False).size()
    .groupby('month', as_index=False).agg(n_sensor_days=('sensor', 'count'))
)
all_months = pd.DataFrame({'month': range(1, 13)})
daily_counts = all_months.merge(daily_counts, on='month', how='left')
sparse_mask  = daily_counts['n_sensor_days'].isna() | (daily_counts['n_sensor_days'] < MIN_SENSOR_DAYS)
VALID_MONTHS: set[int] = set(daily_counts.loc[~sparse_mask, 'month'].tolist())
print(f'Valid months : {sorted(VALID_MONTHS)}')

CONDITIONS = {
    'Urban':  data['clc_group'] == 'Urban',
    'Rural':  data['clc_group'] == 'Rural',
    'Parked': data['moving_status'] == 'Parked',
    'Moving': data['moving_status'] == 'Moving',
}

# ==============================================================================
# 5. AGGREGATION FUNCTIONS
# ==============================================================================

def method_A_pooled_ratio_of_sums(group: pd.DataFrame) -> float:
    """Pool all minutes and compute a single energy-weighted ratio."""
    total_loss  = group['loss_irr_Wh'].sum()
    total_solar = group['E_solar_Wh'].sum()
    denom = total_loss + total_solar
    return (total_loss / denom * 100.0) if denom > 0 else np.nan


def method_B_mean_of_minute_ratios(group: pd.DataFrame) -> float:
    """Simple mean of the per-minute ratio, equal weight per minute."""
    denom = group['loss_irr_Wh'] + group['E_solar_Wh']
    ratio = np.where(denom > 0, group['loss_irr_Wh'] / denom * 100.0, np.nan)
    ratio = ratio[~np.isnan(ratio)]
    return ratio.mean() if len(ratio) > 0 else np.nan


def daily_ratios(group: pd.DataFrame,
                  min_minutes_per_day: int = MIN_MINUTES_PER_DAY) -> pd.DataFrame:
    """
    Compute one relative-loss ratio per (sensor, date) within `group`.

    Returns a DataFrame with columns: sensor, date, ratio_pct, n_minutes.
    Days with fewer than `min_minutes_per_day` valid minutes in this bin are
    dropped, since a ratio computed on a handful of minutes is not a
    meaningful "daily" observation.
    """
    daily = (
        group.groupby(['sensor', 'date'])
        .agg(loss=('loss_irr_Wh', 'sum'),
             solar=('E_solar_Wh', 'sum'),
             n_minutes=('loss_irr_Wh', 'size'))
        .reset_index()
    )
    daily = daily[daily['n_minutes'] >= min_minutes_per_day].copy()
    denom = daily['loss'] + daily['solar']
    daily['ratio_pct'] = np.where(denom > 0, daily['loss'] / denom * 100.0, np.nan)
    daily = daily.dropna(subset=['ratio_pct'])
    return daily[['sensor', 'date', 'ratio_pct', 'n_minutes']]


def jackknife_max_influence(ratios: np.ndarray) -> tuple[float, float, float]:
    """
    Leave-one-out sensitivity of the mean of `ratios`.

    Returns (mean_all_days, mean_excluding_most_extreme_day, max_influence_pp)
    where max_influence_pp is the largest absolute change in the mean caused
    by removing any single day. A large value indicates the mean is fragile
    (driven by one outlier day); a small value indicates the mean reflects
    broad agreement across many days.
    """
    if len(ratios) < 2:
        return (ratios.mean() if len(ratios) else np.nan, np.nan, np.nan)

    mean_all = ratios.mean()
    loo_means = np.array([
        np.delete(ratios, i).mean() for i in range(len(ratios))
    ])
    influences = np.abs(loo_means - mean_all)
    max_idx = np.argmax(influences)
    return mean_all, loo_means[max_idx], influences[max_idx]


def _valid_sensor_days(group: pd.DataFrame,
                        min_minutes_per_day: int) -> pd.DataFrame:
    """
    Return the (sensor, date) pairs within `group` that have at least
    `min_minutes_per_day` minutes — the same day-inclusion rule used by
    daily_ratios() for Method C, applied here so that Methods A and B are
    jackknifed over the identical set of sensor-days for a fair comparison.
    """
    counts = group.groupby(['sensor', 'date']).size().reset_index(name='n_minutes')
    return counts[counts['n_minutes'] >= min_minutes_per_day][['sensor', 'date']]


def jackknife_by_day_method_A(
    group: pd.DataFrame,
    min_minutes_per_day: int = MIN_MINUTES_PER_DAY,
) -> tuple[float, float, float]:
    """
    Leave-one-sensor-day-out sensitivity of Method A (pooled ratio-of-sums),
    computed directly on the raw minute-level `group` — the natural extension
    of jackknife_max_influence() to a method that pools minutes rather than
    pre-aggregated daily ratios.

    For each sensor-day with at least `min_minutes_per_day` valid minutes
    (same rule as daily_ratios(), for a fair like-for-like comparison across
    methods), Method A is recomputed with that day's minutes excluded from
    the pooled sum entirely.

    Returns (value with all days included, value excluding the single most
    influential day, max_influence_pp). NaN influence if fewer than 2 valid
    sensor-days are available (jackknife is not meaningful with only 1 day).
    """
    valid_days = _valid_sensor_days(group, min_minutes_per_day)
    all_value = method_A_pooled_ratio_of_sums(group)
    if len(valid_days) < 2:
        return (all_value, np.nan, np.nan)

    loo_values = []
    for _, day_row in valid_days.iterrows():
        day_mask = ~((group['sensor'] == day_row['sensor'])
                     & (group['date'] == day_row['date']))
        loo_values.append(method_A_pooled_ratio_of_sums(group.loc[day_mask]))
    loo_values = np.array(loo_values, dtype=float)
    valid_loo = ~np.isnan(loo_values)
    if valid_loo.sum() == 0:
        return (all_value, np.nan, np.nan)

    influences = np.abs(loo_values[valid_loo] - all_value)
    max_idx = np.argmax(influences)
    return all_value, loo_values[valid_loo][max_idx], influences[max_idx]


def jackknife_by_day_method_B(
    group: pd.DataFrame,
    min_minutes_per_day: int = MIN_MINUTES_PER_DAY,
) -> tuple[float, float, float]:
    """
    Leave-one-sensor-day-out sensitivity of Method B (mean-of-minute-ratios),
    computed directly on the raw minute-level `group`.

    Same logic as jackknife_by_day_method_A(), applied to Method B: for each
    sensor-day with at least `min_minutes_per_day` valid minutes, Method B
    is recomputed over the remaining minutes with that day fully excluded.
    Unlike Method C's per-day pre-averaging, Method B already gives equal
    weight to every individual minute — this test asks whether one day's
    worth of minutes (which may be a large or small share of the bin's total
    minute count) still swings the equal-weighted mean disproportionately.

    Returns (value with all days included, value excluding the single most
    influential day, max_influence_pp).
    """
    valid_days = _valid_sensor_days(group, min_minutes_per_day)
    all_value = method_B_mean_of_minute_ratios(group)
    if len(valid_days) < 2:
        return (all_value, np.nan, np.nan)

    loo_values = []
    for _, day_row in valid_days.iterrows():
        day_mask = ~((group['sensor'] == day_row['sensor'])
                     & (group['date'] == day_row['date']))
        loo_values.append(method_B_mean_of_minute_ratios(group.loc[day_mask]))
    loo_values = np.array(loo_values, dtype=float)
    valid_loo = ~np.isnan(loo_values)
    if valid_loo.sum() == 0:
        return (all_value, np.nan, np.nan)

    influences = np.abs(loo_values[valid_loo] - all_value)
    max_idx = np.argmax(influences)
    return all_value, loo_values[valid_loo][max_idx], influences[max_idx]


# ==============================================================================
# 6. FIGURE MC-1 -- Sensor-day count heatmap (condition x month)
# ==============================================================================
print('\nGenerating MC-1 : Sensor-day count heatmap...')

count_rows = []
for cond_name, mask in CONDITIONS.items():
    for m in range(1, 13):
        sub = data[mask & (data['month'] == m)]
        d = daily_ratios(sub)
        count_rows.append({'condition': cond_name, 'month': m, 'n_days': len(d)})

df_counts = pd.DataFrame(count_rows)
pivot_counts = df_counts.pivot(index='condition', columns='month', values='n_days')
pivot_counts = pivot_counts.reindex(index=['Urban', 'Rural', 'Parked', 'Moving'],
                                     columns=range(1, 13))

fig, ax = plt.subplots(figsize=(14, 4.5))
im = ax.imshow(pivot_counts.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(12))
ax.set_xticklabels(MONTH_NAMES, fontsize=10)
ax.set_yticks(range(4))
ax.set_yticklabels(pivot_counts.index, fontsize=11)
ax.set_title('Number of Sensor-Days Contributing to Method C\n'
             'by Condition and Month  (fewer days = less stable daily-ratio mean)',
             fontsize=12, fontweight='bold')
for i in range(4):
    for j in range(12):
        v = pivot_counts.values[i, j]
        if not np.isnan(v):
            color = 'white' if v > pivot_counts.values.max() * 0.6 else '#222222'
            ax.text(j, i, f'{int(v)}', ha='center', va='center',
                    fontsize=9, color=color)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('N sensor-days', fontsize=10)
plt.tight_layout()
save_figure(fig, 'mc1_sensor_day_counts')

# ==============================================================================
# 7. IDENTIFY MOST DIVERGENT MONTHLY-BY-CONDITION BINS
# ==============================================================================
print('\nIdentifying most divergent bins (|Method C - Method B|)...')

divergence_rows = []
for cond_name, mask in CONDITIONS.items():
    for m in range(1, 13):
        if m not in VALID_MONTHS:
            continue
        sub = data[mask & (data['month'] == m)]
        if len(sub) < MIN_MINUTES:
            continue
        b = method_B_mean_of_minute_ratios(sub)
        d = daily_ratios(sub)
        if len(d) < 3:
            continue
        c = d['ratio_pct'].mean()
        divergence_rows.append({
            'condition': cond_name, 'month': m,
            'method_B': b, 'method_C': c,
            'delta_CB': c - b, 'n_days': len(d),
        })

df_divergence = pd.DataFrame(divergence_rows)
df_divergence['abs_delta'] = df_divergence['delta_CB'].abs()
top_bins = df_divergence.sort_values('abs_delta', ascending=False).head(TOP_N_BINS)

print('\nTop divergent bins:')
print(top_bins[['condition', 'month', 'method_B', 'method_C', 'delta_CB', 'n_days']]
      .to_string(index=False))

# ==============================================================================
# 8. FIGURE MC-2 -- Daily-ratio distributions for top divergent bins
# ==============================================================================
print('\nGenerating MC-2 : Daily-ratio distributions for divergent bins...')

fig, ax = plt.subplots(figsize=(14, 7))

box_data  = []
box_labels = []
bin_summaries = []

for _, row in top_bins.iterrows():
    cond_name, m = row['condition'], int(row['month'])
    mask = CONDITIONS[cond_name]
    sub = data[mask & (data['month'] == m)]
    d = daily_ratios(sub)

    box_data.append(d['ratio_pct'].values)
    box_labels.append(f"{cond_name}\n{MONTH_NAMES[m - 1]}\n(n={len(d)})")

    mean_all, mean_loo, max_influence = jackknife_max_influence(d['ratio_pct'].values)
    a_all, a_loo, a_influence = jackknife_by_day_method_A(sub)
    b_all, b_loo, b_influence = jackknife_by_day_method_B(sub)
    bin_summaries.append({
        'condition': cond_name, 'month': MONTH_NAMES[m - 1],
        'n_days': len(d),
        'method_A_mean': a_all,
        'method_A_excl_extreme_day': a_loo,
        'jackknife_max_influence_A_pp': a_influence,
        'method_B_mean': b_all,
        'method_B_excl_extreme_day': b_loo,
        'jackknife_max_influence_B_pp': b_influence,
        'method_C_mean': mean_all,
        'method_C_excl_extreme_day': mean_loo,
        'jackknife_max_influence_C_pp': max_influence,
    })

bp = ax.boxplot(box_data, patch_artist=True, widths=0.55,
                medianprops=dict(color='#333333', linewidth=1.8),
                boxprops=dict(facecolor=COLOR_C, alpha=0.55, linewidth=1.2),
                whiskerprops=dict(linewidth=1.2, linestyle='--'),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=4, alpha=0.5, color='#888888'))

# Overlay individual daily-ratio points (strip plot) for full transparency
rng = np.random.default_rng(42)
for i, vals in enumerate(box_data):
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
               color=COLOR_C, alpha=0.6, s=22, zorder=4, edgecolor='white', linewidth=0.4)

# Overlay Method B for reference
for i, row in enumerate(top_bins.itertuples()):
    ax.hlines(row.method_B, i + 0.65, i + 1.35, color=COLOR_B, linewidth=2.2,
              zorder=5, label='Method B (current)' if i == 0 else None)

ax.set_xticks(range(1, len(box_labels) + 1))
ax.set_xticklabels(box_labels, fontsize=9)
ax.set_ylabel('Relative Shading Loss [%]', fontsize=12, labelpad=8)
ax.set_title(f'Daily-Ratio Distributions -- Top {TOP_N_BINS} Most Divergent Bins\n'
             '(green dots: individual sensor-day ratios | red line: Method B for reference)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right', framealpha=0.88)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_figure(fig, 'mc2_daily_ratio_distributions')

# ==============================================================================
# 9. FIGURE MC-3 -- Leave-one-day-out sensitivity
# ==============================================================================
print('Generating MC-3 : Leave-one-day-out sensitivity...')

df_summary = pd.DataFrame(bin_summaries)

x = np.arange(len(df_summary))
w = 0.35

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.bar(x - w / 2, df_summary['method_C_mean'].values, w,
       label='Method C -- all sensor-days', color=COLOR_C, alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.bar(x + w / 2, df_summary['method_C_excl_extreme_day'].values, w,
       label='Method C -- excluding single most extreme day', color=COLOR_OUTLIER,
       alpha=0.85, edgecolor='white', linewidth=0.5)

for i, row in df_summary.iterrows():
    ax.plot([i - w / 2, i + w / 2], [row['method_B_mean'], row['method_B_mean']],
            color=COLOR_B, linewidth=2.0, zorder=5,
            label='Method B (current, all days)' if i == 0 else None)

ax.set_xticks(x)
ax.set_xticklabels([f"{r['condition']}\n{r['month']}" for _, r in df_summary.iterrows()],
                    fontsize=9)
ax.set_ylabel('Relative Shading Loss [%]', fontsize=12, labelpad=8)
ax.set_title('Leave-One-Day-Out Sensitivity of Method C\n'
             'Large gap between bars = mean driven by a single day (fragile);'
             ' small gap = broad agreement across days',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right', framealpha=0.88)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)

for i, row in df_summary.iterrows():
    ax.text(i, max(row['method_C_mean'], row['method_C_excl_extreme_day']) + 1.5,
            f"Δ={row['jackknife_max_influence_C_pp']:.1f}pp\n(n={row['n_days']})",
            ha='center', fontsize=8, color='#444444')

plt.tight_layout()
save_figure(fig, 'mc3_leave_one_day_out')

# ==============================================================================
# 9b. FIGURE MC-4 -- Leave-one-day-out max-influence, Method A vs B vs C
# ==============================================================================
print('Generating MC-4 : Method A vs B vs C leave-one-day-out comparison...')

x4 = np.arange(len(df_summary))
w4 = 0.26

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.bar(x4 - w4, df_summary['jackknife_max_influence_A_pp'].values, w4,
       label='Method A -- pooled ratio-of-sums', color=COLOR_A, alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.bar(x4,      df_summary['jackknife_max_influence_B_pp'].values, w4,
       label='Method B -- mean-of-minute-ratios', color=COLOR_B, alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.bar(x4 + w4, df_summary['jackknife_max_influence_C_pp'].values, w4,
       label='Method C -- mean-of-daily-ratios', color=COLOR_C, alpha=0.85,
       edgecolor='white', linewidth=0.5)

ax.set_xticks(x4)
ax.set_xticklabels([f"{r['condition']}\n{r['month']}\n(n={int(r['n_days'])})"
                    for _, r in df_summary.iterrows()], fontsize=9)
ax.set_ylabel('Leave-One-Sensor-Day-Out Max Influence [pp]', fontsize=11, labelpad=8)
ax.set_title('Single-Day Fragility Across Methods -- Same Bins as MC-2/MC-3\n'
             'A tall Method-C bar with short A/B bars means the fragility is specific '
             'to daily pooling, not a shared property of the underlying data',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper right', framealpha=0.88)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_figure(fig, 'mc4_methodABC_fragility_comparison')

# ==============================================================================
# 10. EXPORT TABLES
# ==============================================================================
print('\nExporting tables...')

# 10a. Every individual daily ratio, for the top divergent bins
detail_rows = []
for _, row in top_bins.iterrows():
    cond_name, m = row['condition'], int(row['month'])
    mask = CONDITIONS[cond_name]
    sub = data[mask & (data['month'] == m)]
    d = daily_ratios(sub)
    for _, drow in d.iterrows():
        detail_rows.append({
            'condition': cond_name, 'month': MONTH_NAMES[m - 1],
            'sensor': drow['sensor'], 'date': drow['date'],
            'ratio_pct': round(drow['ratio_pct'], 3),
            'n_minutes_that_day': int(drow['n_minutes']),
        })

df_detail = pd.DataFrame(detail_rows)
detail_path = OUTPUT_DIR / f'sal_method_c_daily_ratios_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_detail.to_csv(detail_path, index=False)
print(f'  Saved : {detail_path}')

# 10b. Stability summary table
for col in ['method_A_mean', 'method_A_excl_extreme_day', 'jackknife_max_influence_A_pp',
            'method_B_mean', 'method_B_excl_extreme_day', 'jackknife_max_influence_B_pp',
            'method_C_mean', 'method_C_excl_extreme_day', 'jackknife_max_influence_C_pp']:
    df_summary[col] = df_summary[col].round(3)
summary_path = OUTPUT_DIR / f'sal_method_c_stability_table_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_summary.to_csv(summary_path, index=False)
print(f'  Saved : {summary_path}')

# ==============================================================================
# 11. FINAL SUMMARY
# ==============================================================================
print('\n' + '=' * 116)
print('  STABILITY SUMMARY -- Top divergent bins -- Leave-One-Sensor-Day-Out Max Influence, A / B / C')
print('-' * 116)
print(f"  {'Condition':<10} {'Month':<6} {'N days':>7} "
      f"{'Max infl. A':>12} {'Max infl. B':>12} {'Max infl. C':>12} "
      f"{'Method B':>10} {'Method C':>10}")
print('-' * 116)
for _, row in df_summary.iterrows():
    print(f"  {row['condition']:<10} {row['month']:<6} {int(row['n_days']):>7} "
          f"{row['jackknife_max_influence_A_pp']:>11.2f}pp "
          f"{row['jackknife_max_influence_B_pp']:>11.2f}pp "
          f"{row['jackknife_max_influence_C_pp']:>11.2f}pp "
          f"{row['method_B_mean']:>9.2f}% {row['method_C_mean']:>9.2f}%")
print('=' * 116)

print('\n  READING GUIDE:')
print('  A small "max influence" value means that method\'s estimate for that')
print('  bin is supported by broad agreement across many sensor-days -- any')
print('  divergence between methods likely reflects genuine inter-day variability.')
print('  A large "max influence" value (several percentage points) means the')
print('  estimate is dominated by a single day -- treat that bin\'s value with')
print('  caution, or raise MIN_DAILY_RATIOS before reporting it.')
print('  Comparing the three columns (Fig. MC-4): if Method C\'s max influence')
print('  is much larger than A\'s and B\'s for the same bin, the fragility is')
print('  specific to daily pooling (small-sample artefact, explanation 2 above).')
print('  If A and B are similarly fragile, the underlying minute-level data for')
print('  that bin is itself dominated by one day, regardless of aggregation')
print('  method (genuine signal, explanation 1 above).')

print('\nDone.')
