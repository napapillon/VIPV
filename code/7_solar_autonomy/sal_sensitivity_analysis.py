"""
================================================================================
SAL Sensitivity Analysis — Three Relative-Loss Aggregation Methods (A/B/C)
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Reproduces the hourly / monthly / by-condition relative-loss figures of the
SAL analysis with the three aggregation methods overlaid, to judge where and
how strongly the aggregation choice affects the reported relative shading
losses.

Three methods compared
----------------------
  METHOD A — Pooled Ratio-of-Sums (minute-level pooling)
      All 1-minute rows of a bin (e.g. all "Urban, 07:00" minutes of the
      dataset) are pooled; total loss and total captured energy are summed
      once, then divided:
          loss_rel_A = Σ loss_irr / Σ (loss_irr + E_solar) × 100
      Each minute weighs in proportion to its own GHI: a single very sunny
      period can dominate the bin.

  METHOD B — Mean-of-Minute-Ratios (solar_autonomy_losses_method_b.py)
      The per-minute ratio is computed first, then averaged with equal
      weight per minute:
          loss_rel_B = mean( loss_irr_i / (loss_irr_i + E_solar_i) )
      Low-GHI minutes (near sunrise / sunset) can weigh disproportionately
      although their absolute energy is small.

  METHOD C — Mean-of-Daily-Ratios (sensor-day pooling)
      For each (sensor, date), loss and captured energy are summed over the
      minutes of that day in the bin, giving one ratio per sensor-day; the
      daily ratios are averaged with equal weight per sensor-day (the
      sensor-day is also the unit of MIN_SENSOR_DAYS and of Figures 1-3 of
      the solar_autonomy_losses* scripts).
          loss_rel_C = mean_over_days( Σ loss_day / Σ (loss + captured)_day )
      Within-day pooling smooths minute-level energy variance (like A), but
      every sensor-day counts the same whatever its total energy (like B, at
      the day level).

Method C bins need at least MIN_DAILY_RATIOS sensor-day ratios; Method A / B
bins at least MIN_MINUTES minutes; months at least MIN_SENSOR_DAYS
sensor-days.

Figures produced
----------------
  SA-1  : Monthly mean relative loss — A vs B vs C — grouped bar chart
  SA-2  : Monthly difference from Method B — (A-B) and (C-B) — grouped bars
  SA-3  : Hourly relative loss — Urban panel — A vs B vs C
  SA-4  : Hourly relative loss — Rural panel — A vs B vs C
  SA-5  : Hourly relative loss — Parked panel — A vs B vs C
  SA-6  : Hourly relative loss — Moving panel — A vs B vs C
  SA-7  : Monthly relative loss by land-use — Urban panel — A vs B vs C
  SA-8  : Monthly relative loss by land-use — Rural panel — A vs B vs C
  SA-9  : Monthly relative loss by movement — Parked panel — A vs B vs C
  SA-10 : Monthly relative loss by movement — Moving panel — A vs B vs C
  SA-11 : Scatter — Method A vs Method B, across all hourly / monthly bins
  SA-12 : Scatter — Method C vs Method B, across all hourly / monthly bins

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
             sal_sensitivity_v2_<figure>_<CAMPAIGN>_<variant_tag>.png
             sal_sensitivity_v2_table_<CAMPAIGN>_<variant_tag>.csv
                 (A, B, C and their differences per bin)
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

OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "salsa_outputs"

SHOW_FIGURES: bool = False

DELTA_T_H: float = 1.0 / 60.0   # 1-minute data -> hours per interval

# No filtering on land_use_clcv, sun_elev or rad_calib here: clean_data.py
# applies it upstream (choose the input suffix accordingly).
LOCAL_TZ: str = 'Europe/Lisbon'


THRESHOLD_M: float = 30.0    # metres -- Haversine movement threshold
R_EARTH_KM:  float = 6371.0

MIN_SENSOR_DAYS: int = 5
MIN_MINUTES:     int = 600

# Minimum number of daily ratios required for Method C to be considered
# reliable in a given bin (hour x condition, or month x condition).
MIN_DAILY_RATIOS: int = 5

HOUR_START_PLOT: int  = 6
HOUR_END_PLOT:   int  = 20
HOURS_PLOT: list[int] = list(range(HOUR_START_PLOT, HOUR_END_PLOT + 1))

MONTH_NAMES: list[str] = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

# Colour palette -- consistent with the project-wide convention
COLOR_A = '#1565C0'   # Method A -- pooled ratio-of-sums (blue)
COLOR_B = '#E05C5C'   # Method B -- mean-of-minute-ratios (red)
COLOR_C = '#2E7D32'   # Method C -- mean-of-daily-ratios (green)

COLOR_URBAN  = '#E05C5C'
COLOR_RURAL  = '#2E7D32'
COLOR_PARKED = '#F4A800'
COLOR_MOVING = '#1565C0'

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
    path = OUTPUT_DIR / f"sal_sensitivity_v2_{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"
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
    return 2 * R * np.arcsin(np.sqrt(a)) * 1000   # km -> metres


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print('=' * 70)
print('  SAL Sensitivity Analysis -- Methods A / B / C')
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

denom_min = data['loss_irr_Wh'] + data['E_solar_Wh']
data['loss_rel_min_pct'] = np.where(
    denom_min > 0, data['loss_irr_Wh'] / denom_min * 100.0, np.nan
)

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

# ==============================================================================
# 5. THREE AGGREGATION METHODS
# ==============================================================================

def method_A_pooled_ratio_of_sums(group: pd.DataFrame) -> float:
    """
    Method A -- pool all minutes and compute a single energy-weighted ratio.
    Weight of each minute is implicitly proportional to its own GHI.
    """
    total_loss  = group['loss_irr_Wh'].sum()
    total_solar = group['E_solar_Wh'].sum()
    denom = total_loss + total_solar
    return (total_loss / denom * 100.0) if denom > 0 else np.nan


def method_B_mean_of_minute_ratios(group: pd.DataFrame) -> float:
    """
    Method B -- simple mean of the per-minute ratio, equal weight per minute.
    """
    vals = group['loss_rel_min_pct'].dropna()
    return vals.mean() if len(vals) > 0 else np.nan


def method_C_mean_of_daily_ratios(group: pd.DataFrame,
                                   min_days: int = MIN_DAILY_RATIOS) -> float:
    """
    Method C -- compute one ratio-of-sums PER SENSOR-DAY (using only the
    minutes of that day that fall within the group), then average those
    daily ratios with equal weight per sensor-day.

    Returns NaN if fewer than `min_days` valid sensor-day ratios exist,
    to avoid an unstable estimate driven by only one or two days.
    """
    daily = (
        group.groupby(['sensor', 'date'])
        .agg(loss=('loss_irr_Wh', 'sum'), solar=('E_solar_Wh', 'sum'))
        .reset_index()
    )
    denom = daily['loss'] + daily['solar']
    daily_ratio = np.where(denom > 0, daily['loss'] / denom * 100.0, np.nan)
    daily_ratio = daily_ratio[~np.isnan(daily_ratio)]
    if len(daily_ratio) < min_days:
        return np.nan
    return daily_ratio.mean()


def compute_ABC(group: pd.DataFrame) -> tuple[float, float, float]:
    """Convenience wrapper returning (Method A, Method B, Method C) for a group."""
    return (
        method_A_pooled_ratio_of_sums(group),
        method_B_mean_of_minute_ratios(group),
        method_C_mean_of_daily_ratios(group),
    )


# ==============================================================================
# 6. MONTHLY (ALL DATA POOLED)
# ==============================================================================
print('\nComputing monthly A / B / C (all data)...')

monthly_rows = []
for m in range(1, 13):
    sub = data[data['month'] == m]
    if m not in VALID_MONTHS or len(sub) < MIN_MINUTES:
        monthly_rows.append({'month': m, 'method_A': np.nan, 'method_B': np.nan,
                              'method_C': np.nan, 'n_minutes': len(sub)})
        continue
    a, b, c = compute_ABC(sub)
    monthly_rows.append({'month': m, 'method_A': a, 'method_B': b,
                          'method_C': c, 'n_minutes': len(sub)})

df_monthly = pd.DataFrame(monthly_rows)
df_monthly['delta_AB'] = df_monthly['method_A'] - df_monthly['method_B']
df_monthly['delta_CB'] = df_monthly['method_C'] - df_monthly['method_B']

# ==============================================================================
# 7. HOURLY, BY CONDITION
# ==============================================================================
print('Computing hourly A / B / C by condition...')

CONDITIONS = {
    'Urban':  data['clc_group'] == 'Urban',
    'Rural':  data['clc_group'] == 'Rural',
    'Parked': data['moving_status'] == 'Parked',
    'Moving': data['moving_status'] == 'Moving',
}

hourly_rows = []
for cond_name, mask in CONDITIONS.items():
    for h in HOURS_PLOT:
        sub = data[mask & (data['hour'] == h)]
        n = len(sub)
        if n < MIN_MINUTES:
            hourly_rows.append({'condition': cond_name, 'hour': h,
                                 'method_A': np.nan, 'method_B': np.nan,
                                 'method_C': np.nan, 'n_minutes': n})
            continue
        a, b, c = compute_ABC(sub)
        hourly_rows.append({'condition': cond_name, 'hour': h,
                             'method_A': a, 'method_B': b, 'method_C': c,
                             'n_minutes': n})

df_hourly = pd.DataFrame(hourly_rows)
df_hourly['delta_AB'] = df_hourly['method_A'] - df_hourly['method_B']
df_hourly['delta_CB'] = df_hourly['method_C'] - df_hourly['method_B']

# ==============================================================================
# 8. MONTHLY, BY CONDITION
# ==============================================================================
print('Computing monthly A / B / C by condition...')

monthly_cond_rows = []
for cond_name, mask in CONDITIONS.items():
    for m in range(1, 13):
        sub = data[mask & (data['month'] == m)]
        n = len(sub)
        if m not in VALID_MONTHS or n < MIN_MINUTES:
            monthly_cond_rows.append({'condition': cond_name, 'month': m,
                                       'method_A': np.nan, 'method_B': np.nan,
                                       'method_C': np.nan, 'n_minutes': n})
            continue
        a, b, c = compute_ABC(sub)
        monthly_cond_rows.append({'condition': cond_name, 'month': m,
                                   'method_A': a, 'method_B': b, 'method_C': c,
                                   'n_minutes': n})

df_monthly_cond = pd.DataFrame(monthly_cond_rows)
df_monthly_cond['delta_AB'] = df_monthly_cond['method_A'] - df_monthly_cond['method_B']
df_monthly_cond['delta_CB'] = df_monthly_cond['method_C'] - df_monthly_cond['method_B']

# ==============================================================================
# 9. PRINT SUMMARY TABLE
# ==============================================================================
print('\n' + '=' * 90)
print('  MONTHLY SUMMARY -- Relative Loss [%]')
print('-' * 90)
print(f"  {'Month':<8} {'Method A':>12} {'Method B':>12} {'Method C':>12} "
      f"{'A-B':>8} {'C-B':>8} {'N min':>10}")
print('-' * 90)
for _, row in df_monthly.iterrows():
    mname = MONTH_NAMES[int(row['month']) - 1]
    if np.isnan(row['method_A']):
        print(f"  {mname:<8} {'N/A':>12} {'N/A':>12} {'N/A':>12} "
              f"{'N/A':>8} {'N/A':>8} {int(row['n_minutes']):>10,}")
        continue
    print(f"  {mname:<8} {row['method_A']:>11.2f}% {row['method_B']:>11.2f}% "
          f"{row['method_C']:>11.2f}% {row['delta_AB']:>+7.2f} "
          f"{row['delta_CB']:>+7.2f} {int(row['n_minutes']):>10,}")
print('=' * 90)

print('\n  KEY:')
print('  Method A = pooled ratio-of-sums (minute-level energy weighting)')
print('  Method B = mean-of-minute-ratios            [current SAL figs 4-9]')
print('  Method C = mean-of-daily-ratios (sensor-day pooling & equal weight)')


# ==============================================================================
# 10. FIGURE SA-1 -- Monthly A vs B vs C
# ==============================================================================
print('\nGenerating SA-1 : Monthly A vs B vs C...')

x = np.arange(12)
w = 0.26

fig, ax = plt.subplots(figsize=(15, 6.5))
a_vals = df_monthly['method_A'].values
b_vals = df_monthly['method_B'].values
c_vals = df_monthly['method_C'].values

ax.bar(x - w, a_vals, w, label='Method A -- Pooled Ratio-of-Sums',
       color=COLOR_A, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.bar(x,     b_vals, w, label='Method B -- Mean-of-Minute-Ratios (current)',
       color=COLOR_B, alpha=0.80, edgecolor='white', linewidth=0.5)
ax.bar(x + w, c_vals, w, label='Method C -- Mean-of-Daily-Ratios',
       color=COLOR_C, alpha=0.80, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(MONTH_NAMES, fontsize=11)
ax.set_xlabel('Month', fontsize=12, labelpad=8)
ax.set_ylabel('Relative Shading Loss [%]', fontsize=12, labelpad=8)
finite = np.concatenate([a_vals[~np.isnan(a_vals)], b_vals[~np.isnan(b_vals)],
                          c_vals[~np.isnan(c_vals)]])
ax.set_ylim(0, (finite.max() * 1.25 + 2) if len(finite) else 100)
ax.set_title('Monthly Relative Shading Loss -- Three Aggregation Methods',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left', framealpha=0.88)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.01,
         'A: pooled minutes, one ratio  |  B: mean of per-minute ratios  |  '
         'C: mean of per-sensor-day ratios',
         ha='center', fontsize=8.5, color='#666666', style='italic')
plt.tight_layout()
save_figure(fig, 'sa1_monthly_ABC')

# ==============================================================================
# 11. FIGURE SA-2 -- Monthly delta from Method B: (A-B) and (C-B)
# ==============================================================================
print('Generating SA-2 : Monthly delta from Method B...')

fig, ax = plt.subplots(figsize=(15, 6))
w2 = 0.35
delta_ab = df_monthly['delta_AB'].values
delta_cb = df_monthly['delta_CB'].values

ax.bar(x - w2 / 2, delta_ab, w2, label='A − B', color=COLOR_A, alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.bar(x + w2 / 2, delta_cb, w2, label='C − B', color=COLOR_C, alpha=0.85,
       edgecolor='white', linewidth=0.5)
ax.axhline(0, color='#333333', linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(MONTH_NAMES, fontsize=11)
ax.set_xlabel('Month', fontsize=12, labelpad=8)
ax.set_ylabel('Difference from Method B [pp]', fontsize=12, labelpad=8)
ax.set_title('Monthly Deviation of Methods A and C from Method B (current)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.88)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_figure(fig, 'sa2_monthly_delta')

# ==============================================================================
# 12. FIGURES SA-3 to SA-6 -- Hourly, by condition
# ==============================================================================
print('Generating SA-3 to SA-6 : Hourly comparison by condition...')

fig_idx = 3
for cond_name in ['Urban', 'Rural', 'Parked', 'Moving']:
    sub = df_hourly[df_hourly['condition'] == cond_name].sort_values('hour')
    hours   = sub['hour'].values
    a_h, b_h, c_h = sub['method_A'].values, sub['method_B'].values, sub['method_C'].values
    d_ab, d_cb    = sub['delta_AB'].values, sub['delta_CB'].values

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9.5),
                                          gridspec_kw={'height_ratios': [2, 1]})

    ax_top.plot(hours, a_h, color=COLOR_A, linewidth=2.2, marker='o',
                markersize=6, label='Method A -- Pooled Ratio-of-Sums', zorder=4)
    ax_top.plot(hours, b_h, color=COLOR_B, linewidth=2.2, marker='s',
                markersize=6, linestyle='--', label='Method B -- Mean-of-Minute-Ratios (current)',
                zorder=3)
    ax_top.plot(hours, c_h, color=COLOR_C, linewidth=2.2, marker='^',
                markersize=6, linestyle=':', label='Method C -- Mean-of-Daily-Ratios', zorder=3)

    ax_top.set_xticks(hours)
    ax_top.set_xticklabels([f'{h:02d}:00' for h in hours], fontsize=9, rotation=45, ha='right')
    ax_top.set_ylabel('Relative Shading Loss [%]', fontsize=11, labelpad=8)
    finite_h = np.concatenate([v[~np.isnan(v)] for v in [a_h, b_h, c_h] if len(v)])
    ax_top.set_ylim(0, (finite_h.max() * 1.25 + 2) if len(finite_h) else 100)
    ax_top.set_title(f'Hourly Relative Shading Loss -- {cond_name} -- Methods A / B / C',
                     fontsize=12, fontweight='bold')
    ax_top.legend(fontsize=9.5, loc='upper left', framealpha=0.88)
    ax_top.grid(axis='y', linestyle='--', alpha=0.4)
    ax_top.spines[['top', 'right']].set_visible(False)

    w3 = 0.38
    ax_bot.bar(np.array(hours) - w3 / 2, d_ab, w3, label='A − B', color=COLOR_A,
               alpha=0.8, edgecolor='white', linewidth=0.5)
    ax_bot.bar(np.array(hours) + w3 / 2, d_cb, w3, label='C − B', color=COLOR_C,
               alpha=0.8, edgecolor='white', linewidth=0.5)
    ax_bot.axhline(0, color='#333333', linewidth=1.0)
    ax_bot.set_xticks(hours)
    ax_bot.set_xticklabels([f'{h:02d}:00' for h in hours], fontsize=9, rotation=45, ha='right')
    ax_bot.set_xlabel('Hour of day (local time -- Europe/Lisbon)', fontsize=11, labelpad=8)
    ax_bot.set_ylabel('Difference from B [pp]', fontsize=10, labelpad=8)
    ax_bot.set_title('Deviation from Method B (current)', fontsize=10, style='italic')
    ax_bot.legend(fontsize=9, framealpha=0.88)
    ax_bot.grid(axis='y', linestyle='--', alpha=0.4)
    ax_bot.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    save_figure(fig, f'sa{fig_idx}_hourly_{cond_name.lower()}')
    fig_idx += 1

# ==============================================================================
# 13. FIGURES SA-7 to SA-10 -- Monthly, by condition
# ==============================================================================
print('Generating SA-7 to SA-10 : Monthly comparison by condition...')

for cond_name in ['Urban', 'Rural', 'Parked', 'Moving']:
    sub = df_monthly_cond[df_monthly_cond['condition'] == cond_name].sort_values('month')
    months_x = sub['month'].values - 1
    a_m, b_m, c_m = sub['method_A'].values, sub['method_B'].values, sub['method_C'].values
    d_ab, d_cb    = sub['delta_AB'].values, sub['delta_CB'].values

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9.5),
                                          gridspec_kw={'height_ratios': [2, 1]})

    ax_top.bar(months_x - w, a_m, w, label='Method A -- Pooled Ratio-of-Sums',
               color=COLOR_A, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_top.bar(months_x,     b_m, w, label='Method B -- Mean-of-Minute-Ratios (current)',
               color=COLOR_B, alpha=0.80, edgecolor='white', linewidth=0.5)
    ax_top.bar(months_x + w, c_m, w, label='Method C -- Mean-of-Daily-Ratios',
               color=COLOR_C, alpha=0.80, edgecolor='white', linewidth=0.5)

    ax_top.set_xticks(months_x)
    ax_top.set_xticklabels(MONTH_NAMES, fontsize=11)
    ax_top.set_ylabel('Relative Shading Loss [%]', fontsize=11, labelpad=8)
    finite_m = np.concatenate([v[~np.isnan(v)] for v in [a_m, b_m, c_m] if len(v)])
    ax_top.set_ylim(0, (finite_m.max() * 1.25 + 2) if len(finite_m) else 100)
    ax_top.set_title(f'Monthly Relative Shading Loss -- {cond_name} -- Methods A / B / C',
                     fontsize=12, fontweight='bold')
    ax_top.legend(fontsize=9.5, loc='upper left', framealpha=0.88)
    ax_top.grid(axis='y', linestyle='--', alpha=0.4)
    ax_top.spines[['top', 'right']].set_visible(False)

    ax_bot.bar(months_x - w2 / 2, d_ab, w2, label='A − B', color=COLOR_A,
               alpha=0.8, edgecolor='white', linewidth=0.5)
    ax_bot.bar(months_x + w2 / 2, d_cb, w2, label='C − B', color=COLOR_C,
               alpha=0.8, edgecolor='white', linewidth=0.5)
    ax_bot.axhline(0, color='#333333', linewidth=1.0)
    ax_bot.set_xticks(months_x)
    ax_bot.set_xticklabels(MONTH_NAMES, fontsize=11)
    ax_bot.set_xlabel('Month', fontsize=11, labelpad=8)
    ax_bot.set_ylabel('Difference from B [pp]', fontsize=10, labelpad=8)
    ax_bot.set_title('Deviation from Method B (current)', fontsize=10, style='italic')
    ax_bot.legend(fontsize=9, framealpha=0.88)
    ax_bot.grid(axis='y', linestyle='--', alpha=0.4)
    ax_bot.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    save_figure(fig, f'sa{fig_idx}_monthly_{cond_name.lower()}')
    fig_idx += 1

# ==============================================================================
# 14. FIGURE SA-11 -- Scatter Method A vs Method B
# ==============================================================================
print('Generating SA-11 : Scatter A vs B...')

all_cells = pd.concat([
    df_hourly[['method_A', 'method_B', 'method_C', 'condition']],
    df_monthly_cond[['method_A', 'method_B', 'method_C', 'condition']],
], ignore_index=True).dropna(subset=['method_A', 'method_B'])

condition_colors = {
    'Urban': COLOR_URBAN, 'Rural': COLOR_RURAL,
    'Parked': COLOR_PARKED, 'Moving': COLOR_MOVING,
}

fig, ax = plt.subplots(figsize=(8, 8))
for cond_name, color in condition_colors.items():
    sel = all_cells[all_cells['condition'] == cond_name]
    ax.scatter(sel['method_B'], sel['method_A'], c=color, alpha=0.65, s=30, label=cond_name)

lim = max(all_cells['method_A'].max(), all_cells['method_B'].max()) * 1.05 + 2
ax.plot([0, lim], [0, lim], color='#333333', linewidth=1.2, linestyle='--', label='1:1 line')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel('Method B -- Mean-of-Minute-Ratios [%] (current)', fontsize=11)
ax.set_ylabel('Method A -- Pooled Ratio-of-Sums [%]', fontsize=11)
ax.set_title('Method A vs Method B\nacross all hourly & monthly bins',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.88)
ax.grid(linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_figure(fig, 'sa11_scatter_A_vs_B')

# ==============================================================================
# 15. FIGURE SA-12 -- Scatter Method C vs Method B
# ==============================================================================
print('Generating SA-12 : Scatter C vs B...')

all_cells_C = pd.concat([
    df_hourly[['method_C', 'method_B', 'condition']],
    df_monthly_cond[['method_C', 'method_B', 'condition']],
], ignore_index=True).dropna(subset=['method_C', 'method_B'])

fig, ax = plt.subplots(figsize=(8, 8))
for cond_name, color in condition_colors.items():
    sel = all_cells_C[all_cells_C['condition'] == cond_name]
    ax.scatter(sel['method_B'], sel['method_C'], c=color, alpha=0.65, s=30, label=cond_name)

lim_c = max(all_cells_C['method_C'].max(), all_cells_C['method_B'].max()) * 1.05 + 2
ax.plot([0, lim_c], [0, lim_c], color='#333333', linewidth=1.2, linestyle='--', label='1:1 line')
ax.set_xlim(0, lim_c)
ax.set_ylim(0, lim_c)
ax.set_xlabel('Method B -- Mean-of-Minute-Ratios [%] (current)', fontsize=11)
ax.set_ylabel('Method C -- Mean-of-Daily-Ratios [%]', fontsize=11)
ax.set_title('Method C vs Method B\nacross all hourly & monthly bins',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.88)
ax.grid(linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.01,
         'Points below the 1:1 line -> Method C reports lower relative loss than Method B',
         ha='center', fontsize=9, color='#666666', style='italic')
plt.tight_layout()
save_figure(fig, 'sa12_scatter_C_vs_B')

# ==============================================================================
# 16. EXPORT NUMERICAL SUMMARY TABLE
# ==============================================================================
print('\nExporting numerical summary table...')

rows_out: list[dict] = []

for _, row in df_monthly.iterrows():
    rows_out.append({
        'context': 'monthly_all', 'condition': 'All',
        'bin_label': MONTH_NAMES[int(row['month']) - 1], 'bin_value': int(row['month']),
        'n_minutes': int(row['n_minutes']),
        'method_A_pct': row['method_A'], 'method_B_pct': row['method_B'],
        'method_C_pct': row['method_C'],
        'delta_AB_pp': row['delta_AB'], 'delta_CB_pp': row['delta_CB'],
    })

for _, row in df_hourly.iterrows():
    rows_out.append({
        'context': 'hourly_by_condition', 'condition': row['condition'],
        'bin_label': f"{int(row['hour']):02d}:00", 'bin_value': int(row['hour']),
        'n_minutes': int(row['n_minutes']),
        'method_A_pct': row['method_A'], 'method_B_pct': row['method_B'],
        'method_C_pct': row['method_C'],
        'delta_AB_pp': row['delta_AB'], 'delta_CB_pp': row['delta_CB'],
    })

for _, row in df_monthly_cond.iterrows():
    rows_out.append({
        'context': 'monthly_by_condition', 'condition': row['condition'],
        'bin_label': MONTH_NAMES[int(row['month']) - 1], 'bin_value': int(row['month']),
        'n_minutes': int(row['n_minutes']),
        'method_A_pct': row['method_A'], 'method_B_pct': row['method_B'],
        'method_C_pct': row['method_C'],
        'delta_AB_pp': row['delta_AB'], 'delta_CB_pp': row['delta_CB'],
    })

df_out = pd.DataFrame(rows_out)
for col in ['method_A_pct', 'method_B_pct', 'method_C_pct', 'delta_AB_pp', 'delta_CB_pp']:
    df_out[col] = df_out[col].round(4)

csv_path = OUTPUT_DIR / f'sal_sensitivity_v2_table_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_out.to_csv(csv_path, index=False)
print(f'  Saved : {csv_path}')
print(f'  -> {len(df_out):,} rows')

# ==============================================================================
# 17. FINAL SUMMARY
# ==============================================================================
print('\n' + '=' * 74)
print('  AGGREGATED SENSITIVITY STATISTICS')
print('-' * 74)

for label, df_ctx in [('Hourly cells', df_hourly), ('Monthly cells', df_monthly_cond)]:
    valid = df_ctx.dropna(subset=['delta_AB', 'delta_CB'])
    print(f'  {label}:')
    print(f'    A-B : mean = {valid["delta_AB"].mean():+.2f} pp | '
          f'max|delta| = {valid["delta_AB"].abs().max():.2f} pp')
    print(f'    C-B : mean = {valid["delta_CB"].mean():+.2f} pp | '
          f'max|delta| = {valid["delta_CB"].abs().max():.2f} pp')

print('\n  INTERPRETATION:')
print('  Method A pools minutes and weights by raw per-minute GHI: a handful')
print('  of very sunny periods can dominate the bin result.')
print('  Method C pools within each sensor-day first, then weights every')
print('  sensor-day equally: it smooths minute-level energy variance without')
print('  letting the sunniest days dominate across the season.')
print('  Compare SA-11 and SA-12 to see which method departs from B more')
print('  systematically, and in which direction, for your dataset.')
print('=' * 74)

print('\nDone.')
