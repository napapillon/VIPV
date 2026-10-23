"""
================================================================================
SAL Weather Imbalance Diagnostic — kc_clear Distribution Across Conditions
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Method A (pooled ratio of sums) implicitly weights every minute by its GHI.
As GHI is strongly correlated with the clear-sky index
kc_clear = ghi / ghi_clear, an imbalance in the weather mix (clear-sky vs
overcast minutes / days) between two compared conditions (Urban vs Rural,
Parked vs Moving) could make a Method A comparison partly reflect which
condition was sampled under sunnier weather, rather than a difference in
shading. This script tests that possibility at three levels:

  1. MINUTE LEVEL: is the kc_clear distribution of all valid minutes
     different between Urban and Rural (and between Parked and Moving)?
     Two-sample Kolmogorov-Smirnov (KS) test, sensitive to any difference
     in distribution shape, not only the mean.

  2. DAY LEVEL: for each sensor-day, the mean kc_clear of its minutes within
     a condition (days with at least MIN_MINUTES_PER_DAY minutes), compared
     between conditions. This level matters most for Method C, whose unit is
     the sensor-day.

  3. SEASONAL BREAKDOWN: condition x month heatmap of mean kc_clear, to see
     whether an imbalance is concentrated in some months (sampling-calendar
     artefact) or spread over the year (structural difference, e.g. Rural
     routes driven at other times of day or under other regional weather).

kc_clear = ghi / ghi_clear is the clear-sky index (k_c in this project), not
the clearness index k_t = ghi / ghi_extra.

Figures produced
----------------
  WI-1 : Overlaid histograms of minute-level kc_clear — Urban vs Rural
  WI-2 : Overlaid histograms of minute-level kc_clear — Parked vs Moving
  WI-3 : Box plots of day-level mean kc_clear — all four conditions
  WI-4 : Heatmap — mean kc_clear by condition x month

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', 'ghi', 'sun_elev',
             'land_use_clcv', 'ghi_clear'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             sal_weather_imbalance_<figure>_<CAMPAIGN>_<variant_tag>.png
             sal_weather_imbalance_ks_tests_<CAMPAIGN>_<variant_tag>.csv
                 (KS test statistics and p-values)
             sal_weather_imbalance_summary_<CAMPAIGN>_<variant_tag>.csv
                 (mean / median / std kc_clear by condition, minute and
                 day level)
No filtering on sun_elev or rad_calib is applied here (clean_data.py does it
upstream): choose the input suffix accordingly (e.g. 'sd_cleaned_hq').

Requirements
------------
    pip install pandas numpy matplotlib scipy

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.1
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
from scipy.stats import ks_2samp

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

OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "salwid_outputs"

SHOW_FIGURES: bool = False

LOCAL_TZ: str = 'Europe/Lisbon'

MAX_GAP_MINUTES: int = 2
THRESHOLD_M: float = 30.0
R_EARTH_KM:  float = 6371.0

# Minimum minutes within a sensor-day, for a given condition, for that day's
# mean kc_clear to be counted as a valid day-level observation.
MIN_MINUTES_PER_DAY: int = 15

MONTH_NAMES: list[str] = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

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
    Build the dict mapping sensor names to their CSV input paths.

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


_VARIANT_TAG = _variant_tag()
OUTPUT_DIR = OUTPUT_BASE_DIR / CAMPAIGN / _VARIANT_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_FILES = build_sensor_files(CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG)

# Column mapping reflecting the actual CSV schema for each sensor file:
#   Unnamed: 0, Human readable date (Europe/London), J3x lat, J3x lon,
#   J3x Rad raw, J3x Rad calib, ghi, ghi_clear, ghi_extra, bhi, dhi, dni,
#   sun_elev, sun_azim, land_use_clcv, land_use_str_clcv
SENSOR_COLS: dict[str, dict[str, str]] = {
    sensor: {
        'date':      'Human readable date (Europe/London)',
        'ghi':       'ghi',
        'ghi_clear': 'ghi_clear',
        'elev':      'sun_elev',
        'land':      'land_use_clcv',
        'lat':       f'{sensor} lat',
        'lon':       f'{sensor} lon',
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
    path = OUTPUT_DIR / f"sal_weather_imbalance_{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved : {path}')
    show_or_close(fig)


# ==============================================================================
# HAVERSINE -- movement classification
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2, R=R_EARTH_KM,
                         timestamps=None, max_gap_minutes=MAX_GAP_MINUTES):
    """Compute the great-circle distance in metres between consecutive GPS points."""
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
        s           = pd.Series(np.asarray(timestamps))            
        gap_s       = s.diff().dt.total_seconds().fillna(0).values
        dates       = s.dt.date.values
        is_boundary = (
            (gap_s[1:] > max_gap_minutes * 60)
            | (dates[1:] != dates[:-1])
        )
        distances[is_boundary] = 0.0
    
    return distances


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print('=' * 70)
print('  SAL Weather Imbalance Diagnostic')
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
        usecols=[cols['date'], cols['ghi'], cols['ghi_clear'],
                 cols['elev'], cols['land'], cols['lat'], cols['lon']],
    )
    df = df.rename(columns={
        cols['date']:      'datetime',
        cols['ghi']:       'ghi',
        cols['ghi_clear']: 'ghi_clear',
        cols['elev']:      'sun_elev',
        cols['land']:      'land_use_clcv',
        cols['lat']:       'lat',
        cols['lon']:       'lon',
    })

    df['datetime']      = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df['datetime']      = df['datetime'].dt.tz_convert(LOCAL_TZ)
    df['ghi']            = pd.to_numeric(df['ghi'],       errors='coerce')
    df['ghi_clear']      = pd.to_numeric(df['ghi_clear'], errors='coerce')
    df['sun_elev']        = pd.to_numeric(df['sun_elev'],  errors='coerce')
    df['land_use_clcv']  = pd.to_numeric(df['land_use_clcv'], errors='coerce').astype('Int64')

    df = df.dropna(subset=['datetime', 'ghi', 'ghi_clear', 'sun_elev'])

    df['sensor'] = sensor
    df['date']   = df['datetime'].dt.date
    df['month']  = df['datetime'].dt.month

    all_frames.append(df)

if not all_frames:
    raise RuntimeError(
        'No valid data found. Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, '
        'and FILE_SUFFIX/TUKTUK_FILE_SUFFIX.'
    )

data = pd.concat(all_frames, ignore_index=True)
print(f'Total valid measurements : {len(data):,}')

# ==============================================================================
# 2. CLEARNESS INDEX
# ==============================================================================
data['kc_clear'] = np.where(data['ghi_clear'] > 0, data['ghi'] / data['ghi_clear'], np.nan)
data['kc_clear'] = data['kc_clear'].clip(lower=0, upper=1.3)
data = data.dropna(subset=['kc_clear'])

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
        timestamps=df_s['datetime']
    )
    df_s = df_s.iloc[:-1].copy()
    df_s['moving_status'] = np.where(dist >= THRESHOLD_M, 'Moving', 'Parked')
    df_s['sensor']        = sensor
    move_frames.append(df_s[['datetime', 'sensor', 'moving_status']])

move_df = pd.concat(move_frames, ignore_index=True)
data    = data.merge(move_df, on=['datetime', 'sensor'], how='left')

CONDITIONS = {
    'Urban':  data['clc_group'] == 'Urban',
    'Rural':  data['clc_group'] == 'Rural',
    'Parked': data['moving_status'] == 'Parked',
    'Moving': data['moving_status'] == 'Moving',
}

CONDITION_COLORS = {
    'Urban': COLOR_URBAN, 'Rural': COLOR_RURAL,
    'Parked': COLOR_PARKED, 'Moving': COLOR_MOVING,
}

# ==============================================================================
# 4. MINUTE-LEVEL COMPARISON -- WI-1 and WI-2
# ==============================================================================

def plot_kc_histograms(cond_a: str, cond_b: str, fig_name: str) -> tuple[float, float]:
    """
    Overlay normalised histograms of minute-level kc_clear for two conditions,
    and return the two-sample Kolmogorov-Smirnov (statistic, p-value).
    """
    vals_a = data.loc[CONDITIONS[cond_a], 'kc_clear'].values
    vals_b = data.loc[CONDITIONS[cond_b], 'kc_clear'].values

    ks_stat, ks_pvalue = ks_2samp(vals_a, vals_b)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1.3, 40)
    ax.hist(vals_a, bins=bins, density=True, alpha=0.55, color=CONDITION_COLORS[cond_a],
            label=f'{cond_a} (n={len(vals_a):,})', edgecolor='white', linewidth=0.3)
    ax.hist(vals_b, bins=bins, density=True, alpha=0.55, color=CONDITION_COLORS[cond_b],
            label=f'{cond_b} (n={len(vals_b):,})', edgecolor='white', linewidth=0.3)
    ax.axvline(vals_a.mean(), color=CONDITION_COLORS[cond_a], linestyle='--', linewidth=2,
               label=f'{cond_a} mean = {vals_a.mean():.3f}')
    ax.axvline(vals_b.mean(), color=CONDITION_COLORS[cond_b], linestyle='--', linewidth=2,
               label=f'{cond_b} mean = {vals_b.mean():.3f}')

    ax.set_xlabel('Clear-Sky Index k_c = GHI / GHI_clear-sky', fontsize=12, labelpad=8)
    ax.set_ylabel('Probability Density', fontsize=12, labelpad=8)
    ax.set_title(f'Minute-Level kc_clear Distribution -- {cond_a} vs {cond_b}\n'
                 f'Kolmogorov-Smirnov test: D = {ks_stat:.4f}, p-value = {ks_pvalue:.2e}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9.5, framealpha=0.88)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_figure(fig, fig_name)

    return ks_stat, ks_pvalue


print('\nGenerating WI-1 : Urban vs Rural minute-level kc_clear...')
ks_urban_rural = plot_kc_histograms('Urban', 'Rural', 'wi1_urban_vs_rural')

print('Generating WI-2 : Parked vs Moving minute-level kc_clear...')
ks_parked_moving = plot_kc_histograms('Parked', 'Moving', 'wi2_parked_vs_moving')

# ==============================================================================
# 5. DAY-LEVEL COMPARISON -- WI-3
# ==============================================================================
print('\nComputing day-level mean kc_clear per condition...')

day_level_rows = []
for cond_name, mask in CONDITIONS.items():
    sub = data[mask]
    daily = (
        sub.groupby(['sensor', 'date'])
        .agg(mean_kc=('kc_clear', 'mean'), n_minutes=('kc_clear', 'size'))
        .reset_index()
    )
    daily = daily[daily['n_minutes'] >= MIN_MINUTES_PER_DAY]
    for _, row in daily.iterrows():
        day_level_rows.append({
            'condition': cond_name, 'sensor': row['sensor'], 'date': row['date'],
            'mean_kc_clear': row['mean_kc'], 'n_minutes': row['n_minutes'],
        })

df_day_level = pd.DataFrame(day_level_rows)

print('Generating WI-3 : Day-level mean kc_clear box plots...')

fig, ax = plt.subplots(figsize=(9, 6.5))
box_data = [df_day_level.loc[df_day_level['condition'] == c, 'mean_kc_clear'].values
            for c in ['Urban', 'Rural', 'Parked', 'Moving']]
bp = ax.boxplot(box_data, patch_artist=True, widths=0.55, labels=['Urban', 'Rural', 'Parked', 'Moving'],
                medianprops=dict(color='#222222', linewidth=1.8),
                whiskerprops=dict(linewidth=1.2, linestyle='--'),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=4, alpha=0.5))
for patch, cond in zip(bp['boxes'], ['Urban', 'Rural', 'Parked', 'Moving']):
    patch.set_facecolor(CONDITION_COLORS[cond])
    patch.set_alpha(0.6)

for i, cond in enumerate(['Urban', 'Rural', 'Parked', 'Moving']):
    n_days = len(df_day_level[df_day_level['condition'] == cond])
    ax.text(i + 1, -0.05, f'n={n_days}', ha='center', fontsize=9, color='#555555',
            transform=ax.get_xaxis_transform())

ax.set_ylabel('Day-Level Mean kc_clear', fontsize=12, labelpad=8)
ax.set_title('Distribution of Day-Level Mean Clear-Sky Index by Condition\n'
             '(the statistical unit relevant to Method C)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_figure(fig, 'wi3_day_level_boxplots')

# KS tests at day level
ks_day_urban_rural = ks_2samp(
    df_day_level.loc[df_day_level['condition'] == 'Urban', 'mean_kc_clear'],
    df_day_level.loc[df_day_level['condition'] == 'Rural', 'mean_kc_clear'],
)
ks_day_parked_moving = ks_2samp(
    df_day_level.loc[df_day_level['condition'] == 'Parked', 'mean_kc_clear'],
    df_day_level.loc[df_day_level['condition'] == 'Moving', 'mean_kc_clear'],
)

# ==============================================================================
# 6. SEASONAL BREAKDOWN -- WI-4
# ==============================================================================
print('Generating WI-4 : Mean kc_clear by condition x month heatmap...')

seasonal_rows = []
for cond_name, mask in CONDITIONS.items():
    for m in range(1, 13):
        sub = data[mask & (data['month'] == m)]
        seasonal_rows.append({
            'condition': cond_name, 'month': m,
            'mean_kc_clear': sub['kc_clear'].mean() if len(sub) >= 100 else np.nan,
            'n_minutes': len(sub),
        })
df_seasonal = pd.DataFrame(seasonal_rows)
pivot_seasonal = df_seasonal.pivot(index='condition', columns='month', values='mean_kc_clear')
pivot_seasonal = pivot_seasonal.reindex(index=['Urban', 'Rural', 'Parked', 'Moving'], columns=range(1, 13))

fig, ax = plt.subplots(figsize=(14, 4.5))
im = ax.imshow(pivot_seasonal.values, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=0.9)
ax.set_xticks(range(12))
ax.set_xticklabels(MONTH_NAMES, fontsize=10)
ax.set_yticks(range(4))
ax.set_yticklabels(pivot_seasonal.index, fontsize=11)
ax.set_title('Mean kc_clear by Condition and Month\n'
             '(a horizontal-only pattern would indicate a genuine, stable condition-level difference;\n'
             'patchy columns would indicate the imbalance is driven by specific months)',
             fontsize=11.5, fontweight='bold')
for i in range(4):
    for j in range(12):
        v = pivot_seasonal.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9, color='#222222')
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Mean kc_clear', fontsize=10)
plt.tight_layout()
save_figure(fig, 'wi4_seasonal_heatmap')

# ==============================================================================
# 7. EXPORT TABLES
# ==============================================================================
print('\nExporting tables...')

ks_rows = [
    {'comparison': 'Urban vs Rural (minute-level)', 'ks_statistic': ks_urban_rural[0],
     'p_value': ks_urban_rural[1], 'mean_a': data.loc[CONDITIONS['Urban'], 'kc_clear'].mean(),
     'mean_b': data.loc[CONDITIONS['Rural'], 'kc_clear'].mean()},
    {'comparison': 'Parked vs Moving (minute-level)', 'ks_statistic': ks_parked_moving[0],
     'p_value': ks_parked_moving[1], 'mean_a': data.loc[CONDITIONS['Parked'], 'kc_clear'].mean(),
     'mean_b': data.loc[CONDITIONS['Moving'], 'kc_clear'].mean()},
    {'comparison': 'Urban vs Rural (day-level)', 'ks_statistic': ks_day_urban_rural[0],
     'p_value': ks_day_urban_rural[1],
     'mean_a': df_day_level.loc[df_day_level['condition'] == 'Urban', 'mean_kc_clear'].mean(),
     'mean_b': df_day_level.loc[df_day_level['condition'] == 'Rural', 'mean_kc_clear'].mean()},
    {'comparison': 'Parked vs Moving (day-level)', 'ks_statistic': ks_day_parked_moving[0],
     'p_value': ks_day_parked_moving[1],
     'mean_a': df_day_level.loc[df_day_level['condition'] == 'Parked', 'mean_kc_clear'].mean(),
     'mean_b': df_day_level.loc[df_day_level['condition'] == 'Moving', 'mean_kc_clear'].mean()},
]
df_ks = pd.DataFrame(ks_rows)
df_ks['mean_diff'] = df_ks['mean_a'] - df_ks['mean_b']
for col in ['ks_statistic', 'p_value', 'mean_a', 'mean_b', 'mean_diff']:
    df_ks[col] = df_ks[col].round(5)

ks_path = OUTPUT_DIR / f'sal_weather_imbalance_ks_tests_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_ks.to_csv(ks_path, index=False)
print(f'  Saved : {ks_path}')

summary_rows = []
for cond_name, mask in CONDITIONS.items():
    vals_min = data.loc[mask, 'kc_clear']
    vals_day = df_day_level.loc[df_day_level['condition'] == cond_name, 'mean_kc_clear']
    summary_rows.append({
        'condition': cond_name,
        'minute_mean': vals_min.mean(), 'minute_median': vals_min.median(), 'minute_std': vals_min.std(),
        'day_mean': vals_day.mean(), 'day_median': vals_day.median(), 'day_std': vals_day.std(),
        'n_minutes': len(vals_min), 'n_days': len(vals_day),
    })
df_summary = pd.DataFrame(summary_rows)
for col in df_summary.columns:
    if df_summary[col].dtype == float:
        df_summary[col] = df_summary[col].round(4)

summary_path = OUTPUT_DIR / f'sal_weather_imbalance_summary_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_summary.to_csv(summary_path, index=False)
print(f'  Saved : {summary_path}')

# ==============================================================================
# 8. FINAL SUMMARY
# ==============================================================================
print('\n' + '=' * 90)
print('  WEATHER IMBALANCE SUMMARY')
print('-' * 90)
print(df_ks.to_string(index=False))
print('=' * 90)

print('\n  READING GUIDE:')
print('  A significant KS test (p < 0.05) with a non-trivial mean difference means')
print('  the two conditions were sampled under measurably different weather mixes.')
print('  In that case, part of any Method A difference between those conditions may')
print('  reflect the weather imbalance rather than a pure shading effect, and Method C')
print('  (equal weight per sensor-day) is less exposed to this confound.')
print('  A non-significant test, or a significant test with a negligible mean')
print('  difference, means the weather mix is effectively balanced between the two')
print('  conditions, and Method A comparisons between them can be trusted as-is.')

print('\nDone.')
