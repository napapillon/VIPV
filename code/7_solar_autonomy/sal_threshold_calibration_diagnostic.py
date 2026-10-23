"""
================================================================================
SAL Threshold Calibration Diagnostic
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

The SAL analysis (solar_autonomy_losses.py, solar_autonomy_losses_method_b.py
and solar_autonomy_losses_method_c.py) relies on five thresholds that decide
whether a bin, day or month is reliable enough to be reported:

  MIN_MINUTES          minimum minutes in a bin for the absolute-loss box
                       plot and the Method A / B relative-loss line
  MIN_SENSOR_DAYS      minimum sensor-days for a month to be valid at all
                       (affects every figure)
  THRESHOLD_M          Haversine distance (metres) separating "Parked" from
                       "Moving"
  MIN_MINUTES_PER_DAY  minimum minutes a sensor-day must contribute to a bin,
                       for Method C
  MIN_DAILY_RATIOS     minimum number of valid daily ratios for a Method C
                       bin to be reported

Their values (600, 5, 30 m, 15, 6) were round numbers or came from a single
targeted check (the Rural x July leave-one-day-out test behind
MIN_DAILY_RATIOS = 6). This script calibrates each threshold systematically
from the data.

Calibration methodology (one approach per threshold)
----------------------------------------------------
  TC-1  MIN_MINUTES
        For a representative bin, the standard error of the mean per-minute
        relative-loss ratio is estimated by bootstrap (N_BOOTSTRAP
        resamples) over a grid of minimum bin sizes. It should shrink
        roughly as 1/sqrt(n); the threshold should sit at or beyond the
        "elbow" where larger samples add little precision.

  TC-2  MIN_SENSOR_DAYS
        Same logic for the monthly mean of daily solar autonomy
        (km/kWp/day): bootstrap of sensor-days over a grid of minimum
        day counts.

  TC-3  THRESHOLD_M
        A parameter of the Parked / Moving classification itself, not a
        sample-size cut-off: the classification is rebuilt for a grid of
        distances, and the Method B relative loss of Parked and of Moving is
        tracked against the threshold. A defensible choice sits on a plateau
        where both curves are insensitive to the exact value.

  TC-4  MIN_MINUTES_PER_DAY x MIN_DAILY_RATIOS  (Method C, joint)
        Generalises the Rural x July leave-one-day-out check to every hourly
        and monthly bin: for a grid of the two Method C thresholds, the
        fraction of bins whose jackknife max-influence exceeds
        FRAGILITY_CRITERION_PP (2 percentage points) is mapped as a heatmap,
        showing where Method C bins are reliably non-fragile.

Figures produced
----------------
  TC-1 : Bootstrap standard error vs MIN_MINUTES (elbow plot)
  TC-2 : Bootstrap standard error vs MIN_SENSOR_DAYS (elbow plot)
  TC-3 : Relative-loss estimate vs THRESHOLD_M for Parked and Moving
  TC-4 : Heatmap — fraction of fragile Method C bins vs
         (MIN_MINUTES_PER_DAY, MIN_DAILY_RATIOS)

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
             sal_threshold_calibration_<figure>_<CAMPAIGN>_<variant_tag>.png
             sal_threshold_calibration_summary_<CAMPAIGN>_<variant_tag>.csv
                 (numerical results behind each figure)
No filtering on sun_elev or rad_calib is applied here (clean_data.py does it
upstream): choose the input suffix accordingly (e.g. 'sd_cleaned_hq').

Requirements
------------
    pip install pandas numpy matplotlib

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

OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "sal_TC_outputs"

SHOW_FIGURES: bool = False

DELTA_T_H: float = 1.0 / 60.0

# No filtering on land_use_clcv, sun_elev or rad_calib here: clean_data.py
# applies it upstream (choose the input suffix accordingly).
LOCAL_TZ: str = 'Europe/Lisbon'


R_EARTH_KM: float = 6371.0

# Average Battery Electric Vehicle (BEV) energy consumption, for the
# autonomy-based MIN_SENSOR_DAYS calibration (TC-2).
CVE_KWH_PER_100KM: float = 16.6
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0

# Number of bootstrap resamples used for the standard-error curves (TC-1, TC-2).
N_BOOTSTRAP: int = 300

# Fragility criterion for the Method C joint calibration (TC-4): a bin is
# flagged "fragile" if excluding its single most extreme daily ratio shifts
# the Method C mean by more than this many percentage points. This matches
# the criterion already used to justify raising MIN_DAILY_RATIOS to 6.
FRAGILITY_CRITERION_PP: float = 2.0

# Grids searched for each threshold.
MIN_MINUTES_GRID: list[int]         = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000]
MIN_SENSOR_DAYS_GRID: list[int]     = [2, 3, 4, 5, 6, 8, 10, 15, 20]
THRESHOLD_M_GRID: list[float]       = [5, 10, 15, 20, 25, 30, 40, 50, 70, 100]
MIN_MINUTES_PER_DAY_GRID: list[int] = [5, 10, 15, 20, 30]
MIN_DAILY_RATIOS_GRID: list[int]    = [3, 4, 5, 6, 7, 8, 10, 12]

HOUR_START_PLOT: int = 6
HOUR_END_PLOT:   int = 20
HOURS_PLOT: list[int] = list(range(HOUR_START_PLOT, HOUR_END_PLOT + 1))

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
    path = OUTPUT_DIR / f"sal_threshold_calibration_{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved : {path}')
    show_or_close(fig)


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
print('  SAL Threshold Calibration Diagnostic')
print(f'  CAMPAIGN     : {CAMPAIGN}')
print(f'  Data variant : {_VARIANT_TAG}')
print('=' * 70)

all_frames: list[pd.DataFrame] = []
gps_frames: list[pd.DataFrame] = []

for sensor, csv_path in SENSOR_FILES.items():
    cols = SENSOR_COLS[sensor]
    if not csv_path.exists():
        print(f'  [WARNING] Not found, skipping : {csv_path}')
        continue

    df = pd.read_csv(
        csv_path, low_memory=False,
        usecols=[cols['date'], cols['rad'], cols['ghi'],
                 cols['elev'], cols['land'], cols['lat'], cols['lon']],
    )
    df = df.rename(columns={
        cols['date']: 'datetime', cols['rad']: 'rad_calib', cols['ghi']: 'ghi',
        cols['elev']: 'sun_elev', cols['land']: 'land_use_clcv',
        cols['lat']: 'lat', cols['lon']: 'lon',
    })

    df['datetime']  = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df['datetime']  = df['datetime'].dt.tz_convert(LOCAL_TZ)
    for col in ['rad_calib', 'ghi', 'sun_elev', 'land_use_clcv', 'lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['datetime', 'rad_calib', 'ghi', 'lat', 'lon'])

    df['sensor'] = sensor
    df['date']   = df['datetime'].dt.date
    df['month']  = df['datetime'].dt.month
    df['hour']   = df['datetime'].dt.hour

    all_frames.append(df)

    gps = df[['datetime', 'sensor', 'lat', 'lon']].sort_values('datetime').reset_index(drop=True)
    gps_frames.append(gps)

if not all_frames:
    raise RuntimeError('No valid data found. Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX.')

data = pd.concat(all_frames, ignore_index=True)
print(f'Total valid measurements : {len(data):,}')

data['E_solar_kWh'] = data['rad_calib'] / 1000.0 * DELTA_T_H
data['E_solar_Wh']  = data['E_solar_kWh'] * 1000.0
data['loss_irr_Wh'] = np.maximum(0.0, data['ghi'] - data['rad_calib']) * DELTA_T_H
data['loss_auto_km']  = data['loss_irr_Wh'] / 1000.0 / CVE_KWH_PER_KM
data['auto_solar_km'] = data['E_solar_kWh'] / CVE_KWH_PER_KM

denom_min = data['loss_irr_Wh'] + data['E_solar_Wh']
data['loss_rel_min_pct'] = np.where(denom_min > 0, data['loss_irr_Wh'] / denom_min * 100.0, np.nan)

data['clc_group'] = data['land_use_clcv'].map(CLC_URBAN_RURAL).fillna('Unknown')

summary_rows: list[dict] = []

# ==============================================================================
# TC-1 : MIN_MINUTES CALIBRATION -- bootstrap standard error vs bin size
# ==============================================================================
print('\n[TC-1] Calibrating MIN_MINUTES via bootstrap standard error...')

rng = np.random.default_rng(2026)

CONDITIONS = {
    'Urban': data['clc_group'] == 'Urban',
    'Rural': data['clc_group'] == 'Rural',
}

se_curves: dict[str, list[float]] = {c: [] for c in CONDITIONS}
n_available: dict[str, list[int]] = {c: [] for c in CONDITIONS}

for cond_name, mask in CONDITIONS.items():
    # Pool a representative single hour (13:00, typically well-sampled and
    # away from sunrise/sunset edge effects) across all months.
    sub = data[mask & (data['hour'] == 13)]['loss_rel_min_pct'].dropna().values
    for n_min in MIN_MINUTES_GRID:
        if len(sub) < n_min:
            se_curves[cond_name].append(np.nan)
            n_available[cond_name].append(len(sub))
            continue
        boot_means = np.array([
            rng.choice(sub, size=n_min, replace=True).mean()
            for _ in range(N_BOOTSTRAP)
        ])
        se_curves[cond_name].append(boot_means.std())
        n_available[cond_name].append(len(sub))

fig, ax = plt.subplots(figsize=(10, 6.5))
for cond_name, color in [('Urban', COLOR_URBAN), ('Rural', COLOR_RURAL)]:
    ax.plot(MIN_MINUTES_GRID, se_curves[cond_name], color=color, marker='o',
            linewidth=2, markersize=6, label=f'{cond_name} (n available = {n_available[cond_name][0]:,})')

ax.axvline(600, color='#888888', linestyle='--', linewidth=1.3, label='Current MIN_MINUTES = 600')
ax.set_xlabel('Candidate MIN_MINUTES threshold', fontsize=12, labelpad=8)
ax.set_ylabel('Bootstrap standard error of the mean [pp]', fontsize=12, labelpad=8)
ax.set_title('MIN_MINUTES Calibration\n'
             'Bootstrap SE of mean relative loss (13:00 bin) vs minimum sample size',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9.5, framealpha=0.88)
ax.grid(linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.01,
         'Choose a threshold at or beyond the point where the curve flattens: '
         'further increases in n buy little additional precision.',
         ha='center', fontsize=8.5, color='#666666', style='italic')
plt.tight_layout()
save_figure(fig, 'tc1_min_minutes')

for cond_name in CONDITIONS:
    for n_min, se in zip(MIN_MINUTES_GRID, se_curves[cond_name]):
        summary_rows.append({'calibration': 'MIN_MINUTES', 'condition': cond_name,
                             'threshold': n_min, 'bootstrap_se_pp': se})

# ==============================================================================
# TC-2 : MIN_SENSOR_DAYS CALIBRATION -- bootstrap standard error of monthly mean
# ==============================================================================
print('[TC-2] Calibrating MIN_SENSOR_DAYS via bootstrap standard error...')

daily = (
    data.groupby(['sensor', 'date', 'month'], as_index=False)
    .agg(auto_solar_km=('auto_solar_km', 'sum'))
)

month_counts = daily.groupby('month').size()
test_month = month_counts.idxmax()
vals = daily.loc[daily['month'] == test_month, 'auto_solar_km'].values
print(f'  Representative month for TC-2 : {MONTH_NAMES[test_month - 1]} (n={len(vals)} sensor-days)')

se_curve_days: list[float] = []
for n_days in MIN_SENSOR_DAYS_GRID:
    if len(vals) < n_days:
        se_curve_days.append(np.nan)
        continue
    boot_means = np.array([
        rng.choice(vals, size=n_days, replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    se_curve_days.append(boot_means.std())

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(MIN_SENSOR_DAYS_GRID, se_curve_days, color='#1565C0', marker='o',
        linewidth=2, markersize=6, label=f'{MONTH_NAMES[test_month - 1]} (n available = {len(vals)})')
ax.axvline(5, color='#888888', linestyle='--', linewidth=1.3, label='Current MIN_SENSOR_DAYS = 5')
ax.set_xlabel('Candidate MIN_SENSOR_DAYS threshold', fontsize=12, labelpad=8)
ax.set_ylabel('Bootstrap standard error of monthly mean [km/kWp/day]', fontsize=12, labelpad=8)
ax.set_title('MIN_SENSOR_DAYS Calibration\n'
             'Bootstrap SE of monthly mean solar autonomy vs minimum day count',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9.5, framealpha=0.88)
ax.grid(linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_figure(fig, 'tc2_min_sensor_days')

for n_days, se in zip(MIN_SENSOR_DAYS_GRID, se_curve_days):
    summary_rows.append({'calibration': 'MIN_SENSOR_DAYS', 'condition': MONTH_NAMES[test_month - 1],
                         'threshold': n_days, 'bootstrap_se_pp': se})

# ==============================================================================
# TC-3 : THRESHOLD_M CALIBRATION -- Parked/Moving relative loss vs distance
# ==============================================================================
print('[TC-3] Calibrating THRESHOLD_M across a grid of movement distances...')

gps_all = pd.concat(gps_frames, ignore_index=True)

# Precompute inter-point distances ONCE per sensor (independent of threshold);
# only the Parked/Moving assignment changes per threshold.
dist_frames: list[pd.DataFrame] = []
for sensor in gps_all['sensor'].unique():
    g = gps_all[gps_all['sensor'] == sensor].sort_values('datetime').reset_index(drop=True)
    if len(g) < 2:
        continue
    d = haversine_vectorized(g['lat'].values[:-1], g['lon'].values[:-1],
                             g['lat'].values[1:], g['lon'].values[1:])
    g2 = g.iloc[:-1].copy()
    g2['distance_m'] = d
    dist_frames.append(g2[['datetime', 'sensor', 'distance_m']])

dist_df = pd.concat(dist_frames, ignore_index=True)
data_dist = data.merge(dist_df, on=['datetime', 'sensor'], how='inner')

parked_rel_by_threshold: list[float] = []
moving_rel_by_threshold: list[float] = []
parked_n: list[int] = []
moving_n: list[int] = []

for thr in THRESHOLD_M_GRID:
    moving_mask = data_dist['distance_m'] >= thr
    parked_vals = data_dist.loc[~moving_mask, 'loss_rel_min_pct'].dropna().values
    moving_vals = data_dist.loc[moving_mask, 'loss_rel_min_pct'].dropna().values
    parked_rel_by_threshold.append(parked_vals.mean() if len(parked_vals) > 0 else np.nan)
    moving_rel_by_threshold.append(moving_vals.mean() if len(moving_vals) > 0 else np.nan)
    parked_n.append(len(parked_vals))
    moving_n.append(len(moving_vals))

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(THRESHOLD_M_GRID, parked_rel_by_threshold, color=COLOR_PARKED, marker='o',
        linewidth=2, markersize=6, label='Parked -- mean relative loss [%]')
ax.plot(THRESHOLD_M_GRID, moving_rel_by_threshold, color=COLOR_MOVING, marker='s',
        linewidth=2, markersize=6, label='Moving -- mean relative loss [%]')
ax.axvline(30, color='#888888', linestyle='--', linewidth=1.3, label='Current THRESHOLD_M = 30 m')
ax.set_xlabel('Candidate THRESHOLD_M [metres]', fontsize=12, labelpad=8)
ax.set_ylabel('Mean Relative Loss [%] (Method B, for reference)', fontsize=12, labelpad=8)
ax.set_title('THRESHOLD_M Calibration\n'
             'Parked/Moving relative-loss estimate vs the Haversine distance threshold',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9.5, framealpha=0.88)
ax.grid(linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.01,
         'A defensible threshold sits in a flat (plateau) region of both curves; '
         'a region where curves move sharply indicates high sensitivity to the exact choice.',
         ha='center', fontsize=8.5, color='#666666', style='italic')
plt.tight_layout()
save_figure(fig, 'tc3_threshold_m')

for thr, pv, mv, pn, mn in zip(THRESHOLD_M_GRID, parked_rel_by_threshold,
                                moving_rel_by_threshold, parked_n, moving_n):
    summary_rows.append({'calibration': 'THRESHOLD_M', 'condition': 'Parked',
                         'threshold': thr, 'mean_rel_pct': pv, 'n_minutes': pn})
    summary_rows.append({'calibration': 'THRESHOLD_M', 'condition': 'Moving',
                         'threshold': thr, 'mean_rel_pct': mv, 'n_minutes': mn})

# ==============================================================================
# TC-4 : METHOD C JOINT CALIBRATION -- fragile-bin fraction vs (day-minutes, day-count)
# ==============================================================================
print('[TC-4] Calibrating Method C thresholds via generalised leave-one-day-out...')


def daily_ratios_for_bin(df_subset: pd.DataFrame, min_minutes_per_day: int) -> np.ndarray:
    """Per-sensor-day relative-loss ratios for one bin, filtered by day size."""
    daily_grp = (
        df_subset.groupby(['sensor', 'date'])
        .agg(loss=('loss_irr_Wh', 'sum'), solar=('E_solar_Wh', 'sum'),
             n_minutes=('loss_irr_Wh', 'size'))
        .reset_index()
    )
    daily_grp = daily_grp[daily_grp['n_minutes'] >= min_minutes_per_day]
    denom = daily_grp['loss'] + daily_grp['solar']
    ratio = np.where(denom > 0, daily_grp['loss'] / denom * 100.0, np.nan)
    return ratio[~np.isnan(ratio)]


def jackknife_max_influence(ratios: np.ndarray) -> float:
    """Largest change in the mean caused by removing any single value."""
    if len(ratios) < 2:
        return np.nan
    mean_all = ratios.mean()
    loo_means = np.array([np.delete(ratios, i).mean() for i in range(len(ratios))])
    return np.abs(loo_means - mean_all).max()


# Test bins: every (condition, month) combination for Urban/Rural, a broad
# and representative sweep across the seasonal cycle.
test_bins: list[pd.DataFrame] = []
for cond_name, mask in CONDITIONS.items():
    for m in range(1, 13):
        sub = data[mask & (data['month'] == m)]
        if len(sub) > 0:
            test_bins.append(sub)

print(f'  Testing across {len(test_bins)} (condition, month) bins...')

fragile_fraction_grid = np.zeros((len(MIN_DAILY_RATIOS_GRID), len(MIN_MINUTES_PER_DAY_GRID)))

for i, min_days in enumerate(MIN_DAILY_RATIOS_GRID):
    for j, min_min_per_day in enumerate(MIN_MINUTES_PER_DAY_GRID):
        n_tested, n_fragile = 0, 0
        for sub in test_bins:
            ratios = daily_ratios_for_bin(sub, min_min_per_day)
            if len(ratios) < min_days:
                continue
            n_tested += 1
            influence = jackknife_max_influence(ratios)
            if influence > FRAGILITY_CRITERION_PP:
                n_fragile += 1
        fragile_fraction_grid[i, j] = (n_fragile / n_tested * 100.0) if n_tested > 0 else np.nan

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(fragile_fraction_grid, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
ax.set_xticks(range(len(MIN_MINUTES_PER_DAY_GRID)))
ax.set_xticklabels(MIN_MINUTES_PER_DAY_GRID, fontsize=10)
ax.set_yticks(range(len(MIN_DAILY_RATIOS_GRID)))
ax.set_yticklabels(MIN_DAILY_RATIOS_GRID, fontsize=10)
ax.set_xlabel('MIN_MINUTES_PER_DAY', fontsize=12, labelpad=8)
ax.set_ylabel('MIN_DAILY_RATIOS', fontsize=12, labelpad=8)
ax.set_title('Method C Joint Threshold Calibration\n'
             f'% of bins with jackknife max-influence > {FRAGILITY_CRITERION_PP:.0f} pp '
             '(lower = more reliable)',
             fontsize=12, fontweight='bold')
for i in range(len(MIN_DAILY_RATIOS_GRID)):
    for j in range(len(MIN_MINUTES_PER_DAY_GRID)):
        v = fragile_fraction_grid[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=9, color='#222222')
cur_i = MIN_DAILY_RATIOS_GRID.index(6) if 6 in MIN_DAILY_RATIOS_GRID else None
cur_j = MIN_MINUTES_PER_DAY_GRID.index(15) if 15 in MIN_MINUTES_PER_DAY_GRID else None
if cur_i is not None and cur_j is not None:
    ax.add_patch(plt.Rectangle((cur_j - 0.5, cur_i - 0.5), 1, 1, fill=False,
                               edgecolor='black', linewidth=2.5))
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('% fragile bins', fontsize=10)
ax.invert_yaxis()
fig.text(0.5, -0.01,
         'Black box: current choice (MIN_DAILY_RATIOS=6, MIN_MINUTES_PER_DAY=15). '
         'Choose the smallest thresholds (fewest excluded bins) inside a low-fragility region.',
         ha='center', fontsize=8.5, color='#666666', style='italic')
plt.tight_layout()
save_figure(fig, 'tc4_methodC_joint_calibration')

for i, min_days in enumerate(MIN_DAILY_RATIOS_GRID):
    for j, min_min_per_day in enumerate(MIN_MINUTES_PER_DAY_GRID):
        summary_rows.append({
            'calibration': 'MethodC_joint', 'condition': f'days>={min_days}',
            'threshold': min_min_per_day, 'fragile_pct': fragile_fraction_grid[i, j],
        })

# ==============================================================================
# EXPORT SUMMARY
# ==============================================================================
print('\nExporting calibration summary...')

df_summary = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / f'sal_threshold_calibration_summary_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_summary.to_csv(summary_path, index=False)
print(f'  Saved : {summary_path}')

print('\n' + '=' * 74)
print('  READING GUIDE')
print('-' * 74)
print('  TC-1/TC-2 : pick the threshold at the elbow where the SE curve flattens --')
print('              raising it further mostly wastes data without buying precision.')
print('  TC-3      : pick a threshold inside a flat region of both curves -- avoid')
print('              a region where a small change in metres swings the estimate.')
print('  TC-4      : pick the smallest (MIN_DAILY_RATIOS, MIN_MINUTES_PER_DAY) pair')
print('              that keeps the fragile-bin percentage low (e.g. under 10%).')
print('=' * 74)

print('\nDone.')
