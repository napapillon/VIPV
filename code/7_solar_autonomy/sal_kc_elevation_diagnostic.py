"""
================================================================================
SAL Diagnostic — Clear-Sky Index (kc) x Solar Elevation Cross-Tabulation
and Three-Way Aggregation Method Comparison (A / B / C)
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Complements sal_sensitivity_analysis.py by:

  1. Explaining WHY Methods A and B diverge, with the clear-sky index
     kc = ghi / ghi_clear as explanatory variable. Hypothesis tested: within
     an elevation bin, minutes with high kc (irradiance close to the
     clear-sky reference) may show a different per-minute relative loss than
     minutes with low kc (strong attenuation), and this internal correlation
     — not simply "low-GHI minutes" — drives the sign and magnitude of the
     Method A vs Method B gap.

  2. Comparing Method C with Method A:
       METHOD A — Pooled Ratio-of-Sums: all minutes of a bin pooled, loss and
           captured energy summed once, then divided; each minute weighs in
           proportion to its GHI, so a very sunny month / day can dominate.
       METHOD B — Mean-of-Minute-Ratios (solar_autonomy_losses_method_b.py): the
           per-minute ratio averaged with equal weight per minute.
       METHOD C — Mean-of-Daily-Ratios: loss and captured energy summed over
           the minutes of each sensor-day in the bin (energy-weighted within
           the day), one ratio per sensor-day, then averaged with equal
           weight per sensor-day.
     Method C is expected to sit between A and B: it removes the "one sunny
     day dominates" effect of A while keeping the sum-before-divide logic
     within each day.

Figures produced
----------------
  KC-1 : Monthly relative loss — Method A vs B vs C (grouped bar chart)
  KC-2 : Heatmap — mean per-minute relative loss (%) by
         (elevation bin x kc bin)
  KC-3 : Heatmap — minute count (sample weight) by (elevation bin x kc bin);
         cells with fewer than MIN_CELL_MINUTES minutes are masked
  KC-4 : Grouped bars — mean relative loss by kc tercile for two
         representative elevation bins (low vs mid), with minute counts:
         the covariance mechanism behind the Method A vs B gap
  KC-5 : Method A vs Method C — scatter across hourly / monthly bins
         (counterpart of SA-11 of sal_sensitivity_analysis.py)

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', '<SENSOR> Rad calib', 'ghi', 'sun_elev',
             'land_use_clcv', 'ghi_clear'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             sal_kc_diagnostic_<figure>_<CAMPAIGN>_<variant_tag>.png
             sal_kc_diagnostic_crosstab_<CAMPAIGN>_<variant_tag>.csv
                 (elevation x kc cross-tabulation)
             sal_kc_diagnostic_monthly_ABC_<CAMPAIGN>_<variant_tag>.csv
                 (monthly A / B / C comparison)
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

OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "sal_kc_outputs"

SHOW_FIGURES: bool = False

DELTA_T_H: float = 1.0 / 60.0

# No filtering on land_use_clcv, sun_elev or rad_calib here: clean_data.py
# applies it upstream (choose the input suffix accordingly).
LOCAL_TZ: str = 'Europe/Lisbon'


MIN_SENSOR_DAYS: int = 5
MIN_MINUTES:     int = 600

# Minimum minutes required inside a single elevation x kc cell for it
# to be considered statistically reliable (shown on the heatmap; otherwise
# masked as NaN).
MIN_CELL_MINUTES: int = 200

MONTH_NAMES: list[str] = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

COLOR_A = '#1565C0'   # Method A -- pooled ratio-of-sums
COLOR_B = '#E05C5C'   # Method B -- mean-of-minute-ratios
COLOR_C = '#2E7D32'   # Method C -- mean-of-daily-ratios

# ==============================================================================
# FILE RESOLUTION
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
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

SENSOR_FILES = _build_sensor_files(CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG)

# Column mapping reflecting the actual CSV schema:
#   Unnamed: 0, Human readable date (Europe/London), J3x lat, J3x lon,
#   J3x Rad raw, J3x Rad calib, ghi, ghi_clear, ghi_extra, bhi, dhi, dni,
#   sun_elev, sun_azim, land_use, land_use_str
SENSOR_COLS: dict[str, dict[str, str]] = {
    sensor: {
        'date':      'Human readable date (Europe/London)',
        'rad':       f'{sensor} Rad calib',
        'ghi':       'ghi',
        'ghi_clear': 'ghi_clear',
        'elev':      'sun_elev',
        'land': 'land_use_clcv',
        'lat':       f'{sensor} lat',
        'lon':       f'{sensor} lon',
    }
    for sensor in ['J30', 'J31', 'J32', 'J33']
}


def _show_or_close(fig: plt.Figure) -> None:
    """Show the figure if SHOW_FIGURES, otherwise close it."""
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure as sal_kc_diagnostic_<name>_<CAMPAIGN>_<variant_tag>.png."""
    path = OUTPUT_DIR / f"sal_kc_diagnostic_{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved : {path}')
    _show_or_close(fig)


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print('=' * 70)
print('  SAL kc x Elevation Diagnostic  +  Method A / B / C Comparison')
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
        usecols=[cols['date'], cols['rad'], cols['ghi'], cols['ghi_clear'],
                 cols['elev'], cols['land']],
    )
    df = df.rename(columns={
        cols['date']:      'datetime',
        cols['rad']:       'rad_calib',
        cols['ghi']:       'ghi',
        cols['ghi_clear']: 'ghi_clear',
        cols['elev']:      'sun_elev',
        cols['land']: 'land_use_clcv',
    })

    df['datetime']  = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df['datetime']  = df['datetime'].dt.tz_convert(LOCAL_TZ)
    df['rad_calib'] = pd.to_numeric(df['rad_calib'], errors='coerce')
    df['ghi']       = pd.to_numeric(df['ghi'],       errors='coerce')
    df['ghi_clear'] = pd.to_numeric(df['ghi_clear'], errors='coerce')
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
# 2. PER-MINUTE ENERGY COLUMNS + CLEAR-SKY INDEX
# ==============================================================================
data['E_solar_Wh']  = data['rad_calib'] * DELTA_T_H
data['loss_irr_Wh'] = np.maximum(0.0, data['ghi'] - data['rad_calib']) * DELTA_T_H

denom_min = data['loss_irr_Wh'] + data['E_solar_Wh']
data['loss_rel_min_pct'] = np.where(
    denom_min > 0, data['loss_irr_Wh'] / denom_min * 100.0, np.nan
)

# Clear-sky index: kc = GHI / GHI_clear-sky
# ghi_clear is the modelled clear-sky GHI for that instant.
# kc close to 1: measured GHI is close to the clear-sky reference.
# kc close to 0: measured GHI is strongly attenuated relative to that reference.
# kc does not directly measure the direct/diffuse irradiance partition; DNI, DHI,
# or diffuse fraction must be analysed separately for that purpose.
data['kc'] = np.where(
    data['ghi_clear'] > 0, data['ghi'] / data['ghi_clear'], np.nan
)
# Reject values outside the accepted range rather than clipping them to a bin edge.
# Such values may arise from measurement/model mismatch, especially near sunrise.
KC_MIN: float = 0.0
KC_MAX: float = 1.3
data.loc[~data['kc'].between(KC_MIN, KC_MAX, inclusive='both'), 'kc'] = np.nan

data['clc_group'] = data['land_use_clcv'].map(CLC_URBAN_RURAL).fillna('Unknown')

# ==============================================================================
# 3. VALID MONTHS
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
# 4. AGGREGATION FUNCTIONS -- METHODS A, B, C
# ==============================================================================

def method_A_pooled_ratio_of_sums(group: pd.DataFrame) -> float:
    """
    Method A -- pool ALL minutes in the group and compute a single ratio.
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


def method_C_mean_of_daily_ratios(group: pd.DataFrame) -> float:
    """
    Method C -- compute one ratio-of-sums PER SENSOR-DAY (summing only the
    minutes of that day within the group), then average those daily ratios
    with equal weight per sensor-day. This matches the sensor-day unit
    already used throughout the pipeline (MIN_SENSOR_DAYS, Figures 1-3).
    """
    daily = (
        group.groupby(['sensor', 'date'])
        .agg(loss=('loss_irr_Wh', 'sum'), solar=('E_solar_Wh', 'sum'))
        .reset_index()
    )
    denom = daily['loss'] + daily['solar']
    daily_ratio = np.where(denom > 0, daily['loss'] / denom * 100.0, np.nan)
    daily_ratio = daily_ratio[~np.isnan(daily_ratio)]
    return daily_ratio.mean() if len(daily_ratio) > 0 else np.nan


# ==============================================================================
# 5. MONTHLY COMPARISON -- A vs B vs C
# ==============================================================================
print('\nComputing monthly A / B / C comparison...')

monthly_abc = []
for m in range(1, 13):
    sub = data[data['month'] == m]
    if m not in VALID_MONTHS or len(sub) < MIN_MINUTES:
        monthly_abc.append({'month': m, 'method_A': np.nan, 'method_B': np.nan,
                            'method_C': np.nan, 'n_minutes': len(sub)})
        continue
    monthly_abc.append({
        'month':     m,
        'method_A':  method_A_pooled_ratio_of_sums(sub),
        'method_B':  method_B_mean_of_minute_ratios(sub),
        'method_C':  method_C_mean_of_daily_ratios(sub),
        'n_minutes': len(sub),
    })
df_monthly_abc = pd.DataFrame(monthly_abc)

print('\n' + '=' * 82)
print('  MONTHLY -- Method A vs B vs C  [Relative Loss %]')
print('-' * 82)
print(f"  {'Month':<8} {'A (pooled)':>12} {'B (min-mean)':>14} "
      f"{'C (day-mean)':>14} {'A-B':>8} {'C-B':>8}")
print('-' * 82)
for _, row in df_monthly_abc.iterrows():
    mname = MONTH_NAMES[int(row['month']) - 1]
    if np.isnan(row['method_A']):
        print(f"  {mname:<8} {'N/A':>12} {'N/A':>14} {'N/A':>14} {'N/A':>8} {'N/A':>8}")
        continue
    a, b, c = row['method_A'], row['method_B'], row['method_C']
    print(f"  {mname:<8} {a:>11.2f}% {b:>13.2f}% {c:>13.2f}% "
          f"{a-b:>+7.2f} {c-b:>+7.2f}")
print('=' * 82)

# ==============================================================================
# 6. FIGURE KC-1 -- Monthly A vs B vs C grouped bars
# ==============================================================================
print('\nGenerating KC-1 : Monthly Method A vs B vs C...')

x = np.arange(12)
w = 0.26

fig, ax = plt.subplots(figsize=(15, 6.5))
a_vals = df_monthly_abc['method_A'].values
b_vals = df_monthly_abc['method_B'].values
c_vals = df_monthly_abc['method_C'].values

ax.bar(x - w, a_vals, w, label='Method A -- Pooled Ratio-of-Sums (minute pooling)',
       color=COLOR_A, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.bar(x,     b_vals, w, label='Method B -- Mean-of-Minute-Ratios (current)',
       color=COLOR_B, alpha=0.80, edgecolor='white', linewidth=0.5)
ax.bar(x + w, c_vals, w, label='Method C -- Mean-of-Daily-Ratios (sensor-day pooling)',
       color=COLOR_C, alpha=0.80, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(MONTH_NAMES, fontsize=11)
ax.set_xlabel('Month', fontsize=12, labelpad=8)
ax.set_ylabel('Relative Shading Loss [%]', fontsize=12, labelpad=8)
finite_vals = np.concatenate([
    a_vals[~np.isnan(a_vals)], b_vals[~np.isnan(b_vals)], c_vals[~np.isnan(c_vals)]
])
ax.set_ylim(0, finite_vals.max() * 1.25 + 2 if len(finite_vals) else 100)
ax.set_title('Monthly Relative Shading Loss — Three Aggregation Methods',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper left', framealpha=0.88)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.01,
         'A: Σloss/Σ(loss+captured) over all pooled minutes  |  '
         'B: mean(loss_i/(loss_i+captured_i))  |  '
         'C: mean over sensor-days of [Σloss/Σ(loss+captured)] per day',
         ha='center', fontsize=8, color='#666666', style='italic')
plt.tight_layout()
_save(fig, 'kc1_monthly_ABC')

# ==============================================================================
# 7. ELEVATION x kc CROSS-TABULATION
# ==============================================================================
print('\nComputing elevation x kc cross-tabulation...')

elev_edges  = list(range(0, 65, 5))
elev_labels = [f'{lo}-{lo+5}' for lo in elev_edges[:-1]]
data['elev_bin'] = pd.cut(data['sun_elev'], bins=elev_edges,
                          labels=elev_labels, right=False)

kc_edges  = [0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
kc_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.3']
data['kc_bin'] = pd.cut(data['kc'], bins=kc_edges,
                        labels=kc_labels, right=False)

cross_tab_rows = []
for e_label in elev_labels:
    for k_label in kc_labels:
        sub = data[(data['elev_bin'] == e_label) & (data['kc_bin'] == k_label)]
        n = len(sub)
        mean_r = sub['loss_rel_min_pct'].mean() if n >= MIN_CELL_MINUTES else np.nan
        cross_tab_rows.append({
            'elev_bin': e_label, 'kc_bin': k_label,
            'mean_loss_rel_pct': mean_r, 'n_minutes': n,
        })
df_cross = pd.DataFrame(cross_tab_rows)

pivot_mean  = df_cross.pivot(index='kc_bin', columns='elev_bin', values='mean_loss_rel_pct')
pivot_mean  = pivot_mean.reindex(index=kc_labels, columns=elev_labels)
pivot_count = df_cross.pivot(index='kc_bin', columns='elev_bin', values='n_minutes')
pivot_count = pivot_count.reindex(index=kc_labels, columns=elev_labels)

# ==============================================================================
# 8. FIGURE KC-2 -- Heatmap of mean relative loss
# ==============================================================================
print('Generating KC-2 : Heatmap mean relative loss by elevation x kc...')

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(pivot_mean.values, cmap='RdYlGn_r', aspect='auto',
               vmin=np.nanmin(pivot_mean.values), vmax=np.nanmax(pivot_mean.values))

ax.set_xticks(range(len(elev_labels)))
ax.set_xticklabels(elev_labels, fontsize=9, rotation=30, ha='right')
ax.set_yticks(range(len(kc_labels)))
ax.set_yticklabels(kc_labels, fontsize=9)
ax.set_xlabel('Solar Elevation Angle [°]', fontsize=11, labelpad=8)
ax.set_ylabel('Clear-Sky Index k_c = GHI / GHI_clear-sky', fontsize=11, labelpad=8)
ax.set_title('Mean Per-Minute Relative Shading Loss [%]\n'
             'by Solar Elevation and Clear-Sky Index',
             fontsize=12, fontweight='bold')

for i in range(len(kc_labels)):
    for j in range(len(elev_labels)):
        v = pivot_mean.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=8, color='#222222')

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Relative Loss [%]', fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
_save(fig, 'kc2_heatmap_mean_loss')

# ==============================================================================
# 9. FIGURE KC-3 -- Heatmap of minute counts (sample weight)
# ==============================================================================
print('Generating KC-3 : Heatmap minute counts by elevation x kc...')

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(pivot_count.values, cmap='Blues', aspect='auto')

ax.set_xticks(range(len(elev_labels)))
ax.set_xticklabels(elev_labels, fontsize=9, rotation=30, ha='right')
ax.set_yticks(range(len(kc_labels)))
ax.set_yticklabels(kc_labels, fontsize=9)
ax.set_xlabel('Solar Elevation Angle [°]', fontsize=11, labelpad=8)
ax.set_ylabel('Clear-Sky Index $k_c$', fontsize=11, labelpad=8)
ax.set_title('Sample Size (Minute Count)\nby Solar Elevation and Clear-Sky Index',
             fontsize=12, fontweight='bold')

for i in range(len(kc_labels)):
    for j in range(len(elev_labels)):
        v = pivot_count.values[i, j]
        if not np.isnan(v) and v > 0:
            ax.text(j, i, f'{int(v):,}', ha='center', va='center',
                    fontsize=7, color='#222222')

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('N minutes', fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
_save(fig, 'kc3_heatmap_counts')

# ==============================================================================
# 10. FIGURE KC-4 -- Covariance mechanism, two representative elevation bins
# ==============================================================================
print('Generating KC-4 : Covariance mechanism by kc tercile...')

# Representative bins: one low-elevation (where SA-12 showed A > B),
# one mid-elevation (where SA-12 showed the gap shrinking / reversing).
rep_elev_bins = ['0-5', '20-25']

fig, axes = plt.subplots(1, len(rep_elev_bins), figsize=(14, 6))
if len(rep_elev_bins) == 1:
    axes = [axes]

for ax, e_label in zip(axes, rep_elev_bins):
    sub = data[data['elev_bin'] == e_label].dropna(subset=['kc'])
    if len(sub) < MIN_CELL_MINUTES:
        ax.set_title(f'{e_label}° -- insufficient data')
        continue
    # Terciles of kc within this elevation bin
    try:
        sub = sub.copy()
        sub['kc_tercile'] = pd.qcut(sub['kc'], 3,
                                    labels=['Low kc (strong attenuation)', 'Mid kc',
                                'High kc (near clear-sky reference)'])
    except ValueError:
        ax.set_title(f'{e_label}° -- insufficient kc variability')
        continue

    grp = sub.groupby('kc_tercile', observed=True).agg(
        mean_r=('loss_rel_min_pct', 'mean'),
        n=('loss_rel_min_pct', 'count'),
    ).reindex(['Low kc (strong attenuation)', 'Mid kc',
               'High kc (near clear-sky reference)'])

    bars = ax.bar(range(3), grp['mean_r'].values, color=['#90A4AE', '#FFB300', '#E05C5C'],
                  alpha=0.85, edgecolor='white', linewidth=0.8)
    for i, (mv, nv) in enumerate(zip(grp['mean_r'].values, grp['n'].values)):
        if not np.isnan(mv):
            ax.text(i, mv + 1, f'{mv:.1f}%\n(n={int(nv):,})',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(3))
    ax.set_xticklabels(grp.index, fontsize=9)
    ax.set_ylabel('Mean Per-Minute Relative Loss [%]', fontsize=10)
    ax.set_title(f'Elevation {e_label}°', fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Mechanism Check: Does Relative Loss Depend on Clear-Sky Index (kc)?\n'
             'A rising trend from low to high kc indicates that minutes closer to '
             'the clear-sky reference are associated with higher relative loss; '
             'this may contribute to Method A exceeding Method B.',
             fontsize=11.5, fontweight='bold', y=1.04)
plt.tight_layout()
_save(fig, 'kc4_covariance_mechanism')

# ==============================================================================
# 11. FIGURE KC-5 -- Method A vs Method C scatter (hourly + monthly cells)
# ==============================================================================
print('Generating KC-5 : Method A vs Method C scatter...')

CONDITIONS = {
    'Urban':  data['clc_group'] == 'Urban',
    'Rural':  data['clc_group'] == 'Rural',
}

HOURS_PLOT = list(range(6, 21))
abc_cells = []
for cond_name, mask in CONDITIONS.items():
    for h in HOURS_PLOT:
        sub = data[mask & (data['hour'] == h)]
        if len(sub) < MIN_MINUTES:
            continue
        abc_cells.append({
            'condition': cond_name,
            'method_A':  method_A_pooled_ratio_of_sums(sub),
            'method_B':  method_B_mean_of_minute_ratios(sub),
            'method_C':  method_C_mean_of_daily_ratios(sub),
        })

df_abc_cells = pd.DataFrame(abc_cells).dropna()

fig, ax = plt.subplots(figsize=(8, 8))
cond_colors = {'Urban': '#E05C5C', 'Rural': '#2E7D32'}
for cond_name, color in cond_colors.items():
    sel = df_abc_cells[df_abc_cells['condition'] == cond_name]
    ax.scatter(sel['method_C'], sel['method_A'], c=color, alpha=0.7, s=40, label=cond_name)

lim = max(df_abc_cells['method_A'].max(), df_abc_cells['method_C'].max()) * 1.05 + 2
ax.plot([0, lim], [0, lim], color='#333333', linewidth=1.2, linestyle='--', label='1:1 line')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel('Method C -- Mean-of-Daily-Ratios [%]', fontsize=11)
ax.set_ylabel('Method A -- Pooled Ratio-of-Sums [%]', fontsize=11)
ax.set_title('Method A vs Method C\n(hourly bins, Urban & Rural)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.88)
ax.grid(linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
_save(fig, 'kc5_scatter_A_vs_C')

# ==============================================================================
# 12. EXPORT NUMERICAL TABLES
# ==============================================================================
print('\nExporting numerical tables...')

cross_path = OUTPUT_DIR / f'sal_kc_diagnostic_crosstab_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_cross.to_csv(cross_path, index=False)
print(f'  Saved : {cross_path}')

monthly_path = OUTPUT_DIR / f'sal_kc_diagnostic_monthly_ABC_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_monthly_abc.to_csv(monthly_path, index=False)
print(f'  Saved : {monthly_path}')

print('\nDone.')
