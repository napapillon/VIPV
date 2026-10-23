"""
================================================================================
Clear-Sky Fraction Threshold Diagnostic
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Supports the choice of CLEAR_SKY_FRACTION, the threshold of the clear-sky
filter of clean_data.py (Step 3):

    ghi / ghi_clear  >=  CLEAR_SKY_FRACTION   (with ghi_clear > 0)

ghi and ghi_clear are CAMS (Heliosat-4) outputs for the 3 km x 3 km cell:
    ghi        GHI under the actual atmosphere (clouds included)
    ghi_clear  GHI under a cloud-free sky (aerosols included)
Their ratio is the clear-sky index Kc: 1.0 means a cloud-free cell at the
CAMS resolution, lower values partial or full cloud cover. ghi_extra (top of
atmosphere) is not used for Kc: it ignores aerosol and water-vapour
attenuation and would give an overly permissive threshold.

--------------------------------------------------------------------------------
Figures
--------------------------------------------------------------------------------
  F1  Histogram of Kc (all sensors): bimodal distribution; the valley
      between the clear-sky peak (near 1.0) and the cloudy tail is where the
      threshold belongs.
  F2  Cumulative distribution of Kc: share of minutes retained at each
      candidate threshold (data-retention cost).
  F3  Kc vs sun_elev (2-D density): checks for a low-elevation bias, which
      would point to a CAMS modelling artefact.
  F4  Retention vs threshold, per sensor: checks that no single sensor
      drives the distribution.
  F5  Kc vs Kt = ghi / ghi_extra: cross-check against the clearness-index
      thresholds of the literature (e.g. Brito 2023).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (FILE_SUFFIX / TUKTUK_FILE_SUFFIX)
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             clearsky_diagnostic_f1_histogram_<CAMPAIGN>_<variant_tag>.png
             clearsky_diagnostic_f2_cdf_<CAMPAIGN>_<variant_tag>.png
             clearsky_diagnostic_f3_scatter_elev_<CAMPAIGN>_<variant_tag>.png
             clearsky_diagnostic_f4_retention_per_sensor_<CAMPAIGN>_<variant_tag>.png
             clearsky_diagnostic_f5_kt_cross_validation_<CAMPAIGN>_<variant_tag>.png
             clearsky_diagnostic_summary_<CAMPAIGN>_<variant_tag>.csv

Requirements
------------
    pip install numpy pandas matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix: clean_data.py output BEFORE the clear-sky filter ----------
FILE_SUFFIX: str = 'sd_cleaned'          # standard campaign
TUKTUK_FILE_SUFFIX: str = 'sd_cleaned'   # tuk-tuk campaign

# --- Outputs ------------------------------------------------------------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "3_irradiance" / "clearsky_diagnostic_outputs"
SHOW_FIGURES: bool = False

# --- Diagnostic parameters ----------------------------------------------------
# Minimum ghi_clear for a meaningful clear-sky index Kc [W/m²]
MIN_GHI_CLEAR: float = 1.0

# Candidate thresholds marked on all figures
CANDIDATE_THRESHOLDS: list[float] = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

# Value used in clean_data.py (highlighted)
CURRENT_THRESHOLD: float = 0.95

# --- Colour per sensor --------------------------------------------------------
SENSOR_COLORS: dict[str, str] = {
    'J30': '#1565C0',
    'J31': '#2E7D32',
    'J32': '#E65100',
    'J33': '#6A1B9A',
}

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
            s: data_dir / f'{s.lower()}_satellite_landuse_v2_{suffix}.csv'
            for s in sensors
        }
    elif campaign == "tuktuk":
        return {
            s: tuktuk_data_dir / f'{s.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv'
            for s in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Input suffix of the active campaign, used in output paths and names."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


_VARIANT_TAG: str = _variant_tag()
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
)

# One sub-folder per campaign / variant, so that runs never overwrite each other
OUTPUT_DIR: Path = OUTPUT_BASE_DIR / CAMPAIGN / _VARIANT_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# HELPERS
# ==============================================================================

def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure at 300 DPI (embedding CAMPAIGN and the variant tag in the
    filename) and optionally display it."""
    path = OUTPUT_DIR / f'clearsky_diagnostic_{name}_{CAMPAIGN}_{_VARIANT_TAG}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved : {path}')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ==============================================================================
# DATA LOADING
# ==============================================================================

print()
print('═' * 70)
print('  Clear-Sky Fraction Threshold Diagnostic')
print('═' * 70)
print(f'  CAMPAIGN : {CAMPAIGN}')
print(f'  Variant  : {_VARIANT_TAG}')
print()

frames: list[pd.DataFrame] = []

for sensor_id, csv_path in SENSOR_FILES.items():
    if not csv_path.exists():
        print(f'  [WARNING] File not found, skipping : {csv_path}')
        continue

    df = pd.read_csv(csv_path, low_memory=False)
    df['sensor'] = sensor_id

    # Standardise sensor-specific column names
    df = df.rename(columns={
        f'{sensor_id} lat':       'lat',
        f'{sensor_id} lon':       'lon',
        f'{sensor_id} Rad raw':   'rad_raw',
        f'{sensor_id} Rad calib': 'rad_calib',
    })

    # Keep only columns required for this diagnostic
    keep_cols = ['sensor', 'sun_elev', 'ghi', 'ghi_clear', 'ghi_extra', 'rad_calib']
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    for col in ['sun_elev', 'ghi', 'ghi_clear', 'ghi_extra', 'rad_calib']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    frames.append(df)
    print(f'  Loaded {sensor_id}: {len(df):>7,} rows')

if not frames:
    raise FileNotFoundError(
        'No sensor files found.  Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX in CONFIG.'
    )

data = pd.concat(frames, ignore_index=True)
print(f'\n  Total loaded : {len(data):>7,} rows across {len(frames)} sensor(s)')

# ── Apply diagnostic filters ──────────────────────────────────────────────────
# Keep rows with valid CAMS irradiance values.
mask = (
    data['ghi_clear'].notna()
    & (data['ghi_clear'] > MIN_GHI_CLEAR)
    & data['ghi'].notna()
)
data = data[mask].copy()
print(f'  After ghi_clear filter : {len(data):>7,} rows')

# ── Compute clear-sky index Kc ──────────────────────────────────────────────
data['kc'] = data['ghi'] / data['ghi_clear']

# Clip to [0, 1.05] for display — values slightly above 1.0 can occur due to
# sub-grid CAMS variability and are physically acceptable.
data['kc_clipped'] = data['kc'].clip(lower=0.0, upper=1.05)

# Classical clearness index Kt = ghi / ghi_extra (for Figure F5)
data['kt'] = np.where(
    data['ghi_extra'] > 1.0,
    (data['ghi'] / data['ghi_extra']).clip(lower=0.0, upper=1.2),
    np.nan,
)

print('\n  Kc (ghi/ghi_clear) — descriptive statistics:')
print(data['kc'].describe().to_string())
print()

# ==============================================================================
# FIGURE F1 — Histogram of clear-sky index Kc
# ==============================================================================
print('[F1] Histogram of ghi / ghi_clear ...')

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    data['kc_clipped'],
    bins=105,
    range=(0.0, 1.05),
    color='#455A64',
    alpha=0.75,
    edgecolor='none',
    density=True,
    label='All sensors combined',
)

# Mark candidate thresholds
for thr in CANDIDATE_THRESHOLDS:
    lw   = 2.2 if thr == CURRENT_THRESHOLD else 1.2
    ls   = '--' if thr == CURRENT_THRESHOLD else ':'
    col  = '#D32F2F' if thr == CURRENT_THRESHOLD else '#78909C'
    lab  = f'{thr:.2f}  (current default)' if thr == CURRENT_THRESHOLD else f'{thr:.2f}'
    ax.axvline(thr, color=col, linewidth=lw, linestyle=ls, label=lab)

ax.set_xlabel('Clear-sky index  Kc = ghi / ghi_clear', fontsize=12, labelpad=8)
ax.set_ylabel('Probability density', fontsize=12, labelpad=8)
ax.set_title(
    'Distribution of clear-sky index Kc\n'
    r'Clear-sky minutes cluster near 1.0 — place threshold in the valley',
    fontsize=12, fontweight='bold',
)
ax.legend(fontsize=9, framealpha=0.88)
ax.set_xlim(0.0, 1.06)
ax.grid(linestyle='--', alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
fig.text(
    0.5, -0.02,
    f'ghi_clear > {MIN_GHI_CLEAR} W/m²  |  '
    f'n = {len(data):,} minutes  |  Campaign/variant: {CAMPAIGN}/{_VARIANT_TAG}',
    ha='center', fontsize=8.5, color='#666666', style='italic',
)
plt.tight_layout()
_save_fig(fig, 'f1_histogram')

# ==============================================================================
# FIGURE F2 — CDF and data-retention curve
# ==============================================================================
print('[F2] CDF of ghi / ghi_clear ...')

sorted_ratio = np.sort(data['kc_clipped'].dropna().values)
cdf          = np.arange(1, len(sorted_ratio) + 1) / len(sorted_ratio)
retention    = 1.0 - cdf   # fraction retained at each threshold value

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sorted_ratio, retention * 100.0, color='#1565C0', linewidth=2,
        label='% minutes retained  (ghi/ghi_clear >= threshold)')

for thr in CANDIDATE_THRESHOLDS:
    ret_pct = float(np.interp(thr, sorted_ratio, retention * 100.0))
    lw      = 2.0 if thr == CURRENT_THRESHOLD else 1.2
    col     = '#D32F2F' if thr == CURRENT_THRESHOLD else '#78909C'
    ls      = '--' if thr == CURRENT_THRESHOLD else ':'
    ax.axvline(thr, color=col, linewidth=lw, linestyle=ls)
    ax.annotate(
        f'{thr:.2f} → {ret_pct:.1f} %',
        xy=(thr, ret_pct),
        xytext=(thr - 0.01, ret_pct + 3),
        fontsize=8.5,
        color=col,
        ha='right',
    )

ax.set_xlabel('CLEAR_SKY_FRACTION threshold', fontsize=12, labelpad=8)
ax.set_ylabel('% of minutes retained', fontsize=12, labelpad=8)
ax.set_title(
    'Data-retention cost vs clear-sky threshold\n'
    'Read off: "with threshold X, I keep Y % of minutes"',
    fontsize=12, fontweight='bold',
)
ax.set_xlim(0.0, 1.06)
ax.set_ylim(0, 105)
ax.grid(linestyle='--', alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
fig.text(
    0.5, -0.02,
    f'ghi_clear > {MIN_GHI_CLEAR} W/m²  |  '
    f'n = {len(data):,} minutes  |  Campaign/variant: {CAMPAIGN}/{_VARIANT_TAG}',
    ha='center', fontsize=8.5, color='#666666', style='italic',
)
plt.tight_layout()
_save_fig(fig, 'f2_cdf')

# ==============================================================================
# FIGURE F3 — Scatter: Kc vs sun_elev (2-D density)
# ==============================================================================
print('[F3] Scatter Kc vs sun_elev ...')

fig, ax = plt.subplots(figsize=(10, 6))

# Hexbin density plot for large datasets
hb = ax.hexbin(
    data['sun_elev'],
    data['kc_clipped'],
    gridsize=80,
    cmap='Blues',
    mincnt=1,
    bins='log',
)
cb = fig.colorbar(hb, ax=ax, label='log10(count)')
cb.ax.tick_params(labelsize=9)

for thr in CANDIDATE_THRESHOLDS:
    lw  = 2.0 if thr == CURRENT_THRESHOLD else 1.0
    col = '#D32F2F' if thr == CURRENT_THRESHOLD else '#FF7043'
    ls  = '--' if thr == CURRENT_THRESHOLD else ':'
    ax.axhline(thr, color=col, linewidth=lw, linestyle=ls,
               label=f'threshold = {thr:.2f}' if thr == CURRENT_THRESHOLD else f'{thr:.2f}')

ax.set_xlabel('Solar elevation [degrees]', fontsize=12, labelpad=8)
ax.set_ylabel('Clear-sky index  Kc = ghi / ghi_clear', fontsize=12, labelpad=8)
ax.set_title(
    'Clear-sky index Kc vs solar elevation\n'
    'No systematic bias with sun_elev expected for a well-behaved clear-sky model',
    fontsize=12, fontweight='bold',
)
ax.legend(fontsize=9, framealpha=0.88, loc='lower right')
ax.set_ylim(0.0, 1.06)
ax.grid(linestyle='--', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.text(
    0.5, -0.02,
    f'All available solar elevations  |  '
    f'n = {len(data):,} minutes  |  Campaign/variant: {CAMPAIGN}/{_VARIANT_TAG}',
    ha='center', fontsize=8.5, color='#666666', style='italic',
)
plt.tight_layout()
_save_fig(fig, 'f3_scatter_elev')

# ==============================================================================
# FIGURE F4 — Retention fraction per sensor
# ==============================================================================
print('[F4] Retention fraction per sensor ...')

fig, ax = plt.subplots(figsize=(10, 6))

for sensor_id, color in SENSOR_COLORS.items():
    sub = data[data['sensor'] == sensor_id]['kc_clipped'].dropna()
    if len(sub) == 0:
        continue
    sorted_sub  = np.sort(sub.values)
    cdf_sub     = np.arange(1, len(sorted_sub) + 1) / len(sorted_sub)
    ret_sub     = 1.0 - cdf_sub
    ax.plot(sorted_sub, ret_sub * 100.0, color=color, linewidth=2,
            label=f'{sensor_id}  (n={len(sub):,})')

for thr in CANDIDATE_THRESHOLDS:
    lw  = 2.0 if thr == CURRENT_THRESHOLD else 1.0
    col = '#D32F2F' if thr == CURRENT_THRESHOLD else '#90A4AE'
    ls  = '--' if thr == CURRENT_THRESHOLD else ':'
    ax.axvline(thr, color=col, linewidth=lw, linestyle=ls)

ax.set_xlabel('CLEAR_SKY_FRACTION threshold', fontsize=12, labelpad=8)
ax.set_ylabel('% of minutes retained', fontsize=12, labelpad=8)
ax.set_title(
    'Data-retention cost per sensor\n'
    'Divergence between sensors may signal differences in spatial coverage or campaign period',
    fontsize=12, fontweight='bold',
)
ax.legend(fontsize=9.5, framealpha=0.88)
ax.set_xlim(0.0, 1.06)
ax.set_ylim(0, 105)
ax.grid(linestyle='--', alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
fig.text(
    0.5, -0.02,
    f'Campaign/variant: {CAMPAIGN}/{_VARIANT_TAG}',
    ha='center', fontsize=8.5, color='#666666', style='italic',
)
plt.tight_layout()
_save_fig(fig, 'f4_retention_per_sensor')

# ==============================================================================
# FIGURE F5 — Cross-validation: Kc vs Kt (classical clearness index)
# ==============================================================================
print('[F5] Cross-validation Kc vs Kt ...')

df_f5 = data[data['kt'].notna()].copy()

fig, ax = plt.subplots(figsize=(10, 6.5))

hb2 = ax.hexbin(
    df_f5['kt'],
    df_f5['kc_clipped'],
    gridsize=80,
    cmap='Greens',
    mincnt=1,
    bins='log',
    extent=(0.0, 1.2, 0.0, 1.06),
)
cb2 = fig.colorbar(hb2, ax=ax, label='log10(count)')
cb2.ax.tick_params(labelsize=9)

# Horizontal lines: CLEAR_SKY_FRACTION candidates on the Kc axis
for thr in CANDIDATE_THRESHOLDS:
    lw  = 2.0 if thr == CURRENT_THRESHOLD else 1.0
    col = '#D32F2F' if thr == CURRENT_THRESHOLD else '#FF8F00'
    ls  = '--' if thr == CURRENT_THRESHOLD else ':'
    ax.axhline(thr, color=col, linewidth=lw, linestyle=ls,
               label=f'Kc threshold = {thr:.2f}')

# Vertical lines: common Kt thresholds from the literature
for kt_thr, kt_label in [(0.65, 'Kt = 0.65'), (0.70, 'Kt = 0.70')]:
    ax.axvline(kt_thr, color='#1565C0', linewidth=1.2, linestyle='-.', label=kt_label)

ax.set_xlabel('Kt  =  ghi / ghi_extra  (classical clearness index)', fontsize=12, labelpad=8)
ax.set_ylabel('Clear-sky index  Kc = ghi / ghi_clear', fontsize=12, labelpad=8)
ax.set_title(
    'Cross-validation: CAMS clear-sky index Kc vs classical clearness index Kt\n'
    r'Use to align CLEAR_SKY_FRACTION with Kt-based thresholds from the literature',
    fontsize=12, fontweight='bold',
)
ax.legend(fontsize=8.5, framealpha=0.88, ncol=2)
ax.set_xlim(0.0, 1.2)
ax.set_ylim(0.0, 1.06)
ax.grid(linestyle='--', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.text(
    0.5, -0.02,
    f'All available solar elevations  |  '
    f'n = {len(df_f5):,} minutes  |  Campaign/variant: {CAMPAIGN}/{_VARIANT_TAG}',
    ha='center', fontsize=8.5, color='#666666', style='italic',
)
plt.tight_layout()
_save_fig(fig, 'f5_kt_cross_validation')

# ==============================================================================
# SUMMARY CSV
# ==============================================================================
print('\n[Summary] Computing numerical summary ...')

summary_rows: list[dict] = []

for thr in CANDIDATE_THRESHOLDS:
    n_total    = len(data)
    n_retained = int((data['kc'] >= thr).sum())
    pct        = 100.0 * n_retained / max(n_total, 1)

    row: dict = {
        'threshold':            thr,
        'n_total':              n_total,
        'n_retained':           n_retained,
        'pct_retained':         round(pct, 2),
    }

    # Per-sensor retention
    for sensor_id in SENSOR_COLORS:
        sub = data[data['sensor'] == sensor_id]
        n_s = len(sub)
        n_r = int((sub['kc'] >= thr).sum())
        row[f'pct_retained_{sensor_id}'] = round(100.0 * n_r / max(n_s, 1), 2)

    # Corresponding Kt percentile at this Kc level (F5 cross-validation)
    if df_f5['kt'].notna().any():
        kt_at_thr = float(df_f5.loc[df_f5['kc'] >= thr, 'kt'].median())
        row['kt_median_above_threshold'] = round(kt_at_thr, 3)
    else:
        row['kt_median_above_threshold'] = np.nan

    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / f'clearsky_diagnostic_summary_{CAMPAIGN}_{_VARIANT_TAG}.csv'
df_summary.to_csv(summary_path, index=False)
print(f'  Saved : {summary_path}')
print()
print(df_summary.to_string(index=False))

# ==============================================================================
# READING GUIDE
# ==============================================================================
print()
print('═' * 70)
print('  READING GUIDE')
print('─' * 70)
print('  F1  Look for a bimodal distribution.  The valley between the clear-sky')
print('      peak (near 1.0) and the cloudy tail is the natural threshold zone.')
print()
print('  F2  "With threshold X, I retain Y % of minutes."')
print('      Balance between data-retention cost and clear-sky purity.')
print()
print('  F3  Use the full sun_elev range to inspect whether Kc changes')
print('      systematically at low solar elevations.')
print()
print('  F4  Large divergence between sensors indicates unequal spatial or')
print('      temporal coverage.  The threshold choice may affect sensors')
print('      differently; document this in the paper.')
print()
print('  F5  Horizontal lines (Kc threshold) vs vertical lines (Kt threshold).')
print('      Find the Kc value that best corresponds to Kt >= 0.65 or 0.70')
print('      to align with literature definitions of clear sky.')
print('═' * 70)
print()
print('Done.')