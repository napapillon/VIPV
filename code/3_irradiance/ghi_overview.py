"""
================================================================================
GHI Overview — All-Sky vs Clear-Sky vs Extraterrestrial Irradiance
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Quick visual overview of the three CAMS irradiance columns of the sensor
files:

    ghi        : CAMS all-sky GHI                        [W/m²]
    ghi_clear  : CAMS clear-sky GHI (no clouds)          [W/m²]
    ghi_extra  : extraterrestrial (top-of-atmosphere) GHI [W/m²]

--------------------------------------------------------------------------------
Figures
--------------------------------------------------------------------------------
  1  Time series over [TIMESERIES_START, TIMESERIES_END], one panel per
     sensor: ghi, ghi_clear and ghi_extra overlaid (clear-sky envelope,
     cloud attenuation).
  2  Monthly box plots of ghi and ghi_clear, all sensors pooled (seasonal
     pattern, coverage gaps).
  3  Scatter ghi vs ghi_clear, one point per minute, coloured by sensor;
     the 1:1 line is a perfectly clear sky, points below it are attenuated.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (FILE_SUFFIX / TUKTUK_FILE_SUFFIX)
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             required columns: 'Human readable date (Europe/London)', 'ghi',
             'ghi_clear', 'ghi_extra'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             ghi_overview_timeseries_<CAMPAIGN>_<variant_tag>.png
             ghi_overview_monthly_<CAMPAIGN>_<variant_tag>.png
             ghi_overview_scatter_<CAMPAIGN>_<variant_tag>.png

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.2
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

warnings.filterwarnings('ignore')

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

# --- Input suffix (clean_data.py variant) -------------------------------------
FILE_SUFFIX: str = 'sd_cleaned_hq'          # standard campaign
TUKTUK_FILE_SUFFIX: str = 'sd_cleaned_hq'   # tuk-tuk campaign

# --- Sensors to include -------------------------------------------------------
SENSOR_IDS: list[str] = ['J30', 'J31', 'J32', 'J33']

# --- Time-series date range ('YYYY-MM-DD'; None = all data, may be slow) ------
TIMESERIES_START: str | None = '2022-08-01'
TIMESERIES_END:   str | None = '2022-08-31'

# --- Output directory (sub-folder per campaign / variant created) -------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "3_irradiance" / "ghi_outputs"

# --- Display ------------------------------------------------------------------
SHOW_FIGURES: bool = False
DPI: int = 300

# ==============================================================================
# HELPERS
# ==============================================================================

# Colour per sensor
SENSOR_COLOURS: dict[str, str] = {
    'J30': '#1565C0',
    'J31': '#C0392B',
    'J32': '#2E7D32',
    'J33': '#E65100',
}


def _build_sensor_files(campaign: str, sensor_ids: list[str], data_dir: Path,
                        tuktuk_data_dir: Path, suffix: str) -> dict[str, Path]:
    """
    Return a {sensor_id: file_path} dict for all requested sensors.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    if campaign == "standard":
        return {
            sid: data_dir / f'{sid.lower()}_satellite_landuse_v2_{suffix}.csv'
            for sid in sensor_ids
        }
    elif campaign == "tuktuk":
        return {
            sid: tuktuk_data_dir / f'{sid.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv'
            for sid in sensor_ids
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base: Path, campaign: str, variant_tag: str) -> Path:
    """Create and return the output directory for the active campaign/variant."""
    out = base / campaign / variant_tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_output_path(out_dir: Path, name: str, campaign: str, variant_tag: str) -> Path:
    """Build a full output path embedding CAMPAIGN and the variant tag."""
    return out_dir / f'{name}_{campaign}_{variant_tag}.png'


_VARIANT_TAG: str = _variant_tag()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sensor(file_path: Path, sensor_id: str) -> pd.DataFrame:
    """
    Load a single sensor CSV and return a cleaned DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the sensor CSV file.
    sensor_id : str
        Sensor identifier (e.g. 'J31') — added as a column.

    Returns
    -------
    pd.DataFrame with columns: timestamp, sensor, ghi, ghi_clear,
    ghi_extra.  Returns an empty DataFrame if the file is missing.
    """
    if not file_path.exists():
        print(f'  [WARNING] File not found, skipping: {file_path.name}')
        return pd.DataFrame()

    df = pd.read_csv(file_path, low_memory=False)

    # Parse timestamp
    ts_col = 'Human readable date (Europe/London)'
    df['timestamp'] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)  # tz-naive UTC

    df['sensor'] = sensor_id

    required = ['ghi', 'ghi_clear', 'ghi_extra']
    for col in required:
        if col not in df.columns:
            print(f'  [WARNING] Column "{col}" missing in {file_path.name}')
            return pd.DataFrame()

    keep = ['timestamp', 'sensor'] + required
    return df[keep].copy()


def load_all_sensors(sensor_files: dict[str, Path]) -> pd.DataFrame:
    """Load and concatenate all sensor files into a single DataFrame."""
    frames = [load_sensor(path, sid) for sid, path in sensor_files.items()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(
            'No sensor data could be loaded. Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX.'
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    print(f'  Loaded {len(combined):,} rows from '
          f'{", ".join(sorted(combined["sensor"].unique()))}')
    return combined


# ==============================================================================
# FIGURE 1 — TIME-SERIES
# ==============================================================================

def plot_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot ghi, ghi_clear, ghi_extra time-series for each sensor in a
    separate sub-panel.

    Parameters
    ----------
    df       : Full (unfiltered) DataFrame so night-time gaps are visible.
    out_path : Destination PNG file.
    """
    sensors = sorted(df['sensor'].unique())
    n = len(sensors)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Restrict to requested date range (applied here so the full dataset is
    # still available for the other two figures)
    sub = df.copy()
    if TIMESERIES_START:
        sub = sub[sub['timestamp'] >= pd.Timestamp(TIMESERIES_START)]
    if TIMESERIES_END:
        sub = sub[sub['timestamp'] <= pd.Timestamp(TIMESERIES_END)]

    for ax, sid in zip(axes, sensors):
        sdf = sub[sub['sensor'] == sid].sort_values('timestamp')

        ax.plot(sdf['timestamp'], sdf['ghi_extra'],
                color='#BDBDBD', lw=0.8, label='GHI extra (top-of-atmosphere)')
        ax.plot(sdf['timestamp'], sdf['ghi_clear'],
                color='#F4A800', lw=1.0, label='GHI clear-sky (CAMS)')
        ax.plot(sdf['timestamp'], sdf['ghi'],
                color=SENSOR_COLOURS.get(sid, '#333333'), lw=0.9,
                alpha=0.85, label='GHI (CAMS, all-sky)')

        ax.set_ylabel('Irradiance [W/m²]', fontsize=9)
        ax.set_title(f'Sensor {sid}', fontsize=10, fontweight='bold', loc='left')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', lw=0.4, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator())

    fig.suptitle('GHI — CAMS All-Sky vs Clear-Sky vs Extraterrestrial',
                 fontsize=12, fontweight='bold', y=1.01)
    axes[-1].set_xlabel('Date (UTC)', fontsize=9)

    period = ''
    if TIMESERIES_START and TIMESERIES_END:
        period = f'{TIMESERIES_START} → {TIMESERIES_END}'
    elif TIMESERIES_START:
        period = f'from {TIMESERIES_START}'

    fig.text(0.5, -0.01,
             f'Source: CAMS Heliosat  |  Variant: {CAMPAIGN}/{_VARIANT_TAG}'
             + (f'  |  Period: {period}' if period else ''),
             ha='center', fontsize=7, color='grey', style='italic')

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


# ==============================================================================
# FIGURE 2 — MONTHLY BOX PLOTS
# ==============================================================================

def plot_monthly_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    """
    Monthly distributions of ghi and ghi_clear (all sensors
    pooled) as side-by-side box plots.

    Parameters
    ----------
    df       : Input DataFrame.
    out_path : Destination PNG file.
    """
    df = df.copy()
    df['month'] = df['timestamp'].dt.month
    months = sorted(df['month'].unique())
    month_labels = [pd.Timestamp(2000, m, 1).strftime('%b') for m in months]

    ghi_data       = [df.loc[df['month'] == m, 'ghi'].values       for m in months]
    ghi_clear_data = [df.loc[df['month'] == m, 'ghi_clear'].values for m in months]

    x = np.arange(len(months))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    bp1 = ax.boxplot(ghi_data,
                     positions=x - width / 2,
                     widths=width,
                     patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color='black', lw=1.5),
                     boxprops=dict(facecolor='#90CAF9', alpha=0.8))

    bp2 = ax.boxplot(ghi_clear_data,
                     positions=x + width / 2,
                     widths=width,
                     patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color='black', lw=1.5),
                     boxprops=dict(facecolor='#F4A800', alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_ylabel('Irradiance [W/m²]', fontsize=9)
    ax.set_xlabel('Month', fontsize=9)
    ax.set_title('Monthly Distribution of GHI — All-Sky vs Clear-Sky\n'
                 '(all sensors pooled)',
                 fontsize=11, fontweight='bold')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]],
              ['GHI all-sky (CAMS)', 'GHI clear-sky (CAMS)'],
              fontsize=9, loc='upper left')
    ax.grid(axis='y', lw=0.4, alpha=0.5)
    ax.set_ylim(bottom=0)

    fig.text(0.5, -0.02,
             f'Source: CAMS Heliosat  |  Variant: {CAMPAIGN}/{_VARIANT_TAG}  |  '
             f'n = {len(df):,} minutes',
             ha='center', fontsize=7, color='grey', style='italic')

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


# ==============================================================================
# FIGURE 3 — SCATTER: ghi vs ghi_clear
# ==============================================================================

def plot_ghi_vs_clear(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter plot of CAMS all-sky GHI vs CAMS clear-sky GHI, coloured by
    sensor.  Points on the 1:1 diagonal represent clear-sky conditions;
    points below it indicate cloud or aerosol attenuation.

    Parameters
    ----------
    df       : Input DataFrame.
    out_path : Destination PNG file.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    sensors = sorted(df['sensor'].unique())
    for sid in sensors:
        sdf = df[df['sensor'] == sid]
        ax.scatter(sdf['ghi_clear'], sdf['ghi'],
                   s=2,
                   alpha=0.3,
                   color=SENSOR_COLOURS.get(sid, '#333333'),
                   label=f'Sensor {sid}  (n={len(sdf):,})',
                   rasterized=True)

    # 1:1 line
    max_val = max(df['ghi_clear'].max(), df['ghi'].max())
    ax.plot([0, max_val], [0, max_val],
            color='black', lw=1.2, ls='--', label='1:1 (perfect clear sky)')

    ax.set_xlabel('GHI clear-sky — CAMS  [W/m²]', fontsize=10)
    ax.set_ylabel('GHI all-sky — CAMS  [W/m²]', fontsize=10)
    ax.set_title('All-Sky vs Clear-Sky GHI\n'
                 'Points below the diagonal = cloud/aerosol attenuation',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, markerscale=4, loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(lw=0.4, alpha=0.4)
    ax.set_aspect('equal', adjustable='box')

    fig.text(0.5, -0.02,
             f'Source: CAMS Heliosat  |  Variant: {CAMPAIGN}/{_VARIANT_TAG}  |  '
             f'n = {len(df):,} min',
             ha='center', fontsize=7, color='grey', style='italic')

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the sensor files and produce the three overview figures."""
    print('=' * 70)
    print('  GHI OVERVIEW  —  All-Sky / Clear-Sky / Extraterrestrial')
    print(f'  CAMPAIGN : {CAMPAIGN}')
    print(f'  Variant  : {_VARIANT_TAG}')
    print('=' * 70)

    # ── Build paths ───────────────────────────────────────────────────────────
    sensor_files = _build_sensor_files(
        CAMPAIGN, SENSOR_IDS, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
    )
    out_dir = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)

    out_ts      = _build_output_path(out_dir, 'ghi_overview_timeseries', CAMPAIGN, _VARIANT_TAG)
    out_monthly = _build_output_path(out_dir, 'ghi_overview_monthly',    CAMPAIGN, _VARIANT_TAG)
    out_scatter = _build_output_path(out_dir, 'ghi_overview_scatter',    CAMPAIGN, _VARIANT_TAG)

    # ── Load data ─────────────────────────────────────────────────────────────
    print('\n[1/3] Loading sensor CSV files ...')
    df_all = load_all_sensors(sensor_files)

    # ── Figure 1: time-series (uses raw data so night gaps are visible) ───────
    print('\n[2/3] Plotting figures ...')
    print('  → Figure 1: time-series')
    plot_timeseries(df_all, out_ts)

    # ── Figure 2: monthly box plots ───────────────────────────────────────────
    print('  → Figure 2: monthly box plots')
    plot_monthly_boxplots(df_all, out_monthly)

    # ── Figure 3: ghi vs ghi_clear scatter ───────────────────────────────────
    print('  → Figure 3: scatter GHI vs GHI clear-sky')
    plot_ghi_vs_clear(df_all, out_scatter)

    if SHOW_FIGURES:
        plt.show()

    print('\n[3/3] Done.')
    print(f'  Output directory: {out_dir}')


if __name__ == '__main__':
    main()