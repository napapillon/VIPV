"""
================================================================================
Solar Autonomy and Autonomy Losses — Monthly Analysis (Method A)
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Main solar-autonomy script: computes and visualises the daily solar autonomy
and solar autonomy losses per calendar month, pooling all sensors of the
active campaign, with the relative shading loss aggregated by Method A
(pooled ratio of sums), the aggregation retained in this work.
solar_autonomy_losses_method_b.py and solar_autonomy_losses_method_c.py
produce the same nine figures with the two alternative aggregations;
compared with them:

  - Figures 1-2 (solar autonomy bar chart / box plot): identical (no
    relative-loss ratio involved).
  - Figure 3 (monthly absolute / relative loss box plots): the boxes (one
    point per sensor-day) are the same; the green mean line of the right
    panel reports the Method A ratio of the month.
  - Figures 4-9: the left-axis box plots and orange "mean absolute" line are
    the same (absolute Wh/Wp is a plain sum); the right-axis green "mean
    relative" line reports Method A.

Method A pools minutes directly, so it uses the minute-count threshold of the
absolute-loss box plots (MIN_MINUTES); no day-count threshold is needed.

Relative-loss aggregation (Methods A / B / C)
--------------------------------------------
Three scripts produce the same nine figures and differ only in how the
relative shading-loss percentage (%) of Figures 4-9 is aggregated (see
section 2.5.3 of the report, docs/Internship_Report_Papillon.pdf, together
with sal_sensitivity_analysis.py and the SAL diagnostics). The absolute loss
(Wh/Wp, plain sum) is identical in all three.

  METHOD A — Pooled Ratio-of-Sums       (solar_autonomy_losses.py, the
                                          method retained in this work)
      All minutes of a bin are pooled; loss and captured energy are each
      summed once, then divided once:
          loss_rel_A = Σ loss_irr / (Σ loss_irr + Σ E_solar) × 100
      The exact total-energy loss fraction of the bin; each minute weighs
      in proportion to its own GHI.

  METHOD B — Mean-of-Minute-Ratios      (solar_autonomy_losses_method_b.py)
      The per-minute ratio is computed first, then averaged with equal
      weight per minute.

  METHOD C — Mean-of-Daily-Ratios       (solar_autonomy_losses_method_c.py)
      One ratio per sensor-day (loss and captured energy summed over the
      minutes of that day in the bin), averaged with equal weight per
      sensor-day.

Method
------
1. Solar energy per minute [kWh/kWp]:
       E_solar = <SENSOR> Rad calib [W/m²] / 1000 × Δt [h]
2. Solar autonomy per minute [km/kWp]:
       Autonomy = E_solar / CVE,  CVE = 16.6 kWh/100 km = 0.166 kWh/km
       (European Environment Agency / Weiss et al., 2024 — EU average BEV)
3. Irradiance loss per minute (zero when the sensor exceeds GHI):
       Loss_irr = max(0, GHI − <SENSOR> Rad calib) × Δt [h]
4. Autonomy loss per minute [km/kWp]:
       Loss_auto = Loss_irr / 1000 / CVE
5. Daily aggregation: E_solar and Loss_irr summed per (sensor, date), then
   averaged over the sensor-days of each calendar month.
Rows are kept when sun_elev > MIN_SUN_ELEVATION and rad_calib >
MIN_RAD_CALIB (missing land-use values are already removed by clean_data.py).
Parked / Moving: Haversine distance between consecutive 1-minute GPS points,
Moving if >= THRESHOLD_M; pairs across a gap > MAX_GAP_MINUTES or a date
change get a distance of 0.

Sparse data
-----------
Months with fewer than MIN_SENSOR_DAYS valid sensor-days are masked on every
figure (bar omitted, box suppressed, mean line broken, excluded from the
annual mean). Hourly and monthly (panel, bin) cells with fewer than
MIN_MINUTES minute-records are masked the same way.

Figures produced
----------------
  1. Stacked bar chart — monthly mean solar autonomy and autonomy losses
  2. Box plot          — monthly distribution of daily solar autonomy
  3. Box plots (×2)    — monthly absolute and relative irradiance losses
  4. Box plots (×2)    — hourly shading losses by land-use group (Urban / Rural)
  5. Box plots (×2)    — hourly shading losses by movement status (Parked / Moving)
  6. Box plots (×2)    — monthly shading losses by land-use group
  7. Box plots (×2)    — monthly shading losses by movement status
  8. Box plots (×4)    — hourly shading losses by combined condition
                         (Urban-Moving, Urban-Parked, Rural-Moving, Rural-Parked)
  9. Box plots (×4)    — monthly shading losses by combined condition

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
             sal_monthly_barchart_methodA,
             sal_monthly_boxplot_autonomy_methodA,
             sal_monthly_boxplot_losses_methodA,
             sal_hourly_losses_landuse_methodA,
             sal_hourly_losses_movement_methodA,
             sal_monthly_losses_landuse_methodA,
             sal_monthly_losses_movement_methodA,
             sal_hourly_losses_combined_methodA,
             sal_monthly_losses_combined_methodA
                 (_<CAMPAIGN>_<variant_tag>.png)
             sal_monthly_summary_methodA, sal_figure_statistics_methodA
                 (_<CAMPAIGN>_<variant_tag>.csv)

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
import matplotlib.ticker as ticker

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

OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "sal_A_outputs"

SHOW_FIGURES: bool = False

# Average Battery Electric Vehicle (BEV) energy consumption.
# Source: European Environment Agency / Weiss et al. (2024) — EU-wide average.
CVE_KWH_PER_100KM: float = 16.6
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0

DELTA_T_H: float = 1.0 / 60.0   # 1-minute data → hours per interval

LOCAL_TZ: str = 'Europe/Lisbon'

MIN_SUN_ELEVATION: float = 0.0
MIN_RAD_CALIB:     float = 0.0

THRESHOLD_M:     float = 30.0    # metres — Haversine movement threshold
R_EARTH_KM:      float = 6371.0
MAX_GAP_MINUTES: float = 2.0   # overnight / restart gap guard for Haversine

# Months with fewer than MIN_SENSOR_DAYS valid sensor-days are masked on
# every figure (bar omitted, box suppressed, mean line broken).
MIN_SENSOR_DAYS: int = 5

# Minute-level sparse-bin threshold. Used for the absolute-loss box plots
# (unaffected by the A/B/C choice) AND, in this Method A variant, for the
# relative-loss statistic itself, since Method A pools minutes directly --
# no separate day-count threshold is needed here, unlike Method C.
MIN_MINUTES: int = 600

HOUR_START_PLOT: int = 6
HOUR_END_PLOT:   int = 20
HOURS_PLOT: list[int] = list(range(HOUR_START_PLOT, HOUR_END_PLOT + 1))

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
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


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Build the output directory path, embedding the campaign and variant tag."""
    return base_dir / campaign / variant_tag


def _build_output_path(out_dir: Path, base_name: str, campaign: str,
                       variant_tag: str, extension: str) -> Path:
    """Build an output file path that embeds the active campaign and variant tag."""
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


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

MONTH_NAMES: list[str] = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _show_or_close(fig: plt.Figure) -> None:
    """Show the figure interactively if SHOW_FIGURES is True, otherwise close it."""
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def haversine_vectorized(lat1, lon1, lat2, lon2, R=R_EARTH_KM,
                         timestamps=None, max_gap_minutes=MAX_GAP_MINUTES):
    """
    Compute the great-circle distance in metres between consecutive GPS points.

    Uses the Haversine formula — numerically stable for short intra-urban
    distances typical of vehicle trajectory data.

    Parameters
    ----------
    lat1, lon1 : array-like
        Coordinates of point i (decimal degrees, WGS-84).
    lat2, lon2 : array-like
        Coordinates of point i + 1 (decimal degrees, WGS-84).
    R : float
        Earth radius in kilometres (default: 6371.0 — IAU 2012).
    timestamps : array-like of datetime64, optional
        Full timestamp array of length N (i.e. df['datetime'].values).
        When provided, pairs that cross a segment boundary are zeroed:
          - time gap > max_gap_minutes  (overnight stop, sensor restart)
          - calendar-date change        (explicit cross-day guard)
        When None (default), all distances are returned as computed.
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
        s           = pd.Series(np.asarray(timestamps))
        gap_s       = s.diff().dt.total_seconds().fillna(0).values
        dates       = s.dt.date.values
        is_boundary = (
            (gap_s[1:] > max_gap_minutes * 60)
            | (dates[1:] != dates[:-1])
        )
        distances[is_boundary] = 0.0

    return distances


def method_A_relative_loss(df_subset: pd.DataFrame,
                            min_minutes: int = MIN_MINUTES) -> tuple[float, int]:
    """
    Compute the Method A relative shading loss (%) for a subset of minutes.

    All minutes in df_subset are pooled: loss and captured energy are each
    summed once across every minute, then divided once. This is the exact
    total-energy loss fraction for the pooled period — weight of each
    minute is implicitly proportional to its own GHI.

    Parameters
    ----------
    df_subset : pd.DataFrame
        Rows already filtered to the bin of interest (e.g. one hour and one
        land-use condition, or one month and one movement status).
    min_minutes : int
        Minimum number of minutes required to report a value.

    Returns
    -------
    tuple[float, int]
        (pooled_relative_loss_pct, n_minutes). pooled_relative_loss_pct is
        NaN when n_minutes < min_minutes.
    """
    n = len(df_subset)
    if n < min_minutes:
        return np.nan, n
    total_loss  = df_subset['loss_irr_Wh'].sum()
    total_solar = df_subset['E_solar_Wh'].sum()
    denom = total_loss + total_solar
    if denom <= 0:
        return np.nan, n
    return (total_loss / denom * 100.0), n


def build_hourly_abs(df: pd.DataFrame, mask: pd.Series) -> list[np.ndarray]:
    """Per-hour absolute-loss arrays [Wh/Wp], unaffected by the A/B/C choice."""
    sub = df[mask & df['hour'].isin(HOURS_PLOT)]
    return [
        sub.loc[sub['hour'] == h, 'loss_irr_Wh'].dropna().values
        if len(sub.loc[sub['hour'] == h, 'loss_irr_Wh'].dropna()) >= MIN_MINUTES
        else np.array([])
        for h in HOURS_PLOT
    ]


def build_hourly_relA(df: pd.DataFrame, mask: pd.Series) -> tuple[list[float], list[int]]:
    """Per-hour Method A relative-loss values and their contributing minute counts."""
    sub = df[mask & df['hour'].isin(HOURS_PLOT)]
    values, n_minutes = [], []
    for h in HOURS_PLOT:
        v, n = method_A_relative_loss(sub[sub['hour'] == h])
        values.append(v)
        n_minutes.append(n)
    return values, n_minutes


def build_monthly_abs(df: pd.DataFrame, valid_months: set[int], mask: pd.Series) -> list[np.ndarray]:
    """Per-month absolute-loss arrays [Wh/Wp], unaffected by the A/B/C choice."""
    sub = df[mask]
    result = []
    for m in range(1, 13):
        if m not in valid_months:
            result.append(np.array([]))
        else:
            vals = sub.loc[sub['month'] == m, 'loss_irr_Wh'].dropna().values
            result.append(vals if len(vals) >= MIN_MINUTES else np.array([]))
    return result


def build_monthly_relA(df: pd.DataFrame, valid_months: set[int],
                       mask: pd.Series) -> tuple[list[float], list[int]]:
    """Per-month Method A relative-loss values and their contributing minute counts."""
    sub = df[mask]
    values, n_minutes = [], []
    for m in range(1, 13):
        if m not in valid_months:
            values.append(np.nan)
            n_minutes.append(0)
            continue
        v, n = method_A_relative_loss(sub[sub['month'] == m])
        values.append(v)
        n_minutes.append(n)
    return values, n_minutes


def mean_abs_from_lists(lists: list[np.ndarray]) -> list[float]:
    """Mean absolute loss per bin, NaN when under-sampled (unchanged logic)."""
    return [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in lists]


def draw_loss_boxplot(ax, data_list, mean_values, ylabel, title, line_color, line_label,
                      yformat_pct=False):
    """Draw a monthly loss box plot with a connected mean line."""
    ax.boxplot(data_list, positions=range(1, 13), widths=0.55, patch_artist=True,
               medianprops=dict(color='#333333', linewidth=1.8),
               boxprops=dict(facecolor='#E05C5C', alpha=0.70, linewidth=1.2),
               whiskerprops=dict(linewidth=1.2, linestyle='--'),
               capprops=dict(linewidth=1.5),
               flierprops=dict(marker='o', markersize=3, alpha=0.35, color='#888888', markeredgewidth=0))
    ax.plot(range(1, 13), mean_values, color=line_color, linewidth=2.0,
            marker='o', markersize=6, zorder=5, label=line_label)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_xlabel('Month', fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    if yformat_pct:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)


def draw_hourly_boxplot(ax, abs_lists, mean_abs, mean_relA, n_minutes, title):
    """
    Draw hourly shading loss box plots on a twin-axis Axes.

    Left y-axis  : absolute losses [Wh/Wp] — box plots + orange mean line
                   (unchanged from the Method B version).
    Right y-axis : relative losses [%] — green line now reports Method A
                   (pooled ratio-of-sums) instead of Method B.
    """
    positions = np.arange(len(HOURS_PLOT)) + 1
    ax.boxplot(abs_lists, positions=positions, widths=0.55, patch_artist=True,
               medianprops=dict(color='#333333', linewidth=1.8),
               boxprops=dict(facecolor='#E05C5C', alpha=0.65, linewidth=1.2),
               whiskerprops=dict(linewidth=1.2, linestyle='--'),
               capprops=dict(linewidth=1.5),
               flierprops=dict(marker='o', markersize=2.5, alpha=0.25, color='#888888', markeredgewidth=0))
    ax.plot(positions, mean_abs, color='#F4A800', linewidth=2.0, marker='o',
            markersize=6, zorder=5, label='Mean absolute [Wh/Wp]')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{h:02d}:00' for h in HOURS_PLOT], fontsize=9, rotation=45, ha='right')
    ax.set_xlabel('Hour of day (local time — Europe/Lisbon)', fontsize=11, labelpad=8)
    ax.set_ylabel('Shading Losses [Wh / Wp]', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    ax_r = ax.twinx()
    ax_r.plot(positions, mean_relA, color='#2E7D32', linewidth=2.0, marker='o',
              markersize=6, zorder=5, label='Mean relative [%] (Method A)')
    ax_r.set_ylabel('Shading Losses [%]', fontsize=11, labelpad=8, color='#2E7D32')
    ax_r.tick_params(axis='y', labelcolor='#2E7D32')
    ax_r.set_ylim(0, 100)
    ax_r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax_r.spines[['top']].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left', framealpha=0.85)


def draw_monthly_boxplot(ax, abs_lists, mean_abs, mean_relA, n_minutes, title):
    """Monthly counterpart of draw_hourly_boxplot — same layout, x-axis = months."""
    positions = np.arange(1, 13)
    ax.boxplot(abs_lists, positions=positions, widths=0.55, patch_artist=True,
               medianprops=dict(color='#333333', linewidth=1.8),
               boxprops=dict(facecolor='#E05C5C', alpha=0.65, linewidth=1.2),
               whiskerprops=dict(linewidth=1.2, linestyle='--'),
               capprops=dict(linewidth=1.5),
               flierprops=dict(marker='o', markersize=2.5, alpha=0.25, color='#888888', markeredgewidth=0))
    ax.plot(positions, mean_abs, color='#F4A800', linewidth=2.0, marker='o',
            markersize=6, zorder=5, label='Mean absolute [Wh/Wp]')
    ax.set_xticks(positions)
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_xlabel('Month', fontsize=11, labelpad=8)
    ax.set_ylabel('Shading Losses [Wh / Wp]', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    ax_r = ax.twinx()
    ax_r.plot(positions, mean_relA, color='#2E7D32', linewidth=2.0, marker='o',
              markersize=6, zorder=5, label='Mean relative [%] (Method A)')
    ax_r.set_ylabel('Shading Losses [%]', fontsize=11, labelpad=8, color='#2E7D32')
    ax_r.tick_params(axis='y', labelcolor='#2E7D32')
    ax_r.set_ylim(0, 100)
    ax_r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax_r.spines[['top']].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left', framealpha=0.85)


def draw_boxplot_combined(ax, positions, xticklabels, xlabel, abs_lists, mean_abs, mean_relA, title):
    """Shared compact-layout drawer for the 2x2 combined-condition panels (Figures 8-9)."""
    ax.boxplot(abs_lists, positions=positions, widths=0.55, patch_artist=True,
               medianprops=dict(color='#333333', linewidth=1.8),
               boxprops=dict(facecolor='#E05C5C', alpha=0.65, linewidth=1.2),
               whiskerprops=dict(linewidth=1.2, linestyle='--'),
               capprops=dict(linewidth=1.5),
               flierprops=dict(marker='o', markersize=2.5, alpha=0.25, color='#888888', markeredgewidth=0))
    ax.plot(positions, mean_abs, color='#F4A800', linewidth=2.0, marker='o',
            markersize=6, zorder=5, label='Mean absolute [Wh/Wp]')
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=8, rotation=45 if len(xticklabels) > 12 else 0, ha='right' if len(xticklabels) > 12 else 'center')
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel('Shading Losses [Wh / Wp]', fontsize=9, labelpad=6)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    ax_r = ax.twinx()
    ax_r.plot(positions, mean_relA, color='#2E7D32', linewidth=2.0, marker='o',
              markersize=6, zorder=5, label='Mean relative [%] (Method A)')
    ax_r.set_ylabel('Shading Losses [%]', fontsize=9, labelpad=6, color='#2E7D32')
    ax_r.tick_params(axis='y', labelcolor='#2E7D32', labelsize=8)
    ax_r.set_ylim(0, 100)
    ax_r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax_r.spines[['top']].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left', framealpha=0.85)


def _row_relA(fig_id, panel, x_label, x_value, value, n_minutes):
    """One stats-table row for a Method A relative-loss data point."""
    return {
        'figure_id': fig_id, 'panel': panel, 'x_label': x_label, 'x_value': x_value,
        'variable': 'loss_rel_pct_methodA', 'n_obs': n_minutes, 'mean': value,
        'median': np.nan, 'std': np.nan, 'q25': np.nan, 'q75': np.nan,
        'under_sampled': n_minutes < MIN_MINUTES,
        'masked_on_fig': np.isnan(value),
    }


def main() -> None:
    """
    Run the full Solar Autonomy Losses analysis (Method A) and export all
    figures, the monthly summary CSV and the Method A statistics CSV.

    See the module docstring for the full figure list and the relationship
    to solar_autonomy_losses_method_b.py (Method B) and solar_autonomy_losses_method_c.py.
    """

    # ── Resolve paths from CAMPAIGN + variant tag ─────────────────────────────
    _VARIANT_TAG: str = _variant_tag()
    SENSOR_FILES: dict[str, Path] = _build_sensor_files(
        CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
    )
    OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # All Method A outputs carry a "_methodA" tag.
    OUTPUT_IMAGE                  = _build_output_path(OUTPUT_DIR, 'sal_monthly_barchart_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_CSV                    = _build_output_path(OUTPUT_DIR, 'sal_monthly_summary_methodA', CAMPAIGN, _VARIANT_TAG, 'csv')
    OUTPUT_IMAGE_BP                = _build_output_path(OUTPUT_DIR, 'sal_monthly_boxplot_autonomy_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_LOSS              = _build_output_path(OUTPUT_DIR, 'sal_monthly_boxplot_losses_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_HOURLY            = _build_output_path(OUTPUT_DIR, 'sal_hourly_losses_landuse_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MOVE              = _build_output_path(OUTPUT_DIR, 'sal_hourly_losses_movement_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MONTHLY_LANDUSE   = _build_output_path(OUTPUT_DIR, 'sal_monthly_losses_landuse_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MONTHLY_MOVE      = _build_output_path(OUTPUT_DIR, 'sal_monthly_losses_movement_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_HOURLY_COMB       = _build_output_path(OUTPUT_DIR, 'sal_hourly_losses_combined_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MONTHLY_COMB      = _build_output_path(OUTPUT_DIR, 'sal_monthly_losses_combined_methodA', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_STATS                   = _build_output_path(OUTPUT_DIR, 'sal_figure_statistics_methodA', CAMPAIGN, _VARIANT_TAG, 'csv')

    # ==============================================================================
    # 1. LOAD AND POOL DATA FROM ALL SENSORS
    # ==============================================================================
    print('=' * 70)
    print('  Solar Autonomy Monthly Analysis — Method A variant')
    print(f'  CAMPAIGN     : {CAMPAIGN}')
    print(f'  Data variant : {_VARIANT_TAG}')
    print('=' * 70)

    all_frames: list[pd.DataFrame] = []

    for sensor, csv_path in SENSOR_FILES.items():
        cols = SENSOR_COLS[sensor]
        if not csv_path.exists():
            print(f'  [WARNING] File not found, skipping : {csv_path}')
            continue

        print(f'\nLoading sensor {sensor} : {csv_path.name}')
        df = pd.read_csv(csv_path, low_memory=False)
        print(f'  → {len(df):,} rows loaded')

        rename_map = {
            cols['date']: 'datetime', cols['rad']: 'rad_calib', cols['ghi']: 'ghi',
            cols['elev']: 'sun_elev', cols['land']: 'land_use_clcv',
            cols['lat']: 'lat', cols['lon']: 'lon',
        }
        df = df.rename(columns=rename_map)

        needed = ['datetime', 'rad_calib', 'ghi', 'sun_elev', 'land_use_clcv', 'lat', 'lon']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f'  [WARNING] Missing columns {missing}, skipping {sensor}')
            continue
        df = df[needed].copy()

        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
        df['datetime'] = df['datetime'].dt.tz_convert(LOCAL_TZ)
        df = df.dropna(subset=['datetime'])

        for col in ['rad_calib', 'ghi', 'sun_elev', 'land_use_clcv', 'lat', 'lon']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[(df['land_use_clcv'] >= 1) & (df['land_use_clcv'] <= 44)].copy()

        n_before = len(df)
        df = df[df['sun_elev']  > MIN_SUN_ELEVATION].copy()
        df = df[df['rad_calib'] > MIN_RAD_CALIB].copy()
        df = df.dropna(subset=['rad_calib', 'ghi']).reset_index(drop=True)
        print(f'  → {len(df):,} valid rows after filtering  ({n_before - len(df):,} excluded)')

        df['sensor'] = sensor
        df['date']   = df['datetime'].dt.date
        df['month']  = df['datetime'].dt.month
        df['hour']   = df['datetime'].dt.hour

        all_frames.append(df)

    if not all_frames:
        raise RuntimeError('No valid data found. Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX.')

    data = pd.concat(all_frames, ignore_index=True)
    print(f'\nTotal valid measurements across all sensors : {len(data):,}')
    # ==============================================================================
    # 2. COMPUTE PER-MINUTE ENERGY AND LOSSES
    # ==============================================================================
    print('\nComputing per-minute energy and autonomy losses...')

    data['E_solar_kWh'] = data['rad_calib'] / 1000.0 * DELTA_T_H
    data['E_solar_Wh']  = data['E_solar_kWh'] * 1000.0
    data['loss_irr_Wh'] = np.maximum(0.0, data['ghi'] - data['rad_calib']) * DELTA_T_H
    data['loss_auto_km']  = data['loss_irr_Wh'] / 1000.0 / CVE_KWH_PER_KM
    data['auto_solar_km'] = data['E_solar_kWh'] / CVE_KWH_PER_KM

    # Per-minute relative loss [%] — kept for reference; NOT used for Method A,
    # which pools sums directly rather than averaging per-minute ratios.
    data['loss_rel_min_pct'] = (
        data['loss_irr_Wh'] / (data['loss_irr_Wh'] + data['E_solar_Wh']) * 100.0
    )
    data['loss_rel_min_pct'] = data['loss_rel_min_pct'].replace([np.inf, -np.inf], np.nan)

    data['clc_group'] = data['land_use_clcv'].map(CLC_URBAN_RURAL).fillna('Unknown')
    # ==============================================================================
    # 3. DAILY AGGREGATION (Figures 1-2)
    # ==============================================================================
    print('Aggregating to daily totals...')

    daily = (
        data.groupby(['sensor', 'date', 'month'], as_index=False)
        .agg(auto_solar_km=('auto_solar_km', 'sum'), loss_auto_km=('loss_auto_km', 'sum'))
    )
    print(f'  → {len(daily):,} sensor-day records')
    # ==============================================================================
    # 4. MONTHLY AVERAGES
    # ==============================================================================
    print('Computing monthly averages...')

    monthly = (
        daily.groupby('month', as_index=False)
        .agg(auto_solar_mean=('auto_solar_km', 'mean'),
             loss_auto_mean=('loss_auto_km', 'mean'),
             n_sensor_days=('auto_solar_km', 'count'))
    )
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly = all_months.merge(monthly, on='month', how='left')

    sparse_mask = monthly['n_sensor_days'].isna() | (monthly['n_sensor_days'] < MIN_SENSOR_DAYS)
    monthly.loc[sparse_mask, ['auto_solar_mean', 'loss_auto_mean']] = np.nan
    VALID_MONTHS: set[int] = set(monthly.loc[~sparse_mask, 'month'].tolist())

    sparse_months = monthly.loc[sparse_mask, 'month'].tolist()
    if sparse_months:
        names = [MONTH_NAMES[m - 1] for m in sparse_months]
        print(f'\n  [INFO] Months masked (< {MIN_SENSOR_DAYS} sensor-days): {", ".join(names)}')
    # ==============================================================================
    # 5. FIGURE 1 — STACKED BAR CHART (unchanged vs Method B version)
    # ==============================================================================
    print('\nGenerating stacked bar chart (Figure 1)...')

    x, width = np.arange(12), 0.55
    autos  = monthly['auto_solar_mean'].fillna(0).values
    losses = monthly['loss_auto_mean'].fillna(0).values

    fig1, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, autos, width, label='Solar Autonomy (achieved)', color='#F4C518',
           edgecolor='white', linewidth=0.5, zorder=3)
    ax.bar(x, losses, width, label='Solar Autonomy Losses (shading)', color='#E05C5C',
           edgecolor='white', linewidth=0.5, zorder=3, bottom=autos)

    for i, (a, l) in enumerate(zip(autos, losses)):
        m = i + 1
        if m not in VALID_MONTHS:
            continue
        if a > 0.8:
            ax.text(x[i], a / 2, f'{a:.1f}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if l > 0.8:
            ax.text(x[i], a + l / 2, f'{l:.1f}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        total = a + l
        if total > 0.5:
            ax.text(x[i], total + 0.2, f'{total:.1f}', ha='center', va='bottom',
                    fontsize=7.5, color='#333333', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES, fontsize=11)
    ax.set_xlabel('Month', fontsize=12, labelpad=8)
    ax.set_ylabel('km / kWp / day', fontsize=12, labelpad=8)
    ax.set_ylim(0, (autos + losses).max() * 1.18 + 1)
    ax.set_title('Daily Solar Autonomy and Solar Autonomy Losses per Month\n'
                 f'(All sensors pooled — CVE = {CVE_KWH_PER_100KM} kWh/100 km)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    annual_auto = daily.loc[daily['month'].isin(VALID_MONTHS), 'auto_solar_km'].mean()
    annual_loss = daily.loc[daily['month'].isin(VALID_MONTHS), 'loss_auto_km'].mean()
    annual_total = annual_auto + annual_loss
    ax.axhline(annual_auto, color='#c8a000', linewidth=1.3, linestyle='--',
               label=f'Annual mean autonomy ({annual_auto:.1f} km/kWp/day)')
    ax.axhline(annual_total, color='#333333', linewidth=1.0, linestyle=':',
               label=f'Annual mean clear-sky potential ({annual_total:.1f} km/kWp/day)')
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.88)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE}')
    _show_or_close(fig1)
    # ==============================================================================
    # 6. FIGURE 2 — MONTHLY BOX PLOT OF DAILY SOLAR AUTONOMY (unchanged)
    # ==============================================================================
    print('\nGenerating monthly solar autonomy box plot (Figure 2)...')

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    monthly_data_auto = [
        daily.loc[daily['month'] == m, 'auto_solar_km'].dropna().values
        if m in VALID_MONTHS else np.array([])
        for m in range(1, 13)
    ]
    ax2.boxplot(monthly_data_auto, positions=range(1, 13), widths=0.55, patch_artist=True,
                medianprops=dict(color='#333333', linewidth=1.8),
                boxprops=dict(facecolor='#F4C518', alpha=0.75, linewidth=1.2),
                whiskerprops=dict(linewidth=1.2, linestyle='--'),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.35, color='#888888', markeredgewidth=0))

    mean_auto_valid = [
        daily.loc[daily['month'] == m, 'auto_solar_km'].mean() if m in VALID_MONTHS else np.nan
        for m in range(1, 13)
    ]
    ax2.plot(range(1, 13), mean_auto_valid, color='#1565C0', linewidth=2.0,
             marker='o', markersize=6, zorder=5, label='Monthly mean')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_NAMES, fontsize=11)
    ax2.set_xlabel('Month', fontsize=12, labelpad=8)
    ax2.set_ylabel('Solar Autonomy [km / kWp / day]', fontsize=12, labelpad=8)
    ax2.set_title('Monthly Distribution of Daily Solar Autonomy — Box Plot\n'
                  f'(All sensors pooled — CVE = {CVE_KWH_PER_100KM} kWh/100 km)',
                  fontsize=13, fontweight='bold', pad=14)
    ax2.legend(fontsize=11, loc='upper left', framealpha=0.85)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_BP, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_BP}')
    _show_or_close(fig2)
    # ==============================================================================
    # 8. FIGURE 3 — MONTHLY IRRADIANCE LOSS BOX PLOTS
    #    Box distributions (one point per sensor-day) are UNCHANGED — they are
    #    informational, not the debated aggregate statistic. Only the green
    #    connected mean line on the right panel now reports Method A instead of
    #    the mean of the individual daily ratios shown by the box.
    # ==============================================================================
    print('\nGenerating monthly irradiance loss box plots (Figure 3)...')

    daily_losses = (
        data.groupby(['sensor', 'date', 'month'], as_index=False)
        .agg(loss_irr_kWh=('loss_irr_Wh', lambda s: s.sum() / 1000.0),
             E_solar_kWh=('E_solar_kWh', 'sum'))
    )
    daily_losses['loss_rel_pct'] = (
        daily_losses['loss_irr_kWh'] / (daily_losses['loss_irr_kWh'] + daily_losses['E_solar_kWh']) * 100.0
    )
    daily_losses['loss_rel_pct'] = daily_losses['loss_rel_pct'].replace([np.inf, -np.inf], np.nan)

    monthly_loss_abs = [
        daily_losses.loc[daily_losses['month'] == m, 'loss_irr_kWh'].dropna().values
        if m in VALID_MONTHS else np.array([])
        for m in range(1, 13)
    ]
    monthly_loss_rel = [
        daily_losses.loc[daily_losses['month'] == m, 'loss_rel_pct'].dropna().values
        if m in VALID_MONTHS else np.array([])
        for m in range(1, 13)
    ]
    mean_loss_abs = [np.nanmean(v) if (len(v) >= MIN_SENSOR_DAYS and m in VALID_MONTHS) else np.nan
                     for m, v in zip(range(1, 13), monthly_loss_abs)]

    # Method A monthly relative-loss line: pooled ratio-of-sums over ALL minutes
    # of that month (all sensors, all conditions pooled), NOT the mean of the
    # individual daily ratios shown by the box plot above.
    mean_loss_relA = []
    for m in range(1, 13):
        if m not in VALID_MONTHS:
            mean_loss_relA.append(np.nan)
            continue
        v, _n = method_A_relative_loss(data[data['month'] == m])
        mean_loss_relA.append(v)
    fig3, (ax3l, ax3r) = plt.subplots(1, 2, figsize=(16, 6))
    draw_loss_boxplot(ax3l, monthly_loss_abs, mean_loss_abs,
                      ylabel='Irradiance Losses [kWh / kWp / day]',
                      title='Monthly Absolute Irradiance Losses', line_color='#F4A800',
                      line_label='Monthly mean')
    draw_loss_boxplot(ax3r, monthly_loss_rel, mean_loss_relA,
                      ylabel='Irradiance Losses [%]',
                      title='Monthly Relative Irradiance Losses (Method A line; box = per-day distribution)',
                      line_color='#2E7D32', line_label='Monthly mean (Method A)', yformat_pct=True)
    fig3.suptitle('Monthly Distribution of Irradiance Losses due to Shading — Box Plots\n'
                  '(All sensors pooled | left: absolute losses, right: relative losses)',
                  fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_LOSS, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_LOSS}')
    _show_or_close(fig3)
    # ==============================================================================
    # 9. MOVEMENT CLASSIFICATION (needed for Figures 5, 7, 8, 9)
    # ==============================================================================
    print('\nClassifying movement status (Parked / Moving)...')
    moving_status_frames: list[pd.DataFrame] = []
    for sensor, csv_path in SENSOR_FILES.items():
        cols_s = SENSOR_COLS[sensor]
        if not csv_path.exists():
            continue
        df_s = pd.read_csv(csv_path, low_memory=False,
                           usecols=[cols_s['date'], cols_s['lat'], cols_s['lon']])
        df_s = df_s.rename(columns={cols_s['date']: 'datetime', cols_s['lat']: 'lat', cols_s['lon']: 'lon'})
        df_s['datetime'] = pd.to_datetime(df_s['datetime'], utc=True, errors='coerce')
        df_s['datetime'] = df_s['datetime'].dt.tz_convert(LOCAL_TZ)
        df_s = df_s.dropna(subset=['datetime', 'lat', 'lon'])
        for col in ['lat', 'lon']:
            df_s[col] = pd.to_numeric(df_s[col], errors='coerce')
        df_s = df_s.dropna(subset=['lat', 'lon']).sort_values('datetime').reset_index(drop=True)

        dist = haversine_vectorized(df_s['lat'].values[:-1], df_s['lon'].values[:-1],
                                    df_s['lat'].values[1:], df_s['lon'].values[1:],
                                    timestamps=df_s['datetime'].values)
        df_s = df_s.iloc[:-1].copy()
        df_s['moving_status'] = np.where(dist >= THRESHOLD_M, 'Moving', 'Parked')
        df_s['sensor'] = sensor
        moving_status_frames.append(df_s[['datetime', 'sensor', 'moving_status']])

    move_df = pd.concat(moving_status_frames, ignore_index=True)
    data = data.merge(move_df, on=['datetime', 'sensor'], how='left')
    data_move = data.dropna(subset=['moving_status']).copy()
    # ==============================================================================
    # 12. FIGURE 4 — HOURLY SHADING LOSSES BY LAND-USE GROUP
    # ==============================================================================
    print('\nGenerating hourly shading losses by land-use group (Figure 4)...')

    urban_mask = data['clc_group'] == 'Urban'
    rural_mask = data['clc_group'] == 'Rural'

    urban_abs = build_hourly_abs(data, urban_mask)
    rural_abs = build_hourly_abs(data, rural_mask)
    urban_relA, urban_relA_n = build_hourly_relA(data, urban_mask)
    rural_relA, rural_relA_n = build_hourly_relA(data, rural_mask)
    mean_urban_abs = mean_abs_from_lists(urban_abs)
    mean_rural_abs = mean_abs_from_lists(rural_abs)

    fig4, (ax4l, ax4r) = plt.subplots(1, 2, figsize=(16, 6))
    draw_hourly_boxplot(ax4l, urban_abs, mean_urban_abs, urban_relA, urban_relA_n, title='Shading Losses — Urban')
    draw_hourly_boxplot(ax4r, rural_abs, mean_rural_abs, rural_relA, rural_relA_n, title='Shading Losses — Rural')
    fig4.suptitle('Hourly Distribution of Shading Losses — Urban vs Rural\n'
                  '(All sensors pooled | boxes: absolute [Wh/Wp], green line: relative [%], Method A)',
                  fontsize=12, fontweight='bold', y=1.01)
    fig4.text(0.5, -0.01, f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}',
              ha='center', fontsize=8, color='#888888', style='italic')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_HOURLY, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_HOURLY}')
    _show_or_close(fig4)
    # ==============================================================================
    # 13. FIGURE 5 — HOURLY SHADING LOSSES BY MOVEMENT STATUS
    # ==============================================================================
    print('\nGenerating hourly shading losses by movement status (Figure 5)...')

    parked_mask = data_move['moving_status'] == 'Parked'
    moving_mask = data_move['moving_status'] == 'Moving'

    parked_abs = build_hourly_abs(data_move, parked_mask)
    moving_abs = build_hourly_abs(data_move, moving_mask)
    parked_relA, parked_relA_n = build_hourly_relA(data_move, parked_mask)
    moving_relA, moving_relA_n = build_hourly_relA(data_move, moving_mask)
    mean_parked_abs = mean_abs_from_lists(parked_abs)
    mean_moving_abs = mean_abs_from_lists(moving_abs)

    fig5, (ax5l, ax5r) = plt.subplots(1, 2, figsize=(16, 6))
    draw_hourly_boxplot(ax5l, parked_abs, mean_parked_abs, parked_relA, parked_relA_n, title='Shading Losses — Parked')
    draw_hourly_boxplot(ax5r, moving_abs, mean_moving_abs, moving_relA, moving_relA_n, title='Shading Losses — Moving')
    fig5.suptitle('Hourly Distribution of Shading Losses — Parked vs Moving\n'
                  f'(All sensors pooled | threshold: {THRESHOLD_M:.0f} m | '
                  'boxes: absolute [Wh/Wp], green line: relative [%], Method A)',
                  fontsize=12, fontweight='bold', y=1.01)
    fig5.text(0.5, -0.01, f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}',
              ha='center', fontsize=8, color='#888888', style='italic')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MOVE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MOVE}')
    _show_or_close(fig5)
    # ==============================================================================
    # 14. FIGURE 6 — MONTHLY SHADING LOSSES BY LAND-USE GROUP
    # ==============================================================================
    print('\nGenerating monthly shading losses by land-use group (Figure 6)...')

    urban_abs_m = build_monthly_abs(data, VALID_MONTHS, urban_mask)
    rural_abs_m = build_monthly_abs(data, VALID_MONTHS, rural_mask)
    urban_relA_m, urban_relA_m_n = build_monthly_relA(data, VALID_MONTHS, urban_mask)
    rural_relA_m, rural_relA_m_n = build_monthly_relA(data, VALID_MONTHS, rural_mask)
    mean_urban_abs_m = mean_abs_from_lists(urban_abs_m)
    mean_rural_abs_m = mean_abs_from_lists(rural_abs_m)

    fig6, (ax6l, ax6r) = plt.subplots(1, 2, figsize=(16, 6))
    draw_monthly_boxplot(ax6l, urban_abs_m, mean_urban_abs_m, urban_relA_m, urban_relA_m_n, title='Shading Losses — Urban')
    draw_monthly_boxplot(ax6r, rural_abs_m, mean_rural_abs_m, rural_relA_m, rural_relA_m_n, title='Shading Losses — Rural')
    fig6.suptitle('Monthly Distribution of Shading Losses — Urban vs Rural\n'
                  '(All sensors pooled | boxes: absolute [Wh/Wp], green line: relative [%], Method A)',
                  fontsize=12, fontweight='bold', y=1.01)
    fig6.text(0.5, -0.01, f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}',
              ha='center', fontsize=8, color='#888888', style='italic')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MONTHLY_LANDUSE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MONTHLY_LANDUSE}')
    _show_or_close(fig6)
    # ==============================================================================
    # 15. FIGURE 7 — MONTHLY SHADING LOSSES BY MOVEMENT STATUS
    # ==============================================================================
    print('\nGenerating monthly shading losses by movement status (Figure 7)...')

    parked_abs_m = build_monthly_abs(data_move, VALID_MONTHS, parked_mask)
    moving_abs_m = build_monthly_abs(data_move, VALID_MONTHS, moving_mask)
    parked_relA_m, parked_relA_m_n = build_monthly_relA(data_move, VALID_MONTHS, parked_mask)
    moving_relA_m, moving_relA_m_n = build_monthly_relA(data_move, VALID_MONTHS, moving_mask)
    mean_parked_abs_m = mean_abs_from_lists(parked_abs_m)
    mean_moving_abs_m = mean_abs_from_lists(moving_abs_m)

    fig7, (ax7l, ax7r) = plt.subplots(1, 2, figsize=(16, 6))
    draw_monthly_boxplot(ax7l, parked_abs_m, mean_parked_abs_m, parked_relA_m, parked_relA_m_n, title='Shading Losses — Parked')
    draw_monthly_boxplot(ax7r, moving_abs_m, mean_moving_abs_m, moving_relA_m, moving_relA_m_n, title='Shading Losses — Moving')
    fig7.suptitle('Monthly Distribution of Shading Losses — Parked vs Moving\n'
                  f'(All sensors pooled | threshold: {THRESHOLD_M:.0f} m | '
                  'boxes: absolute [Wh/Wp], green line: relative [%], Method A)',
                  fontsize=12, fontweight='bold', y=1.01)
    fig7.text(0.5, -0.01, f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}',
              ha='center', fontsize=8, color='#888888', style='italic')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MONTHLY_MOVE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MONTHLY_MOVE}')
    _show_or_close(fig7)
    # ==============================================================================
    # 16. FIGURES 8-9 — COMBINED CONDITION (Urban/Rural x Moving/Parked)
    # ==============================================================================
    print('\nGenerating combined-condition figures (Figures 8-9)...')

    COMBINED_CONDITIONS: list[dict] = [
        {'label': 'Urban + Moving', 'mask': (data_move['clc_group'] == 'Urban') & (data_move['moving_status'] == 'Moving')},
        {'label': 'Urban + Parked', 'mask': (data_move['clc_group'] == 'Urban') & (data_move['moving_status'] == 'Parked')},
        {'label': 'Rural + Moving', 'mask': (data_move['clc_group'] == 'Rural') & (data_move['moving_status'] == 'Moving')},
        {'label': 'Rural + Parked', 'mask': (data_move['clc_group'] == 'Rural') & (data_move['moving_status'] == 'Parked')},
    ]

    # --- Figure 8 : hourly, combined condition -------------------------------------
    comb_hourly: list[dict] = []
    for cond in COMBINED_CONDITIONS:
        abs_h = build_hourly_abs(data_move, cond['mask'])
        relA_h, n_h = build_hourly_relA(data_move, cond['mask'])
        comb_hourly.append({**cond, 'abs': abs_h, 'mean_abs': mean_abs_from_lists(abs_h),
                            'relA': relA_h, 'n_minutes': n_h})

    fig8, axes8 = plt.subplots(2, 2, figsize=(20, 12))
    hour_positions = np.arange(len(HOURS_PLOT)) + 1
    hour_labels = [f'{h:02d}:00' for h in HOURS_PLOT]
    for ax, cd in zip(axes8.flat, comb_hourly):
        draw_boxplot_combined(ax, hour_positions, hour_labels, 'Hour of day (local time — Europe/Lisbon)',
                              cd['abs'], cd['mean_abs'], cd['relA'], title=f"Shading Losses — {cd['label']}")
    fig8.suptitle('Hourly Distribution of Shading Losses — Urban/Rural x Moving/Parked\n'
                  f'(All sensors pooled | threshold: {THRESHOLD_M:.0f} m | '
                  'boxes: absolute [Wh/Wp], green line: relative [%], Method A)',
                  fontsize=13, fontweight='bold', y=1.01)
    fig8.text(0.5, -0.01, f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}',
              ha='center', fontsize=8, color='#888888', style='italic')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_HOURLY_COMB, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_HOURLY_COMB}')
    _show_or_close(fig8)

    # --- Figure 9 : monthly, combined condition ------------------------------------
    comb_monthly: list[dict] = []
    for cond in COMBINED_CONDITIONS:
        abs_m = build_monthly_abs(data_move, VALID_MONTHS, cond['mask'])
        relA_m, n_m = build_monthly_relA(data_move, VALID_MONTHS, cond['mask'])
        comb_monthly.append({**cond, 'abs': abs_m, 'mean_abs': mean_abs_from_lists(abs_m),
                             'relA': relA_m, 'n_minutes': n_m})

    fig9, axes9 = plt.subplots(2, 2, figsize=(20, 12))
    month_positions = np.arange(1, 13)
    for ax, cd in zip(axes9.flat, comb_monthly):
        draw_boxplot_combined(ax, month_positions, MONTH_NAMES, 'Month',
                              cd['abs'], cd['mean_abs'], cd['relA'], title=f"Shading Losses — {cd['label']}")
    fig9.suptitle('Monthly Distribution of Shading Losses — Urban/Rural x Moving/Parked\n'
                  f'(All sensors pooled | threshold: {THRESHOLD_M:.0f} m | '
                  'boxes: absolute [Wh/Wp], green line: relative [%], Method A)',
                  fontsize=13, fontweight='bold', y=1.01)
    fig9.text(0.5, -0.01, f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}',
              ha='center', fontsize=8, color='#888888', style='italic')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MONTHLY_COMB, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MONTHLY_COMB}')
    _show_or_close(fig9)
    # ==============================================================================
    # 17. EXPORT FIGURE STATISTICS
    # ==============================================================================
    print('\nExporting figure statistics...')

    stats_rows: list[dict] = []

    for i, h in enumerate(HOURS_PLOT):
        hlabel = f'{h:02d}:00'
        stats_rows.append(_row_relA('sal_hourly_losses_landuse_methodA', 'Urban', hlabel, h, urban_relA[i], urban_relA_n[i]))
        stats_rows.append(_row_relA('sal_hourly_losses_landuse_methodA', 'Rural', hlabel, h, rural_relA[i], rural_relA_n[i]))
        stats_rows.append(_row_relA('sal_hourly_losses_movement_methodA', 'Parked', hlabel, h, parked_relA[i], parked_relA_n[i]))
        stats_rows.append(_row_relA('sal_hourly_losses_movement_methodA', 'Moving', hlabel, h, moving_relA[i], moving_relA_n[i]))
        for cd in comb_hourly:
            stats_rows.append(_row_relA('sal_hourly_losses_combined_methodA', cd['label'], hlabel, h, cd['relA'][i], cd['n_minutes'][i]))

    for m in range(1, 13):
        mname = MONTH_NAMES[m - 1]
        stats_rows.append(_row_relA('sal_monthly_losses_landuse_methodA', 'Urban', mname, m, urban_relA_m[m - 1], urban_relA_m_n[m - 1]))
        stats_rows.append(_row_relA('sal_monthly_losses_landuse_methodA', 'Rural', mname, m, rural_relA_m[m - 1], rural_relA_m_n[m - 1]))
        stats_rows.append(_row_relA('sal_monthly_losses_movement_methodA', 'Parked', mname, m, parked_relA_m[m - 1], parked_relA_m_n[m - 1]))
        stats_rows.append(_row_relA('sal_monthly_losses_movement_methodA', 'Moving', mname, m, moving_relA_m[m - 1], moving_relA_m_n[m - 1]))
        for cd in comb_monthly:
            stats_rows.append(_row_relA('sal_monthly_losses_combined_methodA', cd['label'], mname, m, cd['relA'][m - 1], cd['n_minutes'][m - 1]))

    df_stats = pd.DataFrame(stats_rows)
    df_stats = df_stats[['figure_id', 'panel', 'x_label', 'x_value', 'variable',
                          'n_obs', 'mean', 'median', 'std', 'q25', 'q75',
                          'under_sampled', 'masked_on_fig']]
    df_stats.to_csv(OUTPUT_STATS, index=False)
    print(f'Figure statistics saved : {OUTPUT_STATS}')
    print(f'  → {len(df_stats):,} rows | {df_stats["masked_on_fig"].sum()} rows masked '
          f'(fewer than {MIN_MINUTES} contributing minutes)')

    # --- Monthly summary CSV ------------------------------------------------
    monthly.to_csv(OUTPUT_CSV, index=False)
    print(f'Monthly summary saved : {OUTPUT_CSV}')

    # ==========================================================================
    # DONE
    # ==========================================================================
    print('\n' + '=' * 70)
    print('Analysis complete (Method A).')
    print(f'  CAMPAIGN         : {CAMPAIGN}')
    print(f'  Data variant     : {_VARIANT_TAG}')
    print(f'  Output directory : {OUTPUT_DIR}')
    print('=' * 70)


# ==============================================================================
if __name__ == "__main__":
    main()