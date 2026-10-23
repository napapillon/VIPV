"""
================================================================================
Solar Autonomy and Autonomy Losses — Monthly Analysis (Method B)
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Computes and visualises the daily solar autonomy and solar autonomy losses
per calendar month, pooling all sensors of the active campaign. Figures 4-9
report the relative shading loss with Method B (mean of minute ratios).

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
             sal_monthly_barchart, sal_monthly_boxplot_autonomy,
             sal_monthly_boxplot_losses,
             sal_hourly_losses_landuse,
             sal_hourly_losses_movement,
             sal_monthly_losses_landuse,
             sal_monthly_losses_movement,
             sal_hourly_losses_combined,
             sal_monthly_losses_combined
                 (_<CAMPAIGN>_<variant_tag>.png)
             sal_monthly_summary, sal_figure_statistics
                 (_<CAMPAIGN>_<variant_tag>.csv)

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

# --- Output directory ---------------------------------------------------------
# A sub-folder <CAMPAIGN>/<variant tag> is created automatically.
OUTPUT_BASE_DIR = OUTPUT_ROOT / "7_solar_autonomy" / "sal_B_outputs"

# --- Display figures during execution? ----------------------------------------
SHOW_FIGURES: bool = False

# --- Physical / vehicle parameters -------------------------------------------
# Average Battery Electric Vehicle (BEV) energy consumption.
# Source: European Environment Agency / Weiss et al. (2024) — EU-wide average.
CVE_KWH_PER_100KM: float = 16.6             # kWh / 100 km
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0   # kWh / km

# Measurement time resolution: 1-minute data → 1/60 h per record
DELTA_T_H: float = 1.0 / 60.0              # hours per measurement interval

# --- Timezone for local-time parsing ------------------------------------------
LOCAL_TZ: str = 'Europe/Lisbon'

# --- Filters ------------------------------------------------------------------
MIN_SUN_ELEVATION: float = 0.0   # degrees — keep only daytime (sun_elev > 0)
MIN_RAD_CALIB:     float = 0.0   # W/m²   — exclude residual zero/negative readings

# --- Movement classification --------------------------------------------------
# Haversine distance between consecutive 1-minute GPS points:
#   < THRESHOLD_M → Parked
#   ≥ THRESHOLD_M → Moving
THRESHOLD_M:     float = 30.0    # metres
R_EARTH_KM:      float = 6371.0  # mean Earth radius (km) — IAU 2012
MAX_GAP_MINUTES: int = 2        # overnight / restart gap guard for Haversine

# --- Sparse-month threshold ---------------------------------------------------
# Months with fewer than MIN_SENSOR_DAYS valid sensor-days are considered
# under-sampled and are masked on all figures (bar omitted, box suppressed,
# mean line broken).  This prevents statistical artefacts — such as negative
# relative losses — caused by an unrepresentative handful of measurements.
MIN_SENSOR_DAYS: int = 5

# --- Sparse-bin threshold for minute-level figures ----------------------------
# Hourly and monthly panels that split data by land-use or movement status
# (figs 4–7) operate at the 1-minute level.  Any (hour, panel) or
# (month, panel) cell with fewer than MIN_MINUTES observations is masked:
# the box plot receives an empty array (no box drawn) and the mean line
# receives NaN (gap in the line).  This threshold is independent of
# MIN_SENSOR_DAYS and applies only to figs 4–7.
MIN_MINUTES: int = 600

# --- Hour range for hourly shading-loss figures (daytime window) --------------
HOUR_START_PLOT: int = 6
HOUR_END_PLOT:   int = 20
HOURS_PLOT: list[int] = list(range(HOUR_START_PLOT, HOUR_END_PLOT + 1))

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# Input and output paths follow from CAMPAIGN and the input suffix.

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

# --- Column name mappings per sensor ------------------------------------------
# Maps generic field names to the actual column names found in each CSV.
# These are fixed across all FILE_SUFFIX variants (only the file suffix changes,
# not the internal column names).
SENSOR_COLS: dict[str, dict[str, str]] = {
    'J30': {'date': 'Human readable date (Europe/London)',
            'rad':  'J30 Rad calib', 'ghi': 'ghi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J30 lat',       'lon':  'J30 lon'},
    'J31': {'date': 'Human readable date (Europe/London)',
            'rad':  'J31 Rad calib', 'ghi': 'ghi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J31 lat',       'lon':  'J31 lon'},
    'J32': {'date': 'Human readable date (Europe/London)',
            'rad':  'J32 Rad calib', 'ghi': 'ghi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J32 lat',       'lon':  'J32 lon'},
    'J33': {'date': 'Human readable date (Europe/London)',
            'rad':  'J33 Rad calib', 'ghi': 'ghi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J33 lat',       'lon':  'J33 lon'},
}

# ==============================================================================
# CORINE LAND COVER (CLC) MAPPING — Urban / Rural classification
# Imported from clc_mapping.py (code/common/).
# land_use column = grid_code directly (1-based, range 1–44).
# CLC_URBAN_RURAL : dict[int, str]  land_use → 'Urban' / 'Rural'
# ==============================================================================

# Month labels used on the x-axis of monthly charts
MONTH_NAMES: list[str] = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _show_or_close(fig: plt.Figure) -> None:
    """
    Show the figure interactively if SHOW_FIGURES is True, otherwise close it.

    Parameters
    ----------
    fig : matplotlib Figure
        The figure to display or close.
    """
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


def build_hourly_lists(df: pd.DataFrame, mask: pd.Series, col: str) -> list[np.ndarray]:
    """
    Return a list (one array per hour in HOURS_PLOT) of values for a boxplot.

    Hours with fewer than MIN_MINUTES observations receive an empty array
    so that matplotlib renders no box or statistic for those positions.
    Shared by the land-use, movement-status, and combined-condition figures
    (Figures 4, 5, 8) — only the input DataFrame and mask differ.

    Parameters
    ----------
    df   : pd.DataFrame
        Minute-level dataset to draw from (`data` or `data_move`).
    mask : boolean Series aligned with df
        Row selection mask (land-use, movement status, or their intersection).
    col  : str
        Column name to extract (e.g. 'loss_irr_Wh', 'loss_rel_min_pct').

    Returns
    -------
    list of np.ndarray
        One array per hour; empty array when the bin is under-sampled.
    """
    sub = df[mask & df['hour'].isin(HOURS_PLOT)]
    return [
        sub.loc[sub['hour'] == h, col].dropna().values
        if len(sub.loc[sub['hour'] == h, col].dropna()) >= MIN_MINUTES
        else np.array([])
        for h in HOURS_PLOT
    ]


def build_monthly_lists(df: pd.DataFrame, valid_months: set[int],
                        mask: pd.Series, col: str) -> list[np.ndarray]:
    """
    Return a list (one array per calendar month) of values for a boxplot.

    A month is masked (empty array) when either it falls below
    MIN_SENSOR_DAYS (excluded from valid_months) or the (month, panel) cell
    has fewer than MIN_MINUTES minute-records. Shared by the land-use,
    movement-status, and combined-condition figures (Figures 6, 7, 9).

    Parameters
    ----------
    df           : pd.DataFrame
        Minute-level dataset to draw from (`data` or `data_move`).
    valid_months : set[int]
        Calendar months (1-12) meeting the MIN_SENSOR_DAYS threshold.
    mask         : boolean Series aligned with df
        Row selection mask.
    col          : str
        Column name to extract.

    Returns
    -------
    list of np.ndarray
        One array per month (1-12); empty array when the bin is masked.
    """
    sub = df[mask]
    result = []
    for m in range(1, 13):
        if m not in valid_months:
            result.append(np.array([]))
        else:
            vals = sub.loc[sub['month'] == m, col].dropna().values
            result.append(vals if len(vals) >= MIN_MINUTES else np.array([]))
    return result


def draw_loss_boxplot(ax, data_list, mean_values, ylabel, title, line_color,
                      yformat_pct=False):
    """
    Draw a monthly loss box plot with a connected mean line.

    Parameters
    ----------
    ax           : matplotlib Axes
    data_list    : list of 12 arrays, one per calendar month.
                   Sparse months must receive an empty array (np.array([]));
                   matplotlib then draws nothing for that position.
    mean_values  : list of 12 floats — monthly mean to overlay.
                   Use np.nan for sparse months to break the line.
    ylabel       : str — left y-axis label
    title        : str — panel title
    line_color   : str — colour for the mean line
    yformat_pct  : bool — if True, format y-axis ticks as percentages
    """
    ax.boxplot(
        data_list,
        positions=range(1, 13),
        widths=0.55,
        whis=1.5,
        patch_artist=True,
        medianprops=dict(color='#333333', linewidth=1.8),
        boxprops=dict(facecolor='#E05C5C', alpha=0.70, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.35, color='#888888',
                        markeredgewidth=0),
    )
    ax.plot(
        range(1, 13), mean_values,
        color=line_color, linewidth=2.0, marker='o', markersize=6,
        zorder=5, label='Monthly mean',
    )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_xlabel('Month', fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    if yformat_pct:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f'{v:.0f}%')
        )
        ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)


def draw_hourly_boxplot(ax, abs_lists, rel_lists, mean_abs, mean_rel, title):
    """
    Draw hourly shading loss box plots on a twin-axis Axes.

    Left y-axis  : absolute losses [Wh/Wp] — box plots + orange mean line
    Right y-axis : relative losses [%]     — green mean line only

    Under-sampled hours (< MIN_MINUTES records) receive empty arrays so that
    matplotlib draws nothing for those positions; the mean lines show a gap.

    Parameters
    ----------
    ax        : matplotlib Axes (primary)
    abs_lists : list of arrays — one per hour, absolute losses [Wh/Wp]
    rel_lists : list of arrays — one per hour, relative losses [%]
    mean_abs  : list of floats — hourly mean of absolute losses
    mean_rel  : list of floats — hourly mean of relative losses
    title     : str — panel title
    """
    positions = np.arange(len(HOURS_PLOT)) + 1

    ax.boxplot(
        abs_lists,
        positions=positions,
        widths=0.55,
        whis=1.5,
        patch_artist=True,
        medianprops=dict(color='#333333', linewidth=1.8),
        boxprops=dict(facecolor='#E05C5C', alpha=0.65, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=2.5, alpha=0.25,
                        color='#888888', markeredgewidth=0),
    )

    ax.plot(positions, mean_abs,
            color='#F4A800', linewidth=2.0, marker='o', markersize=6,
            zorder=5, label='Mean absolute [Wh/Wp]')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{h:02d}:00' for h in HOURS_PLOT],
                       fontsize=9, rotation=45, ha='right')
    ax.set_xlabel('Hour of day (local time — Europe/Lisbon)',
                  fontsize=11, labelpad=8)
    ax.set_ylabel('Shading Losses [Wh / Wp]', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    ax_r = ax.twinx()
    ax_r.plot(positions, mean_rel,
              color='#2E7D32', linewidth=2.0, marker='o', markersize=6,
              zorder=5, label='Mean relative [%]')
    ax_r.set_ylabel('Shading Losses [%]', fontsize=11, labelpad=8,
                    color='#2E7D32')
    ax_r.tick_params(axis='y', labelcolor='#2E7D32')
    ax_r.set_ylim(0, 100)
    ax_r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax_r.spines[['top']].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=9, loc='upper left', framealpha=0.85)


def draw_monthly_boxplot(ax, abs_lists, rel_lists, mean_abs, mean_rel, title):
    """
    Draw monthly shading loss box plots on a twin-axis Axes.

    Same visual layout as draw_hourly_boxplot but with x-axis = months.
    Under-sampled months (< MIN_SENSOR_DAYS or < MIN_MINUTES) receive empty
    arrays so that matplotlib draws nothing for those positions.

    Left y-axis  : absolute losses [Wh/Wp] — box plots + orange mean line
    Right y-axis : relative losses [%]     — green mean line only

    Parameters
    ----------
    ax        : matplotlib Axes (primary)
    abs_lists : list of 12 arrays — absolute losses [Wh/Wp]
    rel_lists : list of 12 arrays — relative losses [%]
    mean_abs  : list of 12 floats — monthly mean of absolute losses
    mean_rel  : list of 12 floats — monthly mean of relative losses
    title     : str — panel title
    """
    positions = np.arange(1, 13)

    ax.boxplot(
        abs_lists,
        positions=positions,
        widths=0.55,
        whis=1.5,
        patch_artist=True,
        medianprops=dict(color='#333333', linewidth=1.8),
        boxprops=dict(facecolor='#E05C5C', alpha=0.65, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=2.5, alpha=0.25,
                        color='#888888', markeredgewidth=0),
    )

    ax.plot(positions, mean_abs,
            color='#F4A800', linewidth=2.0, marker='o', markersize=6,
            zorder=5, label='Mean absolute [Wh/Wp]')

    ax.set_xticks(positions)
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_xlabel('Month', fontsize=11, labelpad=8)
    ax.set_ylabel('Shading Losses [Wh / Wp]', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    ax_r = ax.twinx()
    ax_r.plot(positions, mean_rel,
              color='#2E7D32', linewidth=2.0, marker='o', markersize=6,
              zorder=5, label='Mean relative [%]')
    ax_r.set_ylabel('Shading Losses [%]', fontsize=11, labelpad=8,
                    color='#2E7D32')
    ax_r.tick_params(axis='y', labelcolor='#2E7D32')
    ax_r.set_ylim(0, 100)
    ax_r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax_r.spines[['top']].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=9, loc='upper left', framealpha=0.85)


def draw_boxplot_combined(ax, positions, xticklabels, xlabel,
                          abs_lists, rel_lists, mean_abs, mean_rel, title,
                          rotate_xticks=False):
    """
    Draw shading-loss box plots for one Urban/Rural × Moving/Parked panel.

    Compact-layout counterpart of draw_hourly_boxplot / draw_monthly_boxplot,
    sized for the 2×2 grid used by Figures 8 and 9. Same colour convention:
    box face #E05C5C, mean absolute line #F4A800 (left axis), mean relative
    line #2E7D32 (right axis).

    Parameters
    ----------
    ax            : matplotlib Axes (primary)
    positions     : array-like — x-axis tick positions
    xticklabels   : list — x-axis tick labels (hour or month labels)
    xlabel        : str — x-axis title
    abs_lists     : list of arrays — absolute losses [Wh/Wp] per bin
    rel_lists     : list of arrays — relative losses [%] per bin
    mean_abs      : list of floats — mean absolute loss per bin
    mean_rel      : list of floats — mean relative loss per bin
    title         : str — panel title
    rotate_xticks : bool — rotate tick labels 45° (used for hourly panels)
    """
    ax.boxplot(
        abs_lists, positions=positions, widths=0.55, whis=1.5,
        patch_artist=True,
        medianprops=dict(color='#333333', linewidth=1.8),
        boxprops=dict(facecolor='#E05C5C', alpha=0.65, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=2.5, alpha=0.25,
                        color='#888888', markeredgewidth=0),
    )
    ax.plot(positions, mean_abs, color='#F4A800', linewidth=2.0, marker='o',
            markersize=6, zorder=5, label='Mean absolute [Wh/Wp]')

    ax.set_xticks(positions)
    if rotate_xticks:
        ax.set_xticklabels(xticklabels, fontsize=8, rotation=45, ha='right')
    else:
        ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel('Shading Losses [Wh / Wp]', fontsize=9, labelpad=6)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    ax_r = ax.twinx()
    ax_r.plot(positions, mean_rel, color='#2E7D32', linewidth=2.0, marker='o',
              markersize=6, zorder=5, label='Mean relative [%]')
    ax_r.set_ylabel('Shading Losses [%]', fontsize=9, labelpad=6, color='#2E7D32')
    ax_r.tick_params(axis='y', labelcolor='#2E7D32', labelsize=8)
    ax_r.set_ylim(0, 100)
    ax_r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax_r.spines[['top']].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
              loc='upper left', framealpha=0.85)


def _stats(arr: np.ndarray) -> dict:
    """Return a dict of descriptive statistics for a 1-D array."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan,
                'q25': np.nan, 'q75': np.nan}
    return {
        'mean':   round(float(np.mean(arr)),              6),
        'median': round(float(np.median(arr)),            6),
        'std':    round(float(np.std(arr, ddof=1)),       6),
        'q25':    round(float(np.percentile(arr, 25)),    6),
        'q75':    round(float(np.percentile(arr, 75)),    6),
    }


def _row(fig_id: str, panel: str, x_label: str, x_value: int,
         variable: str, arr: np.ndarray,
         under_sampled: bool = False,
         threshold: int = 0) -> dict:
    """
    Build one output statistics row.

    Parameters
    ----------
    fig_id        : str  — figure identifier (matches output PNG base name)
    panel         : str  — sub-panel label or 'All'
    x_label       : str  — human-readable x-axis label
    x_value       : int  — numeric x-axis value (for sorting)
    variable      : str  — variable name
    arr           : np.ndarray — raw observations for this cell
    under_sampled : bool — True when the month is flagged by MIN_SENSOR_DAYS
    threshold     : int  — minimum n_obs required; if len(arr) < threshold,
                           statistics are set to NaN (masked on figure).
                           Pass MIN_SENSOR_DAYS for sensor-day figures,
                           MIN_MINUTES for minute-level figures.
    """
    n = len(arr[~np.isnan(arr)])
    s = _stats(arr) if n >= threshold else {
        'mean': np.nan, 'median': np.nan, 'std': np.nan,
        'q25': np.nan,  'q75': np.nan,
    }
    return {
        'figure_id':     fig_id,
        'panel':         panel,
        'x_label':       x_label,
        'x_value':       x_value,
        'variable':      variable,
        'n_obs':         n,
        'mean':          s['mean'],
        'median':        s['median'],
        'std':           s['std'],
        'q25':           s['q25'],
        'q75':           s['q75'],
        'under_sampled': under_sampled,
        'masked_on_fig': under_sampled or (n < threshold),
    }


def main() -> None:
    """
    Run the full Solar Autonomy Losses analysis (Method B) and export all
    figures, the monthly summary CSV, and the per-figure statistics CSV.

    Workflow
    --------
     1. Load and pool minute-level data from all four sensors.
     2. Compute per-minute solar energy and shading losses.
     3. Aggregate to daily sensor-day totals.
     4. Compute monthly averages and flag under-sampled months.
     5. Print the monthly summary table to the console.
     6. Figure 1 : stacked bar chart of monthly autonomy / losses.
     7. Figure 2 : monthly box plot of daily solar autonomy.
     8. Figure 3 : monthly absolute / relative irradiance loss box plots.
     9. Figure 4 : hourly shading losses by land-use group.
    10. Figure 5 : hourly shading losses by movement status.
    11. Figure 6 : monthly shading losses by land-use group.
    12. Figure 7 : monthly shading losses by movement status.
    13. Figure 8 : hourly shading losses by combined condition.
    14. Figure 9 : monthly shading losses by combined condition.
    15. Export the monthly summary CSV.
    16. Export the per-figure statistics CSV.
    """

    # ── Resolve paths from CAMPAIGN + variant tag ─────────────────────────────
    _VARIANT_TAG: str = _variant_tag()
    SENSOR_FILES: dict[str, Path] = _build_sensor_files(
        CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
    )
    OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Output paths (images + CSVs) -----------------------------------------
    # Naming convention: sal_<description>_<FILE_SUFFIX>.<ext>
    OUTPUT_IMAGE                 = _build_output_path(OUTPUT_DIR, 'sal_monthly_barchart', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_CSV                   = _build_output_path(OUTPUT_DIR, 'sal_monthly_summary', CAMPAIGN, _VARIANT_TAG, 'csv')
    OUTPUT_IMAGE_BP               = _build_output_path(OUTPUT_DIR, 'sal_monthly_boxplot_autonomy', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_LOSS             = _build_output_path(OUTPUT_DIR, 'sal_monthly_boxplot_losses', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_HOURLY           = _build_output_path(OUTPUT_DIR, 'sal_hourly_losses_landuse', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MOVE             = _build_output_path(OUTPUT_DIR, 'sal_hourly_losses_movement', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MONTHLY_LANDUSE  = _build_output_path(OUTPUT_DIR, 'sal_monthly_losses_landuse', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MONTHLY_MOVE     = _build_output_path(OUTPUT_DIR, 'sal_monthly_losses_movement', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_HOURLY_COMB      = _build_output_path(OUTPUT_DIR, 'sal_hourly_losses_combined', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_IMAGE_MONTHLY_COMB     = _build_output_path(OUTPUT_DIR, 'sal_monthly_losses_combined', CAMPAIGN, _VARIANT_TAG, 'png')
    OUTPUT_STATS                  = _build_output_path(OUTPUT_DIR, 'sal_figure_statistics', CAMPAIGN, _VARIANT_TAG, 'csv')

    # ==============================================================================
    # 1.  LOAD AND POOL DATA FROM ALL SENSORS
    # ==============================================================================
    print('=' * 70)
    print('  Solar Autonomy Monthly Analysis')
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

        # ── Rename sensor-specific columns to generic names ───────────────────────
        rename_map = {
            cols['date']: 'datetime',
            cols['rad']:  'rad_calib',
            cols['ghi']:  'ghi',
            cols['elev']: 'sun_elev',
            cols['land']: 'land_use_clcv',
            cols['lat']:  'lat',
            cols['lon']:  'lon',
        }
        df = df.rename(columns=rename_map)

        # ── Keep only the columns required for this analysis ──────────────────────
        needed = ['datetime', 'rad_calib', 'ghi', 'sun_elev', 'land_use_clcv', 'lat', 'lon']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f'  [WARNING] Missing columns {missing}, skipping {sensor}')
            continue
        df = df[needed].copy()

        # ── Parse datetime (UTC-aware) and convert to local timezone ──────────────
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
        df['datetime'] = df['datetime'].dt.tz_convert(LOCAL_TZ)
        df = df.dropna(subset=['datetime'])

        # ── Cast all numeric columns ───────────────────────────────────────────────
        for col in ['rad_calib', 'ghi', 'sun_elev', 'land_use_clcv', 'lat', 'lon']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ── Safety filter: exclude any residual out-of-range land_use values ────────
        # clean_data.py enforces 1 <= land_use <= 44; this guard ensures correctness
        # even when FILE_SUFFIX points to a partially cleaned variant.
        df = df[(df['land_use_clcv'] >= 1) & (df['land_use_clcv'] <= 44)].copy()

        # ── Apply analysis filters ─────────────────────────────────────────────────
        # The _cleaned files have already removed land_use == -999 and rad_calib == 0.
        # We apply the remaining filters: daytime only and positive irradiance.
        n_before = len(df)
        df = df[df['sun_elev']  > MIN_SUN_ELEVATION].copy()
        df = df[df['rad_calib'] > MIN_RAD_CALIB].copy()
        df = df.dropna(subset=['rad_calib', 'ghi']).reset_index(drop=True)
        print(f'  → {len(df):,} valid rows after filtering  '
              f'({n_before - len(df):,} excluded)')

        # ── Add sensor label and time-decomposition columns ───────────────────────
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
    # 2.  COMPUTE PER-MINUTE ENERGY AND LOSSES
    # ==============================================================================
    print('\nComputing per-minute energy and autonomy losses...')

    # Solar energy produced per minute interval  [kWh/kWp]
    data['E_solar_kWh'] = data['rad_calib'] / 1000.0 * DELTA_T_H

    # Irradiance loss per minute  [Wh/m²] — zero when sensor output exceeds GHI
    data['loss_irr_Wh'] = np.maximum(0.0, data['ghi'] - data['rad_calib']) * DELTA_T_H

    # Autonomy loss per minute  [km/kWp]:  loss_irr_Wh / 1000 → kWh/m²  → / CVE
    data['loss_auto_km'] = data['loss_irr_Wh'] / 1000.0 / CVE_KWH_PER_KM

    # Solar autonomy per minute  [km/kWp]
    data['auto_solar_km'] = data['E_solar_kWh'] / CVE_KWH_PER_KM

    # ==============================================================================
    # 3.  DAILY AGGREGATION
    # ==============================================================================
    print('Aggregating to daily totals...')

    daily = (
        data
        .groupby(['sensor', 'date', 'month'], as_index=False)
        .agg(
            auto_solar_km = ('auto_solar_km', 'sum'),   # km/kWp/day
            loss_auto_km  = ('loss_auto_km',  'sum'),   # km/kWp/day
        )
    )
    print(f'  → {len(daily):,} sensor-day records')

    # ==============================================================================
    # 4.  MONTHLY AVERAGES
    # ==============================================================================
    print('Computing monthly averages...')

    monthly = (
        daily
        .groupby('month', as_index=False)
        .agg(
            auto_solar_mean = ('auto_solar_km', 'mean'),
            loss_auto_mean  = ('loss_auto_km',  'mean'),
            n_sensor_days   = ('auto_solar_km', 'count'),
        )
    )

    # Ensure all 12 months are represented, filling absent months with NaN
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly = all_months.merge(monthly, on='month', how='left')

    # Flag months with insufficient data as under-sampled.
    # Their values are replaced with NaN so they are silently excluded from all
    # figures and from the annual mean calculation.
    sparse_mask = monthly['n_sensor_days'].isna() | (monthly['n_sensor_days'] < MIN_SENSOR_DAYS)
    monthly.loc[sparse_mask, ['auto_solar_mean', 'loss_auto_mean']] = np.nan
    VALID_MONTHS: set[int] = set(monthly.loc[~sparse_mask, 'month'].tolist())

    sparse_months = monthly.loc[sparse_mask, 'month'].tolist()
    if sparse_months:
        names = [MONTH_NAMES[m - 1] for m in sparse_months]
        print(f'\n  [INFO] Months masked (< {MIN_SENSOR_DAYS} sensor-days): '
              f'{", ".join(names)}')
        print('         These months are excluded from all figures and the '
              'annual mean.')

    # ==============================================================================
    # 5.  PRINT SUMMARY TABLE
    # ==============================================================================
    print('\n' + '─' * 72)
    print(f"  {'Month':<12} {'Solar Autonomy':>18} {'Autonomy Losses':>18} "
          f"{'N sensor-days':>15}")
    print(f"  {'':12} {'[km/kWp/day]':>18} {'[km/kWp/day]':>18} {'':>15}")
    print('─' * 72)

    for _, row in monthly.iterrows():
        mname = MONTH_NAMES[int(row['month']) - 1]
        auto  = f"{row['auto_solar_mean']:.2f}" if not np.isnan(row['auto_solar_mean']) else ' N/A'
        loss  = f"{row['loss_auto_mean']:.2f}"  if not np.isnan(row['loss_auto_mean'])  else ' N/A'
        n     = f"{int(row['n_sensor_days'])}"  if not np.isnan(row['n_sensor_days'])   else ' N/A'
        print(f'  {mname:<12} {auto:>18} {loss:>18} {n:>15}')

    print('─' * 72)
    annual_auto = daily.loc[daily['month'].isin(VALID_MONTHS), 'auto_solar_km'].mean()
    annual_loss = daily.loc[daily['month'].isin(VALID_MONTHS), 'loss_auto_km'].mean()
    print(f"  {'Annual mean':<12} {annual_auto:>18.2f} {annual_loss:>18.2f}")
    print('─' * 72)

    # ==============================================================================
    # 6.  FIGURE — STACKED BAR CHART
    #     Bottom segment (yellow) : solar autonomy actually achieved  [km/kWp/day]
    #     Top segment    (red)    : autonomy lost to shading          [km/kWp/day]
    #     Total bar height        : theoretical clear-sky autonomy    [km/kWp/day]
    # ==============================================================================
    print('\nGenerating stacked bar chart...')

    x      = np.arange(12)
    width  = 0.55

    # Replace NaN with 0 only for bar rendering; sparse months have NaN in monthly
    # so autos/losses will be 0 and no label is drawn (guarded below).
    autos  = monthly['auto_solar_mean'].fillna(0).values
    losses = monthly['loss_auto_mean'].fillna(0).values

    fig1, ax = plt.subplots(figsize=(14, 6))

    ax.bar(
        x, autos, width,
        label='Solar Autonomy (achieved)',
        color='#F4C518',
        edgecolor='white', linewidth=0.5, zorder=3,
    )

    ax.bar(
        x, losses, width,
        label='Solar Autonomy Losses (shading)',
        color='#E05C5C',
        edgecolor='white', linewidth=0.5, zorder=3,
        bottom=autos,
    )

    # Value labels: autonomy inside bottom segment, losses inside top, total above.
    # Sparse months (not in VALID_MONTHS) are silently skipped — no label drawn.
    for i, (a, l) in enumerate(zip(autos, losses)):
        m = i + 1   # calendar month number
        if m not in VALID_MONTHS:
            continue
        if a > 0.8:
            ax.text(x[i], a / 2, f'{a:.1f}',
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if l > 0.8:
            ax.text(x[i], a + l / 2, f'{l:.1f}',
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        total = a + l
        if total > 0.5:
            ax.text(x[i], total + 0.2, f'{total:.1f}',
                    ha='center', va='bottom',
                    fontsize=7.5, color='#333333', fontweight='bold')

    # Axes, labels, grid
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES, fontsize=11)
    ax.set_xlabel('Month', fontsize=12, labelpad=8)
    ax.set_ylabel('km / kWp / day', fontsize=12, labelpad=8)
    ax.set_ylim(0, (autos + losses).max() * 1.18 + 1)
    ax.set_title(
        'Daily Solar Autonomy and Solar Autonomy Losses per Month\n'
        f'(All sensors pooled — CVE = {CVE_KWH_PER_100KM} kWh/100 km  |  '
        'Stacked: bottom = achieved autonomy, top = shading losses)',
        fontsize=12, fontweight='bold', pad=14,
    )
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # Annual mean reference lines — added before the single legend() call so that
    # all handles (bars + reference lines) appear in the same legend entry list.
    annual_total = annual_auto + annual_loss
    ax.axhline(annual_auto, color='#c8a000', linewidth=1.3, linestyle='--',
               label=f'Annual mean autonomy ({annual_auto:.1f} km/kWp/day)')
    ax.axhline(annual_total, color='#333333', linewidth=1.0, linestyle=':',
               label=f'Annual mean clear-sky potential ({annual_total:.1f} km/kWp/day)')

    # Single legend call — includes both bar labels and reference-line labels
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.88)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE}')
    _show_or_close(fig1)

    # ==============================================================================
    # 7.  FIGURE — MONTHLY BOX PLOT OF DAILY SOLAR AUTONOMY
    # ==============================================================================
    print('\nGenerating monthly solar autonomy box plot...')

    fig2, ax2 = plt.subplots(figsize=(14, 6))

    # One array of daily autonomy values per calendar month.
    # Sparse months receive an empty array so no box is drawn.
    monthly_data_auto = [
        daily.loc[daily['month'] == m, 'auto_solar_km'].dropna().values
        if m in VALID_MONTHS else np.array([])
        for m in range(1, 13)
    ]

    ax2.boxplot(
        monthly_data_auto,
        positions=range(1, 13),
        widths=0.55,
        whis=1.5,
        patch_artist=True,
        medianprops=dict(color='#333333', linewidth=1.8),
        boxprops=dict(facecolor='#F4C518', alpha=0.75, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, linestyle='--'),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.35, color='#888888',
                        markeredgewidth=0),
    )

    # Monthly mean connected by a line — break the line at sparse months (NaN)
    mean_auto_valid = [
        daily.loc[daily['month'] == m, 'auto_solar_km'].mean()
        if m in VALID_MONTHS else np.nan
        for m in range(1, 13)
    ]
    ax2.plot(
        range(1, 13), mean_auto_valid,
        color='#1565C0', linewidth=2.0, marker='o', markersize=6,
        zorder=5, label='Monthly mean',
    )

    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_NAMES, fontsize=11)
    ax2.set_xlabel('Month', fontsize=12, labelpad=8)
    ax2.set_ylabel('Solar Autonomy [km / kWp / day]', fontsize=12, labelpad=8)
    ax2.set_title(
        'Monthly Distribution of Daily Solar Autonomy — Box Plot\n'
        f'(All sensors pooled — CVE = {CVE_KWH_PER_100KM} kWh/100 km)',
        fontsize=13, fontweight='bold', pad=14,
    )
    ax2.legend(fontsize=11, loc='upper left', framealpha=0.85)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_BP, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_BP}')
    _show_or_close(fig2)

    # ==============================================================================
    # 8.  FIGURE — MONTHLY BOX PLOTS OF IRRADIANCE LOSSES
    #     Left panel  : absolute losses  [kWh / kWp / day]
    #     Right panel : relative losses  [%]
    # ==============================================================================
    print('\nGenerating monthly irradiance loss box plots...')

    # Daily irradiance losses aggregated from per-minute data
    daily_losses = (
        data
        .groupby(['sensor', 'date', 'month'], as_index=False)
        .agg(
            loss_irr_kWh = ('loss_irr_Wh',  lambda s: s.sum() / 1000.0),   # Wh → kWh
            E_solar_kWh  = ('E_solar_kWh',  'sum'),
        )
    )

    # Relative daily loss [%]: loss / (loss + captured irradiance)
    daily_losses['loss_rel_pct'] = (
        daily_losses['loss_irr_kWh']
        / (daily_losses['loss_irr_kWh'] + daily_losses['E_solar_kWh'])
        * 100.0
    )
    daily_losses['loss_rel_pct'] = daily_losses['loss_rel_pct'].replace(
        [np.inf, -np.inf], np.nan
    )

    # Per-month lists of values — sparse months receive empty arrays so that
    # matplotlib renders no box, whisker, or flier for those positions.
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

    mean_loss_abs = [np.nanmean(v) if (len(v) >= MIN_SENSOR_DAYS and m in VALID_MONTHS)
                     else np.nan
                     for m, v in zip(range(1, 13), monthly_loss_abs)]
    mean_loss_rel = [np.nanmean(v) if (len(v) >= MIN_SENSOR_DAYS and m in VALID_MONTHS)
                     else np.nan
                     for m, v in zip(range(1, 13), monthly_loss_rel)]


    fig3, (ax3l, ax3r) = plt.subplots(1, 2, figsize=(16, 6))

    draw_loss_boxplot(
        ax3l, monthly_loss_abs, mean_loss_abs,
        ylabel='Irradiance Losses [kWh / kWp / day]',
        title='Monthly Absolute Irradiance Losses',
        line_color='#F4A800',
    )
    draw_loss_boxplot(
        ax3r, monthly_loss_rel, mean_loss_rel,
        ylabel='Irradiance Losses [%]',
        title='Monthly Relative Irradiance Losses',
        line_color='#2E7D32',
        yformat_pct=True,
    )

    fig3.suptitle(
        'Monthly Distribution of Irradiance Losses due to Shading — Box Plots\n'
        '(All sensors pooled  |  left: absolute losses,  right: relative losses)',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_LOSS, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_LOSS}')
    _show_or_close(fig3)

    # ==============================================================================
    # 9.  FIGURE — HOURLY SHADING LOSSES BY LAND-USE GROUP
    #     Left panel  : Urban measurements
    #     Right panel : Rural measurements
    #     Each panel combines absolute losses [Wh/Wp] on the left axis
    #     and relative losses [%] on the right axis.
    # ==============================================================================
    print('\nGenerating hourly shading losses by land-use group...')

    # Per-minute relative loss [%]: loss / (loss + captured irradiance)
    data['loss_rel_min_pct'] = (
        data['loss_irr_Wh']
        / (data['loss_irr_Wh'] + data['E_solar_kWh'] * 1000.0)
        * 100.0
    )
    data['loss_rel_min_pct'] = data['loss_rel_min_pct'].replace([np.inf, -np.inf], np.nan)

    # CLC group assignment: land_use = grid_code → Urban / Rural
    data['clc_group'] = data['land_use_clcv'].map(CLC_URBAN_RURAL).fillna('Unknown')

    urban_mask = data['clc_group'] == 'Urban'
    rural_mask = data['clc_group'] == 'Rural'

    urban_abs  = build_hourly_lists(data, urban_mask, 'loss_irr_Wh')
    urban_rel  = build_hourly_lists(data, urban_mask, 'loss_rel_min_pct')
    rural_abs  = build_hourly_lists(data, rural_mask, 'loss_irr_Wh')
    rural_rel  = build_hourly_lists(data, rural_mask, 'loss_rel_min_pct')

    mean_urban_abs = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in urban_abs]
    mean_urban_rel = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in urban_rel]
    mean_rural_abs = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in rural_abs]
    mean_rural_rel = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in rural_rel]


    fig4, (ax4l, ax4r) = plt.subplots(1, 2, figsize=(16, 6))

    draw_hourly_boxplot(
        ax4l, urban_abs, urban_rel, mean_urban_abs, mean_urban_rel,
        title='Shading Losses — Urban',
    )
    draw_hourly_boxplot(
        ax4r, rural_abs, rural_rel, mean_rural_abs, mean_rural_rel,
        title='Shading Losses — Rural',
    )

    fig4.suptitle(
        'Hourly Distribution of Shading Losses — Urban vs Rural\n'
        '(All sensors pooled  |  boxes: absolute [Wh/Wp],  green line: relative [%])',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_HOURLY, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_HOURLY}')
    _show_or_close(fig4)

    # ==============================================================================
    # 10.  FIGURE — HOURLY SHADING LOSSES BY MOVEMENT STATUS
    #      Left panel  : Parked measurements
    #      Right panel : Moving measurements
    #      Same layout as the Urban / Rural figure above.
    #
    #      Movement classification (consistent with vehicle_movement_analysis.py):
    #        Haversine distance between consecutive 1-minute GPS points:
    #          < THRESHOLD_M → Parked
    #          ≥ THRESHOLD_M → Moving
    # ==============================================================================
    print('\nGenerating hourly shading losses by movement status...')


    # Compute movement status per sensor from the raw CSV files,
    # preserving the full time-ordered sequence before any filtering.
    moving_status_frames: list[pd.DataFrame] = []

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
        df_s = df_s.dropna(subset=['datetime', 'lat', 'lon'])

        for col in ['lat', 'lon']:
            df_s[col] = pd.to_numeric(df_s[col], errors='coerce')
        df_s = df_s.dropna(subset=['lat', 'lon']).sort_values('datetime').reset_index(drop=True)

        # Haversine distance from row i to row i + 1, assigned to row i.
        # The last row has no successor and is dropped.
        # timestamps is passed to zero out overnight / restart boundaries.
        dist = haversine_vectorized(
            df_s['lat'].values[:-1], df_s['lon'].values[:-1],
            df_s['lat'].values[1:],  df_s['lon'].values[1:],
            timestamps=df_s['datetime'].values,
        )
        df_s = df_s.iloc[:-1].copy()
        df_s['moving_status'] = np.where(dist >= THRESHOLD_M, 'Moving', 'Parked')
        df_s['sensor']        = sensor

        moving_status_frames.append(df_s[['datetime', 'sensor', 'moving_status']])

    move_df = pd.concat(moving_status_frames, ignore_index=True)

    # Merge movement status into the filtered analysis dataset
    data = data.merge(move_df, on=['datetime', 'sensor'], how='left')
    data_move = data.dropna(subset=['moving_status']).copy()

    # Hourly loss arrays per movement status
    parked_mask = data_move['moving_status'] == 'Parked'
    moving_mask = data_move['moving_status'] == 'Moving'

    parked_abs = build_hourly_lists(data_move, parked_mask, 'loss_irr_Wh')
    parked_rel = build_hourly_lists(data_move, parked_mask, 'loss_rel_min_pct')
    moving_abs = build_hourly_lists(data_move, moving_mask, 'loss_irr_Wh')
    moving_rel = build_hourly_lists(data_move, moving_mask, 'loss_rel_min_pct')

    mean_parked_abs = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in parked_abs]
    mean_parked_rel = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in parked_rel]
    mean_moving_abs = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in moving_abs]
    mean_moving_rel = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in moving_rel]

    fig5, (ax5l, ax5r) = plt.subplots(1, 2, figsize=(16, 6))

    draw_hourly_boxplot(
        ax5l, parked_abs, parked_rel, mean_parked_abs, mean_parked_rel,
        title='Shading Losses — Parked',
    )
    draw_hourly_boxplot(
        ax5r, moving_abs, moving_rel, mean_moving_abs, mean_moving_rel,
        title='Shading Losses — Moving',
    )

    fig5.suptitle(
        'Hourly Distribution of Shading Losses — Parked vs Moving\n'
        f'(All sensors pooled  |  threshold: {THRESHOLD_M:.0f} m  |  '
        'boxes: absolute [Wh/Wp],  green line: relative [%])',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MOVE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MOVE}')
    _show_or_close(fig5)

    # ==============================================================================
    # 11.  FIGURE — MONTHLY SHADING LOSSES BY LAND-USE GROUP
    #      Same logic as the hourly land-use figure, aggregated by month.
    #      Left panel  : Urban measurements
    #      Right panel : Rural measurements
    # ==============================================================================
    print('\nGenerating monthly shading losses by land-use group...')


    # Recomputed against the current `data` (not reused from Figure 4): the
    # movement merge in Section 10 reassigns `data`, so a mask captured
    # beforehand would no longer align with its index.
    urban_mask_cur = data['clc_group'] == 'Urban'
    rural_mask_cur = data['clc_group'] == 'Rural'
    urban_abs_m = build_monthly_lists(data, VALID_MONTHS, urban_mask_cur, 'loss_irr_Wh')
    urban_rel_m = build_monthly_lists(data, VALID_MONTHS, urban_mask_cur, 'loss_rel_min_pct')
    rural_abs_m = build_monthly_lists(data, VALID_MONTHS, rural_mask_cur, 'loss_irr_Wh')
    rural_rel_m = build_monthly_lists(data, VALID_MONTHS, rural_mask_cur, 'loss_rel_min_pct')

    mean_urban_abs_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                        for v in urban_abs_m]
    mean_urban_rel_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                        for v in urban_rel_m]
    mean_rural_abs_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                        for v in rural_abs_m]
    mean_rural_rel_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                        for v in rural_rel_m]


    fig6, (ax6l, ax6r) = plt.subplots(1, 2, figsize=(16, 6))

    draw_monthly_boxplot(
        ax6l, urban_abs_m, urban_rel_m, mean_urban_abs_m, mean_urban_rel_m,
        title='Shading Losses — Urban',
    )
    draw_monthly_boxplot(
        ax6r, rural_abs_m, rural_rel_m, mean_rural_abs_m, mean_rural_rel_m,
        title='Shading Losses — Rural',
    )

    fig6.suptitle(
        'Monthly Distribution of Shading Losses — Urban vs Rural\n'
        '(All sensors pooled  |  boxes: absolute [Wh/Wp],  green line: relative [%])',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MONTHLY_LANDUSE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MONTHLY_LANDUSE}')
    _show_or_close(fig6)

    # ==============================================================================
    # 12.  FIGURE — MONTHLY SHADING LOSSES BY MOVEMENT STATUS
    #      Same logic as the hourly movement figure, aggregated by month.
    #      Left panel  : Parked measurements
    #      Right panel : Moving measurements
    # ==============================================================================
    print('\nGenerating monthly shading losses by movement status...')


    parked_abs_m = build_monthly_lists(data_move, VALID_MONTHS, parked_mask, 'loss_irr_Wh')
    parked_rel_m = build_monthly_lists(data_move, VALID_MONTHS, parked_mask, 'loss_rel_min_pct')
    moving_abs_m = build_monthly_lists(data_move, VALID_MONTHS, moving_mask, 'loss_irr_Wh')
    moving_rel_m = build_monthly_lists(data_move, VALID_MONTHS, moving_mask, 'loss_rel_min_pct')

    mean_parked_abs_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                         for v in parked_abs_m]
    mean_parked_rel_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                         for v in parked_rel_m]
    mean_moving_abs_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                         for v in moving_abs_m]
    mean_moving_rel_m = [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan
                         for v in moving_rel_m]

    fig7, (ax7l, ax7r) = plt.subplots(1, 2, figsize=(16, 6))

    draw_monthly_boxplot(
        ax7l, parked_abs_m, parked_rel_m, mean_parked_abs_m, mean_parked_rel_m,
        title='Shading Losses — Parked',
    )
    draw_monthly_boxplot(
        ax7r, moving_abs_m, moving_rel_m, mean_moving_abs_m, mean_moving_rel_m,
        title='Shading Losses — Moving',
    )

    fig7.suptitle(
        'Monthly Distribution of Shading Losses — Parked vs Moving\n'
        f'(All sensors pooled  |  threshold: {THRESHOLD_M:.0f} m  |  '
        'boxes: absolute [Wh/Wp],  green line: relative [%])',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MONTHLY_MOVE, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MONTHLY_MOVE}')
    _show_or_close(fig7)

    # ==============================================================================
    # 13.  FIGURE — HOURLY SHADING LOSSES BY COMBINED CONDITION  (Figure 8)
    #      Four panels arranged in a 2×2 grid:
    #        Top-left     : Urban  + Moving
    #        Top-right    : Urban  + Parked
    #        Bottom-left  : Rural  + Moving
    #        Bottom-right : Rural  + Parked
    #
    #      Each panel uses the same twin-axis layout as Figures 4 and 5:
    #        Left y-axis  : absolute losses [Wh/Wp] — box plots + mean line
    #        Right y-axis : relative losses [%]     — green mean line only
    #
    #      Box colour follows the project-wide palette:
    #        Urban  → #C0392B (coral red)     Moving → #1565C0 (dark blue)
    #        Rural  → #2E7D32 (forest green)  Parked → #E65100 (deep orange)
    #      Each panel uses the colour of its land-use class (dominant visual axis).
    # ==============================================================================
    print('\nGenerating hourly shading losses by combined condition (Fig 8)...')

    # The four combined conditions and their display properties.
    # All masks operate on data_move which already carries clc_group + moving_status.
    COMBINED_CONDITIONS: list[dict] = [
        {
            'label': 'Urban + Moving',
            'mask':  (data_move['clc_group'] == 'Urban') & (data_move['moving_status'] == 'Moving'),
        },
        {
            'label': 'Urban + Parked',
            'mask':  (data_move['clc_group'] == 'Urban') & (data_move['moving_status'] == 'Parked'),
        },
        {
            'label': 'Rural + Moving',
            'mask':  (data_move['clc_group'] == 'Rural') & (data_move['moving_status'] == 'Moving'),
        },
        {
            'label': 'Rural + Parked',
            'mask':  (data_move['clc_group'] == 'Rural') & (data_move['moving_status'] == 'Parked'),
        },
    ]


    # Pre-compute arrays and means for all four conditions
    comb_hourly: list[dict] = []
    for cond in COMBINED_CONDITIONS:
        abs_h = build_hourly_lists(data_move, cond['mask'], 'loss_irr_Wh')
        rel_h = build_hourly_lists(data_move, cond['mask'], 'loss_rel_min_pct')
        comb_hourly.append({
            **cond,
            'abs': abs_h,
            'rel': rel_h,
            'mean_abs': [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in abs_h],
            'mean_rel': [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in rel_h],
        })

    # 2×2 figure: rows = land-use (Urban / Rural), columns = status (Moving / Parked)
    fig8, axes8 = plt.subplots(2, 2, figsize=(20, 12))
    hour_positions = np.arange(len(HOURS_PLOT)) + 1
    hour_labels = [f'{h:02d}:00' for h in HOURS_PLOT]

    for ax, cd in zip(axes8.flat, comb_hourly):
        draw_boxplot_combined(
            ax, hour_positions, hour_labels,
            'Hour of day (local time — Europe/Lisbon)',
            cd['abs'], cd['rel'], cd['mean_abs'], cd['mean_rel'],
            title=f"Shading Losses — {cd['label']}", rotate_xticks=True,
        )

    fig8.suptitle(
        'Hourly Distribution of Shading Losses — Urban/Rural × Moving/Parked\n'
        f'(All sensors pooled  |  threshold: {THRESHOLD_M:.0f} m  |  '
        'boxes: absolute [Wh/Wp],  green line: relative [%])',
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig8.text(
        0.5, -0.01,
        f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}',
        ha='center', fontsize=8, color='#888888', style='italic',
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_HOURLY_COMB, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_HOURLY_COMB}')
    _show_or_close(fig8)

    # ==============================================================================
    # 14.  FIGURE — MONTHLY SHADING LOSSES BY COMBINED CONDITION  (Figure 9)
    #      Same four conditions as Figure 8, aggregated by calendar month.
    #      Each panel uses the same twin-axis layout as Figures 6 and 7:
    #        Left y-axis  : absolute losses [Wh/Wp] — box plots + mean line
    #        Right y-axis : relative losses [%]     — green mean line only
    # ==============================================================================
    print('\nGenerating monthly shading losses by combined condition (Fig 9)...')


    # Pre-compute arrays and means for all four conditions
    comb_monthly: list[dict] = []
    for cond in COMBINED_CONDITIONS:
        abs_m = build_monthly_lists(data_move, VALID_MONTHS, cond['mask'], 'loss_irr_Wh')
        rel_m = build_monthly_lists(data_move, VALID_MONTHS, cond['mask'], 'loss_rel_min_pct')
        comb_monthly.append({
            **cond,
            'abs': abs_m,
            'rel': rel_m,
            'mean_abs': [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in abs_m],
            'mean_rel': [np.nanmean(v) if len(v) >= MIN_MINUTES else np.nan for v in rel_m],
        })

    # 2×2 figure: rows = land-use (Urban / Rural), columns = status (Moving / Parked)
    fig9, axes9 = plt.subplots(2, 2, figsize=(20, 12))
    month_positions = np.arange(1, 13)

    for ax, cd in zip(axes9.flat, comb_monthly):
        draw_boxplot_combined(
            ax, month_positions, MONTH_NAMES, 'Month',
            cd['abs'], cd['rel'], cd['mean_abs'], cd['mean_rel'],
            title=f"Shading Losses — {cd['label']}",
        )

    fig9.suptitle(
        'Monthly Distribution of Shading Losses — Urban/Rural × Moving/Parked\n'
        f'(All sensors pooled  |  threshold: {THRESHOLD_M:.0f} m  |  '
        'boxes: absolute [Wh/Wp],  green line: relative [%])',
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig9.text(
        0.5, -0.01,
        f'[{CAMPAIGN}/{_VARIANT_TAG}]  —  MIN_MINUTES = {MIN_MINUTES}  |  MIN_SENSOR_DAYS = {MIN_SENSOR_DAYS}',
        ha='center', fontsize=8, color='#888888', style='italic',
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MONTHLY_COMB, dpi=300, bbox_inches='tight')
    print(f'Figure saved : {OUTPUT_IMAGE_MONTHLY_COMB}')
    _show_or_close(fig9)

    # ==============================================================================
    # 15.  EXPORT MONTHLY SUMMARY CSV
    # ==============================================================================
    monthly_out = monthly.copy()
    monthly_out.insert(0, 'month_name',
                       monthly_out['month'].apply(lambda m: MONTH_NAMES[m - 1]))
    monthly_out.columns = [
        'month_name', 'month',
        'auto_solar_mean_km_kWp_day',
        'loss_auto_mean_km_kWp_day',
        'n_sensor_days',
    ]

    # Flag under-sampled months so downstream users are aware
    monthly_out['under_sampled'] = monthly_out['n_sensor_days'].apply(
        lambda n: True if (pd.isna(n) or n < MIN_SENSOR_DAYS) else False
    )

    # Append an annual mean summary row
    annual_row = pd.DataFrame([{
        'month_name':                 'Annual mean',
        'month':                       0,
        'auto_solar_mean_km_kWp_day': round(annual_auto, 3),
        'loss_auto_mean_km_kWp_day':  round(annual_loss, 3),
        'n_sensor_days':               len(daily),
        'under_sampled':               False,
    }])
    monthly_out = pd.concat([monthly_out, annual_row], ignore_index=True)

    monthly_out.to_csv(OUTPUT_CSV, index=False)
    print(f'Monthly summary CSV saved : {OUTPUT_CSV}')

    # ==============================================================================
    # 16.  EXPORT — statistics underlying every figure
    # ==============================================================================
    # One CSV file, one row per (figure, group/panel, x-bin).
    # Columns:
    #   figure_id     : short identifier matching the output PNG base name
    #   panel         : sub-panel label (e.g. 'Urban', 'Parked') or 'All'
    #   x_label       : month name (e.g. 'Jan') or hour string (e.g. '08:00')
    #   x_value       : integer month (1–12) or hour (6–20) for sorting
    #   variable      : what is measured (e.g. 'auto_solar_km', 'loss_irr_Wh')
    #   n_obs         : number of observations (sensor-days or minutes depending on fig)
    #   mean          : arithmetic mean
    #   median        : 50th percentile
    #   std           : standard deviation (ddof=1)
    #   q25           : 25th percentile
    #   q75           : 75th percentile
    #   under_sampled : True when month is flagged by MIN_SENSOR_DAYS threshold
    #   masked_on_fig : True when the cell is hidden on the figure
    # ─────────────────────────────────────────────────────────────────────────────
    # All numeric statistics are rounded to 6 decimal places.
    # Rows masked on the figure have their real n_obs but statistics set to NaN.
    # ==============================================================================

    stats_rows: list[dict] = []


    # ── Figure 1 & 2 — Monthly solar autonomy (sensor-day level) ─────────────────
    fig_id_bar = 'sal_monthly_barchart'
    fig_id_bp  = 'sal_monthly_boxplot_autonomy'

    for m in range(1, 13):
        mname  = MONTH_NAMES[m - 1]
        us     = m not in VALID_MONTHS
        vals   = daily.loc[daily['month'] == m, 'auto_solar_km'].dropna().values
        loss_v = daily.loc[daily['month'] == m, 'loss_auto_km'].dropna().values

        stats_rows.append(_row(fig_id_bar, 'All', mname, m,
                               'auto_solar_km_day', vals, us, MIN_SENSOR_DAYS))
        stats_rows.append(_row(fig_id_bar, 'All', mname, m,
                               'loss_auto_km_day', loss_v, us, MIN_SENSOR_DAYS))
        stats_rows.append(_row(fig_id_bp, 'All', mname, m,
                               'auto_solar_km_day', vals, us, MIN_SENSOR_DAYS))


    # ── Figure 3 — Monthly irradiance losses (sensor-day level) ──────────────────
    fig_id_loss = 'sal_monthly_boxplot_losses'

    for m in range(1, 13):
        mname = MONTH_NAMES[m - 1]
        us    = m not in VALID_MONTHS
        abs_v = daily_losses.loc[daily_losses['month'] == m,
                                  'loss_irr_kWh'].dropna().values
        rel_v = daily_losses.loc[daily_losses['month'] == m,
                                  'loss_rel_pct'].dropna().values

        stats_rows.append(_row(fig_id_loss, 'All', mname, m,
                               'loss_irr_kWh_day', abs_v, us, MIN_SENSOR_DAYS))
        stats_rows.append(_row(fig_id_loss, 'All', mname, m,
                               'loss_rel_pct_day', rel_v, us, MIN_SENSOR_DAYS))


    # ── Figure 4 — Hourly shading losses by land-use group (minute level) ────────
    fig_id_h_lu = 'sal_hourly_losses_landuse'

    for i, h in enumerate(HOURS_PLOT):
        hlabel = f'{h:02d}:00'
        for panel, abs_arr, rel_arr in [
            ('Urban', urban_abs[i], urban_rel[i]),
            ('Rural', rural_abs[i], rural_rel[i]),
        ]:
            stats_rows.append(_row(fig_id_h_lu, panel, hlabel, h,
                                   'loss_irr_Wh_min', abs_arr,
                                   False, MIN_MINUTES))
            stats_rows.append(_row(fig_id_h_lu, panel, hlabel, h,
                                   'loss_rel_pct_min', rel_arr,
                                   False, MIN_MINUTES))


    # ── Figure 5 — Hourly shading losses by movement status (minute level) ───────
    fig_id_h_mv = 'sal_hourly_losses_movement'

    for i, h in enumerate(HOURS_PLOT):
        hlabel = f'{h:02d}:00'
        for panel, abs_arr, rel_arr in [
            ('Parked', parked_abs[i], parked_rel[i]),
            ('Moving', moving_abs[i], moving_rel[i]),
        ]:
            stats_rows.append(_row(fig_id_h_mv, panel, hlabel, h,
                                   'loss_irr_Wh_min', abs_arr,
                                   False, MIN_MINUTES))
            stats_rows.append(_row(fig_id_h_mv, panel, hlabel, h,
                                   'loss_rel_pct_min', rel_arr,
                                   False, MIN_MINUTES))


    # ── Figure 6 — Monthly shading losses by land-use group (minute level) ───────
    fig_id_m_lu = 'sal_monthly_losses_landuse'

    for m in range(1, 13):
        mname = MONTH_NAMES[m - 1]
        us    = m not in VALID_MONTHS
        for panel, abs_arr, rel_arr in [
            ('Urban', urban_abs_m[m - 1], urban_rel_m[m - 1]),
            ('Rural', rural_abs_m[m - 1], rural_rel_m[m - 1]),
        ]:
            stats_rows.append(_row(fig_id_m_lu, panel, mname, m,
                                   'loss_irr_Wh_min', abs_arr, us, MIN_MINUTES))
            stats_rows.append(_row(fig_id_m_lu, panel, mname, m,
                                   'loss_rel_pct_min', rel_arr, us, MIN_MINUTES))


    # ── Figure 7 — Monthly shading losses by movement status (minute level) ──────
    fig_id_m_mv = 'sal_monthly_losses_movement'

    for m in range(1, 13):
        mname = MONTH_NAMES[m - 1]
        us    = m not in VALID_MONTHS
        for panel, abs_arr, rel_arr in [
            ('Parked', parked_abs_m[m - 1], parked_rel_m[m - 1]),
            ('Moving', moving_abs_m[m - 1], moving_rel_m[m - 1]),
        ]:
            stats_rows.append(_row(fig_id_m_mv, panel, mname, m,
                                   'loss_irr_Wh_min', abs_arr, us, MIN_MINUTES))
            stats_rows.append(_row(fig_id_m_mv, panel, mname, m,
                                   'loss_rel_pct_min', rel_arr, us, MIN_MINUTES))


    # ── Figure 8 — Hourly shading losses by combined condition (minute level) ────
    fig_id_h_comb = 'sal_hourly_losses_combined'

    for cd in comb_hourly:
        for i, h in enumerate(HOURS_PLOT):
            hlabel = f'{h:02d}:00'
            stats_rows.append(_row(fig_id_h_comb, cd['label'], hlabel, h,
                                   'loss_irr_Wh_min', cd['abs'][i],
                                   False, MIN_MINUTES))
            stats_rows.append(_row(fig_id_h_comb, cd['label'], hlabel, h,
                                   'loss_rel_pct_min', cd['rel'][i],
                                   False, MIN_MINUTES))


    # ── Figure 9 — Monthly shading losses by combined condition (minute level) ───
    fig_id_m_comb = 'sal_monthly_losses_combined'

    for cd in comb_monthly:
        for m in range(1, 13):
            mname = MONTH_NAMES[m - 1]
            us    = m not in VALID_MONTHS
            stats_rows.append(_row(fig_id_m_comb, cd['label'], mname, m,
                                   'loss_irr_Wh_min', cd['abs'][m - 1],
                                   us, MIN_MINUTES))
            stats_rows.append(_row(fig_id_m_comb, cd['label'], mname, m,
                                   'loss_rel_pct_min', cd['rel'][m - 1],
                                   us, MIN_MINUTES))


    # ── Assemble and save ─────────────────────────────────────────────────────────
    df_stats = pd.DataFrame(stats_rows)

    df_stats = df_stats[[
        'figure_id', 'panel', 'x_label', 'x_value', 'variable',
        'n_obs', 'mean', 'median', 'std', 'q25', 'q75',
        'under_sampled', 'masked_on_fig',
    ]]

    df_stats.to_csv(OUTPUT_STATS, index=False)
    print(f'Figure statistics saved : {OUTPUT_STATS}')
    print(f'  → {len(df_stats):,} rows  |  '
          f'{df_stats["figure_id"].nunique()} figures  |  '
          f'{df_stats["masked_on_fig"].sum()} rows masked on figures')

    # ==========================================================================
    # DONE
    # ==========================================================================
    print('\n' + '=' * 70)
    print('Analysis complete.')
    print(f'  CAMPAIGN         : {CAMPAIGN}')
    print(f'  Data variant     : {_VARIANT_TAG}')
    print(f'  Output directory : {OUTPUT_DIR}')
    print('=' * 70)


# ==============================================================================
if __name__ == "__main__":
    main()