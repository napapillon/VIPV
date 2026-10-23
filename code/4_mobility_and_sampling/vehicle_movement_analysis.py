"""
================================================================================
Vehicle Movement Analysis — Parked vs. Moving Classification
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Classifies each GPS transition between consecutive 1-minute measurements as
"parked" or "moving" from the Haversine great-circle distance:

  - distance <  THRESHOLD_M  →  parked
  - distance >= THRESHOLD_M  →  moving

Transitions with a time gap above MAX_GAP_MINUTES (overnight stop, sensor
restart) are not classified. Only transitions
between HOUR_START and HOUR_END (local time) are counted.

Results are aggregated (A) per hour of day — daily activity pattern — and
(B) per calendar month — seasonal evolution — and plotted as stacked bars,
per sensor and for all sensors combined. A threshold sensitivity analysis
(SENSITIVITY_THRESHOLDS_M) is added when RUN_SENSITIVITY_ANALYSIS is True.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             movement_hourly_<sensor>_<CAMPAIGN>_<variant_tag>.png / .csv
             movement_monthly_<sensor>_<CAMPAIGN>_<variant_tag>.png / .csv
             movement_hourly_combined_<CAMPAIGN>_<variant_tag>.png / .csv
             movement_monthly_combined_<CAMPAIGN>_<variant_tag>.png / .csv
             sensitivity_threshold_<CAMPAIGN>_<variant_tag>_summary.csv
             sensitivity_threshold_<CAMPAIGN>_<variant_tag>_pct_parked.png

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 5.1
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
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
# "standard" : 1st Solar Jinie campaign
# "tuktuk"   : Tuk-tuk campaign
CAMPAIGN: str = "standard"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Latitude / longitude column names per sensor ----------------------------
SENSOR_COLS = {
    'J29': {'lat': 'J29 lat', 'lon': 'J29 lon'},
    'J30': {'lat': 'J30 lat', 'lon': 'J30 lon'},
    'J31': {'lat': 'J31 lat', 'lon': 'J31 lon'},
    'J32': {'lat': 'J32 lat', 'lon': 'J32 lon'},
    'J33': {'lat': 'J33 lat', 'lon': 'J33 lon'},
}

DATE_COL = 'Human readable date (Europe/London)'

# --- Time window (inclusive, local Europe/Lisbon time) -----------------------
HOUR_START = 6    # covers 06:00–06:59
HOUR_END   = 20   # covers 20:00–20:59

# --- Movement threshold -------------------------------------------------------
# Transitions < THRESHOLD_M → parked.  30 m matches the Perez report convention.
THRESHOLD_M     = 30
# Transitions with a longer time gap are not classified [min]
MAX_GAP_MINUTES = 2

# --- Minimum transitions per bar ---------------------------------------------
# Bars below this count are shown in grey with "n = X".  Set to 0 to disable.
MIN_TRANSITIONS = 600

LOCAL_TZ   = 'Europe/Lisbon'
R_EARTH_KM = 6371.0   # IAU 2012 mean Earth radius

# --- Output -------------------------------------------------------------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "4_mobility_and_sampling" / "vma_outputs"
OUTPUT_DPI      = 300

# Set to True to display figures interactively in addition to saving them.
SHOW_FIGURES = False

# --- Sensitivity analysis -----------------------------------------------------
RUN_SENSITIVITY_ANALYSIS  = True
SENSITIVITY_THRESHOLDS_M  = [0, 10, 20, 30, 40, 50]
SENSITIVITY_OUTPUT_PREFIX = 'sensitivity_threshold'

# ==============================================================================
# CONSTANTS
# ==============================================================================

MONTH_NAMES = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor names to their CSV paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv

    Parameters
    ----------
    campaign : str
        "standard" or "tuktuk".
    data_dir : Path
        Directory containing the standard-campaign CSV files.
    tuktuk_data_dir : Path
        Directory containing the tuk-tuk per-sensor CSV files.
    suffix : str
        Active variant tag (FILE_SUFFIX or TUKTUK_FILE_SUFFIX).

    Returns
    -------
    dict[str, Path]
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
    """Return the output directory, embedding campaign + variant tag so
    outputs from different campaigns/variants never overwrite each other."""
    return base_dir / campaign / variant_tag


def _build_output_filename(base_name: str, campaign: str, variant_tag: str,
                           extension: str) -> str:
    """Output file name embedding the campaign and the variant tag."""
    return f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolved at startup from the configuration above
_VARIANT_TAG: str = _variant_tag()
SENSOR_FILES: dict = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
)
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def show_figure(fig):
    """Display a figure interactively when SHOW_FIGURES is True."""
    if SHOW_FIGURES:
        plt.show()
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
        Coordinates of the departure point (decimal degrees, WGS-84).
        Typically df[lat_col].values[:-1].
    lat2, lon2 : array-like
        Coordinates of the arrival point (decimal degrees, WGS-84).
        Typically df[lat_col].values[1:].
    R : float
        Earth radius in kilometres (default: 6371.0 — IAU 2012).
    timestamps : array-like of datetime64, optional
        Full timestamp array of length N (i.e. df[DATE_COL].values).
        When provided, pairs that cross a segment boundary are zeroed:
          - time gap > max_gap_minutes  (overnight stop, sensor restart)
          - calendar-date change        (explicit cross-day guard)
        When None, all distances are returned as computed.
    max_gap_minutes : float
        Maximum allowed gap in minutes between consecutive rows.

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


def compute_stats_by_key(df, lat_col, lon_col, key_col, key_range, threshold_m):
    """
    Classify consecutive GPS transitions as parked or moving, grouped by a key.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sensor data sorted by datetime, containing key_col.
    lat_col, lon_col : str
        Names of the latitude and longitude columns.
    key_col : str
        Column to group by ('hour' or 'month').
    key_range : list[int]
    threshold_m : float
        Distance threshold in metres separating parked from moving.

    Returns
    -------
    parked_count, moved_count, total_count : dict[int, int]
    """
    distances = haversine_vectorized(
        df[lat_col].values[:-1], df[lon_col].values[:-1],
        df[lat_col].values[1:],  df[lon_col].values[1:],
        timestamps=df[DATE_COL].values
    )
    keys = df[key_col].values[:-1]

    parked_count = {k: 0 for k in key_range}
    moved_count  = {k: 0 for k in key_range}
    total_count  = {k: 0 for k in key_range}

    for k, d in zip(keys, distances):
        if k not in parked_count:
            continue
        total_count[k] += 1
        if d < threshold_m:
            parked_count[k] += 1
        else:
            moved_count[k]  += 1

    return parked_count, moved_count, total_count


def compute_percentages(parked_count, moved_count, total_count, key_range):
    """
    Convert raw counts to percentage arrays for plotting.

    Parameters
    ----------
    parked_count, moved_count, total_count : dict[int, int]
    key_range : list[int]

    Returns
    -------
    pct_parked, pct_moved : list[float]
    """
    pct_parked, pct_moved = [], []
    for k in key_range:
        total = total_count[k]
        if total == 0:
            pct_parked.append(0.0)
            pct_moved.append(0.0)
        else:
            pct_parked.append(parked_count[k] / total * 100.0)
            pct_moved.append(moved_count[k]   / total * 100.0)
    return pct_parked, pct_moved


def load_sensor_data(sensor_name, file_path, lat_col, lon_col):
    """
    Load and prepare a single sensor CSV file for movement analysis.

    Steps
    -----
    1. Check file existence.
    2. Read CSV.
    3. Parse datetime and convert to local timezone (Europe/Lisbon).
    4. Drop rows with missing datetime or GPS coordinates.
    5. Sort chronologically, then extract hour-of-day and calendar month.


    Parameters
    ----------
    sensor_name : str
    file_path : Path
    lat_col, lon_col : str

    Returns
    -------
    pd.DataFrame or None
    """
    print(f"  Loading {sensor_name} from '{file_path.name}' ...")

    if not file_path.exists():
        print(f"    [WARNING] File not found — skipping {sensor_name}")
        return None

    df = pd.read_csv(file_path, low_memory=False)
    print(f"    → {len(df):,} rows read")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors='coerce')
    df[DATE_COL] = df[DATE_COL].dt.tz_convert(LOCAL_TZ)

    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[DATE_COL, lat_col, lon_col]).reset_index(drop=True)

    if df.empty:
        print(f"    [WARNING] No valid rows remaining — skipping {sensor_name}")
        return None

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df['hour']  = df[DATE_COL].dt.hour
    df['month'] = df[DATE_COL].dt.month

    print(f"    → {len(df):,} valid rows after preparation")
    return df


def plot_stacked_bar(pct_parked, pct_moved, x_labels, x_axis_label,
                     title, output_path, totals=None):
    """
    Create and save a stacked bar chart showing parked / moving distribution.

    Bars whose total number of transitions is below MIN_TRANSITIONS are drawn
    in grey and annotated with the actual sample size ("n = X").

    Parameters
    ----------
    pct_parked, pct_moved : list[float]
    x_labels : list[str]
    x_axis_label : str
    title : str
    output_path : Path
    totals : list[int] or None
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x     = np.arange(len(x_labels))
    width = 0.6

    unreliable_legend_added = False

    for i, (pp, pm) in enumerate(zip(pct_parked, pct_moved)):
        n = totals[i] if totals is not None else MIN_TRANSITIONS + 1

        if MIN_TRANSITIONS > 0 and n < MIN_TRANSITIONS:
            label = 'Insufficient data' if not unreliable_legend_added else '_nolegend_'
            ax.bar(x[i], 100, width, color='#BDBDBD', alpha=0.70,
                   hatch='//', edgecolor='#9E9E9E', label=label)
            ax.text(x[i], 50, f'n = {n}', ha='center', va='center',
                    fontsize=8, color='#424242', fontweight='bold')
            unreliable_legend_added = True
        else:
            ax.bar(x[i], pp, width, color='#2196F3', alpha=0.90,
                   label='Parked' if i == 0 else '_nolegend_')
            ax.bar(x[i], pm, width, color='#FF5722', alpha=0.90,
                   bottom=pp, label='Moving' if i == 0 else '_nolegend_')

            if pp > 4:
                ax.text(x[i], pp / 2,      f'{pp:.1f}%', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
            if pm > 4:
                ax.text(x[i], pp + pm / 2, f'{pm:.1f}%', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')

    handles, labels_leg = ax.get_legend_handles_labels()
    seen   = set()
    unique = [(h, l) for h, l in zip(handles, labels_leg)
              if l not in seen and not seen.add(l) and l != '_nolegend_']

    legend_map = {l: h for h, l in unique}
    if 'Parked' not in legend_map:
        legend_map['Parked'] = mpatches.Patch(color='#2196F3', alpha=0.90, label='Parked')
    if 'Moving' not in legend_map:
        legend_map['Moving'] = mpatches.Patch(color='#FF5722', alpha=0.90, label='Moving')

    ordered_labels = ['Parked', 'Moving']
    if unreliable_legend_added:
        ordered_labels.append('Insufficient data')
    legend_handles = [legend_map.get(l) for l in ordered_labels if l in legend_map]
    ax.legend(handles=legend_handles,
              labels=[l for l in ordered_labels if l in legend_map],
              loc='upper right', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel('Share (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    if MIN_TRANSITIONS > 0:
        ax.text(0.01, 0.01,
                f'Grey bars: fewer than {MIN_TRANSITIONS} transitions (unreliable)',
                transform=ax.transAxes, fontsize=8, color='#757575',
                va='bottom', ha='left', style='italic')

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    print(f"    Chart saved : {output_path.name}")
    show_figure(fig)


def print_console_summary(label, key_labels, key_range,
                           parked_count, moved_count, total_count):
    """
    Print a formatted breakdown table to the console.

    Parameters
    ----------
    label : str
    key_labels : list[str]
    key_range : list[int]
    parked_count, moved_count, total_count : dict[int, int]
    """
    print(f"\n  {'Key':<8}  {'Total':>8}  {'Parked (%)':>11}  {'Moving (%)':>11}")
    print('  ' + '-' * 46)
    for k, lbl in zip(key_range, key_labels):
        tot = total_count[k]
        pp  = parked_count[k] / tot * 100 if tot else 0.0
        pm  = moved_count[k]  / tot * 100 if tot else 0.0
        print(f"  {lbl:<8}  {tot:>8,}  {pp:>10.1f}%  {pm:>10.1f}%")

    total_all  = sum(total_count.values())
    parked_all = sum(parked_count.values())
    moved_all  = sum(moved_count.values())
    pct_p = parked_all / total_all * 100 if total_all else 0.0
    pct_m = moved_all  / total_all * 100 if total_all else 0.0

    print(f"\n  {'=' * 48}")
    print(f"  OVERALL — {label}")
    print(f"  {'=' * 48}")
    print(f"  Total transitions : {total_all:,}")
    print(f"  Parked            : {parked_all:,}  ({pct_p:.1f}%)")
    print(f"  Moving            : {moved_all:,}  ({pct_m:.1f}%)")
    print(f"  {'=' * 48}")


def save_csv_summary(key_col_name, key_labels, key_range,
                     parked_count, moved_count, total_count, output_path):
    """
    Export the parked / moving aggregation table (plus a TOTAL row) to CSV.

    Parameters
    ----------
    key_col_name : str
        Column header for the key, e.g. 'hour' or 'month'.
    key_labels : list[str]
    key_range : list[int]
    parked_count, moved_count, total_count : dict[int, int]
    output_path : Path
    """
    rows = []
    for k, lbl in zip(key_range, key_labels):
        tot = total_count[k]
        pp  = round(parked_count[k] / tot * 100, 1) if tot else 0.0
        pm  = round(moved_count[k]  / tot * 100, 1) if tot else 0.0
        rows.append({key_col_name: lbl, 'total': tot,
                     'parked_pct': pp, 'moving_pct': pm})

    total_all  = sum(total_count.values())
    parked_all = sum(parked_count.values())
    moved_all  = sum(moved_count.values())
    pct_p = round(parked_all / total_all * 100, 1) if total_all else 0.0
    pct_m = round(moved_all  / total_all * 100, 1) if total_all else 0.0
    rows.append({key_col_name: 'TOTAL', 'total': total_all,
                 'parked_pct': pct_p, 'moving_pct': pct_m})

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"    CSV  saved  : {output_path.name}")


def run_threshold_sensitivity_analysis(sensor_windows, hour_range, output_dir):
    """
    Run a threshold sensitivity analysis on the already-loaded sensor data.

    Reclassifies the same cleaned GPS transitions using several distance
    thresholds and exports a CSV summary and a visual comparison plot.

    Parameters
    ----------
    sensor_windows : dict[str, dict]
        {'J30': {'df': df_window, 'lat_col': ..., 'lon_col': ...}, ...}
    hour_range : list[int]
    output_dir : Path

    Returns
    -------
    pd.DataFrame
        One row per sensor and threshold, plus combined rows.
    """
    print("\n[Sensitivity] Running threshold sensitivity analysis ...")

    thresholds = sorted(set(float(t) for t in SENSITIVITY_THRESHOLDS_M))
    results    = []

    for threshold in thresholds:
        combined_parked = 0
        combined_moved  = 0
        combined_total  = 0

        for sensor_name, info in sensor_windows.items():
            df_window = info['df']
            lat_col   = info['lat_col']
            lon_col   = info['lon_col']

            parked_h, moved_h, total_h = compute_stats_by_key(
                df_window, lat_col, lon_col, 'hour', hour_range, threshold
            )

            parked_total = int(sum(parked_h.values()))
            moved_total  = int(sum(moved_h.values()))
            total        = int(sum(total_h.values()))
            pct_parked   = (parked_total / total * 100) if total else np.nan
            pct_moved    = (moved_total  / total * 100) if total else np.nan

            results.append({
                'sensor':             sensor_name,
                'threshold_m':        threshold,
                'parked_count':       parked_total,
                'moving_count':       moved_total,
                'total_transitions':  total,
                'pct_parked':         pct_parked,
                'pct_moving':         pct_moved,
            })

            combined_parked += parked_total
            combined_moved  += moved_total
            combined_total  += total

        if combined_total:
            results.append({
                'sensor':             'COMBINED',
                'threshold_m':        threshold,
                'parked_count':       combined_parked,
                'moving_count':       combined_moved,
                'total_transitions':  combined_total,
                'pct_parked':         combined_parked / combined_total * 100,
                'pct_moving':         combined_moved  / combined_total * 100,
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("  [WARNING] No sensitivity results generated.")
        return results_df

    csv_path = output_dir / f'{SENSITIVITY_OUTPUT_PREFIX}_{CAMPAIGN}_{_VARIANT_TAG}_summary.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"  -> Sensitivity CSV saved: {csv_path.name}")

    fig, ax = plt.subplots(figsize=(10, 6))

    for sensor_name in sorted(results_df['sensor'].unique()):
        sensor_df = results_df[results_df['sensor'] == sensor_name].sort_values('threshold_m')
        if sensor_name == 'COMBINED':
            ax.plot(sensor_df['threshold_m'], sensor_df['pct_parked'],
                    marker='o', linewidth=3, label='COMBINED')
        else:
            ax.plot(sensor_df['threshold_m'], sensor_df['pct_parked'],
                    marker='o', linewidth=1.5, alpha=0.75, label=sensor_name)

    ax.set_xlabel('Movement threshold (metres)')
    ax.set_ylabel('Parked transitions (%)')
    ax.set_title(
        'Sensitivity analysis — impact of movement threshold on parked classification\n'
        f'Hour window: {HOUR_START:02d}:00–{HOUR_END:02d}:59, local time ({LOCAL_TZ})'
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.grid(True, alpha=0.3)
    ax.legend(title='Sensor')
    fig.tight_layout()

    png_path = output_dir / f'{SENSITIVITY_OUTPUT_PREFIX}_{CAMPAIGN}_{_VARIANT_TAG}_pct_parked.png'
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches='tight')
    print(f"  -> Sensitivity plot saved: {png_path.name}")
    show_figure(fig)

    combined = results_df[results_df['sensor'] == 'COMBINED'].sort_values('threshold_m')
    if not combined.empty:
        first = combined.iloc[0]
        last  = combined.iloc[-1]
        delta = last['pct_parked'] - first['pct_parked']
        print(
            f"  -> Combined parked share changes from {first['pct_parked']:.1f}% "
            f"at {first['threshold_m']:.0f} m to {last['pct_parked']:.1f}% "
            f"at {last['threshold_m']:.0f} m (delta: {delta:+.1f} pp)."
        )

    return results_df


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the sensor files, classify the transitions and write all outputs."""
    print('=' * 70)
    print('VEHICLE MOVEMENT ANALYSIS — Parked vs. Moving Classification')
    print('=' * 70)
    print(f"  CAMPAIGN      : {CAMPAIGN}")
    print(f"  Data variant  : {_VARIANT_TAG}")
    print(f"  Output folder : {OUTPUT_DIR}\n")

    HOUR_RANGE  = list(range(HOUR_START, HOUR_END + 1))
    MONTH_RANGE = list(range(1, 13))

    hour_labels  = [f'{h:02d}:00' for h in HOUR_RANGE]
    month_labels = MONTH_NAMES

    combined_hourly_parked  = {h: 0 for h in HOUR_RANGE}
    combined_hourly_moved   = {h: 0 for h in HOUR_RANGE}
    combined_hourly_total   = {h: 0 for h in HOUR_RANGE}

    combined_monthly_parked = {m: 0 for m in MONTH_RANGE}
    combined_monthly_moved  = {m: 0 for m in MONTH_RANGE}
    combined_monthly_total  = {m: 0 for m in MONTH_RANGE}

    sensors_loaded  = []
    sensor_windows  = {}

    # ==========================================================================
    # 1. PROCESS EACH SENSOR INDIVIDUALLY
    # ==========================================================================
    print("[1/2] Processing individual sensors ...")

    for sensor_name, file_path in SENSOR_FILES.items():
        print(f"\n  ── Sensor {sensor_name} ──")

        lat_col = SENSOR_COLS[sensor_name]['lat']
        lon_col = SENSOR_COLS[sensor_name]['lon']

        df = load_sensor_data(sensor_name, file_path, lat_col, lon_col)
        if df is None or df.empty:
            continue

        df_window = df[df['hour'].between(HOUR_START, HOUR_END)].reset_index(drop=True)

        if df_window.empty:
            print(f"    [WARNING] No data in {HOUR_START:02d}:00–{HOUR_END:02d}:59 — skipping")
            continue

        sensor_windows[sensor_name] = {
            'df':      df_window,
            'lat_col': lat_col,
            'lon_col': lon_col,
        }

        # ── A. HOURLY ANALYSIS ────────────────────────────────────────────────
        parked_h, moved_h, total_h = compute_stats_by_key(
            df_window, lat_col, lon_col, 'hour', HOUR_RANGE, THRESHOLD_M
        )

        print(f"\n  Hourly breakdown — {sensor_name}:")
        print_console_summary(sensor_name, hour_labels, HOUR_RANGE,
                               parked_h, moved_h, total_h)

        for h in HOUR_RANGE:
            combined_hourly_parked[h] += parked_h[h]
            combined_hourly_moved[h]  += moved_h[h]
            combined_hourly_total[h]  += total_h[h]

        pct_p, pct_m = compute_percentages(parked_h, moved_h, total_h, HOUR_RANGE)
        title = (
            f'Hourly parked / moving distribution  —  Sensor {sensor_name}  '
            f'(threshold: {THRESHOLD_M} m,  {HOUR_START:02d}:00 – {HOUR_END:02d}:59)\n'
            f'[{CAMPAIGN} / {_VARIANT_TAG}]'
        )
        fname = _build_output_filename(f'movement_hourly_{sensor_name}', CAMPAIGN, _VARIANT_TAG, 'png')
        plot_stacked_bar(
            pct_p, pct_m, hour_labels,
            'Hour of day (local time — Europe/Lisbon)',
            title,
            OUTPUT_DIR / fname,
            totals=[total_h[h] for h in HOUR_RANGE]
        )

        fname = _build_output_filename(f'movement_hourly_{sensor_name}', CAMPAIGN, _VARIANT_TAG, 'csv')
        save_csv_summary(
            'hour', hour_labels, HOUR_RANGE,
            parked_h, moved_h, total_h,
            OUTPUT_DIR / fname
        )

        # ── B. MONTHLY ANALYSIS ───────────────────────────────────────────────
        parked_m, moved_m, total_m = compute_stats_by_key(
            df_window, lat_col, lon_col, 'month', MONTH_RANGE, THRESHOLD_M
        )

        print(f"\n  Monthly breakdown — {sensor_name}:")
        print_console_summary(sensor_name, month_labels, MONTH_RANGE,
                               parked_m, moved_m, total_m)

        for mo in MONTH_RANGE:
            combined_monthly_parked[mo] += parked_m[mo]
            combined_monthly_moved[mo]  += moved_m[mo]
            combined_monthly_total[mo]  += total_m[mo]

        pct_p, pct_m = compute_percentages(parked_m, moved_m, total_m, MONTH_RANGE)
        title = (
            f'Monthly parked / moving distribution  —  Sensor {sensor_name}  '
            f'(threshold: {THRESHOLD_M} m,  daytime {HOUR_START:02d}:00 – {HOUR_END:02d}:59)\n'
            f'[{CAMPAIGN} / {_VARIANT_TAG}]'
        )
        fname = _build_output_filename(f'movement_monthly_{sensor_name}', CAMPAIGN, _VARIANT_TAG, 'png')
        plot_stacked_bar(
            pct_p, pct_m, month_labels,
            'Month (local time — Europe/Lisbon)',
            title,
            OUTPUT_DIR / fname,
            totals=[total_m[mo] for mo in MONTH_RANGE]
        )

        fname = _build_output_filename(f'movement_monthly_{sensor_name}', CAMPAIGN, _VARIANT_TAG, 'csv')
        save_csv_summary(
            'month', month_labels, MONTH_RANGE,
            parked_m, moved_m, total_m,
            OUTPUT_DIR / fname
        )

        sensors_loaded.append(sensor_name)

    # ==========================================================================
    # 2. COMBINED CHARTS (all sensors aggregated)
    # ==========================================================================
    if sensors_loaded:
        sensor_label = ', '.join(sensors_loaded)
        print(f"\n[2/2] Generating combined charts ({sensor_label}) ...")

        pct_p, pct_m = compute_percentages(
            combined_hourly_parked, combined_hourly_moved,
            combined_hourly_total, HOUR_RANGE
        )
        title = (
            f'Hourly parked / moving distribution  —  All sensors ({sensor_label})\n'
            f'(threshold: {THRESHOLD_M} m,  {HOUR_START:02d}:00 – {HOUR_END:02d}:59)  '
            f'[{CAMPAIGN} / {_VARIANT_TAG}]'
        )
        fname = _build_output_filename('movement_hourly_combined', CAMPAIGN, _VARIANT_TAG, 'png')
        plot_stacked_bar(
            pct_p, pct_m, hour_labels,
            'Hour of day (local time — Europe/Lisbon)',
            title,
            OUTPUT_DIR / fname,
            totals=[combined_hourly_total[h] for h in HOUR_RANGE]
        )
        fname = _build_output_filename('movement_hourly_combined', CAMPAIGN, _VARIANT_TAG, 'csv')
        save_csv_summary(
            'hour', hour_labels, HOUR_RANGE,
            combined_hourly_parked, combined_hourly_moved,
            combined_hourly_total,
            OUTPUT_DIR / fname
        )
        print_console_summary('COMBINED', hour_labels, HOUR_RANGE,
                               combined_hourly_parked, combined_hourly_moved,
                               combined_hourly_total)

        pct_p, pct_m = compute_percentages(
            combined_monthly_parked, combined_monthly_moved,
            combined_monthly_total, MONTH_RANGE
        )
        title = (
            f'Monthly parked / moving distribution  —  All sensors ({sensor_label})\n'
            f'(threshold: {THRESHOLD_M} m,  daytime {HOUR_START:02d}:00 – {HOUR_END:02d}:59)  '
            f'[{CAMPAIGN} / {_VARIANT_TAG}]'
        )
        fname = _build_output_filename('movement_monthly_combined', CAMPAIGN, _VARIANT_TAG, 'png')
        plot_stacked_bar(
            pct_p, pct_m, month_labels,
            'Month (local time — Europe/Lisbon)',
            title,
            OUTPUT_DIR / fname,
            totals=[combined_monthly_total[mo] for mo in MONTH_RANGE]
        )
        fname = _build_output_filename('movement_monthly_combined', CAMPAIGN, _VARIANT_TAG, 'csv')
        save_csv_summary(
            'month', month_labels, MONTH_RANGE,
            combined_monthly_parked, combined_monthly_moved,
            combined_monthly_total,
            OUTPUT_DIR / fname
        )
        print_console_summary('COMBINED', month_labels, MONTH_RANGE,
                               combined_monthly_parked, combined_monthly_moved,
                               combined_monthly_total)

    # ==========================================================================
    # 3. THRESHOLD SENSITIVITY ANALYSIS
    # ==========================================================================
    if RUN_SENSITIVITY_ANALYSIS and sensor_windows:
        print("\n[3/3] Running threshold sensitivity analysis ...")
        run_threshold_sensitivity_analysis(sensor_windows, HOUR_RANGE, OUTPUT_DIR)

    print('\n' + '=' * 70)
    print('Analysis complete.')
    print(f"  CAMPAIGN          : {CAMPAIGN}")
    print(f"  Data variant      : {_VARIANT_TAG}")
    print(f"  Sensors processed : {', '.join(sensors_loaded) if sensors_loaded else 'none'}")
    print(f"  Output directory  : {OUTPUT_DIR}")
    print('=' * 70)


# ==============================================================================
if __name__ == '__main__':
    main()