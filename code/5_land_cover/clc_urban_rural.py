"""
================================================================================
CLC Urban / Rural Distribution Analysis — Hourly and Monthly
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Classifies the sensor measurements into Urban and Rural with the CORINE Land
Cover (CLC) class of each point, then computes their distribution:
  (A) per hour of day     — daily activity pattern
  (B) per calendar month  — seasonal evolution over the year

Method
------
The land_use_clcv column equals the CLC grid_code (1-based, range 1–44).
It is mapped to the CLC legend and to the Urban / Rural grouping with the
shared clc_mapping module. Only the HOUR_START–HOUR_END window (local
Europe/Lisbon time) is kept. Bars with fewer than MIN_OBSERVATIONS
measurements are drawn in grey and annotated with their count.

Sensors whose file is not found are skipped with a warning.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', 'land_use_clcv'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             clc_hourly_<sensor>_<CAMPAIGN>_<variant_tag>.png / .csv
             clc_monthly_<sensor>_<CAMPAIGN>_<variant_tag>.png / .csv
             (<sensor> = J30–J33 or COMBINED)
             optional, for RURAL_DETAIL_SENSORS:
             clc_rural_{detail,by_class,by_hour,by_month}_<sensor>_
             <CAMPAIGN>_<variant_tag>.csv

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

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from clc_mapping import CLC_MAPPING, CLC_URBAN_RURAL

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX = "sd_cleaned"               # standard campaign
TUKTUK_FILE_SUFFIX = "sd_cleaned"        # tuk-tuk campaign

# Shared datetime column name
DATE_COL = 'Human readable date (Europe/London)'

# --- Time window to analyse (inclusive, local Europe/Lisbon time) -------------
# Each value N includes all measurements from N:00 to N:59.
HOUR_START = 6   # earliest hour included (covers 06:00–06:59)
HOUR_END   = 20  # latest   hour included (covers 20:00–20:59)

# --- Timezone -----------------------------------------------------------------
LOCAL_TZ = 'Europe/Lisbon'

# --- Output options -----------------------------------------------------------
# Base output directory.  A sub-folder named after CAMPAIGN / variant tag is
# created automatically so that results from different campaigns/variants
# are kept separate.
OUTPUT_BASE_DIR = OUTPUT_ROOT / "5_land_cover" / "clc_outputs"
OUTPUT_DPI = 300

# Show each figure interactively (in addition to saving it)? False by default
# so batch runs never block on plt.show().
SHOW_FIGURES: bool = False

# Save hourly / monthly summary tables as CSV?
SAVE_CSV = True

# Save charts as PNG?
SAVE_PNG = True

# --- Minimum observations per bar ---------------------------------------------
# Bars with fewer observations than this threshold are considered statistically
# unreliable and are displayed in grey with an "n = X" annotation instead of
# the normal Urban / Rural colours.  Set to 0 to disable the filter entirely.
MIN_OBSERVATIONS = 600

# --- Rural detail export ------------------------------------------------------
# Sensors for which to export rural-detail CSV files (3 files per sensor:
# full row-level detail, count by CLC class, count by hour and CLC class,
# count by month and CLC class).  Use an empty list [] to disable entirely.
RURAL_DETAIL_SENSORS: list[str] = []   # e.g. ['J30', 'J31']

# ==============================================================================
# CORINE LAND COVER MAPPING  –  44 classes
# Imported from clc_mapping.py (code/common/).
# land_use_clcv column = grid_code directly (1-based, range 1–44).
# CLC_MAPPING     : dict[int, CLCEntry]  land_use_clcv → named tuple
# CLC_URBAN_RURAL : dict[int, str]       land_use_clcv → 'Urban' / 'Rural'
# ==============================================================================

# Chart colours for Urban / Rural groups
GROUP_COLORS: dict[str, str] = {
    "Urban": "#E15759",
    "Rural": "#76B7B2",
}

# Month labels used on the x-axis of monthly charts
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Hour-range label shown on chart titles and console output
HOUR_RANGE_LABEL = (
    "full day" if (HOUR_START == 0 and HOUR_END == 23)
    else f"{HOUR_START:02d}:00 – {HOUR_END:02d}:59"
)


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# Input and output paths follow from CAMPAIGN and the input suffix.

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict:
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
        {'J30': Path(...), 'J31': Path(...), ...}
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
    """
    Build the output directory path, embedding the campaign and active
    variant tag so results from different campaigns/variants never
    overwrite each other.

    Parameters
    ----------
    base_dir : Path
        Base output directory set in the configuration section.
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().

    Returns
    -------
    Path
        Full output directory path.
    """
    return base_dir / campaign / variant_tag


def _build_output_filename(base_name: str, campaign: str, variant_tag: str,
                           extension: str) -> str:
    """
    Build an output filename that embeds the active campaign and variant tag.

    This ensures that output files from different campaigns/data variants
    are uniquely named and never overwrite each other even if placed in the
    same folder.

    Parameters
    ----------
    base_name : str
        Descriptive part of the filename (e.g. 'clc_hourly_J31').
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().
    extension : str
        File extension without leading dot (e.g. 'png', 'csv').

    Returns
    -------
    str
        Complete filename, e.g. 'clc_hourly_J31_standard_sd_cleaned_lisbon.png'.
    """
    return f"{base_name}_{campaign}_{variant_tag}.{extension}"


_VARIANT_TAG: str = _variant_tag()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sensor_data(sensor_name: str, file_path: Path) -> pd.DataFrame | None:
    """
    Load and prepare a single sensor CSV file for CLC analysis.

    Steps
    -----
    1. Check that the file exists; warn and return None if not found.
    2. Read the CSV, keeping only the required columns.
    3. Parse and localise the datetime column.
    4. Drop rows with missing datetime or land_use_clcv values.
    5. Exclude out-of-range land_use_clcv values (clean_data.py enforces 1–44).
    6. Map land_use_clcv directly to CLC attributes (land_use_clcv = grid_code, 1-based)
       using the shared clc_mapping module: clc_code, label, category, group.
    7. Extract hour-of-day and calendar month.
    8. Restrict to the requested daytime hour window.

    Parameters
    ----------
    sensor_name : str
        Sensor identifier used in log messages (e.g. 'J31').
    file_path : Path
        Path to the sensor CSV file for this variant.

    Returns
    -------
    pd.DataFrame or None
        Enriched DataFrame ready for analysis, or None if the file is
        missing or produces an empty result after filtering.
    """
    print(f"  Loading {sensor_name} from {file_path.name} ...")

    if not file_path.exists():
        print(f"    [WARNING] File not found — skipping {sensor_name}")
        return None

    # Read only the columns needed for this analysis
    useful_cols = {DATE_COL, "land_use_clcv"}
    df = pd.read_csv(
        file_path,
        usecols=lambda c: c in useful_cols,
        low_memory=False,
    )
    print(f"    -> {len(df):,} rows read")

    # Parse datetime and convert to local timezone
    df[DATE_COL] = (
        pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
        .dt.tz_convert(LOCAL_TZ)
    )

    # Drop rows with missing datetime or land_use_clcv
    df = df.dropna(subset=[DATE_COL, "land_use_clcv"]).reset_index(drop=True)

    # Exclude out-of-range land_use_clcv values (clean_data.py enforces 1–44)
    df["land_use_clcv"] = pd.to_numeric(df["land_use_clcv"], errors="coerce")
    df = df[(df["land_use_clcv"] >= 1) & (df["land_use_clcv"] <= 44)].reset_index(drop=True)

    # Map land_use_clcv directly to CLC attributes (land_use_clcv = grid_code, 1-based)
    df["clc_code"] = df["land_use_clcv"].map(
        lambda lu: CLC_MAPPING[int(lu)].clc_code if int(lu) in CLC_MAPPING else None
    )
    df["label"]    = df["land_use_clcv"].map(
        lambda lu: CLC_MAPPING[int(lu)].lvl3_label if int(lu) in CLC_MAPPING else "Unknown"
    )
    df["category"] = df["land_use_clcv"].map(
        lambda lu: CLC_MAPPING[int(lu)].lvl2_label if int(lu) in CLC_MAPPING else "Unknown"
    )
    df["group"]    = df["land_use_clcv"].map(CLC_URBAN_RURAL).fillna("Unknown")
    df["source"]   = sensor_name

    # Extract time keys used in both analyses
    df["hour"]  = df[DATE_COL].dt.hour    # 0–23
    df["month"] = df[DATE_COL].dt.month   # 1 = January … 12 = December

    # Restrict to the requested daytime hour window
    df = df[df["hour"].between(HOUR_START, HOUR_END)].reset_index(drop=True)

    print(f"    -> {len(df):,} valid rows after filtering  "
          f"(hours {HOUR_START:02d}–{HOUR_END:02d})")
    return df


# ==============================================================================
# AGGREGATION HELPERS
# ==============================================================================

def compute_pivot(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Compute the Urban / Rural percentage breakdown aggregated by a time key.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame produced by load_sensor_data().
    by : str
        Column to group by — either ``'hour'`` or ``'month'``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by the chosen key, with one column per CLC group
        (Urban, Rural, and optionally Unknown), values in percent (0–100).
    """
    counts = df.groupby([by, "group"]).size().rename("n").reset_index()
    counts["pct"] = (
        100.0 * counts["n"] / counts.groupby(by)["n"].transform("sum")
    )
    pivot = counts.pivot(index=by, columns="group", values="pct").fillna(0.0)

    # Ensure standard columns are always present
    for col in ("Urban", "Rural"):
        if col not in pivot.columns:
            pivot[col] = 0.0

    # Keep only columns that have non-zero data (drops 'Unknown' if absent)
    active_cols = [
        c for c in ("Urban", "Rural", "Unknown")
        if c in pivot.columns and pivot[c].sum() > 0
    ]
    return pivot[active_cols]


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_stacked_bar(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    x_labels: list[str],
    x_axis_label: str,
    title: str,
    counts: pd.Series | None = None,
    all_keys: list[int] | None = None,
) -> None:
    """
    Draw a stacked bar chart of Urban / Rural distribution on *ax*.

    Two situations trigger a grey "Insufficient data" bar instead of the
    normal Urban / Rural colours:
      1. The key is present in the pivot but has fewer than MIN_OBSERVATIONS
         observations — the data exist but are statistically unreliable.
      2. The key is completely absent from the pivot (zero observations for
         that hour or month)
    Both cases produce a consistent grey hatched bar annotated with the
    actual sample count ("n = X" or "n = 0"), ensuring no silent blank slots
    in the chart.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes to draw on.
    pivot : pd.DataFrame
        Output of compute_pivot(), indexed by hour or month.
    x_labels : list[str]
        Tick labels for each position in *all_keys* order (e.g. '06:00',
        'Jan').  Must have the same length as *all_keys*.
    x_axis_label : str
        Label printed below the x-axis.
    title : str
        Chart title.
    counts : pd.Series or None
        Number of observations per key value (same index as pivot).
        Required when MIN_OBSERVATIONS > 0; if None the filter is skipped.
    all_keys : list[int] or None
        Complete ordered list of expected key values (e.g. all hours in the
        analysis window, or 1–12 for months).  Keys absent from *pivot* are
        treated as having zero observations and receive a grey bar.
        When None the function falls back to the keys present in *pivot*.
    """
    # Build the full ordered key array, including keys absent from the pivot
    if all_keys is not None:
        keys = np.array(all_keys)
    else:
        keys = np.array(sorted(pivot.index.tolist()))

    bottom = np.zeros(len(keys))

    # Pre-compute per-bar observation counts.
    # Keys absent from counts (and from the pivot) receive count = 0, which
    # will fall below MIN_OBSERVATIONS and trigger the grey-bar rendering.
    n_obs = np.zeros(len(keys), dtype=int)
    if counts is not None:
        for i, k in enumerate(keys):
            n_obs[i] = int(counts.get(k, 0))

    # A key is "unreliable" when its count is strictly below the threshold,
    # OR when it is completely absent from the pivot (zero observations).
    def _is_unreliable(i: int, k: int) -> bool:
        if MIN_OBSERVATIONS > 0 and counts is not None and n_obs[i] < MIN_OBSERVATIONS:
            return True
        if k not in pivot.index:
            return True
        return False

    # ── Pass 1 : draw normal (reliable) bars ──────────────────────────────────
    for group, color in GROUP_COLORS.items():
        if group not in pivot.columns:
            continue
        values = pivot.reindex(keys)[group].fillna(0.0).values

        for i, (k, val) in enumerate(zip(keys, values)):
            # Skip positions that will be overwritten as grey in Pass 2
            if _is_unreliable(i, k):
                continue
            ax.bar(
                k, val, bottom=bottom[i],
                color=color,
                label=group if i == 0 else "_nolegend_",
                width=0.82, edgecolor="white", linewidth=0.4,
            )
            if val > 4:
                ax.text(
                    k, bottom[i] + val / 2,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold",
                )
        bottom += values

    # ── Pass 2 : draw grey bar for every unreliable or empty position ─────────
    # This pass covers both keys that are in the pivot with low counts AND
    # keys that are entirely absent (zero observations, no pivot row at all).
    unreliable_legend_added = False
    for i, k in enumerate(keys):
        if not _is_unreliable(i, k):
            continue
        leg_label = "Insufficient data" if not unreliable_legend_added else "_nolegend_"
        ax.bar(
            k, 100, width=0.82,
            color="#BDBDBD", alpha=0.70,
            hatch="//", edgecolor="#9E9E9E",
            label=leg_label,
        )
        ax.text(
            k, 50, f"n = {n_obs[i]}",
            ha="center", va="center",
            fontsize=8, color="#424242", fontweight="bold",
        )
        unreliable_legend_added = True

    # ── Legend — always show Urban / Rural swatches; add grey if needed ───────
    legend_map = {
        g: mpatches.Patch(color=c, label=g)
        for g, c in GROUP_COLORS.items()
    }
    ordered_labels = [g for g in ("Urban", "Rural") if g in pivot.columns]
    if unreliable_legend_added:
        ordered_labels.append("Insufficient data")
        legend_map["Insufficient data"] = mpatches.Patch(
            facecolor="#BDBDBD", edgecolor="#9E9E9E",
            hatch="//", alpha=0.70, label="Insufficient data",
        )
    ax.legend(
        handles=[legend_map[l] for l in ordered_labels],
        labels=ordered_labels,
        loc="upper left", fontsize=8, framealpha=0.85,
    )

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(keys)
    # x_labels is always aligned with all_keys (same length and same order)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45)
    ax.set_xlabel(x_axis_label, fontsize=9)
    ax.set_ylabel("Share of observations (%)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    # Footnote shown only when the reliability filter is active
    if MIN_OBSERVATIONS > 0:
        ax.text(
            0.01, 0.01,
            f"Grey bars: fewer than {MIN_OBSERVATIONS} observations (unreliable)",
            transform=ax.transAxes, fontsize=8, color="#757575",
            va="bottom", ha="left", style="italic",
        )


def make_hourly_chart(df: pd.DataFrame, label: str) -> plt.Figure:
    """
    Create a stacked bar chart of hourly Urban / Rural distribution.

    Every hour in the analysis window [HOUR_START, HOUR_END] is always
    represented on the x-axis.  Hours with no data at all receive a grey
    "n = 0" bar instead of a silent blank gap.

    Parameters
    ----------
    df    : Enriched DataFrame for one sensor (or the combined dataset).
    label : Sensor name or combined label used in the chart title.

    Returns
    -------
    plt.Figure
    """
    pivot = compute_pivot(df, by="hour")

    # Full expected hour range — includes hours that may be absent from pivot
    all_hours     = list(range(HOUR_START, HOUR_END + 1))
    x_tick_labels = [f"{h:02d}:00" for h in all_hours]

    # Observation count per hour — absent hours get count = 0 via .get(k, 0)
    hour_counts = df.groupby("hour")["group"].count()

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_stacked_bar(
        ax, pivot,
        x_labels=x_tick_labels,
        x_axis_label=f"Hour of day — {LOCAL_TZ}  ({HOUR_RANGE_LABEL})",
        title=f"Urban / Rural Hourly Distribution — {label}  [{HOUR_RANGE_LABEL}]",
        counts=hour_counts,
        all_keys=all_hours,
    )
    fig.tight_layout()
    return fig


def make_monthly_chart(df: pd.DataFrame, label: str) -> plt.Figure:
    """
    Create a stacked bar chart of monthly Urban / Rural distribution.

    All 12 calendar months are always represented on the x-axis.  Months
    with no data at all receive a grey "n = 0" bar instead of a silent
    blank gap.

    Parameters
    ----------
    df    : Enriched DataFrame for one sensor (or the combined dataset).
    label : Sensor name or combined label used in the chart title.

    Returns
    -------
    plt.Figure
    """
    pivot = compute_pivot(df, by="month")

    # Full 12-month range — includes months that may be absent from pivot
    all_months    = list(range(1, 13))
    x_tick_labels = [MONTH_LABELS[m - 1] for m in all_months]

    # Observation count per month — absent months get count = 0 via .get(k, 0)
    month_counts = df.groupby("month")["group"].count()

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_stacked_bar(
        ax, pivot,
        x_labels=x_tick_labels,
        x_axis_label="Month of year",
        title=f"Urban / Rural Monthly Distribution — {label}",
        counts=month_counts,
        all_keys=all_months,
    )
    fig.tight_layout()
    return fig


# ==============================================================================
# CONSOLE SUMMARY
# ==============================================================================

def print_text_summary(df: pd.DataFrame, label: str) -> None:
    """
    Print a formatted hourly and monthly summary to the console, including:
      - Hour-by-hour Urban / Rural / Unknown percentages with dominant CLC class
      - Month-by-month Urban / Rural / Unknown percentages
      - Overall totals
      - Top-5 CLC classes by observation count

    Parameters
    ----------
    df    : Enriched DataFrame for one sensor or the combined dataset.
    label : Sensor name or combined label used in the header.
    """
    SEP  = "=" * 80
    LINE = "-" * 80

    pivot_h = compute_pivot(df, by="hour")
    pivot_m = compute_pivot(df, by="month")
    n_total = len(df)

    print(f"\n{SEP}")
    print(f"  SUMMARY — {label}  [{HOUR_RANGE_LABEL}]  [{CAMPAIGN}/{_VARIANT_TAG}]")
    print(SEP)

    # -- Hourly breakdown ------------------------------------------------------
    print(f"\n  {'Hour':<7} {'Urban':>10} {'Rural':>10} {'Unknown':>9}   "
          f"Dominant Urban CLC class")
    print(LINE)

    for h in range(HOUR_START, HOUR_END + 1):
        # Guard on the columns too: a dataset may have no Rural (or no Urban)
        # measurement at all, e.g. the tuk-tuk campaign inside Lisbon.
        urban   = (
            pivot_h.at[h, "Urban"]
            if (h in pivot_h.index and "Urban" in pivot_h.columns) else 0.0
        )
        rural   = (
            pivot_h.at[h, "Rural"]
            if (h in pivot_h.index and "Rural" in pivot_h.columns) else 0.0
        )
        unknown = (
            pivot_h.at[h, "Unknown"]
            if (h in pivot_h.index and "Unknown" in pivot_h.columns) else 0.0
        )
        urban_subset = df[(df["hour"] == h) & (df["group"] == "Urban")]
        if not urban_subset.empty:
            vc        = urban_subset["label"].value_counts()
            top_label = vc.idxmax()
            top_pct   = 100.0 * vc.iloc[0] / len(urban_subset)
            dominant  = f"{top_label[:38]:<38} ({top_pct:.0f}% of urban)"
        else:
            dominant  = "—"
        print(f"  {h:02d}:00  {urban:>9.1f}% {rural:>9.1f}% {unknown:>8.1f}%   {dominant}")

    # -- Monthly breakdown -----------------------------------------------------
    print(f"\n  {'Month':<9} {'Urban':>10} {'Rural':>10} {'Unknown':>9}")
    print(LINE)

    for m in range(1, 13):
        if m not in pivot_m.index:
            continue
        urban   = pivot_m.at[m, "Urban"]   if "Urban"   in pivot_m.columns else 0.0
        rural   = pivot_m.at[m, "Rural"]   if "Rural"   in pivot_m.columns else 0.0
        unknown = pivot_m.at[m, "Unknown"] if "Unknown" in pivot_m.columns else 0.0
        month_count = len(df[df["month"] == m])
        print(f"  {MONTH_LABELS[m - 1]:<9} {urban:>9.1f}% {rural:>9.1f}% {unknown:>8.1f}%"
              f"   ({month_count:,} observations)")

    # -- Overall totals --------------------------------------------------------
    group_counts = df["group"].value_counts()
    u_tot   = 100.0 * group_counts.get("Urban",   0) / n_total
    r_tot   = 100.0 * group_counts.get("Rural",   0) / n_total
    unk_tot = 100.0 * group_counts.get("Unknown", 0) / n_total

    print(LINE)
    print(f"  TOTAL     {u_tot:>9.1f}% {r_tot:>9.1f}% {unk_tot:>8.1f}%   ({n_total:,} observations)")

    # -- Top-5 CLC classes -----------------------------------------------------
    print(f"\n  Top 5 CLC classes — {label}:")
    top5 = (
        df.groupby(["label", "group", "category"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
        .head(5)
    )
    for _, row in top5.iterrows():
        pct = 100.0 * row["n"] / n_total
        print(f"    [{row['group']:6}]  {row['label'][:42]:<42}  "
              f"{pct:5.1f}%  ({row['category']})")

    print(SEP)


# ==============================================================================
# CSV EXPORT
# ==============================================================================

def save_hourly_csv(df: pd.DataFrame, sensor_name: str, out_dir: Path) -> None:
    """
    Save the hourly Urban / Rural pivot table as a CSV file.

    The output filename embeds CAMPAIGN and the active variant tag so that
    results from different campaigns/variants are kept separate.

    Parameters
    ----------
    df          : Enriched DataFrame for one sensor or the combined dataset.
    sensor_name : Label used in the output filename and 'source' column.
    out_dir     : Destination directory.
    """
    pivot = compute_pivot(df, by="hour")
    pivot_out = pivot.reset_index()
    pivot_out.insert(0, "source", sensor_name)
    hour_counts = df.groupby("hour")["group"].count().rename("n_observations")
    pivot_out = pivot_out.join(hour_counts, on="hour")
    filename = _build_output_filename(f"clc_hourly_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "csv")
    csv_path = out_dir / filename
    pivot_out.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"    -> Saved : {csv_path}")


def save_monthly_csv(df: pd.DataFrame, sensor_name: str, out_dir: Path) -> None:
    """
    Save the monthly Urban / Rural pivot table as a CSV file.

    The output filename embeds CAMPAIGN and the active variant tag so that
    results from different campaigns/variants are kept separate.

    Parameters
    ----------
    df          : Enriched DataFrame for one sensor or the combined dataset.
    sensor_name : Label used in the output filename and 'source' column.
    out_dir     : Destination directory.
    """
    pivot = compute_pivot(df, by="month")
    pivot_out = pivot.reset_index()
    pivot_out.insert(0, "source", sensor_name)
    pivot_out.insert(2, "month_name", pivot_out["month"].map(
        lambda m: MONTH_LABELS[m - 1]
    ))
    month_counts = df.groupby("month")["group"].count().rename("n_observations")
    pivot_out = pivot_out.join(month_counts, on="month")
    filename = _build_output_filename(f"clc_monthly_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "csv")
    csv_path = out_dir / filename
    pivot_out.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"    -> Saved : {csv_path}")


def export_rural_detail(df: pd.DataFrame, sensor_name: str, out_dir: Path) -> None:
    """
    Export Rural-classified observations to detailed CSV files:
      1. Full row-level detail for all rural points.
      2. Count aggregated by CLC class.
      3. Count aggregated by hour and CLC class.
      4. Count aggregated by month and CLC class.

    All output filenames embed CAMPAIGN and the active variant tag.

    Parameters
    ----------
    df          : Enriched DataFrame for one sensor.
    sensor_name : Label used in output filenames (e.g. 'J31').
    out_dir     : Destination directory.
    """
    rural   = df[df["group"] == "Rural"].copy()
    n_rural = len(rural)
    n_total = len(df)

    if n_total == 0:
        print(f"  [WARNING] No data available for {sensor_name} — skipping rural detail export.")
        return

    print(f"\n  Rural observations — {sensor_name}: "
          f"{n_rural:,} / {n_total:,} ({100 * n_rural / n_total:.2f}%)")

    if n_rural == 0:
        print("  No rural observations found.")
        return

    # 1. Full row-level detail
    detail_path = out_dir / _build_output_filename(
        f"clc_rural_detail_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "csv"
    )
    rural.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"    -> Saved : {detail_path}")

    # 2. Count by CLC class
    by_class = (
        rural.groupby(["land_use_clcv", "label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    class_path = out_dir / _build_output_filename(
        f"clc_rural_by_class_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "csv"
    )
    by_class.to_csv(class_path, index=False, encoding="utf-8-sig")
    print(f"    -> Saved : {class_path}")

    # 3. Count by hour and CLC class
    by_hour = (
        rural.groupby(["hour", "land_use_clcv", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["hour", "count"], ascending=[True, False])
    )
    hour_path = out_dir / _build_output_filename(
        f"clc_rural_by_hour_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "csv"
    )
    by_hour.to_csv(hour_path, index=False, encoding="utf-8-sig")
    print(f"    -> Saved : {hour_path}")

    # 4. Count by month and CLC class
    by_month = (
        rural.groupby(["month", "land_use_clcv", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["month", "count"], ascending=[True, False])
    )
    by_month.insert(1, "month_name", by_month["month"].map(lambda m: MONTH_LABELS[m - 1]))
    month_path = out_dir / _build_output_filename(
        f"clc_rural_by_month_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "csv"
    )
    by_month.to_csv(month_path, index=False, encoding="utf-8-sig")
    print(f"    -> Saved : {month_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Run the full CLC Urban / Rural distribution analysis for all sensors.

    Workflow
    --------
    1. Resolve input file paths and output directory from CAMPAIGN and
       the active variant tag.
    2. Create the output directory if it does not already exist.
    3. Attempt to load each sensor file; sensors with missing files are
       skipped with a warning and do not interrupt the run.
    4. For each successfully loaded sensor:
         a. Plot and optionally save the hourly distribution chart.
         b. Plot and optionally save the monthly distribution chart.
         c. Print the console summary (hourly + monthly + top-5 classes).
         d. Optionally save hourly and monthly CSV summaries.
         e. Optionally export rural detail CSV files.
    5. If multiple sensors were loaded, produce combined (all-sensor) charts
       and summaries following the same steps.
    """
    # Validate hour range
    if not (0 <= HOUR_START <= HOUR_END <= 23):
        raise ValueError(
            f"Invalid hour range: HOUR_START={HOUR_START}, HOUR_END={HOUR_END}. "
            "Both must be in [0, 23] with HOUR_START ≤ HOUR_END."
        )

    # ── Resolve paths from CAMPAIGN + variant tag ─────────────────────────────
    SENSOR_FILES = _build_sensor_files(CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG)
    OUTPUT_DIR   = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("  CLC Urban / Rural Distribution Analysis")
    print(f"  CAMPAIGN      : {CAMPAIGN}")
    print(f"  Data variant  : {_VARIANT_TAG}")
    print(f"  Hour range    : {HOUR_RANGE_LABEL}")
    print(f"  Output folder : {OUTPUT_DIR.resolve()}")
    print("=" * 72 + "\n")

    # ── Load all sensor files ──────────────────────────────────────────────────
    sensor_data: dict[str, pd.DataFrame] = {}

    for sensor_name, file_path in SENSOR_FILES.items():
        df = load_sensor_data(sensor_name, file_path)
        if df is not None and not df.empty:
            sensor_data[sensor_name] = df

    print()

    if not sensor_data:
        print("  [ERROR] No sensor data could be loaded.")
        print("  Check that DATA_DIR/TUKTUK_DATA_DIR contains files matching the pattern:")
        print(f"    (CAMPAIGN={CAMPAIGN!r}, variant={_VARIANT_TAG!r})")
        return

    loaded = ", ".join(sensor_data.keys())
    print(f"  Sensors loaded : {loaded}\n")

    # ── Per-sensor processing ──────────────────────────────────────────────────
    for sensor_name, df in sensor_data.items():

        print(f"  Processing {sensor_name} ...")

        # -- Hourly chart (always produced) -------------------------------------
        fig_h = make_hourly_chart(df, label=sensor_name)
        if SAVE_PNG:
            png_name = _build_output_filename(
                f"clc_hourly_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "png"
            )
            png_h = OUTPUT_DIR / png_name
            fig_h.savefig(png_h, dpi=OUTPUT_DPI, bbox_inches="tight")
            print(f"    -> Saved : {png_h}")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig_h)

        # -- Monthly chart (always produced) -------------------------------------
        fig_m = make_monthly_chart(df, label=sensor_name)
        if SAVE_PNG:
            png_name = _build_output_filename(
                f"clc_monthly_{sensor_name}", CAMPAIGN, _VARIANT_TAG, "png"
            )
            png_m = OUTPUT_DIR / png_name
            fig_m.savefig(png_m, dpi=OUTPUT_DPI, bbox_inches="tight")
            print(f"    -> Saved : {png_m}")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig_m)

        # -- Console summary ---------------------------------------------------
        print_text_summary(df, label=sensor_name)

        # -- CSV summaries -----------------------------------------------------
        if SAVE_CSV:
            save_hourly_csv(df, sensor_name, OUTPUT_DIR)
            save_monthly_csv(df, sensor_name, OUTPUT_DIR)

        # -- Rural detail export -----------------------------------------------
        if sensor_name in RURAL_DETAIL_SENSORS:
            export_rural_detail(df, sensor_name, OUTPUT_DIR)

        print()

    # ── Combined charts (only when multiple sensors were loaded) ───────────────
    if len(sensor_data) > 1:
        combined_label = " + ".join(sensor_data.keys())
        combined_df    = pd.concat(sensor_data.values(), ignore_index=True)

        print(f"  Processing combined dataset ({combined_label}) ...")

        # -- Combined hourly chart (always produced) -----------------------------
        fig_hc = make_hourly_chart(combined_df, label=f"{combined_label} (combined)")
        if SAVE_PNG:
            png_name = _build_output_filename(
                "clc_hourly_COMBINED", CAMPAIGN, _VARIANT_TAG, "png"
            )
            png_hc = OUTPUT_DIR / png_name
            fig_hc.savefig(png_hc, dpi=OUTPUT_DPI, bbox_inches="tight")
            print(f"    -> Saved : {png_hc}")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig_hc)

        # -- Combined monthly chart (always produced) ----------------------------
        fig_mc = make_monthly_chart(combined_df, label=f"{combined_label} (combined)")
        if SAVE_PNG:
            png_name = _build_output_filename(
                "clc_monthly_COMBINED", CAMPAIGN, _VARIANT_TAG, "png"
            )
            png_mc = OUTPUT_DIR / png_name
            fig_mc.savefig(png_mc, dpi=OUTPUT_DPI, bbox_inches="tight")
            print(f"    -> Saved : {png_mc}")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig_mc)

        # -- Combined console summary ------------------------------------------
        print_text_summary(combined_df, label=f"COMBINED ({combined_label})")

        # -- Combined CSV summaries --------------------------------------------
        if SAVE_CSV:
            save_hourly_csv(combined_df, "COMBINED", OUTPUT_DIR)
            save_monthly_csv(combined_df, "COMBINED", OUTPUT_DIR)

    print("\n" + "=" * 72)
    print("  Analysis complete.")
    print(f"  CAMPAIGN      : {CAMPAIGN}")
    print(f"  Data variant  : {_VARIANT_TAG}")
    print(f"  Output folder : {OUTPUT_DIR.resolve()}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()