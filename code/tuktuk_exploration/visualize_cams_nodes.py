"""
================================================================================
CAMS Node Time-Series Visualisation — Tuk-Tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Visualises the CAMS solar irradiance time series downloaded from the
Copernicus Atmosphere Monitoring Service (CAMS / ADS) API by
join_cams_tuktuk.py. One multi-panel figure per grid node found in
CAMS_TEMP_DIR:
    Panel 1 — Daily GHI profiles (all days overlaid, colour-coded by date)
    Panel 2 — Clear-sky GHI vs all-sky GHI scatter, coloured by kc
    Panel 3 — Monthly GHI energy totals (kWh/m²)
    Panel 4 — Diurnal percentile envelope of the clear-sky index kc
Rows with sun elevation below MIN_SUN_ELEV_DEG are excluded; the scatter and
kc plots also exclude GHI or GHI_clear below MIN_GHI_WM2.

CAMS file format
----------------
CAMS format v2, semicolon-separated:
  - comment lines start with '#'; the last one is the column header;
  - data rows use interval notation, e.g.
        2026-06-29T00:00:00.0/2026-06-29T00:01:00.0;0.0;...
  - values are irradiation accumulated over the interval (Wh/m²); dividing
    by the interval duration in hours gives W/m²:
        1-minute : ÷ (1/60)    15-minute : ÷ 0.25    1-hour : ÷ 1.0

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : CAMS_TEMP_DIR = CACHE_DIR / cams  (cache of join_cams_tuktuk.py)
             cams_<lat>_<lon>_<date_begin>_<date_end>_<time_step>.csv
    output : OUTPUT_ROOT / tuktuk_exploration / cams_viz /
             cams_node_<lat>_<lon>.png  (one per node)

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import sys
import warnings
from io import StringIO
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CACHE_DIR, OUTPUT_ROOT  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Directory containing the raw CAMS CSV files (CAMS_TEMP_DIR) --------------
# Same directory as CAMS_TEMP_DIR in join_cams_tuktuk.py.
CAMS_TEMP_DIR: Path = CACHE_DIR / "cams"

# --- Output directory ---------------------------------------------------------
OUTPUT_DIR: Path = OUTPUT_ROOT / "tuktuk_exploration" / "cams_viz"

# --- Figure -------------------------------------------------------------------
FIGURE_DPI:     int  = 300
SHOW_FIGURE:    bool = False   # Set True to open the figure interactively

# --- Solar elevation filter ---------------------------------------------------
# Rows with sun elevation below this threshold are excluded from the plots.
# This removes night-time noise and very-low-angle artefacts.
MIN_SUN_ELEV_DEG: float = 10.0

# --- Minimum irradiance for scatter / kc plots --------------------------------
# Rows with GHI or GHI_clear below this are excluded to avoid numerical noise.
MIN_GHI_WM2: float = 25.0

# --- Clear-sky index (kc) colour scale bounds ---------------------------------
KC_VMIN: float = 0.0
KC_VMAX: float = 1.0

# --- Palette ------------------------------------------------------------------
COLOR_GHI:       str = "#1565C0"   # all-sky GHI
COLOR_GHI_CLEAR: str = "#E65100"   # clear-sky GHI
COLOR_GHI_EXTRA: str = "#888888"   # extra-terrestrial (TOA)
COLOR_MONTHLY:   str = "#2E7D32"


# ==============================================================================
# STEP 1 — DISCOVER FILES
# ==============================================================================

def discover_cams_files(temp_dir: Path) -> list[Path]:
    """
    Return all CAMS raw CSV files found in temp_dir.

    Expected naming pattern (from join_cams_tuktuk.py):
        cams_<lat>_<lon>_<date_begin>_<date_end>_<time_step>.csv

    Parameters
    ----------
    temp_dir : Path
        Directory to search.

    Returns
    -------
    list of Path, sorted alphabetically.
    """
    files = sorted(temp_dir.glob("cams_*.csv"))
    return files


def parse_node_coords(filepath: Path) -> tuple[float, float] | tuple[None, None]:
    """
    Extract the (lat, lon) grid node coordinates from the file name.

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    (lat, lon) floats, or (None, None) if the pattern does not match.
    """
    # Pattern: cams_<lat>_<lon>_<YYYY-MM-DD>_<YYYY-MM-DD>_<step>.csv
    m = re.match(r"cams_([+-]?\d+\.\d+)_([+-]?\d+\.\d+)_", filepath.name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def infer_time_step_from_name(filepath: Path) -> str:
    """
    Infer the CAMS time step string from the file name suffix.

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    One of '1minute', '15minute', '1hour', '1day'.  Defaults to '1minute'.
    """
    name = filepath.stem.lower()
    for candidate in ("15minute", "1minute", "1hour", "1day"):
        if candidate in name:
            return candidate
    return "1minute"


# ==============================================================================
# STEP 2 — PARSE ONE CAMS CSV FILE
# ==============================================================================

_STEP_TO_HOURS: dict[str, float] = {
    "1minute":  1.0 / 60.0,
    "15minute": 15.0 / 60.0,
    "1hour":    1.0,
    "1day":     24.0,
}

# CAMS format v2 column names → pipeline column names
_COL_MAP: dict[str, str] = {
    "GHI":           "ghi",
    "Clear sky GHI": "ghi_clear",
    "TOA":           "ghi_extra",
    "BHI":           "bhi",
    "DHI":           "dhi",
    "BNI":           "dni",
    "Reliability":   "reliability",
}


def parse_cams_csv(raw_path: Path,
                   time_step: str | None = None) -> pd.DataFrame:
    """
    Parse one raw CAMS CSV file (format v2) into a clean irradiance DataFrame.

    The function:
    1. Skips metadata comment lines ('#') and reads the embedded header.
    2. Parses the start of each observation interval as a UTC timestamp.
    3. Converts Wh/m² accumulated irradiation → W/m² average irradiance.
    4. Clips all irradiance columns to ≥ 0.

    Parameters
    ----------
    raw_path : Path
        Path to the raw CAMS CSV file.
    time_step : str or None
        CAMS time step (e.g. '1minute', '15minute', '1hour').
        If None, inferred from the file name.

    Returns
    -------
    pd.DataFrame
        Columns:
            timestamp_utc   — UTC datetime (timezone-aware)
            ghi             — all-sky GHI             [W/m²]
            ghi_clear       — clear-sky GHI (McClear)  [W/m²]
            ghi_extra       — top-of-atmosphere (TOA)  [W/m²]
            bhi             — beam horizontal           [W/m²]
            dhi             — diffuse horizontal        [W/m²]
            dni             — direct normal             [W/m²]
            reliability     — CAMS reliability flag     (raw)
        Rows where the timestamp cannot be parsed are dropped.
    """
    if time_step is None:
        time_step = infer_time_step_from_name(raw_path)

    interval_h = _STEP_TO_HOURS.get(time_step, 1.0 / 60.0)

    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # The last '#'-prefixed line containing ';' is the column header
    header_line    = None
    first_data_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and ";" in stripped:
            header_line = stripped.lstrip("# ")
        elif not stripped.startswith("#") and stripped:
            first_data_idx = i
            break

    if header_line is None or first_data_idx is None:
        raise ValueError(
            f"Cannot locate column header or data rows in: {raw_path}"
        )

    raw = pd.read_csv(
        StringIO(header_line + "\n" + "".join(lines[first_data_idx:])),
        sep=";",
        skipinitialspace=True,
    )
    raw.columns = [c.strip() for c in raw.columns]

    # ── Timestamp: start of each observation interval ─────────────────────────
    ts_col = "Observation period" if "Observation period" in raw.columns else raw.columns[0]
    raw["timestamp_utc"] = (
        raw[ts_col]
        .astype(str)
        .str.split("/").str[0]
        .str.replace("T", " ", regex=False)
        .str[:16]
    )
    raw["timestamp_utc"] = pd.to_datetime(
        raw["timestamp_utc"], utc=True, errors="coerce"
    )
    raw = raw.dropna(subset=["timestamp_utc"]).reset_index(drop=True)

    # ── Build output DataFrame: convert Wh/m² → W/m² ──────────────────────────
    out = pd.DataFrame({"timestamp_utc": raw["timestamp_utc"]})
    irr_cols = ["ghi", "ghi_clear", "ghi_extra", "bhi", "dhi", "dni"]

    for out_col, src_col in _COL_MAP.items():
        # Note: _COL_MAP maps pipeline name → CAMS name; iterate reversed
        pass

    for cams_col, pipe_col in _COL_MAP.items():
        if cams_col in raw.columns:
            out[pipe_col] = pd.to_numeric(raw[cams_col], errors="coerce") / interval_h
        else:
            out[pipe_col] = np.nan

    for col in irr_cols:
        if col in out.columns:
            out[col] = out[col].clip(lower=0.0)

    return out


# ==============================================================================
# STEP 3 — DERIVED QUANTITIES
# ==============================================================================

def add_sun_elevation(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """
    Compute solar elevation angle for each timestamp and append it as
    'sun_elev' (degrees above horizon).

    Uses a simplified Spencer formula (accuracy ±0.01°).  pvlib is not
    required; the result is sufficient for daytime filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'timestamp_utc' (tz-aware UTC).
    lat, lon : float
        Node coordinates in decimal degrees (WGS-84).

    Returns
    -------
    pd.DataFrame with 'sun_elev' column added.
    """
    df = df.copy()
    ts = df["timestamp_utc"]

    # Day of year and fractional hour (UTC)
    doy  = ts.dt.day_of_year.values.astype(float)
    hour = (ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0).values

    # Spencer solar declination (radians)
    B    = (2.0 * np.pi / 365.0) * (doy - 1.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(B)     + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
        - 0.002697 * np.cos(3 * B) + 0.00148  * np.sin(3 * B)
    )  # already in radians

    # Equation of time (minutes)
    eot = (
        229.18 * (
            0.000075
            + 0.001868 * np.cos(B)    - 0.032077 * np.sin(B)
            - 0.014615 * np.cos(2*B) - 0.04089  * np.sin(2*B)
        )
    )

    # Solar noon correction: local solar time (hours)
    lst  = hour + lon / 15.0 + eot / 60.0
    ha   = np.radians((lst - 12.0) * 15.0)   # hour angle (radians)
    lat_r = np.radians(lat)

    sin_elev = (
        np.sin(lat_r) * np.sin(decl)
        + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    )
    df["sun_elev"] = np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))
    return df


def add_clear_sky_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the clear-sky index kc = GHI / GHI_clear and append it as 'kc'.

    kc measures cloud attenuation relative to the McClear clear-sky model.
    It is distinct from the atmospheric clearness index kt = GHI / GHI_extra,
    which quantifies total atmospheric attenuation (clouds + aerosols + air mass).

    kc is clipped to [0, 1.2] to remove unphysical artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ghi' and 'ghi_clear'.

    Returns
    -------
    pd.DataFrame with 'kc' column added.
    """
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        kc = np.where(
            df["ghi_clear"] > 0,
            df["ghi"] / df["ghi_clear"],
            np.nan,
        )
    df["kc"] = np.clip(kc, 0.0, 1.2)
    return df


# ==============================================================================
# STEP 4 — FIGURE
# ==============================================================================

def make_node_figure(df: pd.DataFrame,
                     lat: float, lon: float,
                     filepath: Path) -> plt.Figure:
    """
    Build a 4-panel diagnostic figure for one CAMS grid node.

    Panels
    ------
    1. Daily GHI profiles  — all days overlaid, colour-coded by date.
    2. All-sky vs clear-sky GHI scatter — points coloured by kc.
    3. Monthly GHI energy totals (kWh/m²).
    4. Diurnal kc envelope — median ± 25th/75th percentile by hour.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed CAMS data for one node, with sun_elev and kc columns.
    lat, lon : float
        Node coordinates.
    filepath : Path
        Source file path (used only for the figure annotation).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Daytime subset ─────────────────────────────────────────────────────────
    day = df[df["sun_elev"] >= MIN_SUN_ELEV_DEG].copy()
    scatter_mask = day["ghi"] >= MIN_GHI_WM2
    day_s = day[scatter_mask].copy()

    # Unique dates for colour mapping
    dates      = day["timestamp_utc"].dt.date
    unique_dates = sorted(dates.unique())
    n_dates    = max(len(unique_dates), 1)
    cmap_days  = cm.get_cmap("viridis", n_dates)
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # ── Panel 1: Daily GHI profiles ───────────────────────────────────────────
    day_plot = day.copy()
    day_plot["hour_frac"] = (
        day_plot["timestamp_utc"].dt.hour
        + day_plot["timestamp_utc"].dt.minute / 60.0
    )
    for d in unique_dates:
        subset = day_plot[day_plot["timestamp_utc"].dt.date == d]
        color  = cmap_days(date_to_idx[d] / max(n_dates - 1, 1))
        ax1.plot(
            subset["hour_frac"], subset["ghi"],
            color=color, linewidth=0.6, alpha=0.6,
        )

    # Overlay mean clear-sky profile
    if "ghi_clear" in day_plot.columns:
        mean_clear = day_plot.groupby("hour_frac")["ghi_clear"].mean()
        ax1.plot(
            mean_clear.index, mean_clear.values,
            color=COLOR_GHI_CLEAR, linewidth=1.4,
            linestyle="--", label="Mean clear-sky GHI",
        )

    ax1.set_xlabel("Hour (UTC)", fontsize=9)
    ax1.set_ylabel("GHI  [W/m²]", fontsize=9)
    ax1.set_title("Daily GHI Profiles", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 24)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, framealpha=0.8)

    # Colour bar for dates
    sm = cm.ScalarMappable(
        cmap=cmap_days,
        norm=mcolors.Normalize(vmin=0, vmax=n_dates - 1),
    )
    sm.set_array([])
    cbar1 = fig.colorbar(sm, ax=ax1, fraction=0.035, pad=0.02)
    cbar1.set_ticks([0, n_dates - 1])
    if n_dates > 1:
        cbar1.set_ticklabels([
            str(unique_dates[0]),
            str(unique_dates[-1]),
        ])
    cbar1.set_label("Date", fontsize=7.5)
    cbar1.ax.tick_params(labelsize=7)

    # ── Panel 2: All-sky vs clear-sky GHI scatter ─────────────────────────────
    sc = ax2.scatter(
        day_s["ghi_clear"], day_s["ghi"],
        c=day_s["kc"].clip(KC_VMIN, KC_VMAX),
        cmap="RdYlGn",
        vmin=KC_VMIN, vmax=KC_VMAX,
        s=2, alpha=0.5, linewidths=0, rasterized=True,
    )
    max_val = max(day_s["ghi_clear"].max(), day_s["ghi"].max()) * 1.05
    ax2.plot([0, max_val], [0, max_val],
             color="#555555", linewidth=0.9, linestyle="--", label="1:1 line")
    ax2.set_xlabel("Clear-sky GHI  [W/m²]", fontsize=9)
    ax2.set_ylabel("All-sky GHI  [W/m²]", fontsize=9)
    ax2.set_title("All-sky vs Clear-sky GHI", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.8)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    cbar2 = fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.02)
    cbar2.set_label("kc  (GHI / GHI_clear)", fontsize=7.5)
    cbar2.ax.tick_params(labelsize=7)

    # ── Panel 3: Monthly GHI energy totals ────────────────────────────────────
    # Integrate: W/m² × Δt_hours → Wh/m²; aggregate to kWh/m²
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly["timestamp_utc"].dt.to_period("M")
    # Determine time step from median interval
    if len(df_monthly) > 1:
        dt_hours = float(
            df_monthly["timestamp_utc"]
            .diff()
            .dt.total_seconds()
            .median()
        ) / 3600.0
    else:
        dt_hours = 1.0 / 60.0

    monthly = (
        df_monthly[df_monthly["ghi"] >= 0]
        .groupby("month")["ghi"]
        .sum()
        * dt_hours
        / 1000.0  # Wh/m² → kWh/m²
    )
    months_str = [str(p) for p in monthly.index]
    x_pos      = np.arange(len(months_str))
    ax3.bar(x_pos, monthly.values,
            color=COLOR_MONTHLY, alpha=0.85, width=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(months_str, rotation=45, ha="right", fontsize=7.5)
    ax3.set_ylabel("GHI energy  [kWh/m²]", fontsize=9)
    ax3.set_title("Monthly GHI Energy Totals", fontsize=10, fontweight="bold")
    ax3.set_ylim(bottom=0)

    # Annotate bars
    for xi, val in zip(x_pos, monthly.values):
        ax3.text(xi, val + 0.5, f"{val:.0f}", ha="center", va="bottom",
                 fontsize=7)

    # ── Panel 4: Diurnal kc envelope ──────────────────────────────────────────
    if not day_s.empty:
        day_s["hour_int"] = day_s["timestamp_utc"].dt.hour
        kc_stats = day_s.groupby("hour_int")["kc"].agg(
            med="median", p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
        )
        ax4.fill_between(
            kc_stats.index,
            kc_stats["p25"], kc_stats["p75"],
            alpha=0.3, color=COLOR_GHI, label="IQR (25th–75th percentile)",
        )
        ax4.plot(kc_stats.index, kc_stats["med"],
                 color=COLOR_GHI, linewidth=1.8, label="Median kc")
        ax4.axhline(0.95, color="#C0392B", linewidth=1.0,
                    linestyle=":", label="Clear-sky threshold (0.95)")
        ax4.set_xlabel("Hour (UTC)", fontsize=9)
        ax4.set_ylabel("kc  (GHI / GHI_clear)", fontsize=9)
        ax4.set_title("Diurnal Clear-sky Index kc", fontsize=10, fontweight="bold")
        ax4.set_xlim(0, 23)
        ax4.set_ylim(0, 1.2)
        ax4.legend(fontsize=8, framealpha=0.8)
    else:
        ax4.text(0.5, 0.5, "Insufficient data for kc envelope",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=9, color="#888888")
        ax4.set_title("Diurnal Clear-sky Index kc", fontsize=10, fontweight="bold")

    # ── Figure-level title and annotation ─────────────────────────────────────
    fig.suptitle(
        f"CAMS Node  {lat:.4f}°N  {lon:.4f}°E",
        fontsize=12, fontweight="bold", y=1.01,
    )
    n_days  = len(unique_dates)
    n_rows  = len(df)
    date_range = (
        f"{unique_dates[0]}  →  {unique_dates[-1]}"
        if n_dates > 1 else str(unique_dates[0]) if unique_dates else "—"
    )
    fig.text(
        0.5, -0.01,
        (
            f"Source: {filepath.name}  |  "
            f"Records: {n_rows:,}  |  "
            f"Days: {n_days}  |  "
            f"Period: {date_range}  |  "
            f"Sun elevation filter: ≥ {MIN_SUN_ELEV_DEG}°"
        ),
        ha="center", fontsize=7, color="#888888", style="italic",
    )

    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Discover, parse, and visualise all CAMS node files in CAMS_TEMP_DIR."""
    sep = "═" * 68
    print(sep)
    print("  CAMS NODE IRRADIANCE VISUALISER")
    print(f"  Source directory : {CAMS_TEMP_DIR}")
    print(f"  Output directory : {OUTPUT_DIR}")
    print(sep)

    # ── Discover files ─────────────────────────────────────────────────────────
    files = discover_cams_files(CAMS_TEMP_DIR)
    if not files:
        print(
            f"\n  [ERROR] No CAMS CSV files found in:\n"
            f"          {CAMS_TEMP_DIR}\n"
            f"  Run join_cams_tuktuk.py first to populate this directory."
        )
        return
    print(f"\n  Found {len(files)} CAMS file(s):\n")
    for f in files:
        print(f"    {f.name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each file ──────────────────────────────────────────────────────
    for i, filepath in enumerate(files, start=1):
        lat, lon = parse_node_coords(filepath)
        if lat is None:
            print(f"\n  [WARNING] Cannot parse coordinates from: {filepath.name} — skipped.")
            continue

        print(f"\n{'─' * 68}")
        print(f"  [{i}/{len(files)}] Node ({lat:.4f}°N, {lon:.4f}°E)")
        print(f"           File: {filepath.name}")

        # Parse
        try:
            df = parse_cams_csv(filepath)
        except Exception as exc:
            print(f"  [ERROR] Could not parse {filepath.name}: {exc}")
            continue

        print(f"  Parsed   : {len(df):,} records")
        if df.empty:
            print("  [WARNING] Empty dataset — skipped.")
            continue

        # Add derived columns
        df = add_sun_elevation(df, lat, lon)
        df = add_clear_sky_index(df)

        n_day = int((df["sun_elev"] >= MIN_SUN_ELEV_DEG).sum())
        print(f"  Daytime rows (sun_elev ≥ {MIN_SUN_ELEV_DEG}°): {n_day:,}")

        # Print a brief summary
        day = df[df["sun_elev"] >= MIN_SUN_ELEV_DEG]
        if not day.empty:
            print(
                f"  GHI       : mean {day['ghi'].mean():.1f} W/m²  |  "
                f"max {day['ghi'].max():.1f} W/m²"
            )
            print(
                f"  GHI_clear : mean {day['ghi_clear'].mean():.1f} W/m²  |  "
                f"max {day['ghi_clear'].max():.1f} W/m²"
            )
            if "kc" in day.columns:
                day_s = day[day["ghi"] >= MIN_GHI_WM2]
                if not day_s.empty:
                    print(
                        f"  kc        : mean {day_s['kc'].mean():.3f}  |  "
                        f"median {day_s['kc'].median():.3f}"
                    )

        # Build figure
        print("  Building figure ...")
        try:
            fig = make_node_figure(df, lat, lon, filepath)
        except Exception as exc:
            print(f"  [ERROR] Figure generation failed: {exc}")
            continue

        # Save
        out_name = f"cams_node_{lat:.4f}_{lon:.4f}.png"
        out_path = OUTPUT_DIR / out_name
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved    → {out_path}")

        if SHOW_FIGURE:
            plt.show()
        else:
            plt.close(fig)

    print(f"\n{sep}")
    print(f"  Done.  {len(files)} node(s) processed.")
    print(f"  Figures written to: {OUTPUT_DIR}")
    print(f"{sep}\n")


# ==============================================================================
if __name__ == "__main__":
    main()
