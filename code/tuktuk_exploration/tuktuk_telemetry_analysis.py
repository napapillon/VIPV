"""
================================================================================
Tuk-Tuk Telemetry — Temperature, Battery Voltage and State of Charge
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Visualisation of the onboard telemetry fields of the J33 tuk-tuk campaign:

    temp_C     : sensor temperature (°C)         — decoded from raw ÷ 100
    voltage_V  : battery voltage (V)              — decoded from raw mV ÷ 1000
    s_raw      : undocumented field, read here as the battery state of charge
                 (SoC) in centipercent (0–10 000 ≈ 0–100 %),
                 SoC_pct = s_raw / SOC_SCALE — a working hypothesis, not a
                 confirmed encoding (see below)

The unit conversions of temp_C and voltage_V are applied by decode_tuktuk.py.

Figures produced
----------------
  Fig 1 — Full-campaign time series, 3 stacked panels (one per field),
          shared x-axis in local time.
  Fig 2 — Per-day facet grid: one row per field, one column per day, for
          comparing intra-day patterns across the campaign.
  Fig 3 — Distributions (histogram + KDE), one subplot per field.
  Fig 4 — Pair plot of temp_C, voltage_V and SoC_pct (3×3, marginal
          histograms, Pearson r).
  Fig 5 — Voltage vs SoC, coloured by temperature.
  Fig 6 — Daily box plots of each field.

Assumptions / caveats
---------------------
  - SoC = s_raw / 100 (%). Consistent with the observed range (565–8 763 →
    ~6–88 %) and with the positive correlation with voltage_V (r ≈ 0.68); to
    be checked against the Solar Jinie firmware documentation when available.
  - Local time = UTC + UTC_OFFSET_H (1 h, Portuguese summer time, WEST).
  - The voltage range 3.47–4.11 V is consistent with a single-cell Li-ion /
    LiPo battery (nominal 3.0–4.2 V).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : TUKTUK_DIR / j33_tuktuk.csv  (decode_tuktuk.py output)
             columns: 'timestamp', 'temp_C', 'voltage_V', 's_raw'
    output : OUTPUT_ROOT / tuktuk_exploration / telemetry /
             fig1_campaign_timeseries.png, fig2_daily_facets.png,
             fig3_distributions.png, fig4_pairplot.png,
             fig5_voltage_soc_temperature.png, fig6_daily_boxplots.png

Requirements
------------
    pip install pandas numpy matplotlib scipy

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR   : Path = TUKTUK_DIR
OUTPUT_DIR : Path = OUTPUT_ROOT / "tuktuk_exploration" / "telemetry"

# Input file (output of decode_tuktuk.py)
INPUT_FILE = DATA_DIR / "j33_tuktuk.csv"

# Portugal Summer Time offset (UTC+1)
UTC_OFFSET_H = 1

# SoC conversion factor: s_raw / SOC_SCALE = SoC in %
# Based on observed range 565–8 763 mapping to ~6–88 % when divided by 100.
# Update if firmware documentation clarifies the encoding.
SOC_SCALE = 100.0

# Color palette — consistent with project conventions
COLOR_TEMP    = "#E05C5C"   # red   — temperature
COLOR_VOLTAGE = "#1565C0"   # blue  — voltage
COLOR_SOC     = "#2E7D32"   # green — state of charge

# Figure resolution (dpi)
FIG_DPI = 150

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load the decoded J33 tuk-tuk CSV and add derived columns.

    Parameters
    ----------
    filepath : Path
        Path to j33_tuktuk.csv (output of decode_tuktuk.py).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp (UTC), timestamp_local, date,
        temp_C, voltage_V, SoC_pct, A0.
    """
    df = pd.read_csv(filepath)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_local"] = df["timestamp"] + pd.Timedelta(hours=UTC_OFFSET_H)
    df["date"] = df["timestamp_local"].dt.date

    # Derive SoC in % from s_raw
    df["SoC_pct"] = df["s_raw"] / SOC_SCALE

    # Sort chronologically
    df = df.sort_values("timestamp_local").reset_index(drop=True)

    return df


# ==============================================================================
# FIGURE 1 — Full campaign time series
# ==============================================================================

def plot_campaign_timeseries(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Three stacked subplots (temp_C, voltage_V, SoC_pct) over the full
    campaign duration. X-axis is shared local time.
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"hspace": 0.08}
    )

    fields = [
        ("temp_C",    "Temperature (°C)",         COLOR_TEMP,    0.7),
        ("voltage_V", "Battery Voltage (V)",       COLOR_VOLTAGE, 0.7),
        ("SoC_pct",   "State of Charge (%)",       COLOR_SOC,     0.7),
    ]

    for ax, (col, ylabel, color, alpha) in zip(axes, fields):
        valid = df[col].notna()
        ax.plot(
            df.loc[valid, "timestamp_local"],
            df.loc[valid, col],
            color=color, linewidth=0.6, alpha=alpha
        )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.tick_params(axis="y", labelsize=9)

        # Annotate mean
        mean_val = df.loc[valid, col].mean()
        ax.axhline(mean_val, color=color, linewidth=0.9, linestyle=":",
                   alpha=0.8, label=f"Mean = {mean_val:.2f}")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    # Format x-axis on the bottom panel
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    axes[-1].set_xlabel("Date (local time, UTC+1)", fontsize=10)

    fig.suptitle("J33 Tuk-Tuk — Onboard Telemetry: Full Campaign", fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"SoC interpreted as s_raw ÷ {int(SOC_SCALE)} — pending firmware confirmation  |  "
        f"UTC+1 (Portugal Summer Time)  |  n = {len(df):,} measurements",
        ha="center", fontsize=7, style="italic", color="grey"
    )

    out_path = output_dir / "fig1_campaign_timeseries.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ==============================================================================
# FIGURE 2 — Per-day facet grid
# ==============================================================================

def plot_daily_facets(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Grid of line charts: rows = fields (temp, voltage, SoC), columns = days.
    Allows intra-day shape comparison across the campaign.
    """
    days = sorted(df["date"].unique())
    n_days = len(days)

    fields = [
        ("temp_C",    "Temp (°C)",     COLOR_TEMP),
        ("voltage_V", "Voltage (V)",   COLOR_VOLTAGE),
        ("SoC_pct",   "SoC (%)",       COLOR_SOC),
    ]
    n_fields = len(fields)

    fig, axes = plt.subplots(
        n_fields, n_days,
        figsize=(2.5 * n_days, 2.8 * n_fields),
        sharey="row"
    )

    # Ensure 2D indexing even with a single day
    if n_days == 1:
        axes = axes.reshape(n_fields, 1)

    for col_idx, day in enumerate(days):
        day_df = df[df["date"] == day]

        for row_idx, (field, ylabel, color) in enumerate(fields):
            ax = axes[row_idx, col_idx]
            valid = day_df[field].notna()
            ax.plot(
                day_df.loc[valid, "timestamp_local"],
                day_df.loc[valid, field],
                color=color, linewidth=0.9
            )

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[6, 9, 12, 15, 18]))
            ax.tick_params(axis="x", labelsize=6, rotation=45)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

            # Column header: day label on top row only
            if row_idx == 0:
                ax.set_title(str(day), fontsize=7, fontweight="bold")

            # Row ylabel on left column only
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=8)

    fig.suptitle("J33 Tuk-Tuk — Daily Telemetry Profiles", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = output_dir / "fig2_daily_facets.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ==============================================================================
# FIGURE 3 — Distributions (histogram + KDE)
# ==============================================================================

def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    One subplot per field: histogram with overlaid KDE, median and mean
    annotated as vertical lines.
    """
    fields = [
        ("temp_C",    "Temperature (°C)",    COLOR_TEMP,    np.arange(20, 80, 1)),
        ("voltage_V", "Battery Voltage (V)", COLOR_VOLTAGE, np.arange(3.45, 4.15, 0.01)),
        ("SoC_pct",   "State of Charge (%)", COLOR_SOC,     np.arange(0, 100, 1)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.3)

    for ax, (col, xlabel, color, bins) in zip(axes, fields):
        values = df[col].dropna().values

        # Histogram (density=True to normalise)
        ax.hist(values, bins=bins, color=color, alpha=0.4, density=True, label="Histogram")

        # KDE
        try:
            kde = gaussian_kde(values, bw_method="scott")
            x_range = np.linspace(values.min(), values.max(), 400)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2.0, label="KDE")
        except Exception:
            pass

        # Mean and median
        mean_v   = values.mean()
        median_v = np.median(values)
        ax.axvline(mean_v,   color=color,   linewidth=1.4, linestyle="--",
                   label=f"Mean = {mean_v:.2f}")
        ax.axvline(median_v, color="black", linewidth=1.0, linestyle=":",
                   label=f"Median = {median_v:.2f}")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.tick_params(labelsize=9)

    fig.suptitle("J33 Tuk-Tuk — Telemetry Distributions", fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.02,
        f"n = {len(df):,} measurements  |  campaign 2026-06-29 → 2026-07-13",
        ha="center", fontsize=8, style="italic", color="grey"
    )

    out_path = output_dir / "fig3_distributions.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ==============================================================================
# FIGURE 4 — Pair-plot scatter matrix
# ==============================================================================

def plot_pairplot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    3×3 scatter matrix for temp_C, voltage_V, SoC_pct.
    Diagonal: histograms.  Off-diagonal: scatter + Pearson r annotation.
    """
    cols   = ["temp_C", "voltage_V", "SoC_pct"]
    labels = ["Temperature (°C)", "Voltage (V)", "SoC (%)"]
    colors_diag = [COLOR_TEMP, COLOR_VOLTAGE, COLOR_SOC]
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(10, 9))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    sub = df[cols].dropna()

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            if row == col:
                # Diagonal: histogram
                ax.hist(sub[cols[row]], bins=40, color=colors_diag[row],
                        alpha=0.6, density=True)
                ax.set_yticks([])
            else:
                # Off-diagonal: scatter
                ax.scatter(
                    sub[cols[col]], sub[cols[row]],
                    s=2, alpha=0.25, color="#555555", rasterized=True
                )
                # Pearson r
                r = sub[cols[col]].corr(sub[cols[row]])
                ax.text(
                    0.05, 0.92, f"r = {r:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    color="navy", fontweight="bold"
                )

            # Labels on outer edges only
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=8)
            else:
                ax.set_yticklabels([])
            if row == n - 1:
                ax.set_xlabel(labels[col], fontsize=8)
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    fig.suptitle("J33 Tuk-Tuk — Telemetry Cross-Correlation Matrix",
                 fontsize=12, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"n = {len(sub):,} valid rows  |  Pearson r shown on off-diagonal panels",
        ha="center", fontsize=8, style="italic", color="grey"
    )

    out_path = output_dir / "fig4_pairplot.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ==============================================================================
# FIGURE 5 — Voltage vs SoC coloured by temperature
# ==============================================================================

def plot_voltage_soc_temp(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter: x = SoC (%), y = voltage (V), colour = temperature (°C).
    Reveals the thermal dependence of the voltage–SoC relationship.
    """
    sub = df[["SoC_pct", "voltage_V", "temp_C"]].dropna()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    norm = Normalize(vmin=sub["temp_C"].quantile(0.02),
                     vmax=sub["temp_C"].quantile(0.98))
    cmap = plt.get_cmap("RdYlBu_r")

    ax.scatter(
        sub["SoC_pct"], sub["voltage_V"],
        c=sub["temp_C"], cmap=cmap, norm=norm,
        s=5, alpha=0.5, rasterized=True
    )

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("State of Charge (%)", fontsize=11)
    ax.set_ylabel("Battery Voltage (V)", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=10)

    # Annotate Pearson r
    r = sub["SoC_pct"].corr(sub["voltage_V"])
    ax.text(0.03, 0.94, f"r(SoC, V) = {r:.3f}",
            transform=ax.transAxes, fontsize=10, color="navy", fontweight="bold")

    fig.suptitle("J33 Tuk-Tuk — Battery Voltage vs State of Charge (coloured by Temperature)",
                 fontsize=12, fontweight="bold")
    fig.text(
        0.5, -0.02,
        f"n = {len(sub):,} measurements  |  "
        f"SoC = s_raw ÷ {int(SOC_SCALE)} — pending firmware confirmation",
        ha="center", fontsize=8, style="italic", color="grey"
    )

    out_path = output_dir / "fig5_voltage_soc_temperature.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ==============================================================================
# FIGURE 6 — Per-day box plots
# ==============================================================================

def plot_daily_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Three stacked sub-panels, each showing box plots per calendar day.
    Summarises day-to-day variability in each telemetry channel.
    """
    days = [str(d) for d in sorted(df["date"].unique())]
    day_str = [str(d) for d in df["date"]]

    fields = [
        ("temp_C",    "Temperature (°C)",    COLOR_TEMP),
        ("voltage_V", "Battery Voltage (V)", COLOR_VOLTAGE),
        ("SoC_pct",   "State of Charge (%)", COLOR_SOC),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"hspace": 0.10})

    for ax, (col, ylabel, color) in zip(axes, fields):
        data_by_day = [
            df.loc[day_str_i == d, col].dropna().values
            for d, day_str_i in [(d, np.array(day_str)) for d in days]
        ]

        bp = ax.boxplot(
            data_by_day,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker=".", color=color, alpha=0.3, markersize=3),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.tick_params(axis="y", labelsize=9)

    # Shared x-axis labels (day labels)
    axes[-1].set_xticks(range(1, len(days) + 1))
    axes[-1].set_xticklabels(days, rotation=30, ha="right", fontsize=8)
    axes[-1].set_xlabel("Date (local)", fontsize=10)

    fig.suptitle("J33 Tuk-Tuk — Daily Telemetry Box Plots", fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        "Each box: IQR; whiskers: 1.5×IQR; dots: outliers",
        ha="center", fontsize=8, style="italic", color="grey"
    )

    out_path = output_dir / "fig6_daily_boxplots.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Load the decoded tuk-tuk data and draw Figures 1–6."""
    print("=" * 60)
    print("tuktuk_telemetry_analysis.py")
    print("=" * 60)

    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    df = load_data(INPUT_FILE)
    print(f"  {len(df):,} valid measurements loaded")
    print(f"  Date range (local): {df['date'].min()} → {df['date'].max()}")
    print(f"  Days with data    : {df['date'].nunique()}")
    print()

    # Print summary statistics
    print("Summary statistics:")
    summary = df[["temp_C", "voltage_V", "SoC_pct"]].describe().round(3)
    summary.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    print(summary.to_string())
    print()

    # Generate figures
    print("Generating figures...")
    plot_campaign_timeseries(df, OUTPUT_DIR)
    plot_daily_facets(df, OUTPUT_DIR)
    plot_distributions(df, OUTPUT_DIR)
    plot_pairplot(df, OUTPUT_DIR)
    plot_voltage_soc_temp(df, OUTPUT_DIR)
    plot_daily_boxplots(df, OUTPUT_DIR)

    print()
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
