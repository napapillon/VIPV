"""
================================================================================
Tuk-Tuk Telemetry — Exploration of the Unknown Solar Jinie Fields
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Exploratory analysis of the two undocumented fields of the tuk-tuk Solar
Jinie telemetry:

    s_raw   : undocumented. Hypotheses (not confirmed): an internal energy
              counter, or the battery state of charge in centipercent (the
              working hypothesis of tuktuk_telemetry_analysis.py)
    A1_raw  : undocumented. Hypothesis (not confirmed): the voltage on
              analog input 1 (solar charging panel)

  Figure 1 — Time series, one row per day, two columns (s_raw, A1_raw), with
             A0 in grey on a secondary axis: intra-day evolution of each field.
  Figure 2 — Relationship with other parameters: A0, ghi_fcul, voltage_V and
             temp_C vs s_raw and A1_raw (4 scatter subplots per field).

Local time is UTC + UTC_OFFSET_H (1 h, Portuguese summer time).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : TUKTUK_DIR / j33_tuktuk_ghi.csv  (join_ghi_fcul_tuktuk.py output)
             columns: 'timestamp', 'A0', 'voltage_V', 'temp_C', 's_raw',
             'A1_raw', 'ghi_fcul'
    output : OUTPUT_ROOT / tuktuk_exploration / unknown_fields /
             unknown_fields_timeseries.png
             unknown_fields_scatter_s_raw.png, unknown_fields_scatter_A1_raw.png

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
import os
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR: Path = TUKTUK_DIR
INPUT_FILE   = os.path.join(DATA_DIR, "j33_tuktuk_ghi.csv")
OUTPUT_DIR: Path = OUTPUT_ROOT / "tuktuk_exploration" / "unknown_fields"

UTC_OFFSET_H = 1   # Portugal summer time

# Colors
COLOR_S    = "#1565C0"   # blue  for s_raw
COLOR_A1   = "#E65100"   # orange for A1_raw
COLOR_REF  = "#2E7D32"   # green for reference variables

# ==============================================================================
# LOAD
# ==============================================================================

def load_data(filepath):
    """Load the tuk-tuk CSV, add local time and local date, sort by time."""
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_local"] = df["timestamp"] + pd.Timedelta(hours=UTC_OFFSET_H)
    df["date_local"] = df["timestamp_local"].dt.date
    df = df.sort_values("timestamp_local")
    return df

# ==============================================================================
# FIGURE 1 — Daily time series
# ==============================================================================

def plot_time_series(df, output_dir):
    """
    One row per day. Left column: s_raw. Right column: A1_raw.
    Both overlaid with A0 (grey, secondary axis) for reference.
    """
    days = sorted(df["date_local"].unique())
    n_days = len(days)

    fig, axes = plt.subplots(
        nrows=n_days, ncols=2,
        figsize=(14, 2.8 * n_days),
        sharex=False,
    )

    for i, day in enumerate(days):
        day_df = df[df["date_local"] == day].copy()

        for j, (field, color, label, ylim) in enumerate([
            ("s_raw",   COLOR_S,  "s_raw (unknown)",          (0, 9500)),
            ("A1_raw",  COLOR_A1, "A1_raw (mV, unknown)",     (1300, 1620)),
        ]):
            ax = axes[i][j]
            ax2 = ax.twinx()

            # A0 in background (grey, secondary axis)
            ax2.scatter(day_df["timestamp_local"], day_df["A0"],
                        color="lightgrey", s=4, zorder=1, label="A0")
            ax2.set_ylabel("A0 (W/m²)", fontsize=7, color="grey")
            ax2.tick_params(axis="y", labelcolor="grey", labelsize=7)
            ax2.set_ylim(0, 1400)

            # Main field
            ax.scatter(day_df["timestamp_local"], day_df[field],
                       color=color, s=6, zorder=3)
            ax.set_ylim(ylim)
            ax.set_ylabel(label, fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

            if j == 0:
                ax.set_ylabel(f"{day}\n{label}", fontsize=8)

    fig.suptitle("Unknown fields — daily time series (grey = A0 reference)",
                 fontsize=12, y=1.002)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "unknown_fields_timeseries.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ==============================================================================
# FIGURE 2 — Scatter vs other parameters
# ==============================================================================

def plot_scatter_vs_params(df, output_dir):
    """
    For each unknown field, 4 scatter subplots against:
    A0, ghi_fcul, voltage_V, temp_C.
    """
    ref_params = [
        ("A0",         "A0 (W/m²)"),
        ("ghi_fcul",   "GHI FCUL (W/m²)"),
        ("voltage_V",  "Battery voltage (V)"),
        ("temp_C",     "Temperature (°C)"),
    ]

    for field, color, field_label in [
        ("s_raw",  COLOR_S,  "s_raw"),
        ("A1_raw", COLOR_A1, "A1_raw (mV)"),
    ]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for ax, (param, param_label) in zip(axes, ref_params):
            ax.scatter(df[param], df[field],
                       color=color, s=5, alpha=0.4, linewidths=0)

            # Linear regression
            valid = df[[param, field]].dropna()
            if len(valid) > 10:
                coeffs = np.polyfit(valid[param], valid[field], 1)
                x_line = np.linspace(valid[param].min(), valid[param].max(), 100)
                ax.plot(x_line, np.polyval(coeffs, x_line),
                        color="black", linewidth=1, linestyle="--")
                r = valid[param].corr(valid[field])
                ax.set_title(f"r = {r:.3f}", fontsize=9)

            ax.set_xlabel(param_label, fontsize=9)
            ax.set_ylabel(field_label, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

        fig.suptitle(f"{field_label} — scatter vs other parameters", fontsize=12)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"unknown_fields_scatter_{field}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Load the data and draw the time-series and scatter figures."""
    print(f"Loading: {INPUT_FILE}")
    df = load_data(INPUT_FILE)
    print(f"  {len(df)} measurements, {df['date_local'].nunique()} days")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFigure 1 — time series...")
    plot_time_series(df, OUTPUT_DIR)

    print("Figure 2 — scatter vs parameters...")
    plot_scatter_vs_params(df, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
