"""
================================================================================
Daily Irradiance Curves — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Plots, for each day of the tuk-tuk campaign, the daily irradiance curve
("bell") of the onboard sensor against the FCUL pyranometer:

    A0        : raw sensor irradiance [W/m²] (sensor tilted at 12°, not
                tilt-corrected), as points
    ghi_fcul  : GHI of the FCUL rooftop pyranometer [W/m²], horizontal
                reference, as a line

Times are shown in local Portuguese summer time (UTC+1, UTC_OFFSET_H). The
Y axis is fixed (Y_MIN, Y_MAX) so that days can be compared.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <TUKTUK_BASE><INPUT_SUFFIX>.csv in TUKTUK_DIR, e.g.
             j33_tuktuk_ghi.csv (output of join_ghi_fcul_tuktuk.py) or any
             later stage of the tuk-tuk join chain
    output : OUTPUT_DIR / daily_bells / irradiance_<YYYY-MM-DD>.png
             (one figure per day with data)

Requirements
------------
    pip install pandas matplotlib

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


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input file ---------------------------------------------------------------
DATA_DIR: Path = TUKTUK_DIR

TUKTUK_BASE  = "j33_tuktuk"
INPUT_SUFFIX = "_ghi"
INPUT_FILE   = os.path.join(DATA_DIR, f"{TUKTUK_BASE}{INPUT_SUFFIX}.csv")

# --- Output directory ---------------------------------------------------------
OUTPUT_DIR: Path = OUTPUT_ROOT / "3_irradiance" / "daily_irradiance_bells"

# --- Display ------------------------------------------------------------------
UTC_OFFSET_H = 1          # local time offset (Portugal summer time = UTC+1)

Y_MIN = 0                 # fixed Y axis for all figures [W/m²]
Y_MAX = 1400

COLOR_A0       = "#E05C5C"   # sensor (red)
COLOR_GHI_FCUL = "#1565C0"   # pyranometer (blue)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Plot one irradiance figure per day of data."""
    # --- Load data ---
    print(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Convert to local time for display (UTC+1 in summer)
    df["timestamp_local"] = df["timestamp"] + pd.Timedelta(hours=UTC_OFFSET_H)
    df["date_local"] = df["timestamp_local"].dt.date

    print(f"  {len(df)} measurements across {df['date_local'].nunique()} days")

    # --- Output directory ---
    fig_dir = os.path.join(OUTPUT_DIR, "daily_bells")
    os.makedirs(fig_dir, exist_ok=True)

    # --- One figure per day ---
    days = sorted(df["date_local"].unique())

    for day in days:
        day_df = df[df["date_local"] == day].copy()
        day_df = day_df.sort_values("timestamp_local")

        n_pts = len(day_df)
        print(f"  Plotting {day}  ({n_pts} measurements)")

        fig, ax = plt.subplots(figsize=(10, 5))

        # --- ghi_fcul: pyranometer reference as a line ---
        ax.plot(
            day_df["timestamp_local"],
            day_df["ghi_fcul"],
            color=COLOR_GHI_FCUL,
            linewidth=1.2,
            label="GHI – FCUL pyranometer",
            zorder=2,
        )

        # --- A0: sensor measurements as points ---
        ax.scatter(
            day_df["timestamp_local"],
            day_df["A0"],
            color=COLOR_A0,
            s=8,
            linewidths=0,
            label="A0 – Jinie33 sensor (raw, 12° tilt)",
            zorder=3,
        )

        # --- Formatting ---
        ax.set_xlim(
            day_df["timestamp_local"].min().replace(hour=6,  minute=0, second=0),
            day_df["timestamp_local"].max().replace(hour=21, minute=0, second=0),
        )
        ax.set_ylim(Y_MIN, Y_MAX)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=45)

        ax.set_xlabel("Local time (UTC+1)", fontsize=11)
        ax.set_ylabel("Irradiance (W/m²)", fontsize=11)
        ax.set_title(f"Daily irradiance — {day}", fontsize=13)

        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        fig.text(
            0.5, 0.01,
            f"Input: {TUKTUK_BASE}{INPUT_SUFFIX}.csv  |  n={n_pts} measurements  |  A0 not tilt-corrected",
            ha="center", fontsize=7, style="italic", color="grey",
        )

        fig.tight_layout(rect=[0, 0.03, 1, 1])

        # --- Save ---
        out_path = os.path.join(fig_dir, f"irradiance_{day}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_path}")

    print(f"\nDone. {len(days)} figures written to: {fig_dir}")


if __name__ == "__main__":
    main()
