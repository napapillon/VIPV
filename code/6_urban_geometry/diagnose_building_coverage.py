"""
================================================================================
Building-Height Join Coverage by COS2023 Category
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Cross-tabulates the "no building found nearby" rate (n_buildings_nearby == 0
in the building-height buffer join) against the COS2023 level-4 category, to
test whether a high non-coverage rate is explained by genuinely
building-free categories (wide roads, green spaces, open land) rather than
by a data or radius problem.

cos23_n4_label (join_cos_standard.py / join_cos23_tuktuk.py) and
n_buildings_nearby (join_bh_standard.py / join_bh_tuktuk.py) are columns of
the same per-sensor file, so both values are read on the same row.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'cos23_n4_label', 'n_buildings_nearby'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             diagnose_building_coverage_<CAMPAIGN>_<variant_tag>.csv
                 full table: cos23_n4_label, n_points, pct_no_building
             diagnose_building_coverage_<CAMPAIGN>_<variant_tag>.png
                 top TOP_N_CATEGORIES categories by point count, coloured by
                 non-coverage rate (green = well covered, red = often no
                 building nearby), n annotated per bar
    The top-20 table is also printed to the console.

Requirements
------------
    pip install pandas matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.0
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import pandas as pd


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
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# Base output directory. A subdirectory named after CAMPAIGN / variant tag is
# created automatically (e.g. diag_bh_coverage_outputs/standard/sd_cleaned_lisbon/).
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "diag_bh_coverage_outputs"

# Number of top categories (by point count) shown in the bar chart.
TOP_N_CATEGORIES: int = 20

# Minimum points a category needs to appear in the bar chart at all
# (avoids single-digit-sample bars that are not informative).
MIN_POINTS_FOR_CHART: int = 30

DPI: int = 300
SHOW_FIGURES: bool = False


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor identifiers to their CSV input paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    sensors = ['j30', 'j31', 'j32', 'j33']
    if campaign == "standard":
        return {
            s: data_dir / f"{s}_satellite_landuse_v2_{suffix}.csv"
            for s in sensors
        }
    elif campaign == "tuktuk":
        sep = f"_{suffix}" if suffix else ""
        return {
            s: tuktuk_data_dir / f"{s}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv"
            for s in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output directory, embedding campaign + variant tag."""
    return base_dir / campaign / variant_tag


SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)


# ==============================================================================
# LOAD DATA
# ==============================================================================

def load_data(sensor_files: dict[str, Path]) -> pd.DataFrame:
    """
    Load cos23_n4_label and n_buildings_nearby from each per-sensor CSV and
    concatenate across sensors. Both columns already coexist on the same
    row for the same measurement — no alignment/merge needed.

    Parameters
    ----------
    sensor_files : dict[str, Path]
        Mapping of sensor id to its CSV path (from SENSOR_FILES).

    Returns
    -------
    pd.DataFrame
        Columns: cos23_n4_label, n_buildings_nearby, sensor.
    """
    frames = []
    for sensor_id, file_path in sensor_files.items():
        if not file_path.exists():
            print(f"  [WARNING] File not found — skipping {sensor_id}: {file_path.name}")
            continue
        df = pd.read_csv(
            file_path, low_memory=False,
            usecols=["cos23_n4_label", "n_buildings_nearby"],
        )
        df["sensor"] = sensor_id
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
            "FILE_SUFFIX/TUKTUK_FILE_SUFFIX. Note this diagnostic requires "
            "cos23_n4_label (join_cos_standard.py) and n_buildings_nearby "
            "(join_bh_standard.py) to already be present in the input file."
        )

    df = pd.concat(frames, ignore_index=True)

    for col in ("cos23_n4_label", "n_buildings_nearby"):
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found in the loaded data. "
                "Ensure the join_cos_standard.py / join_bh_standard.py "
                "(or their tuk-tuk equivalents) steps have already run."
            )

    return df


# ==============================================================================
# FIGURE
# ==============================================================================

def plot_coverage_bar_chart(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart of non-coverage rate by COS2023 category, for the
    TOP_N_CATEGORIES largest categories (by point count) with at least
    MIN_POINTS_FOR_CHART points. Bars are coloured on a green-to-red scale
    by pct_no_building (green = well covered, red = frequently no building).
    """
    plot_data = (
        summary[summary["n_points"] >= MIN_POINTS_FOR_CHART]
        .sort_values("n_points", ascending=False)
        .head(TOP_N_CATEGORIES)
        .sort_values("pct_no_building")
    )
    if plot_data.empty:
        print(f"  [warn] No category has >= {MIN_POINTS_FOR_CHART} points; "
              f"skipping bar chart.")
        return

    colors = plt.cm.RdYlGn_r(plot_data["pct_no_building"].values / 100.0)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(plot_data))))
    bars = ax.barh(plot_data["cos23_n4_label"], plot_data["pct_no_building"],
                    color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)

    for bar, (_, row) in zip(bars, plot_data.iterrows()):
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{row['pct_no_building']:.1f}%  (n={row['n_points']:,})",
                va="center", fontsize=8, color="#333333")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Rate of measurements with no building within the buffer [%]",
                  fontsize=10, fontweight="bold")
    ax.set_title(f"Building-Height Non-Coverage Rate by COS2023 Category\n"
                 f"(top {len(plot_data)} categories with ≥ {MIN_POINTS_FOR_CHART} points)",
                 fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.text(
        0.5, -0.01,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  Green = building usually found nearby. "
        "Red = frequently no building within the buffer radius — check whether "
        "this is a genuinely building-free category (roads, green space, water) "
        "or a coverage/radius problem.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved : {out_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the sensor files, tabulate the no-building rate per COS2023 category, save CSV and figure."""
    print("=" * 70)
    print("  Building-Height Coverage Diagnostic by COS2023 Category")
    print(f"  CAMPAIGN     : {CAMPAIGN}")
    print(f"  Data variant : {_VARIANT_TAG}")
    print("=" * 70)

    print("\nLoading sensor data...")
    combined = load_data(SENSOR_FILES)
    print(f"  {len(combined):,} rows loaded across "
          f"{combined['sensor'].nunique()} sensor(s)")

    combined["no_building_nearby"] = (combined["n_buildings_nearby"] == 0)

    print(f"\nTotal rows: {len(combined):,}")
    print(f"Overall 'no building nearby' rate: "
          f"{combined['no_building_nearby'].mean() * 100:.1f}%\n")

    summary = (
        combined.groupby("cos23_n4_label")
        .agg(n_points=("no_building_nearby", "size"),
             pct_no_building=("no_building_nearby", "mean"))
        .assign(pct_no_building=lambda d: (d["pct_no_building"] * 100).round(1))
        .sort_values("n_points", ascending=False)
        .reset_index()
    )

    print("Non-coverage rate by COS2023 category (sorted by sample size):")
    print(summary.head(20).to_string(index=False))

    # ── Outputs ────────────────────────────────────────────────────────────────
    print("\nWriting outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / f"diagnose_building_coverage_{CAMPAIGN}_{_VARIANT_TAG}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Saved : {csv_path}  ({len(summary)} categories, full table)")

    png_path = OUTPUT_DIR / f"diagnose_building_coverage_{CAMPAIGN}_{_VARIANT_TAG}.png"
    plot_coverage_bar_chart(summary, png_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
