"""
================================================================================
COS2023 Urban Breakdown — Land-Use Category Analysis
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Four exploratory figures breaking down the VIPV sensor measurements by
COS2023 land-use category (DGT):

  1. Distribution of measurements by COS2023 level-4 category
  2. Distribution of measurements by COS2023 level-1 branch
  3. Relative autonomy loss (%) by level-4 category (n annotated per bar)
  4. Diffuse ratio k_d = DHI / GHI distribution by level-4 category

Categories with fewer than MIN_FIG_ZONE_POINTS_COS23 points are left out of
the figures and counted in a footnote.

Method
------
Autonomy / loss metrics use energy-weighted aggregation (ratio of sums), as
in solar_autonomy_losses.py:
    E_solar_kWh   = rad_calib [W/m²] / 1000 × Δt_h
    loss_irr_kWh  = max(0, GHI − rad_calib) / 1000 × Δt_h
    auto_solar_km = E_solar_kWh  / CVE_kWh_per_km
    loss_auto_km  = loss_irr_kWh / CVE_kWh_per_km
    loss_pct (per category) = Σ loss_auto_km / (Σ auto_solar_km + Σ loss_auto_km) × 100

The sensor-prefixed columns ('<SENSOR> Rad calib', '<SENSOR> lat',
'<SENSOR> lon') are renamed to generic rad_calib / lat / lon so that all
sensors can be pooled.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> Rad calib', '<SENSOR> lat', '<SENSOR> lon',
             'ghi', 'dhi', 'cos23_n4_code', 'cos23_n4_label', 'cos23_n1_label'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             fig1_distribution_level4_<CAMPAIGN>_<variant_tag>.png
             fig2_distribution_level1_<CAMPAIGN>_<variant_tag>.png
             fig3_losses_by_category_<CAMPAIGN>_<variant_tag>.png
             fig4_kd_by_category_<CAMPAIGN>_<variant_tag>.png

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.2
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
import numpy as np
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
# created automatically.
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "5_land_cover" / "cos23_urban_outputs"

# Minimum number of measurement points required for a COS2023 category to
# appear in figures. Categories below this threshold are excluded and counted
# in the footnote.
MIN_FIG_ZONE_POINTS_COS23: int = 600

# Physical / vehicle parameters
CVE_KWH_PER_100KM: float = 16.6          # consumption per 100 km [kWh/100 km]
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0
DELTA_T_H:         float = 1.0 / 60.0    # time step [h]  (1-minute data)

DPI:          int  = 300
SHOW_FIGURES: bool = False

# ==============================================================================
# DERIVED PATHS AND LABELS  (resolved from CAMPAIGN and the input suffix)
# ==============================================================================

def _build_subtitle(campaign: str, variant_tag: str) -> str:
    """Return the figure subtitle identifying the campaign and variant."""
    if campaign == "standard":
        return f"Standard campaign — variant: {variant_tag}"
    return f"Tuk-tuk campaign — variant: {variant_tag}"


def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        file_suffix: str, tuktuk_file_suffix: str) -> dict:
    """Build the dict mapping sensor identifiers to their CSV input paths."""
    sensors = ['j30', 'j31', 'j32', 'j33']
    if campaign == "standard":
        return {
            s: data_dir / f"{s}_satellite_landuse_v2_{file_suffix}.csv"
            for s in sensors
        }
    elif campaign == "tuktuk":
        sep = f"_{tuktuk_file_suffix}" if tuktuk_file_suffix else ""
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
    """Return the output subdirectory, embedding campaign + variant tag."""
    return base_dir / campaign / variant_tag


def _build_output_path(out_dir: Path, base_name: str,
                       campaign: str, variant_tag: str) -> Path:
    """Build a figure output path embedding the active campaign and variant tag."""
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.png"


SENSOR_FILES: dict = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, FILE_SUFFIX, TUKTUK_FILE_SUFFIX
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR: Path  = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
FIG_SUBTITLE: str = _build_subtitle(CAMPAIGN, _VARIANT_TAG)

# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

def load_and_prepare(sensor_files: dict) -> pd.DataFrame:
    """
    Load per-sensor CSV files, consolidate sensor-prefixed columns (e.g.
    "J31 Rad calib", "J31 lat") into generic rad_calib / lat / lon, and
    compute autonomy / loss / diffuse-ratio metrics.
    """
    frames = []
    for sensor_id, file_path in sensor_files.items():
        tag = sensor_id.upper()
        if not file_path.exists():
            print(f"  [WARNING] File not found — skipping {tag}: {file_path.name}")
            continue

        df = pd.read_csv(file_path, low_memory=False)
        print(f"  {tag}: {len(df):,} rows read from '{file_path.name}'")

        rename_map = {
            f"{tag} Rad calib": "rad_calib",
            f"{tag} lat":       "lat",
            f"{tag} lon":       "lon",
        }
        missing = [c for c in rename_map if c not in df.columns]
        if missing:
            print(f"  [WARNING] {tag}: missing columns {missing} — skipping.")
            continue
        df = df.rename(columns=rename_map)
        df["sensor"] = sensor_id
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
            "FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    df = pd.concat(frames, ignore_index=True)

    # Energy metrics — energy-weighted aggregation (ratio of sums).
    diff_w = df["ghi"] - df["rad_calib"]
    df["e_solar_kWh"]  = df["rad_calib"] / 1000.0 * DELTA_T_H
    df["loss_irr_kWh"] = (diff_w / 1000.0 * DELTA_T_H).clip(lower=0.0)
    df["auto_solar_km"] = df["e_solar_kWh"]  / CVE_KWH_PER_KM
    df["loss_auto_km"]  = df["loss_irr_kWh"] / CVE_KWH_PER_KM

    # Diffuse fraction k_d = DHI / GHI (undefined when GHI = 0).
    df["k_d"] = np.nan
    mask_kd = df["ghi"] > 0
    df.loc[mask_kd, "k_d"] = df.loc[mask_kd, "dhi"] / df.loc[mask_kd, "ghi"]

    return df


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def categories_above_threshold(
    df: pd.DataFrame,
    group_col: str,
    threshold: int,
) -> tuple[pd.Series, int, int]:
    """
    Split categories in `group_col` by the minimum point-count threshold.

    Returns (counts_above_threshold sorted descending, n_categories_excluded,
    n_points_excluded).
    """
    counts   = df.groupby(group_col).size().sort_values(ascending=False)
    kept     = counts[counts >= threshold]
    excluded = counts[counts <  threshold]
    return kept, int(len(excluded)), int(excluded.sum())


def build_color_map(categories) -> dict:
    """Assign a stable colour per category (tab20) so it matches across figures."""
    cmap = plt.get_cmap("tab20")
    return {cat: cmap(i % 20) for i, cat in enumerate(categories)}


def _save_and_show(fig: plt.Figure, out_path: Path) -> None:
    """Save figure to disk at DPI resolution and optionally display it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  saved: {out_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ==============================================================================
# FIG 1  —  Distribution of measurements by COS2023 level-4 category
# ==============================================================================

def plot_fig1_distribution_level4(df: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart of measurement counts per COS2023 level-4 category.
    Categories below MIN_FIG_ZONE_POINTS_COS23 are excluded and footnoted.
    """
    counts, n_excl_cat, n_excl_pts = categories_above_threshold(
        df, "cos23_n4_label", MIN_FIG_ZONE_POINTS_COS23
    )
    colors = build_color_map(counts.index)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(counts))))
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=[colors[c] for c in counts.index])
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.invert_yaxis()
    for yi, v in zip(y, counts.values):
        ax.text(v, yi, f"  {v:,}", va="center", fontsize=7.5, color="#333333")
    ax.set_xlabel("Number of measurements", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Distribution of Measurements by COS2023 Category (level 4)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.text(
        0.5, 1.04, FIG_SUBTITLE,
        transform=ax.transAxes, ha="center", fontsize=9, color="#555555",
    )
    fig.text(
        0.5, -0.01,
        f"[COS2023, threshold ≥ {MIN_FIG_ZONE_POINTS_COS23} pts]  —  "
        f"{n_excl_cat} categor{'y' if n_excl_cat == 1 else 'ies'} excluded "
        f"({n_excl_pts:,} pts)  —  campaign: {CAMPAIGN} / {_VARIANT_TAG}",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 2  —  Distribution of measurements by COS2023 level-1 branch
# ==============================================================================

def plot_fig2_distribution_level1(df: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart of measurement counts per COS2023 level-1 branch,
    with each bar annotated with its percentage of the total.
    """
    counts, n_excl_cat, n_excl_pts = categories_above_threshold(
        df, "cos23_n1_label", MIN_FIG_ZONE_POINTS_COS23
    )
    colors = build_color_map(counts.index)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(counts))))
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=[colors[c] for c in counts.index])
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.invert_yaxis()
    for yi, v in zip(y, counts.values):
        pct = v / len(df) * 100.0
        ax.text(
            v, yi, f"  {v:,} ({pct:.1f}%)",
            va="center", fontsize=8, color="#333333",
        )
    ax.set_xlabel("Number of measurements", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Distribution of Measurements by COS2023 Branch (level 1)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.text(
        0.5, 1.04, FIG_SUBTITLE,
        transform=ax.transAxes, ha="center", fontsize=9, color="#555555",
    )
    fig.text(
        0.5, -0.01,
        f"[COS2023, threshold ≥ {MIN_FIG_ZONE_POINTS_COS23} pts]  —  "
        f"{n_excl_cat} branch{'es' if n_excl_cat != 1 else ''} excluded "
        f"({n_excl_pts:,} pts)  —  campaign: {CAMPAIGN} / {_VARIANT_TAG}",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 3  —  Relative autonomy loss by COS2023 level-4 category
# ==============================================================================

def plot_fig3_losses_by_category(df: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart of relative autonomy loss (%) per COS2023 level-4
    category (energy-weighted ratio of sums). Sample size n is annotated on
    every bar, since categories with very different n are not directly
    comparable.
    """
    counts, n_excl_cat, n_excl_pts = categories_above_threshold(
        df, "cos23_n4_label", MIN_FIG_ZONE_POINTS_COS23
    )

    rows = []
    for cat in counts.index:
        sub      = df[df["cos23_n4_label"] == cat]
        auto_sum = sub["auto_solar_km"].sum()
        loss_sum = sub["loss_auto_km"].sum()
        denom    = auto_sum + loss_sum
        loss_pct = (loss_sum / denom * 100.0) if denom > 0 else np.nan
        rows.append({"category": cat, "n": len(sub), "loss_pct": loss_pct})

    plot_df = (
        pd.DataFrame(rows)
        .dropna(subset=["loss_pct"])
        .sort_values("loss_pct", ascending=False)
    )
    colors = build_color_map(plot_df["category"])

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["loss_pct"],
            color=[colors[c] for c in plot_df["category"]])
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["category"], fontsize=8)
    ax.invert_yaxis()
    for yi, (loss_pct, n) in enumerate(
        zip(plot_df["loss_pct"], plot_df["n"])
    ):
        ax.text(
            loss_pct, yi, f"  {loss_pct:.1f}%  (n={n:,})",
            va="center", fontsize=7.5, color="#333333",
        )
    ax.set_xlabel("Relative Autonomy Loss [%]", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Relative Autonomy Loss by COS2023 Category (level 4)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.text(
        0.5, 1.04, FIG_SUBTITLE,
        transform=ax.transAxes, ha="center", fontsize=9, color="#555555",
    )
    fig.text(
        0.5, -0.01,
        f"[COS2023, threshold ≥ {MIN_FIG_ZONE_POINTS_COS23} pts]  —  "
        f"{n_excl_cat} categor{'y' if n_excl_cat == 1 else 'ies'} excluded "
        f"({n_excl_pts:,} pts)  —  "
        f"loss_pct = Σ loss_auto / (Σ auto_solar + Σ loss_auto) × 100  —  "
        f"campaign: {CAMPAIGN} / {_VARIANT_TAG}",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 4  —  Diffuse ratio k_d distribution by COS2023 level-4 category
# ==============================================================================

def plot_fig4_kd_by_category(df: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal box plot of the diffuse ratio k_d = DHI / GHI across COS2023
    level-4 categories, ordered by decreasing median. Outliers hidden.
    """
    counts, n_excl_cat, n_excl_pts = categories_above_threshold(
        df, "cos23_n4_label", MIN_FIG_ZONE_POINTS_COS23
    )

    medians = (
        df[df["cos23_n4_label"].isin(counts.index)]
        .groupby("cos23_n4_label")["k_d"]
        .median()
        .sort_values(ascending=False)
    )
    order        = medians.index
    data_per_cat = [
        df.loc[df["cos23_n4_label"] == cat, "k_d"].dropna().values
        for cat in order
    ]
    colors = build_color_map(order)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(order))))
    bp = ax.boxplot(
        data_per_cat, vert=False, patch_artist=True,
        showfliers=False, labels=list(order),
    )
    for patch, cat in zip(bp["boxes"], order):
        patch.set_facecolor(colors[cat])
        patch.set_alpha(0.85)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel(
        "Diffuse ratio  k_d = DHI / GHI  [—]",
        fontsize=10, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Diffuse Ratio (k_d) Distribution by COS2023 Category (level 4)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.text(
        0.5, 1.04, FIG_SUBTITLE,
        transform=ax.transAxes, ha="center", fontsize=9, color="#555555",
    )
    fig.text(
        0.5, -0.01,
        f"[COS2023, threshold ≥ {MIN_FIG_ZONE_POINTS_COS23} pts]  —  "
        f"{n_excl_cat} categor{'y' if n_excl_cat == 1 else 'ies'} excluded "
        f"({n_excl_pts:,} pts)  —  outliers not shown  —  campaign: {CAMPAIGN} / {_VARIANT_TAG}",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the sensor files and produce the four COS2023 breakdown figures."""
    print("=" * 78)
    print(f"COS2023 urban breakdown  —  CAMPAIGN = {CAMPAIGN!r}")
    print(f"  CAMPAIGN    : {CAMPAIGN}")
    if CAMPAIGN == "standard":
        print(f"  FILE_SUFFIX : {FILE_SUFFIX!r}")
    else:
        print(f"  TUKTUK_FILE_SUFFIX : {TUKTUK_FILE_SUFFIX!r}")
    print(f"  OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"  threshold   : {MIN_FIG_ZONE_POINTS_COS23} pts")
    print("=" * 78)

    print("\nLoading and preparing data...")
    df = load_and_prepare(SENSOR_FILES)
    print(f"  Total valid rows: {len(df):,}")

    print("\n[Fig 1] Distribution by level-4 category...")
    plot_fig1_distribution_level4(
        df,
        _build_output_path(OUTPUT_DIR, "fig1_distribution_level4", CAMPAIGN, _VARIANT_TAG),
    )

    print("\n[Fig 2] Distribution by level-1 branch...")
    plot_fig2_distribution_level1(
        df,
        _build_output_path(OUTPUT_DIR, "fig2_distribution_level1", CAMPAIGN, _VARIANT_TAG),
    )

    print("\n[Fig 3] Relative autonomy loss by category...")
    plot_fig3_losses_by_category(
        df,
        _build_output_path(OUTPUT_DIR, "fig3_losses_by_category", CAMPAIGN, _VARIANT_TAG),
    )

    print("\n[Fig 4] Diffuse ratio (k_d) by category...")
    plot_fig4_kd_by_category(
        df,
        _build_output_path(OUTPUT_DIR, "fig4_kd_by_category", CAMPAIGN, _VARIANT_TAG),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()