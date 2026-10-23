"""
================================================================================
Building-Height Data-Quality Diagnostic (GlobalBuildingAtlas)
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Data-quality diagnostic for the Lisbon building-height dataset
(GlobalBuildingAtlas, Zhu et al., 2025, ESSD). The heights are not
user-edited OpenStreetMap tags: they are estimated by a machine-learning
model from PlanetScope satellite imagery. This script lets a human check,
visually and statistically, that the dataset can be trusted for the VIPV
canyon analysis: is coverage dense enough, are the heights in a plausible
range, and are tall buildings where they are expected in Lisbon?

Figures produced
----------------
Raw building dataset, clipped to the sensor-route bounding box (Figs A–C,
K–M; campaign-independent except for that bounding box):
  Fig A - height_map
      Buildings coloured by height on an OSM basemap — the most direct
      "does this look right" check.
  Fig B - density_hexbin
      Hexbin of building COUNT (not height): gaps in the source dataset
      itself, independent of where the sensors drove.
  Fig C - height_and_footprint_histograms
      Distribution of height (and footprint area, when the file has a
      footprint column), with the implausible range (<= 0 m or
      > HEIGHT_MAX_PLAUSIBLE_M) marked.
  Fig K - dataset_spatial_extent
      All building centroids on a basemap, coloured by the dataset 'source'
      field: total coverage of the metropolitan area.
  Fig L - building_count_stats
      Summary panel: building count, height completeness, count per source,
      and comparison with the ARpoly50 benchmark of Zhu et al. (2025)
      (Table 3, EU: 69.3 %).
  Fig M - height_by_freguesia
      Mean building height per GADM Level-3 freguesia (parish), as a
      choropleth (skipped if GADM_LEVEL3_SHP is missing).

Building-height join of the sensor files (Figs D–J):
  Fig D - join_coverage_map
      Measurement points coloured by the number of buildings found in their
      BUFFER_RADIUS_M buffer (bh_mapping) — shows WHERE the join fails, as
      opposed to the aggregate rate printed by diagnose_building_coverage.py.
  Fig E - height_by_cos23_category
      Mean building height per COS2023 level-4 category (categories with
      >= MIN_BIN_POINTS points). Vertical residential should sit clearly
      above horizontal residential and open-land categories.
  Fig F - nan_rate_by_cos23
      Zero-building rate (n_buildings_nearby == 0) per COS2023 category:
      shows whether the points without building are concentrated in
      genuinely open categories (roads, parks) rather than in dense urban
      fabric.
  Fig G - n_buildings_distribution
      Histogram + CDF of n_buildings_nearby over all measurement points.
  Fig H - height_vs_n_buildings_boxplots
      Boxplots of mean_building_height_m binned by n_buildings_nearby: do
      points with very few buildings (n = 1) show anomalous heights?
  Fig I - mean_vs_max_height_scatter
      mean_building_height_m vs max_building_height_m, coloured by
      n_buildings_nearby: how much the tallest building dominates the mean.
  Fig J - nan_rate_hexbin
      Zero-building rate aggregated into hexagonal cells over the route.

The console also reports the building count in the bounding box, the share
of non-positive or implausibly tall heights, and summary quantiles.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon', 'cos23_n4_label',
             'n_buildings_nearby', 'mean_building_height_m',
             'max_building_height_m' (missing columns are filled with NaN)
    input  : BUILDINGS_PATH — GlobalBuildingAtlas building centroids with
             heights (Lisbon metropolitan area clip)
             BOUNDARIES_DIR / gadm41_PRT_2.shp (Lisbon outline, optional)
             BOUNDARIES_DIR / gadm41_PRT_3.shp (freguesias, Fig M, optional)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             <figure name>_<CAMPAIGN>_<variant_tag>.png (names listed above)

Requirements
------------
    pip install pandas numpy matplotlib geopandas contextily

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.1
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BOUNDARIES_DIR, BUILDINGS_PATH, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
import bh_mapping as bhm  # noqa: E402


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Raw building file (campaign-independent) ---------------------------------
BUILDINGS_FILE: Path = BUILDINGS_PATH

# --- Input directories (one per campaign) -------------------------------------
# The sensor files feed the join diagnostics (Figs D-J) and give the bounding
# box of the measurement route used to clip the building file.
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

BBOX_MARGIN_DEG: float = 0.01  # margin around the sensor route [deg]

# --- GADM boundary (optional decoration on the maps) --------------------------
SHAPEFILE_DIR: Path | None = BOUNDARIES_DIR
LISBON_MUNICIPALITY_NAME: str = "Lisboa"
# GADM level 3 (freguesia / parish) shapefile — used for Fig M.
# Set to None to skip Fig M.  Download from gadm.org (country = Portugal, level 3).
GADM_LEVEL3_SHP: Path | None = SHAPEFILE_DIR / "gadm41_PRT_3.shp" if SHAPEFILE_DIR else None

# --- Output -------------------------------------------------------------------
# A subdirectory named after CAMPAIGN / variant tag is created automatically
# (e.g. building_height_quality_outputs/standard/sd_cleaned/).
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "building_height_quality_outputs"
OUTPUT_DPI: int = 300
SHOW_FIGURES: bool = False

# --- Plausibility thresholds --------------------------------------------------
HEIGHT_MAX_PLAUSIBLE_M: float = 100.0  # taller than this in Lisbon is very likely noise
MIN_BIN_POINTS: int = 60  # minimum points per COS2023 category (Fig E)

# --- Palette (consistent with the VIPV pipeline) ------------------------------
CMAP_HEIGHT: str = "viridis"
CMAP_DENSITY: str = "magma"
CMAP_COVERAGE: str = "YlOrRd"
BASEMAP_PROVIDER = ctx.providers.CartoDB.Positron


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

SENSOR_IDS: list[str] = ["j30", "j31", "j32", "j33"]


def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor identifiers to their CSV input paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    if campaign == "standard":
        return {
            s: data_dir / f"{s}_satellite_landuse_v2_{suffix}.csv"
            for s in SENSOR_IDS
        }
    elif campaign == "tuktuk":
        return {
            s: tuktuk_data_dir / f"{s}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
            for s in SENSOR_IDS
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _latlon_cols(campaign: str, sensor: str) -> tuple[str, str]:
    """Return the (lat_col, lon_col) names for a given sensor and campaign.

    Both campaigns use sensor-prefixed columns ('J31 lat' / 'J31 lon'):
    normalize_tuktuk.py gives the tuk-tuk files the standard schema.
    """
    if campaign in ("standard", "tuktuk"):
        tag = sensor.upper()
        return f"{tag} lat", f"{tag} lon"
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


def _out_png(name: str) -> Path:
    """Return the PNG output path of a figure, embedding campaign + variant tag."""
    return OUTPUT_DIR / f"{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"


# ==============================================================================
# 1. BOUNDING BOX + LOADING
# ==============================================================================

def detect_sensor_bbox(
    campaign: str,
    sensor_files: dict[str, Path],
    margin_deg: float,
) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) covering all available sensors.

    Both campaigns use sensor-prefixed columns ("J31 lat" / "J31 lon").
    """
    lat_min, lat_max = float("inf"), float("-inf")
    lon_min, lon_max = float("inf"), float("-inf")
    found_any = False

    for sensor, path in sensor_files.items():
        lat_col, lon_col = _latlon_cols(campaign, sensor)
        if not path.exists():
            print(f"  [warn] sensor file not found, skipped: {path}")
            continue
        df = pd.read_csv(path, usecols=[lat_col, lon_col]).dropna()
        if df.empty:
            continue
        found_any = True
        lat_min = min(lat_min, df[lat_col].min())
        lat_max = max(lat_max, df[lat_col].max())
        lon_min = min(lon_min, df[lon_col].min())
        lon_max = max(lon_max, df[lon_col].max())

    if not found_any:
        raise RuntimeError("No valid lat/lon values found in any sensor file.")

    return (lon_min - margin_deg, lat_min - margin_deg,
            lon_max + margin_deg, lat_max + margin_deg)


# ==============================================================================
# 1b. COMBINED PER-SENSOR JOIN DATA (Figs D-J)
# ==============================================================================

def load_sensor_join_data(campaign: str, sensor_files: dict[str, Path]) -> pd.DataFrame:
    """
    Load and concatenate, across sensors, the columns needed by Figs D-J:
    lat, lon, cos23_n4_label, n_buildings_nearby, mean_building_height_m,
    max_building_height_m. All four join columns are on the same row of the
    same per-sensor file (added upstream of clean_data.py by
    join_cos_standard.py / join_cos23_tuktuk.py and join_bh_standard.py /
    join_bh_tuktuk.py), so no alignment or merge is needed.

    Missing columns are tolerated (e.g. an older FILE_SUFFIX variant produced
    before a given join step): they are filled with NaN and a warning is
    printed, so the caller's individual figures can still run on whichever
    columns are actually available.
    """
    wanted_cols = ["cos23_n4_label", "n_buildings_nearby",
                   "mean_building_height_m", "max_building_height_m"]
    frames: list[pd.DataFrame] = []

    for sensor, path in sensor_files.items():
        lat_col, lon_col = _latlon_cols(campaign, sensor)
        if not path.exists():
            print(f"  [warn] sensor file not found, skipped: {path}")
            continue

        header = pd.read_csv(path, nrows=0).columns
        present = [c for c in wanted_cols if c in header]
        missing = [c for c in wanted_cols if c not in header]
        if missing:
            print(f"  [warn] {sensor}: missing columns {missing} "
                  f"(run the corresponding join_*.py step first)")

        usecols = [lat_col, lon_col] + present
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
        for c in missing:
            df[c] = np.nan
        df["sensor"] = sensor
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. Check DATA_DIR/"
            "TUKTUK_DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    return pd.concat(frames, ignore_index=True)


def load_lisbon_boundary() -> gpd.GeoDataFrame | None:
    """Load the GADM Level-2 Lisboa municipality polygon, or None if unavailable."""
    if SHAPEFILE_DIR is None:
        return None
    shp_path = SHAPEFILE_DIR / "gadm41_PRT_2.shp"
    if not shp_path.is_file():
        print(f"  [warn] GADM shapefile not found: {shp_path}")
        return None
    gdf = gpd.read_file(shp_path)
    name_col = "NAME_2" if "NAME_2" in gdf.columns else gdf.columns[2]
    lisbon = gdf[gdf[name_col] == LISBON_MUNICIPALITY_NAME]
    return lisbon if not lisbon.empty else None


def try_add_basemap(ax: plt.Axes) -> None:
    """Add an OSM-style basemap; warn and continue if the tiles are unreachable."""
    try:
        ctx.add_basemap(ax, source=BASEMAP_PROVIDER, crs="EPSG:3857", attribution_size=5)
    except Exception as exc:
        print(f"  [warn] basemap fetch failed ({exc!s}); plain background used instead.")


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    """Save the figure at OUTPUT_DPI, optionally show it, then close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ==============================================================================
# 2. CONSOLE DIAGNOSTICS
# ==============================================================================

def print_quality_diagnostics(gdf: gpd.GeoDataFrame) -> None:
    """Print summary statistics on the raw footprint/height data quality."""
    heights = gdf["_height_m"]
    n = len(gdf)
    n_nonpositive = (heights <= 0).sum()
    n_missing = heights.isna().sum()
    n_implausible = (heights > HEIGHT_MAX_PLAUSIBLE_M).sum()

    print("\n" + "-" * 78)
    print("RAW DATA-QUALITY SUMMARY (footprints within the sensor bbox)")
    print("-" * 78)
    print(f"  Total footprints                    : {n:,}")
    print(f"  Missing height (NaN)                 : {n_missing:,} "
          f"({n_missing / n * 100:.1f}%)")
    print(f"  Non-positive height (<=0 m)          : {n_nonpositive:,} "
          f"({n_nonpositive / n * 100:.1f}%)")
    print(f"  Implausibly tall (> {HEIGHT_MAX_PLAUSIBLE_M:.0f} m)         : "
          f"{n_implausible:,} ({n_implausible / n * 100:.1f}%)")
    print("\n  Height quantiles [m]:")
    print(heights.describe(percentiles=[.05, .25, .5, .75, .95]).to_string())
    print("-" * 78)


# ==============================================================================
# 3. FIG A - HEIGHT CHOROPLETH MAP
# ==============================================================================

def fig_height_map(gdf: gpd.GeoDataFrame, lisbon_gdf: gpd.GeoDataFrame | None,
                    out_path: Path) -> None:
    """Fig A — buildings coloured by height over the measurement area."""
    plot_gdf = gdf[(gdf["_height_m"] > 0) &
                   (gdf["_height_m"] <= HEIGHT_MAX_PLAUSIBLE_M)].to_crs(epsg=3857)
    if plot_gdf.empty:
        print("  [WARN] Fig A: no plausible-height footprints to plot; skipping.")
        return

    fig, ax = plt.subplots(figsize=(11, 10))
    plot_gdf.plot(
        ax=ax, column="_height_m", cmap=CMAP_HEIGHT, linewidth=0,
        legend=True, legend_kwds={"label": "Building height [m]", "shrink": 0.7},
        zorder=3,
    )
    if lisbon_gdf is not None:
        lisbon_gdf.to_crs(epsg=3857).boundary.plot(
            ax=ax, color="#111111", linewidth=1.3, zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    ax.set_title("Building heights over the measurement area (GlobalBuildingAtlas)",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"{len(plot_gdf):,} footprints shown (heights in 0-"
        f"{HEIGHT_MAX_PLAUSIBLE_M:.0f} m range) - visually check that tall "
        f"clusters match known dense/vertical neighbourhoods.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 4. FIG B - DENSITY HEXBIN
# ==============================================================================

def fig_density_hexbin(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Fig B — hexbin of the building count (dataset coverage gaps)."""
    centroids = gdf.to_crs(epsg=3857).geometry.centroid
    fig, ax = plt.subplots(figsize=(11, 10))
    hb = ax.hexbin(centroids.x, centroids.y, gridsize=60, cmap=CMAP_DENSITY,
                   mincnt=1, zorder=3)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.7)
    cbar.set_label("Buildings per hex cell", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    ax.set_title("Footprint density over the measurement area",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        "Reveals gaps in the source dataset itself (independent of where the "
        "sensors drove) - empty cells over built-up areas are a coverage red flag.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 5. FIG C - HISTOGRAMS
# ==============================================================================

def fig_histograms(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Fig C — histograms of building height (and footprint area, if present)."""
    footprint_col = next((c for c in bhm.FOOTPRINT_COL_CANDIDATES if c in gdf.columns), None)

    n_panels = 2 if footprint_col else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    heights = gdf["_height_m"].dropna()
    ax.hist(heights.clip(upper=HEIGHT_MAX_PLAUSIBLE_M * 1.5), bins=60,
            color="#1565C0", alpha=0.85)
    ax.axvspan(-50, 0, color="#E05C5C", alpha=0.15, label="Implausible (<= 0 m)")
    ax.axvspan(HEIGHT_MAX_PLAUSIBLE_M, HEIGHT_MAX_PLAUSIBLE_M * 1.5, color="#E05C5C",
              alpha=0.15, label=f"Implausible (> {HEIGHT_MAX_PLAUSIBLE_M:.0f} m)")
    ax.set_xlabel("Building height [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_title("Height distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)

    if footprint_col:
        ax2 = axes[1]
        footprints = gdf[footprint_col].dropna()
        clip_val = np.nanpercentile(footprints, 99)
        ax2.hist(footprints.clip(upper=clip_val), bins=60, color="#2E7D32", alpha=0.85)
        ax2.set_xlabel("Footprint area [m²] (clipped at 99th pct)", fontsize=10,
                       fontweight="bold")
        ax2.set_ylabel("Count", fontsize=10, fontweight="bold")
        ax2.set_title("Footprint area distribution", fontsize=11, fontweight="bold")
        ax2.grid(linestyle="--", alpha=0.3)

    fig.suptitle("Raw building dataset - distribution checks", fontsize=13,
                fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 6. FIG D - JOIN COVERAGE MAP (sensor points, not raw footprints)
# ==============================================================================

def fig_join_coverage_map(join_df: pd.DataFrame, out_path: Path) -> None:
    """join_df: combined per-sensor DataFrame from load_sensor_join_data(),
    already carrying generic 'lat'/'lon' columns regardless of campaign."""
    df = join_df.dropna(subset=["lat", "lon", "n_buildings_nearby"])
    if df.empty:
        print("  [WARN] Fig D: no rows with lat/lon + join result; skipping.")
        return

    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 10))
    n_clip = np.clip(pts["n_buildings_nearby"].values, 0,
                     np.nanpercentile(pts["n_buildings_nearby"], 98))
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=n_clip, cmap=CMAP_COVERAGE,
                    s=8, alpha=0.7, linewidths=0, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Buildings found in buffer (clipped at 98th pct)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    n_zero = (df["n_buildings_nearby"] == 0).sum()
    ax.set_title("Sensor route - buffer join coverage", fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"{len(df):,} measurement points - {n_zero:,} "
        f"({n_zero / len(df) * 100:.1f}%) found zero buildings within "
        f"{bhm.BUFFER_RADIUS_M:.0f} m - shows WHERE the join fails, not just how often.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 7. FIG E - MEAN HEIGHT BY COS2023 CATEGORY (sanity check)
# ==============================================================================

def fig_height_by_cos23_category(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig E — mean building height per COS2023 level-4 category.

    join_df: combined per-sensor DataFrame from load_sensor_join_data().
    """
    df = join_df.dropna(subset=["cos23_n4_label", "mean_building_height_m"])
    if df.empty:
        print("  [WARN] Fig E: no rows with both cos23_n4_label and "
              "mean_building_height_m present; skipping.")
        return

    summary = (
        df.groupby("cos23_n4_label")["mean_building_height_m"]
        .agg(mean="mean", std="std", n="count")
        .query("n >= @MIN_BIN_POINTS")
        .sort_values("mean")
    )
    if summary.empty:
        print(f"  [WARN] Fig E: no COS2023 category reaches {MIN_BIN_POINTS} "
              "points; skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(summary))))
    ax.barh(summary.index, summary["mean"], xerr=summary["std"],
           color="#1565C0", alpha=0.85, capsize=3)
    ax.set_xlabel("Mean building height in the 50 m buffer [m]", fontsize=10,
                 fontweight="bold")
    ax.set_title("Mean building height by COS2023 category (sanity check)",
                fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.text(
        0.5, -0.02 - 0.01 * len(summary),
        f"Categories with >= {MIN_BIN_POINTS} points only. Expected pattern: "
        "vertical residential clearly above horizontal residential and open-land "
        "categories - if the order looks wrong, treat it as a data/join issue "
        "before drawing scientific conclusions.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 8. FIG F - NaN RATE BY COS2023 CATEGORY
# ==============================================================================

def fig_nan_rate_by_cos23(join_df: pd.DataFrame, out_path: Path) -> None:
    """Zero-building rate per COS2023 land-use category.

    The key paper argument: NaN points cluster in genuinely open categories
    (wide roads, parks, water), NOT in the dense urban fabric the analysis targets.

    join_df: combined per-sensor DataFrame from load_sensor_join_data().
    """
    df = join_df.dropna(subset=["cos23_n4_label", "n_buildings_nearby"]).copy()
    if df.empty:
        print("  [WARN] Fig F: no rows with both cos23_n4_label and "
              "n_buildings_nearby present; skipping.")
        return
    df["is_zero"] = df["n_buildings_nearby"] == 0

    summary = (
        df.groupby("cos23_n4_label")
        .agg(zero_rate=("is_zero", "mean"), n=("is_zero", "count"))
        .query("n >= @MIN_BIN_POINTS")
        .sort_values("zero_rate")
    )
    if summary.empty:
        print(f"  [WARN] Fig F: no category with >= {MIN_BIN_POINTS} points; skipping.")
        return

    colours = ["#E05C5C" if r > 0.30 else "#F4A800" if r > 0.10 else "#2E7D32"
               for r in summary["zero_rate"]]

    overall_rate = df["is_zero"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(summary))))
    bars = ax.barh(summary.index, summary["zero_rate"] * 100,
                   color=colours, alpha=0.85)
    ax.axvline(overall_rate, color="#1565C0", linewidth=1.2, linestyle="--",
               label=f"Overall rate ({overall_rate:.0f}%)")
    ax.set_xlabel("Share of points with zero buildings in 50 m buffer [%]",
                  fontsize=10, fontweight="bold")
    ax.set_title("Zero-building join rate by COS2023 land-use category",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    # Annotate n
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"n={row['n']:,}", va="center", fontsize=7, color="#555555")
    fig.text(
        0.5, -0.02,
        "Green (<10%) = dataset covers this category well. "
        "Red (>30%) = structural gap for this land-use type. "
        "Blue line = overall average.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 9. FIG G - n_buildings_nearby DISTRIBUTION + CDF
# ==============================================================================

def fig_n_buildings_distribution(join_df: pd.DataFrame, out_path: Path) -> None:
    """Histogram + cumulative distribution of n_buildings_nearby.

    join_df: combined per-sensor DataFrame from load_sensor_join_data().
    """
    n_vals = join_df["n_buildings_nearby"].dropna().astype(int)
    if n_vals.empty:
        print("  [WARN] Fig G: no valid n_buildings_nearby values; skipping.")
        return

    counts = n_vals.value_counts().sort_index()
    cap = min(counts.index.max(), 20)  # cap display at 20 for readability
    counts = counts[counts.index <= cap]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    colours_hist = ["#E05C5C" if i == 0 else "#1565C0" for i in counts.index]
    ax1.bar(counts.index, counts.values, color=colours_hist, alpha=0.85, width=0.7)
    ax1.set_xlabel("n_buildings_nearby", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Number of GPS measurement points", fontsize=10, fontweight="bold")
    ax1.set_title("Distribution of buildings found\nper measurement point",
                  fontsize=11, fontweight="bold")
    ax1.set_xticks(counts.index)
    if cap == 20:
        ax1.set_xticklabels(
            [str(i) if i < 20 else "20+" for i in counts.index], fontsize=8)
    n_zero = n_vals[n_vals == 0].sum() if 0 in n_vals.values else (n_vals == 0).sum()
    n_zero = int((n_vals == 0).sum())
    ax1.text(0, ax1.get_ylim()[1] * 0.95,
             f"n=0: {n_zero:,}\n({n_zero / len(n_vals) * 100:.1f}%)",
             ha="center", va="top", fontsize=8, color="#E05C5C", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # CDF
    sorted_vals = np.sort(n_vals.values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cdf * 100, color="#1565C0", linewidth=2)
    ax2.axhline(100 * (n_vals == 0).mean(), color="#E05C5C", linewidth=1.2,
                linestyle="--", label=f"Zero-building share ({(n_vals==0).mean()*100:.1f}%)")
    ax2.set_xlabel("n_buildings_nearby", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cumulative % of points", fontsize=10, fontweight="bold")
    ax2.set_title("Cumulative distribution (CDF)\nof n_buildings_nearby",
                  fontsize=11, fontweight="bold")
    ax2.set_xlim(0, cap)
    ax2.legend(fontsize=8)
    ax2.grid(linestyle="--", alpha=0.3)

    fig.suptitle("n_buildings_nearby — coverage distribution across measurement points",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 10. FIG H - HEIGHT BOXPLOTS BY n_buildings_nearby
# ==============================================================================

def fig_height_vs_n_buildings_boxplots(join_df: pd.DataFrame,
                                       out_path: Path) -> None:
    """Boxplots of mean_building_height_m binned by n_buildings_nearby.

    If points with n=1 show very different (higher variance) heights than
    points with n>=5, that suggests the single captured building may be
    unrepresentative of the surrounding canyon.

    join_df: combined per-sensor DataFrame from load_sensor_join_data().
    """
    df = join_df[["n_buildings_nearby", "mean_building_height_m"]].dropna(
        subset=["mean_building_height_m"])
    df = df[df["n_buildings_nearby"] > 0]  # exclude zero-coverage points

    if df.empty:
        print("  [WARN] Fig H: no valid rows after filtering; skipping.")
        return

    # Adaptive bins based on actual distribution of n_buildings_nearby.
    # With a sparse dataset (48k buildings, 50m buffer), most covered points
    # have n in the range 1-10+. Bins: 1-2, 3-5, 6-10, 11-20, 20+
    def label_n(n: int) -> str:
        if n <= 2:
            return "1–2"
        if n <= 5:
            return "3–5"
        if n <= 10:
            return "6–10"
        if n <= 20:
            return "11–20"
        return "20+"

    df["n_bin"] = df["n_buildings_nearby"].apply(label_n)
    order = ["1–2", "3–5", "6–10", "11–20", "20+"]
    groups = [df[df["n_bin"] == b]["mean_building_height_m"].values
              for b in order if b in df["n_bin"].values]
    labels = [b for b in order if b in df["n_bin"].values]

    if not groups:
        print("  [WARN] Fig H: no non-empty bins after grouping; skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="#F4A800", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3,
                                    markerfacecolor="#888888"))
    colours_box = ["#abd9e9", "#74add1", "#4575b4", "#313695", "#1565C0"]
    for patch, col in zip(bp["boxes"], colours_box[:len(bp["boxes"])]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    # Annotate n per bin
    for i, (lbl, grp) in enumerate(zip(labels, groups), start=1):
        ax.text(i, ax.get_ylim()[0] - 0.5, f"n={len(grp):,}",
                ha="center", va="top", fontsize=7.5, color="#555555")

    ax.set_xlabel("n_buildings_nearby — number of buildings captured in 50 m buffer",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("mean_building_height_m [m]", fontsize=10, fontweight="bold")
    ax.set_title("Building height distribution by number of buildings\nin the buffer",
                 fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.text(
        0.5, 0.01,
        "Bins adapted to the actual distribution (sparse dataset). "
        "Higher n = richer spatial context for the height estimate. "
        "Stable median across bins = the height signal is robust to buffer density.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 11. FIG I - MEAN vs MAX HEIGHT SCATTER
# ==============================================================================

def fig_mean_vs_max_scatter(join_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of mean_building_height_m vs max_building_height_m.

    Coloured by n_buildings_nearby. Points on the diagonal have mean ≈ max
    (homogeneous neighbourhood). Points far below the diagonal have one very
    tall outlier dominating the max while the average is much lower.

    join_df: combined per-sensor DataFrame from load_sensor_join_data().
    """
    df = join_df[["mean_building_height_m", "max_building_height_m",
                   "n_buildings_nearby"]].dropna(
        subset=["mean_building_height_m", "max_building_height_m"])
    df = df[df["n_buildings_nearby"] > 0]

    if df.empty:
        print("  [WARN] Fig I: no valid rows; skipping.")
        return

    # Downsample for display performance (scatter with 80k pts is slow)
    if len(df) > 15_000:
        df = df.sample(n=15_000, random_state=42)

    n_clip = np.clip(df["n_buildings_nearby"].values, 1,
                     np.nanpercentile(df["n_buildings_nearby"], 95))

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["mean_building_height_m"], df["max_building_height_m"],
                    c=n_clip, cmap="viridis", s=6, alpha=0.5, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label("n_buildings_nearby (clipped at 95th pct)", fontsize=9)

    # 1:1 line
    lim = max(df["max_building_height_m"].quantile(0.99),
              df["mean_building_height_m"].quantile(0.99))
    ax.plot([0, lim], [0, lim], color="#E05C5C", linewidth=1.2,
            linestyle="--", label="mean = max (n=1 or homogeneous)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("mean_building_height_m [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("max_building_height_m [m]", fontsize=10, fontweight="bold")
    ax.set_title("Mean vs max building height\ncoloured by n_buildings_nearby",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)

    # Correlation
    r = df[["mean_building_height_m", "max_building_height_m"]].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            fontsize=9, va="top", color="#333333")
    fig.text(
        0.5, 0.01,
        "Points on the red line: mean = max (single building or uniform canyon). "
        "Points far below: one very tall outlier dominates max. "
        "Colour shows how many buildings contributed to the mean.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 12. FIG J - SPATIAL NaN RATE HEXBIN
# ==============================================================================

def fig_nan_rate_hexbin(join_df: pd.DataFrame, out_path: Path) -> None:
    """Hexagonal map of the zero-building rate over the GPS measurement route.

    Complements Fig B (raw dataset gaps) by showing which areas the sensors
    actually visited AND found no buildings — the operational blind spots.

    join_df: combined per-sensor DataFrame from load_sensor_join_data(),
    already carrying generic 'lat'/'lon' columns regardless of campaign.
    """
    df = join_df.dropna(subset=["lat", "lon", "n_buildings_nearby"]).copy()
    if df.empty:
        print("  [WARN] Fig J: no rows with lat/lon + join result; skipping.")
        return
    df["is_zero"] = (df["n_buildings_nearby"] == 0).astype(float)

    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 10))
    hb = ax.hexbin(pts.geometry.x, pts.geometry.y, C=df["is_zero"].values,
                   gridsize=50, reduce_C_function=np.mean,
                   cmap="RdYlGn_r", mincnt=5, zorder=3,
                   vmin=0, vmax=1)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.7)
    cbar.set_label("Share of points with zero buildings in 50 m buffer", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    ax.set_title("Spatial distribution of zero-building join rate",
                 fontsize=13, fontweight="bold")
    n_zero = int((df["n_buildings_nearby"] == 0).sum())
    fig.text(
        0.5, 0.01,
        f"Green = well-covered areas. Red = join failures concentrated here. "
        f"Overall: {n_zero:,} / {len(df):,} points ({n_zero/len(df)*100:.1f}%) "
        f"found zero buildings. Min 5 points per hex cell shown.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 13. FIG K - DATASET SPATIAL EXTENT
# ==============================================================================

def fig_dataset_spatial_extent(gdf: gpd.GeoDataFrame,
                                lisbon_gdf: gpd.GeoDataFrame | None,
                                out_path: Path) -> None:
    """All building centroids coloured by dataset source field.

    Shows the spatial extent of the raw GeoJSON and which sources
    contribute to different parts of the metropolitan area.
    """
    plot_gdf = gdf.to_crs(epsg=3857).copy()

    # Try to use the 'source' field; fall back to a single colour
    source_col = next((c for c in ["source", "Source", "SOURCE"] if c in plot_gdf.columns),
                      None)

    fig, ax = plt.subplots(figsize=(11, 10))

    if source_col and plot_gdf[source_col].nunique() <= 10:
        sources = plot_gdf[source_col].fillna("unknown").unique()
        palette = plt.cm.tab10.colors
        for i, src in enumerate(sorted(sources)):
            sub = plot_gdf[plot_gdf[source_col].fillna("unknown") == src]
            ax.scatter(sub.geometry.x, sub.geometry.y,
                       s=2, alpha=0.5, linewidths=0,
                       color=palette[i % len(palette)],
                       label=f"{src} ({len(sub):,})", zorder=3)
        ax.legend(fontsize=7, loc="lower right", markerscale=4,
                  title="Source", title_fontsize=8)
    else:
        ax.scatter(plot_gdf.geometry.x, plot_gdf.geometry.y,
                   s=2, alpha=0.4, linewidths=0, color="#1565C0", zorder=3)

    if lisbon_gdf is not None:
        lisbon_gdf.to_crs(epsg=3857).boundary.plot(
            ax=ax, color="#111111", linewidth=1.3, zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    ax.set_title("Spatial extent and source breakdown\nof the GlobalBuildingAtlas clip",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"{len(plot_gdf):,} building centroids shown. Each dot = one building. "
        "Coverage gaps over the basemap indicate areas absent from the dataset.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 14. FIG L - BUILDING COUNT SUMMARY PANEL
# ==============================================================================

def fig_building_count_summary(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Summary statistics panel for the raw dataset.

    Panels: (1) count by source, (2) height completeness breakdown,
    (3) comparison bar vs Zhu et al. 2025 EU ARpoly50 benchmark (69.3%).
    """
    heights = gdf["_height_m"]
    n_total = len(gdf)
    n_valid = int(((heights > 0) & (heights <= HEIGHT_MAX_PLAUSIBLE_M)).sum())
    n_zero_or_neg = int((heights <= 0).sum())
    n_missing = int(heights.isna().sum())
    n_implausible = int((heights > HEIGHT_MAX_PLAUSIBLE_M).sum())

    source_col = next((c for c in ["source", "Source", "SOURCE"] if c in gdf.columns), None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Panel 1 — count by source
    ax1 = axes[0]
    if source_col and gdf[source_col].nunique() <= 15:
        src_counts = gdf[source_col].fillna("unknown").value_counts()
        src_counts.plot(kind="barh", ax=ax1, color="#1565C0", alpha=0.85)
        ax1.set_xlabel("Number of buildings", fontsize=9, fontweight="bold")
        ax1.set_title("Building count\nby source", fontsize=11, fontweight="bold")
        ax1.grid(axis="x", linestyle="--", alpha=0.3)
        for i, (val, lbl) in enumerate(zip(src_counts.values, src_counts.index)):
            ax1.text(val + n_total * 0.005, i, f"{val:,}", va="center", fontsize=7.5)
    else:
        ax1.text(0.5, 0.5, f"Total:\n{n_total:,}\nbuildings",
                 ha="center", va="center", fontsize=18, fontweight="bold",
                 color="#1565C0", transform=ax1.transAxes)
        ax1.set_title("Total building count", fontsize=11, fontweight="bold")
        ax1.axis("off")

    # Panel 2 — height completeness breakdown (pie)
    ax2 = axes[1]
    sizes = [n_valid, n_zero_or_neg, n_missing, n_implausible]
    labels_pie = [
        f"Valid height\n{n_valid:,} ({n_valid/n_total*100:.1f}%)",
        f"Zero/negative\n{n_zero_or_neg:,} ({n_zero_or_neg/n_total*100:.1f}%)",
        f"Missing (NaN)\n{n_missing:,} ({n_missing/n_total*100:.1f}%)",
        f"Implausible >100m\n{n_implausible:,} ({n_implausible/n_total*100:.1f}%)",
    ]
    colours_pie = ["#2E7D32", "#F4A800", "#888888", "#E05C5C"]
    sizes_nonzero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colours_pie) if s > 0]
    ax2.pie([x[0] for x in sizes_nonzero],
            labels=[x[1] for x in sizes_nonzero],
            colors=[x[2] for x in sizes_nonzero],
            startangle=90, textprops={"fontsize": 8})
    ax2.set_title("Height completeness\nbreakdown", fontsize=11, fontweight="bold")

    # Panel 3 — benchmark comparison bar
    ax3 = axes[2]
    completeness_pct = n_valid / n_total * 100
    # ARpoly50 EU benchmark from Zhu et al. 2025, Table 3
    benchmark_recall = 69.3
    categories = ["This dataset\n(height completeness)", "Zhu et al. 2025\n(EU ARpoly50)"]
    values = [completeness_pct, benchmark_recall]
    bar_cols = ["#1565C0", "#888888"]
    bars = ax3.bar(categories, values, color=bar_cols, alpha=0.85, width=0.4)
    ax3.set_ylim(0, 110)
    ax3.set_ylabel("Percentage [%]", fontsize=9, fontweight="bold")
    ax3.set_title("Comparison with\npublished benchmark", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax3.axhline(100, color="#E05C5C", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.text(1, benchmark_recall - 8,
             "Building recall\nvs ground truth\n(Zhu et al. 2025,\nTable 3 EU)",
             ha="center", fontsize=7, color="#555555")

    fig.suptitle(
        f"GlobalBuildingAtlas — building count and height quality summary "
        f"({n_total:,} buildings in sensor bbox)",
        fontsize=12.5, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 15. FIG M - MEAN HEIGHT BY FREGUESIA (GADM Level 3)
# ==============================================================================

def fig_height_by_freguesia(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    """Choropleth of mean building height aggregated at the parish (freguesia) level.

    Requires GADM Level-3 shapefile (gadm41_PRT_3.shp).
    Reveals spatial heterogeneity in the dataset across administrative units.
    """
    if GADM_LEVEL3_SHP is None or not Path(GADM_LEVEL3_SHP).is_file():
        print(f"  [warn] Fig M: GADM level-3 shapefile not found "
              f"({GADM_LEVEL3_SHP}); skipping.")
        return

    parishes = gpd.read_file(GADM_LEVEL3_SHP)

    # Spatial join: assign each building centroid to a parish
    buildings_4326 = gdf[gdf["_height_m"].notna() & (gdf["_height_m"] > 0)].copy()
    if buildings_4326.crs.to_epsg() != 4326:
        buildings_4326 = buildings_4326.to_crs(epsg=4326)
    if parishes.crs.to_epsg() != 4326:
        parishes = parishes.to_crs(epsg=4326)

    joined = gpd.sjoin(buildings_4326[["_height_m", "geometry"]],
                       parishes[["geometry", "NAME_3"]],
                       how="left", predicate="within")

    parish_stats = (
        joined.groupby("NAME_3")["_height_m"]
        .agg(mean_height="mean", n="count")
        .reset_index()
    )

    parishes_plot = parishes.merge(parish_stats, on="NAME_3", how="left")
    parishes_plot = parishes_plot[parishes_plot["mean_height"].notna()]
    if parishes_plot.empty:
        print("  [WARN] Fig M: no parishes with height data after join; skipping.")
        return

    parishes_plot = parishes_plot.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 10))
    parishes_plot.plot(
        ax=ax, column="mean_height", cmap="YlOrRd", linewidth=0.4,
        edgecolor="#555555", legend=True,
        legend_kwds={"label": "Mean building height [m]", "shrink": 0.7},
        zorder=3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    ax.set_title("Mean building height by freguesia (parish)\nGlobalBuildingAtlas",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"{len(parishes_plot)} parishes shown. Only buildings with valid height "
        f"(0–{HEIGHT_MAX_PLAUSIBLE_M:.0f} m) included. Parishes with no buildings "
        "in dataset appear blank.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the building file and the sensor join data, then draw Figs A–M."""
    print("=" * 78)
    print("Building-height data-quality diagnostic (GlobalBuildingAtlas)")
    print(f"  CAMPAIGN         : {CAMPAIGN}")
    print(f"  Data variant     : {_VARIANT_TAG}")
    print(f"  BUILDINGS_FILE   : {BUILDINGS_FILE}")
    print(f"  OUTPUT_DIR       : {OUTPUT_DIR}")
    print("=" * 78)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BUILDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Building file not found at {BUILDINGS_FILE} - check "
            "BUILDINGS_PATH in config.py."
        )

    print("\n[1/14] Detecting sensor bounding box...")
    bbox = detect_sensor_bbox(CAMPAIGN, SENSOR_FILES, BBOX_MARGIN_DEG)
    print(f"  bbox = {bbox}")

    print("\n[2/14] Loading raw building footprints in this bbox...")
    buildings_gdf = bhm.load_building_heights(BUILDINGS_FILE, bbox=bbox)
    print_quality_diagnostics(buildings_gdf)

    lisbon_gdf = load_lisbon_boundary()

    # ── Figures A–E ───────────────────────────────────────────────────────────
    print("\n[3/14] Fig A - height choropleth map...")
    fig_height_map(buildings_gdf, lisbon_gdf, _out_png("height_map"))

    print("\n[4/14] Fig B - footprint density hexbin...")
    fig_density_hexbin(buildings_gdf, _out_png("density_hexbin"))

    print("\n[5/14] Fig C - height / footprint histograms...")
    fig_histograms(buildings_gdf, _out_png("height_and_footprint_histograms"))

    print("\n[6-12/14] Loading per-sensor join data for Figs D-J...")
    try:
        join_df = load_sensor_join_data(CAMPAIGN, SENSOR_FILES)
        print(f"  {len(join_df):,} rows loaded across "
              f"{join_df['sensor'].nunique()} sensor(s)")
    except FileNotFoundError as exc:
        print(f"  [warn] {exc} - skipping Figs D-J.")
        join_df = None

    print("\n[6/14] Fig D - join-coverage map...")
    if join_df is not None:
        fig_join_coverage_map(join_df, _out_png("join_coverage_map"))
    else:
        print("  [warn] no join data - skipping Fig D.")

    print("\n[7/14] Fig E - mean height by COS2023 category...")
    if join_df is not None:
        fig_height_by_cos23_category(join_df, _out_png("height_by_cos23_category"))
    else:
        print("  [warn] no join data - skipping Fig E.")

    # ── Figures F–M ───────────────────────────────────────────────────────────
    print("\n[8/14] Fig F - NaN rate by COS2023 category...")
    if join_df is not None:
        fig_nan_rate_by_cos23(join_df, _out_png("nan_rate_by_cos23"))
    else:
        print("  [warn] no join data - skipping Fig F.")

    print("\n[9/14] Fig G - n_buildings_nearby distribution + CDF...")
    if join_df is not None:
        fig_n_buildings_distribution(join_df, _out_png("n_buildings_distribution"))
    else:
        print("  [warn] no join data - skipping Fig G.")

    print("\n[10/14] Fig H - height boxplots by n_buildings_nearby...")
    if join_df is not None:
        fig_height_vs_n_buildings_boxplots(join_df, _out_png("height_vs_n_buildings_boxplots"))
    else:
        print("  [warn] no join data - skipping Fig H.")

    print("\n[11/14] Fig I - mean vs max height scatter...")
    if join_df is not None:
        fig_mean_vs_max_scatter(join_df, _out_png("mean_vs_max_height_scatter"))
    else:
        print("  [warn] no join data - skipping Fig I.")

    print("\n[12/14] Fig J - spatial NaN rate hexbin...")
    if join_df is not None:
        fig_nan_rate_hexbin(join_df, _out_png("nan_rate_hexbin"))
    else:
        print("  [warn] no join data - skipping Fig J.")

    print("\n[13/14] Fig K - dataset spatial extent...")
    fig_dataset_spatial_extent(
        buildings_gdf, lisbon_gdf, _out_png("dataset_spatial_extent"))

    print("\n[14/14] Fig L - building count summary panel + Fig M - height by freguesia...")
    fig_building_count_summary(buildings_gdf, _out_png("building_count_stats"))
    fig_height_by_freguesia(buildings_gdf, _out_png("height_by_freguesia"))

    print("\nDone. All figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()