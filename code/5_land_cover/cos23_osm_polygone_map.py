"""
================================================================================
COS2023 OSM Polygon Overlay Map
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Draws the COS2023 land-use polygons over an OSM basemap, in the viewport of
the measurements, with the legend exported as a separate figure.

  - A polygon category is coloured only if it holds at least
    MIN_POLYGON_POINTS_COS23 measurement points; categories below the
    threshold are hidden or drawn in grey (LOW_COUNT_POLYGON_MODE).
  - Polygon colours follow the COS code hierarchy (COLOR_STRATEGY,
    COLOR_GROUP_LEVEL).

COS2023 polygons
----------------
The polygons are read directly from the official COS2023 GeoPackage
(COS_GPKG_FILE). The level-4 code field name (COS23n4_C or COS23_n4_C) is
auto-detected and renamed to 'COS23_n4_C', the name used by the colouring
helpers of this script (cos_code_group, build_subcategory_color_map, ...).

Viewport
--------
Auto-detected from the measurement point cloud (plus VIEWPORT_MARGIN_FRAC)
unless VIEWPORT_OVERRIDE is set to a fixed (lon_min, lat_min, lon_max,
lat_max) tuple.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon', 'cos23_n4_label'
    input  : COS23_PATH (COS2023 GeoPackage, DGT)
             BOUNDARIES_DIR / gadm41_PRT_2.shp (Lisbon outline, DRAW_BOUNDARY)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             cos23_map_polygon_overlay_<CAMPAIGN>_<variant_tag>.png / .pdf
             cos23_map_polygon_overlay_legend_<CAMPAIGN>_<variant_tag>.png / .pdf

Requirements
------------
    pip install pandas numpy matplotlib geopandas contextily shapely pyproj

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.2
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import colorsys
import sys
import textwrap
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BOUNDARIES_DIR, COS23_PATH, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
import cos23_mapping as cos23  # noqa: E402


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

# Path to the official COS2023 GeoPackage, downloaded once from DGT:
#   https://geo2.dgterritorio.gov.pt/cos/S2/COS2023/COS2023v1-S2-gpkg.zip
COS_GPKG_FILE: Path = COS23_PATH

# Candidate field names for the level-4 code column in the GeoPackage, tried
# in order (see join_cos_standard.py for the full explanation).
CODE_FIELD_CANDIDATES: list[str] = ["COS23n4_C", "COS23_n4_C"]

# Base output directory.
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "5_land_cover" / "cos23_polygons_outputs"

# GADM Level-2 shapefile directory.
SHAPEFILE_DIR: Path = BOUNDARIES_DIR
LISBON_MUNICIPALITY_NAME: str = "Lisboa"
# Draw the GADM municipality boundary outline.
DRAW_BOUNDARY: bool = True


# ==============================================================================
# THRESHOLD AND COLOURING OPTIONS
# ==============================================================================

# Minimum number of measurement points required for a COS2023 category to be
# displayed as a coloured polygon category.
MIN_POLYGON_POINTS_COS23: int = 600

# Behaviour for polygon categories below the threshold:
#   "hide" -> do not draw these polygons
#   "grey" -> draw them in neutral grey and group them as "Other"
LOW_COUNT_POLYGON_MODE: str = "hide"

# Colouring strategy:
#   "subcategory" -> colours are structured by COS code hierarchy
#   "dynamic"     -> scalable general-purpose colour palette
COLOR_STRATEGY: str = "subcategory"

# COS hierarchy level used as the colour family.
#   2 = good compromise between grouped colours and readability.
COLOR_GROUP_LEVEL: int = 2


# ==============================================================================
# FIGURE SETTINGS
# ==============================================================================

POLYGON_ALPHA: float = 0.5
OTHER_POLYGON_COLOR: str = "#BBBBBB"

# Point parameters.
# POINT_ALPHA = 0 means points are hidden.
POINT_COLOR: str = "#E6007A"
POINT_SIZE: float = 5.0
POINT_ALPHA: float = 0.0

# Basemap provider.
BASEMAP_PROVIDER = ctx.providers.CartoDB.Positron

VIEWPORT_MARGIN_FRAC: float = 0.05
DPI: int = 300

# Manual fixed viewport override (lon_min, lat_min, lon_max, lat_max) in
# WGS-84. Set to None (default) to always auto-detect from the loaded point
# cloud.
VIEWPORT_OVERRIDE: tuple[float, float, float, float] | None = None

# Standalone legend settings.
SHOW_COUNTS_IN_LEGEND: bool = True
WRAP_LEGEND_LABELS: bool = True
LEGEND_LABEL_WRAP_WIDTH: int = 48


# ==============================================================================
# CAMPAIGN-DERIVED SETTINGS
# ==============================================================================

def _area_label(campaign: str) -> str:
    """Return the campaign label shown in the figure titles."""
    return "Standard campaign" if campaign == "standard" else "Tuk-tuk campaign"


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


def _build_output_path(
    out_dir: Path,
    base_name: str,
    campaign: str,
    variant_tag: str,
    extension: str = "png",
) -> Path:
    """Build an output path embedding the active campaign and variant tag."""
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


SENSOR_FILES: dict = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, FILE_SUFFIX, TUKTUK_FILE_SUFFIX
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
AREA_LABEL: str = _area_label(CAMPAIGN)


# ==============================================================================
# DATA LOADING AND PROJECTION
# ==============================================================================

def wgs84_to_webmercator(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject WGS-84 longitude / latitude arrays to Web Mercator (EPSG:3857)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return x, y


def load_points(sensor_files: dict) -> pd.DataFrame:
    """Load per-sensor CSVs, consolidate lat/lon columns into generic 'lat'/'lon'."""
    frames = []
    for sensor_id, file_path in sensor_files.items():
        tag = sensor_id.upper()
        if not file_path.exists():
            print(f"  [WARNING] File not found — skipping {tag}: {file_path.name}")
            continue
        df = pd.read_csv(file_path, low_memory=False)
        lat_col, lon_col = f"{tag} lat", f"{tag} lon"
        if lat_col not in df.columns or lon_col not in df.columns:
            print(f"  [WARNING] {tag}: missing {lat_col}/{lon_col} — skipping.")
            continue
        df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
        df["sensor"] = sensor_id
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
            "FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def _detect_code_field(columns: list[str]) -> str:
    """Auto-detect the level-4 COS23 code field name in the GeoPackage."""
    for candidate in CODE_FIELD_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(
        f"None of the expected level-4 code fields {CODE_FIELD_CANDIDATES} "
        f"were found in the GeoPackage.\nAvailable columns: {columns}"
    )


def load_cos23_polygons(gpkg_path: Path) -> gpd.GeoDataFrame:
    """
    Load the national COS2023 GeoPackage into a GeoDataFrame in EPSG:4326.

    Auto-detects whichever level-4 code field spelling is present
    (COS23n4_C vs COS23_n4_C) and renames it to the canonical 'COS23_n4_C'
    used by the colouring helpers of this script (cos_code_group,
    build_subcategory_color_map, ...).
    """
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    code_field = _detect_code_field(gdf.columns.tolist())
    if code_field != "COS23_n4_C":
        gdf = gdf.rename(columns={code_field: "COS23_n4_C"})
        print(f"  detected level-4 code field '{code_field}' "
              f"-> renamed to canonical 'COS23_n4_C'")
    else:
        print(f"  detected level-4 code field: '{code_field}'")
    return gdf


def categories_above_threshold(
    df: pd.DataFrame,
    group_col: str,
    threshold: int,
) -> pd.Series:
    """
    Return category counts sorted descending for categories with at least
    `threshold` measurement points.
    """
    counts = df.groupby(group_col).size().sort_values(ascending=False)
    return counts[counts >= threshold]


def resolve_viewport(
    pts: pd.DataFrame,
    viewport_override: tuple[float, float, float, float] | None,
    margin_frac: float,
) -> tuple[float, float, float, float]:
    """
    Compute the Web Mercator axis limits for the map figure.

    If viewport_override is None (default), auto-detects from the point
    cloud with margin_frac applied on all sides.
    """
    if viewport_override is None:
        xmin, xmax = pts["x_merc"].min(), pts["x_merc"].max()
        ymin, ymax = pts["y_merc"].min(), pts["y_merc"].max()

        mx = (xmax - xmin) * margin_frac
        my = (ymax - ymin) * margin_frac

        return xmin - mx, ymin - my, xmax + mx, ymax + my

    lon_min, lat_min, lon_max, lat_max = viewport_override

    x0, y0 = wgs84_to_webmercator(
        np.array([lon_min]),
        np.array([lat_min]),
    )
    x1, y1 = wgs84_to_webmercator(
        np.array([lon_max]),
        np.array([lat_max]),
    )

    return float(x0[0]), float(y0[0]), float(x1[0]), float(y1[0])


def try_add_basemap(ax: plt.Axes) -> None:
    """Fetch and add the OSM basemap tiles to `ax`."""
    try:
        ctx.add_basemap(
            ax,
            source=BASEMAP_PROVIDER,
            crs="EPSG:3857",
            zoom="auto",
            attribution_size=7,
        )
    except Exception as exc:
        print(f"  [WARNING] Could not fetch basemap tiles: {exc}")


def draw_municipality_boundary(ax: plt.Axes) -> None:
    """Overlay the Lisbon municipality boundary from the GADM Level-2 shapefile."""
    shp_path = SHAPEFILE_DIR / "gadm41_PRT_2.shp"

    if not shp_path.is_file():
        print(f"  [warn] GADM shapefile not found: {shp_path}")
        return

    gdf2 = gpd.read_file(shp_path).to_crs(epsg=3857)
    name_col = "NAME_2" if "NAME_2" in gdf2.columns else gdf2.columns[2]
    lisbon = gdf2[gdf2[name_col] == LISBON_MUNICIPALITY_NAME]

    if not lisbon.empty:
        lisbon.boundary.plot(
            ax=ax,
            color="#111111",
            linewidth=1.4,
            zorder=3,
        )
    else:
        print(
            f"  [warn] Municipality '{LISBON_MUNICIPALITY_NAME}' not found "
            f"in column '{name_col}'."
        )


# ==============================================================================
# FIGURE SIZE HELPER
# ==============================================================================

def compute_figsize(
    viewport: tuple[float, float, float, float],
    base: float = 10.0,
) -> tuple[float, float]:
    """Compute a figure size matching the aspect ratio of the viewport."""
    xmin, ymin, xmax, ymax = viewport

    w = xmax - xmin
    h = ymax - ymin
    ratio = w / h

    if ratio >= 1.0:
        return base, base / ratio

    return base * ratio, base


# ==============================================================================
# COLOUR HELPERS
# ==============================================================================

def cos_code_group(code: str, level: int = 2) -> str:
    """
    Return the COS hierarchy group for a code.
    """
    if pd.isna(code):
        return "unknown"

    parts = str(code).split(".")
    level = max(1, min(level, len(parts)))

    return ".".join(parts[:level])


def lighten_color(color, amount: float = 0.0):
    """Lighten a colour by mixing it with white (amount 0.0 = original, 1.0 = white)."""
    r, g, b = mcolors.to_rgb(color)

    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount

    return r, g, b, 1.0


def build_subcategory_color_map(
    categories: list[str],
    label_to_code: dict[str, str],
    group_level: int = 2,
) -> dict:
    """
    Build colours structured by COS hierarchy: categories in the same group
    receive related colours, varied by lightness within the group.
    """
    categories = list(categories)

    if not categories:
        return {}

    category_groups = {
        label: cos_code_group(label_to_code.get(label, "unknown"), group_level)
        for label in categories
    }

    groups = sorted(set(category_groups.values()))

    tab20 = plt.get_cmap("tab20")
    base_group_colors = {}

    for i, group in enumerate(groups):
        if i < tab20.N:
            base_group_colors[group] = tab20(i)
        else:
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.88)
            base_group_colors[group] = (r, g, b, 1.0)

    colors = {}

    for group in groups:
        group_categories = [
            label for label in categories
            if category_groups[label] == group
        ]

        # Keep initial order from `categories`, which is usually count-descending.
        n = len(group_categories)
        base_color = base_group_colors[group]

        if n == 1:
            lightness_values = [0.0]
        else:
            lightness_values = np.linspace(0.0, 0.55, n)

        for label, lightness in zip(group_categories, lightness_values):
            colors[label] = lighten_color(base_color, float(lightness))

    return colors


def build_dynamic_color_map(categories) -> dict:
    """
    Scalable fallback colour logic: tab20+tab20b+tab20c up to 60 categories,
    deterministic HSV colours beyond that.
    """
    categories = list(categories)
    n = len(categories)

    if n == 0:
        return {}

    if n <= 60:
        palette = []

        for cmap_name in ["tab20", "tab20b", "tab20c"]:
            cmap = plt.get_cmap(cmap_name)
            palette.extend([cmap(i) for i in range(cmap.N)])

        return {
            cat: palette[i]
            for i, cat in enumerate(categories)
        }

    golden_ratio_conjugate = 0.618033988749895
    saturation_cycle = [0.62, 0.72, 0.82]
    value_cycle = [0.90, 0.78, 0.66]

    colors = {}

    for i, cat in enumerate(categories):
        hue = (i * golden_ratio_conjugate) % 1.0
        saturation = saturation_cycle[i % len(saturation_cycle)]
        value = value_cycle[(i // len(saturation_cycle)) % len(value_cycle)]

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[cat] = (r, g, b, 1.0)

    return colors


# ==============================================================================
# LEGEND HELPER
# ==============================================================================

def save_standalone_legend(
    colors: dict,
    kept_counts: pd.Series,
    out_png: Path,
    out_pdf: Path,
    other_count: int = 0,
    include_points: bool = False,
    n_points: int | None = None,
) -> None:
    """Save the COS2023 legend as a separate PNG and PDF file."""
    n_categories = len(colors)
    n_extra = int(other_count > 0 and LOW_COUNT_POLYGON_MODE == "grey") + int(include_points)
    n_items = n_categories + n_extra

    if n_items == 0:
        print("  [warn] No legend items to save.")
        return

    if n_items <= 25:
        ncol = 1
    elif n_items <= 60:
        ncol = 2
    elif n_items <= 120:
        ncol = 3
    else:
        ncol = 4

    rows = int(np.ceil(n_items / ncol))

    fig_width = 5.8 * ncol
    fig_height = max(2.8, 0.34 * rows + 1.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    handles = []

    for label, color in colors.items():
        legend_label = str(label)

        if SHOW_COUNTS_IN_LEGEND and label in kept_counts.index:
            legend_label = f"{legend_label} — n={int(kept_counts.loc[label]):,}"

        if WRAP_LEGEND_LABELS:
            legend_label = textwrap.fill(
                legend_label,
                width=LEGEND_LABEL_WRAP_WIDTH,
            )

        handles.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor="none",
                alpha=POLYGON_ALPHA,
                label=legend_label,
            )
        )

    if other_count > 0 and LOW_COUNT_POLYGON_MODE == "grey":
        handles.append(
            mpatches.Patch(
                facecolor=OTHER_POLYGON_COLOR,
                edgecolor="none",
                alpha=POLYGON_ALPHA,
                label=(
                    f"Other COS2023 categories below threshold "
                    f"— {other_count} categories"
                ),
            )
        )

    if include_points:
        if n_points is None:
            point_label = "Measurement points"
        else:
            point_label = f"Measurement points — n={n_points:,}"

        handles.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color="w",
                markerfacecolor=POINT_COLOR,
                markersize=8,
                label=point_label,
            )
        )

    fig.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        title=f"COS2023 category ≥ {MIN_POLYGON_POINTS_COS23} points",
        title_fontsize=10,
        fontsize=8,
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=1.5,
        borderpad=0.8,
        labelspacing=0.75,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"  saved legend (PNG): {out_png}")

    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    print(f"  saved legend (PDF): {out_pdf}")

    plt.close(fig)


# ==============================================================================
# FIG A — POLYGON OVERLAY MAP
# ==============================================================================

def plot_fig_a_polygon_overlay(
    pts: pd.DataFrame,
    kept_counts: pd.Series,
    colors: dict,
    polygons_merc: gpd.GeoDataFrame | None,
    viewport: tuple[float, float, float, float],
    out_png: Path,
    out_pdf: Path,
    out_legend_png: Path,
    out_legend_pdf: Path,
    other_count: int = 0,
) -> None:
    """
    Draw Fig A: semi-transparent COS2023 land-use polygon overlay.
    The legend is exported separately.
    """
    xmin, ymin, xmax, ymax = viewport

    fig, ax = plt.subplots(figsize=compute_figsize(viewport))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    try_add_basemap(ax)

    # COS2023 polygon overlay.
    if polygons_merc is not None:
        visible_categories = set(str(cat) for cat in colors.keys())

        for label, sub in polygons_merc.groupby("cos23_n4_label", dropna=False):
            label_str = str(label)

            if label_str in visible_categories:
                color = colors[label_str]
            else:
                if LOW_COUNT_POLYGON_MODE == "hide":
                    continue
                elif LOW_COUNT_POLYGON_MODE == "grey":
                    color = OTHER_POLYGON_COLOR
                else:
                    raise ValueError(
                        f"Unknown LOW_COUNT_POLYGON_MODE="
                        f"{LOW_COUNT_POLYGON_MODE!r}. "
                        "Choose 'hide' or 'grey'."
                    )

            sub.plot(
                ax=ax,
                color=color,
                alpha=POLYGON_ALPHA,
                edgecolor="none",
                zorder=2,
            )

    # Municipality boundary, Lisbon mode only.
    if DRAW_BOUNDARY:
        draw_municipality_boundary(ax)

    # Measurement points.
    # With POINT_ALPHA = 0, these are hidden.
    ax.scatter(
        pts["x_merc"],
        pts["y_merc"],
        c=POINT_COLOR,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        zorder=4,
        rasterized=True,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(
        f"Measurement Points — {AREA_LABEL}\n"
        f"OpenStreetMap basemap with COS2023 polygon overlay",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    fig.text(
        0.5,
        0.015,
        f"COS2023 categories shown when n ≥ {MIN_POLYGON_POINTS_COS23} "
        f"— low-count mode: {LOW_COUNT_POLYGON_MODE} "
        f"— basemap © OpenStreetMap contributors "
        f"— {CAMPAIGN}/{_VARIANT_TAG}",
        ha="center",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    # Keep the output figure stable.
    # Do not use bbox_inches="tight" for the map, otherwise the final size
    # can change depending on title/footer content.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=DPI)
    print(f"  saved map (PNG): {out_png}")

    fig.savefig(out_pdf, dpi=DPI)
    print(f"  saved map (PDF): {out_pdf}")

    plt.close(fig)

    # Save standalone legend.
    save_standalone_legend(
        colors=colors,
        kept_counts=kept_counts,
        out_png=out_legend_png,
        out_pdf=out_legend_pdf,
        other_count=other_count,
        include_points=POINT_ALPHA > 0,
        n_points=len(pts),
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load points and COS2023 polygons, then draw the overlay map and legend."""
    print("=" * 78)
    print(f"COS2023 OSM map — CAMPAIGN = {CAMPAIGN!r}")
    print(f"  CAMPAIGN      : {CAMPAIGN}")
    if CAMPAIGN == "standard":
        print(f"  FILE_SUFFIX   : {FILE_SUFFIX!r}")
    else:
        print(f"  TUKTUK_FILE_SUFFIX : {TUKTUK_FILE_SUFFIX!r}")
    print(f"  COS_GPKG_FILE : {COS_GPKG_FILE}")
    print(f"  OUTPUT_DIR    : {OUTPUT_DIR}")
    print("=" * 78)

    if LOW_COUNT_POLYGON_MODE not in {"hide", "grey"}:
        raise ValueError(
            f"Invalid LOW_COUNT_POLYGON_MODE={LOW_COUNT_POLYGON_MODE!r}. "
            "Choose 'hide' or 'grey'."
        )

    if COLOR_STRATEGY not in {"subcategory", "dynamic"}:
        raise ValueError(
            f"Invalid COLOR_STRATEGY={COLOR_STRATEGY!r}. "
            "Choose 'subcategory' or 'dynamic'."
        )

    # ------------------------------------------------------------------
    # Step 1 — Load measurement points.
    # ------------------------------------------------------------------
    print("\n[1/6] Loading measurement points...")

    pts = load_points(SENSOR_FILES)
    print(f"  {len(pts):,} points loaded")

    if "cos23_n4_label" not in pts.columns:
        raise KeyError(
            "Column 'cos23_n4_label' not found in the points CSV. "
            "Check the output of join_cos_standard.py / join_cos23_tuktuk.py."
        )

    n_unknown = (pts["cos23_n4_label"] == "Unknown / missing").sum()

    if n_unknown > 0:
        pct = n_unknown / len(pts) * 100.0
        print(
            f"  [warn] {n_unknown:,} points ({pct:.2f}%) labelled "
            f"'Unknown / missing'."
        )

    # ------------------------------------------------------------------
    # Step 2 — Determine point categories passing the threshold.
    # ------------------------------------------------------------------
    print(
        f"\n[2/6] Determining categories "
        f"(threshold ≥ {MIN_POLYGON_POINTS_COS23} pts)..."
    )

    kept_counts = categories_above_threshold(
        pts,
        "cos23_n4_label",
        MIN_POLYGON_POINTS_COS23,
    )

    visible_categories = [str(cat) for cat in kept_counts.index]

    n_total_point_cats = pts["cos23_n4_label"].nunique()
    n_hidden_point_cats = n_total_point_cats - len(visible_categories)

    print(
        f"  {len(visible_categories)} categories selected for display"
    )
    print(
        f"  {n_hidden_point_cats} point categories below threshold"
    )

    # ------------------------------------------------------------------
    # Step 3 — Project points and resolve viewport.
    # ------------------------------------------------------------------
    print("\n[3/6] Projecting points to Web Mercator and resolving viewport...")

    x_pts, y_pts = wgs84_to_webmercator(
        pts["lon"].values,
        pts["lat"].values,
    )

    pts["x_merc"] = x_pts
    pts["y_merc"] = y_pts

    viewport = resolve_viewport(
        pts,
        VIEWPORT_OVERRIDE,
        VIEWPORT_MARGIN_FRAC,
    )

    print(
        f"  viewport EPSG:3857 = "
        f"({viewport[0]:.0f}, {viewport[1]:.0f}, "
        f"{viewport[2]:.0f}, {viewport[3]:.0f})"
    )

    # ------------------------------------------------------------------
    # Step 4 — Load and project COS2023 polygons.
    # ------------------------------------------------------------------
    polygons_merc = None

    if COS_GPKG_FILE.exists():
        print("\n[4/6] Loading and projecting COS2023 polygons...")

        polygons = load_cos23_polygons(COS_GPKG_FILE)

        polygons["cos23_n4_label"] = polygons["COS23_n4_C"].apply(
            cos23.get_label
        )

        polygons["cos23_n4_label"] = polygons["cos23_n4_label"].astype(str)
        polygons["COS23_n4_C"] = polygons["COS23_n4_C"].astype(str)

        polygons_merc = polygons.to_crs(epsg=3857)

        print(f"  {len(polygons_merc):,} polygons projected")

    else:
        print(
            f"\n[4/6] COS2023 GeoPackage not found — map will be drawn "
            f"without polygon overlay.\n  ({COS_GPKG_FILE})\n"
            f"  Download from:\n"
            f"    https://geo2.dgterritorio.gov.pt/cos/S2/COS2023/COS2023v1-S2-gpkg.zip"
        )

    # ------------------------------------------------------------------
    # Step 5 — Build category colour map.
    # ------------------------------------------------------------------
    print("\n[5/6] Building category colour map...")

    label_to_code: dict[str, str] = {}

    if polygons_merc is not None:
        label_code_df = (
            polygons_merc[["cos23_n4_label", "COS23_n4_C"]]
            .dropna()
            .drop_duplicates()
        )

        label_to_code = dict(
            zip(
                label_code_df["cos23_n4_label"].astype(str),
                label_code_df["COS23_n4_C"].astype(str),
            )
        )

    if COLOR_STRATEGY == "subcategory":
        colors = build_subcategory_color_map(
            categories=visible_categories,
            label_to_code=label_to_code,
            group_level=COLOR_GROUP_LEVEL,
        )
        print(
            f"  colour strategy: subcategory "
            f"(COS group level {COLOR_GROUP_LEVEL})"
        )

    elif COLOR_STRATEGY == "dynamic":
        colors = build_dynamic_color_map(visible_categories)
        print("  colour strategy: dynamic palette")

    else:
        raise ValueError(
            f"Unknown COLOR_STRATEGY={COLOR_STRATEGY!r}. "
            "Choose 'subcategory' or 'dynamic'."
        )

    if LOW_COUNT_POLYGON_MODE == "grey":
        other_count = max(0, n_hidden_point_cats)
    else:
        other_count = 0

    print(f"  {len(colors)} visible categories assigned colours")

    if LOW_COUNT_POLYGON_MODE == "hide":
        print("  categories below threshold will be hidden")
    else:
        print("  categories below threshold will be drawn in grey")

    # ------------------------------------------------------------------
    # Step 6 — Draw and save map + separate legend.
    # ------------------------------------------------------------------
    print("\n[6/6] Drawing map and standalone legend...")

    plot_fig_a_polygon_overlay(
        pts=pts,
        kept_counts=kept_counts,
        colors=colors,
        polygons_merc=polygons_merc,
        viewport=viewport,
        out_png=_build_output_path(
            OUTPUT_DIR,
            "cos23_map_polygon_overlay",
            CAMPAIGN,
            _VARIANT_TAG,
            "png",
        ),
        out_pdf=_build_output_path(
            OUTPUT_DIR,
            "cos23_map_polygon_overlay",
            CAMPAIGN,
            _VARIANT_TAG,
            "pdf",
        ),
        out_legend_png=_build_output_path(
            OUTPUT_DIR,
            "cos23_map_polygon_overlay_legend",
            CAMPAIGN,
            _VARIANT_TAG,
            "png",
        ),
        out_legend_pdf=_build_output_path(
            OUTPUT_DIR,
            "cos23_map_polygon_overlay_legend",
            CAMPAIGN,
            _VARIANT_TAG,
            "pdf",
        ),
        other_count=other_count,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()