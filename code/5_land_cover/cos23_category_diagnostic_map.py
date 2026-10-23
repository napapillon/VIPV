"""
================================================================================
COS2023 Category Diagnostic Map
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Diagnostic map for COS2023-joined sensor data. Displays the measurement
points coloured by their COS2023 category on an OSM basemap.

Points belonging to FILTER_COS23_CATEGORIES are drawn in distinct tab20
colours on top. All other points are drawn in neutral grey underneath, so the
spatial context of the full route is kept while the selected categories stay
visually dominant.

Answers "where exactly are my measurements labelled as category X?" — e.g.
checking that "Road network" points fall on actual roads, or investigating
why "Unknown / missing" points cluster in a specific area.

Category filter
---------------
FILTER_COS23_CATEGORIES lists the COS2023 level-4 categories to highlight.
Each entry can be the exact label or the level-4 code (e.g. "1.5.1.1" for
"Road network"); codes are resolved to labels with
cos23_mapping.COS23_NOMENCLATURE before any filtering, so both forms give the
same outputs.

An empty list [] is the exploratory mode: every category is shown, but those
below EXPLORATORY_MIN_POINTS are grouped into a single grey "Other" entry,
which keeps the legend readable (up to 93 COS2023 level-4 categories).

    FILTER_COS23_CATEGORIES = ["1.5.1.1"]              # code for "Road network"
    FILTER_COS23_CATEGORIES = ["Road network"]          # same, full label
    FILTER_COS23_CATEGORIES = ["1.1.1.1", "1.1.2.1"]    # two residential codes
    FILTER_COS23_CATEGORIES = ["Unknown / missing"]     # unmatched points
    FILTER_COS23_CATEGORIES = []                        # exploratory mode

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
    input  : BOUNDARIES_DIR / gadm41_PRT_2.shp (optional, Lisbon outline)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             cos23_diag_map_<CAMPAIGN>_<variant_tag>_<filter_tag>.png
                 (Fig A — scatter)
             cos23_diag_hexbin_<CAMPAIGN>_<variant_tag>_<filter_tag>.png
                 (Fig B — hexbin, if SHOW_HEXBIN)
             cos23_diag_heatmap_<CAMPAIGN>_<variant_tag>_<filter_tag>.png
                 (Fig C — KDE, if SHOW_KDE)

Requirements
------------
    pip install pandas numpy matplotlib geopandas contextily shapely pyproj scipy

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.3
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import re
import sys
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BOUNDARIES_DIR, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import contextily as ctx
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import gaussian_kde

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

# ---- Category filter ---------------------------------------------------------
# COS2023 level-4 categories to highlight — each entry can be the exact
# label string OR the shorter level-4 code (e.g. "1.5.1.1" instead of
# "Road network"); codes are resolved to labels at startup.
# Use [] to show all categories in distinct colours (exploratory mode).
# Use ["Unknown / missing"] to diagnose unmatched points.
FILTER_COS23_CATEGORIES: list[str] = [
    "Unknown / missing",
]

# Exploratory mode only (FILTER_COS23_CATEGORIES = []): categories below
# this point count are grouped into a single grey "Other" entry, keeping
# the legend readable when many COS2023 classes are present. Matches the
# project-wide COS23 threshold convention (cos23_urban_breakdown.py,
# cos23_osm_polygone_map.py).
EXPLORATORY_MIN_POINTS: int = 600

# ---- Output ------------------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "5_land_cover" / "cos23_diag_outputs"

# ---- GADM shapefiles (optional) ----------------------------------------------
# Used to draw the Lisbon municipality boundary outline on top of the basemap.
# Set to None to skip (outline will be absent).
SHAPEFILE_DIR: Path | None = BOUNDARIES_DIR
LISBON_MUNICIPALITY_NAME: str = "Lisboa"

# ---- Viewport ----------------------------------------------------------------
# The viewport is auto-detected from the measurement point cloud (with
# VIEWPORT_MARGIN_FRAC applied on all sides) for every campaign/variant.
# Set VIEWPORT_OVERRIDE to a fixed (lon_min, lat_min, lon_max, lat_max) tuple
# if you need a specific fixed view regardless of the loaded data's extent.
VIEWPORT_OVERRIDE: tuple[float, float, float, float] | None = None
VIEWPORT_MARGIN_FRAC: float = 0.08   # extra margin (auto-detection only)

# ---- Rendering parameters ----------------------------------------------------
# BASEMAP_PROVIDER = ctx.providers.OpenStreetMap.Mapnik
BASEMAP_PROVIDER = ctx.providers.CartoDB.Positron   # lighter alternative

# Style for points NOT in FILTER_COS23_CATEGORIES (background context layer).
OTHER_COLOR: str   = "#BBBBBB"
OTHER_ALPHA: float = 0.25
OTHER_SIZE:  float = 4.0

# Style for highlighted category points.
HIGHLIGHT_ALPHA: float = 0.80
HIGHLIGHT_SIZE:  float = 8.0

DPI: int = 300

# ---- Figure toggles ----------------------------------------------------------
# Fig B (hexbin) is fast (~instant) and on by default.
# Fig C (KDE) is slow on large datasets and off by default; enable when you
# need a smooth continuous density surface rather than discrete hex cells.
SHOW_HEXBIN: bool = True
SHOW_KDE:    bool = False

# ---- Hexbin density map (Fig B) parameters -----------------------------------
HEXBIN_CELL_SIZE_M: int = 200

# Minimum number of points for a hex cell to be rendered. Cells with fewer
# points are left transparent (basemap shows through). Raising this value
# removes isolated low-count cells and focuses attention on real clusters.
HEXBIN_MIN_COUNT: int = 3

# Colormap for the hexbin map. "YlOrRd" on a light basemap is readable.
HEXBIN_COLORMAP: str = "YlOrRd"

# Alpha transparency of the hexbin overlay.
HEXBIN_ALPHA: float = 0.75

# ---- Heatmap (KDE) parameters (Fig C) ---------------------------------------
# Bandwidth for the Gaussian kernel density estimate.
# "scott" lets scipy choose automatically. Or pass a value in EPSG:3857 metres.
KDE_BANDWIDTH: float | str = "scott"

# Number of grid cells per axis for the KDE evaluation grid.
KDE_GRID_STEPS: int = 300

# Colormap for the KDE heatmap.
KDE_COLORMAP: str = "YlOrRd"

# Alpha transparency of the KDE overlay.
KDE_ALPHA: float = 0.55

# Percentile used to cap the KDE colour scale maximum.
# Values above this percentile are saturated to the maximum colour, preventing
# a single dominant hot spot from collapsing all other variation to near-zero.
# Set to 100.0 to disable clipping and use the true maximum.
KDE_VMAX_PERCENTILE: float = 99.0

# ==============================================================================
# DERIVED SETTINGS  (resolved automatically — do not edit)
# ==============================================================================

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


def _resolve_filter_categories(raw: list[str]) -> list[str]:
    """
    Resolve FILTER_COS23_CATEGORIES entries, allowing COS2023 level-4 codes
    (e.g. "1.5.1.1") as shorthand for their label. Entries not
    found in COS23_NOMENCLATURE are assumed to already be a label and are
    passed through unchanged.
    """
    return [cos23.COS23_NOMENCLATURE.get(entry, entry) for entry in raw]


def _make_filter_tag(categories: list[str]) -> str:
    """
    Build a short, filesystem-safe tag from the active category filter,
    for embedding in the output filename.

    Rules:
    - Empty list  → "all_categories"
    - Single item → first 60 chars of the slugified label (lowercased,
                    non-alphanumeric → underscore).
    - Multiple    → "N_categories" (e.g. "3_categories")

    The tag is embedded in the output filename so different filter runs
    never overwrite each other.
    """
    if not categories:
        return "all_categories"
    if len(categories) == 1:
        slug = re.sub(r"[^a-z0-9]+", "_", categories[0].lower()).strip("_")
        return slug[:60]
    return f"{len(categories)}_categories"


def _build_output_path(out_dir: Path, campaign: str, variant_tag: str, filter_tag: str) -> Path:
    """Return the Fig A scatter map output path."""
    return out_dir / f"cos23_diag_map_{campaign}_{variant_tag}_{filter_tag}.png"


def _build_hexbin_path(out_dir: Path, campaign: str, variant_tag: str, filter_tag: str) -> Path:
    """Return the Fig B hexbin map output path."""
    return out_dir / f"cos23_diag_hexbin_{campaign}_{variant_tag}_{filter_tag}.png"


def _build_heatmap_path(out_dir: Path, campaign: str, variant_tag: str, filter_tag: str) -> Path:
    """Return the Fig C KDE heatmap output path."""
    return out_dir / f"cos23_diag_heatmap_{campaign}_{variant_tag}_{filter_tag}.png"


SENSOR_FILES: dict = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, FILE_SUFFIX, TUKTUK_FILE_SUFFIX
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR:   Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
FILTER_COS23_CATEGORIES = _resolve_filter_categories(FILTER_COS23_CATEGORIES)
FILTER_TAG:   str  = _make_filter_tag(FILTER_COS23_CATEGORIES)

# ==============================================================================
# HELPERS
# ==============================================================================

def wgs84_to_webmercator(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject WGS-84 lon/lat arrays to Web Mercator (EPSG:3857)."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return t.transform(np.asarray(lon, dtype=float),
                       np.asarray(lat, dtype=float))


def bbox_wgs84_to_mercator(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    margin_frac: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Convert a WGS-84 bounding box to Web Mercator axis limits
    (xmin, ymin, xmax, ymax) with an optional fractional margin.

    wgs84_to_webmercator always returns 1-D arrays; [0] indexing extracts
    plain Python floats compatible with matplotlib axis limit setters.
    """
    x0, y0 = wgs84_to_webmercator(np.array([lon_min]), np.array([lat_min]))
    x1, y1 = wgs84_to_webmercator(np.array([lon_max]), np.array([lat_max]))
    xmin, xmax = float(x0[0]), float(x1[0])
    ymin, ymax = float(y0[0]), float(y1[0])
    mx = (xmax - xmin) * margin_frac
    my = (ymax - ymin) * margin_frac
    return xmin - mx, ymin - my, xmax + mx, ymax + my


def load_points(sensor_files: dict) -> pd.DataFrame:
    """
    Load per-sensor CSVs and consolidate lat/lon columns into generic
    'lat'/'lon'. Returns rows with 'lat', 'lon', 'cos23_n4_label';
    missing-coordinate rows dropped.
    """
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


def build_color_map(categories: list[str]) -> dict[str, tuple]:
    """Assign stable tab20 colours to a list of category labels."""
    cmap = plt.get_cmap("tab20")
    return {cat: cmap(i % 20) for i, cat in enumerate(categories)}


def categories_above_threshold(
    df: pd.DataFrame, group_col: str, threshold: int
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


def resolve_viewport(
    pts: pd.DataFrame,
    viewport_override: tuple[float, float, float, float] | None,
    margin_frac: float,
) -> tuple[float, float, float, float]:
    """
    Compute Web Mercator axis limits for the map figure.

    If viewport_override is None (default), auto-detects from the point
    cloud with margin_frac applied on all sides. Otherwise, projects the
    given (lon_min, lat_min, lon_max, lat_max) WGS-84 box directly.
    """
    if viewport_override is None:
        xmin, xmax = pts["x_merc"].min(), pts["x_merc"].max()
        ymin, ymax = pts["y_merc"].min(), pts["y_merc"].max()
        mx = (xmax - xmin) * margin_frac
        my = (ymax - ymin) * margin_frac
        return xmin - mx, ymin - my, xmax + mx, ymax + my
    else:
        lon_min, lat_min, lon_max, lat_max = viewport_override
        return bbox_wgs84_to_mercator(lon_min, lat_min, lon_max, lat_max)


def try_add_basemap(ax: plt.Axes) -> None:
    """Add OSM basemap tiles; warn and continue if unreachable."""
    try:
        ctx.add_basemap(ax, source=BASEMAP_PROVIDER, crs="EPSG:3857",
                        zoom="auto", attribution_size=6)
    except Exception as exc:
        print(f"  [WARNING] Could not fetch basemap tiles: {exc}")


def draw_lisbon_boundary(ax: plt.Axes) -> None:
    """Draw the Lisbon municipality boundary from the GADM Level-2 shapefile.
    Skips gracefully if SHAPEFILE_DIR is None or the file is not found."""
    if SHAPEFILE_DIR is None:
        return
    shp_path = SHAPEFILE_DIR / "gadm41_PRT_2.shp"
    if not shp_path.is_file():
        print(f"  [warn] GADM shapefile not found: {shp_path}")
        return
    gdf2     = gpd.read_file(shp_path).to_crs(epsg=3857)
    name_col = "NAME_2" if "NAME_2" in gdf2.columns else gdf2.columns[2]
    lisbon   = gdf2[gdf2[name_col] == LISBON_MUNICIPALITY_NAME]
    if not lisbon.empty:
        lisbon.boundary.plot(ax=ax, color="#111111", linewidth=1.5, zorder=5)
    else:
        print(f"  [warn] '{LISBON_MUNICIPALITY_NAME}' not found in GADM column "
              f"'{name_col}'.")


def build_legend_handles(
    pts: pd.DataFrame,
    highlight_cats: list[str],
    colors: dict[str, tuple],
) -> list:
    """
    Build legend handles for the highlighted categories + "Other" group.
    `highlight_cats` is pre-resolved by the caller (either the user filter,
    or the thresholded category list in exploratory mode).
    """
    handles = []
    for cat in highlight_cats:
        n = (pts["cos23_n4_label"] == cat).sum()
        handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=colors[cat], markersize=8,
                label=f"{cat}\n(n={n:,})",
            )
        )
    n_other = (~pts["cos23_n4_label"].isin(highlight_cats)).sum()
    if n_other > 0:
        handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=OTHER_COLOR, markersize=6,
                label=f"Other categories (n={n_other:,})",
            )
        )
    return handles


# ==============================================================================
# HEXAGONAL BIN DENSITY MAP  (Fig B — fast, default ON)
# ==============================================================================

def plot_hexbin(
    highlight_pts: pd.DataFrame,
    viewport: tuple[float, float, float, float],
    title: str,
    n_shown: int,
    n_total: int,
    out_path: Path,
) -> None:
    """
    Draw a hexagonal bin density map of the highlighted category points,
    coloured by point count on a log scale (near-instant vs. KDE, at the
    cost of a discrete grid instead of a continuous surface).

    Cell size is controlled by HEXBIN_CELL_SIZE_M; gridsize is derived
    automatically from the viewport width. Cells below HEXBIN_MIN_COUNT
    are left transparent.
    """
    if highlight_pts.empty:
        print("  [hexbin] no highlighted points — skipped.")
        return

    xmin, ymin, xmax, ymax = viewport

    # Derive gridsize so each hexagon covers ~HEXBIN_CELL_SIZE_M metres.
    # matplotlib hexbin uses gridsize = number of hexagons along the x-axis.
    viewport_width_m = xmax - xmin
    gridsize = max(10, int(viewport_width_m / HEXBIN_CELL_SIZE_M))

    fig, ax = plt.subplots(figsize=compute_figsize(viewport))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    try_add_basemap(ax)

    # Draw hexbin with log-normalised colour scale.
    # mincnt clips cells below the minimum count threshold (they are rendered
    # as transparent, so the basemap shows through in empty areas).
    hb = ax.hexbin(
        highlight_pts["x_merc"].values,
        highlight_pts["y_merc"].values,
        gridsize=gridsize,
        cmap=HEXBIN_COLORMAP,
        bins="log",           # logarithmic colour scale
        mincnt=HEXBIN_MIN_COUNT,
        alpha=HEXBIN_ALPHA,
        linewidths=0.0,
        zorder=3,
        extent=(xmin, xmax, ymin, ymax),
    )

    # Colour bar.
    cbar = fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Point count per cell (log scale)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    draw_lisbon_boundary(ax)

    ax.set_title(
        f"{title}\n— hexagonal bin density map ({HEXBIN_CELL_SIZE_M} m cells) —",
        fontsize=11, fontweight="bold", pad=12,
    )
    fig.text(
        0.5, -0.01,
        f"[COS2023 diagnostic — hexbin]  {CAMPAIGN}/{_VARIANT_TAG}  —  "
        f"filter: {FILTER_TAG}  —  "
        f"{n_shown:,} pts / {n_total:,} total  —  "
        f"cell size ≈ {HEXBIN_CELL_SIZE_M} m  —  "
        f"min count = {HEXBIN_MIN_COUNT}  —  "
        f"basemap © OpenStreetMap contributors",
        ha="center", fontsize=7, color="#888888", style="italic",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  saved (hexbin): {out_path}")
    plt.close(fig)


# ==============================================================================
# KDE HEATMAP FIGURE  (Fig C — slow, default OFF)
# ==============================================================================

def plot_heatmap(
    highlight_pts: pd.DataFrame,
    viewport: tuple[float, float, float, float],
    title: str,
    n_shown: int,
    n_total: int,
    out_path: Path,
) -> None:
    """
    Draw a Gaussian KDE heatmap of the highlighted category points, revealing
    spatial concentration that overlapping points hide on a scatter plot
    (e.g. a vehicle parked at one spot for hours). Evaluated on a regular
    grid over the viewport and rendered as a semi-transparent overlay; the
    lowest-density cells are clipped to full transparency. Skipped in
    exploratory mode (no filter — all categories combined, not meaningful).
    """
    if len(highlight_pts) < 2:
        print("  [heatmap] fewer than 2 highlighted points — skipped.")
        return

    xmin, ymin, xmax, ymax = viewport

    # ── KDE evaluation on a regular grid ──────────────────────────────
    xi = np.linspace(xmin, xmax, KDE_GRID_STEPS)
    yi = np.linspace(ymin, ymax, KDE_GRID_STEPS)
    xx, yy = np.meshgrid(xi, yi)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values    = np.vstack([
        highlight_pts["x_merc"].values,
        highlight_pts["y_merc"].values,
    ])

    try:
        kernel = gaussian_kde(values, bw_method=KDE_BANDWIDTH)
        zz = kernel(positions).reshape(xx.shape)
    except np.linalg.LinAlgError:
        # Collinear points (e.g. single parked-car route) -> fixed bandwidth.
        print("  [heatmap] KDE singular matrix — retrying with bw=0.1")
        kernel = gaussian_kde(values, bw_method=0.1)
        zz = kernel(positions).reshape(xx.shape)

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=compute_figsize(viewport))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    try_add_basemap(ax)

    # ── Colour normalisation with percentile cap ──────────────────────
    # Caps at percentile_N instead of max so a single dominant hot spot
    # does not compress the rest of the scale to near-zero.
    vmax   = float(np.percentile(zz, KDE_VMAX_PERCENTILE))
    vmin   = float(zz.min())
    z_norm = np.clip((zz - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)

    cmap   = plt.get_cmap(KDE_COLORMAP)
    rgba   = cmap(z_norm)
    # Cells below the 5th percentile made fully transparent (basemap shows through).
    alpha_threshold = np.percentile(z_norm, 5)
    rgba[..., 3] = np.where(
        z_norm <= alpha_threshold, 0.0, KDE_ALPHA * z_norm
    )

    ax.imshow(
        rgba,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        aspect="auto",
        zorder=3,
        interpolation="bilinear",
    )

    # Colour bar — scale reflects the clipped range [vmin, vmax].
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label(
        f"KDE density [a.u.]\n(colour scale clipped at p{KDE_VMAX_PERCENTILE:.0f})",
        fontsize=7,
    )
    cbar.ax.tick_params(labelsize=7)

    draw_lisbon_boundary(ax)

    ax.set_title(
        f"{title}\n— KDE density heatmap —",
        fontsize=11, fontweight="bold", pad=12,
    )
    fig.text(
        0.5, -0.01,
        f"[COS2023 diagnostic — heatmap]  {CAMPAIGN}/{_VARIANT_TAG}  —  "
        f"filter: {FILTER_TAG}  —  "
        f"{n_shown:,} pts in KDE / {n_total:,} total  —  "
        f"bandwidth: {KDE_BANDWIDTH}  —  "
        f"colour scale clipped at p{KDE_VMAX_PERCENTILE:.0f}  —  "
        f"basemap © OpenStreetMap contributors",
        ha="center", fontsize=7, color="#888888", style="italic",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  saved (heatmap): {out_path}")
    plt.close(fig)


# ==============================================================================
# FIGURE SIZE HELPER
# ==============================================================================

def compute_figsize(
    viewport: tuple[float, float, float, float],
    base: float = 10.0,
) -> tuple[float, float]:
    """Compute a figure size matching the aspect ratio of the viewport."""
    xmin, ymin, xmax, ymax = viewport
    ratio = (xmax - xmin) / (ymax - ymin)
    if ratio >= 1.0:
        return base, base / ratio
    else:
        return base * ratio, base


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the sensor points and draw the COS2023 diagnostic maps."""
    output_path = _build_output_path(OUTPUT_DIR, CAMPAIGN, _VARIANT_TAG, FILTER_TAG)

    print("=" * 78)
    print(f"COS2023 category diagnostic map  —  CAMPAIGN = {CAMPAIGN!r}")
    print(f"  CAMPAIGN           : {CAMPAIGN}")
    if CAMPAIGN == "standard":
        print(f"  FILE_SUFFIX        : {FILE_SUFFIX!r}")
    else:
        print(f"  TUKTUK_FILE_SUFFIX : {TUKTUK_FILE_SUFFIX!r}")
    print(f"  OUTPUT_FILE        : {output_path}")
    if FILTER_COS23_CATEGORIES:
        n_cats = len(FILTER_COS23_CATEGORIES)
        print(
            f"  Filter ({n_cats} categor"
            f"{'y' if n_cats == 1 else 'ies'}):"
        )
        for cat in FILTER_COS23_CATEGORIES:
            print(f"    • {cat}")
    else:
        print(f"  Filter: none (exploratory mode — categories ≥ {EXPLORATORY_MIN_POINTS} pts shown)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Step 1 — Load and inspect points.
    # ------------------------------------------------------------------
    print("\n[1/4] Loading measurement points...")
    pts = load_points(SENSOR_FILES)
    print(f"  {len(pts):,} points loaded")

    n_unknown = (pts["cos23_n4_label"] == "Unknown / missing").sum()
    if n_unknown > 0:
        print(
            f"  [info] {n_unknown:,} points ({n_unknown / len(pts) * 100:.1f}%) "
            f"labelled 'Unknown / missing' — add to filter to diagnose."
        )

    # Validate filter labels and print per-category counts.
    if FILTER_COS23_CATEGORIES:
        present = set(pts["cos23_n4_label"].unique())
        for cat in FILTER_COS23_CATEGORIES:
            if cat not in present:
                print(
                    f"  [warn] Category not found in data: {cat!r}\n"
                    f"         Check spelling against cos23_mapping."
                    f"COS23_NOMENCLATURE."
                )
            else:
                n = (pts["cos23_n4_label"] == cat).sum()
                print(
                    f"  → {cat!r}: {n:,} pts "
                    f"({n / len(pts) * 100:.1f}%)"
                )

    # ------------------------------------------------------------------
    # Step 2 — Project to Web Mercator.
    # ------------------------------------------------------------------
    print("\n[2/4] Projecting points to Web Mercator...")
    pts["x_merc"], pts["y_merc"] = wgs84_to_webmercator(
        pts["lon"].values, pts["lat"].values
    )

    # ------------------------------------------------------------------
    # Step 3 — Resolve viewport and category colour map.
    # ------------------------------------------------------------------
    print("\n[3/4] Resolving viewport and categories...")
    viewport = resolve_viewport(
        pts, VIEWPORT_OVERRIDE, VIEWPORT_MARGIN_FRAC,
    )
    print(
        f"  (xmin={viewport[0]:.0f}, ymin={viewport[1]:.0f}, "
        f"xmax={viewport[2]:.0f}, ymax={viewport[3]:.0f})  [EPSG:3857 m]"
    )

    colors = build_color_map(FILTER_COS23_CATEGORIES)

    # Exploratory mode: threshold categories the same way as the rest of
    # the pipeline, grouping low-count classes into "Other" so the legend
    # stays readable with up to 93 possible COS2023 level-4 categories.
    n_excl_cat = n_excl_pts = 0
    if FILTER_COS23_CATEGORIES:
        highlight_cats = FILTER_COS23_CATEGORIES
    else:
        kept_exp, n_excl_cat, n_excl_pts = categories_above_threshold(
            pts, "cos23_n4_label", EXPLORATORY_MIN_POINTS
        )
        highlight_cats = kept_exp.index.tolist()
        colors = build_color_map(highlight_cats)
        print(
            f"  {len(highlight_cats)} categories ≥ {EXPLORATORY_MIN_POINTS} pts, "
            f"{n_excl_cat} grouped as 'Other' ({n_excl_pts:,} pts)"
        )

    # ------------------------------------------------------------------
    # Step 4 — Draw and save figures (Fig A: scatter, Fig B: heatmap).
    # ------------------------------------------------------------------
    print("\n[4/4] Drawing figures...")

    fig, ax = plt.subplots(figsize=compute_figsize(viewport))
    ax.set_xlim(viewport[0], viewport[2])
    ax.set_ylim(viewport[1], viewport[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    try_add_basemap(ax)

    # Grey "Other" layer rendered first (lowest z-order): points below
    # threshold (exploratory mode) or outside the filter (filter mode).
    is_other  = ~pts["cos23_n4_label"].isin(highlight_cats)
    other_pts = pts[is_other]
    if not other_pts.empty:
        ax.scatter(
            other_pts["x_merc"].values, other_pts["y_merc"].values,
            color=OTHER_COLOR, s=OTHER_SIZE, alpha=OTHER_ALPHA,
            linewidths=0, zorder=3, rasterized=True,
        )
    # Highlighted categories on top, in the resolved order (filter order,
    # or decreasing frequency in exploratory mode).
    for cat in highlight_cats:
        sub = pts[pts["cos23_n4_label"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["x_merc"].values, sub["y_merc"].values,
            color=colors[cat], s=HIGHLIGHT_SIZE, alpha=HIGHLIGHT_ALPHA,
            linewidths=0, zorder=4, rasterized=True,
        )

    # Lisbon boundary (drawn above points for visibility).
    draw_lisbon_boundary(ax)

    # ── Legend ────────────────────────────────────────────────────────
    handles = build_legend_handles(pts, highlight_cats, colors)
    ax.legend(
        handles=handles, fontsize=7, loc="lower left",
        framealpha=0.90, title="COS2023 category", title_fontsize=8,
    )

    # ── Title ─────────────────────────────────────────────────────────
    if FILTER_COS23_CATEGORIES:
        if len(FILTER_COS23_CATEGORIES) == 1:
            title = FILTER_COS23_CATEGORIES[0]
        else:
            short = [
                (c[:28] + "…" if len(c) > 28 else c)
                for c in FILTER_COS23_CATEGORIES
            ]
            title = f"{len(FILTER_COS23_CATEGORIES)} categories — " + ", ".join(short)
    else:
        title = f"All COS2023 categories ≥ {EXPLORATORY_MIN_POINTS} pts (exploratory mode)"

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    # ── Footnote ──────────────────────────────────────────────────────
    n_shown = sum((pts["cos23_n4_label"] == cat).sum() for cat in highlight_cats)
    excl_note = (
        f"  —  {n_excl_cat} categories grouped as 'Other' ({n_excl_pts:,} pts)"
        if not FILTER_COS23_CATEGORIES else ""
    )
    fig.text(
        0.5, -0.01,
        f"[COS2023 diagnostic]  {CAMPAIGN}/{_VARIANT_TAG}  —  "
        f"filter: {FILTER_TAG}  —  "
        f"{n_shown:,} highlighted / {len(pts):,} total{excl_note}  —  "
        f"basemap © OpenStreetMap contributors",
        ha="center", fontsize=7, color="#888888", style="italic",
    )

    # ── Save scatter map (Fig A) ───────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"\n  saved (scatter): {output_path}")
    plt.close(fig)

    # Collect highlighted points once, shared by Fig B and Fig C.
    if FILTER_COS23_CATEGORIES:
        highlight_mask = pts["cos23_n4_label"].isin(FILTER_COS23_CATEGORIES)
        highlight_pts  = pts[highlight_mask]
    else:
        highlight_pts = pd.DataFrame()   # not used in exploratory mode

    # ── Fig B — Hexbin density map (fast, default ON) ─────────────────
    if SHOW_HEXBIN and FILTER_COS23_CATEGORIES:
        print("\n  [Fig B] Drawing hexbin density map...")
        plot_hexbin(
            highlight_pts=highlight_pts,
            viewport=viewport,
            title=title,
            n_shown=n_shown,
            n_total=len(pts),
            out_path=_build_hexbin_path(OUTPUT_DIR, CAMPAIGN, _VARIANT_TAG, FILTER_TAG),
        )
    elif not FILTER_COS23_CATEGORIES:
        print(
            "\n  [Fig B] Hexbin skipped in exploratory mode "
            "(all categories combined — not meaningful)."
        )
    else:
        print("\n  [Fig B] Hexbin disabled (SHOW_HEXBIN = False).")

    # ── Fig C — KDE heatmap (slow, default OFF) ───────────────────────
    if SHOW_KDE and FILTER_COS23_CATEGORIES:
        print("\n  [Fig C] Drawing KDE heatmap (this may take a while)...")
        plot_heatmap(
            highlight_pts=highlight_pts,
            viewport=viewport,
            title=title,
            n_shown=n_shown,
            n_total=len(pts),
            out_path=_build_heatmap_path(OUTPUT_DIR, CAMPAIGN, _VARIANT_TAG, FILTER_TAG),
        )
    elif not FILTER_COS23_CATEGORIES:
        print(
            "\n  [Fig C] KDE skipped in exploratory mode "
            "(all categories combined — not meaningful)."
        )
    else:
        print("\n  [Fig C] KDE disabled (SHOW_KDE = False).")

    print("\nDone.")


if __name__ == "__main__":
    main()