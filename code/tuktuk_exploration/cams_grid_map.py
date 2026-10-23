"""
================================================================================
CAMS Discretisation Grid Map — Tuk-Tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Publication-ready map of the spatial discretisation grid over the tuk-tuk
GPS measurement points of the Lisbon campaign.

The grid is the one used to limit CAMS API calls: GPS coordinates are
snapped to the nearest node of a regular grid of spacing GRID_DEG, and one
CAMS time series is downloaded per distinct node (join_cams_tuktuk.py, same
GRID_DEG). It is a methodological choice, not an intrinsic CAMS spatial
resolution (the CRS-TIME service interpolates to any requested coordinate).

Layout
------
Single-panel map:
  - OpenStreetMap basemap (CartoDB Positron, via contextily; skipped if
    contextily is missing or the tiles cannot be fetched)
  - grid cells, colour-filled and labelled with their node coordinates
  - grid-node markers
  - GPS measurement points (alpha-controlled; hidden if SHOW_POINTS = False)
  - FCUL reference station marker
  - legend, scale bar, figure annotation

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : TUKTUK_DIR / j33_tuktuk.csv  (decode_tuktuk.py output, or any
             downstream variant such as j33_tuktuk_ghi.csv)
             columns: 'lat', 'lon'
    output : OUTPUT_ROOT / tuktuk_exploration / cams_grid /
             cams_grid_map_tuktuk.png

Requirements
------------
    pip install pandas numpy matplotlib pyproj contextily

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math
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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pyproj import Transformer

try:
    import contextily as ctx
    CTX_AVAILABLE = True
except ImportError:
    CTX_AVAILABLE = False
    print("[INFO] contextily not installed — basemap will be skipped.")

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input file ---------------------------------------------------------------
# Decoded tuk-tuk CSV (output of decode_tuktuk.py or any downstream variant).
# Required columns: lat, lon.
INPUT_FILE: Path = TUKTUK_DIR / "j33_tuktuk.csv"

# --- Output -------------------------------------------------------------------
OUTPUT_DIR:  Path = OUTPUT_ROOT / "tuktuk_exploration" / "cams_grid"
OUTPUT_FILE: str  = "cams_grid_map_tuktuk.png"

# --- Spatial discretisation grid ----------------------------------------------
# Spacing of the regular grid used to snap GPS coordinates and limit API calls
# (same value as GRID_DEG in join_cams_tuktuk.py).
# This is a methodological choice — not an intrinsic CAMS spatial resolution
# (the CRS-TIME service interpolates to any requested coordinate on the fly).
GRID_DEG: float = 0.05

# --- Map extent margin --------------------------------------------------------
# Extra margin added around the GPS point cloud, in degrees.
MAP_MARGIN_DEG: float = 0.04

# --- Measurement points -------------------------------------------------------
SHOW_POINTS:  bool  = True
POINTS_ALPHA: float = 0.3
POINTS_SIZE:  float = 4.0
POINTS_COLOR: str   = "#D4A017"

# --- Grid cell style ---------------------------------------------------------
PIXEL_FILL_COLOR: str   = "#1565C0"
PIXEL_FILL_ALPHA: float = 0.10
PIXEL_EDGE_COLOR: str   = "#1565C0"
PIXEL_EDGE_WIDTH: float = 1.4

# --- FCUL reference station ---------------------------------------------------
FCUL_LAT:   float = 38.7565
FCUL_LON:   float = -9.1564
SHOW_FCUL:  bool  = True

# --- Figure -------------------------------------------------------------------
FIGURE_DPI:     int   = 300
FIGURE_SIZE_IN: tuple = (8.0, 7.0)
SHOW_FIGURE:    bool  = False


# ==============================================================================
# COORDINATE HELPERS
# ==============================================================================

_to_merc  = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def to_merc(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS-84 lon/lat to Web Mercator (EPSG:3857) in metres."""
    return _to_merc.transform(
        np.asarray(lon, dtype=float),
        np.asarray(lat, dtype=float),
    )


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_gps_points(filepath: Path) -> pd.DataFrame:
    """
    Load GPS coordinates from a decoded tuk-tuk CSV.

    Parameters
    ----------
    filepath : Path
        Path to the decoded CSV (decode_tuktuk.py output or downstream variant).
        Must contain 'lat' and 'lon' columns in decimal degrees (WGS-84).

    Returns
    -------
    pd.DataFrame
        Columns: lat, lon — rows with missing or implausible coordinates dropped.
    """
    df = pd.read_csv(filepath)

    missing = [c for c in ("lat", "lon") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}\n"
            f"Expected the decoded format produced by decode_tuktuk.py."
        )

    pts = df[["lat", "lon"]].dropna()
    # Sanity filter: keep only coordinates plausible for Lisbon
    pts = pts[pts["lat"].between(38.0, 40.0) & pts["lon"].between(-10.0, -8.0)]
    print(f"  {len(pts):,} valid GPS points loaded")
    return pts.reset_index(drop=True)


# ==============================================================================
# CAMS GRID
# ==============================================================================

def compute_cams_pixels(pts: pd.DataFrame, step: float) -> pd.DataFrame:
    """
    Identify all distinct grid nodes touched by the GPS trajectory.

    Each GPS point is snapped to the nearest CAMS grid node by rounding to
    the nearest multiple of `step`.  The distinct (cams_lat, cams_lon) pairs
    represent the nodes for which a CAMS time series will be downloaded.

    Parameters
    ----------
    pts : pd.DataFrame
        GPS points with 'lat' and 'lon' columns.
    step : float
        CAMS grid spacing in decimal degrees.

    Returns
    -------
    pd.DataFrame
        One row per distinct node, columns: cams_lat, cams_lon, n_points.
    """
    snapped = pts.copy()
    snapped["cams_lat"] = ((snapped["lat"] / step).round() * step).round(4)
    snapped["cams_lon"] = ((snapped["lon"] / step).round() * step).round(4)

    pixels = (
        snapped.groupby(["cams_lat", "cams_lon"])
        .size()
        .reset_index(name="n_points")
        .sort_values(["cams_lat", "cams_lon"])
        .reset_index(drop=True)
    )
    return pixels


# ==============================================================================
# FIGURE
# ==============================================================================

def draw_scale_bar(ax: plt.Axes,
                   x0: float, y0: float,
                   length_km: float = 2.0) -> None:
    """
    Draw a simple scale bar on a Web Mercator axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x0, y0 : float
        Lower-left anchor in Web Mercator metres.
    length_km : float
        Scale bar length in kilometres.
    """
    length_m = length_km * 1000.0
    tick_h = 80
    ax.plot([x0, x0 + length_m], [y0, y0],
            color="black", linewidth=2.5, solid_capstyle="butt", zorder=10)
    for xv in (x0, x0 + length_m):
        ax.plot([xv, xv], [y0 - tick_h, y0 + tick_h],
                color="black", linewidth=1.5, zorder=10)
    ax.text(x0 + length_m / 2, y0 + 200,
            f"{length_km:.0f} km",
            ha="center", va="bottom", fontsize=7.5, zorder=10)


def make_figure(pts: pd.DataFrame,
                pixels: pd.DataFrame,
                step: float) -> plt.Figure:
    """
    Build the CAMS grid map figure.

    Parameters
    ----------
    pts : pd.DataFrame
        GPS measurement points (lat, lon).
    pixels : pd.DataFrame
        Grid node table from compute_cams_pixels().
    step : float
        CAMS grid spacing in decimal degrees.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Map extent in Web Mercator ─────────────────────────────────────────────
    lat_min = pts["lat"].min() - MAP_MARGIN_DEG
    lat_max = pts["lat"].max() + MAP_MARGIN_DEG
    lon_min = pts["lon"].min() - MAP_MARGIN_DEG
    lon_max = pts["lon"].max() + MAP_MARGIN_DEG

    x_min, y_min = to_merc(np.array([lon_min]), np.array([lat_min]))
    x_max, y_max = to_merc(np.array([lon_max]), np.array([lat_max]))
    x_min, x_max = float(x_min[0]), float(x_max[0])
    y_min, y_max = float(y_min[0]), float(y_max[0])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ── Basemap ────────────────────────────────────────────────────────────────
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                            crs="EPSG:3857", zoom="auto", attribution_size=6)
        except Exception as exc:
            print(f"  [WARNING] Basemap unavailable: {exc}")
            ax.set_facecolor("#f5f5f0")
    else:
        ax.set_facecolor("#f5f5f0")

    # ── GPS measurement points ─────────────────────────────────────────────────
    if SHOW_POINTS:
        x_pts, y_pts = to_merc(pts["lon"].values, pts["lat"].values)
        ax.scatter(x_pts, y_pts,
                   s=POINTS_SIZE, c=POINTS_COLOR, alpha=POINTS_ALPHA,
                   linewidths=0, zorder=3, rasterized=True)

    # ── Grid cell rectangles ──────────────────────────────────────────────────
    lat_km = step * 111.0
    lon_km = step * 111.0 * math.cos(math.radians(pts["lat"].mean()))

    for _, px in pixels.iterrows():
        clat, clon = px["cams_lat"], px["cams_lon"]
        corners_lon = [clon - step/2, clon + step/2,
                       clon + step/2, clon - step/2, clon - step/2]
        corners_lat = [clat - step/2, clat - step/2,
                       clat + step/2, clat + step/2, clat - step/2]
        cx_m, cy_m = to_merc(np.array(corners_lon), np.array(corners_lat))

        ax.fill(cx_m, cy_m,
                color=PIXEL_FILL_COLOR, alpha=PIXEL_FILL_ALPHA, zorder=4)
        ax.plot(cx_m, cy_m,
                color=PIXEL_EDGE_COLOR, linewidth=PIXEL_EDGE_WIDTH,
                zorder=5, solid_capstyle="round")

        # Centre label
        x_c, y_c = to_merc(np.array([clon]), np.array([clat]))
        ax.text(float(x_c[0]), float(y_c[0]) + 120,
                f"{clat:.2f}°N  {clon:.2f}°E",
                ha="center", va="bottom", fontsize=6.0,
                color=PIXEL_EDGE_COLOR, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", ec="none", alpha=0.75))

    # ── Pixel centre markers ───────────────────────────────────────────────────
    px_x, px_y = to_merc(pixels["cams_lon"].values, pixels["cams_lat"].values)
    ax.scatter(px_x, px_y,
               s=60, c=PIXEL_EDGE_COLOR, marker="+",
               linewidths=1.5, zorder=7)

    # ── FCUL marker ───────────────────────────────────────────────────────────
    if SHOW_FCUL:
        fx, fy = to_merc(np.array([FCUL_LON]), np.array([FCUL_LAT]))
        ax.scatter(fx, fy,
                   s=90, c="#C0392B", marker="s",
                   edgecolors="white", linewidths=0.8, zorder=8)
        ax.text(float(fx[0]) + 80, float(fy[0]) + 200,
                "FCUL", fontsize=7.5, color="#C0392B",
                fontweight="bold", zorder=8)

    # ── Scale bar ──────────────────────────────────────────────────────────────
    draw_scale_bar(ax,
                   x0=x_min + (x_max - x_min) * 0.05,
                   y0=y_min + (y_max - y_min) * 0.04,
                   length_km=2.0)

    # ── Axis tick labels in lat/lon ────────────────────────────────────────────
    x_ticks = np.linspace(x_min, x_max, 5)
    y_ticks = np.linspace(y_min, y_max, 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    lon_labels = _to_wgs84.transform(x_ticks, np.full_like(x_ticks, y_min))[0]
    lat_labels = _to_wgs84.transform(np.full_like(y_ticks, x_min), y_ticks)[1]
    ax.set_xticklabels([f"{v:.3f}°" for v in lon_labels], fontsize=7)
    ax.set_yticklabels([f"{v:.3f}°" for v in lat_labels], fontsize=7)
    ax.set_xlabel("Longitude (WGS-84)", fontsize=8.5)
    ax.set_ylabel("Latitude (WGS-84)", fontsize=8.5)

    # ── Legend ─────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=PIXEL_FILL_COLOR, alpha=0.35,
                       edgecolor=PIXEL_EDGE_COLOR, linewidth=1.2,
                       label=f"Grid cell ({step}° ≈ {lat_km:.1f}×{lon_km:.1f} km)"),
        mlines.Line2D([], [], marker="+", color=PIXEL_EDGE_COLOR,
                      markersize=8, linewidth=0,
                      label=f"Grid nodes (n={len(pixels)})"),
    ]
    if SHOW_POINTS:
        handles.append(
            mlines.Line2D([], [], marker="o", color=POINTS_COLOR,
                          alpha=min(POINTS_ALPHA * 5, 1.0),
                          markersize=4, linewidth=0,
                          label=f"Measurements (n={len(pts):,})")
        )
    if SHOW_FCUL:
        handles.append(
            mlines.Line2D([], [], marker="s", color="#C0392B",
                          markersize=6, linewidth=0, label="FCUL pyranometer")
        )
    ax.legend(handles=handles, loc="upper right",
              fontsize=7.5, framealpha=0.9, edgecolor="#cccccc")

    # ── Title and annotation ───────────────────────────────────────────────────
    ax.set_title("CAMS Spatial Grid — Tuk-tuk Campaign (Lisbon)",
                 fontsize=11, fontweight="bold", pad=10)
    fig.text(
        0.5, -0.01,
        f"Discretisation grid: {step}°  |  "
        f"Grid nodes: {len(pixels)}  |  "
        f"GPS points: {len(pts):,}  |  "
        f"Basemap © OpenStreetMap contributors / CartoDB",
        ha="center", fontsize=6.5, color="#888888", style="italic",
    )

    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the GPS points, compute the grid cells and save the map."""
    print("=" * 68)
    print("  CAMS GRID MAP — TUK-TUK CAMPAIGN")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_DIR / OUTPUT_FILE}")
    print("=" * 68)

    print("\n[1/3] Loading GPS points ...")
    pts = load_gps_points(INPUT_FILE)
    if pts.empty:
        print("  ERROR: no valid GPS points found.")
        return

    print("\n[2/3] Computing discretisation grid nodes ...")
    pixels = compute_cams_pixels(pts, GRID_DEG)
    lat_km = GRID_DEG * 111.0
    lon_km = GRID_DEG * 111.0 * math.cos(math.radians(pts["lat"].mean()))
    print(f"  {len(pixels)} distinct node(s) — ~{lat_km:.1f}×{lon_km:.1f} km each:")
    for _, px in pixels.iterrows():
        print(f"    ({px['cams_lat']:.4f}°N, {px['cams_lon']:.4f}°E)"
              f"  →  {px['n_points']:,} measurements")

    print("\n[3/3] Drawing figure ...")
    fig = make_figure(pts, pixels, GRID_DEG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Figure saved → {out_path}")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    print("=" * 68 + "\n")


# ==============================================================================
if __name__ == "__main__":
    main()