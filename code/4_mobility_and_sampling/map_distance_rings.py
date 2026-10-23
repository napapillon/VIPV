"""
================================================================================
Lisbon Measurement Map with Distance Rings around FCUL Station
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Maps all GPS measurement locations of the Solar Jinie sensors on the Lisbon
region, each point coloured by its geodesic distance from the FCUL ground
reference station (rooftop pyranometer). Distance rings at 3, 9, 18 and
32 km are drawn around FCUL, matching the spatial-uncertainty analysis of
the 3_irradiance scripts.

Optional layers: GADM administrative boundaries (USE_SHAPEFILES, levels in
SHAPEFILE_LEVELS) and a web basemap (USE_BASEMAP, needs internet access).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon'
    shapefiles (optional) : BOUNDARIES_DIR / gadm41_PRT_<level>.shp (GADM 4.1)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             mdr_map_distance_rings_<CAMPAIGN>_<variant_tag>.png

Requirements
------------
    pip install pandas numpy matplotlib geopandas contextily shapely pyproj

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.1
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BOUNDARIES_DIR, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import contextily as ctx
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR = STANDARD_DIR                  # CAMPAIGN = "standard"
TUKTUK_DATA_DIR = TUKTUK_DIR             # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output directory ----------------------------------------------------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "4_mobility_and_sampling" / "mdr_outputs"

# --- FCUL reference station coordinates (WGS-84 decimal degrees) --------------
# Pyranometer on the rooftop of building C1, Faculty of Sciences,
# University of Lisbon, campus Cidade Universitária.
FCUL_LAT = 38.7565    # Google Maps: 38.756594832619776
FCUL_LON = -9.1564    # Google Maps: -9.156483276747663

# --- Distance ring radii (km) -------------------------------------------------
# Chosen to match the spatial uncertainty analysis:
# 3 km  → CAMS pixel size (~3 × 3 km)
# 9 km  → intermediate zone
# 18 km → threshold beyond which ground-based reference should become unreliable
# 32 km → outer limit of the dataset
RING_RADII_KM: list[int] = [3, 9, 18, 32]

# --- Earth radius -------------------------------------------------------------
R_EARTH_KM: float = 6371.0   # Mean Earth radius (km) – IAU 2012

# --- Colour map for distance bands --------------------------------------------
# One colour per band: [0–3 km] [3–9 km] [9–18 km] [18–32 km] [> 32 km]
DISTANCE_COLORS: list[str] = [
    "#2166ac",   # band 0 : d < 3 km   (blue)
    "#4dac26",   # band 1 : 3–9 km     (green)
    "#f4a582",   # band 2 : 9–18 km    (light orange)
    "#d6604d",   # band 3 : 18–32 km   (red-orange)
    "#7b2d8b",   # band 4 : d ≥ 32 km  (purple)
]

# --- Shapefile options --------------------------------------------------------
USE_SHAPEFILES: bool = True
SHAPEFILE_DIR: Path  = BOUNDARIES_DIR
# Administrative levels: 1 = districts, 2 = municipalities
SHAPEFILE_LEVELS: list[int] = [1, 2]

# --- Map extent (WGS-84 degrees) – centred on Greater Lisbon ------------------
# Extended to fully contain the 32 km ring around FCUL (≈ 0.29° lat / 0.37° lon)
MAP_WEST:  float = -9.70
MAP_EAST:  float = -8.65
MAP_SOUTH: float = 38.40
MAP_NORTH: float = 39.10

# --- Basemap tile provider ----------------------------------------------------
# ctx.providers.CartoDB.Positron      → light, publication-friendly (default)
# ctx.providers.OpenStreetMap.Mapnik  → full colour street map
BASEMAP_PROVIDER = ctx.providers.CartoDB.Positron

# False: GADM shapefiles only, on a plain background (no internet needed)
USE_BASEMAP: bool = True

# Background colour used when USE_BASEMAP is False
BACKGROUND_COLOR: str = "#f5f5f0"   # light off-white, easy on the eye

# --- Figure resolution --------------------------------------------------------
OUTPUT_DPI: int = 300

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# Input and output paths follow from CAMPAIGN and the input suffix.

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor names to their CSV input paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv

    Parameters
    ----------
    campaign : str
        "standard" or "tuktuk".
    data_dir : Path
        Directory containing the standard-campaign CSV files.
    tuktuk_data_dir : Path
        Directory containing the tuk-tuk per-sensor CSV files.
    suffix : str
        Active variant tag (FILE_SUFFIX or TUKTUK_FILE_SUFFIX).

    Returns
    -------
    dict[str, Path]
        {'j30': Path(...), 'j31': Path(...), ...}
    """
    sensors = ['j30', 'j31', 'j32', 'j33']
    if campaign == "standard":
        return {
            s: data_dir / f"{s}_satellite_landuse_v2_{suffix}.csv"
            for s in sensors
        }
    elif campaign == "tuktuk":
        return {
            s: tuktuk_data_dir / f"{s}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
            for s in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """
    Build the output directory path, embedding the campaign and active
    variant tag so results from different campaigns/variants never
    overwrite each other.

    Parameters
    ----------
    base_dir : Path
        Base output directory set in the configuration section.
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().

    Returns
    -------
    Path
        Full output directory path (not yet created).
    """
    return base_dir / campaign / variant_tag


def _build_output_path(out_dir: Path, base_name: str, campaign: str,
                       variant_tag: str, extension: str) -> Path:
    """
    Build an output file path that embeds the active campaign and variant tag.

    Parameters
    ----------
    out_dir : Path
        Output directory (already resolved).
    base_name : str
        Descriptive part of the filename (e.g. 'mdr_map_distance_rings').
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().
    extension : str
        File extension without leading dot (e.g. 'png').

    Returns
    -------
    Path
        Complete file path, e.g.
        out_dir / 'mdr_map_distance_rings_standard_sd_cleaned.png'.
    """
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolved at startup from the configuration above
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR:   Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_FILE:  Path = _build_output_path(
    OUTPUT_DIR, 'mdr_map_distance_rings', CAMPAIGN, _VARIANT_TAG, 'png'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def haversine_km(lat1: float, lon1: float,
                 lat2: np.ndarray, lon2: np.ndarray,
                 R: float = R_EARTH_KM) -> np.ndarray:
    """
    Compute the great-circle distance (km) between a fixed reference point
    (lat1, lon1) and an array of points (lat2, lon2) using the Haversine
    formula (equation 3.1 of the thesis).

    Parameters
    ----------
    lat1, lon1 : float
        Reference point in decimal degrees (WGS-84).
    lat2, lon2 : array-like
        Target points in decimal degrees (WGS-84).
    R : float
        Mean Earth radius in km (default: 6371.0).

    Returns
    -------
    np.ndarray
        Distances in kilometres.
    """
    phi1    = np.radians(lat1)
    phi2    = np.radians(np.asarray(lat2, dtype=float))
    dphi    = np.radians(np.asarray(lat2, dtype=float) - lat1)
    dlambda = np.radians(np.asarray(lon2, dtype=float) - lon1)

    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(a))   # kilometres


def classify_distance(distances_km: np.ndarray,
                      radii: list[int]) -> np.ndarray:
    """
    Assign each measurement point to a distance band.

    Band 0 : d <  radii[0]
    Band k : radii[k-1] ≤ d < radii[k]   (k = 1 … n-1)
    Band n : d ≥  radii[-1]

    Parameters
    ----------
    distances_km : np.ndarray
        Distances from FCUL (km).
    radii : list of int
        Sorted ring radii in km.

    Returns
    -------
    np.ndarray of int
        Band index (0-based) for each point.
    """
    bands = np.zeros(len(distances_km), dtype=int)
    for k, r in enumerate(radii):
        bands[distances_km >= r] = k + 1
    return bands


def wgs84_to_webmercator(lon: np.ndarray,
                         lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert WGS-84 (EPSG:4326) coordinates to Web Mercator (EPSG:3857)
    for compatibility with contextily basemap tiles.

    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude in decimal degrees.

    Returns
    -------
    (x, y) : tuple of np.ndarray
        Projected coordinates in metres (EPSG:3857).
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return x, y


def circle_webmercator(center_lat: float, center_lon: float,
                       radius_km: float,
                       n_pts: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the (x, y) vertices of a geodesically correct circle in
    Web Mercator (EPSG:3857).

    The circle is built in a local azimuthal equidistant projection centred
    on the reference point (which preserves distances in all directions) and
    then re-projected to EPSG:3857. This avoids the ellipse distortion that
    would result from drawing the circle directly in geographic coordinates.

    Parameters
    ----------
    center_lat, center_lon : float
        Centre of the circle in WGS-84 decimal degrees.
    radius_km : float
        Radius of the circle in kilometres.
    n_pts : int
        Number of vertices (higher = smoother circle).

    Returns
    -------
    (x, y) : tuple of np.ndarray
        Circle vertices in Web Mercator metres.
    """
    # Local azimuthal equidistant CRS centred on FCUL
    aeqd_crs = (
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m"
    )

    # Circle vertices in local metric space (metres)
    angles  = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    r_m     = radius_km * 1000.0
    x_local = r_m * np.cos(angles)
    y_local = r_m * np.sin(angles)

    # aeqd → WGS-84 → Web Mercator
    to_wgs84 = Transformer.from_crs(aeqd_crs, "EPSG:4326", always_xy=True)
    lon_circle, lat_circle = to_wgs84.transform(x_local, y_local)
    x_merc, y_merc = wgs84_to_webmercator(lon_circle, lat_circle)
    return x_merc, y_merc


def load_sensor_data(filepath: Path,
                     sensor_label: str) -> pd.DataFrame | None:
    """
    Load one CSV sensor file and return valid latitude/longitude pairs.

    Filtering steps applied in order:
        1. lat/lon columns are coerced to float (handles stray text values).
        2. NaN coordinates are dropped.
        3. Coordinates outside valid WGS-84 ranges are dropped.
        4. Zero coordinates (GPS not fixed) are dropped.

    Column name detection
    ---------------------
    The function first looks for the exact column names produced by the
    data pipeline, e.g. 'J31 lat' and 'J31 lon' (uppercase sensor ID).
    If not found, it falls back to a case-insensitive search, excluding
    'land_use_clcv' and 'land_use_str_clcv' which also contain the substring 'lan'.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.
    sensor_label : str
        Sensor key, e.g. 'j31'. Used to build expected column names.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['lat', 'lon'] for valid rows,
        or None if the file is missing or no valid rows remain.
    """
    if not filepath.is_file():
        print(f"  [WARNING] File not found: {filepath}  – skipped.")
        return None

    df = pd.read_csv(filepath, low_memory=False)

    # ── Step 1 : locate lat / lon columns ─────────────────────────────────────
    sensor_upper = sensor_label.upper()
    lat_col = f"{sensor_upper} lat"
    lon_col = f"{sensor_upper} lon"

    # Fallback: case-insensitive search, explicitly excluding land_use columns
    if lat_col not in df.columns:
        candidates = [
            c for c in df.columns
            if "lat" in c.lower() and "land" not in c.lower()
        ]
        lat_col = candidates[0] if candidates else None

    if lon_col not in df.columns:
        candidates = [
            c for c in df.columns
            if "lon" in c.lower() and "land" not in c.lower()
        ]
        lon_col = candidates[0] if candidates else None

    if lat_col is None or lon_col is None:
        print(f"  [WARNING] Could not find lat/lon columns in "
              f"{filepath.name} – skipped.")
        return None

    # ── Steps 2-4 : clean coordinates ─────────────────────────────────────────
    subset = df[[lat_col, lon_col]].copy()
    subset.columns = ["lat", "lon"]

    # Coerce to float: converts any stray text / comma-decimal values to NaN
    subset["lat"] = pd.to_numeric(subset["lat"], errors="coerce")
    subset["lon"] = pd.to_numeric(subset["lon"], errors="coerce")

    subset = subset.dropna()
    subset = subset[
        subset["lat"].between(-90, 90) & subset["lon"].between(-180, 180)
    ]
    subset = subset[(subset["lat"] != 0.0) & (subset["lon"] != 0.0)]

    if subset.empty:
        print(f"  [WARNING] No valid GPS points in {filepath.name} – skipped.")
        return None

    print(f"  {sensor_label}: {len(subset):,} valid GPS points "
          f"loaded from '{filepath.name}'")
    return subset.reset_index(drop=True)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the sensor files and draw the distance-ring map."""
    print("=" * 70)
    print("  Lisbon Measurement Map with Distance Rings")
    print(f"  CAMPAIGN     : {CAMPAIGN}")
    print(f"  Data variant : {_VARIANT_TAG}")
    print("=" * 70)
    print(f"\n  Input files  : {DATA_DIR}")
    print(f"  Output file  : {OUTPUT_FILE}")

    # ── 1. Load all sensor CSV files ──────────────────────────────────────────
    print("\n[1/5] Loading sensor CSV files …")
    frames: list[pd.DataFrame] = []
    for label, path in SENSOR_FILES.items():
        df = load_sensor_data(path, label)
        if df is not None:
            df["sensor"] = label
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
            "FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    all_data = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(all_data):,} measurement points across all sensors.")

    # ── 2. Compute Haversine distances from FCUL ──────────────────────────────
    print("\n[2/5] Computing Haversine distances from FCUL …")
    all_data["dist_km"] = haversine_km(
        FCUL_LAT, FCUL_LON,
        all_data["lat"].values,
        all_data["lon"].values,
    )
    all_data["band"] = classify_distance(
        all_data["dist_km"].values, RING_RADII_KM
    )

    print(f"  Distance range: "
          f"{all_data['dist_km'].min():.2f} – "
          f"{all_data['dist_km'].max():.2f} km")

    band_labels_str = (
        [f"d < {RING_RADII_KM[0]} km"]
        + [f"{RING_RADII_KM[i]}–{RING_RADII_KM[i+1]} km"
           for i in range(len(RING_RADII_KM) - 1)]
        + [f"d ≥ {RING_RADII_KM[-1]} km"]
    )
    for i, lbl in enumerate(band_labels_str):
        n = (all_data["band"] == i).sum()
        print(f"    Band {i} [{lbl}]: {n:,} points")

    # ── 3. Project coordinates to Web Mercator (EPSG:3857) ────────────────────
    print("\n[3/5] Projecting coordinates to Web Mercator (EPSG:3857) …")
    x_pts, y_pts = wgs84_to_webmercator(
        all_data["lon"].values,
        all_data["lat"].values,
    )
    all_data["x_merc"] = x_pts
    all_data["y_merc"] = y_pts

    # FCUL reference point
    fcul_x, fcul_y = wgs84_to_webmercator(
        np.array([FCUL_LON]), np.array([FCUL_LAT])
    )
    fcul_x, fcul_y = float(fcul_x[0]), float(fcul_y[0])

    # Map extent
    x_min, y_min = wgs84_to_webmercator(
        np.array([MAP_WEST]), np.array([MAP_SOUTH])
    )
    x_max, y_max = wgs84_to_webmercator(
        np.array([MAP_EAST]), np.array([MAP_NORTH])
    )

    # ── 4. Load shapefiles (optional) ─────────────────────────────────────────
    shp_layers: dict = {}
    if USE_SHAPEFILES:
        print("\n[4/5] Loading GADM shapefiles …")
        try:
            import geopandas as gpd
            for level in SHAPEFILE_LEVELS:
                shp_path = SHAPEFILE_DIR / f"gadm41_PRT_{level}.shp"
                if shp_path.is_file():
                    gdf = gpd.read_file(shp_path).to_crs("EPSG:3857")
                    shp_layers[level] = gdf
                    print(f"  Level {level}: loaded ({len(gdf)} features)")
                else:
                    print(f"  [WARNING] Not found: {shp_path}")
        except ImportError:
            print("  [WARNING] geopandas not installed – shapefiles skipped.")
    else:
        print("\n[4/5] Shapefiles disabled (USE_SHAPEFILES = False).")

    # ── 5. Draw the map ───────────────────────────────────────────────────────
    print("\n[5/5] Drawing figure …")

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- Basemap --------------------------------------------------------------
    ax.set_xlim(x_min[0], x_max[0])
    ax.set_ylim(y_min[0], y_max[0])
    if USE_BASEMAP:
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
            ax.set_facecolor(BACKGROUND_COLOR)
    else:
        # Plain background – shapefile outlines provide all geographic context
        ax.set_facecolor(BACKGROUND_COLOR)

    # --- Shapefile outlines ---------------------------------------------------
    if shp_layers:
        for level, gdf in shp_layers.items():
            lw = 0.6 if level == 2 else 1.2
            gdf.boundary.plot(ax=ax, color="#333333", linewidth=lw, zorder=2)

    # --- Scatter plot : outermost bands first so closer points plot on top ----
    # s     : marker size in points² – increased for visibility in sparse areas
    # alpha : transparency – kept below 0.6 to limit saturation in dense areas
    for band_idx in range(len(RING_RADII_KM), -1, -1):
        mask = all_data["band"] == band_idx
        ax.scatter(
            all_data.loc[mask, "x_merc"],
            all_data.loc[mask, "y_merc"],
            c=DISTANCE_COLORS[band_idx],
            s=6,
            alpha=0.50,
            linewidths=0,
            zorder=3,
            rasterized=True,   # reduces file size for dense point clouds
        )

    # --- Distance rings -------------------------------------------------------
    ring_styles = [
        {"lw": 2.2, "ls": "-"},
        {"lw": 2.0, "ls": "--"},
        {"lw": 2.0, "ls": "-."},
        {"lw": 1.8, "ls": ":"},
    ]
    # Each ring is drawn with the colour of the band it ENCLOSES:
    #   ring at  3 km → DISTANCE_COLORS[0]  (blue,   d < 3 km)
    #   ring at  9 km → DISTANCE_COLORS[1]  (green,  3–9 km)
    #   ring at 18 km → DISTANCE_COLORS[2]  (orange, 9–18 km)
    #   ring at 32 km → DISTANCE_COLORS[3]  (red,    18–32 km)
    for i, (r_km, style) in enumerate(zip(RING_RADII_KM, ring_styles)):
        cx, cy = circle_webmercator(FCUL_LAT, FCUL_LON, r_km)
        ax.plot(
            cx, cy,
            color=DISTANCE_COLORS[i],
            linewidth=style["lw"],
            linestyle=style["ls"],
            zorder=4,
        )
        # Label at the top of each ring, same colour as the ring
        ax.text(
            fcul_x,
            fcul_y + r_km * 1000 * 1.012,
            f"{r_km} km",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=DISTANCE_COLORS[i],
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    # --- FCUL marker ----------------------------------------------------------
    ax.plot(
        fcul_x, fcul_y,
        marker="*", markersize=16,
        color="gold", markeredgecolor="#333333", markeredgewidth=0.8,
        zorder=7, linestyle="None",
    )

    # --- Legend ---------------------------------------------------------------
    legend_band_labels = (
        [f"d < {RING_RADII_KM[0]} km"]
        + [f"{RING_RADII_KM[i]} ≤ d < {RING_RADII_KM[i+1]} km"
           for i in range(len(RING_RADII_KM) - 1)]
        + [f"d ≥ {RING_RADII_KM[-1]} km"]
    )
    legend_patches = [
        mpatches.Patch(color=DISTANCE_COLORS[i], label=legend_band_labels[i])
        for i in range(len(legend_band_labels))
    ]
    legend_patches.append(
        mlines.Line2D(
            [], [],
            marker="*", color="w",
            markerfacecolor="gold", markeredgecolor="#333333",
            markersize=12, label="FCUL – reference station",
        )
    )
    ax.legend(
        handles=legend_patches,
        title="Distance from FCUL",
        title_fontsize=9,
        fontsize=8.5,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.12),    # shift upward to clear scale bar and OSM credit
        bbox_transform=ax.transAxes,
        framealpha=0.9,
        edgecolor="#aaaaaa",
    )

    # --- Scale bar (Web Mercator metres) --------------------------------------
    # At lat ≈ 38.75°, 10 km in EPSG:3857 ≈ 12 600 m (meridional scale factor)
    sb_length_m = 10_000
    sb_x0 = float(x_min[0]) + 0.04 * (float(x_max[0]) - float(x_min[0]))
    sb_y0 = float(y_min[0]) + 0.04 * (float(y_max[0]) - float(y_min[0]))
    ax.plot([sb_x0, sb_x0 + sb_length_m], [sb_y0, sb_y0],
            color="black", lw=3, zorder=8)
    ax.text(sb_x0 + sb_length_m / 2, sb_y0 + 300,
            "10 km", ha="center", fontsize=8, zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # --- North arrow ----------------------------------------------------------
    arrow_x = float(x_max[0]) - 0.04 * (float(x_max[0]) - float(x_min[0]))
    arrow_y = float(y_max[0]) - 0.08 * (float(y_max[0]) - float(y_min[0]))
    arrow_len = 0.025 * (float(y_max[0]) - float(y_min[0]))
    ax.annotate(
        "", xy=(arrow_x, arrow_y + arrow_len),
        xytext=(arrow_x, arrow_y),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        zorder=9,
    )
    ax.text(arrow_x, arrow_y + arrow_len + 300,
            "N", ha="center", va="bottom",
            fontsize=10, fontweight="bold", zorder=9)

    # --- Title and layout -----------------------------------------------------
    ax.set_axis_off()
    ax.set_title(
        "Measurement locations within 3, 9, 18 and 32 km\n"
        f"of the FCUL solar radiation reference station  [{CAMPAIGN} / {_VARIANT_TAG}]",
        fontsize=12, pad=12,
    )
    fig.tight_layout()

    # ── Save ------------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches="tight")
    print(f"\n  Figure saved → {OUTPUT_FILE}  ({OUTPUT_DPI} dpi)")
    plt.show()

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


# ==============================================================================
if __name__ == "__main__":
    main()