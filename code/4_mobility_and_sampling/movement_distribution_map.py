"""
================================================================================
Movement Distribution Map — Moving vs. Parked GPS Point Visualisation
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Plots GPS measurement locations recorded by the Solar Jinie sensors
(J30, J31, J32, J33) on a map of continental Portugal, with a zoomed panel
showing the Municipality of Lisbon, coloured by vehicle movement status
(Moving or Parked).

Each GPS transition between two consecutive 1-minute measurements is
classified using the Haversine great-circle distance:

  - Distance < THRESHOLD_M metres  →  Parked
  - Distance ≥ THRESHOLD_M metres  →  Moving

The status is assigned to the *departure* point of each transition (i.e. row i
receives the label computed from the displacement toward row i+1).  The last
row of each sensor file is always dropped because no forward transition exists.
Transitions with a time gap above MAX_GAP_MINUTES are not classified (same
rule as vehicle_movement_analysis.py).

──────────────────────────────────────────────────────────────────────────────
DISPLAY_MODE controls which movement-status figure(s) are produced:

  "all"    — one figure with all classified points (golden yellow)
  "moving" — one figure with Moving points only (blue)
  "parked" — one figure with Parked points only (orange)
  "both"   — two figures: Moving (blue) + Parked (orange)

──────────────────────────────────────────────────────────────────────────────
CLC LAND-USE FILTER — two independent, stackable options:

  Option 1 — FILTER_URBAN_RURAL
      'Urban', 'Rural', or None (default).

  Option 2 — FILTER_LAND_USE
      List of specific CLC land_use values (grid_codes, 1–44), or [] (default).
      Takes precedence over FILTER_URBAN_RURAL when both are set.

──────────────────────────────────────────────────────────────────────────────
HEXBIN DENSITY MAP (representativity diagnostic)

  Setting SHOW_HEXBIN = True (default) produces an additional figure for each
  movement-status subset: a hexagonal bin density map overlaid on an OSM
  basemap, zoomed to the Lisbon area.

  Purpose: detect spatial concentration biases.  When a vehicle is parked at
  one location for many minutes, the scatter map looks plausible (many dots
  in the same colour), but the hexbin immediately reveals a dominant hot spot
  through its colour scale.  If the hexbin is nearly uniform, the subset is
  spatially representative.  If one or a few cells dominate, the subset is
  biased and statistical conclusions drawn from it should be treated with
  caution.

  Each hexagonal cell aggregates all points that fall within it and is
  coloured by count on a logarithmic scale, which simultaneously shows both
  the dominant cluster and secondary concentrations.  Cells below
  HEXBIN_MIN_COUNT are left transparent (basemap shows through).

  The viewport is auto-detected from the point cloud of all loaded sensors
  (before the movement filter) with a fractional margin, so Moving and Parked
  figures share the same spatial extent and are directly comparable.

  An internet connection is required to fetch basemap tiles; if unavailable,
  the basemap is silently skipped and a plain white background is used.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', 'land_use_clcv'
    shapefiles (Fig A) : SHAPEFILE_DIR / gadm41_PRT_1.shp (districts, main
             panel) and gadm41_PRT_2.shp (municipalities, Lisbon panel and
             hexbin boundary), GADM 4.1
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             Fig A — Portugal + Lisbon map:
               movement_distribution_map_<CAMPAIGN>_<variant_tag>_<mode>[_<filter_tag>].png
             Fig B — hexbin density map (Lisbon area, OSM basemap):
               movement_hexbin_<CAMPAIGN>_<variant_tag>_<mode>[_<filter_tag>].png
             <mode> = all | moving | parked

Requirements
------------
    pip install pandas numpy matplotlib geopandas shapely pyproj contextily

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 4.1
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
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import box as shapely_box

# contextily is optional: if missing or if basemap tiles are unreachable the
# hexbin figure is still produced on a plain white background.
try:
    import contextily as ctx
    _CTX_AVAILABLE = True
except ImportError:
    _CTX_AVAILABLE = False

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from clc_mapping import (
    CLC_MAPPING,
    CLC_LVL3_LABEL,
    CLC_URBAN_RURAL,
    LAND_USE_MIN,
    LAND_USE_MAX,
)

warnings.filterwarnings('ignore')

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
FILE_SUFFIX: str = 'sd_cleaned'          # standard campaign
TUKTUK_FILE_SUFFIX: str = 'sd_cleaned'   # tuk-tuk campaign

# --- Display mode -------------------------------------------------------------
# Controls which movement-status figure(s) are produced.
#   "all"    — all classified points (golden yellow)
#   "moving" — Moving points only (blue)
#   "parked" — Parked points only (orange)
#   "both"   — Moving (blue) + Parked (orange) — two separate figures
DISPLAY_MODE: str = 'both'   # "all" | "moving" | "parked" | "both"

# --- CLC land-use filter — Option 1: Urban / Rural group ----------------------
# Restrict ALL figures to points belonging to the chosen CLC group.
# Allowed values: 'Urban', 'Rural', or None (no filter).
# When both FILTER_URBAN_RURAL and FILTER_LAND_USE are set, FILTER_LAND_USE
# takes precedence and FILTER_URBAN_RURAL is ignored (with a printed warning).
FILTER_URBAN_RURAL: str | None = None   # 'Urban' | 'Rural' | None

# --- CLC land-use filter — Option 2: specific land_use values -----------------
# Restrict ALL figures to one or more specific CLC classes (= grid_code, 1–44).
# Use [] to disable.  Takes precedence over FILTER_URBAN_RURAL.
# See CLC_MAPPING in clc_mapping.py for the full class list.
FILTER_LAND_USE: list = []   # e.g. [4] or [1, 2] or []

# --- Hexbin density map (representativity diagnostic) -------------------------
# Produce an additional hexbin figure for each movement-status subset?
SHOW_HEXBIN: bool = True

# Target physical size of each hexagonal cell in metres.
# Smaller = finer grid (more detail, more cells); larger = coarser grid.
# Recommended range: 100 m (dense urban analysis) to 500 m (broad overview).
HEXBIN_CELL_SIZE_M: int = 200

# Minimum number of points for a hex cell to be rendered.
# Cells below this count are left transparent; basemap shows through.
# Raise to reduce visual clutter from isolated low-count cells.
HEXBIN_MIN_COUNT: int = 3

# Colormap for the hexbin overlay ('YlOrRd' reads well on a light basemap).
HEXBIN_COLORMAP: str = 'YlOrRd'

# Transparency of the hexbin overlay (0 = fully transparent, 1 = opaque).
HEXBIN_ALPHA: float = 0.75

# Fractional margin added around the auto-detected viewport (0.05 = 5%).
# Ignored when HEXBIN_BBOX is set.
HEXBIN_VIEWPORT_MARGIN: float = 0.05

# Manual bounding box for the hexbin figure (WGS-84 decimal degrees).
# Set to None to use the auto-detected extent from the point cloud + margin.
# Set to (lon_min, lat_min, lon_max, lat_max) to fix the viewport yourself.
# Points outside the bbox are excluded from the hexbin figure but are still
# counted in the legend total (n = X shows the full subset size).
HEXBIN_BBOX: tuple | None = (-9.52, 38.64, -9.04, 38.87)

# Basemap tile provider for the hexbin figure (None = CartoDB Positron).
# Requires an internet connection; silently skipped if unreachable.
# Other option: ctx.providers.OpenStreetMap.Mapnik
HEXBIN_BASEMAP_PROVIDER = None   # resolved at runtime (see _get_basemap_provider)

# --- Latitude / longitude column names per sensor -----------------------------
SENSOR_COLS: dict = {
    'J30': {'lat': 'J30 lat', 'lon': 'J30 lon'},
    'J31': {'lat': 'J31 lat', 'lon': 'J31 lon'},
    'J32': {'lat': 'J32 lat', 'lon': 'J32 lon'},
    'J33': {'lat': 'J33 lat', 'lon': 'J33 lon'},
}

# Datetime column (same in both campaigns)
DATE_COL: str = 'Human readable date (Europe/London)'

# --- Movement classification (as in vehicle_movement_analysis.py) -------------
THRESHOLD_M: float = 30.0      # parked / moving threshold [m]
MAX_GAP_MINUTES: int = 2       # longer gaps are not classified [min]

# --- Timezone -----------------------------------------------------------------
LOCAL_TZ: str = 'Europe/Lisbon'

# --- Earth radius (Haversine formula) -----------------------------------------
R_EARTH_KM: float = 6371.0   # mean Earth radius — IAU 2012

# --- GADM shapefiles directory ------------------------------------------------
SHAPEFILE_DIR: Path = BOUNDARIES_DIR

# --- Lisboa municipality name as it appears in the GADM NAME_2 column ---------
LISBON_NAME: str = 'Lisboa'

# --- Zoom rectangle padding (degrees) around the Lisboa bounding box ----------
RECT_MARGIN: float = 0.12

# --- Rendering mode for the Lisbon panel (Fig A only) -------------------------
CLIP_LISBON_POINTS: bool = True
ZOOM_BORDERS: str = 'viewport'   # 'viewport' | 'lisbon_only'

# --- Point colours ------------------------------------------------------------
POINT_COLOR_ALL:    str = '#D4A017'   # golden yellow — all points
POINT_COLOR_MOVING: str = '#1565C0'   # project blue  — Moving
POINT_COLOR_PARKED: str = '#E65100'   # project orange — Parked

# --- Fig A point style --------------------------------------------------------
POINT_SIZE_MAIN: float = 3.0
POINT_SIZE_ZOOM: float = 7.0
POINT_ALPHA:     float = 0.55

# --- Output -------------------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "4_mobility_and_sampling" / "mdm_outputs"
OUTPUT_DPI: int = 300

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        file_suffix: str, tuktuk_file_suffix: str) -> dict:
    """
    Build the dict mapping sensor identifiers to their CSV input paths.

    Parameters
    ----------
    campaign : str
        "standard" or "tuktuk".
    data_dir : Path
        Directory containing the standard-campaign CSV files.
    tuktuk_data_dir : Path
        Directory containing the tuk-tuk per-sensor CSV files.
    file_suffix : str
        FILE_SUFFIX value (standard campaign).
    tuktuk_file_suffix : str
        TUKTUK_FILE_SUFFIX value (tuk-tuk campaign).

    Returns
    -------
    dict[str, Path]
    """
    sensors = ['j30', 'j31', 'j32', 'j33']
    if campaign == "standard":
        return {
            sensor: data_dir / f'{sensor}_satellite_landuse_v2_{file_suffix}.csv'
            for sensor in sensors
        }
    elif campaign == "tuktuk":
        sep = f"_{tuktuk_file_suffix}" if tuktuk_file_suffix else ""
        return {
            sensor: tuktuk_data_dir / f'{sensor}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv'
            for sensor in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign (used in
    output directory / filename construction)."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output directory path, embedding the campaign and active
    variant tag (base_dir/campaign/variant_tag/)."""
    return base_dir / campaign / variant_tag


SENSOR_FILES = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, FILE_SUFFIX, TUKTUK_FILE_SUFFIX
)
_VARIANT_TAG = _variant_tag()
OUTPUT_DIR   = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)


# ==============================================================================
# CLC FILTER RESOLUTION
# ==============================================================================

def resolve_clc_filter(
    filter_land_use:    list,
    filter_urban_rural: str | None,
) -> tuple:
    """
    Resolve the active CLC filter from the two configuration parameters.

    FILTER_LAND_USE takes precedence over FILTER_URBAN_RURAL when both are set.

    Parameters
    ----------
    filter_land_use : list[int]
        Explicit list of land_use values (grid_codes, 1–44).  Empty = inactive.
    filter_urban_rural : str or None
        'Urban', 'Rural', or None.

    Returns
    -------
    active_lu_values : list[int] or None
        Land_use values to keep, or None if no filter is active.
    filter_tag : str or None
        Short string for output filenames (e.g. 'urban', 'lu4'), or None.
    filter_desc : str or None
        Human-readable description for figure titles and log messages, or None.
    """
    if filter_land_use:
        invalid = [lu for lu in filter_land_use if lu not in CLC_MAPPING]
        if invalid:
            raise ValueError(
                f'FILTER_LAND_USE contains invalid value(s): {invalid}\n'
                f'Valid range is {LAND_USE_MIN}–{LAND_USE_MAX}.  '
                'See CLC_MAPPING in clc_mapping.py for the full list.'
            )
        if filter_urban_rural is not None:
            print(
                '  [INFO] Both FILTER_LAND_USE and FILTER_URBAN_RURAL are set. '
                'FILTER_LAND_USE takes precedence; FILTER_URBAN_RURAL is ignored.'
            )
        tag = 'lu' + '_'.join(str(lu) for lu in sorted(filter_land_use))
        if len(filter_land_use) == 1:
            desc = CLC_LVL3_LABEL[filter_land_use[0]]
        else:
            desc = ', '.join(CLC_LVL3_LABEL[lu] for lu in sorted(filter_land_use))
        return filter_land_use, tag, desc

    if filter_urban_rural is not None:
        if filter_urban_rural not in ('Urban', 'Rural'):
            raise ValueError(
                f"FILTER_URBAN_RURAL must be 'Urban', 'Rural', or None, "
                f"got '{filter_urban_rural}'."
            )
        values = [lu for lu, g in CLC_URBAN_RURAL.items()
                  if g == filter_urban_rural]
        return values, filter_urban_rural.lower(), f'{filter_urban_rural} areas (CLC)'

    return None, None, None


# ==============================================================================
# COORDINATE HELPERS
# ==============================================================================

def wgs84_to_webmercator(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple:
    """
    Reproject WGS-84 lon/lat arrays to Web Mercator (EPSG:3857).

    Web Mercator is the native CRS of OSM basemap tiles and is required by
    contextily.add_basemap.  All hexbin computations are performed in this CRS
    so that physical distances (metres) are directly meaningful for gridsize.

    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude in decimal degrees (WGS-84).

    Returns
    -------
    x_merc, y_merc : numpy.ndarray
        Easting and northing in EPSG:3857 metres.
    """
    t = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    return t.transform(np.asarray(lon, dtype=float),
                       np.asarray(lat, dtype=float))


def _get_basemap_provider():
    """Return the contextily basemap provider, resolving the default lazily."""
    if not _CTX_AVAILABLE:
        return None
    if HEXBIN_BASEMAP_PROVIDER is not None:
        return HEXBIN_BASEMAP_PROVIDER
    return ctx.providers.CartoDB.Positron


def _try_add_basemap(ax: plt.Axes) -> None:
    """
    Add OSM basemap tiles to an EPSG:3857 axis.

    Fails silently if contextily is not installed or if the tile server is
    unreachable (no internet connection), so the hexbin figure is still
    produced on a plain white background.
    """
    provider = _get_basemap_provider()
    if provider is None:
        return
    try:
        ctx.add_basemap(ax, source=provider, crs='EPSG:3857',
                        zoom='auto', attribution_size=6)
    except Exception as exc:
        print(f'  [WARNING] Could not fetch basemap tiles: {exc}')


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2, R=R_EARTH_KM,
                         timestamps=None, max_gap_minutes=MAX_GAP_MINUTES):
    """
    Compute the great-circle distance in metres between consecutive GPS points.

    Parameters
    ----------
    lat1, lon1 : array-like
        Coordinates of the departure point (decimal degrees, WGS-84).
        Typically df[lat_col].values[:-1].
    lat2, lon2 : array-like
        Coordinates of the arrival point (decimal degrees, WGS-84).
        Typically df[lat_col].values[1:].
    R : float
        Earth radius in kilometres (default: 6371.0 — IAU 2012).
    timestamps : array-like of datetime64, optional
        Full timestamp array of length N (i.e. df['datetime'].values).
        When provided, pairs that cross a segment boundary are zeroed:
          - time gap > max_gap_minutes  (overnight stop, sensor restart)
          - calendar-date change        (explicit cross-day guard)
        When None (default), all distances are returned as computed — safe
        for point-to-fixed-reference uses (distance rings, FCUL proximity).
    max_gap_minutes : float
        Maximum allowed gap in minutes between consecutive rows.
        Defaults to MAX_GAP_MINUTES (defined in the CONFIG section).

    Returns
    -------
    numpy.ndarray of float, shape (N-1,)
        Great-circle distances in metres.
        Boundary pairs are returned as 0.0 when timestamps is provided.
    """
    lat1 = np.asarray(lat1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    phi1    = np.radians(lat1)
    phi2    = np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(np.asarray(lon2, dtype=float)
                         - np.asarray(lon1, dtype=float))
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    distances = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * 1000.0

    if timestamps is not None:
        s         = pd.Series(np.asarray(timestamps))
        gap_s     = s.diff().dt.total_seconds().fillna(0).values
        dates     = s.dt.date.values
        # gap_s[i+1] = elapsed seconds between row i and row i+1;
        # distances[i] covers that same pair -> compare gap_s[1:]
        is_boundary = (
            (gap_s[1:] > max_gap_minutes * 60)
            | (dates[1:] != dates[:-1])
        )
        distances[is_boundary] = 0.0

    return distances


def load_and_classify_sensor(
    sensor_name: str,
    file_path:   Path,
    lat_col:     str,
    lon_col:     str,
) -> pd.DataFrame | None:
    """
    Load a sensor CSV and attach a Moving/Parked label to each GPS row.

    Steps
    -----
    1. Check file existence; skip with a warning if absent.
    2. Read the CSV.
    3. Parse the datetime column and convert to Europe/Lisbon timezone.
    4. Cast lat/lon to float; drop rows with missing or zero GPS values.
    5. Keep the land_use_clcv column when present.
    6. Sort chronologically.
    7. Compute Haversine distances between consecutive rows.
    8. Assign 'Moving' or 'Parked' to each departure point.
       The last row is dropped (no forward neighbour).

    Parameters
    ----------
    sensor_name : str
        Short identifier for log messages (e.g. 'J31').
    file_path : Path
        Path to the cleaned sensor CSV file.
    lat_col, lon_col : str
        Column names for latitude and longitude.

    Returns
    -------
    pd.DataFrame with columns ['lat', 'lon', 'moving_status'] and optionally
    ['land_use_clcv'], or None if the file cannot be loaded.
    """
    print(f"  Loading {sensor_name} from '{file_path.name}' ...")

    if not file_path.exists():
        print(f'    [WARNING] File not found — skipping {sensor_name}')
        return None

    df = pd.read_csv(file_path, low_memory=False)
    print(f'    → {len(df):,} rows read')

    if DATE_COL not in df.columns:
        print(f"    [WARNING] Column '{DATE_COL}' not found — skipping {sensor_name}")
        return None

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors='coerce')
    df[DATE_COL] = df[DATE_COL].dt.tz_convert(LOCAL_TZ)

    for col in (lat_col, lon_col):
        if col not in df.columns:
            print(f"    [WARNING] Column '{col}' not found — skipping {sensor_name}")
            return None
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[DATE_COL, lat_col, lon_col]).reset_index(drop=True)
    df = df[(df[lat_col] != 0.0) & (df[lon_col] != 0.0)].reset_index(drop=True)

    if df.empty:
        print(f'    [WARNING] No valid rows remaining — skipping {sensor_name}')
        return None

    has_land_use = 'land_use_clcv' in df.columns
    if has_land_use:
        df['land_use_clcv'] = pd.to_numeric(df['land_use_clcv'], errors='coerce')

    df = df.sort_values(DATE_COL).reset_index(drop=True)

    distances = haversine_vectorized(
        df[lat_col].values[:-1], df[lon_col].values[:-1],
        df[lat_col].values[1:],  df[lon_col].values[1:],
        # DATE_COL is already parsed and converted to LOCAL_TZ above
        timestamps=df[DATE_COL].values,
    )
    df_dep = df.iloc[:-1].copy()
    df_dep['moving_status'] = np.where(distances >= THRESHOLD_M, 'Moving', 'Parked')

    keep_cols = [lat_col, lon_col, 'moving_status']
    if has_land_use:
        keep_cols.append('land_use_clcv')

    result = df_dep[keep_cols].copy()
    result = result.rename(columns={lat_col: 'lat', lon_col: 'lon'})

    n_moving = (result['moving_status'] == 'Moving').sum()
    n_parked = (result['moving_status'] == 'Parked').sum()
    print(f'    → {len(result):,} transitions classified  '
          f'(Moving: {n_moving:,}  |  Parked: {n_parked:,})')

    return result.reset_index(drop=True)


def apply_clc_filter(
    df:              pd.DataFrame,
    active_lu_vals:  list | None,
    filter_desc:     str | None,
) -> pd.DataFrame:
    """
    Apply the resolved CLC land-use filter to the pooled DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled sensor data.
    active_lu_vals : list[int] or None
        Land_use values to keep; None means no filter.
    filter_desc : str or None
        Label for log messages.

    Returns
    -------
    pd.DataFrame
        Filtered (or unmodified) DataFrame.
    """
    if active_lu_vals is None:
        return df

    if 'land_use_clcv' not in df.columns:
        raise RuntimeError(
            'A CLC land-use filter is active but the sensor CSV files do not '
            "contain a 'land_use_clcv' column.  Make sure the data was "
            'produced by join_clc_standard.py / join_clc_tuktuk.py (the '
            'CLC-vector join, the project\'s single land-use reference).'
        )

    n_before = len(df)
    df = df[df['land_use_clcv'].isin(active_lu_vals)].reset_index(drop=True)
    n_after  = len(df)
    print(f'  CLC filter: {filter_desc}')
    print(f'  → {n_after:,} / {n_before:,} points retained  '
          f'({100 * n_after / max(n_before, 1):.1f}%)')
    if df.empty:
        print('  [WARNING] No points remain — check FILTER_LAND_USE / FILTER_URBAN_RURAL.')
    return df


# ==============================================================================
# FIG A — PORTUGAL + LISBON MAP HELPER
# ==============================================================================

def _draw_and_save_map(
    pts:            pd.DataFrame,
    point_color:    str,
    legend_label:   str,
    main_title:     str,
    output_file:    Path,
    gdf1:           gpd.GeoDataFrame,
    gdf2:           gpd.GeoDataFrame,
    lisbon_gdf:     gpd.GeoDataFrame,
    lisbon_bounds:  tuple,
    footnote_extra: str = '',
) -> None:
    """
    Render and save one Portugal + Lisbon two-panel map (Fig A).

    Left panel  : Continental Portugal with district boundaries and scatter
                  points coloured by movement status.  A dashed rectangle
                  marks the Lisboa zoom area.
    Right panel : Zoomed view of the Municipality of Lisbon with municipality
                  boundaries.  Dotted connector lines link the zoom rectangle
                  to the right panel corners.

    Parameters
    ----------
    pts : pd.DataFrame
        Points to plot, with float columns ['lat', 'lon'].
    point_color : str
        Hex colour for the scatter markers.
    legend_label : str
        Text shown in the legend patch.
    main_title : str
        Title above the left (Portugal) panel.
    output_file : Path
        Full output PNG path.
    gdf1 : GeoDataFrame
        GADM level-1 (district) polygons.
    gdf2 : GeoDataFrame
        GADM level-2 (municipality) polygons.
    lisbon_gdf : GeoDataFrame
        Lisboa municipality polygon.
    lisbon_bounds : tuple
        (minx, miny, maxx, maxy) bounding box in WGS-84 degrees.
    footnote_extra : str
        Additional text for the figure footnote (e.g. the active CLC filter).
    """
    if pts.empty:
        print(f"  [WARNING] No points to plot for '{legend_label}' — skipping Fig A.")
        return

    print(f'\n  [Fig A] Drawing: {output_file.name}  ({len(pts):,} points) ...')

    fig, (ax_pt, ax_ls) = plt.subplots(
        1, 2, figsize=(13, 11),
        gridspec_kw={'width_ratios': [2.2, 1]},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.04)

    # ── Left panel: Continental Portugal ──────────────────────────────────────
    ax_pt.set_facecolor('white')
    for sp in ax_pt.spines.values():
        sp.set_visible(False)

    gdf1.boundary.plot(ax=ax_pt, color='#444444', linewidth=0.8, zorder=2)

    ax_pt.scatter(
        pts['lon'].values, pts['lat'].values,
        c=point_color, s=POINT_SIZE_MAIN, alpha=POINT_ALPHA,
        linewidths=0, zorder=3, rasterized=True,
    )

    rx = lisbon_bounds[0] - RECT_MARGIN
    ry = lisbon_bounds[1] - RECT_MARGIN
    rw = (lisbon_bounds[2] - lisbon_bounds[0]) + 2 * RECT_MARGIN
    rh = (lisbon_bounds[3] - lisbon_bounds[1]) + 2 * RECT_MARGIN

    ax_pt.add_patch(mpatches.Rectangle(
        (rx, ry), rw, rh,
        linewidth=1.2, edgecolor='#111111', facecolor='none',
        linestyle='--', zorder=5,
    ))
    ax_pt.set_xlim(-9.65, -6.05)
    ax_pt.set_ylim(36.75, 42.25)
    ax_pt.set_xticks([])
    ax_pt.set_yticks([])
    ax_pt.set_title(main_title, fontsize=11, fontweight='bold', pad=8)

    # ── Right panel: Municipality of Lisbon ───────────────────────────────────
    ax_ls.set_facecolor('white')
    for sp in ax_ls.spines.values():
        sp.set_linewidth(1.0)
        sp.set_edgecolor('#333333')

    zoom_margin = 0.04

    if ZOOM_BORDERS == 'viewport':
        viewport_box = shapely_box(
            lisbon_bounds[0] - zoom_margin, lisbon_bounds[1] - zoom_margin,
            lisbon_bounds[2] + zoom_margin, lisbon_bounds[3] + zoom_margin,
        )
        gdf2[gdf2.geometry.intersects(viewport_box)].boundary.plot(
            ax=ax_ls, color='#888888', linewidth=0.5, zorder=2,
        )

    lisbon_gdf.boundary.plot(ax=ax_ls, color='#111111', linewidth=1.4, zorder=3)

    if CLIP_LISBON_POINTS:
        lisbon_polygon = lisbon_gdf.geometry.union_all()
        pts_gdf = gpd.GeoDataFrame(
            pts,
            geometry=[ShapelyPoint(lon, lat)
                      for lon, lat in zip(pts['lon'].values, pts['lat'].values)],
            crs='EPSG:4326',
        )
        pts_zoom = pts_gdf[pts_gdf.geometry.within(lisbon_polygon)]
        print(f'       Lisbon panel: {len(pts_zoom):,} / {len(pts):,} points inside Lisboa.')
    else:
        pts_zoom = pts

    ax_ls.scatter(
        pts_zoom['lon'].values, pts_zoom['lat'].values,
        c=point_color, s=POINT_SIZE_ZOOM, alpha=POINT_ALPHA,
        linewidths=0, zorder=4, rasterized=True,
    )
    ax_ls.set_xlim(lisbon_bounds[0] - zoom_margin, lisbon_bounds[2] + zoom_margin)
    ax_ls.set_ylim(lisbon_bounds[1] - zoom_margin, lisbon_bounds[3] + zoom_margin)
    ax_ls.set_xticks([])
    ax_ls.set_yticks([])
    ax_ls.set_title('Municipality of Lisbon', fontsize=10, fontweight='bold', pad=6)

    # ── Connector lines (zoom rect → Lisbon panel) ────────────────────────────
    fig.canvas.draw()

    def data_to_fig(ax, xd, yd):
        """Convert data coordinates of `ax` to figure coordinates."""
        display = ax.transData.transform((xd, yd))
        return fig.transFigure.inverted().transform(display)

    bb_ls    = ax_ls.get_position()
    top_rect = data_to_fig(ax_pt, rx + rw, ry + rh)
    bot_rect = data_to_fig(ax_pt, rx + rw, ry)

    for start, end in [(top_rect, (bb_ls.x0, bb_ls.y1)),
                       (bot_rect, (bb_ls.x0, bb_ls.y0))]:
        fig.add_artist(plt.Line2D(
            [start[0], end[0]], [start[1], end[1]],
            transform=fig.transFigure, color='#555555',
            linewidth=0.8, linestyle=':', zorder=10, clip_on=False,
        ))

    # ── Legend ────────────────────────────────────────────────────────────────
    ax_pt.legend(
        handles=[mpatches.Patch(
            facecolor=point_color, edgecolor='none',
            label=f'{legend_label}  (n = {len(pts):,})',
        )],
        loc='lower right', fontsize=9, frameon=True, framealpha=0.9,
    )

    # ── Footnote ─────────────────────────────────────────────────────────────
    footnote = (
        f'Movement threshold: {THRESHOLD_M:.0f} m  |  '
        f'Data variant: {FILE_SUFFIX}  |  VIPV Research — FCUL'
    )
    if footnote_extra:
        footnote += f'  |  Filter: {footnote_extra}'
    fig.text(0.5, 0.005, footnote, ha='center', va='bottom',
             fontsize=7, style='italic', color='#666666')

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    print(f'       Fig A saved → {output_file}')
    plt.show()
    plt.close(fig)


# ==============================================================================
# FIG B — HEXBIN DENSITY MAP (representativity diagnostic)
# ==============================================================================

def _draw_and_save_hexbin(
    pts:            pd.DataFrame,
    point_color:    str,
    legend_label:   str,
    main_title:     str,
    output_file:    Path,
    viewport_merc:  tuple,
    gdf2:           gpd.GeoDataFrame,
    footnote_extra: str = '',
) -> None:
    """
    Render and save a hexagonal bin density map on an OSM basemap (Fig B).

    Purpose — representativity diagnostic

    The viewport is shared across all sub-figures produced in a single run
    (Moving and Parked figures use the same spatial extent), making direct
    visual comparison straightforward.

    Parameters
    ----------
    pts : pd.DataFrame
        Points to map, with float columns ['lat', 'lon'].
    point_color : str
        Colour used for the district boundary outline drawn on the figure.
    legend_label : str
        Movement-status label shown in the figure title (e.g. 'Moving points').
    main_title : str
        Full figure title string.
    output_file : Path
        Full output PNG path.
    viewport_merc : tuple
        (xmin, ymin, xmax, ymax) in EPSG:3857 metres — shared across all
        hexbin figures in this run so Moving and Parked are comparable.
    gdf2 : GeoDataFrame
        GADM level-2 (municipality) polygons in WGS-84 (reprojected internally).
    footnote_extra : str
        Additional text for the figure footnote.
    """
    if pts.empty:
        print(f"  [WARNING] No points to plot for '{legend_label}' — skipping Fig B.")
        return

    print(f'\n  [Fig B] Drawing hexbin: {output_file.name}  ({len(pts):,} points) ...')

    # ── Reproject points to EPSG:3857 (Web Mercator) ─────────────────────────
    x_merc, y_merc = wgs84_to_webmercator(pts['lon'].values, pts['lat'].values)

    xmin, ymin, xmax, ymax = viewport_merc

    # ── Derive gridsize from target physical cell size ────────────────────────
    # gridsize = number of hexagons along the x-axis.  Derived from the
    # viewport width so the physical cell size matches HEXBIN_CELL_SIZE_M
    # regardless of the chosen spatial extent.
    viewport_width_m = xmax - xmin
    gridsize = max(10, int(viewport_width_m / HEXBIN_CELL_SIZE_M))

    # ── Figure aspect ratio from viewport ─────────────────────────────────────
    aspect = (xmax - xmin) / max(ymax - ymin, 1)
    fig_w  = min(12.0, max(7.0, 10.0 * aspect))
    fig_h  = fig_w / aspect

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── OSM basemap ───────────────────────────────────────────────────────────
    _try_add_basemap(ax)

    # ── Hexbin overlay ────────────────────────────────────────────────────────
    hb = ax.hexbin(
        x_merc, y_merc,
        gridsize=gridsize,
        cmap=HEXBIN_COLORMAP,
        bins='log',                    # logarithmic colour scale
        mincnt=HEXBIN_MIN_COUNT,       # transparent below this count
        alpha=HEXBIN_ALPHA,
        linewidths=0.0,
        zorder=3,
        extent=(xmin, xmax, ymin, ymax),
    )

    # ── Colour bar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label('Point count per cell (log scale)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Lisbon municipality boundary ──────────────────────────────────────────
    try:
        gdf2_merc = gdf2.to_crs(epsg=3857)
        name_col  = 'NAME_2' if 'NAME_2' in gdf2_merc.columns else gdf2_merc.columns[2]
        lisbon_m  = gdf2_merc[gdf2_merc[name_col] == LISBON_NAME]
        if not lisbon_m.empty:
            lisbon_m.boundary.plot(
                ax=ax, color='#111111', linewidth=1.5, zorder=5,
            )
    except Exception as exc:
        print(f'  [WARNING] Could not draw Lisbon boundary on hexbin: {exc}')

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f'{main_title}\n'
        f'Hexagonal bin density map  —  cell size ≈ {HEXBIN_CELL_SIZE_M} m  '
        f'(log scale)',
        fontsize=11, fontweight='bold', pad=10,
    )

    # ── Footnote ──────────────────────────────────────────────────────────────
    footnote = (
        f'{len(pts):,} points  |  '
        f'threshold: {THRESHOLD_M:.0f} m  |  '
        f'min count: {HEXBIN_MIN_COUNT}  |  '
        f'variant: {FILE_SUFFIX}  |  '
        f'basemap © OpenStreetMap contributors  |  VIPV Research — FCUL'
    )
    if footnote_extra:
        footnote += f'  |  filter: {footnote_extra}'
    fig.text(0.5, 0.005, footnote, ha='center', va='bottom',
             fontsize=6, style='italic', color='#666666')

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    print(f'       Fig B saved → {output_file}')
    plt.show()
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load and classify the sensor data, then draw the selected figures."""
    print('=' * 70)
    print('  MOVEMENT DISTRIBUTION MAP — Moving vs. Parked GPS Points')
    print('=' * 70)
    print(f'  CAMPAIGN      : {CAMPAIGN}')
    if CAMPAIGN == "standard":
        print(f'  FILE_SUFFIX   : "{FILE_SUFFIX}"')
    else:
        print(f'  TUKTUK_FILE_SUFFIX : "{TUKTUK_FILE_SUFFIX}"')
    print(f'  Display mode  : DISPLAY_MODE       = "{DISPLAY_MODE}"')
    print(f'  Threshold     : THRESHOLD_M        = {THRESHOLD_M:.0f} m')
    print(f'  CLC filter 1  : FILTER_URBAN_RURAL = {FILTER_URBAN_RURAL!r}')
    print(f'  CLC filter 2  : FILTER_LAND_USE    = {FILTER_LAND_USE!r}')
    print(f'  Hexbin map    : SHOW_HEXBIN         = {SHOW_HEXBIN}')
    if SHOW_HEXBIN:
        print(f'  Hexbin cell   : HEXBIN_CELL_SIZE_M = {HEXBIN_CELL_SIZE_M} m')
    print(f'  Output folder : {OUTPUT_DIR}')

    # ── Validate DISPLAY_MODE ─────────────────────────────────────────────────
    valid_modes = ('all', 'moving', 'parked', 'both')
    if DISPLAY_MODE not in valid_modes:
        raise ValueError(
            f"DISPLAY_MODE must be one of {valid_modes!r}, got '{DISPLAY_MODE}'."
        )

    # ── Resolve CLC filter ────────────────────────────────────────────────────
    active_lu_vals, filter_tag, filter_desc = resolve_clc_filter(
        FILTER_LAND_USE, FILTER_URBAN_RURAL,
    )
    if filter_desc:
        print(f'\n  Active CLC filter : {filter_desc}')
    else:
        print('\n  No CLC filter active — all land-use classes included.')

    # ── 1. Load and classify sensors ──────────────────────────────────────────
    print(f'\n[1/4] Loading and classifying sensor data ({_VARIANT_TAG}) ...')

    frames: list = []
    for sensor_key, file_path in SENSOR_FILES.items():
        sensor_name = sensor_key.upper()
        if sensor_name not in SENSOR_COLS:
            print(f'  [WARNING] No column mapping for {sensor_name} — skipping.')
            continue
        df = load_and_classify_sensor(
            sensor_name, file_path,
            SENSOR_COLS[sensor_name]['lat'],
            SENSOR_COLS[sensor_name]['lon'],
        )
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            'No sensor CSV files could be loaded.  '
            'Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and '
            'FILE_SUFFIX/TUKTUK_FILE_SUFFIX in the configuration section.'
        )

    all_pts = pd.concat(frames, ignore_index=True)

    # ── 2. Apply CLC filter ───────────────────────────────────────────────────
    print('\n[2/4] Applying CLC land-use filter ...')
    all_pts = apply_clc_filter(all_pts, active_lu_vals, filter_desc)

    if all_pts.empty:
        print('  Nothing to plot — exiting.')
        return

    n_total  = len(all_pts)
    n_moving = (all_pts['moving_status'] == 'Moving').sum()
    n_parked = (all_pts['moving_status'] == 'Parked').sum()
    print('\n  ══ Dataset summary (after filter) ══')
    print(f'  Total : {n_total:,}')
    print(f'  Moving: {n_moving:,}  ({100 * n_moving / n_total:.1f}%)')
    print(f'  Parked: {n_parked:,}  ({100 * n_parked / n_total:.1f}%)')

    # ── 3. Load GADM shapefiles ───────────────────────────────────────────────
    print('\n[3/4] Loading GADM shapefiles ...')

    shp1 = SHAPEFILE_DIR / 'gadm41_PRT_1.shp'
    shp2 = SHAPEFILE_DIR / 'gadm41_PRT_2.shp'
    for p in (shp1, shp2):
        if not p.is_file():
            raise FileNotFoundError(
                f'Shapefile not found: {p}\n'
                'Download from https://gadm.org/ and update SHAPEFILE_DIR.'
            )

    gdf1 = gpd.read_file(shp1)
    gdf2 = gpd.read_file(shp2)
    print(f'  Districts: {len(gdf1)} features  |  '
          f'Municipalities: {len(gdf2)} features')

    name_col   = 'NAME_2' if 'NAME_2' in gdf2.columns else gdf2.columns[2]
    lisbon_gdf = gdf2[gdf2[name_col] == LISBON_NAME].copy()
    if lisbon_gdf.empty:
        lisbon_gdf = gdf2[gdf2[name_col].str.lower() == LISBON_NAME.lower()].copy()
    if lisbon_gdf.empty:
        raise ValueError(
            f"Municipality '{LISBON_NAME}' not found in GADM column '{name_col}'.\n"
            f"Available (first 30): {sorted(gdf2[name_col].unique())[:30]}"
        )

    lisbon_bounds = lisbon_gdf.total_bounds   # (minx, miny, maxx, maxy) in WGS-84
    print(f'  Lisboa bounds (WGS-84): {lisbon_bounds.round(4)}')

    # ── Pre-compute the shared hexbin viewport in EPSG:3857 ──────────────────
    # Moving and Parked hexbin figures always share the same spatial extent so
    # they are directly comparable.
    #
    # Two modes:
    #   HEXBIN_BBOX is set → convert the user-supplied WGS-84 bbox to EPSG:3857.
    #                        Points outside the bbox are clipped by hexbin's
    #                        extent parameter (excluded from cell counts).
    #   HEXBIN_BBOX is None → auto-detect from the full point cloud + margin.
    if SHOW_HEXBIN:
        if HEXBIN_BBOX is not None:
            lon_min, lat_min, lon_max, lat_max = HEXBIN_BBOX
            x0, y0 = wgs84_to_webmercator(
                np.array([lon_min, lon_max]),
                np.array([lat_min, lat_max]),
            )
            viewport_merc = (float(x0[0]), float(y0[0]),
                             float(x0[1]), float(y0[1]))
            print(
                f'\n  Hexbin viewport: manual bbox  '
                f'(lon [{lon_min}, {lon_max}]  lat [{lat_min}, {lat_max}])'
            )
        else:
            print('\n  Hexbin viewport: auto-detected from point cloud ...')
            x_all, y_all = wgs84_to_webmercator(
                all_pts['lon'].values, all_pts['lat'].values,
            )
            xmin_m, xmax_m = float(x_all.min()), float(x_all.max())
            ymin_m, ymax_m = float(y_all.min()), float(y_all.max())
            mx = (xmax_m - xmin_m) * HEXBIN_VIEWPORT_MARGIN
            my = (ymax_m - ymin_m) * HEXBIN_VIEWPORT_MARGIN
            viewport_merc = (xmin_m - mx, ymin_m - my,
                             xmax_m + mx, ymax_m + my)

        print(
            f'  Viewport EPSG:3857 (m): '
            f'x=[{viewport_merc[0]:.0f}, {viewport_merc[2]:.0f}]  '
            f'y=[{viewport_merc[1]:.0f}, {viewport_merc[3]:.0f}]  '
            f'→ width {(viewport_merc[2]-viewport_merc[0])/1000:.1f} km  '
            f'height {(viewport_merc[3]-viewport_merc[1])/1000:.1f} km  '
            f'→ hexbin gridsize ≈ {max(10, int((viewport_merc[2]-viewport_merc[0])/HEXBIN_CELL_SIZE_M))}'
        )
    else:
        viewport_merc = None

    # ── 4. Build filename / title helpers ─────────────────────────────────────
    def _outfile_a(mode_tag: str) -> Path:
        """Output path of the Portugal + Lisbon map (Fig A)."""
        parts = ['movement_distribution_map', CAMPAIGN, _VARIANT_TAG, mode_tag]
        if filter_tag:
            parts.append(filter_tag)
        return OUTPUT_DIR / ('_'.join(parts) + '.png')

    def _outfile_b(mode_tag: str) -> Path:
        """Output path of the hexbin density map (Fig B)."""
        parts = ['movement_hexbin', CAMPAIGN, _VARIANT_TAG, mode_tag]
        if filter_tag:
            parts.append(filter_tag)
        return OUTPUT_DIR / ('_'.join(parts) + '.png')

    def _title(base: str) -> str:
        """Figure title with the active CLC filter, if any."""
        return f'{base} — {filter_desc}' if filter_desc else base

    map_kwargs_a = dict(
        gdf1=gdf1, gdf2=gdf2,
        lisbon_gdf=lisbon_gdf, lisbon_bounds=lisbon_bounds,
        footnote_extra=filter_desc or '',
    )

    # ── 5. Draw figure(s) ────────────────────────────────────────────────────
    print(f'\n[4/4] Drawing figure(s) (DISPLAY_MODE="{DISPLAY_MODE}") ...')

    # Helper: draw both Fig A and Fig B for a given subset
    def _produce_figures(pts_sub: pd.DataFrame, mode_tag: str,
                         color: str, label: str, title_base: str) -> None:
        """Draw Fig A (and Fig B when SHOW_HEXBIN) for one movement subset."""
        # Fig A — Portugal + Lisbon two-panel map
        _draw_and_save_map(
            pts=pts_sub, point_color=color,
            legend_label=label,
            main_title=_title(title_base),
            output_file=_outfile_a(mode_tag),
            **map_kwargs_a,
        )
        # Fig B — Hexbin density map
        if SHOW_HEXBIN:
            _draw_and_save_hexbin(
                pts=pts_sub, point_color=color,
                legend_label=label,
                main_title=_title(title_base),
                output_file=_outfile_b(mode_tag),
                viewport_merc=viewport_merc,
                gdf2=gdf2,
                footnote_extra=filter_desc or '',
            )

    # ── (A) All classified points ─────────────────────────────────────────────
    if DISPLAY_MODE == 'all':
        _produce_figures(
            all_pts, 'all', POINT_COLOR_ALL,
            'All measurement points',
            'Continental Portugal — All measurements',
        )

    # ── (B) Moving points only ────────────────────────────────────────────────
    if DISPLAY_MODE in ('moving', 'both'):
        moving_pts = all_pts[all_pts['moving_status'] == 'Moving'].reset_index(drop=True)
        print(f'\n  Moving subset: {len(moving_pts):,} points '
              f'({100 * len(moving_pts) / max(n_total, 1):.1f}% of post-filter total)')
        _produce_figures(
            moving_pts, 'moving', POINT_COLOR_MOVING,
            'Moving points',
            'Continental Portugal — Moving measurements',
        )

    # ── (C) Parked points only ────────────────────────────────────────────────
    if DISPLAY_MODE in ('parked', 'both'):
        parked_pts = all_pts[all_pts['moving_status'] == 'Parked'].reset_index(drop=True)
        print(f'\n  Parked subset: {len(parked_pts):,} points '
              f'({100 * len(parked_pts) / max(n_total, 1):.1f}% of post-filter total)')
        _produce_figures(
            parked_pts, 'parked', POINT_COLOR_PARKED,
            'Parked points',
            'Continental Portugal — Parked measurements',
        )

    print('\nDone.')


# ==============================================================================
if __name__ == '__main__':
    main()