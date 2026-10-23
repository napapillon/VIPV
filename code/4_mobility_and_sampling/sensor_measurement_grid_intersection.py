"""
================================================================================
Sensor Measurement Grid Intersection
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Intersects the Solar Jinie measurements with a regular GRID_RES = 0.03°
(~3 km, about the CAMS pixel) grid over mainland Portugal, in four panels:
  (a) the regular grid over mainland Portugal
  (b) grid pixels with at least one measurement (MEASURED_PIXEL_COLOUR)
  (c) number of measurements per pixel, colour gradient (full extent)
  (d) close-up on the Lisbon municipality
The composite figure and each panel are saved as PDF and PNG (10 files).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon'
    boundary : BOUNDARIES_DIR / gadm41_PRT_0.shp (GADM 4.1, national
             boundary); if absent, the Natural Earth low-resolution boundary
             is downloaded from GitHub (internet access needed)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             smgi_composite_<CAMPAIGN>_<variant_tag>.pdf / .png
             smgi_panel_<a|b|c|d>_<CAMPAIGN>_<variant_tag>.pdf / .png

Requirements
------------
    pip install pandas numpy matplotlib geopandas

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
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import geopandas as gpd

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
FILE_SUFFIX = "sd_cleaned"               # standard campaign
TUKTUK_FILE_SUFFIX = "sd_cleaned"        # tuk-tuk campaign

# --- Datetime column (same in both campaigns) ---------------------------------
DATE_COL = 'Human readable date (Europe/London)'

# --- Grid resolution [decimal degrees] ----------------------------------------
GRID_RES: float = 0.03

# --- Spatial bounding boxes ---------------------------------------------------
# Mainland Portugal
LAT_MIN, LAT_MAX = 36.9, 42.2
LON_MIN, LON_MAX = -9.6, -6.1

# Lisbon close-up (panel d)
LISBON_LAT_MIN, LISBON_LAT_MAX = 38.65, 38.95
LISBON_LON_MIN, LISBON_LON_MAX = -9.35, -8.95

# --- Visual options -----------------------------------------------------------
# Colour used in panel (b) to mark pixels with at least one measurement
MEASURED_PIXEL_COLOUR: str = '#FFC300'

# --- Boundary data ------------------------------------------------------------
# GADM shapefiles (gadm.org); if gadm41_PRT_0.shp is absent, the Natural Earth
# boundary is downloaded from GitHub
GADM_DIR = BOUNDARIES_DIR

# --- Output -------------------------------------------------------------------
# Sub-folder per campaign / variant created automatically
OUTPUT_BASE_DIR = OUTPUT_ROOT / "4_mobility_and_sampling" / "smgi_outputs"
OUTPUT_DPI = 300

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# Input and output paths follow from CAMPAIGN and the input suffix.

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        file_suffix: str, tuktuk_file_suffix: str) -> dict:
    """Build the dict mapping sensor names to their CSV input paths.

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
        {'J30': Path(...), 'J31': Path(...), ...}
    """
    sensors = ['J30', 'J31', 'J32', 'J33']
    if campaign == "standard":
        return {
            sensor: data_dir / f"{sensor.lower()}_satellite_landuse_v2_{file_suffix}.csv"
            for sensor in sensors
        }
    elif campaign == "tuktuk":
        sep = f"_{tuktuk_file_suffix}" if tuktuk_file_suffix else ""
        return {
            sensor: tuktuk_data_dir / f"{sensor.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv"
            for sensor in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign.

    Used throughout for output directory / filename construction, so that
    results from different campaigns or variants are always kept separate.
    """
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Build the output directory path from the base directory, campaign,
    and active variant tag.

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


def _build_output_stem(base_name: str, campaign: str, variant_tag: str) -> str:
    """Build an output filename stem that embeds the campaign and variant tag.

    The pattern is:  <base_name>_<campaign>_<variant_tag>

    Parameters
    ----------
    base_name : str
        Descriptive prefix, e.g. 'smgi_composite' or 'smgi_panel_a'.
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().

    Returns
    -------
    str
        Stem string, e.g. 'smgi_composite_standard_sd_cleaned_lisbon'.
    """
    return f"{base_name}_{campaign}_{variant_tag}"


# Resolve paths at import time so they are available throughout the module
SENSOR_FILES = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, FILE_SUFFIX, TUKTUK_FILE_SUFFIX
)
_VARIANT_TAG = _variant_tag()
OUTPUT_DIR   = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_sensor_data(sensor_name: str, file_path: Path) -> pd.DataFrame | None:
    """Load and prepare one cleaned sensor CSV file.

    Renames sensor-specific columns (e.g. 'J31 lat') to generic internal
    names ('lat', 'lon', 'rad_raw', 'rad_calib') and parses timestamps.
    Applies a minimal safety filter to guard against any residual sentinel
    values that may have survived the cleaning pipeline.

    Parameters
    ----------
    sensor_name : str
        Sensor identifier used as column prefix, e.g. ``'J31'``.
    file_path : Path
        Path to the cleaned CSV file for this sensor.

    Returns
    -------
    pd.DataFrame or None
        Cleaned DataFrame with standardised columns, or ``None`` if the
        file does not exist or cannot be loaded.

    Notes
    -----
    Timestamps stored as ``Europe/London`` (with DST-aware offsets) are parsed
    with ``utc=True``, which converts them to UTC without ambiguity across
    winter (+00:00) and summer (+01:00) periods.
    """
    if not file_path.exists():
        print(f'  [WARNING] File not found, skipping : {file_path}')
        return None

    print(f'  Loading {file_path.name} ...')
    df = pd.read_csv(file_path, low_memory=False)

    # Remove spurious index columns produced by double-export (e.g. 'Unnamed: 0')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # Rename sensor-specific columns to generic internal names
    rename_map = {
        DATE_COL:                        'date_str',
        f'{sensor_name} lat':       'lat',
        f'{sensor_name} lon':       'lon',
        f'{sensor_name} Rad raw':   'rad_raw',
        f'{sensor_name} Rad calib': 'rad_calib',
    }
    df = df.rename(columns=rename_map)

    # Parse timestamps: utc=True converts Europe/London offsets to UTC.
    # Result dtype: DatetimeTZDtype(tz=UTC)
    df['datetime_utc'] = pd.to_datetime(df['date_str'], utc=True)
    df = df.drop(columns=['date_str'])

    # Safety filter — remove any residual sentinel land-use values.
    df = df[df['land_use_clcv'] != -999]

    # Keep only daytime records (sun above horizon) and valid coordinates
    df = df[df['sun_elev'] > 0]
    df = df.dropna(subset=['lat', 'lon', 'rad_calib', 'ghi'])

    # Clip to mainland Portugal bounding box
    df = df[
        (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)
    ]

    df['sensor'] = sensor_name
    print(f'    {len(df):>7,} valid rows retained')
    return df


def load_all_sensors(sensor_files: dict) -> pd.DataFrame:
    """Load and concatenate all sensor CSV files.

    Parameters
    ----------
    sensor_files : dict
        Mapping of sensor name to cleaned CSV path (see ``SENSOR_FILES``).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all valid sensor records.

    Raises
    ------
    FileNotFoundError
        If none of the listed CSV files can be loaded.
    """
    frames = []
    for sensor_name, file_path in sensor_files.items():
        print(f'\n  -- Sensor {sensor_name} --')
        df = load_sensor_data(sensor_name, file_path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            'No sensor CSV files could be loaded. '
            'Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and '
            'FILE_SUFFIX/TUKTUK_FILE_SUFFIX.'
        )
    return pd.concat(frames, ignore_index=True)


# ==============================================================================
# 2. GRID ASSIGNMENT & AGGREGATION
# ==============================================================================

def assign_grid_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each measurement to the nearest grid pixel.

    Each pixel is identified by the coordinates of its lower-left corner,
    consistent with a centroid at (lon_px + GRID_RES/2, lat_px + GRID_RES/2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``lat`` and ``lon`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns: ``lat_px`` and ``lon_px``.
    """
    lat_edges = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
    lon_edges = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)

    # Labels = lower-left corner of each bin; right=False → left-closed intervals
    df['lat_px'] = pd.cut(
        df['lat'], bins=lat_edges, labels=lat_edges[:-1], right=False
    ).astype(float)
    df['lon_px'] = pd.cut(
        df['lon'], bins=lon_edges, labels=lon_edges[:-1], right=False
    ).astype(float)

    return df.dropna(subset=['lat_px', 'lon_px'])


def compute_pixel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of measurements per grid pixel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``lat_px`` and ``lon_px`` columns (from
        :func:`assign_grid_pixels`).

    Returns
    -------
    pd.DataFrame
        One row per occupied pixel with columns
        ``lat_px``, ``lon_px``, ``n_measurements``.
    """
    return (
        df.groupby(['lat_px', 'lon_px'], observed=True)
          .size()
          .reset_index(name='n_measurements')
    )


def compute_pixel_mean_irradiance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean calibrated irradiance per grid pixel.

    This optional aggregation is useful for spatial irradiance maps,
    e.g. shading loss analysis by region.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``lat_px``, ``lon_px``, and ``rad_calib`` columns.

    Returns
    -------
    pd.DataFrame
        One row per occupied pixel with columns
        ``lat_px``, ``lon_px``, ``mean_irradiance_Wm2``.
    """
    return (
        df.groupby(['lat_px', 'lon_px'], observed=True)['rad_calib']
          .mean()
          .reset_index(name='mean_irradiance_Wm2')
    )


# ==============================================================================
# 3. MAPPING HELPERS
# ==============================================================================

def draw_grid_lines(ax, lat_min=LAT_MIN, lat_max=LAT_MAX,
                    lon_min=LON_MIN, lon_max=LON_MAX,
                    res=GRID_RES, **kwargs) -> None:
    """Overlay a regular lat/lon grid on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    lat_min, lat_max, lon_min, lon_max : float
        Spatial extent of the grid.
    res : float
        Grid spacing in decimal degrees.
    **kwargs
        Forwarded to ``axhline`` / ``axvline``.
    """
    style = dict(color='#bbbbbb', lw=0.25, alpha=0.7)
    style.update(kwargs)
    for lat in np.arange(lat_min, lat_max + res, res):
        ax.axhline(lat, **style)
    for lon in np.arange(lon_min, lon_max + res, res):
        ax.axvline(lon, **style)


def draw_pixel_rectangles(ax, counts_df: pd.DataFrame,
                           cmap, norm, alpha: float = 0.85) -> None:
    """Draw one coloured rectangle per occupied grid pixel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    counts_df : pd.DataFrame
        Pixel count table from :func:`compute_pixel_counts`.
    cmap : matplotlib.colors.Colormap
    norm : matplotlib.colors.Normalize
    alpha : float
        Rectangle opacity.
    """
    for _, row in counts_df.iterrows():
        colour = cmap(norm(row['n_measurements']))
        rect = mpatches.Rectangle(
            (row['lon_px'], row['lat_px']),
            GRID_RES, GRID_RES,
            linewidth=0,
            facecolor=colour,
            alpha=alpha,
        )
        ax.add_patch(rect)


def set_map_extent(ax, lon_min: float, lon_max: float,
                   lat_min: float, lat_max: float) -> None:
    """Set axis limits and enforce equal aspect ratio."""
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal')


def style_axes(ax, title: str = '', xlabel: str = 'Longitude',
               ylabel: str = 'Latitude', fontsize: int = 9) -> None:
    """Apply consistent styling to a map subplot."""
    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold',
                 loc='left', pad=4)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)


# ==============================================================================
# 4. COLOUR NORMALISATION
# ==============================================================================

def build_colour_norm(counts: pd.DataFrame) -> tuple:
    """Compute colour map and normalisation from pixel counts.

    Uses a linear scale capped at the 95th percentile so that extreme
    hotspots do not wash out the rest of the map.

    Parameters
    ----------
    counts : pd.DataFrame
        Pixel count table from :func:`compute_pixel_counts`.

    Returns
    -------
    tuple
        ``(cmap, norm, scale_label)`` ready to pass to drawing functions.
    """
    n_min     = 1
    n_max     = int(counts['n_measurements'].quantile(0.95))
    n_abs_max = int(counts['n_measurements'].max())
    norm      = mcolors.Normalize(vmin=n_min, vmax=n_max)
    cmap      = plt.colormaps['viridis']
    scale_label = 'linear scale (capped at 95th percentile)'
    print(f'  [INFO] Colour scale : {scale_label}  '
          f'(min={n_min}, cap={n_max}, absolute max={n_abs_max})')
    return cmap, norm, scale_label


# ==============================================================================
# 5. INDIVIDUAL PANEL FIGURES
# ==============================================================================

def plot_panel_a(portugal_gdf: gpd.GeoDataFrame,
                 ax=None) -> plt.Figure | None:
    """(a) Regular 0.03° grid over mainland Portugal — no measurements.

    Can be called in two modes:
      - **Standalone** (``ax=None``): creates its own figure and returns it.
      - **Composite** (``ax`` provided): draws into the given axes, returns None.

    Parameters
    ----------
    portugal_gdf : gpd.GeoDataFrame
        National boundary geometry (CRS: EPSG:4326).
    ax : matplotlib.axes.Axes, optional
        Target axes for composite mode.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 7))
    else:
        fig = None

    portugal_gdf.plot(ax=ax, facecolor='none', edgecolor='black', lw=1.2)
    draw_grid_lines(ax)
    set_map_extent(ax, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    style_axes(ax, title='(a) — Regular 0.03° grid' if standalone else '(a)')

    if standalone:
        fig.suptitle('Regular 0.03° grid over mainland Portugal',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
    return fig


def plot_panel_b(portugal_gdf: gpd.GeoDataFrame,
                 counts: pd.DataFrame,
                 ax=None) -> plt.Figure | None:
    """(b) Grid pixels with at least one measurement (amber).

    Parameters
    ----------
    portugal_gdf : gpd.GeoDataFrame
    counts : pd.DataFrame
        Pixel count table from :func:`compute_pixel_counts`.
    ax : matplotlib.axes.Axes, optional
        Target axes for composite mode.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 7))
    else:
        fig = None

    portugal_gdf.boundary.plot(ax=ax, color='black', lw=1.2)
    for _, row in counts.iterrows():
        rect = mpatches.Rectangle(
            (row['lon_px'], row['lat_px']),
            GRID_RES, GRID_RES,
            linewidth=0,
            facecolor=MEASURED_PIXEL_COLOUR,
            alpha=0.85,
        )
        ax.add_patch(rect)
    set_map_extent(ax, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    style_axes(ax, title='(b) — Pixels with measurements' if standalone else '(b)')

    if standalone:
        fig.suptitle('Grid pixels with at least one Solar Jinie measurement',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
    return fig


def plot_panel_c(portugal_gdf: gpd.GeoDataFrame,
                 counts: pd.DataFrame,
                 cmap, norm, scale_label: str,
                 ax=None) -> plt.Figure | None:
    """(c) Measurement density across mainland Portugal.

    Parameters
    ----------
    portugal_gdf : gpd.GeoDataFrame
    counts : pd.DataFrame
    cmap : matplotlib.colors.Colormap
    norm : matplotlib.colors.Normalize
    scale_label : str
        Colour scale description used in the colorbar label.
    ax : matplotlib.axes.Axes, optional
        Target axes for composite mode.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 7))
    else:
        fig = ax.get_figure()

    portugal_gdf.boundary.plot(ax=ax, color='black', lw=1.2)
    draw_pixel_rectangles(ax, counts, cmap, norm)
    set_map_extent(ax, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    style_axes(ax, title='(c) — Measurement density' if standalone else '(c)')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('Number of measurements per pixel', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if standalone:
        fig.suptitle('Measurement density per 0.03° grid pixel — mainland Portugal',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        return fig
    return None


def plot_panel_d(portugal_gdf: gpd.GeoDataFrame,
                 counts: pd.DataFrame,
                 cmap, norm,
                 ax=None) -> plt.Figure | None:
    """(d) Measurement density — Lisbon municipality close-up.

    Parameters
    ----------
    portugal_gdf : gpd.GeoDataFrame
    counts : pd.DataFrame
    cmap : matplotlib.colors.Colormap
    norm : matplotlib.colors.Normalize
    ax : matplotlib.axes.Axes, optional
        Target axes for composite mode.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = None

    portugal_gdf.boundary.plot(ax=ax, color='black', lw=1.5)
    draw_pixel_rectangles(ax, counts, cmap, norm)
    set_map_extent(ax,
                   LISBON_LON_MIN, LISBON_LON_MAX,
                   LISBON_LAT_MIN, LISBON_LAT_MAX)
    style_axes(ax, title='(d) — Lisbon close-up' if standalone else '(d)')

    if standalone:
        fig.suptitle('Measurement density — Lisbon municipality close-up',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
    return fig


# ==============================================================================
# 6. COMPOSITE FOUR-PANEL FIGURE
# ==============================================================================

def build_composite_figure(portugal_gdf: gpd.GeoDataFrame,
                            counts: pd.DataFrame,
                            cmap, norm,
                            scale_label: str) -> plt.Figure:
    """Assemble the four panels into a single composite figure.

    Calls each ``plot_panel_*`` function in composite mode (passing the
    target axes) so that all rendering logic stays in one place.

    Parameters
    ----------
    portugal_gdf : gpd.GeoDataFrame
    counts : pd.DataFrame
    cmap : matplotlib.colors.Colormap
    norm : matplotlib.colors.Normalize
    scale_label : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(13, 15))
    gs  = GridSpec(
        2, 2, figure=fig,
        hspace=0.32, wspace=0.28,
        left=0.07, right=0.95, top=0.93, bottom=0.05,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_panel_a(portugal_gdf, ax=ax_a)
    plot_panel_b(portugal_gdf, counts, ax=ax_b)
    plot_panel_c(portugal_gdf, counts, cmap, norm, scale_label, ax=ax_c)
    plot_panel_d(portugal_gdf, counts, cmap, norm, ax=ax_d)

    fig.suptitle(
        'Spatial intersection of Solar Jinie sensor measurements '
        'and the 0.03° regular grid',
        fontsize=11, fontweight='bold', y=0.97,
    )
    return fig


# ==============================================================================
# 7. BOUNDARY LOADING
# ==============================================================================

def load_portugal_boundary() -> gpd.GeoDataFrame:
    """Load mainland Portugal national boundary (GADM level 0).

    Tries the local GADM shapefile first; falls back to Natural Earth
    via GitHub if the local file is not found.

    Returns
    -------
    gpd.GeoDataFrame
        Boundary geometry in EPSG:4326 (WGS 84).
    """
    local = GADM_DIR / 'gadm41_PRT_0.shp'
    if local.exists():
        print('  [INFO] Using local GADM shapefile (level 0 — national boundary).')
        return gpd.read_file(local).to_crs(epsg=4326)

    print('  [WARNING] Local GADM shapefile not found — downloading Natural Earth boundary ...')
    url = (
        'https://raw.githubusercontent.com/nvkelso/natural-earth-vector'
        '/master/geojson/ne_110m_admin_0_countries.geojson'
    )
    world = gpd.read_file(url)
    pt    = world[world['NAME'] == 'Portugal'].copy()
    if pt.empty:
        raise RuntimeError('Could not find Portugal in Natural Earth data.')
    return pt.to_crs(epsg=4326)


# ==============================================================================
# 8. SAVE HELPER
# ==============================================================================

def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a figure as both PDF (300 dpi) and PNG (200 dpi).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    stem : Path
        Full path stem without extension, e.g. ``Path('/output/my_figure')``.
        The function appends ``.pdf`` and ``.png`` automatically.
    """
    for ext, dpi in (('pdf', OUTPUT_DPI), ('png', 200)):
        out = stem.with_suffix(f'.{ext}')
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        print(f'    → Saved : {out}')


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the full grid intersection pipeline for all sensors."""
    print('\nSensor Measurement Grid Intersection')
    print('=' * 70)
    print(f'  CAMPAIGN            : {CAMPAIGN}')
    if CAMPAIGN == "standard":
        print(f'  FILE_SUFFIX         : "{FILE_SUFFIX}"')
    else:
        print(f'  TUKTUK_FILE_SUFFIX  : "{TUKTUK_FILE_SUFFIX}"')

    # 1. Create output directory if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'  Output directory    : {OUTPUT_DIR}')

    # 2. Load and merge all sensor data
    print(f'\n[1/5] Loading sensor data ({_VARIANT_TAG}) ...')
    data = load_all_sensors(SENSOR_FILES)
    print(f'\n  Total : {len(data):>7,} valid measurements across all sensors')

    # 3. Assign measurements to grid pixels and compute counts
    print('\n[2/5] Assigning measurements to 0.03° grid pixels ...')
    data   = assign_grid_pixels(data)
    counts = compute_pixel_counts(data)
    print(
        f'  {len(counts):,} pixels with at least one measurement\n'
        f'  Measurements per pixel — '
        f'min: {counts["n_measurements"].min()}, '
        f'max: {counts["n_measurements"].max()}, '
        f'median: {counts["n_measurements"].median():.0f}'
    )

    # 4. Load Portugal boundary geometry
    print('\n[3/5] Loading Portugal boundary ...')
    portugal = load_portugal_boundary()

    # 5. Compute shared colour normalisation
    print('\n[4/5] Computing colour normalisation ...')
    cmap, norm, scale_label = build_colour_norm(counts)

    # 6. Generate and save all figures
    print('\n[5/5] Generating figures ...')

    # Composite four-panel figure
    print('\n  Composite figure ...')
    fig_composite = build_composite_figure(portugal, counts, cmap, norm, scale_label)
    save_figure(
        fig_composite,
        OUTPUT_DIR / _build_output_stem('smgi_composite', CAMPAIGN, _VARIANT_TAG)
    )
    plt.close(fig_composite)

    # Individual panel figures
    panels = {
        'panel_a': plot_panel_a(portugal),
        'panel_b': plot_panel_b(portugal, counts),
        'panel_c': plot_panel_c(portugal, counts, cmap, norm, scale_label),
        'panel_d': plot_panel_d(portugal, counts, cmap, norm),
    }
    for name, fig in panels.items():
        print(f'\n  {name} ...')
        save_figure(
            fig,
            OUTPUT_DIR / _build_output_stem(f'smgi_{name}', CAMPAIGN, _VARIANT_TAG)
        )
        plt.close(fig)

    print('\n' + '=' * 70)
    print('  Grid intersection complete. All figures saved.')
    print(f'  Output directory : {OUTPUT_DIR}')
    print('=' * 70 + '\n')


# ==============================================================================
if __name__ == '__main__':
    main()