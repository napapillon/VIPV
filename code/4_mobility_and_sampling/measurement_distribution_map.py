"""
================================================================================
Portugal Measurement Distribution Map
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Maps all GPS measurement locations of the Solar Jinie sensors on continental
Portugal, with a zoom panel on the Municipality of Lisbon. Three figures:
  (A) all measurement points   (golden yellow)
  (B) urban points only        (coral red)    — CLC_URBAN_RURAL = 'Urban'
  (C) rural points only        (forest green) — CLC_URBAN_RURAL = 'Rural'
The Urban / Rural group comes from land_use_clcv (clc_mapping.py); rows
outside grid_code 1–44 are excluded.

--------------------------------------------------------------------------------
Layout (each figure)
--------------------------------------------------------------------------------
  Left panel  – Continental Portugal with district boundaries (GADM level 1)
                and the measurement points of the subset; a dashed rectangle
                marks the Lisboa zoom area.
  Right panel – Zoom on the Municipality of Lisbon with municipality
                boundaries (GADM level 2); dotted lines connect the zoom
                rectangle to the corners of the panel.

Rendering modes of the Lisbon panel (CLIP_LISBON_POINTS, ZOOM_BORDERS):
  1. all points, all borders in the viewport : False, 'viewport'
  2. Lisbon points, Lisbon border only        : True,  'lisbon_only'
  3. Lisbon points, all borders [default]     : True,  'viewport'

Optional CLC filter: FILTER_LAND_USE (list of grid_codes, e.g. [4] for road
and rail networks) restricts all figures to those classes; the urban / rural
figures are then skipped and file names get a '_lu<codes>' suffix.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon' (detected from the
             headers), 'land_use_clcv'; missing files are skipped
    shapefiles : SHAPEFILE_DIR / gadm41_PRT_1.shp (districts) and
             gadm41_PRT_2.shp (municipalities), GADM 4.1 —
             https://gadm.org/download_country.html (country = Portugal)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             measurement_distribution_map_<CAMPAIGN>_<variant_tag>.png
             measurement_distribution_map_<CAMPAIGN>_<variant_tag>_urban.png
             measurement_distribution_map_<CAMPAIGN>_<variant_tag>_rural.png

Requirements
------------
    pip install pandas numpy matplotlib geopandas shapely

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 5.1
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BOUNDARIES_DIR, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import box as shapely_box

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from clc_mapping import CLC_URBAN_RURAL, CLC_MAPPING, CLC_LVL3_LABEL

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

# --- GADM shapefiles directory ------------------------------------------------
SHAPEFILE_DIR = BOUNDARIES_DIR

# --- Measurement point style --------------------------------------------------
POINT_COLOR:      str   = "#D4A017"   # golden yellow
POINT_SIZE_MAIN:  float = 3.0         # marker size in the main panel (points²)
POINT_SIZE_ZOOM:  float = 7.0         # marker size in the Lisbon panel (points²)
POINT_ALPHA:      float = 0.55        # marker transparency (0 = transparent, 1 = opaque)

# --- Point colours for the Urban / Rural sub-figures -------------------------
# These replace POINT_COLOR in the respective filtered figures so that the
# three outputs are visually distinct at a glance.
POINT_COLOR_URBAN: str = "#C0392B"   # coral red   — urban subset
POINT_COLOR_RURAL: str = "#2E7D32"   # forest green — rural subset

# --- Lisboa municipality name as it appears in GADM NAME_2 column -------------
LISBON_NAME: str = "Lisboa"

# --- Zoom rectangle padding (degrees) around Lisboa bounding box --------------
RECT_MARGIN: float = 0.12

# --- Rendering mode for the Lisbon panel (see "Rendering modes" above) -------
# Whether to clip measurement points to the Lisboa municipality polygon only.
#   True  → only points strictly inside the municipality are shown  [default]
#   False → all points in the viewport bounding box are shown
CLIP_LISBON_POINTS: bool = True

# Which administrative boundaries to draw in the Lisbon panel:
#   'viewport'    → all municipalities that intersect the viewport  [default]
#   'lisbon_only' → Lisboa municipality outline only (no neighbours)
ZOOM_BORDERS: str = 'viewport'   # 'viewport' | 'lisbon_only'

# --- CLC class filter (optional) ----------------------------------------------
# Set FILTER_LAND_USE to a list of land_use values (= grid_code, 1-based) to
# restrict ALL figures to points from those specific CLC classes only.
# Leave as an empty list [] to keep the default behaviour (all classes shown).
#
# Use CLC_MAPPING keys from clc_mapping.py for reference.  Examples:
#
#   Single class:
#     FILTER_LAND_USE = [4]    # Road and rail networks and associated land (122)
#
#   Multiple classes:
#     FILTER_LAND_USE = [1, 2] # Continuous + Discontinuous urban fabric
#
# When active, every figure title and legend label is updated automatically
# to reflect the active filter, and output filenames include a "_luXX" suffix
# so they never overwrite the unfiltered outputs.
FILTER_LAND_USE: list[int] = []   # [] = no filter — show all classes

# --- Output (sub-folder per campaign / variant created automatically) --------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "4_mobility_and_sampling" / "mdm_outputs"
OUTPUT_DPI: int = 300

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# All input file paths, the output directory, and the output filename are
# derived automatically from DATA_DIR and FILE_SUFFIX, so that changing
# FILE_SUFFIX is the only edit needed to switch between data variants.

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        file_suffix: str, tuktuk_file_suffix: str) -> dict:
    """
    Build the dict mapping sensor names to their CSV input paths.

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
        {'j30': Path(...), 'j31': Path(...), ...}
    """
    sensors = ['j30', 'j31', 'j32', 'j33']
    if campaign == "standard":
        return {
            sensor: data_dir / f"{sensor}_satellite_landuse_v2_{file_suffix}.csv"
            for sensor in sensors
        }
    elif campaign == "tuktuk":
        sep = f"_{tuktuk_file_suffix}" if tuktuk_file_suffix else ""
        return {
            sensor: tuktuk_data_dir / f"{sensor}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv"
            for sensor in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign (used in
    output directory / filename construction)."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """
    Build the output directory path, embedding the campaign and active
    variant tag so results from different campaigns / variants never
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


def _build_output_filename(base_name: str, campaign: str, variant_tag: str,
                           extension: str) -> str:
    """
    Build an output filename that embeds the campaign and active variant tag.

    Parameters
    ----------
    base_name : str
        Descriptive part of the filename (e.g. 'measurement_distribution_map').
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().
    extension : str
        File extension without leading dot (e.g. 'png').

    Returns
    -------
    str
        Complete filename, e.g.
        'measurement_distribution_map_standard_sd_cleaned.png'.
    """
    return f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolve paths at import time so they are available throughout the module
SENSOR_FILES = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, FILE_SUFFIX, TUKTUK_FILE_SUFFIX
)
_VARIANT_TAG = _variant_tag()
OUTPUT_DIR   = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def find_coord_col(columns: list[str], keyword: str) -> str | None:
    """
    Return the first column name whose text best matches a coordinate keyword.

    The search priority is:
      1. Column name ends with the keyword  (e.g. 'J31 lat')
      2. Column name starts with the keyword
      3. Keyword appears as a whole word    (avoids 'lon' matching 'London')

    Parameters
    ----------
    columns : list[str]
        Column names of the DataFrame.
    keyword : str
        Target keyword, typically 'lat' or 'lon'.

    Returns
    -------
    str or None
        Matching column name, or None if no match is found.
    """
    for c in columns:
        if c.lower().endswith(keyword):
            return c
    for c in columns:
        if c.lower().startswith(keyword):
            return c
    for c in columns:
        if re.search(r'\b' + keyword + r'\b', c.lower()):
            return c
    return None


def load_sensor_data(filepath: Path, sensor_label: str) -> pd.DataFrame | None:
    """
    Load GPS coordinates and CLC land-use group from a cleaned sensor CSV file.

    The land_use filter is applied on the full DataFrame before lat/lon
    extraction to keep row indices aligned throughout.  Coordinates are cast
    to float via pd.to_numeric so that matplotlib never receives non-numeric
    values (which would trigger a categorical-axis ConversionError).

    A 'clc_group' column ('Urban' | 'Rural') is derived from land_use_clcv via
    CLC_URBAN_RURAL (imported from clc_mapping.py).  Rows whose land_use_clcv
    value is outside the valid range 1–44 receive 'Unknown' and are excluded
    from the urban / rural sub-figures.

    Parameters
    ----------
    filepath     : Path to the cleaned CSV file.
    sensor_label : Short identifier ('j30', 'j31', ...) used in log messages.

    Returns
    -------
    pd.DataFrame with float columns ['lat', 'lon'], integer column
    ['land_use_clcv'], and string column ['clc_group'], or None if the file is
    missing or contains no valid GPS rows after filtering.
    """
    if not filepath.is_file():
        print(f"  [WARNING] File not found — skipping: {filepath}")
        return None

    df = pd.read_csv(filepath, low_memory=False)
    print(f"  {sensor_label}: {len(df):,} rows read from '{filepath.name}'")

    # Auto-detect lat / lon column names
    lat_col = find_coord_col(list(df.columns), 'lat')
    lon_col = find_coord_col(list(df.columns), 'lon')

    if lat_col is None or lon_col is None:
        print(f"  [WARNING] Lat/lon columns not found in {filepath.name} — skipping.")
        return None

    # Exclude out-of-range land_use_clcv values
    if 'land_use_clcv' in df.columns:
        df['land_use_clcv'] = pd.to_numeric(df['land_use_clcv'], errors='coerce')
        n_before = len(df)
        df = df[(df['land_use_clcv'] >= 1) & (df['land_use_clcv'] <= 44)].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"    -> {n_dropped:,} rows excluded (land_use_clcv out of range 1–44)")
    else:
        df['land_use_clcv'] = pd.NA   # column absent: create a placeholder

    # Extract coordinates and land_use_clcv, cast to correct types
    subset = df[[lat_col, lon_col, 'land_use_clcv']].copy()
    subset.columns = ['lat', 'lon', 'land_use_clcv']
    subset['lat']           = pd.to_numeric(subset['lat'],           errors='coerce')
    subset['lon']           = pd.to_numeric(subset['lon'],           errors='coerce')
    subset['land_use_clcv'] = pd.to_numeric(subset['land_use_clcv'], errors='coerce').astype('Int64')

    # Drop rows with missing coordinates or zero-coordinate GPS fix
    subset = subset.dropna(subset=['lat', 'lon'])
    subset = subset[(subset['lat'] != 0.0) & (subset['lon'] != 0.0)]

    # Derive CLC group (Urban / Rural / Unknown) from land_use_clcv (= grid_code, 1-based)
    subset['clc_group'] = subset['land_use_clcv'].map(CLC_URBAN_RURAL).fillna('Unknown')

    print(f"    -> {len(subset):,} valid GPS points retained")
    if len(subset) > 0:
        print(f"       lat [{subset['lat'].min():.3f}, {subset['lat'].max():.3f}]  "
              f"lon [{subset['lon'].min():.3f}, {subset['lon'].max():.3f}]")
        urban_n = (subset['clc_group'] == 'Urban').sum()
        rural_n = (subset['clc_group'] == 'Rural').sum()
        print(f"       Urban: {urban_n:,}  |  Rural: {rural_n:,}  "
              f"|  Unknown: {len(subset) - urban_n - rural_n:,}")

    return subset.reset_index(drop=True) if len(subset) > 0 else None


# ==============================================================================
# FIGURE DRAWING HELPER
# ==============================================================================

def _draw_and_save_map(
    pts:          pd.DataFrame,
    point_color:  str,
    legend_label: str,
    main_title:   str,
    output_file:  Path,
    gdf1:         gpd.GeoDataFrame,
    gdf2:         gpd.GeoDataFrame,
    lisbon_gdf:   gpd.GeoDataFrame,
    lisbon_bounds: tuple,
) -> None:
    """
    Render and save one measurement distribution map figure.

    The layout is identical for all three sub-figures (all points, urban only,
    rural only): a wide left panel showing continental Portugal and a narrow
    right panel zooming into the Municipality of Lisbon.  Only the point
    colour and the figure title differ between calls.

    Parameters
    ----------
    pts           : DataFrame with float columns ['lat', 'lon'].
    point_color   : Hex colour string for the scatter markers.
    legend_label  : Text shown in the legend patch (e.g. 'Urban points').
    main_title    : Title placed above the left (Portugal) panel.
    output_file   : Full path to the output PNG file.
    gdf1          : GeoDataFrame of GADM level-1 district boundaries.
    gdf2          : GeoDataFrame of GADM level-2 municipality boundaries.
    lisbon_gdf    : GeoDataFrame of the Lisboa municipality polygon.
    lisbon_bounds : (minx, miny, maxx, maxy) bounding box of Lisbon.
    """
    if pts.empty:
        print(f"  [WARNING] No points to plot for '{legend_label}' — skipping figure.")
        return

    print(f"\n  Drawing figure: {output_file.name}  ({len(pts):,} points) ...")

    # Wide left panel (Portugal) + narrow right panel (Lisbon zoom)
    fig, (ax_pt, ax_ls) = plt.subplots(
        1, 2,
        figsize=(13, 11),
        gridspec_kw={'width_ratios': [2.2, 1]},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.04)

    # ==========================================================================
    # Left panel : Continental Portugal
    # ==========================================================================
    ax_pt.set_facecolor('white')
    for sp in ax_pt.spines.values():
        sp.set_visible(False)

    # District boundary lines
    gdf1.boundary.plot(ax=ax_pt, color='#444444', linewidth=0.8, zorder=2)

    # Measurement points.
    # .values converts to plain numpy float arrays, preventing the categorical-
    # axis ConversionError that occurs when pandas Series with object dtype
    # are passed directly to matplotlib scatter.
    ax_pt.scatter(
        pts['lon'].values,
        pts['lat'].values,
        c=point_color,
        s=POINT_SIZE_MAIN,
        alpha=POINT_ALPHA,
        linewidths=0,
        zorder=3,
        rasterized=True,   # reduces file size for dense point clouds
    )

    # Dashed rectangle marking the Lisboa zoom area on the main map
    rx = lisbon_bounds[0] - RECT_MARGIN
    ry = lisbon_bounds[1] - RECT_MARGIN
    rw = (lisbon_bounds[2] - lisbon_bounds[0]) + 2 * RECT_MARGIN
    rh = (lisbon_bounds[3] - lisbon_bounds[1]) + 2 * RECT_MARGIN

    rect = mpatches.Rectangle(
        (rx, ry), rw, rh,
        linewidth=1.2, edgecolor='#111111', facecolor='none',
        linestyle='--', zorder=5,
    )
    ax_pt.add_patch(rect)

    # Fixed map extent for continental Portugal.
    # set_aspect is intentionally omitted so xlim/ylim are fully respected
    # and all measurement points remain visible within the panel.
    ax_pt.set_xlim(-9.65, -6.05)
    ax_pt.set_ylim(36.75, 42.25)
    ax_pt.set_xticks([])
    ax_pt.set_yticks([])
    ax_pt.set_title(main_title, fontsize=11, fontweight='bold', pad=8)

    # ==========================================================================
    # Right panel : Municipality of Lisbon
    # ==========================================================================
    ax_ls.set_facecolor('white')
    for sp in ax_ls.spines.values():
        sp.set_linewidth(1.0)
        sp.set_edgecolor('#333333')

    zoom_margin = 0.04

    # --- Boundary layers in the Lisbon panel ----------------------------------
    # ZOOM_BORDERS = 'viewport' : draw all municipalities that intersect the
    #   viewport bounding box (includes Almada, Loures, Amadora, etc.).
    # ZOOM_BORDERS = 'lisbon_only' : draw only the Lisboa municipality outline.
    if ZOOM_BORDERS == 'viewport':
        viewport_box = shapely_box(
            lisbon_bounds[0] - zoom_margin,
            lisbon_bounds[1] - zoom_margin,
            lisbon_bounds[2] + zoom_margin,
            lisbon_bounds[3] + zoom_margin,
        )
        viewport_munis = gdf2[gdf2.geometry.intersects(viewport_box)]
        viewport_munis.boundary.plot(
            ax=ax_ls, color='#888888', linewidth=0.5, zorder=2
        )

    # Lisboa municipality boundary (thick outline on top in both modes)
    lisbon_gdf.boundary.plot(
        ax=ax_ls, color='#111111', linewidth=1.4, zorder=3
    )

    # --- Measurement points in the Lisbon panel -------------------------------
    # CLIP_LISBON_POINTS = True  : only points strictly inside the Lisboa
    #   polygon are shown (excludes neighbouring municipalities).
    # CLIP_LISBON_POINTS = False : all points in the viewport bounding box
    #   are shown regardless of which municipality they fall in.
    if CLIP_LISBON_POINTS:
        lisbon_polygon = lisbon_gdf.geometry.union_all()
        pts_gdf = gpd.GeoDataFrame(
            pts,
            geometry=[ShapelyPoint(lon, lat)
                      for lon, lat in zip(pts['lon'].values,
                                          pts['lat'].values)],
            crs='EPSG:4326',
        )
        pts_zoom = pts_gdf[pts_gdf.geometry.within(lisbon_polygon)]
        print(f"    Lisbon panel: {len(pts_zoom):,} points inside Lisboa "
              f"out of {len(pts):,} total.")
    else:
        pts_zoom = pts   # no spatial filter applied

    ax_ls.scatter(
        pts_zoom['lon'].values,
        pts_zoom['lat'].values,
        c=point_color,
        s=POINT_SIZE_ZOOM,
        alpha=POINT_ALPHA,
        linewidths=0,
        zorder=4,
        rasterized=True,
    )

    ax_ls.set_xlim(lisbon_bounds[0] - zoom_margin, lisbon_bounds[2] + zoom_margin)
    ax_ls.set_ylim(lisbon_bounds[1] - zoom_margin, lisbon_bounds[3] + zoom_margin)
    ax_ls.set_xticks([])
    ax_ls.set_yticks([])
    ax_ls.set_title('Municipality of Lisbon', fontsize=10, fontweight='bold', pad=6)

    # ==========================================================================
    # Connector lines from zoom rectangle to Lisbon panel corners
    # Coordinates are converted through the transform pipeline (data → display
    # → figure fraction) to avoid axis unit errors that arise when
    # ConnectionPatch handles geopandas-initialised axes.
    # ==========================================================================
    fig.canvas.draw()   # force layout so transforms are accurate

    def data_to_fig(ax, xd, yd):
        """Convert one data-space point to figure-fraction coordinates."""
        display = ax.transData.transform((xd, yd))
        return fig.transFigure.inverted().transform(display)

    bb_ls = ax_ls.get_position()   # Bbox of right panel in figure fraction

    top_rect = data_to_fig(ax_pt, rx + rw, ry + rh)
    bot_rect = data_to_fig(ax_pt, rx + rw, ry)
    top_ls   = (bb_ls.x0, bb_ls.y1)
    bot_ls   = (bb_ls.x0, bb_ls.y0)

    for start, end in [(top_rect, top_ls), (bot_rect, bot_ls)]:
        line = plt.Line2D(
            [start[0], end[0]], [start[1], end[1]],
            transform=fig.transFigure,
            color='#555555', linewidth=0.8, linestyle=':',
            zorder=10, clip_on=False,
        )
        fig.add_artist(line)

    # ==========================================================================
    # Legend
    # ==========================================================================
    legend_handle = mpatches.Patch(
        facecolor=point_color, edgecolor='none', label=legend_label
    )
    ax_pt.legend(
        handles=[legend_handle],
        loc='lower right',
        fontsize=9,
        frameon=True,
        framealpha=0.9,
    )

    # ==========================================================================
    # Save and display
    # ==========================================================================
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    print(f"    Figure saved -> {output_file}")

    plt.show()
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the sensor files and draw the all / urban / rural maps."""
    print("=" * 70)
    print("  Portugal Measurement Distribution Map")
    print("=" * 70)
    print(f'  CAMPAIGN            : {CAMPAIGN}')
    if CAMPAIGN == "standard":
        print(f'  FILE_SUFFIX         : "{FILE_SUFFIX}"')
    else:
        print(f'  TUKTUK_FILE_SUFFIX  : "{TUKTUK_FILE_SUFFIX}"')

    # ---- 1. Load sensor CSV files --------------------------------------------
    print(f"\n[1/4] Loading sensor CSV files ({_VARIANT_TAG}) ...")

    frames: list[pd.DataFrame] = []
    for label, path in SENSOR_FILES.items():
        df = load_sensor_data(path, label)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
            "FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    all_pts = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(all_pts):,} measurement points across all sensors.")

    # ---- 2. Load GADM shapefiles ---------------------------------------------
    print("\n[2/4] Loading GADM shapefiles ...")

    shp1_path = SHAPEFILE_DIR / 'gadm41_PRT_1.shp'   # districts
    shp2_path = SHAPEFILE_DIR / 'gadm41_PRT_2.shp'   # municipalities

    for p in (shp1_path, shp2_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Shapefile not found: {p}\n"
                "Download GADM Portugal shapefiles from https://gadm.org/ "
                "and update SHAPEFILE_DIR."
            )

    gdf1 = gpd.read_file(shp1_path)   # district boundaries
    gdf2 = gpd.read_file(shp2_path)   # municipality boundaries

    print(f"  District shapefile    : {len(gdf1)} features  (CRS: {gdf1.crs})")
    print(f"  Municipality shapefile: {len(gdf2)} features  (CRS: {gdf2.crs})")

    # ---- 3. Isolate Lisboa municipality --------------------------------------
    print(f"\n[3/4] Extracting '{LISBON_NAME}' municipality boundary ...")

    name_col = 'NAME_2' if 'NAME_2' in gdf2.columns else gdf2.columns[2]
    lisbon_gdf = gdf2[gdf2[name_col] == LISBON_NAME].copy()
    if lisbon_gdf.empty:
        lisbon_gdf = gdf2[gdf2[name_col].str.lower() == LISBON_NAME.lower()].copy()
    if lisbon_gdf.empty:
        raise ValueError(
            f"Municipality '{LISBON_NAME}' not found in column '{name_col}'.\n"
            f"Available (first 30): {sorted(gdf2[name_col].unique())[:30]}\n"
            "Adjust LISBON_NAME in the configuration section."
        )

    lisbon_bounds = lisbon_gdf.total_bounds   # (minx, miny, maxx, maxy)
    print(f"  Found {len(lisbon_gdf)} polygon(s).  "
          f"Bounds: {lisbon_bounds.round(4)}")

    # Shared keyword arguments passed to every _draw_and_save_map() call
    map_kwargs = dict(
        gdf1=gdf1,
        gdf2=gdf2,
        lisbon_gdf=lisbon_gdf,
        lisbon_bounds=lisbon_bounds,
    )

    # ---- 4. Apply optional CLC class filter ------------------------------------
    # Validate FILTER_LAND_USE entries and build a descriptive tag
    if FILTER_LAND_USE:
        invalid = [lu for lu in FILTER_LAND_USE if lu not in CLC_MAPPING]
        if invalid:
            raise ValueError(
                f"FILTER_LAND_USE contains invalid land_use value(s): {invalid}\n"
                f"Valid range is 1–44.  Check clc_mapping.py for the full list."
            )
        # Build a short tag for filenames: "lu4" or "lu1_2" etc.
        lu_tag = 'lu' + '_'.join(str(lu) for lu in sorted(FILTER_LAND_USE))
        # Build a human-readable label for titles and legends
        if len(FILTER_LAND_USE) == 1:
            lu_label = CLC_LVL3_LABEL[FILTER_LAND_USE[0]]
        else:
            lu_label = ', '.join(CLC_LVL3_LABEL[lu] for lu in FILTER_LAND_USE)
        all_pts = all_pts[all_pts['land_use_clcv'].isin(FILTER_LAND_USE)].reset_index(drop=True)
        print(f"\n  CLC class filter active: {lu_label}")
        print(f"  Filtered to {len(all_pts):,} points  (land_use_clcv ∈ {FILTER_LAND_USE})")
        if all_pts.empty:
            print("  [WARNING] No points remain after CLC filter — check FILTER_LAND_USE.")
            return
    else:
        lu_tag   = None
        lu_label = None

    def _suffixed(base: str) -> str:
        """Append the CLC filter tag to a filename suffix when active."""
        return base + f'_{lu_tag}' if lu_tag else base

    # ---- 5. Draw the figures --------------------------------------------------
    print("\n[4/4] Drawing figures ...")

    # ── (A) All measurement points  (or the CLC-filtered subset) ─────────────
    output_all = OUTPUT_DIR / _build_output_filename(
        'measurement_distribution_map', CAMPAIGN, _suffixed(_VARIANT_TAG), 'png'
    )
    title_all   = f'Continental Portugal — {lu_label}' if lu_label else 'Continental Portugal'
    legend_all  = lu_label if lu_label else 'Measurement points'
    _draw_and_save_map(
        pts=all_pts,
        point_color=POINT_COLOR,
        legend_label=legend_all,
        main_title=title_all,
        output_file=output_all,
        **map_kwargs,
    )

    # When a CLC class filter is active, Urban/Rural sub-figures are skipped:
    # the filtered set may already contain only one group, and mixing a class
    # filter with an Urban/Rural split adds little information.
    if lu_tag:
        print("\n  CLC class filter active — Urban/Rural sub-figures skipped.")
        print("\nDone.")
        return

    # ── (B) Urban points only ─────────────────────────────────────────────────
    urban_pts = all_pts[all_pts['clc_group'] == 'Urban'].reset_index(drop=True)
    print(f"\n  Urban subset: {len(urban_pts):,} points "
          f"({100 * len(urban_pts) / max(len(all_pts), 1):.1f}% of total)")

    output_urban = OUTPUT_DIR / _build_output_filename(
        'measurement_distribution_map', CAMPAIGN, _VARIANT_TAG + '_urban', 'png'
    )
    _draw_and_save_map(
        pts=urban_pts,
        point_color=POINT_COLOR_URBAN,
        legend_label='Urban points (CLC)',
        main_title='Continental Portugal — Urban areas',
        output_file=output_urban,
        **map_kwargs,
    )

    # ── (C) Rural points only ─────────────────────────────────────────────────
    rural_pts = all_pts[all_pts['clc_group'] == 'Rural'].reset_index(drop=True)
    print(f"\n  Rural subset: {len(rural_pts):,} points "
          f"({100 * len(rural_pts) / max(len(all_pts), 1):.1f}% of total)")

    output_rural = OUTPUT_DIR / _build_output_filename(
        'measurement_distribution_map', CAMPAIGN, _VARIANT_TAG + '_rural', 'png'
    )
    _draw_and_save_map(
        pts=rural_pts,
        point_color=POINT_COLOR_RURAL,
        legend_label='Rural points (CLC)',
        main_title='Continental Portugal — Rural areas',
        output_file=output_rural,
        **map_kwargs,
    )

    print("\nDone.")


# ==============================================================================
if __name__ == '__main__':
    main()