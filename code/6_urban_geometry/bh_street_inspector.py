"""
================================================================================
Building-Height (BH) Join — Single-Sensor Street Inspector
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Interactive diagnostic map of the building-height (BH) join, for ONE sensor
at a time. The standalone HTML map (no server needed) shows:
  - the GPS measurement points of the selected sensor, coloured by
    n_buildings_nearby;
  - the building centroids (GlobalBuildingAtlas) of the loaded area;
  - on click on a GPS point, a popup with its BH columns and the
    BUFFER_RADIUS_M buffer circle used by the join.
The area can be restricted to a bounding box (USE_BBOX / STREET_BBOX), and
the numbers of points and buildings drawn are capped (MAX_GPS_POINTS,
MAX_BUILDINGS) to keep the HTML file manageable.

Controls in the browser
-----------------------
  - Click a coloured point -> popup with all BH columns + buffer drawn
  - Click a blue dot        -> popup with that building's height, var, floors
  - Layer panel (top-right) -> toggle GPS points / buildings / buffer
  - Search box (top-left)   -> OpenStreetMap geocoder (type a street name)
Internet access is needed in the browser only, for the basemap tiles.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py output (or a filter / time-range subset) of
             SENSOR_ID, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon', BH join columns
    input  : BUILDINGS_PATH — GlobalBuildingAtlas (Zhu et al., 2025, ESSD),
             Lisbon metropolitan area clip, building centroids with heights
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             bh_street_inspector_<sensor>_<CAMPAIGN>_<variant_tag>.html

Requirements
------------
    pip install folium geopandas pandas

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.3
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BUILDINGS_PATH, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import folium
import geopandas as gpd
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Basemap selector ---------------------------------------------------------
BASEMAP: str = "osm"   # "osm" | "clean"
if BASEMAP not in ("osm", "clean"):
    raise ValueError(f"BASEMAP must be 'osm' or 'clean', got '{BASEMAP}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned_lisbon"   # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Sensor to inspect (one at a time) ----------------------------------------
SENSOR_ID: str = "j31"

# Path to the building centroids GeoJSON.
BUILDINGS_FILE: Path = BUILDINGS_PATH

# Buffer radius used during the join (for display only — no recomputation here).
# Must match the value used by join_bh_standard.py / join_bh_tuktuk.py (bh_mapping.BUFFER_RADIUS_M).
BUFFER_RADIUS_M: float = 50.0

# --- Output -------------------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "bh_inspector_outputs"

# Open the generated HTML automatically in the default browser.
AUTO_OPEN_BROWSER: bool = True

# --- Optional spatial filter --------------------------------------------------
# Set USE_BBOX = True and define STREET_BBOX to restrict both GPS points and
# building centroids to a small area (e.g. one street or neighbourhood).
# Format: (lon_min, lat_min, lon_max, lat_max) in WGS-84 degrees.
# Set USE_BBOX = False to load the full dataset (slower, larger HTML file).
USE_BBOX: bool = True
# STREET_BBOX: tuple[float, float, float, float] = (
#     -9.145, 38.715, -9.130, 38.725  # example: Avenida da Liberdade area
# )
STREET_BBOX: tuple[float, float, float, float] = (
    -9.156641, 38.719682, -9.124891, 38.74479  # Arroios
)
# BBOX TUKTUK AVENIDADE
# STREET_BBOX: tuple[float, float, float, float] = (
#     -9.15835, 38.712703, -9.137917, 38.731439
# )
# BBOX TUKTUK BAIXA
# STREET_BBOX: tuple[float, float, float, float] = (
#     -9.148627, 38.703601, -9.121926, 38.716718
# )
# BBOX TUKTUK BELEM-ALCANTARA
# STREET_BBOX: tuple[float, float, float, float] = (
#     -9.220309, 38.689481, -9.15815, 38.707286
# )

# Max GPS points to render (sub-sampled randomly if exceeded, to keep the
# HTML file manageable).  Set to None to render all points.
MAX_GPS_POINTS: int | None = 5_000

# Max building centroids to render.  Set to None for all.
MAX_BUILDINGS: int | None = 10_000


# ==============================================================================
# FILENAME / COLUMN RESOLUTION
# ==============================================================================
# Input path, GPS columns and output path follow from CAMPAIGN, SENSOR_ID and
# the input suffix.

SENSORS: list[str] = ["j30", "j31", "j32", "j33"]


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    if CAMPAIGN == "standard":
        return FILE_SUFFIX
    elif CAMPAIGN == "tuktuk":
        return TUKTUK_FILE_SUFFIX
    else:
        raise ValueError(f"Unknown CAMPAIGN '{CAMPAIGN}'.")


def _build_input_path(campaign: str, sensor_id: str, suffix: str) -> Path:
    """
    Build the input CSV path for one sensor, following the pipeline's
    per-campaign naming convention.

    Standard campaign : <DATA_DIR>/<sensor>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <TUKTUK_DATA_DIR>/<sensor>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    if campaign == "standard":
        if sensor_id not in SENSORS:
            raise ValueError(
                f"SENSOR_ID '{sensor_id}' is not valid for CAMPAIGN='standard'. "
                f"Expected one of {SENSORS}."
            )
        return DATA_DIR / f"{sensor_id}_satellite_landuse_v2_{suffix}.csv"
    elif campaign == "tuktuk":
        if sensor_id not in SENSORS:
            raise ValueError(
                f"SENSOR_ID '{sensor_id}' is not valid for CAMPAIGN='tuktuk'. "
                f"Expected one of {SENSORS}."
            )
        return TUKTUK_DATA_DIR / f"{sensor_id}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _get_latlon_cols(campaign: str, sensor_id: str) -> tuple[str, str]:
    """
    Return the (lat_col, lon_col) names for the current campaign.

    Both campaigns : sensor-prefixed, e.g. "J31 lat" / "J31 lon".
    """
    if campaign in ("standard", "tuktuk"):
        return f"{sensor_id.upper()} lat", f"{sensor_id.upper()} lon"
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _build_output_path(base_dir: Path, sensor_id: str, campaign: str, variant_tag: str) -> Path:
    """Build the output HTML path, embedding CAMPAIGN + sensor + variant tag."""
    out_dir = base_dir / campaign / variant_tag
    return out_dir / f"bh_street_inspector_{sensor_id}_{campaign}_{variant_tag}.html"


# Resolved at startup from the configuration above.
_VARIANT_TAG: str = _variant_tag()
INPUT_PATH: Path = _build_input_path(CAMPAIGN, SENSOR_ID, _VARIANT_TAG)
LAT_COL, LON_COL = _get_latlon_cols(CAMPAIGN, SENSOR_ID)
OUTPUT_HTML: Path = _build_output_path(OUTPUT_BASE_DIR, SENSOR_ID, CAMPAIGN, _VARIANT_TAG)
OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# COLOUR HELPERS
# ==============================================================================

def _n_buildings_colour(n: float) -> str:
    """Map n_buildings_nearby to a colour for the GPS point circle marker."""
    if pd.isna(n):
        return "#888888"   # grey   — GPS invalid / not computed
    n = int(n)
    if n == 0:
        return "#E74C3C"   # red    — open space, no buildings found
    if n <= 3:
        return "#F39C12"   # amber  — sparse
    if n <= 8:
        return "#27AE60"   # green  — typical urban
    return "#8E44AD"       # purple — dense (many buildings in buffer)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_gps(path: Path, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Load the sensor CSV and keep the rows with valid GPS coordinates."""
    if not path.exists():
        raise FileNotFoundError(
            f"GPS file not found:\n  {path}\n"
            f"Check CAMPAIGN ({CAMPAIGN!r}), SENSOR_ID ({SENSOR_ID!r}) and "
            f"FILE_SUFFIX/TUKTUK_FILE_SUFFIX in the CONFIGURATION block."
        )
    df = pd.read_csv(path, low_memory=False)
    for col in (lat_col, lon_col):
        if col not in df.columns:
            raise KeyError(
                f"Expected GPS column '{col}' not found in {path.name}.\n"
                f"Available columns: {df.columns.tolist()}"
            )
    # Drop rows without GPS
    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
    print(f"  GPS rows (valid coords): {len(df):,}")
    return df


def load_buildings(bbox: tuple | None) -> gpd.GeoDataFrame:
    """Load the building centroids, restricted to *bbox* when given."""
    if not BUILDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Building file not found:\n  {BUILDINGS_FILE}\n"
            "Check BUILDINGS_PATH in config.py."
        )
    gdf = gpd.read_file(BUILDINGS_FILE, bbox=bbox) if bbox else \
          gpd.read_file(BUILDINGS_FILE)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    print(f"  Building centroids loaded: {len(gdf):,}")
    return gdf


# ==============================================================================
# OPTIONAL BBOX FILTER FOR GPS POINTS
# ==============================================================================

def filter_by_bbox(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Keep the GPS points inside *bbox* (lon_min, lat_min, lon_max, lat_max)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    mask = (
        (df[lon_col] >= lon_min) & (df[lon_col] <= lon_max) &
        (df[lat_col] >= lat_min) & (df[lat_col] <= lat_max)
    )
    return df[mask].reset_index(drop=True)


# ==============================================================================
# MAP BUILDING
# ==============================================================================

def build_map(df: pd.DataFrame, buildings: gpd.GeoDataFrame,
               lat_col: str, lon_col: str) -> folium.Map:
    """Build and return the Folium map object."""

    # ── Centre the map on the data ────────────────────────────────────────────
    centre_lat = df[lat_col].median()
    centre_lon = df[lon_col].median()
    if BASEMAP == "osm":
        tiles = "OpenStreetMap"
    elif BASEMAP == "clean":
        tiles = "CartoDB positron"
    else:
        raise ValueError(f"Unknown BASEMAP '{BASEMAP}'.")
    fmap = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=15,
        tiles=tiles,
    )

    # ── Layer groups (toggled from the layer panel) ───────────────────────────
    gps_layer  = folium.FeatureGroup(name=f"GPS points ({SENSOR_ID.upper()}, {CAMPAIGN})", show=True)
    bldg_layer = folium.FeatureGroup(name="Building centroids (GBA)", show=True)
    buf_layer  = folium.FeatureGroup(name="50 m buffers (on click)", show=True)

    # ── GPS points ────────────────────────────────────────────────────────────
    # Sub-sample if needed to keep HTML size manageable
    df_plot = df if MAX_GPS_POINTS is None else df.sample(
        n=min(MAX_GPS_POINTS, len(df)), random_state=42
    ).reset_index(drop=True)
    print(f"  Rendering {len(df_plot):,} GPS points ...")

    # Identify BH columns present
    bh_cols = [c for c in
               ["mean_building_height_m", "max_building_height_m", "n_buildings_nearby"]
               if c in df.columns]

    for _, row in df_plot.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        n_bldg = row.get("n_buildings_nearby", float("nan"))
        colour = _n_buildings_colour(n_bldg)

        # Build popup content
        popup_lines = [
            f"<b>Sensor:</b> {SENSOR_ID.upper()} ({CAMPAIGN})<br>",
            f"<b>Coords:</b> {lat:.6f}, {lon:.6f}<br>",
            "<hr style='margin:4px 0'>",
        ]
        for col in bh_cols:
            val = row.get(col)
            label = col.replace("_", " ")
            popup_lines.append(
                f"<b>{label}:</b> "
                f"{'NaN' if pd.isna(val) else f'{val:.1f}'}<br>"
            )
        # Add a few other useful columns if present
        for extra in ["timestamp", "ghi", "ghi_clear", "land_use_str_clcv", "cos23_n4_label"]:
            if extra in row.index and pd.notna(row[extra]):
                if extra in ("ghi", "ghi_clear"):
                    popup_lines.append(f"<b>{extra}:</b> {row[extra]:.2f} W/m²<br>")
                else:
                    popup_lines.append(f"<b>{extra}:</b> {row[extra]}<br>")

        popup_html = "".join(popup_lines)

        mean_h = row.get("mean_building_height_m", float("nan"))
        tooltip = (
            f"n_bldg={int(n_bldg) if pd.notna(n_bldg) else 'NaN'}  "
            f"mean={mean_h:.1f} m" if pd.notna(mean_h) else "no buildings nearby"
        )

        # The circle marker (GPS point)
        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            popup=folium.Popup(
                folium.IFrame(popup_html, width=280, height=180),
                max_width=300,
            ),
            tooltip=tooltip,
        )
        gps_layer.add_child(marker)

        # Buffer circle (drawn as a separate layer so it can be toggled)
        folium.Circle(
            location=[lat, lon],
            radius=BUFFER_RADIUS_M,
            color=colour,
            fill=False,
            weight=1,
            opacity=0.3,
        ).add_to(buf_layer)

    # ── Building centroids ────────────────────────────────────────────────────
    bldg_plot = buildings if MAX_BUILDINGS is None else buildings.sample(
        n=min(MAX_BUILDINGS, len(buildings)), random_state=42
    )
    print(f"  Rendering {len(bldg_plot):,} building centroids ...")

    for _, row in bldg_plot.iterrows():
        geom = row.geometry
        blat, blon = geom.y, geom.x

        h    = row.get("height", float("nan"))
        var_ = row.get("var",    float("nan"))
        flrs = row.get("est_floors", float("nan"))
        src  = row.get("source", "?")
        bid  = row.get("id", "?")

        popup_html = (
            f"<b>ID:</b> {bid}<br>"
            f"<b>Source:</b> {src}<br>"
            f"<b>Height:</b> {h:.2f} m<br>"
            f"<b>Variance:</b> {var_:.3f}<br>"
            f"<b>Est. floors:</b> {int(flrs) if pd.notna(flrs) else '?'}<br>"
            f"<b>Footprint:</b> {row.get('footprint_m2', '?')} m²<br>"
        )

        folium.CircleMarker(
            location=[blat, blon],
            radius=3,
            color="#1565C0",
            fill=True,
            fill_color="#1565C0",
            fill_opacity=0.5,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"H={h:.1f} m | src={src}",
        ).add_to(bldg_layer)

    # ── Assemble layers ───────────────────────────────────────────────────────
    fmap.add_child(bldg_layer)
    fmap.add_child(gps_layer)
    fmap.add_child(buf_layer)
    folium.LayerControl(collapsed=False).add_to(fmap)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        border: 1px solid #ccc; font-family: sans-serif; font-size: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,.2);">
      <b>GPS points — n_buildings_nearby</b><br>
      <span style="color:#888">●</span> NaN (GPS invalid)<br>
      <span style="color:#E74C3C">●</span> 0 — open space / gap<br>
      <span style="color:#F39C12">●</span> 1–3 — sparse<br>
      <span style="color:#27AE60">●</span> 4–8 — typical urban<br>
      <span style="color:#8E44AD">●</span> 9+ — dense<br>
      <hr style="margin:6px 0">
      <span style="color:#1565C0">●</span> Building centroids (GBA)<br>
      <span style="color:#999">○</span> 50 m buffer radius
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load GPS points and buildings, build the map and save it as HTML."""
    print("═" * 68)
    print("  BH STREET INSPECTOR — building-height join diagnostic")
    print(f"  Campaign      : {CAMPAIGN}")
    print(f"  Sensor        : {SENSOR_ID.upper()}")
    print(f"  Variant tag   : {_VARIANT_TAG!r}")
    print(f"  Input file    : {INPUT_PATH}")
    print(f"  GPS columns   : {LAT_COL!r} / {LON_COL!r}")
    print(f"  Buffer radius : {BUFFER_RADIUS_M} m")
    print(f"  Basemap       : {BASEMAP}")
    print(f"  Spatial filter: {'YES — bbox ' + str(STREET_BBOX) if USE_BBOX else 'NO — full dataset'}")
    print("═" * 68)

    if CAMPAIGN == "standard" and SENSOR_ID in ("j32", "j33"):
        print(
            f"\n  [NOTE] {SENSOR_ID.upper()} has never had data in the standard "
            "campaign — this is expected. Nothing to inspect here; pick j30 "
            "or j31, or switch CAMPAIGN to 'tuktuk'.\n"
        )

    bbox = STREET_BBOX if USE_BBOX else None

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/3] Loading GPS data ...")
    df = load_gps(INPUT_PATH, LAT_COL, LON_COL)
    if USE_BBOX:
        df = filter_by_bbox(df, LAT_COL, LON_COL, STREET_BBOX)
        print(f"  After bbox filter: {len(df):,} rows")

    if df.empty:
        print("  [WARN] No GPS rows to display — stopping here.")
        return

    print("\n[2/3] Loading building centroids ...")
    buildings = load_buildings(bbox)

    # ── Build map ─────────────────────────────────────────────────────────────
    print("\n[3/3] Building map ...")
    fmap = build_map(df, buildings, LAT_COL, LON_COL)

    # ── Save ──────────────────────────────────────────────────────────────────
    fmap.save(str(OUTPUT_HTML))
    print(f"\n  ✓ Saved → {OUTPUT_HTML}")
    if AUTO_OPEN_BROWSER:
        import webbrowser
        webbrowser.open(OUTPUT_HTML.as_uri())
    else:
        print("  Open this file in your browser to inspect the data "
              "(set AUTO_OPEN_BROWSER = True to do this automatically).")
    print("═" * 68)


if __name__ == "__main__":
    main()