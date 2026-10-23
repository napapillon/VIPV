"""
================================================================================
Building-Height (BH) Join — All-Sensor Diagnostic Map
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Interactive map to check visually the building-height join
(GlobalBuildingAtlas) on the GPS measurement points of ALL sensors of the
selected campaign. The self-contained HTML map shows:
  - the GPS measurement points (all sensors), coloured by
    mean_building_height_m;
  - the building centroids inside the area;
  - optionally, the BUFFER_RADIUS_M buffer circle around each GPS point;
  - a popup on each marker with all BH columns, coordinates and sensor id.
The join is not re-run: the script only reads the joined sensor files.
Sensors whose file is missing are skipped.

Filtering options
-----------------
    AREA_BBOX      : (lat_min, lon_min, lat_max, lon_max) box; only data
                     inside it is drawn. Main parameter to change when
                     inspecting a specific street. None = full dataset (slow).
    FOCUS_ZOOM     : initial zoom level (17 = street level, 14 = district).
    SHOW_BUFFERS   : draw the buffer circle around each GPS point.
    BUFFER_RADIUS_M: must match the radius used by the join (bh_mapping).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon', BH join columns
    input  : BUILDINGS_PATH — GlobalBuildingAtlas building centroids
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             bh_diagnostic_map_<CAMPAIGN>_<variant_tag>.html

Requirements
------------
    pip install folium pandas

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.1
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import webbrowser
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BUILDINGS_PATH, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import folium
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
FILE_SUFFIX: str = "sd_cleaned_lisbon"   # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# Path to the raw building GeoJSON (to overlay building centroids).
BUILDINGS_GEOJSON: Path = BUILDINGS_PATH

# --- Output -------------------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "bh_diagnostic_outputs"

# Open the generated HTML automatically in the default browser.
AUTO_OPEN_BROWSER: bool = True

# --- Area of interest (key parameter) -----------------------------------------
# Bounding box (lat_min, lon_min, lat_max, lon_max) that controls WHAT IS DRAWN.
# Only GPS points AND building centroids within this box are loaded.
# Change this to inspect a specific street — nothing outside the box is drawn.
#
# How to get coordinates for a street:
#   Open Google Maps → right-click any point on the street → copy lat/lon.
#   Define a small rectangle around the block you want to check (~200-500 m side).
#
# Set to None to load the full dataset (slow, produces a cluttered view).
AREA_BBOX: tuple[float, float, float, float] | None = (38.70, -9.16, 38.72, -9.134)
# (lat_min, lon_min, lat_max, lon_max)
FOCUS_ZOOM: int = 14
# Other examples:
# AREA_BBOX = (38.710, -9.135, 38.718, -9.126)   # Alfama / Mouraria
# AREA_BBOX = (38.714, -9.148, 38.722, -9.138)   # Avenida da Liberdade area
# AREA_BBOX = None                                 # full dataset (slow)

# Used only when AREA_BBOX is None, to centre the map.
FOCUS_LAT: float = 38.7223
FOCUS_LON: float = -9.1393

# --- Buffer circles -----------------------------------------------------------
# Draw the join radius around each GPS point (useful on a small AREA_BBOX).
SHOW_BUFFERS: bool = True

# Must match the value used by join_bh_standard.py / join_bh_tuktuk.py
# (bh_mapping.BUFFER_RADIUS_M).
BUFFER_RADIUS_M: float = 50.0


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# Input and output paths follow from CAMPAIGN and the input suffix.

STANDARD_SENSORS: list[str] = ["j30", "j31", "j32", "j33"]
TUKTUK_SENSORS: list[str] = ["j29", "j30", "j31", "j32", "j33"]


def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor names to their CSV input paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    if campaign == "standard":
        return {
            s: data_dir / f"{s}_satellite_landuse_v2_{suffix}.csv"
            for s in STANDARD_SENSORS
        }
    elif campaign == "tuktuk":
        return {
            s: tuktuk_data_dir / f"{s}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
            for s in TUKTUK_SENSORS
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _get_latlon_cols(campaign: str, sensor_id: str) -> tuple[str, str]:
    """
    Return the (lat_col, lon_col) names for a sensor.

    Both campaigns use sensor-prefixed columns ("J31 lat" / "J31 lon"):
    normalize_tuktuk.py gives the tuk-tuk files the standard schema.
    """
    if campaign in ("standard", "tuktuk"):
        return f"{sensor_id.upper()} lat", f"{sensor_id.upper()} lon"
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    if CAMPAIGN == "standard":
        return FILE_SUFFIX
    elif CAMPAIGN == "tuktuk":
        return TUKTUK_FILE_SUFFIX
    else:
        raise ValueError(f"Unknown CAMPAIGN '{CAMPAIGN}'.")


def _build_output_path(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Build the output HTML path, embedding CAMPAIGN + variant tag."""
    out_dir = base_dir / campaign / variant_tag
    return out_dir / f"bh_diagnostic_map_{campaign}_{variant_tag}.html"


# Resolved at startup from the configuration above.
_VARIANT_TAG: str = _variant_tag()
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
)
OUTPUT_HTML: Path = _build_output_path(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# COLOUR SCALE  (GPS points coloured by mean_building_height_m)
# ==============================================================================

def height_to_colour(h: float | None) -> str:
    """Map mean building height [m] to a hex colour for the GPS marker."""
    if h is None or pd.isna(h):
        return "#808080"   # grey = no building in buffer
    if h < 5:
        return "#2166ac"   # deep blue  — very low
    if h < 10:
        return "#74add1"   # light blue
    if h < 15:
        return "#abd9e9"
    if h < 20:
        return "#fee090"   # yellow
    if h < 30:
        return "#f46d43"   # orange
    return "#d73027"       # red — tall (>30 m)


# ==============================================================================
# PER-SENSOR DATA LOADING
# ==============================================================================

def load_sensor_data(filepath: Path, sensor_id: str, campaign: str) -> pd.DataFrame | None:
    """
    Load one sensor's CSV file and return a DataFrame with unified 'lat'/'lon'
    columns plus the BH columns and a 'sensor' identifier column.

    Returns None if the file is missing or has no valid GPS rows (this is
    the expected outcome for j32/j33 in the standard campaign).
    """
    if not filepath.is_file():
        print(f"  [WARNING] File not found: {filepath.name} — skipped "
              f"(expected for j32/j33 in the standard campaign).")
        return None

    lat_col, lon_col = _get_latlon_cols(campaign, sensor_id)
    df = pd.read_csv(filepath, low_memory=False)

    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"  [WARNING] Columns '{lat_col}'/'{lon_col}' not found in "
              f"{filepath.name} — skipped.")
        return None

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    df = df.dropna(subset=[lat_col, lon_col])
    df = df[df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)]
    df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]

    if df.empty:
        print(f"  [WARNING] No valid GPS points in {filepath.name} — skipped.")
        return None

    # Unify lat/lon column names across sensors/campaigns
    keep_cols = [lat_col, lon_col] + [
        c for c in ["mean_building_height_m", "max_building_height_m",
                     "n_buildings_nearby", "land_use_clcv", "ghi", "ghi_clear"]
        if c in df.columns
    ]
    out = df[keep_cols].rename(columns={lat_col: "_lat", lon_col: "_lon"})
    out["sensor"] = sensor_id

    print(f"  {sensor_id}: {len(out):,} valid GPS points loaded "
          f"from '{filepath.name}'")
    return out.reset_index(drop=True)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load all sensor files and buildings in AREA_BBOX, build the map, save it as HTML."""
    print("=" * 68)
    print("  BH Diagnostic Map — building-height join verification")
    print(f"  Campaign     : {CAMPAIGN}")
    print(f"  Variant tag  : {_VARIANT_TAG!r}")
    print("=" * 68)

    # ── Load all sensor CSV files ──────────────────────────────────────────────
    print("\n[1/4] Loading sensor CSV files ...")
    frames: list[pd.DataFrame] = []
    for sensor_id, path in SENSOR_FILES.items():
        df_sensor = load_sensor_data(path, sensor_id, CAMPAIGN)
        if df_sensor is not None:
            frames.append(df_sensor)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. Check CAMPAIGN, "
            "DATA_DIR/TUKTUK_DATA_DIR, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    df = pd.concat(frames, ignore_index=True)
    lat_col, lon_col = "_lat", "_lon"
    print(f"\n  Total: {len(df):,} valid GPS points across all sensors.")

    # Geographic filter: keep only points inside AREA_BBOX
    if AREA_BBOX is not None:
        lat_min, lon_min, lat_max, lon_max = AREA_BBOX
        df = df[
            df[lat_col].between(lat_min, lat_max) &
            df[lon_col].between(lon_min, lon_max)
        ]
        print(f"  {len(df):,} rows inside AREA_BBOX {AREA_BBOX}")
        if df.empty:
            print("  [WARN] No GPS points found in this bounding box.")
            print("         Check AREA_BBOX coordinates (lat_min, lon_min, lat_max, lon_max).")
            return
    else:
        lat_min, lon_min = df[lat_col].min(), df[lon_col].min()
        lat_max, lon_max = df[lat_col].max(), df[lon_col].max()

    # ── Load building centroids ────────────────────────────────────────────────
    buildings_df: pd.DataFrame | None = None
    if BUILDINGS_GEOJSON.exists():
        print("\n[2/4] Loading building centroids from GeoJSON ...")
        import json
        with open(BUILDINGS_GEOJSON, encoding="utf-8") as f:
            geojson = json.load(f)

        # Filter buildings to the same bounding box (+ buffer margin)
        margin = BUFFER_RADIUS_M / 111_000  # ~0.00045° per metre at Lisbon latitude
        rows = []
        for feat in geojson.get("features", []):
            geom = feat.get("geometry", {})
            if geom.get("type") != "Point":
                continue
            blon, blat = geom["coordinates"]
            if not (lat_min - margin <= blat <= lat_max + margin
                    and lon_min - margin <= blon <= lon_max + margin):
                continue
            props = feat.get("properties", {})
            rows.append({
                "blat": blat,
                "blon": blon,
                "height": props.get("height"),
                "footprint_m2": props.get("footprint_m2"),
                "id": props.get("id"),
                "source": props.get("source"),
            })

        buildings_df = pd.DataFrame(rows)
        print(f"  {len(buildings_df):,} building centroids loaded in this area")
    else:
        print("\n[2/4] Building GeoJSON not found — centroids will not be shown.")
        print(f"  (path checked: {BUILDINGS_GEOJSON})")

    # ── Build the map ──────────────────────────────────────────────────────────
    print("\n[3/4] Building interactive map ...")

    # Auto-centre on AREA_BBOX if set, otherwise use FOCUS_LAT/LON
    if AREA_BBOX is not None:
        map_lat = (AREA_BBOX[0] + AREA_BBOX[2]) / 2
        map_lon = (AREA_BBOX[1] + AREA_BBOX[3]) / 2
        map_zoom = FOCUS_ZOOM
    else:
        map_lat, map_lon, map_zoom = FOCUS_LAT, FOCUS_LON, 14

    m = folium.Map(
        location=[map_lat, map_lon],
        zoom_start=map_zoom,
        tiles="CartoDB positron",
    )

    # Layer: building centroids (small grey triangles)
    if buildings_df is not None and not buildings_df.empty:
        bldg_layer = folium.FeatureGroup(name="Building centroids (GeoJSON)", show=True)
        for _, row in buildings_df.iterrows():
            h = row["height"]
            h_str = f"{h:.1f} m" if pd.notna(h) else "N/A"
            popup_html = (
                f"<b>Building centroid</b><br>"
                f"Height: <b>{h_str}</b><br>"
                f"Footprint: {row['footprint_m2']} m²<br>"
                f"ID: {row['id']}<br>"
                f"Source: {row['source']}<br>"
                f"Lat: {row['blat']:.6f} | Lon: {row['blon']:.6f}"
            )
            folium.RegularPolygonMarker(
                location=[row["blat"], row["blon"]],
                number_of_sides=3,
                radius=5,
                color="#444444",
                fill=True,
                fill_color="#888888",
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"Bldg {h_str}",
            ).add_to(bldg_layer)
        bldg_layer.add_to(m)

    # Layer: GPS measurement points coloured by mean_building_height_m
    gps_layer = folium.FeatureGroup(name=f"GPS points ({CAMPAIGN}, coloured by mean BH)", show=True)

    bh_col_present = "mean_building_height_m" in df.columns

    for _, row in df.iterrows():
        lat_val = row[lat_col]
        lon_val = row[lon_col]
        mean_h  = row.get("mean_building_height_m") if bh_col_present else None
        max_h   = row.get("max_building_height_m")  if "max_building_height_m"  in df.columns else None
        n_bldg  = row.get("n_buildings_nearby")     if "n_buildings_nearby"     in df.columns else None

        colour = height_to_colour(mean_h)

        mean_h_str = f"{mean_h:.1f} m" if pd.notna(mean_h) else "no building in buffer"
        max_h_str  = f"{max_h:.1f} m"  if (max_h  is not None and pd.notna(max_h))  else "—"
        n_str      = str(int(n_bldg))  if (n_bldg is not None and pd.notna(n_bldg)) else "—"

        popup_html = (
            f"<b>GPS measurement point</b><br>"
            f"Sensor: <b>{row['sensor'].upper()}</b> ({CAMPAIGN})<br>"
            f"Lat: {lat_val:.6f} | Lon: {lon_val:.6f}<br>"
            f"<br>"
            f"mean_building_height_m : <b>{mean_h_str}</b><br>"
            f"max_building_height_m  : {max_h_str}<br>"
            f"n_buildings_nearby     : {n_str}<br>"
        )

        # Add any extra columns that exist in the CSV
        for extra_col in ["land_use_clcv", "ghi", "ghi_clear"]:
            if extra_col in df.columns and pd.notna(row.get(extra_col)):
                popup_html += f"{extra_col}: {row[extra_col]}<br>"

        folium.CircleMarker(
            location=[lat_val, lon_val],
            radius=5,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{row['sensor'].upper()} — mean BH: {mean_h_str}",
        ).add_to(gps_layer)

        # Optional buffer circle
        if SHOW_BUFFERS:
            folium.Circle(
                location=[lat_val, lon_val],
                radius=BUFFER_RADIUS_M,
                color=colour,
                fill=False,
                weight=1,
                opacity=0.4,
            ).add_to(gps_layer)

    gps_layer.add_to(m)

    # Legend (HTML injection)
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 12px 16px; border-radius: 8px;
                border: 1px solid #ccc; font-family: sans-serif; font-size: 12px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
      <b>mean_building_height_m</b><br>
      <span style="color:#808080">■</span> NaN (no building in buffer)<br>
      <span style="color:#2166ac">■</span> &lt; 5 m<br>
      <span style="color:#74add1">■</span> 5–10 m<br>
      <span style="color:#abd9e9">■</span> 10–15 m<br>
      <span style="color:#fee090">■</span> 15–20 m<br>
      <span style="color:#f46d43">■</span> 20–30 m<br>
      <span style="color:#d73027">■</span> &gt; 30 m<br>
      <br>
      <span style="color:#888888">▲</span> Building centroid (GeoJSON)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Save and (optionally) open ─────────────────────────────────────────────
    print(f"\n[4/4] Saving map to: {OUTPUT_HTML}")
    m.save(str(OUTPUT_HTML))
    print("  Done.")

    if AUTO_OPEN_BROWSER:
        print("\n  Opening map in browser ...")
        webbrowser.open(OUTPUT_HTML.as_uri())
    else:
        print("\n  Set AUTO_OPEN_BROWSER = True to open the map automatically.")

    print("\n" + "=" * 68)
    print("  Tip: zoom to street level (Ctrl+scroll), click any marker")
    print("  to see mean/max/n_buildings values for that GPS point.")
    print("  Toggle building centroids via the layer control (top right).")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()