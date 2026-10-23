"""
================================================================================
Land-Use Classification Inspector — 4-Method Interactive Diagnostic Map
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Builds an interactive, browser-based diagnostic map comparing four land-use /
land-cover classification sources at every GPS measurement point, with all
sensors of the active campaign combined on a single map.

The script produces:
  • One self-contained interactive HTML map per campaign / variant tag
  • Console report of the source column used for each method
  • Console warnings for any coordinate carrying inconsistent classifications

Classification methods compared
-------------------------------
  legacy  Raw 'land_use' column of the original per-sensor files (a satellite
          raster lookup made before this pipeline). Known to be PARTIALLY
          WRONG — kept for QA reference only. Standard campaign only.

  clcr    CLC 2018 raster, from the official 100 m GeoTIFF, by
          join_clc_raster_standard.py → land_use_clcr. Kept for cross-checks.
          Standard campaign only (no raster join for the tuk-tuk campaign).

  clcv    CLC 2018 vector — the project's land-use reference → land_use_clcv
          (join_clc_standard.py / join_clc_tuktuk.py). Default colouring.

  cos     COS2023 vector (Direção-Geral do Território) → cos23_n4_code
          (join_cos_standard.py / join_cos23_tuktuk.py).

A method that is unavailable for the loaded campaign is disabled in the
browser UI.

Column resolution
-----------------
The source column of each method is taken from a candidate list per campaign
(METHOD_COLUMN_CANDIDATES); the first match is used and printed. For the
tuk-tuk campaign, 'clcv' tries 'land_use_clcv' (written by join_clc_tuktuk.py)
and falls back to 'land_use'; the tuk-tuk 'land_use' column is never used as
the 'legacy' method, so 'legacy' and 'clcr' are unavailable for
CAMPAIGN = "tuktuk".

Combined sensors and deduplication
----------------------------------
All sensors of the active campaign are merged onto one map, each point tagged
with the sensor(s) that recorded it. Measurements at the exact same GPS
coordinate (e.g. many 1-minute records while parked) are collapsed into one
map point carrying the measurement count and the sensors involved. A console
warning is printed if a coordinate carries more than one class for the same
method (classification is a deterministic function of location, so this
flags an upstream data-quality issue); the first non-null value is kept.

Performance for large point counts
----------------------------------
No point is dropped or subsampled, so that every deduplicated point can be
inspected when zooming in. Points are added in small batches via
requestAnimationFrame (with a live progress readout) and drawn with Leaflet's
Canvas renderer, so the browser does not freeze at tens of thousands of
points. Only Leaflet core (bundled by folium) is used — no external plugin
CDN.

Interactive controls
--------------------
Control panel (top-right), retractable legend (bottom-left):

  • Colour points by any available classification method. Clicking a point
    always shows the results of all available methods.

  • Polygon overlays (semi-transparent, toggled independently): CLC vector and
    COS2023 vector polygons, coloured with the same palette as the
    corresponding point-colouring method for visual cross-checking.

  • Minimum measurement count: show only points backed by at least N
    measurements. Combined with the disagreement filter (logical AND), this
    isolates disagreements at locations sampled many times rather than
    one-off GPS fixes.

  • Disagreement diagnostic: pick two methods (A / B) and show only points
    where they assign a different class. Points where either method is
    missing are excluded. A live counter reports how many points pass the
    active filters.

Colour scheme: CLC classes (legacy / clcr / clcv share the same 44-class
palette, grouped by CORINE level-1 branch with lightness varied within each
branch) and COS2023 classes (grouped by their 9 level-1 branches) have fixed
colours, independent of the classes loaded in a given run, so the same class
is always the same colour.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon', 'land_use',
             'land_use_clcr', 'land_use_clcv', 'cos23_n4_code'
    input  : CLC_VECTOR_PATH (CLC 2018 GeoPackage), COS23_PATH (COS2023
             GeoPackage) — polygon overlays, clipped to the points' extent
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             landuse_classification_inspector_<CAMPAIGN>_<variant_tag>.html

Open the HTML file in a browser (or set AUTO_OPEN_BROWSER = True). Internet
access is needed in the browser only, for the OpenStreetMap basemap tiles.

Requirements
------------
    pip install folium geopandas pandas shapely pyproj

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.2
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
import sys
import warnings
import webbrowser
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CLC_VECTOR_PATH, COS23_PATH, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import folium
import geopandas as gpd
import pandas as pd

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from clc_mapping import CLC_MAPPING, CLC_LVL3_LABEL  # noqa: E402
import cos23_mapping as cos23  # noqa: E402

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
# "standard" : 1st Solar Jinie campaign
# "tuktuk"   : Tuk-tuk campaign
CAMPAIGN: str = "standard"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Vector polygon sources for the toggleable overlay layers -----------------
CLC_VECTOR_GPKG: Path = CLC_VECTOR_PATH
COS_VECTOR_GPKG: Path = COS23_PATH

# --- Default point-colouring method -------------------------------------------
# One of: "legacy", "clcr", "clcv", "cos".  Falls back to the first available
# method (with a console warning) if unavailable for the active campaign.
DEFAULT_COLOR_METHOD: str = "clcv"

# --- Polygon overlay clipping -------------------------------------------------
# Margin in degrees added around the loaded points' bounding box before clipping
# the national-scale CLC / COS polygon layers, so the overlay extends slightly
# beyond the point cloud itself.
POLYGON_CLIP_MARGIN_DEG: float = 0.02   # ~2 km at Lisbon's latitude

# --- Polygon simplification ---------------------------------------------------
# Tolerance in decimal degrees applied to every clipped polygon before it is
# embedded in the HTML.  COS2023 has a very fine minimum mapping unit (many
# small, highly detailed parcels), so on a wide extent the raw polygon layer can
# reach hundreds of MB of coordinates.  0.00008 deg ~ 9 m at Lisbon's latitude:
# visually imperceptible at the zoom levels used here, but it cuts the vertex
# count — and the file size — drastically.  Set to 0.0 for exact boundaries.
POLYGON_SIMPLIFY_TOLERANCE_DEG: float = 0.00008

# --- Polygon count warning threshold ------------------------------------------
# If either layer still exceeds this many features after clipping and
# simplification, a console warning suggests a narrower subset or a coarser
# tolerance.  The map is still generated — nothing is ever silently dropped.
POLYGON_COUNT_WARNING_THRESHOLD: int = 4000

# --- Output -------------------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "5_land_cover" / "landuse_inspector_outputs"

# Set to True to open the generated map in the default browser after saving.
AUTO_OPEN_BROWSER: bool = False


# ==============================================================================
# CONSTANTS
# ==============================================================================

# --- Classification methods ---------------------------------------------------
METHOD_KEYS: list[str] = ['legacy', 'clcr', 'clcv', 'cos']

# Which colour palette / code type each method uses ('clc' = 1-44 grid_code,
# 'cos' = dotted level-4 code string).
METHOD_KIND: dict[str, str] = {
    'legacy': 'clc',
    'clcr':   'clc',
    'clcv':   'clc',
    'cos':    'cos',
}

METHOD_DISPLAY_NAMES: dict[str, str] = {
    'legacy': 'CLC raster — legacy (pre-pipeline, partially wrong)',
    'clcr':   'CLC raster — new (join_clc_raster_standard.py)',
    'clcv':   'CLC vector — reference (land_use_clcv)',
    'cos':    'COS2023 vector (DGT)',
}

# Candidate source columns per campaign, tried in order (see the
# "Column resolution" section of the module docstring).
METHOD_COLUMN_CANDIDATES: dict[str, dict[str, list[str]]] = {
    'standard': {
        'legacy': ['land_use'],
        'clcr':   ['land_use_clcr'],
        'clcv':   ['land_use_clcv'],
        'cos':    ['cos23_n4_code'],
    },
    'tuktuk': {
        'legacy': [],                            # no pre-pipeline raster column
        'clcr':   [],                            # no tuk-tuk raster join script
        'clcv':   ['land_use_clcv', 'land_use'],  # see module docstring
        'cos':    ['cos23_n4_code'],
    },
}

# --- Sensors per campaign -----------------------------------------------------
STANDARD_SENSORS: list[str] = ['j30', 'j31', 'j32', 'j33']
TUKTUK_SENSORS: list[str]   = ['j29', 'j30', 'j31', 'j32', 'j33']


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor names to their CSV paths.

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
    """
    if campaign == "standard":
        return {
            sensor: data_dir / f"{sensor}_satellite_landuse_v2_{suffix}.csv"
            for sensor in STANDARD_SENSORS
        }
    elif campaign == "tuktuk":
        return {
            sensor: tuktuk_data_dir / f"{sensor}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
            for sensor in TUKTUK_SENSORS
        }
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


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output directory, embedding campaign + variant tag so
    outputs from different campaigns/variants never overwrite each other."""
    return base_dir / campaign / variant_tag


def _build_output_filename(base_name: str, campaign: str, variant_tag: str,
                           extension: str) -> str:
    """Build an output filename embedding the active campaign and variant tag."""
    return f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolved at startup from the configuration above
_VARIANT_TAG: str = _variant_tag()
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
)
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE: Path = OUTPUT_DIR / _build_output_filename(
    'landuse_classification_inspector', CAMPAIGN, _VARIANT_TAG, 'html'
)


# ==============================================================================
# CLC REVERSE LOOKUP
# ==============================================================================
# Maps the 3-digit CLC code used in the vector GeoPackage (Code_18) back to the
# project's 1-44 grid_code convention used by land_use_clcv and clc_mapping.py.

CLC_CODE_TO_LAND_USE: dict[int, int] = {
    entry.clc_code: land_use for land_use, entry in CLC_MAPPING.items()
}


# ==============================================================================
# COLOUR HELPERS
# ==============================================================================
# Deterministic, class-based colour palettes.  A given class always renders in
# the same colour regardless of which classes happen to be present in a run,
# across methods, sensors and campaigns — essential for point/polygon coherence
# and for the disagreement diagnostic to be visually meaningful.

def _lighten_hex(hex_color: str, amount: float) -> str:
    """Lighten a hex colour by mixing it with white (amount=0 → unchanged, 1 → white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * amount)
    g = round(g + (255 - g) * amount)
    b = round(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# Official CORINE level-1 branches (first digit of the 3-digit CLC code).
CLC_GROUP_BASE_COLORS: dict[int, str] = {
    1: "#C0392B",   # Artificial surfaces      (matches project Urban colour)
    2: "#F1C40F",   # Agricultural areas
    3: "#2E7D32",   # Forest and semi-natural areas (matches project Rural colour)
    4: "#8E44AD",   # Wetlands
    5: "#1565C0",   # Water bodies              (matches project Moving colour)
}


def clc_level1_group(clc_code: int) -> int:
    """First digit of the 3-digit CLC code = official CORINE level-1 group (1-5)."""
    return clc_code // 100


def build_clc_color_map(land_use_codes) -> dict[int, str]:
    """
    Build a colour for every CLC grid_code (1-44) in `land_use_codes`,
    grouped by CORINE level-1 branch and varied by lightness within each
    group. Sorted by grid_code (not by frequency) so the result never
    depends on which classes happen to be loaded in this particular run.
    """
    codes = sorted(set(int(c) for c in land_use_codes if int(c) in CLC_MAPPING))
    colors: dict[int, str] = {}
    by_group: dict[int, list[int]] = {}
    for code in codes:
        group = clc_level1_group(CLC_MAPPING[code].clc_code)
        by_group.setdefault(group, []).append(code)
    for group, group_codes in by_group.items():
        base = CLC_GROUP_BASE_COLORS.get(group, "#7F8C8D")
        n = len(group_codes)
        lightness_values = [0.0] if n == 1 else [i / (n - 1) * 0.55 for i in range(n)]
        for code, lightness in zip(sorted(group_codes), lightness_values):
            colors[code] = _lighten_hex(base, lightness)
    return colors


# Base palette for the 9 official COS2023 level-1 branches.
COS_GROUP_PALETTE: list[str] = [
    "#1565C0", "#2E7D32", "#C0392B", "#8E44AD", "#E67E22",
    "#16A085", "#D35400", "#7F8C8D", "#2C3E50",
]


def cos_level1_group(code: str) -> str:
    """First segment of the dotted COS level-4 code (e.g. '1.2.3.4' → '1')."""
    if not code or pd.isna(code):
        return "unknown"
    return str(code).split(".")[0]


def build_cos_color_map(cos_codes) -> dict[str, str]:
    """Same principle as build_clc_color_map(), for COS2023 codes."""
    codes = sorted(set(
        str(c) for c in cos_codes
        if c and str(c) not in ("nan", "-999", "Unknown / missing")
    ))
    by_group: dict[str, list[str]] = {}
    for code in codes:
        by_group.setdefault(cos_level1_group(code), []).append(code)
    groups = sorted(by_group.keys())
    colors: dict[str, str] = {}
    for i, group in enumerate(groups):
        base = COS_GROUP_PALETTE[i % len(COS_GROUP_PALETTE)]
        group_codes = sorted(by_group[group])
        n = len(group_codes)
        lightness_values = [0.0] if n == 1 else [j / (n - 1) * 0.55 for j in range(n)]
        for code, lightness in zip(group_codes, lightness_values):
            colors[code] = _lighten_hex(base, lightness)
    return colors


def clc_label(code: int) -> str:
    """Return the CLC level-3 label of a 1-44 grid_code."""
    return CLC_LVL3_LABEL.get(int(code), "Unknown / missing")


def cos_label(code: str) -> str:
    """Return the COS2023 level-4 label of a dotted code."""
    try:
        return cos23.get_label(code)
    except Exception:
        return "Unknown / missing"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def _resolve_column(df_columns: set, candidates: list[str]) -> str | None:
    """Return the first candidate column present in the file, or None."""
    for c in candidates:
        if c in df_columns:
            return c
    return None


def load_and_merge_sensors(
    campaign: str,
    sensor_files: dict[str, Path],
    method_candidates: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load every sensor's CSV for the active campaign, keep only lat/lon +
    whichever candidate column resolves for each classification method,
    tag each row with its sensor, and concatenate into a single DataFrame.

    Returns
    -------
    (merged_df, resolved_cols) where resolved_cols maps method → the
    actual column name that was found and used (locked in from the first
    sensor file where it is found; identical for every subsequent sensor).
    """
    frames = []
    resolved_cols: dict[str, str] = {}

    for sensor_id, path in sensor_files.items():
        tag = sensor_id.upper()
        if not path.exists():
            print(f"  [SKIP] {tag}: file not found → {path.name}")
            continue

        df = pd.read_csv(path, low_memory=False)
        lat_col, lon_col = f"{tag} lat", f"{tag} lon"
        if lat_col not in df.columns or lon_col not in df.columns:
            print(f"  [SKIP] {tag}: missing {lat_col!r}/{lon_col!r}")
            continue

        cols_present = set(df.columns)
        keep = {lat_col, lon_col}
        for method, default_candidates in method_candidates.items():
            candidates = [resolved_cols[method]] if method in resolved_cols else default_candidates
            col = _resolve_column(cols_present, candidates)
            if col:
                keep.add(col)
                resolved_cols.setdefault(method, col)

        sub = df[[c for c in keep if c in df.columns]].copy()
        sub = sub.rename(columns={lat_col: "lat", lon_col: "lon"})
        sub["sensor"] = tag

        n_before = len(sub)
        sub = sub.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        print(f"  {tag}: {len(sub):,} / {n_before:,} rows with valid GPS")
        frames.append(sub)

    if not frames:
        raise RuntimeError(
            "No sensor files could be loaded - check CAMPAIGN / FILE_SUFFIX / "
            "TUKTUK_FILE_SUFFIX and the DATA_DIR / TUKTUK_DATA_DIR paths."
        )

    merged = pd.concat(frames, ignore_index=True)
    return merged, resolved_cols


def normalize_codes(df: pd.DataFrame, resolved_cols: dict[str, str]) -> pd.DataFrame:
    """Coerce each resolved method column to a clean, missing-aware type."""
    for method, col in resolved_cols.items():
        if col not in df.columns:
            continue
        if METHOD_KIND[method] == "clc":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~df[col].between(1, 44), col] = pd.NA
        else:  # "cos"
            df[col] = df[col].astype("string")
            df.loc[df[col].isin(["nan", "-999", "Unknown / missing", ""]), col] = pd.NA
    return df


def deduplicate_points(df: pd.DataFrame, resolved_cols: dict[str, str]) -> pd.DataFrame:
    """
    Collapse rows sharing the EXACT same (lat, lon) into a single point,
    keeping a measurement count, the set of sensors involved, and (for
    each available method) the first non-null classification code plus a
    distinct-value count used for the QA warning below.
    """
    agg_kwargs = {
        "n": ("lat", "size"),
        "sensors": ("sensor", lambda s: ",".join(sorted(set(s)))),
    }
    for method, col in resolved_cols.items():
        agg_kwargs[f"__code_{method}"] = (col, "first")
        agg_kwargs[f"__nun_{method}"] = (col, "nunique")

    out = df.groupby(["lat", "lon"], as_index=False, sort=False).agg(**agg_kwargs)

    for method in resolved_cols:
        nun_col = f"__nun_{method}"
        n_inconsistent = int((out[nun_col] > 1).sum())
        if n_inconsistent:
            print(
                f"  [WARNING] {n_inconsistent:,} location(s) have more than one "
                f"distinct '{method}' classification at the EXACT same GPS "
                f"coordinate - keeping the first non-null value seen for each. "
                f"This should not normally happen (classification is a "
                f"deterministic function of location) and may indicate an "
                f"upstream data-quality issue worth investigating."
            )
    return out


def compute_bbox(dedup_df: pd.DataFrame, margin_deg: float) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) of the point cloud + margin."""
    return (
        float(dedup_df["lon"].min() - margin_deg),
        float(dedup_df["lat"].min() - margin_deg),
        float(dedup_df["lon"].max() + margin_deg),
        float(dedup_df["lat"].max() + margin_deg),
    )


# ==============================================================================
# POLYGON LOADING
# ==============================================================================

def _simplify_and_warn(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    """
    Simplify geometries (see POLYGON_SIMPLIFY_TOLERANCE_DEG) to keep the
    embedded GeoJSON a manageable size, and warn (without dropping
    anything) if the feature count is still very high after clipping.
    """
    if POLYGON_SIMPLIFY_TOLERANCE_DEG > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf["geometry"].simplify(
            POLYGON_SIMPLIFY_TOLERANCE_DEG, preserve_topology=True
        )
        # simplify() can occasionally collapse a sliver polygon to empty -
        # drop those rather than emit invalid/empty GeoJSON geometries.
        gdf = gdf[~gdf["geometry"].is_empty].reset_index(drop=True)

    if len(gdf) > POLYGON_COUNT_WARNING_THRESHOLD:
        print(
            f"  [WARNING] {layer_name} polygon overlay still has {len(gdf):,} "
            f"features after clipping + simplification (threshold: "
            f"{POLYGON_COUNT_WARNING_THRESHOLD:,}). The map will still be "
            f"generated, but the output file may be large and slow to open "
            f"/ render. Consider: a smaller FILE_SUFFIX subset (e.g. "
            f"'sd_cleaned_lisbon' instead of the full 'sd_cleaned'), a "
            f"smaller POLYGON_CLIP_MARGIN_DEG, or a coarser "
            f"POLYGON_SIMPLIFY_TOLERANCE_DEG."
        )
    return gdf


def load_clc_polygons(gpkg_path: Path, bbox_wgs84) -> gpd.GeoDataFrame | None:
    """
    Load the CLC 2018 vector GeoPackage, map its 3-digit Code_18 to the
    project's 1-44 land_use_clcv grid_code, reproject to EPSG:4326, and
    clip to the point cloud's bounding box (+ margin).
    """
    if not gpkg_path.exists():
        print(f"  [WARNING] CLC vector file not found → polygon overlay disabled\n         ({gpkg_path})")
        return None

    gdf = gpd.read_file(gpkg_path)

    if "Code_18" not in gdf.columns:
        candidates = [c for c in gdf.columns if "code" in c.lower() or "clc" in c.lower()]
        if not candidates:
            print("  [WARNING] 'Code_18' column not found in the CLC GeoPackage - polygon overlay disabled.")
            return None
        print(f"  [INFO] 'Code_18' not found - using {candidates[0]!r} instead.")
        gdf = gdf.rename(columns={candidates[0]: "Code_18"})

    gdf["Code_18"] = pd.to_numeric(gdf["Code_18"], errors="coerce")

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=3035)
    gdf = gdf.to_crs(epsg=4326)

    gdf["land_use_clcv"] = gdf["Code_18"].map(CLC_CODE_TO_LAND_USE)
    gdf = gdf.dropna(subset=["land_use_clcv", "geometry"]).copy()
    gdf["land_use_clcv"] = gdf["land_use_clcv"].astype(int)

    if bbox_wgs84 is not None:
        lon_min, lat_min, lon_max, lat_max = bbox_wgs84
        gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max]

    gdf = _simplify_and_warn(gdf, "CLC")
    return gdf[["land_use_clcv", "geometry"]].reset_index(drop=True)


def load_cos_polygons(gpkg_path: Path, bbox_wgs84) -> gpd.GeoDataFrame | None:
    """
    Load the COS2023 vector GeoPackage, auto-detect the level-4 code
    field (COS23n4_C vs COS23_n4_C, same convention as join_cos_standard.py),
    reproject to EPSG:4326, and clip to the point cloud's bounding box.
    """
    if not gpkg_path.exists():
        print(f"  [WARNING] COS vector file not found → polygon overlay disabled\n         ({gpkg_path})")
        return None

    gdf = gpd.read_file(gpkg_path)

    code_field = None
    for candidate in ("COS23n4_C", "COS23_n4_C"):
        if candidate in gdf.columns:
            code_field = candidate
            break
    if code_field is None:
        print("  [WARNING] COS level-4 code field not found (tried COS23n4_C / COS23_n4_C) - polygon overlay disabled.")
        return None

    gdf = gdf.rename(columns={code_field: "cos23_n4_code"})
    gdf["cos23_n4_code"] = gdf["cos23_n4_code"].astype(str)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    if bbox_wgs84 is not None:
        lon_min, lat_min, lon_max, lat_max = bbox_wgs84
        gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max]

    gdf = _simplify_and_warn(gdf, "COS")
    return gdf[["cos23_n4_code", "geometry"]].reset_index(drop=True)


def polygons_to_geojson(gdf, code_col: str, kind: str, color_map: dict, label_fn) -> dict:
    """Attach {code, label, color} properties and serialise to a GeoJSON dict."""
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}

    gdf = gdf.copy()
    if kind == "clc":
        gdf["code"] = gdf[code_col].astype(int)
    else:
        gdf["code"] = gdf[code_col].astype(str)
    gdf["label"] = gdf["code"].map(label_fn)
    gdf["color"] = gdf["code"].map(lambda c: color_map.get(c, "#888888"))

    gdf = gdf[["code", "label", "color", "geometry"]]
    return json.loads(gdf.to_json())


# ==============================================================================
# COLOUR MAPS, LABELS AND POINT RECORDS
# ==============================================================================

def build_method_color_maps(
    dedup_df: pd.DataFrame,
    resolved_cols: dict[str, str],
    clc_polygon_codes,
    cos_polygon_codes,
) -> dict[str, dict]:
    """
    Build one colour map per available method, over the UNION of codes
    present in the points AND in the polygon overlay layer, so point and
    polygon colours always agree for a given method.
    """
    color_maps: dict[str, dict] = {}
    for method in resolved_cols:
        code_col = f"__code_{method}"
        if code_col not in dedup_df.columns:
            continue
        point_codes = dedup_df[code_col].dropna().tolist()
        if METHOD_KIND[method] == "clc":
            all_codes = set(int(c) for c in point_codes) | set(int(c) for c in clc_polygon_codes)
            color_maps[method] = build_clc_color_map(all_codes)
        else:
            all_codes = set(str(c) for c in point_codes) | set(str(c) for c in cos_polygon_codes)
            color_maps[method] = build_cos_color_map(all_codes)
    return color_maps


def build_point_records(
    dedup_df: pd.DataFrame,
    resolved_cols: dict[str, str],
    color_maps: dict[str, dict],
) -> list[dict]:
    """Build one JSON-ready record per deduplicated point (code, label, colour per method)."""
    records = []
    for row in dedup_df.to_dict(orient="records"):
        rec = {
            "lat": round(float(row["lat"]), 6),
            "lon": round(float(row["lon"]), 6),
            "n": int(row["n"]),
            "sensors": row["sensors"],
        }
        for method in resolved_cols:
            code_key = f"__code_{method}"
            if code_key not in row:
                continue
            code = row[code_key]
            if pd.isna(code):
                rec[method] = None
                continue
            if METHOD_KIND[method] == "clc":
                code_i = int(code)
                rec[method] = {
                    "code": code_i,
                    "label": clc_label(code_i),
                    "color": color_maps[method].get(code_i, "#888888"),
                }
            else:
                code_s = str(code)
                rec[method] = {
                    "code": code_s,
                    "label": cos_label(code_s),
                    "color": color_maps[method].get(code_s, "#888888"),
                }
        records.append(rec)
    return records


def build_legend_entries(
    dedup_df: pd.DataFrame,
    resolved_cols: dict[str, str],
    color_maps: dict[str, dict],
) -> dict[str, list[dict]]:
    """Build the legend entries (label, colour, point count) of each method."""
    legends: dict[str, list[dict]] = {}
    for method in resolved_cols:
        code_col = f"__code_{method}"
        if code_col not in dedup_df.columns:
            continue
        counts = dedup_df[code_col].value_counts(dropna=True)
        items = []
        for code, n in counts.items():
            if METHOD_KIND[method] == "clc":
                code_i = int(code)
                items.append({
                    "label": clc_label(code_i),
                    "color": color_maps[method].get(code_i, "#888888"),
                    "n": int(n),
                })
            else:
                code_s = str(code)
                items.append({
                    "label": cos_label(code_s),
                    "color": color_maps[method].get(code_s, "#888888"),
                    "n": int(n),
                })
        n_missing = int(dedup_df[code_col].isna().sum())
        if n_missing:
            items.append({"label": "Unknown / missing", "color": "#888888", "n": n_missing})
        legends[method] = items
    return legends


# ==============================================================================
# HTML / JS TEMPLATE
# ==============================================================================
# Raw string with @@TOKEN@@ placeholders, substituted via .replace().  This is
# deliberately not an f-string or .format() template: the JS body is full of
# literal { } braces, which those mechanisms would misinterpret.

_HEAD_ASSETS = """
<style>
  .landuse-control-panel, .landuse-legend-panel {
      background: white; padding: 10px 12px; border-radius: 8px;
      border: 1px solid #ccc; font-family: sans-serif; font-size: 12.5px;
      box-shadow: 2px 2px 6px rgba(0,0,0,.25); max-width: 280px;
  }
  .landuse-control-panel h4, .landuse-legend-panel h4 { margin: 2px 0 6px 0; font-size: 13px; }
  .landuse-control-panel label { display:block; margin: 2px 0; cursor: pointer; }
  .landuse-control-panel select { max-width: 260px; }
  .landuse-legend-panel { max-height: 320px; overflow-y: auto; }
  #legend-header { cursor: pointer; font-weight: bold; margin-bottom: 4px; }
  .landuse-legend-item { display:flex; align-items:center; margin:2px 0; }
  .landuse-swatch { width:12px; height:12px; border-radius:2px; margin-right:6px; flex-shrink:0; border:1px solid rgba(0,0,0,.2); display:inline-block; }
  .landuse-legend-label { flex:1; }
  .landuse-legend-n { color:#888; margin-left:6px; }
  .landuse-loading { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
      background:white; padding:10px 18px; border-radius:6px; font-family:sans-serif;
      box-shadow:0 0 8px rgba(0,0,0,.3); z-index:9999; }
  hr.landuse-sep { margin:8px 0; border:none; border-top:1px solid #ddd; }
</style>
"""

_BODY_SCRIPT = """
<div id="landuse-loading-indicator" class="landuse-loading">Loading points...</div>
<script>
(function () {
    const DATA = @@DATA_JSON@@;

    document.addEventListener("DOMContentLoaded", function () {
        const MAP = window["@@MAP_VAR@@"];
        if (!MAP) { console.error("Map object @@MAP_VAR@@ not found"); return; }

        let currentMethod = DATA.defaultMethod;
        const markers = [];

        const clusterGroup = L.layerGroup();

        function colorFor(pt, method) {
            const m = pt[method];
            return (m && m.color) ? m.color : '#888888';
        }

        function buildPopupHTML(pt) {
            let html = '<div style="font-family:sans-serif;font-size:12.5px;max-width:260px">';
            html += '<b>' + pt.n + ' measurement' + (pt.n > 1 ? 's' : '') + '</b>';
            if (pt.sensors) html += ' &mdash; ' + pt.sensors;
            html += '<hr class="landuse-sep">';
            DATA.methodOrder.forEach(function (key) {
                const meta = DATA.methods[key];
                if (!meta || !meta.available) return;
                const m = pt[key];
                const label = m ? m.label : 'Unknown / missing';
                const color = m ? m.color : '#888888';
                html += '<div><span class="landuse-swatch" style="background:' + color +
                        '"></span><b>' + meta.label + ':</b> ' + label + '</div>';
            });
            html += '<div style="color:#888;font-size:11px;margin-top:4px">lat ' +
                    pt.lat.toFixed(5) + ', lon ' + pt.lon.toFixed(5) + '</div></div>';
            return html;
        }

        clusterGroup.addTo(MAP);

        // Build markers in small batches via requestAnimationFrame instead
        // of one giant synchronous loop, so the browser tab never freezes
        // and the loading indicator reflects real progress - essential
        // once the point count reaches into the tens of thousands.
        const CHUNK_SIZE = 1500;
        let pointIndex = 0;
        const loadingEl = document.getElementById('landuse-loading-indicator');

        function addPointsChunk() {
            const end = Math.min(pointIndex + CHUNK_SIZE, DATA.points.length);
            const chunkMarkers = [];
            for (; pointIndex < end; pointIndex++) {
                const pt = DATA.points[pointIndex];
                const marker = L.circleMarker([pt.lat, pt.lon], {
                    radius: 5, weight: 1, color: '#333333',
                    fillColor: colorFor(pt, currentMethod), fillOpacity: 0.85
                });
                marker.bindPopup(buildPopupHTML(pt), { maxWidth: 300 });
                marker._ptData = pt;
                marker._visible = true;
                markers.push(marker);
                chunkMarkers.push(marker);
            }
            chunkMarkers.forEach(function (mk) { clusterGroup.addLayer(mk); });

            if (loadingEl) {
                loadingEl.textContent = 'Loading points... (' +
                    pointIndex.toLocaleString() + ' / ' + DATA.points.length.toLocaleString() + ')';
            }

            if (pointIndex < DATA.points.length) {
                requestAnimationFrame(addPointsChunk);
            } else {
                if (loadingEl) loadingEl.style.display = 'none';
                if (markers.length > 0) {
                    const group = L.featureGroup(markers);
                    MAP.fitBounds(group.getBounds().pad(0.05));
                }
                updateLegend(currentMethod);
                const countEl = document.getElementById('landuse-filter-count');
                if (countEl) {
                    countEl.textContent = markers.length.toLocaleString() + ' / ' +
                        markers.length.toLocaleString() + ' points shown';
                }
            }
        }

        function setColorMethod(method) {
            currentMethod = method;
            markers.forEach(function (mk) {
                if (mk._visible) mk.setStyle({ fillColor: colorFor(mk._ptData, method) });
            });
            updateLegend(method);
        }

        function updateLegend(method) {
            const items = DATA.legends[method] || [];
            const container = document.getElementById('legend-content');
            if (!container) return;
            let html = '<div style="font-weight:bold;margin-bottom:4px">' +
                        (DATA.methods[method] ? DATA.methods[method].label : method) + '</div>';
            items.forEach(function (it) {
                html += '<div class="landuse-legend-item">' +
                        '<span class="landuse-swatch" style="background:' + it.color + '"></span>' +
                        '<span class="landuse-legend-label">' + it.label + '</span>' +
                        '<span class="landuse-legend-n">' + it.n.toLocaleString() + '</span></div>';
            });
            container.innerHTML = html;
        }

        // ---- Polygon overlays (transparent, same colours as points) ----
        function polyStyle(feature) {
            return { fillColor: feature.properties.color, color: feature.properties.color,
                     weight: 0.6, fillOpacity: 0.35 };
        }
        function polyPopup(feature, layer) {
            layer.bindPopup('<b>' + feature.properties.label + '</b>');
        }
        const clcPolyLayer = L.geoJSON(DATA.clcPolygons, { style: polyStyle, onEachFeature: polyPopup });
        const cosPolyLayer = L.geoJSON(DATA.cosPolygons, { style: polyStyle, onEachFeature: polyPopup });

        // ---- Point filtering: minimum measurement count AND/OR disagreement ----
        // The two criteria are combined (logical AND), so the user can isolate
        // e.g. "disagreements at locations backed by many measurements" - the
        // most significant disagreements, rather than one-off GPS samples.
        function applyFilter() {
            const onlyDisagree = document.getElementById('landuse-only-disagree').checked;
            const mA = document.getElementById('landuse-compare-a').value;
            const mB = document.getElementById('landuse-compare-b').value;

            const minNRaw = parseInt(document.getElementById('landuse-min-n').value, 10);
            const minN = (isNaN(minNRaw) || minNRaw < 1) ? 1 : minNRaw;

            markers.forEach(function (mk) {
                const pt = mk._ptData;
                let show = (pt.n >= minN);
                if (show && onlyDisagree) {
                    const a = pt[mA], b = pt[mB];
                    const validA = a && a.code !== null && a.code !== undefined;
                    const validB = b && b.code !== null && b.code !== undefined;
                    show = validA && validB && (String(a.code) !== String(b.code));
                }
                mk._visible = show;
            });

            clusterGroup.clearLayers();
            const visible = [];
            markers.forEach(function (mk) {
                if (mk._visible) {
                    mk.setStyle({ fillColor: colorFor(mk._ptData, currentMethod) });
                    visible.push(mk);
                }
            });
            visible.forEach(function (mk) { clusterGroup.addLayer(mk); });

            const countEl = document.getElementById('landuse-filter-count');
            if (countEl) {
                countEl.textContent = visible.length.toLocaleString() + ' / ' +
                    markers.length.toLocaleString() + ' points shown';
            }
        }

        // ---- Control panel ----
        const ClassificationControl = L.Control.extend({
            options: { position: 'topright' },
            onAdd: function () {
                const div = L.DomUtil.create('div', 'landuse-control-panel');
                let methodOptions = '';
                let radioHtml = '';
                DATA.methodOrder.forEach(function (key) {
                    const meta = DATA.methods[key];
                    if (!meta || !meta.available) return;
                    methodOptions += '<option value="' + key + '">' + meta.label + '</option>';
                    const checked = (key === DATA.defaultMethod) ? 'checked' : '';
                    radioHtml += '<label><input type="radio" name="landuse-color-method" value="' + key +
                                 '" ' + checked + '> ' + meta.label + '</label>';
                });
                div.innerHTML =
                    '<h4>Colour points by</h4>' + radioHtml +
                    '<hr class="landuse-sep">' +
                    '<h4>Polygon overlays (semi-transparent)</h4>' +
                    '<label><input type="checkbox" id="landuse-toggle-clc"> CLC vector</label>' +
                    '<label><input type="checkbox" id="landuse-toggle-cos"> COS2023 vector</label>' +
                    '<hr class="landuse-sep">' +
                    '<h4>Minimum measurements per point</h4>' +
                    '<label>Show points with at least ' +
                    '<input type="number" id="landuse-min-n" value="1" min="1" step="1" ' +
                    'style="width:70px"> measurement(s)</label>' +
                    '<hr class="landuse-sep">' +
                    '<h4>Disagreement diagnostic</h4>' +
                    '<label>A: <select id="landuse-compare-a">' + methodOptions + '</select></label>' +
                    '<label>B: <select id="landuse-compare-b">' + methodOptions + '</select></label>' +
                    '<label><input type="checkbox" id="landuse-only-disagree"> Show disagreements only</label>' +
                    '<div id="landuse-filter-count" style="margin-top:6px;color:#555"></div>';
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);
                return div;
            }
        });
        MAP.addControl(new ClassificationControl());

        const LegendControl = L.Control.extend({
            options: { position: 'bottomleft' },
            onAdd: function () {
                const div = L.DomUtil.create('div', 'landuse-legend-panel');
                div.innerHTML = '<div id="legend-header">Legend &#9662;</div><div id="legend-content"></div>';
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);
                return div;
            }
        });
        MAP.addControl(new LegendControl());

        // ---- Wire up events (elements exist only after addControl ran) ----
        document.querySelectorAll('input[name="landuse-color-method"]').forEach(function (el) {
            el.addEventListener('change', function () { setColorMethod(this.value); });
        });
        document.getElementById('landuse-toggle-clc').addEventListener('change', function () {
            if (this.checked) { clcPolyLayer.addTo(MAP); } else { MAP.removeLayer(clcPolyLayer); }
        });
        document.getElementById('landuse-toggle-cos').addEventListener('change', function () {
            if (this.checked) { cosPolyLayer.addTo(MAP); } else { MAP.removeLayer(cosPolyLayer); }
        });
        document.getElementById('landuse-compare-a').addEventListener('change', applyFilter);
        document.getElementById('landuse-compare-b').addEventListener('change', applyFilter);
        document.getElementById('landuse-only-disagree').addEventListener('change', applyFilter);
        document.getElementById('landuse-min-n').addEventListener('change', applyFilter);
        document.getElementById('legend-header').addEventListener('click', function () {
            const el = document.getElementById('legend-content');
            el.style.display = (el.style.display === 'none') ? 'block' : 'none';
        });

        // Pre-select two different methods for the disagreement dropdowns
        const availableKeys = DATA.methodOrder.filter(function (k) {
            return DATA.methods[k] && DATA.methods[k].available;
        });
        if (availableKeys.length >= 2) {
            document.getElementById('landuse-compare-a').value = availableKeys[0];
            document.getElementById('landuse-compare-b').value = availableKeys[availableKeys.length - 1];
        }

        // Legend starts empty until the first chunk lands; kick off the
        // asynchronous, non-blocking point loading now that every control
        // and event handler above is wired up.
        updateLegend(currentMethod);
        addPointsChunk();
    });
})();
</script>
"""


def build_map(
    points: list[dict],
    legends: dict[str, list[dict]],
    methods_meta: dict[str, dict],
    clc_geojson: dict,
    cos_geojson: dict,
    default_method: str,
    center: list[float],
) -> folium.Map:
    """Assemble the folium map: embedded point/polygon data, control panel and legend."""
    m = folium.Map(
        location=center, zoom_start=12, tiles="OpenStreetMap",
        control_scale=True, prefer_canvas=True,
    )
    map_var = m.get_name()

    m.get_root().header.add_child(folium.Element(_HEAD_ASSETS))

    data_blob = {
        "points": points,
        "legends": legends,
        "methods": methods_meta,
        "methodOrder": METHOD_KEYS,
        "clcPolygons": clc_geojson,
        "cosPolygons": cos_geojson,
        "defaultMethod": default_method,
    }
    data_json = json.dumps(data_blob, ensure_ascii=True)
    # Defensive: an embedded "</" (however unlikely in official CLC/COS
    # nomenclature labels) would make the browser's HTML parser terminate
    # the <script> tag early, silently breaking the whole page. Escaping
    # it is a no-op for normal content and a safety net otherwise.
    data_json = data_json.replace("</", "<\\/")

    script_html = (
        _BODY_SCRIPT
        .replace("@@MAP_VAR@@", map_var)
        .replace("@@DATA_JSON@@", data_json)
    )
    m.get_root().html.add_child(folium.Element(script_html))
    return m


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Load the sensor files and build the interactive classification map."""
    print('=' * 70)
    print('LAND-USE CLASSIFICATION INSPECTOR — 4-Method Diagnostic Map')
    print('=' * 70)
    print(f"  CAMPAIGN      : {CAMPAIGN}")
    print(f"  Data variant  : {_VARIANT_TAG}")
    print(f"  Output folder : {OUTPUT_DIR}\n")

    method_candidates = METHOD_COLUMN_CANDIDATES[CAMPAIGN]

    # ==========================================================================
    # 1. LOAD AND MERGE SENSOR FILES
    # ==========================================================================
    print('[1/6] Loading and merging sensor files ...')

    merged, resolved_cols = load_and_merge_sensors(
        CAMPAIGN, SENSOR_FILES, method_candidates
    )

    print('\n  Resolved source column per method:')
    for method in METHOD_KEYS:
        if method in resolved_cols:
            print(f"    {method:>7s} → {resolved_cols[method]}")
        else:
            print(f"    {method:>7s} → not available for this campaign")

    merged = normalize_codes(merged, resolved_cols)
    print(f"\n  Total merged rows: {len(merged):,}")

    # ==========================================================================
    # 2. DEDUPLICATE EXACT GPS COORDINATES
    # ==========================================================================
    print('\n[2/6] Deduplicating exact GPS coordinates ...')

    dedup_df = deduplicate_points(merged, resolved_cols)
    collapse_pct = 100 * (1 - len(dedup_df) / max(len(merged), 1))
    print(f"  {len(dedup_df):,} unique GPS coordinates "
          f"(from {len(merged):,} measurements, {collapse_pct:.1f}% collapsed)")

    bbox = compute_bbox(dedup_df, POLYGON_CLIP_MARGIN_DEG)

    # ==========================================================================
    # 3. LOAD VECTOR POLYGON OVERLAYS
    # ==========================================================================
    print('\n[3/6] Loading CLC / COS vector polygons ...')

    clc_gdf = load_clc_polygons(CLC_VECTOR_GPKG, bbox)
    cos_gdf = load_cos_polygons(COS_VECTOR_GPKG, bbox)
    print(f"  CLC polygons in view : {0 if clc_gdf is None else len(clc_gdf):,}")
    print(f"  COS polygons in view : {0 if cos_gdf is None else len(cos_gdf):,}")

    # ==========================================================================
    # 4. BUILD COLOUR MAPS
    # ==========================================================================
    print('\n[4/6] Building colour maps (coherent across points and polygons) ...')

    clc_poly_codes = [] if clc_gdf is None else clc_gdf['land_use_clcv'].dropna().tolist()
    cos_poly_codes = [] if cos_gdf is None else cos_gdf['cos23_n4_code'].dropna().tolist()
    color_maps = build_method_color_maps(
        dedup_df, resolved_cols, clc_poly_codes, cos_poly_codes
    )

    # ==========================================================================
    # 5. BUILD POINT RECORDS, LEGENDS AND POLYGON GEOJSON
    # ==========================================================================
    print('\n[5/6] Building point records, legends and polygon GeoJSON ...')

    points  = build_point_records(dedup_df, resolved_cols, color_maps)
    legends = build_legend_entries(dedup_df, resolved_cols, color_maps)

    methods_meta = {
        key: {'label': METHOD_DISPLAY_NAMES[key], 'available': key in resolved_cols}
        for key in METHOD_KEYS
    }

    empty_fc = {'type': 'FeatureCollection', 'features': []}
    clc_geojson = (
        polygons_to_geojson(clc_gdf, 'land_use_clcv', 'clc',
                            color_maps.get('clcv', {}), clc_label)
        if 'clcv' in color_maps else empty_fc
    )
    cos_geojson = (
        polygons_to_geojson(cos_gdf, 'cos23_n4_code', 'cos',
                            color_maps.get('cos', {}), cos_label)
        if 'cos' in color_maps else empty_fc
    )

    if DEFAULT_COLOR_METHOD in resolved_cols:
        default_method = DEFAULT_COLOR_METHOD
    else:
        default_method = next(iter(resolved_cols))
        print(f"  [WARNING] DEFAULT_COLOR_METHOD '{DEFAULT_COLOR_METHOD}' is not "
              f"available for this campaign — falling back to '{default_method}'.")

    # ==========================================================================
    # 6. RENDER THE INTERACTIVE HTML MAP
    # ==========================================================================
    print('\n[6/6] Rendering the interactive HTML map ...')

    center = [float(dedup_df['lat'].mean()), float(dedup_df['lon'].mean())]
    fmap = build_map(
        points=points,
        legends=legends,
        methods_meta=methods_meta,
        clc_geojson=clc_geojson,
        cos_geojson=cos_geojson,
        default_method=default_method,
        center=center,
    )
    fmap.save(str(OUTPUT_FILE))

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        print(
            f"\n  [NOTE] {size_mb:.0f} MB is large for a single HTML file. "
            f"Opening it in a text editor may be slow, since the embedded data "
            f"is written as one long line, but the browser should still handle "
            f"it. See the polygon warnings above to shrink it further."
        )

    print('\n' + '=' * 70)
    print('Analysis complete.')
    print(f"  CAMPAIGN         : {CAMPAIGN}")
    print(f"  Data variant     : {_VARIANT_TAG}")
    print(f"  Points mapped    : {len(points):,}")
    print(f"  CLC polygons     : {len(clc_geojson['features']):,}")
    print(f"  COS polygons     : {len(cos_geojson['features']):,}")
    print(f"  Output file      : {OUTPUT_FILE.name}  ({size_mb:.1f} MB)")
    print(f"  Output directory : {OUTPUT_DIR}")
    print('=' * 70)

    if AUTO_OPEN_BROWSER:
        webbrowser.open(f"file://{OUTPUT_FILE.resolve()}")


# ==============================================================================
if __name__ == '__main__':
    main()