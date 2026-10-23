"""
================================================================================
Street-Width (OSM) Data-Quality / Coverage Diagnostic
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Data-quality / coverage diagnostic for the OSM-derived street width
(sw_mapping, joined by join_sw_standard.py / join_sw_tuktuk.py), the
counterpart of building_height_quality_diagnostic.py for building height.
It lets a human check, visually and statistically, that the OSM street
width can be trusted for the H/W urban-canyon analysis.

OSM network: bounding box and cache
-----------------------------------
Unlike the building file, the street network is not a local file:
sw_mapping.load_street_network() queries the Overpass API. To limit those
queries:
  - The bounding box of the network is computed from the sensor rows that
    have a building height (RESTRICT_BBOX_TO_VALID_BH, for both campaigns),
    since W is only used together with H in the H/W ratio. This mirrors
    RESTRICT_TO_VALID_BH of join_sw_standard.py (join_sw_tuktuk.py has no
    such restriction).
  - The downloaded edges are cached on disk (CACHE_DIR / osm_network), keyed
    on CAMPAIGN + variant tag. The cache is reused only if the stored bbox
    matches the current one (rounded to 4 decimals, ~11 m), otherwise the
    network is downloaded again; FORCE_REFRESH_CACHE = True always
    re-downloads (e.g. after a change in sw_mapping). Only the columns used
    here are cached (geometry, width_source, street_width_m, highway); list
    values of raw osmnx attributes are converted to strings, which
    GeoPackage requires.
Internet access is therefore needed only on a cache miss.

Only the public interface of sw_mapping is used: load_street_network(bbox)
returning (graph_proj, edges_gdf), with edges_gdf carrying 'street_width_m'
and 'width_source'. A figure whose expected column is missing prints a
warning and is skipped.

Figures produced
----------------
  Fig A - street_network_by_source_map
      The OSM street network (all edges in the bbox), coloured by
      width_source ('osm_tag' / 'lanes_estimate' / 'highway_default'):
      where each width value comes from, independent of the driven route.
  Fig B - join_coverage_by_source_map
      Measurement points coloured by the width_source actually joined into
      the sensor files: which quality tier the real route lands on.
  Fig C - street_width_histogram
      Distribution of street_width_m (matched rows only), by width_source.
  Fig D - mean_width_by_cos23_category
      Mean street width per COS2023 level-4 category (categories with
      >= MIN_BIN_POINTS matched points).
  Fig E - width_source_composition_by_cos23
      width_source composition per COS2023 category: which land-use types
      rely most on the highway-type fallback vs an explicit OSM width tag.
  Fig F - unmatched_rate_by_cos23
      Join-failure rate (width_source == 'unmatched_too_far') per COS2023
      category — counterpart of Fig F of the BH diagnostic.
  Fig G - edge_match_distance_distribution
      Histogram + CDF of edge_match_dist_m (point-to-nearest-edge distance):
      match QUALITY, as opposed to the match RATE of Fig F.
  Fig H - street_width_vs_building_height
      street_width_m vs mean_building_height_m hexbin: checks whether H and
      W are anti-correlated (taller buildings on narrower streets), with the
      exact inputs of the canyon analysis.
  Fig I - width_by_clc_group
      Street width by CLC Urban / Rural group (boxplot), a coarser
      cross-check of Figs D-F.

The console also reports the network-level and point-level width_source
composition and street_width_m quantiles.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon', 'street_width_m',
             'width_source', 'edge_match_dist_m', 'cos23_n4_label',
             'land_use_clcv', 'mean_building_height_m' (missing columns are
             filled with NaN)
    input  : OSM street network (Overpass API, via sw_mapping), cached in
             CACHE_DIR / osm_network / <CAMPAIGN>_<variant_tag>_edges.gpkg
             (+ _meta.json)
             BOUNDARIES_DIR / gadm41_PRT_2.shp (Lisbon outline, optional)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             <figure name>_<CAMPAIGN>_<variant_tag>.png (names listed above)

Requirements
------------
    pip install pandas numpy matplotlib geopandas contextily osmnx

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
import sys
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BOUNDARIES_DIR, CACHE_DIR, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
import sw_mapping as swm  # noqa: E402
from clc_mapping import CLC_URBAN_RURAL  # noqa: E402


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

BBOX_MARGIN_DEG: float = 0.02  # same margin as join_sw_standard.py / join_sw_tuktuk.py

# --- Restrict bbox (and therefore the OSM download) to BH-covered rows -------
# W is only used together with H (H/W). Mirrors RESTRICT_TO_VALID_BH of
# join_sw_standard.py, applied here for both campaigns.
RESTRICT_BBOX_TO_VALID_BH: bool = True

# --- On-disk cache for the downloaded OSM network -----------------------------
CACHE_DIR: Path = CACHE_DIR / "osm_network"
FORCE_REFRESH_CACHE: bool = False

# --- GADM boundary (optional decoration on the maps) --------------------------
SHAPEFILE_DIR: Path | None = BOUNDARIES_DIR
LISBON_MUNICIPALITY_NAME: str = "Lisboa"

# --- Output -------------------------------------------------------------------
# A subdirectory named after CAMPAIGN / variant tag is created automatically
# (e.g. street_width_quality_outputs/standard/sd_cleaned/).
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "street_width_quality_outputs"
OUTPUT_DPI: int = 300
SHOW_FIGURES: bool = False

# --- Thresholds ---------------------------------------------------------------
MIN_BIN_POINTS: int = 60  # minimum matched points per COS2023 category

# --- Palette ------------------------------------------------------------------
# width_source categorical palette (specific to this script).
WIDTH_SOURCE_COLORS: dict[str, str] = {
    "osm_tag":           "#2E7D32",  # green      - explicit OSM width= tag (most reliable)
    "lanes_estimate":    "#F4A800",  # amber      - lane-count estimate (medium reliability)
    "highway_default":   "#E65100",  # orange     - highway-type fallback (least reliable)
    "unmatched_too_far": "#E05C5C",  # salmon     - no OSM edge within MAX_MATCH_DISTANCE_M
    "invalid_gps":       "#9E9E9E",  # grey       - excluded upstream (bad GPS / rad_calib)
    "no_bh_coverage":    "#BDBDBD",  # light grey - excluded upstream (no BH value, standard only)
}
# Project CLC Urban / Rural colours.
CLC_GROUP_COLORS: dict[str, str] = {"Urban": "#C0392B", "Rural": "#2E7D32"}
CMAP_COVERAGE = "YlOrRd"
BASEMAP_PROVIDER = ctx.providers.CartoDB.Positron

# width_source values that represent an actual estimated width (as opposed
# to an excluded / unmatched row). Reused across several figures below.
MATCHED_SOURCES: tuple[str, ...] = ("osm_tag", "lanes_estimate", "highway_default")


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

SENSOR_IDS_STANDARD: list[str] = ["j30", "j31", "j32", "j33"]
# Tuk-tuk list includes J29, as in join_sw_tuktuk.py.
SENSOR_IDS_TUKTUK: list[str] = ["j29", "j30", "j31", "j32", "j33"]


def _sensor_ids(campaign: str) -> list[str]:
    """Return the sensor list of the campaign."""
    if campaign == "standard":
        return SENSOR_IDS_STANDARD
    elif campaign == "tuktuk":
        return SENSOR_IDS_TUKTUK
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor identifiers to their CSV input paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    if campaign == "standard":
        return {s: data_dir / f"{s}_satellite_landuse_v2_{suffix}.csv"
                for s in _sensor_ids(campaign)}
    elif campaign == "tuktuk":
        return {s: tuktuk_data_dir / f"{s}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
                for s in _sensor_ids(campaign)}
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _latlon_cols(sensor: str) -> tuple[str, str]:
    """Return the (lat_col, lon_col) names of a sensor.

    Both campaigns use sensor-prefixed columns ('J31 lat' / 'J31 lon'):
    normalize_tuktuk.py gives the tuk-tuk files the standard schema.
    """
    tag = sensor.upper()
    return f"{tag} lat", f"{tag} lon"


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output directory, embedding campaign + variant tag."""
    return base_dir / campaign / variant_tag


SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)


def _out_png(name: str) -> Path:
    """Return the PNG output path of a figure, embedding campaign + variant tag."""
    return OUTPUT_DIR / f"{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"


# ==============================================================================
# 1. BOUNDING BOX DETECTION (restricted to BH-covered rows)
# ==============================================================================

def detect_bbox(sensor_files: dict[str, Path], margin_deg: float,
                restrict_to_bh: bool) -> tuple[float, float, float, float]:
    """
    Return (lon_min, lat_min, lon_max, lat_max) covering all available
    sensors, optionally restricted to rows with a valid
    mean_building_height_m (see module docstring: street width is only
    useful once H is already known).

    Only files that actually exist are included; missing files, and files
    missing 'mean_building_height_m' when restrict_to_bh is True, are
    handled with a printed warning rather than raising.
    """
    lat_min, lat_max = float("inf"), float("-inf")
    lon_min, lon_max = float("inf"), float("-inf")
    found_any = False

    for sensor, path in sensor_files.items():
        lat_col, lon_col = _latlon_cols(sensor)
        if not path.exists():
            print(f"  [warn] sensor file not found, skipped: {path}")
            continue

        header = pd.read_csv(path, nrows=0).columns
        has_bh = "mean_building_height_m" in header
        usecols = [lat_col, lon_col]
        if restrict_to_bh and has_bh:
            usecols.append("mean_building_height_m")
        elif restrict_to_bh and not has_bh:
            print(f"  [warn] {sensor}: 'mean_building_height_m' not found - "
                  f"cannot restrict this file to BH-covered rows; using all "
                  f"valid lat/lon rows instead. Run join_bh_*.py first if "
                  f"this is unexpected.")

        df = pd.read_csv(path, usecols=usecols)
        mask = df[lat_col].notna() & df[lon_col].notna()
        if restrict_to_bh and has_bh:
            n_before = mask.sum()
            mask = mask & df["mean_building_height_m"].notna()
            n_dropped = n_before - mask.sum()
            if n_dropped:
                print(f"  [bbox] {sensor}: dropped {n_dropped:,} row(s) "
                      f"without a valid building-height value")

        if not mask.any():
            continue
        found_any = True
        lat_min = min(lat_min, df.loc[mask, lat_col].min())
        lat_max = max(lat_max, df.loc[mask, lat_col].max())
        lon_min = min(lon_min, df.loc[mask, lon_col].min())
        lon_max = max(lon_max, df.loc[mask, lon_col].max())

    if not found_any:
        raise RuntimeError(
            "No valid lat/lon values found across all sensor files"
            + (" with a usable building-height value." if restrict_to_bh else ".")
        )

    return (lon_min - margin_deg, lat_min - margin_deg,
            lon_max + margin_deg, lat_max + margin_deg)


# ==============================================================================
# 2. CACHED OSM NETWORK LOAD
# ==============================================================================

def _cache_paths(campaign: str, variant_tag: str) -> tuple[Path, Path]:
    """Return the (edges GeoPackage, metadata JSON) cache paths of a campaign / variant."""
    stem = f"{campaign}_{variant_tag}"
    return CACHE_DIR / f"{stem}_edges.gpkg", CACHE_DIR / f"{stem}_meta.json"


def _prepare_edges_for_cache(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Select a lean, GeoPackage-safe subset of columns for the on-disk cache.

    Raw osmnx edge attributes (osmid, highway, lanes, ...) can contain
    Python lists when an edge merges several OSM ways, which GeoPackage
    cannot serialise directly - stringify defensively rather than let
    to_file() raise partway through a run. Only the columns these
    diagnostics actually use are kept.
    """
    keep_candidates = ["width_source", "street_width_m", "highway", "geometry"]
    keep = [c for c in keep_candidates if c in edges_gdf.columns]
    if "geometry" not in keep:
        raise KeyError(
            "edges_gdf returned by sw_mapping.load_street_network has no "
            "'geometry' column - cannot cache or plot it. Check the "
            "sw_mapping.py module's return value."
        )
    out = edges_gdf[keep].copy()
    for col in out.columns:
        if col == "geometry":
            continue
        out[col] = out[col].apply(lambda v: str(v) if isinstance(v, (list, dict)) else v)
    return out


def load_edges_cached(campaign: str, variant_tag: str,
                      bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Return the OSM edges GeoDataFrame (with pre-computed 'street_width_m' /
    'width_source' columns - see sw_mapping.load_street_network) for this
    bbox, using an on-disk cache keyed on CAMPAIGN + variant tag.

    The cache is considered valid only if the stored bbox matches the
    current one (rounded to 4 decimal places, ~11 m) - if the sensor data
    changes (new points -> a different bbox), the cache is automatically
    treated as stale and a fresh download is triggered. Set
    FORCE_REFRESH_CACHE = True to always re-download regardless of cache
    state (e.g. after a sw_mapping.py change).
    """
    edges_path, meta_path = _cache_paths(campaign, variant_tag)
    bbox_rounded = [round(float(v), 4) for v in bbox]

    if not FORCE_REFRESH_CACHE and edges_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("bbox") == bbox_rounded:
                print(f"  [cache] hit -> {edges_path.name} "
                      f"(downloaded {meta.get('downloaded_at', 'unknown date')}, "
                      f"{meta.get('n_edges', '?')} edges)")
                return gpd.read_file(edges_path)
            print(f"  [cache] stale (bbox changed: cached={meta.get('bbox')}, "
                  f"current={bbox_rounded}) -> re-downloading")
        except Exception as exc:
            print(f"  [cache] unreadable cache entry ({exc!s}) -> re-downloading")
    elif FORCE_REFRESH_CACHE:
        print("  [cache] FORCE_REFRESH_CACHE=True -> re-downloading")
    else:
        print("  [cache] miss -> downloading")

    print(f"  Downloading OSM street network via sw_mapping.load_street_network "
          f"(bbox={tuple(bbox_rounded)}) ...")
    _, edges_gdf = swm.load_street_network(bbox)
    edges_gdf = _prepare_edges_for_cache(edges_gdf)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    edges_gdf.to_file(edges_path, driver="GPKG")
    meta_path.write_text(json.dumps({
        "bbox": bbox_rounded,
        "campaign": campaign,
        "variant_tag": variant_tag,
        "n_edges": len(edges_gdf),
        "downloaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"  [cache] written -> {edges_path.name} ({len(edges_gdf):,} edges)")
    return edges_gdf


# ==============================================================================
# 3. COMBINED PER-SENSOR JOIN DATA
# ==============================================================================

def load_sensor_join_data(sensor_files: dict[str, Path]) -> pd.DataFrame:
    """
    Load and concatenate, across sensors, the columns needed by Figs B-I:
    lat, lon, street_width_m, width_source, edge_match_dist_m,
    cos23_n4_label, land_use_clcv, mean_building_height_m. All of these
    already coexist as columns on the same per-sensor file (added upstream
    of clean_data.py by the respective join_*.py steps) - no cross-file
    alignment needed.

    Missing columns are tolerated (e.g. an older FILE_SUFFIX variant
    produced before a given join step): filled with NaN and a warning is
    printed, so individual figures can still run on whatever is available.
    """
    wanted_cols = ["street_width_m", "width_source", "edge_match_dist_m",
                   "cos23_n4_label", "land_use_clcv", "mean_building_height_m"]
    frames: list[pd.DataFrame] = []

    for sensor, path in sensor_files.items():
        lat_col, lon_col = _latlon_cols(sensor)
        if not path.exists():
            print(f"  [warn] sensor file not found, skipped: {path}")
            continue

        header = pd.read_csv(path, nrows=0).columns
        present = [c for c in wanted_cols if c in header]
        missing = [c for c in wanted_cols if c not in header]
        if missing:
            print(f"  [warn] {sensor}: missing columns {missing} "
                  f"(run the corresponding join_*.py step first)")

        usecols = [lat_col, lon_col] + present
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
        for c in missing:
            df[c] = np.nan
        df["sensor"] = sensor
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. Check DATA_DIR/"
            "TUKTUK_DATA_DIR, CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    combined = pd.concat(frames, ignore_index=True)
    if "land_use_clcv" in combined.columns:
        combined["clc_group"] = combined["land_use_clcv"].map(CLC_URBAN_RURAL)
    return combined


def load_lisbon_boundary() -> gpd.GeoDataFrame | None:
    """Load the GADM Level-2 Lisboa municipality polygon, or None if unavailable."""
    if SHAPEFILE_DIR is None:
        return None
    shp_path = SHAPEFILE_DIR / "gadm41_PRT_2.shp"
    if not shp_path.is_file():
        print(f"  [warn] GADM shapefile not found: {shp_path}")
        return None
    gdf = gpd.read_file(shp_path)
    name_col = "NAME_2" if "NAME_2" in gdf.columns else gdf.columns[2]
    lisbon = gdf[gdf[name_col] == LISBON_MUNICIPALITY_NAME]
    return lisbon if not lisbon.empty else None


def try_add_basemap(ax: plt.Axes) -> None:
    """Add an OSM-style basemap; warn and continue if the tiles are unreachable."""
    try:
        ctx.add_basemap(ax, source=BASEMAP_PROVIDER, crs="EPSG:3857", attribution_size=5)
    except Exception as exc:
        print(f"  [warn] basemap fetch failed ({exc!s}); plain background used instead.")


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    """Save the figure at OUTPUT_DPI, optionally show it, then close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ==============================================================================
# 4. CONSOLE DIAGNOSTICS
# ==============================================================================

def print_coverage_summary(edges_gdf: gpd.GeoDataFrame, join_df: pd.DataFrame) -> None:
    """Print the network-level and point-level width_source composition and width quantiles."""
    print("\n" + "-" * 78)
    print("STREET-WIDTH COVERAGE SUMMARY")
    print("-" * 78)

    if "width_source" in edges_gdf.columns:
        print("\n  Network-level composition (all OSM edges in the restricted bbox):")
        print((edges_gdf["width_source"].value_counts(normalize=True) * 100)
             .round(1).to_string())
    else:
        print("\n  [warn] 'width_source' not found on edges_gdf - "
              "check sw_mapping.load_street_network's return value.")

    if "width_source" in join_df.columns:
        print("\n  Point-level composition (actual sensor-route join):")
        print((join_df["width_source"].value_counts(normalize=True) * 100)
             .round(1).to_string())

    if "street_width_m" in join_df.columns and "width_source" in join_df.columns:
        matched = join_df.loc[join_df["width_source"].isin(MATCHED_SOURCES), "street_width_m"]
        if len(matched) > 0:
            print("\n  street_width_m (matched points only):")
            print(matched.describe(percentiles=[.05, .25, .5, .75, .95]).to_string())
    print("-" * 78)


# ==============================================================================
# 5. FIG A - NETWORK MAP BY width_source (edge-level)
# ==============================================================================

def fig_network_by_source_map(edges_gdf: gpd.GeoDataFrame,
                              lisbon_gdf: gpd.GeoDataFrame | None,
                              out_path: Path) -> None:
    """Fig A — OSM street network coloured by width_source."""
    if "width_source" not in edges_gdf.columns:
        print("  [WARN] Fig A: 'width_source' column not found on edges_gdf; skipping.")
        return

    gdf = edges_gdf.copy()
    if gdf.crs is None:
        print("  [warn] edges_gdf has no CRS set; assuming EPSG:3763 "
              "(Portugal TM06, per sw_mapping.py's documented projection).")
        gdf = gdf.set_crs("EPSG:3763")
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 10))
    present_sources = [s for s in WIDTH_SOURCE_COLORS if s in gdf["width_source"].unique()]
    for source in present_sources:
        sub = gdf[gdf["width_source"] == source]
        sub.plot(ax=ax, color=WIDTH_SOURCE_COLORS[source], linewidth=1.1,
                 label=f"{source} (n={len(sub):,})", zorder=3)
    other = gdf[~gdf["width_source"].isin(present_sources)]
    if not other.empty:
        other.plot(ax=ax, color="#000000", linewidth=1.1,
                  label=f"other/unlabelled (n={len(other):,})", zorder=3)

    if lisbon_gdf is not None:
        lisbon_gdf.to_crs(epsg=3857).boundary.plot(
            ax=ax, color="#111111", linewidth=1.3, zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_title("OSM street network - width provenance (width_source)",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"{len(gdf):,} edges in the bbox (restricted to BH-covered sensor "
        "coverage - see module docstring). Green = explicit OSM width tag "
        "(most reliable); amber = lane-count estimate; orange = "
        "highway-type fallback (least reliable).",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 6. FIG B - JOIN COVERAGE MAP BY width_source (point-level, sensor route)
# ==============================================================================

def fig_join_coverage_by_source_map(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig B — measurement points coloured by the joined width_source."""
    df = join_df.dropna(subset=["lat", "lon", "width_source"])
    if df.empty:
        print("  [WARN] Fig B: no rows with lat/lon + width_source; skipping.")
        return

    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 10))
    for source, colour in WIDTH_SOURCE_COLORS.items():
        sub = pts[pts["width_source"] == source]
        if sub.empty:
            continue
        ax.scatter(sub.geometry.x, sub.geometry.y, c=colour, s=6, alpha=0.7,
                  linewidths=0, label=f"{source} (n={len(sub):,})", zorder=3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    try_add_basemap(ax)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9, markerscale=2)
    ax.set_title("Sensor route - street-width join coverage by source",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"{len(df):,} measurement points - shows WHERE each width-source "
        "tier lands along the actual driven route, as opposed to the "
        "network's raw composition in Fig A.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 7. FIG C - STREET-WIDTH HISTOGRAM BY SOURCE
# ==============================================================================

def fig_width_histogram(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig C — histogram of street_width_m (matched rows), by width_source."""
    df = join_df.dropna(subset=["street_width_m", "width_source"])
    df = df[df["width_source"].isin(MATCHED_SOURCES)]
    if df.empty:
        print("  [WARN] Fig C: no matched rows with a valid street_width_m; skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, df["street_width_m"].quantile(0.99), 50)
    sources = [s for s in MATCHED_SOURCES if s in df["width_source"].unique()]
    ax.hist(
        [df.loc[df["width_source"] == s, "street_width_m"] for s in sources],
        bins=bins, stacked=True, color=[WIDTH_SOURCE_COLORS[s] for s in sources],
        label=[f"{s} (n={(df['width_source'] == s).sum():,})" for s in sources],
        alpha=0.9,
    )
    ax.set_xlabel("street_width_m [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_title("Street-width distribution (matched rows), by source",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    fig.text(
        0.5, -0.02,
        f"mean = {df['street_width_m'].mean():.2f} m, "
        f"median = {df['street_width_m'].median():.2f} m "
        "(x-axis clipped at the 99th percentile for display).",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 8. FIG D - MEAN WIDTH BY COS2023 CATEGORY
# ==============================================================================

def fig_width_by_cos23_category(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig D — mean street width per COS2023 level-4 category."""
    df = join_df.dropna(subset=["cos23_n4_label", "street_width_m"])
    df = df[df["width_source"].isin(MATCHED_SOURCES)]
    if df.empty:
        print("  [WARN] Fig D: no matched rows with cos23_n4_label; skipping.")
        return

    summary = (
        df.groupby("cos23_n4_label")["street_width_m"]
        .agg(mean="mean", std="std", n="count")
        .query("n >= @MIN_BIN_POINTS")
        .sort_values("mean")
    )
    if summary.empty:
        print(f"  [WARN] Fig D: no COS2023 category reaches {MIN_BIN_POINTS} "
              "matched points; skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(summary))))
    ax.barh(summary.index, summary["mean"], xerr=summary["std"],
           color="#1565C0", alpha=0.85, capsize=3)
    ax.set_xlabel("Mean street width [m]", fontsize=10, fontweight="bold")
    ax.set_title("Mean street width by COS2023 category", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.text(
        0.5, -0.02 - 0.01 * len(summary),
        f"Categories with >= {MIN_BIN_POINTS} matched points only (osm_tag / "
        "lanes_estimate / highway_default).",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 9. FIG E - width_source COMPOSITION BY COS2023 CATEGORY
# ==============================================================================

def fig_source_composition_by_cos23(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig E — width_source composition per COS2023 category."""
    df = join_df.dropna(subset=["cos23_n4_label", "width_source"]).copy()
    if df.empty:
        print("  [WARN] Fig E: no rows with both cos23_n4_label and width_source; skipping.")
        return

    counts = df.groupby(["cos23_n4_label", "width_source"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    counts = counts.loc[totals[totals >= MIN_BIN_POINTS].index]
    if counts.empty:
        print(f"  [WARN] Fig E: no category reaches {MIN_BIN_POINTS} points; skipping.")
        return
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    sort_col = "highway_default" if "highway_default" in shares.columns else shares.columns[0]
    shares = shares.sort_values(sort_col)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(shares))))
    left = np.zeros(len(shares))
    for source in (*MATCHED_SOURCES, "unmatched_too_far"):
        if source not in shares.columns:
            continue
        ax.barh(shares.index, shares[source], left=left,
               color=WIDTH_SOURCE_COLORS[source], label=source, alpha=0.9)
        left += shares[source].values
    ax.set_xlabel("Share of points [%]", fontsize=10, fontweight="bold")
    ax.set_title("width_source composition by COS2023 category",
                fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.text(
        0.5, -0.02 - 0.01 * len(shares),
        f"Categories with >= {MIN_BIN_POINTS} points only. Categories dominated "
        "by orange (highway_default) rely most heavily on the least-precise "
        "width estimate.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 10. FIG F - UNMATCHED (JOIN-FAILURE) RATE BY COS2023 CATEGORY
# ==============================================================================

def fig_unmatched_rate_by_cos23(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig F — 'unmatched_too_far' rate per COS2023 category."""
    df = join_df.dropna(subset=["cos23_n4_label", "width_source"]).copy()
    if df.empty:
        print("  [WARN] Fig F: no rows with both cos23_n4_label and width_source; skipping.")
        return
    df["is_unmatched"] = df["width_source"] == "unmatched_too_far"

    summary = (
        df.groupby("cos23_n4_label")
        .agg(unmatched_rate=("is_unmatched", "mean"), n=("is_unmatched", "count"))
        .query("n >= @MIN_BIN_POINTS")
        .sort_values("unmatched_rate")
    )
    if summary.empty:
        print(f"  [WARN] Fig F: no category with >= {MIN_BIN_POINTS} points; skipping.")
        return

    colours = ["#E05C5C" if r > 0.30 else "#F4A800" if r > 0.10 else "#2E7D32"
              for r in summary["unmatched_rate"]]
    overall_rate = df["is_unmatched"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(summary))))
    bars = ax.barh(summary.index, summary["unmatched_rate"] * 100, color=colours, alpha=0.85)
    ax.axvline(overall_rate, color="#1565C0", linewidth=1.2, linestyle="--",
              label=f"Overall rate ({overall_rate:.0f}%)")
    ax.set_xlabel("Share of points unmatched (no OSM edge within range) [%]",
                 fontsize=10, fontweight="bold")
    ax.set_title("Street-width join-failure rate by COS2023 category",
                fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
               f"n={row['n']:,}", va="center", fontsize=7, color="#555555")
    fig.text(
        0.5, -0.02,
        "Green (<10%) = well covered by the OSM network. Red (>30%) = "
        "structural gap for this land-use type. Blue line = overall average.",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 11. FIG G - EDGE-MATCH-DISTANCE DISTRIBUTION (match QUALITY)
# ==============================================================================

def fig_match_distance_distribution(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig G — histogram + CDF of edge_match_dist_m."""
    d = join_df["edge_match_dist_m"].dropna() if "edge_match_dist_m" in join_df.columns else pd.Series(dtype=float)
    d = d[d >= 0]
    if d.empty:
        print("  [WARN] Fig G: no valid edge_match_dist_m values; skipping.")
        return

    max_match_dist = getattr(swm, "MAX_MATCH_DISTANCE_M", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(d.clip(upper=d.quantile(0.99)), bins=50, color="#1565C0", alpha=0.85)
    if max_match_dist is not None:
        ax1.axvline(max_match_dist, color="#E05C5C", linestyle="--", linewidth=1.3,
                   label=f"MAX_MATCH_DISTANCE_M = {max_match_dist:.0f} m")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("edge_match_dist_m [m]", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax1.set_title("Point-to-nearest-edge distance", fontsize=11, fontweight="bold")
    ax1.grid(linestyle="--", alpha=0.3)

    sorted_d = np.sort(d.values)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax2.plot(sorted_d, cdf * 100, color="#1565C0", linewidth=1.5)
    if max_match_dist is not None:
        ax2.axvline(max_match_dist, color="#E05C5C", linestyle="--", linewidth=1.3)
    ax2.set_xlabel("edge_match_dist_m [m]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cumulative share of matched points [%]", fontsize=10, fontweight="bold")
    ax2.set_title("Cumulative distribution", fontsize=11, fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.3)

    fig.suptitle("Street-width match quality (edge_match_dist_m)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 12. FIG H - STREET WIDTH vs BUILDING HEIGHT
# ==============================================================================

def fig_width_vs_building_height(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig H — street width vs mean building height (hexbin)."""
    df = join_df.dropna(subset=["street_width_m", "mean_building_height_m"])
    df = df[df["width_source"].isin(MATCHED_SOURCES)]
    if df.empty:
        print("  [WARN] Fig H: no rows with both street_width_m and "
              "mean_building_height_m; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(df["street_width_m"], df["mean_building_height_m"],
                   gridsize=40, cmap=CMAP_COVERAGE, mincnt=1)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label("Measurement points per hex cell", fontsize=9)

    corr = df["street_width_m"].corr(df["mean_building_height_m"])
    ax.set_xlabel("Street width W [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean building height H [m]", fontsize=10, fontweight="bold")
    ax.set_title("Street width vs. building height", fontsize=12.5, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    fig.text(
        0.5, -0.02,
        f"Pearson r = {corr:.2f} across {len(df):,} points. A negative "
        "correlation would corroborate the reported H/W anti-correlation "
        "pattern in Lisbon (taller buildings on narrower streets).",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# 13. FIG I - STREET WIDTH BY CLC URBAN/RURAL GROUP
# ==============================================================================

def fig_width_by_clc_group(join_df: pd.DataFrame, out_path: Path) -> None:
    """Fig I — street width by CLC Urban / Rural group (boxplot)."""
    if "clc_group" not in join_df.columns:
        print("  [WARN] Fig I: 'clc_group' not available (needs land_use_clcv); skipping.")
        return
    df = join_df.dropna(subset=["clc_group", "street_width_m"])
    df = df[df["width_source"].isin(MATCHED_SOURCES)]
    df = df[df["clc_group"].isin(["Urban", "Rural"])]
    if df.empty:
        print("  [WARN] Fig I: no matched rows with a CLC Urban/Rural group; skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    data = [df.loc[df["clc_group"] == g, "street_width_m"] for g in ("Urban", "Rural")]
    bp = ax.boxplot(data, labels=["Urban", "Rural"], patch_artist=True, widths=0.5)
    for patch, group in zip(bp["boxes"], ("Urban", "Rural")):
        patch.set_facecolor(CLC_GROUP_COLORS[group])
        patch.set_alpha(0.75)
    ax.set_ylabel("Street width [m]", fontsize=10, fontweight="bold")
    ax.set_title("Street width by CLC Urban/Rural group", fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    y_bottom = ax.get_ylim()[0]
    for i, g in enumerate(("Urban", "Rural"), start=1):
        n = (df["clc_group"] == g).sum()
        ax.text(i, y_bottom, f"n={n:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Detect the bbox, load the (cached) OSM network and sensor join data, draw Figs A–I."""
    print("=" * 78)
    print("Street-width data-quality / coverage diagnostic")
    print(f"  CAMPAIGN                  : {CAMPAIGN}")
    print(f"  Data variant               : {_VARIANT_TAG}")
    print(f"  RESTRICT_BBOX_TO_VALID_BH  : {RESTRICT_BBOX_TO_VALID_BH}")
    print(f"  CACHE_DIR                  : {CACHE_DIR}")
    print(f"  FORCE_REFRESH_CACHE        : {FORCE_REFRESH_CACHE}")
    print(f"  OUTPUT_DIR                 : {OUTPUT_DIR}")
    print("=" * 78)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/12] Detecting bounding box (restricted to BH-covered rows)...")
    bbox = detect_bbox(SENSOR_FILES, BBOX_MARGIN_DEG, RESTRICT_BBOX_TO_VALID_BH)
    print(f"  bbox (W, S, E, N) = {tuple(round(float(v), 4) for v in bbox)}")

    print("\n[2/12] Loading OSM street network (cached)...")
    edges_gdf = load_edges_cached(CAMPAIGN, _VARIANT_TAG, bbox)
    print(f"  {len(edges_gdf):,} edges loaded")

    print("\n[3/12] Loading per-sensor join data...")
    join_df = load_sensor_join_data(SENSOR_FILES)
    print(f"  {len(join_df):,} rows loaded across {join_df['sensor'].nunique()} sensor(s)")

    print_coverage_summary(edges_gdf, join_df)

    lisbon_gdf = load_lisbon_boundary()

    print("\n[4/12] Fig A - network map by width_source...")
    fig_network_by_source_map(edges_gdf, lisbon_gdf, _out_png("street_network_by_source_map"))

    print("\n[5/12] Fig B - join coverage map by width_source...")
    fig_join_coverage_by_source_map(join_df, _out_png("join_coverage_by_source_map"))

    print("\n[6/12] Fig C - street-width histogram by source...")
    fig_width_histogram(join_df, _out_png("street_width_histogram"))

    print("\n[7/12] Fig D - mean width by COS2023 category...")
    fig_width_by_cos23_category(join_df, _out_png("mean_width_by_cos23_category"))

    print("\n[8/12] Fig E - width_source composition by COS2023 category...")
    fig_source_composition_by_cos23(join_df, _out_png("width_source_composition_by_cos23"))

    print("\n[9/12] Fig F - unmatched rate by COS2023 category...")
    fig_unmatched_rate_by_cos23(join_df, _out_png("unmatched_rate_by_cos23"))

    print("\n[10/12] Fig G - match-distance distribution...")
    fig_match_distance_distribution(join_df, _out_png("edge_match_distance_distribution"))

    print("\n[11/12] Fig H - street width vs building height...")
    fig_width_vs_building_height(join_df, _out_png("street_width_vs_building_height"))

    print("\n[12/12] Fig I - street width by CLC Urban/Rural group...")
    fig_width_by_clc_group(join_df, _out_png("width_by_clc_group"))

    print("\nDone. All figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
