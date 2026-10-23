"""
================================================================================
Street-Width Mapping — Shared Module
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Nearest-edge join between VIPV measurement points and the OpenStreetMap (OSM)
street network (via osmnx), giving a per-point street width for the W side
of the urban-canyon H/W ratio (building heights: bh_mapping.py). Used by
join_sw_standard.py, join_sw_tuktuk.py and the urban-geometry scripts.

--------------------------------------------------------------------------------
Why OSM rather than building footprints
--------------------------------------------------------------------------------
GlobalBuildingAtlas gives reliable building heights, but its footprints are
not facade-accurate enough to measure a street width from the gap between
opposite buildings (diagnose_building_coverage.py). The OSM network geometry
and its 'width' / 'lanes' tags are a more direct source for the width.

--------------------------------------------------------------------------------
What "street width" means here
--------------------------------------------------------------------------------
The estimate is a CARRIAGEWAY width (paved road), in order of preference:
    1) OSM 'width' tag                       -> width_source 'osm_tag'
    2) 'lanes' tag x LANE_WIDTH_M            -> 'lanes_estimate'
    3) default width of the highway type     -> 'highway_default'
It is NOT the facade-to-facade canyon width of Brito et al. (2024), which
also includes sidewalks and setbacks: the carriageway width likely
UNDERESTIMATES W, so the SVF loss derived from it is an upper bound.
SIDEWALK_ALLOWANCE_M is an explicit, editable correction for this (0.0 by
default, raw carriageway width).

--------------------------------------------------------------------------------
Matching
--------------------------------------------------------------------------------
Each valid point is matched to its nearest edge (osmnx.distance.nearest_edges)
in EPSG:3763 (ETRS89 / Portugal TM06). Matches farther than
MAX_MATCH_DISTANCE_M (parking area, courtyard, OSM coverage gap) are
discarded: street_width_m = NaN, width_source = 'unmatched_too_far'. Every
matched point keeps its width_source, so any result can report how much of
it rests on a tagged width rather than a fallback. Rows with missing lat/lon
keep NaN (not computed).

--------------------------------------------------------------------------------
Network access
--------------------------------------------------------------------------------
load_street_network() downloads the network through the Overpass API and
needs internet access. The self-test uses a small synthetic graph instead.

Self-test
---------
    python sw_mapping.py      (synthetic graph, no network access needed)

Requirements
------------
    pip install numpy pandas geopandas networkx "osmnx>=2.0" shapely
    (load_street_network uses the osmnx 2.x signature graph_from_bbox(bbox, ...))

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.1
================================================================================
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Projected CRS and OSM network type ----------------------------------------
METRIC_CRS: str = "EPSG:3763"  # ETRS89 / Portugal TM06, same as bh_mapping.py
NETWORK_TYPE: str = "drive"    # osmnx network type: "all", "drive", "walk", ...

# --- Carriageway width estimate [m], in order of preference --------------------
#   1) OSM 'width' tag, if present and parseable (metres)
#   2) 'lanes' tag x LANE_WIDTH_M
#   3) DEFAULT_WIDTH_BY_HIGHWAY.get(highway_type, GLOBAL_DEFAULT_WIDTH_M)
LANE_WIDTH_M: float = 3.0  # typical urban lane width assumption (2.75-3.5 m range)
DEFAULT_WIDTH_BY_HIGHWAY: dict[str, float] = {
    "motorway": 12.0, "motorway_link": 6.0,
    "trunk": 11.0, "trunk_link": 6.0,
    "primary": 9.0, "primary_link": 5.5,
    "secondary": 8.0, "secondary_link": 5.0,
    "tertiary": 7.0, "tertiary_link": 5.0,
    "unclassified": 6.0,
    "residential": 6.0,
    "living_street": 5.0,
    "service": 4.0,
    "pedestrian": 4.0,
}
GLOBAL_DEFAULT_WIDTH_M: float = 6.0  # highway type not in the dict above

# --- Carriageway-to-canyon correction [m] (module docstring) -------------------
# 0.0 keeps the raw carriageway width.
SIDEWALK_ALLOWANCE_M: float = 0.0

# --- Maximum point-to-edge distance [m] -----------------------------------------
# Farther matches are discarded (point likely off the mapped network).
MAX_MATCH_DISTANCE_M: float = 30.0


# ==============================================================================
# WIDTH TAG PARSING
# ==============================================================================

def _parse_width_value(raw) -> float | None:
    """
    Parse an OSM 'width' tag value into a single float [m].

    Accepted forms: a number, a numeric string (optionally ending in 'm'), a
    ';'- or ','-separated list of values, or a Python list of any of these
    (osmnx simplification merges consecutive ways). Lists are averaged.
    Missing or unparseable values return None (never guessed).
    """
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, list):
        vals = [v for v in (_parse_width_value(x) for x in raw) if v is not None]
        return float(np.mean(vals)) if vals else None
    if isinstance(raw, str):
        vals = []
        for part in raw.replace(",", ";").split(";"):
            part = part.strip().rstrip("m").strip()
            try:
                vals.append(float(part))
            except ValueError:
                continue
        return float(np.mean(vals)) if vals else None
    return None


def _parse_lane_count(raw) -> float | None:
    """Same parsing logic as _parse_width_value, applied to the 'lanes' tag."""
    return _parse_width_value(raw)


def _first_highway_tag(raw) -> str | None:
    """Return the first 'highway' value when osmnx simplification merged
    edges of different types into a list."""
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


def estimate_edge_width_m(edge_row: pd.Series) -> tuple[float, str]:
    """
    Estimate the carriageway width [m] and its provenance tag ('osm_tag',
    'lanes_estimate' or 'highway_default') for one edge row of the
    GeoDataFrame returned by osmnx.graph_to_gdfs.
    """
    w = _parse_width_value(edge_row.get("width"))
    if w is not None and w > 0:
        return w, "osm_tag"

    lanes = _parse_lane_count(edge_row.get("lanes"))
    if lanes is not None and lanes > 0:
        return lanes * LANE_WIDTH_M, "lanes_estimate"

    highway = _first_highway_tag(edge_row.get("highway"))
    default_w = DEFAULT_WIDTH_BY_HIGHWAY.get(highway, GLOBAL_DEFAULT_WIDTH_M)
    return default_w, "highway_default"


# ==============================================================================
# NETWORK LOADING  (requires internet access)
# ==============================================================================

def load_street_network(
    bbox: tuple[float, float, float, float],
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
    """
    Download the OSM street network within `bbox` (west, south, east, north
    in EPSG:4326), project it to METRIC_CRS, and attach a per-edge
    'street_width_m' / 'width_source' estimate.

    Requires internet access to the Overpass API (via osmnx).
    """
    print(f"Downloading OSM street network for bbox {bbox} "
          f"(network_type='{NETWORK_TYPE}') ...")
    graph = ox.graph_from_bbox(bbox, network_type=NETWORK_TYPE)
    graph_proj = ox.project_graph(graph, to_crs=METRIC_CRS)
    edges = ox.graph_to_gdfs(graph_proj, nodes=False)
    print(f"  {len(edges):,} street edges loaded")

    widths_sources = [estimate_edge_width_m(row) for _, row in edges.iterrows()]
    edges["street_width_m"] = [w + SIDEWALK_ALLOWANCE_M for w, _ in widths_sources]
    edges["width_source"] = [s for _, s in widths_sources]

    n_by_source = edges["width_source"].value_counts()
    print(f"  width provenance across edges: {n_by_source.to_dict()}")
    return graph_proj, edges


# ==============================================================================
# NEAREST-EDGE JOIN
# ==============================================================================

def join_street_width_nearest(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    graph_proj: nx.MultiDiGraph,
    edges_gdf: gpd.GeoDataFrame,
    max_match_dist_m: float = MAX_MATCH_DISTANCE_M,
) -> pd.DataFrame:
    """
    For each row in `df`, find the nearest OSM street edge and assign its
    'street_width_m' / 'width_source', plus the point-to-edge match distance
    ('edge_match_dist_m'). Matches beyond `max_match_dist_m` are discarded
    (street_width_m -> NaN, width_source -> 'unmatched_too_far').

    Rows with missing lat/lon keep NaN / None in the new columns
    (not computed).
    """
    valid_mask = df[lat_col].notna() & df[lon_col].notna()

    points_gdf = gpd.GeoDataFrame(
        df.loc[valid_mask, [lat_col, lon_col]].copy(),
        geometry=gpd.points_from_xy(
            df.loc[valid_mask, lon_col], df.loc[valid_mask, lat_col]
        ),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    x = points_gdf.geometry.x.to_numpy()
    y = points_gdf.geometry.y.to_numpy()
    nearest_edges, dists = ox.distance.nearest_edges(graph_proj, x, y, return_dist=True)

    edge_lookup = edges_gdf[["street_width_m", "width_source"]]

    widths = np.empty(len(nearest_edges), dtype=float)
    sources = np.empty(len(nearest_edges), dtype=object)
    for i, uvk in enumerate(nearest_edges):
        row = edge_lookup.loc[tuple(uvk)]
        widths[i] = row["street_width_m"]
        sources[i] = row["width_source"]

    result = df.copy()
    for col in ["street_width_m", "edge_match_dist_m"]:
        result[col] = np.nan
    result["width_source"] = pd.array([None] * len(df), dtype=object)

    valid_idx = df.loc[valid_mask].index
    result.loc[valid_idx, "street_width_m"] = widths
    result.loc[valid_idx, "width_source"] = sources
    result.loc[valid_idx, "edge_match_dist_m"] = dists

    too_far = result["edge_match_dist_m"] > max_match_dist_m
    result.loc[too_far, "street_width_m"] = np.nan
    result.loc[too_far, "width_source"] = "unmatched_too_far"

    result["street_width_m"] = pd.to_numeric(result["street_width_m"], errors="coerce")
    result["edge_match_dist_m"] = pd.to_numeric(result["edge_match_dist_m"], errors="coerce")
    return result



# ==============================================================================
# SELF-TEST  (synthetic graph, no network access needed)
# ==============================================================================

def _build_synthetic_graph() -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
    """
    Two street segments in a local metric frame (no reprojection needed):
      edge A: a narrow residential street with an explicit width tag (5 m),
              running along y=0 from x=0 to x=100.
      edge B: a wide primary road with a 'lanes' tag (no width tag),
              running along x=200 from y=0 to y=100 (3 lanes -> 9 m at
              LANE_WIDTH_M=3.0).
    """
    from shapely.geometry import LineString

    graph = nx.MultiDiGraph(crs=METRIC_CRS)
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=100.0, y=0.0)
    graph.add_node(3, x=200.0, y=0.0)
    graph.add_node(4, x=200.0, y=100.0)

    graph.add_edge(1, 2, key=0, width="5",
                    highway="residential",
                    geometry=LineString([(0, 0), (100, 0)]))
    graph.add_edge(3, 4, key=0, lanes="3",
                    highway="primary",
                    geometry=LineString([(200, 0), (200, 100)]))

    edges = ox.graph_to_gdfs(graph, nodes=False)
    widths_sources = [estimate_edge_width_m(row) for _, row in edges.iterrows()]
    edges["street_width_m"] = [w + SIDEWALK_ALLOWANCE_M for w, _ in widths_sources]
    edges["width_source"] = [s for _, s in widths_sources]
    return graph, edges


def _self_test() -> None:
    """Width estimate and nearest-edge matching on the synthetic graph."""
    graph, edges = _build_synthetic_graph()
    print("Synthetic edges:")
    print(edges[["highway", "width", "lanes", "street_width_m", "width_source"]]
          if "width" in edges.columns and "lanes" in edges.columns
          else edges[["highway", "street_width_m", "width_source"]])

    # Points given directly in the synthetic metric frame: nearest_edges is
    # called directly, bypassing the EPSG:4326 input of
    # join_street_width_nearest().
    x = np.array([50.0, 200.0, 1000.0])   # point 3 is far from both edges
    y = np.array([1.0, 50.0, 1000.0])
    nearest, dists = ox.distance.nearest_edges(graph, x, y, return_dist=True)

    lookup = edges[["street_width_m", "width_source"]]
    widths = [lookup.loc[tuple(uvk), "street_width_m"] for uvk in nearest]
    sources = [lookup.loc[tuple(uvk), "width_source"] for uvk in nearest]

    print(f"\nPoint 1 (near edge A, residential, width tag) "
          f"-> width={widths[0]}, source={sources[0]}, dist={dists[0]:.2f} m")
    print(f"Point 2 (near edge B, primary, lanes tag)      "
          f"-> width={widths[1]}, source={sources[1]}, dist={dists[1]:.2f} m")
    print(f"Point 3 (far from both edges)                  "
          f"-> nearest dist={dists[2]:.1f} m "
          f"({'would be flagged unmatched_too_far' if dists[2] > MAX_MATCH_DISTANCE_M else 'within match distance'})")

    assert widths[0] == 5.0, "Point 1 should get the explicit width tag (5 m)"
    assert sources[0] == "osm_tag"
    assert widths[1] == 3.0 * 3.0, "Point 2 should get 3 lanes x LANE_WIDTH_M"
    assert sources[1] == "lanes_estimate"
    assert dists[2] > MAX_MATCH_DISTANCE_M, "Point 3 should be far enough to be flagged"
    print("\nSelf-test passed.")


if __name__ == "__main__":
    _self_test()