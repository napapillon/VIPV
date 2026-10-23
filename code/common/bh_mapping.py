"""
================================================================================
Building-Height Mapping — Shared Module
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Buffer join between VIPV measurement points and the building-height dataset
(GlobalBuildingAtlas, Lisbon metropolitan area clip — BUILDINGS_PATH in
config.py). Used by join_bh_standard.py, join_bh_tuktuk.py and the urban-
geometry scripts.

--------------------------------------------------------------------------------
Method
--------------------------------------------------------------------------------
Unlike the land-cover joins (point-in-polygon, one match per point), this is a
BUFFER join: every building geometry intersecting a BUFFER_RADIUS_M disc
around the point is considered nearby, and the heights of these buildings are
aggregated into three columns:

    mean_building_height_m  : mean height of the nearby buildings [m]
    max_building_height_m   : height of the tallest nearby building [m]
    n_buildings_nearby      : number of nearby buildings

A buffer rather than the single nearest building is used because the shading
of a street canyon depends on the buildings on BOTH sides of the street. This
gives a continuous proxy of the height of the surrounding built environment,
without the street direction and width needed for a full H/W ratio
(sw_mapping.py provides the width).

n_buildings_nearby = 0 is a genuine result (open space, park, wide square).
Rows with missing lat/lon keep NaN in the three columns (not computed).

--------------------------------------------------------------------------------
Column detection
--------------------------------------------------------------------------------
The height column of the source file is detected from HEIGHT_COL_CANDIDATES
(first match) and renamed '_height_m'. load_building_heights() prints the
detected columns and the full column list; if detection fails, add the real
column name to HEIGHT_COL_CANDIDATES. The footprint column is detected for
information only (not used).

--------------------------------------------------------------------------------
CRS
--------------------------------------------------------------------------------
Buffering by a distance in metres requires a projected metric CRS: points and
buildings are projected to EPSG:3763 (ETRS89 / Portugal TM06), the official
Portuguese grid, also used by COS2023.

Self-test
---------
    python bh_mapping.py      (synthetic data, no building file needed)

Requirements
------------
    pip install numpy pandas geopandas shapely

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Projected CRS for metric buffering ----------------------------------------
METRIC_CRS: str = "EPSG:3763"  # ETRS89 / Portugal TM06

# --- Buffer radius [m] ----------------------------------------------------------
# 30 m left 53.5 % of the COS 'predominantly vertical residential' points
# without any building (inner courtyards, set-back blocks of the 1960s-80s
# housing estates: Telheiras, Chelas, Lumiar); 50 m brings this down to
# 12.3 % while remaining a local street-canyon scale
# (diagnose_building_coverage.py).
BUFFER_RADIUS_M: float = 50.0

# --- Candidate column names (first match is used) ------------------------------
HEIGHT_COL_CANDIDATES: list[str] = [
    "height", "Height", "HEIGHT", "height_m", "building_height", "h_mean",
]
FOOTPRINT_COL_CANDIDATES: list[str] = [
    "footprint_m2", "footprint", "area_m2", "area", "Area",
]


def _detect_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first entry of `candidates` present in `columns`, else None."""
    for c in candidates:
        if c in columns:
            return c
    return None


# ==============================================================================
# LOADING
# ==============================================================================

def load_building_heights(
    geojson_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """
    Load the building GeoJSON, optionally restricted at read time to `bbox`
    (xmin, ymin, xmax, ymax in EPSG:4326), since the file covers the whole
    Lisbon metropolitan area.

    Prints the detected height / footprint columns and the full column list,
    and warns if the loaded bounds do not overlap `bbox` (CRS mismatch).
    The height column is renamed '_height_m'.
    """
    print(f"Loading building footprints from {geojson_path} ...")
    gdf = gpd.read_file(geojson_path, bbox=bbox) if bbox is not None \
        else gpd.read_file(geojson_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    print(f"  {len(gdf):,} building footprints loaded")
    print(f"  CRS: {gdf.crs}")
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"  bounds: ({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f})")
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        print(f"  requested bbox: ({bx0:.4f}, {by0:.4f}, {bx1:.4f}, {by1:.4f})")
        if not (minx <= bx1 and maxx >= bx0 and miny <= by1 and maxy >= by0):
            print("  [WARN] loaded bounds do not overlap the requested bbox at "
                  "all - check the CRS above, this strongly suggests a "
                  "coordinate system mismatch rather than a real data gap.")
    print(f"  columns found: {list(gdf.columns)}")

    height_col = _detect_column(list(gdf.columns), HEIGHT_COL_CANDIDATES)
    footprint_col = _detect_column(list(gdf.columns), FOOTPRINT_COL_CANDIDATES)

    if height_col is None:
        raise RuntimeError(
            "Could not auto-detect a height column. Add the actual column "
            "name to HEIGHT_COL_CANDIDATES at the top of this module - the "
            f"available columns are: {list(gdf.columns)}"
        )
    print(f"  detected height column     : '{height_col}'")
    print(f"  detected footprint column  : "
          f"{repr(footprint_col) if footprint_col else '(not found - optional, not used yet)'}")

    gdf = gdf.rename(columns={height_col: "_height_m"})
    return gdf


# ==============================================================================
# BUFFER JOIN
# ==============================================================================

def join_building_height_buffer(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    buildings_gdf: gpd.GeoDataFrame,
    radius_m: float = BUFFER_RADIUS_M,
) -> pd.DataFrame:
    """
    For each row of `df`, find every building within `radius_m` metres and
    aggregate their heights into 'mean_building_height_m',
    'max_building_height_m' and 'n_buildings_nearby'.

    n_buildings_nearby = 0 is a genuine result (open space, park, wide
    square). Rows with missing lat/lon keep NaN in the three columns
    (not computed, distinct from "computed as 0").
    """
    valid_mask = df[lat_col].notna() & df[lon_col].notna()

    points_gdf = gpd.GeoDataFrame(
        df.loc[valid_mask, [lat_col, lon_col]].copy(),
        geometry=gpd.points_from_xy(
            df.loc[valid_mask, lon_col], df.loc[valid_mask, lat_col]
        ),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    buildings_metric = buildings_gdf.to_crs(METRIC_CRS)

    buffers_gdf = points_gdf.copy()
    buffers_gdf["geometry"] = buffers_gdf.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        buffers_gdf, buildings_metric[["_height_m", "geometry"]],
        how="left", predicate="intersects",
    )

    agg = joined.groupby(joined.index)["_height_m"].agg(
        mean_building_height_m="mean",
        max_building_height_m="max",
        n_buildings_nearby="count",
    )

    result = df.copy()
    for col in ["mean_building_height_m", "max_building_height_m", "n_buildings_nearby"]:
        result[col] = np.nan
    # Index-aligned assignment (not positional .values): agg.index is the
    # original df index of the valid rows, whatever the row order.
    result.loc[agg.index, "mean_building_height_m"] = agg["mean_building_height_m"]
    result.loc[agg.index, "max_building_height_m"] = agg["max_building_height_m"]
    result.loc[agg.index, "n_buildings_nearby"] = agg["n_buildings_nearby"]

    # Re-cast to numeric: a partial .loc assignment can leave an object dtype
    # in some pandas versions, which breaks numeric summaries (.describe()).
    for col in ["mean_building_height_m", "max_building_height_m", "n_buildings_nearby"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result



# ==============================================================================
# SELF-TEST  (synthetic data, no real building file needed)
# ==============================================================================

def _self_test() -> None:
    """Buffer join on two points and three synthetic buildings."""
    from shapely.geometry import Point

    # Two points ~10 m apart (Lisbon-ish coords), three synthetic buildings:
    # building A (height 20) near point 1 only, building B (height 40) near
    # both points, building C (height 10) far from both (outside any buffer).
    points_df = pd.DataFrame({"lat": [38.7200, 38.7201], "lon": [-9.1400, -9.1399]})

    buildings = gpd.GeoDataFrame(
        {"height": [20.0, 40.0, 10.0]},
        geometry=[
            Point(-9.14005, 38.72003).buffer(0.0002),   # building A: near point 1
            Point(-9.13995, 38.72005).buffer(0.0002),   # building B: near both
            Point(-9.20000, 38.80000).buffer(0.0002),   # building C: far away
        ],
        crs="EPSG:4326",
    ).rename(columns={"height": "_height_m"})

    result = join_building_height_buffer(points_df, "lat", "lon", buildings, radius_m=40.0)

    assert result.loc[0, "n_buildings_nearby"] >= 1, "Point 1 should see at least building A"
    assert result.loc[1, "n_buildings_nearby"] >= 1, "Point 2 should see at least building B"
    assert pd.notna(result.loc[0, "mean_building_height_m"])
    print("Self-test passed:")
    print(result[["lat", "lon", "mean_building_height_m",
                  "max_building_height_m", "n_buildings_nearby"]])


if __name__ == "__main__":
    _self_test()