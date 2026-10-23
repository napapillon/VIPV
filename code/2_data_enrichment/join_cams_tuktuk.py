"""
================================================================================
CAMS Satellite Irradiance Join — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Appends CAMS satellite irradiance and solar position to every measurement of
the tuk-tuk campaign, giving the tuk-tuk files the same irradiance columns as
the standard Solar Jinie files.

--------------------------------------------------------------------------------
Columns added
--------------------------------------------------------------------------------
    ghi        : CAMS all-sky global horizontal irradiance        [W/m²]
    ghi_clear  : CAMS clear-sky GHI (McClear)                     [W/m²]
    ghi_extra  : top-of-atmosphere horizontal irradiance          [W/m²]
    bhi        : CAMS beam horizontal irradiance                  [W/m²]
    dhi        : CAMS diffuse horizontal irradiance               [W/m²]
    dni        : CAMS direct normal irradiance                    [W/m²]
    sun_elev   : solar elevation at the GPS position              [deg]
    sun_azim   : solar azimuth, clockwise from North              [deg]

--------------------------------------------------------------------------------
Method
--------------------------------------------------------------------------------
1. GPS positions are snapped to a regular grid of spacing GRID_DEG.
2. One CAMS time series (cams-solar-radiation-timeseries, ADS API) is
   downloaded per distinct grid node, shared by all sensors. Raw downloads are
   cached in CACHE_DIR / "cams" and reused while node, date range and time
   step are unchanged.
3. CAMS values (Wh/m² per interval, converted to W/m²) are linearly
   interpolated in time onto each measurement of the node; timestamps outside
   the CAMS range receive NaN.
4. Solar position is computed with pvlib (NREL SPA) at the actual GPS
   position of each measurement.

The grid limits the number of API calls (CAMS itself computes any requested
coordinate): 0.05° (~5 km at Lisbon) gives a handful of nodes for the
campaign.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_tuktuk<INPUT_SUFFIX>.csv
    output : <sensor_id>_tuktuk<INPUT_SUFFIX>_cams.csv

Tuk-tuk join chain:
    decode_tuktuk.py         ->  <sensor_id>_tuktuk.csv
    join_ghi_fcul_tuktuk.py  ->  _ghi
    join_cams_tuktuk.py      ->  _ghi_cams                     <- this script
    join_cos23_tuktuk.py     ->  _ghi_cams_cos
    join_clc_tuktuk.py       ->  _ghi_cams_cos_clc
    join_bh_tuktuk.py        ->  _ghi_cams_cos_clc_bh
    join_sw_tuktuk.py        ->  _ghi_cams_cos_clc_bh_sw
    normalize_tuktuk.py      ->  _ghi_cams_cos_clc_bh_sw_norm   ->  clean_data.py

--------------------------------------------------------------------------------
ADS API setup (one-time)
--------------------------------------------------------------------------------
1. Create a free account at https://ads.atmosphere.copernicus.eu
2. Create the file ~/.cdsapirc (Windows: C:/Users/<you>/.cdsapirc):
       url: https://ads.atmosphere.copernicus.eu/api
       key: <YOUR-PERSONAL-ACCESS-TOKEN>
3. Accept the terms of use of the dataset at
   https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries

Requirements
------------
    pip install cdsapi numpy pandas pvlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from io import StringIO
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TUKTUK_DIR, CACHE_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import cdsapi
import numpy as np
import pandas as pd
import pvlib

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = TUKTUK_DIR

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j29", "j30", "j31", "j32", "j33"]

# --- Input suffix (pipeline stage of the input files) -------------------------
#   "_ghi" -> after join_ghi_fcul_tuktuk.py
INPUT_SUFFIX: str = "_ghi"

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_cams"

# --- Grid spacing for CAMS requests [deg] -------------------------------------
#   0.05° ≈ 5 km at Lisbon  ->  ~6 nodes for the campaign
#   0.10° ≈ 10 km           ->  ~2 nodes
GRID_DEG: float = 0.05

# --- Altitude sent with the CAMS request [m] ----------------------------------
# Representative of Lisbon; negligible effect on irradiance.
CAMS_ALTITUDE_M: float = 50.0

# --- CAMS time step: "1minute" | "15minute" | "1hour" | "1day" | "1month" -----
# "1minute" matches the measurement cadence.
CAMS_TIME_STEP: str = "1minute"

# --- Cache of raw CAMS downloads ----------------------------------------------
# Set CLEAR_CACHE = True for one run to force fresh downloads.
CAMS_TEMP_DIR: Path = CACHE_DIR / "cams"
CLEAR_CACHE:   bool = False

# ==============================================================================
# STEP 1 — LOAD TUKTUK DATA
# ==============================================================================

def load_tuktuk(filepath: Path) -> pd.DataFrame:
    """
    Load a tuk-tuk CSV file and parse its timestamp column to UTC-aware
    datetimes.

    Parameters
    ----------
    filepath : Path
        Path to the input CSV.

    Returns
    -------
    pd.DataFrame
        All rows, with 'timestamp' as timezone-aware UTC datetime.
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ==============================================================================
# STEP 2 — SNAP GPS COORDINATES TO GRID NODES
# ==============================================================================

def snap_to_grid(df: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    """
    Assign each measurement to the nearest node of the discretisation grid.

    Adds two columns to the dataframe:
        cams_lat : snapped latitude  (decimal degrees)
        cams_lon : snapped longitude (decimal degrees)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'lat' and 'lon' columns (decimal degrees).
    grid_deg : float
        Grid spacing in decimal degrees.

    Returns
    -------
    pd.DataFrame
        Input dataframe with 'cams_lat' and 'cams_lon' columns appended.
    """
    df = df.copy()
    df["cams_lat"] = ((df["lat"] / grid_deg).round() * grid_deg).round(4)
    df["cams_lon"] = ((df["lon"] / grid_deg).round() * grid_deg).round(4)
    return df


def collect_all_nodes(sensor_dfs: dict[str, pd.DataFrame]) -> set[tuple[float, float]]:
    """
    Collect all distinct grid nodes across all sensor datasets.

    Pooling nodes across sensors avoids redundant CAMS downloads when two
    sensors cover the same geographic area.

    Parameters
    ----------
    sensor_dfs : dict
        Mapping sensor_id → dataframe (with 'cams_lat', 'cams_lon' columns).

    Returns
    -------
    set of (cams_lat, cams_lon) tuples.
    """
    nodes: set[tuple[float, float]] = set()
    for df in sensor_dfs.values():
        pairs = df[["cams_lat", "cams_lon"]].drop_duplicates().values
        nodes.update((round(lat, 4), round(lon, 4)) for lat, lon in pairs)
    return nodes


# ==============================================================================
# STEP 3 — DOWNLOAD AND PARSE CAMS DATA
# ==============================================================================

def _date_range_from_dfs(sensor_dfs: dict[str, pd.DataFrame]) -> tuple[str, str]:
    """
    Derive a CAMS download date range covering all sensors, extended by one
    day on each side for full interpolation coverage.
    """
    all_ts = pd.concat([df["timestamp"] for df in sensor_dfs.values()])
    return (
        (all_ts.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        (all_ts.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )


def download_cams_node(lat: float, lon: float, altitude: float,
                       date_begin: str, date_end: str,
                       time_step: str, output_path: Path) -> None:
    """
    Request the CAMS solar radiation time series for one grid node from the
    ADS API.

    The ADS 'cams-solar-radiation-timeseries' dataset provides:
        - All-sky GHI, BHI, DHI, BNI   (Meteosat / Heliosat)
        - Clear-sky GHI, BHI, DHI, BNI  (McClear model)
        - Top-of-atmosphere irradiance   (TOA → used as ghi_extra)

    Parameters
    ----------
    lat, lon : float
        Grid node coordinates (decimal degrees, WGS-84).
    altitude : float
        Elevation above sea level in metres.
    date_begin, date_end : str
        Inclusive date range 'YYYY-MM-DD'.
    time_step : str
        CAMS time step: '1minute', '15minute', '1hour', etc.
    output_path : Path
        Destination path for the raw API response (CSV format).
    """
    client = cdsapi.Client()
    client.retrieve(
        "cams-solar-radiation-timeseries",
        {
            "sky_type":       "observed_cloud",
            "location":       {"latitude": lat, "longitude": lon},
            "altitude":       [str(int(altitude))],
            "date":           [f"{date_begin}/{date_end}"],
            "time_step":      time_step,
            "time_reference": "universal_time",
            "data_format":    "csv",
        },
        str(output_path),
    )


def parse_cams_csv(raw_path: Path, time_step: str,
                   node_lat: float, node_lon: float) -> pd.DataFrame:
    """
    Parse one raw CAMS CSV file into a clean irradiance DataFrame.

    File format (CAMS format v2, semicolon-separated):
        - Lines starting with '#' are metadata / comments.
        - The last comment line is the column header, e.g.:
              # Observation period;TOA;Clear sky GHI;...;GHI;BHI;DHI;BNI;Reliability
        - Data rows use the interval notation:
              2026-06-29T00:00:00.0/2026-06-29T00:01:00.0;0.0;...

    Unit conversion
    ---------------
    The ADS returns accumulated irradiation in Wh/m² per interval.
    Dividing by the interval duration in hours gives average irradiance in W/m²:
        1-minute  : ÷ (1/60)
        15-minute : ÷ 0.25

    Parameters
    ----------
    raw_path : Path
        Path to the raw CAMS CSV file.
    time_step : str
        CAMS time step string used for the download.
    node_lat, node_lon : float
        Grid node coordinates (added as identification columns).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp_utc, cams_lat, cams_lon,
                 ghi, ghi_clear, ghi_extra, bhi, dhi, dni
        All irradiance values in W/m², clipped to ≥ 0.
    """
    step_to_hours: dict[str, float] = {
        "1minute":  1.0 / 60.0,
        "15minute": 15.0 / 60.0,
        "1hour":    1.0,
        "1day":     24.0,
    }
    interval_h = step_to_hours.get(time_step, 1.0 / 60.0)

    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_line    = None
    first_data_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and ";" in stripped:
            header_line = stripped.lstrip("# ")
        elif not stripped.startswith("#") and stripped:
            first_data_idx = i
            break

    if header_line is None or first_data_idx is None:
        raise ValueError(
            f"Could not locate column header or data rows in: {raw_path}"
        )

    raw = pd.read_csv(
        StringIO(header_line + "\n" + "".join(lines[first_data_idx:])),
        sep=";",
        skipinitialspace=True,
    )
    raw.columns = [c.strip() for c in raw.columns]

    # Parse the START of the observation period as timestamp_utc
    ts_col = "Observation period" if "Observation period" in raw.columns else raw.columns[0]
    raw["timestamp_utc"] = (
        raw[ts_col]
        .astype(str)
        .str.split("/").str[0]
        .str.replace("T", " ", regex=False)
        .str[:16]
    )
    raw["timestamp_utc"] = pd.to_datetime(
        raw["timestamp_utc"], utc=True, errors="coerce"
    )
    raw = raw.dropna(subset=["timestamp_utc"]).reset_index(drop=True)

    # Map CAMS column names to pipeline output names and convert Wh/m² → W/m²
    # CAMS format v2 columns:
    #   TOA, Clear sky GHI, Clear sky BHI, Clear sky DHI, Clear sky BNI,
    #   GHI, BHI, DHI, BNI, Reliability
    col_map: dict[str, str] = {
        "ghi":       "GHI",
        "ghi_clear": "Clear sky GHI",
        "ghi_extra": "TOA",
        "bhi":       "BHI",
        "dhi":       "DHI",
        "dni":       "BNI",
    }

    out = pd.DataFrame({
        "timestamp_utc": raw["timestamp_utc"],
        "cams_lat":      node_lat,
        "cams_lon":      node_lon,
    })
    for out_col, src_col in col_map.items():
        if src_col in raw.columns:
            out[out_col] = pd.to_numeric(raw[src_col], errors="coerce") / interval_h
        else:
            print(f"    [WARNING] CAMS column '{src_col}' not found — filled with NaN.")
            out[out_col] = float("nan")

    for col in ["ghi", "ghi_clear", "ghi_extra", "bhi", "dhi", "dni"]:
        out[col] = out[col].clip(lower=0.0)

    return out


def download_all_nodes(nodes: set[tuple[float, float]],
                       date_begin: str, date_end: str,
                       time_step: str, altitude: float,
                       temp_dir: Path) -> dict[tuple[float, float], pd.DataFrame]:
    """
    Download and parse CAMS data for every distinct grid node.

    Downloads are shared across all sensors: each node is downloaded once
    regardless of how many sensors cover it.

    Cache behaviour
    ---------------
    Raw CSV files are kept in temp_dir after parsing and reused on subsequent
    runs.  The cache filename encodes the node coordinates, date range, and
    time step, so it is automatically invalidated if any of these parameters
    change.  To force a fresh download, delete the files in CAMS_TEMP_DIR
    manually or set CLEAR_CACHE = True in the CONFIG section.

    Parameters
    ----------
    nodes : set of (lat, lon) tuples
        Grid nodes to download.
    date_begin, date_end : str
        Inclusive date range 'YYYY-MM-DD'.
    time_step : str
        CAMS time step string.
    altitude : float
        Representative altitude in metres.
    temp_dir : Path
        Directory for cached raw download files.

    Returns
    -------
    dict mapping (cams_lat, cams_lon) → parsed CAMS DataFrame.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    sorted_nodes = sorted(nodes)
    n = len(sorted_nodes)
    node_data: dict[tuple[float, float], pd.DataFrame] = {}

    for i, (lat, lon) in enumerate(sorted_nodes, start=1):
        print(f"\n  Node {i}/{n} : ({lat:.4f}°N, {lon:.4f}°E)")

        # Cache filename encodes all parameters that affect the download result.
        # A change in date range, time step, or coordinates invalidates the cache.
        cache_filename = (
            f"cams_{lat:.4f}_{lon:.4f}"
            f"_{date_begin}_{date_end}"
            f"_{time_step}.csv"
        )
        cache_path = temp_dir / cache_filename

        if CLEAR_CACHE and cache_path.exists():
            cache_path.unlink()
            print("    Cache cleared.")

        if cache_path.exists():
            print(f"    [CACHE] Using cached file: {cache_path.name}")
        else:
            print(f"    Downloading {date_begin} → {date_end} [{time_step}] ...")
            download_cams_node(lat, lon, altitude, date_begin, date_end,
                               time_step, cache_path)
            print(f"    Saved to cache: {cache_path.name}")

        print("    Parsing ...")
        cams = parse_cams_csv(cache_path, time_step, lat, lon)
        node_data[(lat, lon)] = cams
        print(f"    {len(cams):,} records  "
              f"| GHI {cams['ghi'].min():.1f}–{cams['ghi'].max():.1f} W/m²")

    return node_data


# ==============================================================================
# STEP 4 — COMPUTE SOLAR POSITION WITH PVLIB
# ==============================================================================

def compute_solar_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute solar elevation and azimuth for each measurement using the
    pvlib NREL SPA algorithm and the actual GPS coordinates.

    Azimuth convention: 0° = North, 90° = East, 180° = South, 270° = West
    (pvlib default, consistent with the J30–J33 sensor pipeline).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'timestamp' (UTC-aware), 'lat', 'lon' columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with 'sun_elev' and 'sun_azim' columns appended.
    """
    df = df.copy()
    lats       = df["lat"].values
    lons       = df["lon"].values
    timestamps = df["timestamp"]

    sun_elevs = np.full(len(df), np.nan)
    sun_azims = np.full(len(df), np.nan)

    pos_df  = pd.DataFrame({"lat": lats, "lon": lons})
    grouped = pos_df.groupby(["lat", "lon"], sort=False)

    for (lat, lon), idx in grouped.groups.items():
        if pd.isna(lat) or pd.isna(lon):
            continue
        loc = pvlib.location.Location(
            latitude=lat, longitude=lon, altitude=0.0, tz="UTC"
        )
        sp = loc.get_solarposition(timestamps.iloc[idx])
        sun_elevs[idx] = sp["elevation"].values
        sun_azims[idx] = sp["azimuth"].values

    df["sun_elev"] = np.round(sun_elevs, 4)
    df["sun_azim"] = np.round(sun_azims, 4)
    return df


# ==============================================================================
# STEP 5 — INTERPOLATE CAMS ONTO TUKTUK TIMESTAMPS
# ==============================================================================

def interpolate_cams_per_node(df: pd.DataFrame,
                               node_data: dict[tuple[float, float], pd.DataFrame]
                               ) -> pd.DataFrame:
    """
    For each measurement, linearly interpolate the CAMS time series of its
    assigned grid node onto the measurement timestamp.

    Measurements outside the CAMS time range for their node receive NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'timestamp', 'cams_lat', 'cams_lon' columns.
    node_data : dict
        Mapping (cams_lat, cams_lon) → parsed CAMS DataFrame.

    Returns
    -------
    pd.DataFrame
        Input dataframe with CAMS columns appended:
        ghi, ghi_clear, ghi_extra, bhi, dhi, dni.
    """
    cams_cols = ["ghi", "ghi_clear", "ghi_extra", "bhi", "dhi", "dni"]

    result = df.copy()
    for col in cams_cols:
        result[col] = np.nan

    for (node_lat, node_lon), cams_df in node_data.items():
        mask = (df["cams_lat"] == node_lat) & (df["cams_lon"] == node_lon)
        if not mask.any():
            continue

        tuktuk_ts    = pd.DatetimeIndex(df.loc[mask, "timestamp"])
        cams_indexed = cams_df.set_index("timestamp_utc")[cams_cols]
        all_times    = tuktuk_ts.union(cams_indexed.index)
        cams_interp  = (
            cams_indexed
            .reindex(all_times)
            .interpolate(method="time", limit_area="inside")
        )

        extracted = cams_interp.reindex(tuktuk_ts)
        for col in cams_cols:
            result.loc[mask, col] = extracted[col].values

        n_nan = result.loc[mask, "ghi"].isna().sum()
        print(f"    Node ({node_lat:.4f}°N, {node_lon:.4f}°E) : "
              f"{mask.sum():,} rows  ({n_nan} NaN outside CAMS range)")

    return result


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(sensor_id: str,
                   node_data: dict[tuple[float, float], pd.DataFrame]) -> pd.DataFrame | None:
    """
    Join CAMS data and solar position onto one sensor's tuk-tuk dataset.

    Parameters
    ----------
    sensor_id : str
        Sensor identifier (e.g. 'j33').
    node_data : dict
        Pre-downloaded CAMS node data (shared across all sensors).

    Returns
    -------
    pd.DataFrame or None
        Processed dataframe (also written to disk), or None if input missing.
    """
    input_file  = DATA_DIR / f"{sensor_id}_tuktuk{INPUT_SUFFIX}.csv"
    output_file = DATA_DIR / f"{sensor_id}_tuktuk{INPUT_SUFFIX}{OUTPUT_SUFFIX}.csv"

    print(f"\n{'─' * 68}")
    print(f"  Sensor : {sensor_id.upper()}")
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")
    print(f"{'─' * 68}")

    if not input_file.exists():
        print("  [SKIP] Input file not found.")
        return None

    df = load_tuktuk(input_file)
    print(f"  {len(df):,} rows loaded  "
          f"({df['timestamp'].min()} → {df['timestamp'].max()})")

    df = snap_to_grid(df, GRID_DEG)

    print("  Computing solar position ...")
    df = compute_solar_position(df)
    print(f"  sun_elev : {df['sun_elev'].min():.1f}° – {df['sun_elev'].max():.1f}°")

    print("  Interpolating CAMS ...")
    df = interpolate_cams_per_node(df, node_data)

    df = df.drop(columns=["cams_lat", "cams_lon"])
    df.to_csv(output_file, index=False)

    print("\n  CAMS columns added:")
    for col in ["ghi", "ghi_clear", "ghi_extra", "bhi", "dhi", "dni"]:
        if col in df.columns:
            n_ok = df[col].notna().sum()
            print(f"    {col:10s}  {n_ok:,} valid  "
                  f"({df[col].min():.1f} – {df[col].max():.1f} W/m²)")
    print(f"  Written → {output_file}")
    return df


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  JOIN CAMS SATELLITE DATA — TUK-TUK CAMPAIGN")
    print(f"  Sensors     : {SENSOR_IDS}")
    print(f"  Grid        : {GRID_DEG}°  |  Time step : {CAMS_TIME_STEP}")
    print(f"  Data dir    : {DATA_DIR}")
    print("=" * 68)

    # ── Step 1: Load all available sensor files ───────────────────────────────
    print("\n[1/3] Loading sensor data and snapping to grid ...")
    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor_id in SENSOR_IDS:
        input_file = DATA_DIR / f"{sensor_id}_tuktuk{INPUT_SUFFIX}.csv"
        if not input_file.exists():
            print(f"  [SKIP] {sensor_id}: file not found.")
            continue
        df = load_tuktuk(input_file)
        df = snap_to_grid(df, GRID_DEG)
        sensor_dfs[sensor_id] = df
        nodes = df[["cams_lat", "cams_lon"]].drop_duplicates()
        print(f"  {sensor_id.upper()} : {len(df):,} rows  |  {len(nodes)} grid node(s)")

    if not sensor_dfs:
        print("  No valid sensor files found. Check DATA_DIR and SENSOR_IDS.")
        return

    # ── Step 2: Download CAMS — once per node, shared across all sensors ──────
    all_nodes  = collect_all_nodes(sensor_dfs)
    date_begin, date_end = _date_range_from_dfs(sensor_dfs)
    print(f"\n[2/3] Downloading CAMS data "
          f"({len(all_nodes)} node(s), {date_begin} → {date_end}) ...")
    node_data = download_all_nodes(
        nodes      = all_nodes,
        date_begin = date_begin,
        date_end   = date_end,
        time_step  = CAMS_TIME_STEP,
        altitude   = CAMS_ALTITUDE_M,
        temp_dir   = CAMS_TEMP_DIR,
    )

    # ── Step 3: Process each sensor independently ─────────────────────────────
    print("\n[3/3] Processing sensors ...")
    for sensor_id in sensor_dfs:
        process_sensor(sensor_id, node_data)

    print(f"\n{'=' * 68}")
    print(f"  Done.  Next step : clean_data.py  "
          f"(INPUT_SUFFIX = '{INPUT_SUFFIX}{OUTPUT_SUFFIX}')")
    print(f"{'=' * 68}\n")


# ==============================================================================
if __name__ == "__main__":
    main()
