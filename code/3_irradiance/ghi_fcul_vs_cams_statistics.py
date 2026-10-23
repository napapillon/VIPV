"""
================================================================================
FCUL Pyranometer GHI vs CAMS Satellite GHI — Error Statistics by Distance Ring
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Error statistics of the CAMS satellite GHI relative to the FCUL rooftop
pyranometer (Eko MS-802), by distance ring between the vehicle and the
station (same data and matching as ghi_fcul_vs_cams_scatter.py).

    residual = ghi_cams − ghi_fcul   [W/m²]   (positive: CAMS overestimates)

Four metrics per ring, one bar-chart panel each:
    MBE   mean(residuals)                 systematic bias
    MAE   mean(|residuals|)               mean error magnitude, no sign
                                          compensation
    RMSE  sqrt(mean(residuals²))          weights large errors more
    STD   std(residuals, ddof=1)          dispersion around the mean
                                          ('Desvio Padrão' of Figure 3.22)

An optional clear-sky filter (ENABLE_CLEAR_SKY_FILTER, ghi_cams / ghi_clear
>= MIN_CLEAR_SKY_INDEX) restricts the comparison to near-clear-sky minutes;
it is disabled by default.

Distance rings (RING_BOUNDARIES_KM; points beyond 32 km are excluded):
    0–3 km   : inner zone (≈ CAMS pixel, ~3 × 3 km²)
    3–9 km   : intermediate zone
    9–18 km  : outer reliable zone
    18–32 km : far zone (highest spatial uncertainty)

--------------------------------------------------------------------------------
Data and timestamp matching
--------------------------------------------------------------------------------
Standard campaign: the FCUL pyranometer GHI comes from the Excel files of
FCUL_FILES (FCUL_DIR, 1-minute averages, column 'Avg_GHI', 'GMT TIME' = UTC):
    GHI2021.xlsx        (sheet DATA_2021)
    GHI2022_part2.xlsx  (sheet DATA_2022)
    GHI2023.xlsx        (sheet DATA_2023)
Both series are rounded to the UTC minute and joined on that key. Data notes
from the provider: January–May 2022 are absent (no matches for that period);
negative GHI values (night-time instrument offset) are clipped to 0 W/m².

Tuk-tuk campaign: the sensor files already carry 'ghi_fcul'
(join_ghi_fcul_tuktuk.py), so no Excel file or timestamp join is needed.

Rows kept: ghi_fcul >= MIN_IRRADIANCE_WM2 and ghi_clear > 0. No threshold is
applied to the CAMS ghi or to sun_elev here: use a '_hq' variant of
clean_data.py (Step 2) for those.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (FILE_SUFFIX / TUKTUK_FILE_SUFFIX), one
             file per sensor
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', 'ghi', 'ghi_clear', 'sun_elev' (+ 'ghi_fcul',
             tuk-tuk)
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             mdr_ghi_fcul_vs_cams_statistics_rings_<CAMPAIGN>_<variant_tag>.png

Requirements
------------
    pip install pandas numpy matplotlib openpyxl

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.1
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FCUL_DIR, OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- FCUL pyranometer Excel files (GMT TIME column = UTC) ---------------------
# Used only when CAMPAIGN = "standard" (tuk-tuk files already carry
# 'ghi_fcul', join_ghi_fcul_tuktuk.py). Provider notes: January–May 2022
# absent; negative GHI clipped to 0 W/m² in load_fcul_ghi().
FCUL_DATA_DIR: Path = FCUL_DIR

FCUL_FILES: list[tuple[Path, str]] = [
    (FCUL_DATA_DIR / 'GHI2021.xlsx',       'DATA_2021'),
    (FCUL_DATA_DIR / 'GHI2022_part2.xlsx', 'DATA_2022'),
    (FCUL_DATA_DIR / 'GHI2023.xlsx',       'DATA_2023'),
]

# --- Sensor input directories (one per campaign) ------------------------------
SENSOR_DATA_DIR: Path = STANDARD_DIR     # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py variant) -------------------------------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output base directory ----------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "3_irradiance" / "mdr_outputs"

# --- FCUL reference station coordinates (WGS-84 decimal degrees) --------------
# Pyranometer on the rooftop of building C1, Faculty of Sciences,
# University of Lisbon, campus Cidade Universitária.
FCUL_LAT: float = 38.7565    # Google Maps: 38.756594832619776
FCUL_LON: float = -9.1564    # Google Maps: -9.156483276747663

# --- Distance ring boundaries (km) --------------------------------------------
# Points at or beyond 32 km are excluded — outside the analysis domain.
RING_BOUNDARIES_KM: list[int] = [0, 3, 9, 18, 32]

# --- Earth radius -------------------------------------------------------------
R_EARTH_KM: float = 6371.0   # Mean Earth radius (km) – IAU 2012

# --- Filter 1: Minimum irradiance threshold ----------------------------------
MIN_IRRADIANCE_WM2: float = 25.0   # W/m², ghi_fcul only

# --- Filter 2: Clear-sky index (kc_clear = ghi_cams / ghi_clear) -------------
ENABLE_CLEAR_SKY_FILTER: bool  = False
MIN_CLEAR_SKY_INDEX:     float = 0.85

# --- Ring colours (identical to map_distance_rings.py → DISTANCE_COLORS) ------
RING_COLORS: list[str] = [
    "#2166ac",   # 0–3 km   : blue
    "#4dac26",   # 3–9 km   : green
    "#f4a582",   # 9–18 km  : light orange
    "#d6604d",   # 18–32 km : red-orange
]

# --- Bar chart styling --------------------------------------------------------
BAR_ALPHA:      float = 0.88
BAR_EDGE_COLOR: str   = 'white'
BAR_WIDTH:      float = 0.6

# --- Layout -------------------------------------------------------------------
FIGURE_SIZE: tuple = (14, 11)

# --- Figure resolution --------------------------------------------------------
OUTPUT_DPI: int = 300

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# Sensor input and output paths follow from CAMPAIGN and the input suffix;
# FCUL_FILES do not depend on them.

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    sensors = ['j30', 'j31', 'j32', 'j33']
    if campaign == "standard":
        return {
            s: data_dir / f"{s}_satellite_landuse_v2_{suffix}.csv"
            for s in sensors
        }
    elif campaign == "tuktuk":
        return {
            s: tuktuk_data_dir / f"{s}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
            for s in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output directory, embedding campaign + variant tag so
    outputs from different campaigns/variants never overwrite each other."""
    return base_dir / campaign / variant_tag


def _build_output_path(out_dir: Path, base_name: str, campaign: str,
                       variant_tag: str, extension: str) -> Path:
    """Build an output file path that embeds the active campaign and variant tag."""
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolve all paths at startup — do NOT edit these directly; change CAMPAIGN /
# FILE_SUFFIX / TUKTUK_FILE_SUFFIX above instead.
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, SENSOR_DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR:   Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_FILE:  Path = _build_output_path(
    OUTPUT_DIR, 'mdr_ghi_fcul_vs_cams_statistics_rings', CAMPAIGN, _VARIANT_TAG, 'png'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def haversine_km(lat1: float, lon1: float,
                 lat2: np.ndarray, lon2: np.ndarray,
                 R: float = R_EARTH_KM) -> np.ndarray:
    """
    Compute the great-circle distance (km) between a fixed reference point
    (lat1, lon1) and an array of points (lat2, lon2) using the Haversine
    formula.

    Parameters
    ----------
    lat1, lon1 : float
        Reference point in decimal degrees (WGS-84).
    lat2, lon2 : array-like
        Target points in decimal degrees (WGS-84).
    R : float
        Mean Earth radius in km (default: 6371.0).

    Returns
    -------
    np.ndarray
        Distances in kilometres.
    """
    phi1    = np.radians(lat1)
    phi2    = np.radians(np.asarray(lat2, dtype=float))
    dphi    = np.radians(np.asarray(lat2, dtype=float) - lat1)
    dlambda = np.radians(np.asarray(lon2, dtype=float) - lon1)

    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(a))   # kilometres


def classify_into_rings(distances_km: np.ndarray,
                        boundaries: list) -> np.ndarray:
    """
    Assign each measurement to a distance ring index (0-based).

    Ring i covers: boundaries[i] ≤ d < boundaries[i+1].
    Points outside [boundaries[0], boundaries[-1]) return index -1.
    Callers must filter these out with filter_to_analysis_range().

    Parameters
    ----------
    distances_km : np.ndarray
        Distances from FCUL in kilometres.
    boundaries : list of int
        Sorted ring boundary values [0, r1, r2, …, rN].

    Returns
    -------
    np.ndarray of int
        Ring index (0-based) for each point, or -1 if out of range.
    """
    distances_km = np.asarray(distances_km)
    ring_indices = np.full(len(distances_km), -1, dtype=int)
    for i in range(len(boundaries) - 1):
        mask = (distances_km >= boundaries[i]) & (distances_km < boundaries[i + 1])
        ring_indices[mask] = i
    return ring_indices


def create_ring_labels(boundaries: list) -> list[str]:
    """
    Build human-readable labels for each distance ring.

    For boundaries [0, 3, 9, 18, 32] returns:
        ['0–3 km', '3–9 km', '9–18 km', '18–32 km']

    Parameters
    ----------
    boundaries : list of int
        Sorted ring boundary values in kilometres.

    Returns
    -------
    list of str
        Human-readable ring labels.
    """
    return [
        f"{boundaries[i]}–{boundaries[i + 1]} km"
        for i in range(len(boundaries) - 1)
    ]


def filter_to_analysis_range(df: pd.DataFrame,
                              boundaries: list) -> pd.DataFrame:
    """
    Retain only measurements within the analysis distance domain
    [boundaries[0], boundaries[-1]).

    Points at or beyond boundaries[-1] (≥ 32 km) are outside all defined
    rings and must not be silently absorbed into the last ring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'distance_km' column already computed.
    boundaries : list of int
        Sorted ring boundary values [0, r1, r2, …, rN].

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only in-range measurements.
    """
    d_max = boundaries[-1]
    mask  = (df['distance_km'] >= boundaries[0]) & (df['distance_km'] < d_max)
    n_out = int((~mask).sum())
    if n_out > 0:
        print(f"  [INFO] {n_out:,} measurements beyond {d_max} km excluded "
              f"(outside analysis domain).")
    return df[mask].reset_index(drop=True)


# ==============================================================================
# DATA LOADING  (identical logic to ghi_fcul_vs_cams_scatter.py)
# ==============================================================================

def load_fcul_ghi(fcul_files: list) -> pd.DataFrame:
    """
    Load and concatenate FCUL pyranometer GHI data from all Excel files.

    Each file contains columns:
        - 'GMT TIME'  : UTC timestamp (1-minute resolution, tz-naive)
        - 'Avg_GHI'   : averaged GHI measurement [W/m²]

    The first rows of each sheet contain unit headers and are dropped by
    filtering on rows where 'GMT TIME' parses as a valid datetime.

    Data notes (from the FCUL data provider):
    - January–May 2022 are absent. The join produces no matches for this period.
    - Negative GHI values are clipped to 0 W/m² per data provider instruction.

    Parameters
    ----------
    fcul_files : list of (Path, str)
        List of (file_path, sheet_name) tuples.

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp_utc', 'ghi_fcul'].
        'timestamp_utc' is a timezone-naive UTC datetime rounded to the minute.
        Rows with ghi_fcul < MIN_IRRADIANCE_WM2 are excluded.
    """
    frames = []
    for file_path, sheet_name in fcul_files:
        if not file_path.is_file():
            print(f"  [WARNING] FCUL file not found: {file_path}  – skipped.")
            continue

        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Drop header/unit rows: keep only rows where GMT TIME parses as datetime
        df['GMT TIME'] = pd.to_datetime(df['GMT TIME'], errors='coerce')
        df = df[df['GMT TIME'].notna()].copy()

        df['Avg_GHI'] = pd.to_numeric(df['Avg_GHI'], errors='coerce')
        df = df.dropna(subset=['Avg_GHI'])

        # Clip negative values to 0 W/m² (instrument night-time offset)
        # Per data provider: "any negative value in GHI can be assumed as 0 W/m²"
        df['Avg_GHI'] = df['Avg_GHI'].clip(lower=0.0)

        # Round to minute — ensures exact join key
        df['timestamp_utc'] = df['GMT TIME'].dt.floor('min')

        sub = df[['timestamp_utc', 'Avg_GHI']].rename(
            columns={'Avg_GHI': 'ghi_fcul'}
        )
        frames.append(sub)
        print(f"  {file_path.name} ({sheet_name}): {len(sub):,} rows loaded")

    if not frames:
        raise FileNotFoundError(
            "No FCUL Excel files could be loaded. "
            "Please check the paths defined in FCUL_FILES."
        )

    fcul = pd.concat(frames, ignore_index=True)

    # Remove duplicates (overlapping periods across files) — keep first
    fcul = fcul.drop_duplicates(subset='timestamp_utc').sort_values('timestamp_utc')

    # Apply minimum irradiance threshold (after clipping — zeros excluded here)
    n_before  = len(fcul)
    fcul = fcul[fcul['ghi_fcul'] >= MIN_IRRADIANCE_WM2].reset_index(drop=True)
    n_excluded = n_before - len(fcul)
    print(f"\n  FCUL total: {len(fcul):,} valid rows "
          f"({n_excluded:,} night/near-zero excluded, "
          f"threshold {MIN_IRRADIANCE_WM2} W/m²)")

    return fcul


def load_sensor_data(file_path: Path, sensor_name: str,
                     include_ghi_fcul: bool = False) -> pd.DataFrame:
    """
    Load one cleaned sensor CSV file and extract the columns needed for the
    FCUL vs CAMS GHI statistics.

    Columns extracted and renamed:
        timestamp_utc  ← 'Human readable date (Europe/London)'  (UTC, tz-aware)
        lat            ← 'J3x lat'
        lon            ← 'J3x lon'
        ghi_cams       ← 'ghi'       (CAMS satellite GHI, W/m²)
        ghi_clear      ← 'ghi_clear' (CAMS clear-sky GHI, W/m²)
        sun_elev       ← 'sun_elev'  (solar elevation angle, degrees)
        ghi_fcul       ← 'ghi_fcul'  (tuk-tuk campaign only, see below)

    The timestamp is converted to a timezone-naive UTC datetime rounded to the
    minute so it can be joined to the FCUL time series (standard campaign only).

    No threshold is applied to ghi (CAMS) here: select a "..._hq" variant
    (clean_data.py Step 2) for the ghi > 25 W/m² floor.

    Parameters
    ----------
    file_path : Path
        Path to the cleaned sensor CSV file.
    sensor_name : str
        Sensor identifier, e.g. 'j31'.
    include_ghi_fcul : bool
        Set True for the tuk-tuk campaign, whose per-sensor files already
        carry a 'ghi_fcul' column — joined in-pipeline by join_ghi_fcul_tuktuk.py,
        well before clean_data.py — unlike the standard campaign, which has
        no FCUL column at all and must be matched against the separate
        Excel files (load_fcul_ghi + a timestamp merge in main()).

    Returns
    -------
    pd.DataFrame
        Columns: [timestamp_utc, lat, lon, ghi_cams, ghi_clear, sun_elev,
                  sensor] (+ ghi_fcul if include_ghi_fcul).
        Returns an empty DataFrame if the file is missing or unusable.
    """
    if not file_path.is_file():
        print(f"  [WARNING] File not found: {file_path}  – skipped.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as exc:
        print(f"  [ERROR] Could not read {file_path.name}: {exc}")
        return pd.DataFrame()

    # ── Locate sensor-specific columns ────────────────────────────────────────
    sensor_upper = sensor_name.upper()   # 'j31' → 'J31'
    col_lat = f"{sensor_upper} lat"
    col_lon = f"{sensor_upper} lon"

    # Fallback: case-insensitive search (excludes land_use_clcv columns)
    if col_lat not in df.columns:
        candidates = [c for c in df.columns
                      if 'lat' in c.lower() and 'land' not in c.lower()]
        col_lat = candidates[0] if candidates else None

    if col_lon not in df.columns:
        candidates = [c for c in df.columns
                      if 'lon' in c.lower() and 'land' not in c.lower()]
        col_lon = candidates[0] if candidates else None

    col_ts    = 'Human readable date (Europe/London)'
    col_ghi   = 'ghi'
    col_clear = 'ghi_clear'
    col_elev  = 'sun_elev'

    required = [col_ts, col_lat, col_lon, col_ghi, col_clear, col_elev]
    if include_ghi_fcul:
        required.append('ghi_fcul')
    missing  = [c for c in required if c is None or c not in df.columns]
    if missing:
        print(f"  [ERROR] Missing columns in {file_path.name}: {missing}")
        return pd.DataFrame()

    cols_to_take = [col_ts, col_lat, col_lon, col_ghi, col_clear, col_elev]
    out_names    = ['ts_raw', 'lat', 'lon', 'ghi_cams', 'ghi_clear', 'sun_elev']
    if include_ghi_fcul:
        cols_to_take.append('ghi_fcul')
        out_names.append('ghi_fcul')

    data = df[cols_to_take].copy()
    data.columns = out_names

    # ── Parse timestamp → UTC, tz-naive, rounded to minute ────────────────────
    data['ts_raw'] = pd.to_datetime(data['ts_raw'], utc=True, errors='coerce')
    data = data[data['ts_raw'].notna()].copy()
    data['timestamp_utc'] = (data['ts_raw']
                             .dt.tz_convert('UTC')
                             .dt.tz_localize(None)
                             .dt.floor('min'))
    data = data.drop(columns='ts_raw')

    # ── Coerce numeric columns ─────────────────────────────────────────────────
    numeric_cols = ['lat', 'lon', 'ghi_cams', 'ghi_clear', 'sun_elev']
    if include_ghi_fcul:
        numeric_cols.append('ghi_fcul')
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # ── Drop NaN and invalid coordinates ──────────────────────────────────────
    data = data.dropna()
    data = data[data['lat'].between(-90, 90) & data['lon'].between(-180, 180)]
    data = data[(data['lat'] != 0.0) & (data['lon'] != 0.0)]

    # ── Avoid division by zero on ghi_clear ───────────────────────────────────
    data = data[data['ghi_clear'] > 0]

    # ── Tuk-tuk campaign: apply the ghi_fcul irradiance filter here directly
    # (standard campaign applies it in load_fcul_ghi(), before the merge) ─────
    if include_ghi_fcul:
        data = data[data['ghi_fcul'] >= MIN_IRRADIANCE_WM2]

    if data.empty:
        print(f"  [WARNING] No valid rows in {file_path.name} – skipped.")
        return pd.DataFrame()

    data['sensor'] = sensor_upper
    print(f"  {sensor_upper}: {len(data):,} valid rows "
          f"loaded from '{file_path.name}'")
    return data.reset_index(drop=True)


# ==============================================================================
# FILTER FUNCTIONS
# ==============================================================================

def filter_clear_sky(data: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only near-clear-sky conditions where ghi_cams / ghi_clear ≥
    MIN_CLEAR_SKY_INDEX (Filter 2).

    Parameters
    ----------
    data : pd.DataFrame
        Matched data with 'ghi_cams' and 'ghi_clear' columns.

    Returns
    -------
    pd.DataFrame
    """
    csi      = data['ghi_cams'] / data['ghi_clear']
    filtered = data[csi >= MIN_CLEAR_SKY_INDEX].copy()
    removed  = len(data) - len(filtered)
    print(f"    Removed {removed:,} partly-cloudy measurements "
          f"(ghi_cams / ghi_clear < {MIN_CLEAR_SKY_INDEX:.2f})")
    return filtered.reset_index(drop=True)


# ==============================================================================
# STATISTICS AND PLOTTING
# ==============================================================================

def compute_ring_statistics(matched: pd.DataFrame,
                             ring_labels: list[str]) -> pd.DataFrame:
    """
    Compute error statistics of CAMS GHI relative to FCUL GHI per ring.

    Residual:  e = ghi_cams − ghi_fcul   [W/m²]
    A positive value means CAMS overestimates relative to FCUL.

    Metrics computed per ring:
        n     : number of matched measurements
        MBE   : Mean Bias Error        = mean(e)
        MAE   : Mean Absolute Error    = mean(|e|)
        RMSE  : Root Mean Square Error = sqrt(mean(e²))
        STD   : Standard deviation     = std(e, ddof=1)

    Parameters
    ----------
    matched : pd.DataFrame
        Matched data with columns 'ring', 'ghi_cams', 'ghi_fcul'.
    ring_labels : list of str
        Human-readable ring labels (same order as ring indices).

    Returns
    -------
    pd.DataFrame
        One row per ring with columns [ring, n, MBE, MAE, RMSE, STD].
    """
    rows = []
    for i, label in enumerate(ring_labels):
        ring_data = matched[matched['ring'] == i]
        n         = len(ring_data)

        if n == 0:
            rows.append({'ring': label, 'n': 0,
                         'MBE': np.nan, 'MAE': np.nan,
                         'RMSE': np.nan, 'STD': np.nan})
            continue

        e    = ring_data['ghi_cams'].values - ring_data['ghi_fcul'].values
        mbe  = float(np.mean(e))
        mae  = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e ** 2)))
        std  = float(np.std(e, ddof=1))

        rows.append({'ring': label, 'n': n,
                     'MBE': mbe, 'MAE': mae, 'RMSE': rmse, 'STD': std})

    return pd.DataFrame(rows)


def plot_metric_bars(ax: plt.Axes,
                     stats: pd.DataFrame,
                     metric: str,
                     ylabel: str,
                     title: str,
                     ring_colors: list[str],
                     show_value_labels: bool = True) -> None:
    """
    Draw a bar chart of one error metric across all distance rings.

    Each bar uses the ring-specific colour from map_distance_rings.py.
    The value is annotated above each bar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    stats : pd.DataFrame
        Output of compute_ring_statistics(), one row per ring.
    metric : str
        Column name in stats to plot (e.g. 'STD', 'RMSE', 'MAE', 'MBE').
    ylabel : str
        Y-axis label.
    title : str
        Panel title.
    ring_colors : list of str
        Hex colours, one per ring (from RING_COLORS).
    show_value_labels : bool
        If True, annotate the numeric value above each bar.
    """
    x_pos  = np.arange(len(stats))
    values = stats[metric].values

    bars = ax.bar(
        x_pos,
        np.abs(values) if metric == 'MBE' else values,
        width     = BAR_WIDTH,
        color     = ring_colors[:len(stats)],
        alpha     = BAR_ALPHA,
        edgecolor = BAR_EDGE_COLOR,
        linewidth = 0.8,
    )

    # For MBE, draw the actual signed value (bar shows absolute, sign in label)
    if show_value_labels:
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            label = f"{val:+.1f}" if metric == 'MBE' else f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(np.nanmax(np.abs(values)), 1) * 0.02,
                label,
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
            )

    # Horizontal zero line for MBE (signed metric)
    if metric == 'MBE':
        ax.axhline(0, color='#333333', linewidth=0.8, linestyle='--')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats['ring'], fontsize=10)
    ax.set_xlabel('Distance from FCUL Reference Station',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Run the full pipeline:
        1. Load FCUL data (standard: Excel + merge; tuk-tuk: already joined).
        2. Load sensor CSV files (CAMS GHI + GPS).
        3. Apply the clear-sky index filter, if enabled.
        4. Compute Haversine distances and classify into rings.
        5. Compute error statistics per ring (MBE, MAE, RMSE, STD).
        6. Plot a 2 × 2 figure of bar charts, one panel per metric.
        7. Save the figure and print the full statistics table.
    """
    print("=" * 70)
    print("  FCUL vs CAMS GHI — ERROR STATISTICS BY DISTANCE RING")
    print(f"  CAMPAIGN     : {CAMPAIGN}")
    print(f"  Data variant : {_VARIANT_TAG}")
    print("=" * 70)
    print(f"\n  Sensor files : {SENSOR_DATA_DIR if CAMPAIGN == 'standard' else TUKTUK_DATA_DIR}")
    print(f"  Output file  : {OUTPUT_FILE}")

    if CAMPAIGN == "standard":
        # Standard campaign: FCUL reference lives in separate Excel files
        # and must be matched to the sensor data by UTC minute.
        print("\n[1/4] Loading FCUL pyranometer Excel files ...")
        fcul = load_fcul_ghi(FCUL_FILES)
        print(f"\n  FCUL time series: {len(fcul):,} valid 1-minute records "
              f"({fcul['timestamp_utc'].min().date()} → "
              f"{fcul['timestamp_utc'].max().date()})")

        print("\n[2/4] Loading sensor CSV files ...")
        frames: list[pd.DataFrame] = []
        for sensor_name, file_path in SENSOR_FILES.items():
            df = load_sensor_data(file_path, sensor_name, include_ghi_fcul=False)
            if not df.empty:
                frames.append(df)

        if not frames:
            raise FileNotFoundError(
                "No sensor CSV files could be loaded. "
                "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
                "FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
            )

        sensors = pd.concat(frames, ignore_index=True)
        print(f"\n  Sensors total: {len(sensors):,} valid rows "
              f"from {', '.join(sorted(sensors['sensor'].unique()))}")

        fcul['timestamp_utc']    = fcul['timestamp_utc'].dt.floor('min')
        sensors['timestamp_utc'] = sensors['timestamp_utc'].dt.floor('min')

        matched   = pd.merge(sensors, fcul, on='timestamp_utc', how='inner')
        n_sensors = len(sensors)
        n_matched = len(matched)
        print(f"  Sensor rows  : {n_sensors:,}")
        print(f"  Matched rows : {n_matched:,}  "
              f"({n_matched / n_sensors * 100:.1f}% of sensor rows)")

        if n_matched == 0:
            raise RuntimeError(
                "No matching timestamps found between FCUL and sensor data. "
                "Check that the date ranges overlap and that time zones are correct."
            )

    else:  # CAMPAIGN == "tuktuk"
        # Tuk-tuk campaign: ghi_fcul is already a column in the per-sensor
        # file (joined in-pipeline by join_ghi_fcul_tuktuk.py, well before
        # clean_data.py) — no separate Excel loading or timestamp merge
        # is needed; the sensor data already IS the matched dataset.
        print("\n[1/4] Skipping separate FCUL loading — the tuk-tuk pipeline "
              "already joins ghi_fcul in-pipeline (join_ghi_fcul_tuktuk.py).")

        print("\n[2/4] Loading sensor CSV files (ghi_fcul already included) ...")
        frames = []
        for sensor_name, file_path in SENSOR_FILES.items():
            df = load_sensor_data(file_path, sensor_name, include_ghi_fcul=True)
            if not df.empty:
                frames.append(df)

        if not frames:
            raise FileNotFoundError(
                "No sensor CSV files could be loaded. "
                "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
                "FILE_SUFFIX/TUKTUK_FILE_SUFFIX. Note the tuk-tuk campaign "
                "requires the join_ghi_fcul_tuktuk.py step to already be present "
                "(ghi_fcul column)."
            )

        matched = pd.concat(frames, ignore_index=True)
        print(f"\n  Sensor rows total: {len(matched):,}  "
              f"({', '.join(sorted(matched['sensor'].unique()))})")

    # ── Apply the clear-sky index filter, if enabled ───────────────────────
    print("\n[3/4] Applying filters ...")
    n_before = len(matched)

    if ENABLE_CLEAR_SKY_FILTER:
        print(f"  Filter 2 — clear-sky index "
              f"(threshold {MIN_CLEAR_SKY_INDEX:.2f}):")
        matched = filter_clear_sky(matched)
    else:
        print("  Filter 2 — clear-sky index    : DISABLED")

    n_after = len(matched)
    print(f"\n  Before filtering : {n_before:,}")
    print(f"  After  filtering : {n_after:,}  "
          f"({n_before - n_after:,} removed, "
          f"{(n_before - n_after) / n_before * 100:.1f}%)")

    if n_after == 0:
        raise RuntimeError(
            "No measurements remain after filtering. "
            "Consider relaxing MIN_CLEAR_SKY_INDEX."
        )

    # ── Distance computation and ring classification ──────────────────────
    print("\n[4/4] Computing distances, classifying into rings, "
          "and computing statistics ...")

    matched['distance_km'] = haversine_km(
        FCUL_LAT, FCUL_LON,
        matched['lat'].values, matched['lon'].values,
    )
    print(f"  Distance range: "
          f"{matched['distance_km'].min():.2f} – "
          f"{matched['distance_km'].max():.2f} km")

    matched = filter_to_analysis_range(matched, RING_BOUNDARIES_KM)

    matched['ring'] = classify_into_rings(
        matched['distance_km'].values, RING_BOUNDARIES_KM
    )
    ring_labels = create_ring_labels(RING_BOUNDARIES_KM)

    # ── Compute statistics ────────────────────────────────────────────────────
    stats = compute_ring_statistics(matched, ring_labels)

    # ── Build 2 × 2 figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    axes_flat = axes.flatten()

    panels = [
        ('STD',  'Standard Deviation  [W/m²]',
         'Standard Deviation of Residuals'),
        ('RMSE', 'RMSE  [W/m²]',
         'Root Mean Square Error'),
        ('MAE',  'MAE  [W/m²]',
         'Mean Absolute Error'),
        ('MBE',  '|MBE|  [W/m²]  (sign annotated)',
         'Mean Bias Error  (+  CAMS overestimates)'),
    ]

    for ax, (metric, ylabel, title) in zip(axes_flat, panels):
        plot_metric_bars(ax, stats, metric, ylabel, title, RING_COLORS)

    # Active filters in the figure subtitle
    active = [f"Min GHI (ghi_fcul): {MIN_IRRADIANCE_WM2:.0f} W/m²"]
    if ENABLE_CLEAR_SKY_FILTER:
        active.append(f"CSI ≥ {MIN_CLEAR_SKY_INDEX:.2f}")

    fig.suptitle(
        f'CAMS Satellite GHI vs FCUL Pyranometer — Error Statistics by Distance Ring  [{CAMPAIGN} / {_VARIANT_TAG}]\n'
        f'Residual = CAMS − FCUL   |   '
        f'Sensors: {", ".join(sorted(matched["sensor"].unique()))}   |   '
        f'{", ".join(active)}',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ERROR STATISTICS SUMMARY  (residual = CAMS − FCUL, W/m²)")
    print("=" * 70)
    print(f"  {'Ring':>10}  {'n':>7}  {'MBE':>8}  {'MAE':>8}  "
          f"{'RMSE':>8}  {'STD':>8}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for _, row in stats.iterrows():
        print(f"  {row['ring']:>10}  {int(row['n']):>7,}  "
              f"{row['MBE']:>+8.2f}  {row['MAE']:>8.2f}  "
              f"{row['RMSE']:>8.2f}  {row['STD']:>8.2f}")
    print(f"\n  Figure saved → {OUTPUT_FILE}  ({OUTPUT_DPI} dpi)")
    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


# ==============================================================================
if __name__ == "__main__":
    main()