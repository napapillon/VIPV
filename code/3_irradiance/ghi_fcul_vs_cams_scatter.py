"""
================================================================================
FCUL Pyranometer GHI vs CAMS Satellite GHI — Scatter Plot by Distance Ring
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Compares the ground GHI of the FCUL rooftop pyranometer (Eko MS-802) with the
CAMS satellite GHI carried by the Solar Jinie sensor files at the same UTC
minute, by distance ring between the vehicle and the station. One scatter
panel per ring:

    X : FCUL pyranometer GHI  [W/m²]  (ground reference)
    Y : CAMS satellite GHI    [W/m²]

with a forced-origin regression y = slope · x. The slope measures the
systematic bias of CAMS relative to FCUL; its change with distance shows the
spatial representativeness of the satellite pixel relative to the station.

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
             mdr_ghi_fcul_vs_cams_scatter_rings_<CAMPAIGN>_<variant_tag>.png

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
# absent; negative GHI clipped to 0 W/m² in load_fcul_ghi(); gaps and
# anomalies documented in the NOTES_ sheet of each file.
FCUL_DATA_DIR = FCUL_DIR

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

# --- Ring colours (identical to map_distance_rings.py → DISTANCE_COLORS) ------
RING_COLORS: list[str] = [
    "#2166ac",   # 0–3 km   : blue
    "#4dac26",   # 3–9 km   : green
    "#f4a582",   # 9–18 km  : light orange
    "#d6604d",   # 18–32 km : red-orange
]

# Darker variants for the regression line (better contrast on scatter cloud)
RING_REGRESSION_COLORS: list[str] = [
    "#0d3d6e",   # 0–3 km   : dark blue
    "#2a6614",   # 3–9 km   : dark green
    "#c0392b",   # 9–18 km  : dark red
    "#7b1a10",   # 18–32 km : very dark red
]

# --- Scatter plot styling ------------------------------------------------------
SCATTER_ALPHA:  float = 0.15        # transparency (many overlapping points)
SCATTER_SIZE:   int   = 4           # marker size in points²
REGRESSION_LW:  float = 2.0         # regression line width
IDENTITY_COLOR: str   = '#333333'   # 1:1 reference line colour
IDENTITY_LW:    float = 1.2         # 1:1 line width
IDENTITY_LS:    str   = '--'        # 1:1 line style
AXIS_MAX_WM2:   int   = 1400        # upper limit for both axes (W/m²)

# --- Layout -------------------------------------------------------------------
FIGURE_SIZE:  tuple = (14, 12)
SUBPLOT_ROWS: int   = 2
SUBPLOT_COLS: int   = 2

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
    OUTPUT_DIR, 'mdr_ghi_fcul_vs_cams_scatter_rings', CAMPAIGN, _VARIANT_TAG, 'png'
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
    Callers must exclude these with filter_to_analysis_range() before calling
    this function.

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

    Points at or beyond boundaries[-1] (e.g. ≥ 32 km) are outside all
    defined rings and must not be silently absorbed into the last ring.

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
# DATA LOADING
# ==============================================================================

def load_fcul_ghi(fcul_files: list) -> pd.DataFrame:
    """
    Load and concatenate FCUL pyranometer GHI data from all Excel files.

    Each file contains columns:
        - 'GMT TIME'   : UTC timestamp (1-minute resolution)
        - 'Avg_GHI'    : averaged GHI measurement [W/m²]

    The first two rows of each sheet contain unit headers and are dropped by
    filtering on rows where 'GMT TIME' parses as a valid datetime.

    Data notes (from the FCUL data provider)
    -----------------------------------------
    - January – May 2022 are missing from the supplied files, so this period
      has no matched rows (expected). The 2022 data starts in
      GHI2022_part2.xlsx.
    - Negative GHI values (night-time instrument offset) are clipped to
      0 W/m², as instructed by the provider; MIN_IRRADIANCE_WM2 is then
      applied to the clipped series (night-time and near-zero records).

    Parameters
    ----------
    fcul_files : list of (Path, str)
        List of (file_path, sheet_name) tuples.

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp_utc', 'ghi_fcul']
        'timestamp_utc' is a timezone-naive UTC datetime rounded to the minute.
        Rows with ghi_fcul < MIN_IRRADIANCE_WM2 are already excluded.
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

        # Round to minute (already 1-min data, but ensures exact join key)
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
    FCUL vs CAMS GHI comparison.

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
        Sensor identifier, e.g. 'j31'. Used for log messages and the
        'sensor' column in the output.
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
    # Convert to tz-naive UTC (strip timezone info after ensuring UTC)
    data['timestamp_utc'] = data['ts_raw'].dt.tz_convert('UTC') \
                                          .dt.tz_localize(None) \
                                          .dt.floor('min')
    data = data.drop(columns='ts_raw')

    # ── Coerce numeric columns ─────────────────────────────────────────────────
    numeric_cols = ['lat', 'lon', 'ghi_cams', 'ghi_clear', 'sun_elev']
    if include_ghi_fcul:
        numeric_cols.append('ghi_fcul')
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # ── Drop NaN, invalid coordinates ─────────────────────────────────────────
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

    data['sensor'] = sensor_name.upper()
    print(f"  {sensor_upper}: {len(data):,} valid rows "
          f"loaded from '{file_path.name}'")
    return data.reset_index(drop=True)


# ==============================================================================
# FILTER FUNCTIONS
# ==============================================================================

# ==============================================================================
# REGRESSION AND PLOTTING
# ==============================================================================

def fit_forced_origin(x: np.ndarray,
                      y: np.ndarray) -> tuple[float, float]:
    """
    Fit a forced-origin linear regression  y = slope · x  using OLS.

    The closed-form least-squares solution with intercept constrained to zero:
        slope = Σ(x · y) / Σ(x²)

    Parameters
    ----------
    x, y : array-like
        Paired irradiance values (no NaNs).

    Returns
    -------
    slope : float
        Best-fit slope of the forced-origin regression.
    r_squared : float
        Coefficient of determination R² relative to a zero-intercept model.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope     = np.sum(x * y) / np.sum(x ** 2)
    y_pred    = slope * x
    ss_res    = np.sum((y - y_pred) ** 2)
    ss_tot    = np.sum(y ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, r_squared


def plot_scatter_ring(ax: plt.Axes,
                      x_data: np.ndarray,
                      y_data: np.ndarray,
                      ring_label: str,
                      n_points: int,
                      scatter_color: str,
                      regression_color: str) -> None:
    """
    Draw one scatter panel (one distance ring) on a given Axes object.

    The panel shows:
        - Scatter cloud coloured by ring (same palette as map_distance_rings.py)
        - Forced-origin regression line  y = slope · x
        - 1:1 identity reference line
        - Annotation box with slope, R², and point count

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_data : array-like
        FCUL pyranometer GHI values  (W/m²)  — X-axis (ground reference).
    y_data : array-like
        CAMS satellite GHI values    (W/m²)  — Y-axis.
    ring_label : str
        Human-readable ring label, e.g. '0–3 km'.
    n_points : int
        Number of data points (shown in title and annotation box).
    scatter_color : str
        Hex colour for the scatter points (from RING_COLORS).
    regression_color : str
        Hex colour for the regression line (from RING_REGRESSION_COLORS).
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)

    # ── Scatter cloud ─────────────────────────────────────────────────────────
    ax.scatter(x, y,
               color=scatter_color, alpha=SCATTER_ALPHA,
               s=SCATTER_SIZE, linewidths=0,
               rasterized=True, label='Measurements')

    # ── Forced-origin regression ──────────────────────────────────────────────
    slope, r2 = fit_forced_origin(x, y)
    x_line    = np.array([0, AXIS_MAX_WM2])
    ax.plot(x_line, slope * x_line,
            color=regression_color, linewidth=REGRESSION_LW,
            label=f'y = {slope:.4f}x')

    # ── 1:1 identity line ─────────────────────────────────────────────────────
    ax.plot([0, AXIS_MAX_WM2], [0, AXIS_MAX_WM2],
            color=IDENTITY_COLOR, linewidth=IDENTITY_LW,
            linestyle=IDENTITY_LS, label='y = x  (1:1)')

    # ── Annotation box ────────────────────────────────────────────────────────
    ax.text(0.04, 0.96,
            f"y = {slope:.4f}x\n$R^2$ = {r2:.4f}\nn = {n_points:,}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#AAAAAA', alpha=0.9))

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0, AXIS_MAX_WM2)
    ax.set_ylim(0, AXIS_MAX_WM2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('FCUL Pyranometer GHI  [W/m²]',  fontsize=11)
    ax.set_ylabel('CAMS Satellite GHI    [W/m²]',   fontsize=11)
    ax.set_title(f'Distance ring  {ring_label}\n({n_points:,} matched measurements)',
                 fontsize=12, fontweight='bold', pad=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Run the full pipeline:
        1. Load FCUL data (standard: Excel + merge; tuk-tuk: already joined).
        2. Load sensor CSV files (CAMS GHI + GPS).
        3. Compute Haversine distances and classify into rings.
        4. Plot one scatter panel per ring (FCUL GHI vs CAMS GHI).
        5. Save the 2 × 2 figure.
    """
    print("=" * 70)
    print("  FCUL PYRANOMETER GHI vs CAMS SATELLITE GHI — SCATTER BY RING")
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
        matched = pd.merge(sensors, fcul, on='timestamp_utc', how='inner')

        n_sensors = len(sensors)
        n_matched = len(matched)
        match_pct = n_matched / n_sensors * 100 if n_sensors else 0
        print(f"  Sensor rows       : {n_sensors:,}")
        print(f"  Matched rows      : {n_matched:,}  ({match_pct:.1f}% of sensor rows)")

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

    if len(matched) == 0:
        raise RuntimeError(
            "No measurements remain. Consider relaxing MIN_IRRADIANCE_WM2."
        )

    # ── Distance classification ────────────────────────────────────────────
    print("\n[3/4] Computing distances and classifying into rings ...")

    matched['distance_km'] = haversine_km(
        FCUL_LAT, FCUL_LON,
        matched['lat'].values, matched['lon'].values,
    )
    print(f"  Distance range: "
          f"{matched['distance_km'].min():.2f} – "
          f"{matched['distance_km'].max():.2f} km")

    # Exclude points beyond the outer ring boundary
    matched = filter_to_analysis_range(matched, RING_BOUNDARIES_KM)

    matched['ring'] = classify_into_rings(
        matched['distance_km'].values, RING_BOUNDARIES_KM
    )
    ring_labels = create_ring_labels(RING_BOUNDARIES_KM)

    print("\n  Ring distribution after all filters:")
    for i, label in enumerate(ring_labels):
        count = int((matched['ring'] == i).sum())
        pct   = count / len(matched) * 100
        print(f"    {label:>10}: {count:>7,} matched measurements ({pct:5.1f}%)")

    # ── Step 4: Build the 2 × 2 scatter figure ────────────────────────────────
    print("\n[4/4] Generating scatter plots ...")
    fig, axes = plt.subplots(SUBPLOT_ROWS, SUBPLOT_COLS, figsize=FIGURE_SIZE)
    axes_flat = axes.flatten()

    for ring_idx, (ax, label) in enumerate(zip(axes_flat, ring_labels)):
        ring_data = matched[matched['ring'] == ring_idx]
        n_pts     = len(ring_data)

        if n_pts == 0:
            ax.set_visible(False)
            print(f"  Ring {label}: no data — panel hidden")
            continue

        print(f"  Ring {label}: {n_pts:,} points  →  plotting ...")
        plot_scatter_ring(
            ax,
            ring_data['ghi_fcul'], ring_data['ghi_cams'],
            label, n_pts,
            scatter_color    = RING_COLORS[ring_idx],
            regression_color = RING_REGRESSION_COLORS[ring_idx],
        )

    subtitle = f"Min irradiance (ghi_fcul): {MIN_IRRADIANCE_WM2:.0f} W/m²"

    fig.suptitle(
        f'CAMS Satellite GHI vs FCUL Pyranometer GHI by Distance Ring  [{CAMPAIGN} / {_VARIANT_TAG}]\n'
        f'Reference: FCUL Eko MS-802 pyranometer   —   {subtitle}',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Sensors         : {', '.join(sorted(matched['sensor'].unique()))}")
    print(f"  Matched & plotted: {len(matched):,} measurements")
    print()
    print(f"  {'Ring':>10}  {'n':>7}  {'slope':>8}  {'R²':>8}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}")
    for i, label in enumerate(ring_labels):
        ring_data = matched[matched['ring'] == i]
        if len(ring_data) > 0:
            slope, r2 = fit_forced_origin(
                ring_data['ghi_fcul'].values,
                ring_data['ghi_cams'].values,
            )
            print(f"  {label:>10}  {len(ring_data):>7,}  {slope:>8.4f}  {r2:>8.4f}")
    print(f"\n  Figure saved → {OUTPUT_FILE}  ({OUTPUT_DPI} dpi)")
    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


# ==============================================================================
if __name__ == "__main__":
    main()