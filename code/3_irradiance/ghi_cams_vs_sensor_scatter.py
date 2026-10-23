"""
================================================================================
CAMS Satellite GHI vs Calibrated Sensor GHI — Scatter Plot by Distance Ring
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Compares the calibrated irradiance of the Solar Jinie onboard sensors with
the CAMS satellite GHI at the same place and time, by distance ring from the
FCUL reference station. One scatter panel per ring:

    X : calibrated onboard sensor irradiance  [W/m²]  ('<SENSOR> Rad calib')
    Y : CAMS satellite GHI                    [W/m²]  ('ghi')

with a forced-origin regression y = slope · x, whose slope measures the
systematic bias of CAMS relative to the onboard sensor.

Unlike ghi_fcul_vs_cams_scatter.py, whose reference is the unobstructed
FCUL pyranometer, the reference here is the moving sensor itself: deep
shading, cloud passages and tunnels are all kept, so that the full range of
in-vehicle irradiance is visible. The "arc" of
points above the 1:1 line and the cluster of near-zero sensor values under
high CAMS values are features of interest, not artefacts.

Distance rings (RING_BOUNDARIES_KM; points beyond 32 km are excluded):
    0–3 km   : inner zone (≈ CAMS pixel, ~3 × 3 km²)
    3–9 km   : intermediate zone
    9–18 km  : outer reliable zone
    18–32 km : far zone (highest spatial uncertainty)

Rows kept: valid coordinates (non-zero), rad_calib >= MIN_IRRADIANCE_WM2,
ghi_clear > 0. No threshold is applied to ghi here: the ghi > 25 W/m² floor
is the Step 2 filter of clean_data.py, present in the '_hq' variants only.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (FILE_SUFFIX / TUKTUK_FILE_SUFFIX), one
             file per sensor
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             required columns: '<SENSOR> lat', '<SENSOR> lon',
             '<SENSOR> Rad calib', 'ghi', 'ghi_clear', 'sun_elev'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             mdr_ghi_cams_vs_sensor_scatter_rings_<CAMPAIGN>_<variant_tag>.png

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.1
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

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

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py variant) -------------------------------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output directory ----------------------------------------------------------
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

# --- Minimum sensor irradiance (rad_calib) [W/m²] ----------------------------
MIN_IRRADIANCE_WM2: float = 25.0

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
# Input and output paths follow from CAMPAIGN and the input suffix.

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
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR:   Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_FILE:  Path = _build_output_path(
    OUTPUT_DIR, 'mdr_ghi_cams_vs_sensor_scatter_rings', CAMPAIGN, _VARIANT_TAG, 'png'
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


def load_sensor_data(file_path: Path, sensor_name: str) -> pd.DataFrame:
    """
    Load one cleaned sensor CSV file and extract the columns needed for the
    sensor vs CAMS GHI scatter analysis.

    Columns extracted and renamed:
        lat      ← J3x lat
        lon      ← J3x lon
        rad_cal  ← J3x Rad calib   (calibrated onboard irradiance, W/m²)
        ghi      ← ghi             (CAMS satellite GHI, W/m²)
        ghi_clear← ghi_clear       (CAMS clear-sky GHI, W/m²)
        sun_elev ← sun_elev        (solar elevation angle, degrees)

    The input files are clean_data.py outputs (invalid, incomplete and
    sentinel rows already removed). Here, rows with missing or zero
    coordinates are dropped, MIN_IRRADIANCE_WM2 is applied to rad_cal
    (night-time noise) and ghi_clear > 0 is required (no division by zero if
    the clear-sky index is computed downstream).

    Parameters
    ----------
    file_path : Path
        Path to the cleaned sensor CSV file.
    sensor_name : str
        Sensor identifier, e.g. 'j31'. Used for log messages and the
        'sensor' column in the output.

    Returns
    -------
    pd.DataFrame
        Columns: [lat, lon, rad_cal, ghi, ghi_clear, sun_elev, sensor].
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
    col_lat     = f"{sensor_upper} lat"
    col_lon     = f"{sensor_upper} lon"
    col_rad_cal = f"{sensor_upper} Rad calib"

    # Fallback for lat/lon: case-insensitive search (excludes land_use columns)
    if col_lat not in df.columns:
        candidates = [c for c in df.columns
                      if 'lat' in c.lower() and 'land' not in c.lower()]
        col_lat = candidates[0] if candidates else None

    if col_lon not in df.columns:
        candidates = [c for c in df.columns
                      if 'lon' in c.lower() and 'land' not in c.lower()]
        col_lon = candidates[0] if candidates else None

    required = [col_lat, col_lon, col_rad_cal, 'ghi', 'ghi_clear', 'sun_elev']
    missing  = [c for c in required if c is None or c not in df.columns]
    if missing:
        print(f"  [ERROR] Missing columns in {file_path.name}: {missing}")
        return pd.DataFrame()

    data = df[[col_lat, col_lon, col_rad_cal,
               'ghi', 'ghi_clear', 'sun_elev']].copy()
    data.columns = ['lat', 'lon', 'rad_cal', 'ghi', 'ghi_clear', 'sun_elev']

    # ── Coerce numeric columns ─────────────────────────────────────────────────
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # ── Drop NaN and invalid coordinates ──────────────────────────────────────
    data = data.dropna()
    data = data[data['lat'].between(-90, 90) & data['lon'].between(-180, 180)]
    data = data[(data['lat'] != 0.0) & (data['lon'] != 0.0)]

    # ── Minimum irradiance on rad_cal (the ghi floor is clean_data.py Step 2) ─
    data = data[
        (data['rad_cal']   >= MIN_IRRADIANCE_WM2) &
        (data['ghi_clear'] >  0)                    # avoid division-by-zero
    ]

    if data.empty:
        print(f"  [WARNING] No valid rows in {file_path.name} – skipped.")
        return pd.DataFrame()

    data['sensor'] = sensor_upper
    n_valid = len(data)
    print(f"  {sensor_upper}: {n_valid:,} valid rows "
          f"loaded from '{file_path.name}'")
    return data.reset_index(drop=True)


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

    The scatter intentionally includes all sky conditions (shading, partial
    cloud cover, tunnels) — these are features of interest in the VIPV context,
    not noise to be removed.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_data : array-like
        Calibrated sensor irradiance values  (W/m²)  — X-axis.
    y_data : array-like
        CAMS satellite GHI values            (W/m²)  — Y-axis.
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
    ax.set_xlabel('Calibrated Sensor GHI  [W/m²]',  fontsize=11)
    ax.set_ylabel('CAMS Satellite GHI     [W/m²]',   fontsize=11)
    ax.set_title(f'Distance ring  {ring_label}\n({n_points:,} measurements)',
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
        1. Load all cleaned sensor CSV files.
           (minimum irradiance on rad_calib, valid coordinates).
        2. Compute Haversine distances from FCUL and classify into rings.
        3. Plot one scatter panel per ring (sensor GHI vs CAMS GHI).
        4. Save the 2 × 2 figure.
    """
    print("=" * 70)
    print("  CAMS SATELLITE GHI vs CALIBRATED SENSOR GHI — SCATTER BY RING")
    print(f"  CAMPAIGN     : {CAMPAIGN}")
    print(f"  Data variant : {_VARIANT_TAG}")
    print("=" * 70)
    print(f"\n  Input files  : {DATA_DIR if CAMPAIGN == 'standard' else TUKTUK_DATA_DIR}")
    print(f"  Output file  : {OUTPUT_FILE}")

    # ── Step 1: Load all sensor data ──────────────────────────────────────────
    print("\n[1/3] Loading sensor CSV files ...")

    frames: list[pd.DataFrame] = []
    for sensor_name, file_path in SENSOR_FILES.items():
        df = load_sensor_data(file_path, sensor_name)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, and "
            "FILE_SUFFIX/TUKTUK_FILE_SUFFIX."
        )

    combined = pd.concat(frames, ignore_index=True)
    total_loaded = len(combined)
    print(f"\n  Total: {total_loaded:,} measurements "
          f"from {', '.join(sorted(combined['sensor'].unique()))}")

    n_after = len(combined)
    if n_after == 0:
        raise RuntimeError(
            "No measurements remain after filtering. "
            "Consider raising MIN_IRRADIANCE_WM2."
        )

    # ── Step 2: Distance computation and ring classification ──────────────────
    print("\n[2/3] Computing distances from FCUL and classifying into rings ...")

    combined['distance_km'] = haversine_km(
        FCUL_LAT, FCUL_LON,
        combined['lat'].values, combined['lon'].values,
    )
    print(f"  Distance range: "
          f"{combined['distance_km'].min():.2f} – "
          f"{combined['distance_km'].max():.2f} km")

    # Exclude points beyond the outer ring boundary
    combined = filter_to_analysis_range(combined, RING_BOUNDARIES_KM)

    combined['ring'] = classify_into_rings(
        combined['distance_km'].values, RING_BOUNDARIES_KM
    )
    ring_labels = create_ring_labels(RING_BOUNDARIES_KM)

    print("\n  Ring distribution:")
    for i, label in enumerate(ring_labels):
        count = int((combined['ring'] == i).sum())
        pct   = count / len(combined) * 100
        print(f"    {label:>10}: {count:>7,} measurements ({pct:5.1f}%)")

    # ── Step 3: Build the 2 × 2 scatter figure ────────────────────────────────
    print("\n[3/3] Generating scatter plots ...")

    fig, axes = plt.subplots(SUBPLOT_ROWS, SUBPLOT_COLS, figsize=FIGURE_SIZE)
    axes_flat = axes.flatten()

    for ring_idx, (ax, label) in enumerate(zip(axes_flat, ring_labels)):
        ring_data = combined[combined['ring'] == ring_idx]
        n_pts     = len(ring_data)

        if n_pts == 0:
            ax.set_visible(False)
            print(f"  Ring {label}: no data — panel hidden")
            continue

        print(f"  Ring {label}: {n_pts:,} points  →  plotting ...")
        plot_scatter_ring(
            ax,
            ring_data['rad_cal'].values, ring_data['ghi'].values,
            label, n_pts,
            scatter_color    = RING_COLORS[ring_idx],
            regression_color = RING_REGRESSION_COLORS[ring_idx],
        )

    # Figure subtitle
    subtitle = f"Min irradiance (rad_cal): {MIN_IRRADIANCE_WM2:.0f} W/m²"

    fig.suptitle(
        f'CAMS Satellite GHI vs Calibrated Sensor GHI by Distance Ring  [{CAMPAIGN} / {_VARIANT_TAG}]\n'
        f'All sky conditions (shading included)   —   {subtitle}',
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
    print(f"  Sensors       : {', '.join(sorted(combined['sensor'].unique()))}")
    print(f"  Total plotted : {len(combined):,} measurements")
    print()
    print(f"  {'Ring':>10}  {'n':>7}  {'slope':>8}  {'R²':>8}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}")
    for i, label in enumerate(ring_labels):
        ring_data = combined[combined['ring'] == i]
        if len(ring_data) > 0:
            slope, r2 = fit_forced_origin(
                ring_data['rad_cal'].values,
                ring_data['ghi'].values,
            )
            print(f"  {label:>10}  {len(ring_data):>7,}  {slope:>8.4f}  {r2:>8.4f}")
    print(f"\n  Figure saved → {OUTPUT_FILE}  ({OUTPUT_DPI} dpi)")
    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


# ==============================================================================
if __name__ == "__main__":
    main()