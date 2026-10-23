"""
================================================================================
Measurement Distribution by Distance Rings — Stacked Bar Chart by Sensor
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Variant of measurement_rings_bar_chart.py in which each distance-ring bar is
split into one coloured segment per sensor. Configuration constants are the
same as in measurement_rings_bar_chart.py, so both charts can be compared
directly.
    0–3 km   : inner zone (≈ CAMS pixel size)
    3–9 km   : intermediate zone (highest measurement density)
    9–18 km  : outer reliable zone
    18–32 km : far zone (fewer measurements, higher uncertainty)
Points beyond 32 km are not counted. No extra filtering is applied: the
input files are clean_data.py outputs.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: '<SENSOR> lat', '<SENSOR> lon'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             mdr_measurement_rings_stacked_by_sensor_<CAMPAIGN>_<variant_tag>.png

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
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR = STANDARD_DIR                  # CAMPAIGN = "standard"
TUKTUK_DATA_DIR = TUKTUK_DIR             # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output directory ----------------------------------------------------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "4_mobility_and_sampling" / "mdr_outputs"

# --- FCUL reference station coordinates (WGS-84 decimal degrees) --------------
# Pyranometer on the rooftop of building C1, Faculty of Sciences,
# University of Lisbon, campus Cidade Universitária.
FCUL_LAT: float = 38.7565    # Google Maps: 38.756594832619776
FCUL_LON: float = -9.1564    # Google Maps: -9.156483276747663

# --- Distance ring boundaries (km) --------------------------------------------
# 3 km  → CAMS pixel size (~3 × 3 km)
# 9 km  → intermediate zone
# 18 km → threshold beyond which ground-based reference becomes unreliable
# 32 km → outer limit of the dataset
# Points strictly beyond MAX_DISTANCE_KM are excluded before classification:
# they fall outside all defined rings and must not be silently absorbed.
RING_BOUNDARIES_KM: list[int] = [0, 3, 9, 18, 32]
MAX_DISTANCE_KM:    int       = RING_BOUNDARIES_KM[-1]   # 32 km

# --- Earth radius -------------------------------------------------------------
R_EARTH_KM: float = 6371.0   # Mean Earth radius (km) – IAU 2012

# --- Sensor colours -----------------------------------------------------------
# One distinctive colour per sensor for the stacked bar segments
SENSOR_COLORS: dict[str, str] = {
    'j30': '#2166ac',   # blue
    'j31': '#4dac26',   # green
    'j32': '#d6604d',   # red-orange
    'j33': '#9970ab',   # purple
}

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
    """
    Build an output file path that embeds the active campaign and variant tag.

    Parameters
    ----------
    out_dir : Path
        Output directory (already resolved).
    base_name : str
        Descriptive part of the filename
        (e.g. 'mdr_measurement_rings_stacked_by_sensor').
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().
    extension : str
        File extension without leading dot (e.g. 'png').

    Returns
    -------
    Path
        Complete file path,
        e.g. out_dir / 'mdr_measurement_rings_stacked_by_sensor_standard_sd_cleaned.png'.
    """
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolved at startup from the configuration above
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR:   Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_FILE:  Path = _build_output_path(
    OUTPUT_DIR, 'mdr_measurement_rings_stacked_by_sensor', CAMPAIGN, _VARIANT_TAG, 'png'
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS  (same logic as measurement_rings_bar_chart.py)
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
    Assign each measurement to a distance ring.

    Ring i covers: boundaries[i] ≤ d < boundaries[i+1].

    All input distances must be strictly within [boundaries[0], boundaries[-1]).
    Points at or beyond boundaries[-1] must be excluded BEFORE calling this
    function (see main() filtering step). Passing out-of-range distances raises
    a ValueError to prevent silent data corruption.

    Parameters
    ----------
    distances_km : np.ndarray
        Distances from FCUL in kilometres. Must all be < boundaries[-1].
    boundaries : list of int
        Sorted ring boundary values in km.

    Returns
    -------
    np.ndarray of int
        Ring index (0-based) for each distance.

    Raises
    ------
    ValueError
        If any distance is >= boundaries[-1].
    """
    distances_km = np.asarray(distances_km)

    if np.any(distances_km >= boundaries[-1]):
        raise ValueError(
            f"classify_into_rings() received distances >= {boundaries[-1]} km. "
            f"Exclude out-of-range points before calling this function."
        )

    ring_indices = np.zeros(len(distances_km), dtype=int)
    for i in range(len(boundaries) - 1):
        mask = (distances_km >= boundaries[i]) & (distances_km < boundaries[i + 1])
        ring_indices[mask] = i
    return ring_indices


def create_ring_labels(boundaries: list) -> list[str]:
    """
    Build human-readable labels for each distance ring.

    For boundaries [0, 3, 9, 18, 32] the function returns:
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


def load_sensor_data(file_path: Path, sensor_name: str) -> pd.DataFrame:
    """
    Load one cleaned sensor CSV file and return valid latitude/longitude pairs.

    The input files are the output of clean_data.py, which has already removed
    invalid rows (land_use_clcv out of range, zero irradiance, etc.).
    This function therefore only validates coordinate columns.

    Column name detection
    ---------------------
    The function first looks for the exact column names produced by the data
    pipeline, e.g. 'J31 lat' and 'J31 lon' (uppercase sensor ID).  If not
    found, it falls back to a case-insensitive search, excluding 'land_use_clcv'
    and 'land_use_str' which also contain the substring 'lat' / 'lon'.

    Parameters
    ----------
    file_path : Path
        Path to the cleaned CSV file.
    sensor_name : str
        Sensor identifier, e.g. 'j31'. Used to build expected column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['lat', 'lon', 'sensor'] for valid rows.
        Returns an empty DataFrame if the file is missing or unusable.
    """
    if not file_path.is_file():
        print(f"  [WARNING] File not found: {file_path}  – skipped.")
        return pd.DataFrame(columns=['lat', 'lon', 'sensor'])

    try:
        df = pd.read_csv(file_path, low_memory=False)

        # ── Locate lat / lon columns ──────────────────────────────────────────
        sensor_upper = sensor_name.upper()   # 'j31' → 'J31'
        lat_col = f"{sensor_upper} lat"
        lon_col = f"{sensor_upper} lon"

        # Fallback: case-insensitive search, excluding land_use columns
        if lat_col not in df.columns:
            candidates = [c for c in df.columns
                          if 'lat' in c.lower() and 'land' not in c.lower()]
            lat_col = candidates[0] if candidates else None

        if lon_col not in df.columns:
            candidates = [c for c in df.columns
                          if 'lon' in c.lower() and 'land' not in c.lower()]
            lon_col = candidates[0] if candidates else None

        if lat_col is None or lon_col is None:
            print(f"  [WARNING] Could not find lat/lon columns in "
                  f"{file_path.name} – skipped.")
            return pd.DataFrame(columns=['lat', 'lon', 'sensor'])

        # ── Extract and validate coordinates ──────────────────────────────────
        subset = df[[lat_col, lon_col]].copy()
        subset.columns = ['lat', 'lon']

        # Coerce to float: converts any stray text / comma-decimal values to NaN
        subset['lat'] = pd.to_numeric(subset['lat'], errors='coerce')
        subset['lon'] = pd.to_numeric(subset['lon'], errors='coerce')

        subset = subset.dropna()
        subset = subset[
            subset['lat'].between(-90, 90) & subset['lon'].between(-180, 180)
        ]
        subset = subset[(subset['lat'] != 0.0) & (subset['lon'] != 0.0)]

        if subset.empty:
            print(f"  [WARNING] No valid GPS points in {file_path.name} – skipped.")
            return pd.DataFrame(columns=['lat', 'lon', 'sensor'])

        subset['sensor'] = sensor_name
        print(f"  {sensor_upper}: {len(subset):,} valid GPS points "
              f"loaded from '{file_path.name}'")
        return subset.reset_index(drop=True)

    except Exception as exc:
        print(f"  [ERROR] Could not load {sensor_name}: {exc}")
        return pd.DataFrame(columns=['lat', 'lon', 'sensor'])


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Run the full pipeline: load cleaned sensor data, compute Haversine distances
    from FCUL, classify into distance rings, and save the stacked bar chart.
    """
    print("=" * 70)
    print("  MEASUREMENT DISTRIBUTION BY DISTANCE RINGS — stacked by sensor")
    print(f"  CAMPAIGN     : {CAMPAIGN}")
    print(f"  Data variant : {_VARIANT_TAG}")
    print("=" * 70)
    print(f"\n  Input files  : {DATA_DIR}")
    print(f"  Output file  : {OUTPUT_FILE}")

    # ── 1. Load all sensor CSV files ──────────────────────────────────────────
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

    combined       = pd.concat(frames, ignore_index=True)
    total_count    = len(combined)
    sensors_loaded = sorted(combined['sensor'].unique())
    print(f"\n  Total: {total_count:,} measurements "
          f"from {', '.join(s.upper() for s in sensors_loaded)}")

    # ── 2. Compute distances and classify into rings ───────────────────────────
    print("\n[2/3] Computing Haversine distances from FCUL ...")

    combined['distance_km'] = haversine_km(
        FCUL_LAT, FCUL_LON,
        combined['lat'].values, combined['lon'].values,
    )
    print(f"  Distance range: "
          f"{combined['distance_km'].min():.2f} – "
          f"{combined['distance_km'].max():.2f} km")

    # Exclude points beyond the outer ring boundary — they fall outside all
    # defined analysis zones and must not be silently absorbed into the last ring.
    mask_in_range  = combined['distance_km'] < MAX_DISTANCE_KM
    n_out_of_range = int((~mask_in_range).sum())
    if n_out_of_range:
        print(f"  [INFO] {n_out_of_range:,} points beyond {MAX_DISTANCE_KM} km "
              f"excluded (outside all defined rings).")
    combined    = combined[mask_in_range].reset_index(drop=True)
    total_count = len(combined)
    print(f"  Points within rings: {total_count:,}")

    # ── 3. Classify into rings and build stacked bar chart ────────────────────
    print("\n[3/3] Classifying into distance rings and creating bar chart ...")

    combined['ring'] = classify_into_rings(
        combined['distance_km'].values, RING_BOUNDARIES_KM
    )
    ring_labels = create_ring_labels(RING_BOUNDARIES_KM)

    # Build a count matrix: rows = rings, columns = sensors
    count_matrix = pd.DataFrame(
        0, index=range(len(ring_labels)), columns=sensors_loaded
    )
    for ring_idx in range(len(ring_labels)):
        ring_data = combined[combined['ring'] == ring_idx]
        for sensor in sensors_loaded:
            count_matrix.loc[ring_idx, sensor] = int(
                (ring_data['sensor'] == sensor).sum()
            )

    print("\n  Ring distribution by sensor:")
    header = (f"  {'Ring':>10}"
              + "".join(f"  {s.upper():>8}" for s in sensors_loaded)
              + f"  {'Total':>8}")
    print(header)
    for i, label in enumerate(ring_labels):
        row       = count_matrix.loc[i]
        row_total = int(row.sum())
        row_str   = (f"  {label:>10}"
                     + "".join(f"  {int(row[s]):>8,}" for s in sensors_loaded)
                     + f"  {row_total:>8,}")
        print(row_str)

    # ── Build stacked bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    x_positions = np.arange(len(ring_labels))
    bar_width   = 0.6
    bottoms     = np.zeros(len(ring_labels))

    for sensor in sensors_loaded:
        counts = count_matrix[sensor].values.astype(float)
        bars   = ax.bar(
            x_positions,
            counts,
            bar_width,
            bottom    = bottoms,
            color     = SENSOR_COLORS.get(sensor, '#888888'),
            edgecolor = 'white',
            linewidth = 0.8,
            alpha     = 0.90,
            label     = sensor.upper(),
        )

        # Value labels inside each segment (only if > 1% of total)
        for bar, count, bottom in zip(bars, counts, bottoms):
            if count > total_count * 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + count / 2,
                    f'{int(count):,}',
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white',
                )
        bottoms += counts

    # Total label on top of each bar
    for i, total in enumerate(bottoms):
        ax.text(
            x_positions[i],
            total + total_count * 0.005,
            f'{int(total):,}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            color='#222222',
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ring_labels, fontsize=11)
    ax.set_xlabel('Distance from FCUL Reference Station',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Measurements',
                  fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribution of Solar Irradiance Measurements by Distance Ring\n'
        f'Faculty of Sciences, University of Lisbon (FCUL) — [{CAMPAIGN} / {_VARIANT_TAG}]',
        fontsize=14, fontweight='bold', pad=20,
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x):,}')
    )
    ax.tick_params(axis='y', labelsize=11)
    ax.set_ylim(0, bottoms.max() * 1.12)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(
            facecolor=SENSOR_COLORS.get(s, '#888888'),
            edgecolor='white',
            label=s.upper(),
        )
        for s in sensors_loaded
    ]
    ax.legend(
        handles        = legend_handles,
        title          = 'Sensor',
        title_fontsize = 10,
        fontsize       = 10,
        loc            = 'upper right',
        framealpha     = 0.9,
    )

    # ── Stats text box (upper-left — legend occupies upper-right) ────────────
    stats_text = (
        f'Measurements (within {MAX_DISTANCE_KM} km): {total_count:,}\n'
        f'Sensors: {", ".join(s.upper() for s in sensors_loaded)}\n'
        f'Reference: FCUL Station\n'
        f'({FCUL_LAT:.4f}°N, {abs(FCUL_LON):.4f}°W)'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform           = ax.transAxes,
        fontsize            = 9,
        verticalalignment   = 'top',
        horizontalalignment = 'left',
        bbox                = dict(boxstyle='round,pad=0.5', facecolor='white',
                                   edgecolor='gray', alpha=0.9),
    )

    plt.tight_layout()

    # ── Save figure ───────────────────────────────────────────────────────────
    fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Measurements loaded : {total_count:,}")
    print(f"  Distance range      : "
          f"{combined['distance_km'].min():.2f} – "
          f"{combined['distance_km'].max():.2f} km")
    print()
    print(f"  {'Ring':>10}", end="")
    for s in sensors_loaded:
        print(f"  {s.upper():>8}", end="")
    print(f"  {'Total':>8}  {'%':>6}")
    for i, label in enumerate(ring_labels):
        row      = count_matrix.loc[i]
        row_tot  = int(row.sum())
        row_pct  = row_tot / total_count * 100
        print(f"  {label:>10}", end="")
        for s in sensors_loaded:
            print(f"  {int(row[s]):>8,}", end="")
        print(f"  {row_tot:>8,}  {row_pct:>5.1f}%")
    print(f"\n  Figure saved → {OUTPUT_FILE}  ({OUTPUT_DPI} dpi)")
    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


# ==============================================================================
if __name__ == "__main__":
    main()