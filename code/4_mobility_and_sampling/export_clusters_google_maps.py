"""
================================================================================
Export Stationary Clusters to Excel for Google My Maps
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Detects the stationary measurement clusters of the sensor data and exports
them to an Excel file ready to import into Google My Maps.

A GPS location is stationary when the same coordinate pair (rounded to
COORD_PRECISION decimals) appears at least STATIONARY_THRESHOLD times across
all sensors. Parked vehicles are expected and valid in the VIPV dataset: the
clusters are exported for geographic inspection only. Clusters beyond
MAX_DISTANCE_KM (32 km) from FCUL are excluded.

Excel sheets:
  1. Google_MyMaps_Import  — Name, Latitude, Longitude, Description (the
                             columns Google My Maps needs)
  2. Full_Details          — all cluster attributes, coloured by distance ring
  3. Summary_by_Ring       — cluster counts and measurement totals per ring

Import into Google My Maps
--------------------------
  1. Go to https://www.google.com/maps/d/
  2. Create a new map → Add layer
  3. Import → upload this file → select the "Google_MyMaps_Import" sheet
  4. Choose the Latitude / Longitude columns for the location
  5. Choose Name or Description as the marker title

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             mdr_stationary_clusters_google_maps_<CAMPAIGN>_<variant_tag>.xlsx

Requirements
------------
    pip install pandas numpy openpyxl

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
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector ---------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output directory -----------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "4_mobility_and_sampling" / "mdr_outputs"

# --- FCUL reference station coordinates (WGS-84 decimal degrees) --------------
# Pyranometer on the rooftop of building C1, Faculty of Sciences,
# University of Lisbon, campus Cidade Universitária.
FCUL_LAT: float = 38.7565    # Google Maps: 38.756594832619776
FCUL_LON: float = -9.1564    # Google Maps: -9.156483276747663

# --- Stationary cluster detection ---------------------------------------------
# A coordinate pair (rounded to COORD_PRECISION decimal degrees) that appears
# at least STATIONARY_THRESHOLD times is flagged as a stationary cluster.
# At ~1 measurement/minute, 120 readings ≈ 2 hours parked at the same spot.
STATIONARY_THRESHOLD: int = 120   # minimum occurrences to flag as stationary
COORD_PRECISION:      int = 4     # decimal places for rounding (~11 m resolution)

# --- Earth radius -------------------------------------------------------------
R_EARTH_KM: float = 6371.0   # Mean Earth radius (km) – IAU 2012

# --- Distance ring definitions ------------------------------------------------
# Points strictly beyond MAX_DISTANCE_KM are excluded before classification:
# they fall outside all defined rings and must not be silently absorbed.
RING_BOUNDARIES_KM: list[int] = [0, 3, 9, 18, 32]
MAX_DISTANCE_KM:    int       = RING_BOUNDARIES_KM[-1]   # 32 km
RING_LABELS: list[str]        = ['0–3 km', '3–9 km', '9–18 km', '18–32 km']

# --- Ring background colours for Excel (openpyxl hex, no '#') ----------------
RING_FILLS: dict[str, str] = {
    '0–3 km'  : 'BDD7EE',   # light blue
    '3–9 km'  : 'E2EFDA',   # light green
    '9–18 km' : 'FCE4D6',   # light orange
    '18–32 km': 'FFDDC1',   # light salmon
}


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# All input file paths and the output directory are derived automatically from
# CAMPAIGN, DATA_DIR / TUKTUK_DATA_DIR, OUTPUT_BASE_DIR, and FILE_SUFFIX /
# TUKTUK_FILE_SUFFIX, so that changing CAMPAIGN (and the matching suffix) is
# the only edit needed to switch between pipeline variants.

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor names to their CSV input paths.

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
        {'j30': Path(...), 'j31': Path(...), ...}
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
    """
    Build the output directory path, embedding the campaign and active
    variant tag so results from different campaigns/variants never
    overwrite each other.

    Parameters
    ----------
    base_dir : Path
        Base output directory set in the configuration section.
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().

    Returns
    -------
    Path
        Full output directory path (not yet created).
    """
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
        (e.g. 'mdr_stationary_clusters_google_maps').
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().
    extension : str
        File extension without leading dot (e.g. 'xlsx').

    Returns
    -------
    Path
        Complete file path, e.g.
        out_dir / 'mdr_stationary_clusters_google_maps_standard_sd_cleaned.xlsx'.
    """
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolved at startup from the configuration above
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR:   Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_FILE:  Path = _build_output_path(
    OUTPUT_DIR, 'mdr_stationary_clusters_google_maps', CAMPAIGN, _VARIANT_TAG, 'xlsx'
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


def ring_label(dist_km: float) -> str:
    """
    Return the distance ring label for a given distance from FCUL.

    This function must only be called for distances strictly within
    [0, MAX_DISTANCE_KM). Out-of-range points must be excluded in main()
    before any call to this function.

    Parameters
    ----------
    dist_km : float
        Distance in kilometres. Must be < MAX_DISTANCE_KM.

    Returns
    -------
    str
        Ring label, e.g. '3–9 km'.

    Raises
    ------
    ValueError
        If dist_km >= MAX_DISTANCE_KM.
    """
    if dist_km >= MAX_DISTANCE_KM:
        raise ValueError(
            f"ring_label() received distance {dist_km:.2f} km >= "
            f"{MAX_DISTANCE_KM} km. Exclude out-of-range points before "
            f"calling this function."
        )
    for i in range(len(RING_BOUNDARIES_KM) - 1):
        if dist_km < RING_BOUNDARIES_KM[i + 1]:
            return RING_LABELS[i]
    return RING_LABELS[-1]


def load_sensor(file_path: Path, sensor_name: str) -> pd.DataFrame:
    """
    Load one sensor CSV file and return valid latitude/longitude pairs.

    The input files are clean_data.py outputs (invalid rows already
    removed), so only the coordinate columns are validated here. Points near
    FCUL are kept by clean_data.py: stationary clusters at or near the FCUL
    rooftop are a real feature of the dataset.

    Column name detection
    ---------------------
    The function first looks for the exact column names produced by the data
    pipeline, e.g. 'J31 lat' and 'J31 lon' (uppercase sensor ID). This naming
    convention is identical for both campaigns. If not found, it falls back
    to a case-insensitive search, excluding 'land_use' and 'land_use_str'
    (and their '_clcv' variants) which also contain the substring 'lat' / 'lon'.

    Parameters
    ----------
    file_path : Path
        Path to the sensor CSV file.
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

    df = pd.read_csv(file_path, low_memory=False)

    # ── Locate lat / lon columns ──────────────────────────────────────────────
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

    # ── Extract and validate coordinates ──────────────────────────────────────
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

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Run the full pipeline: load sensor data for the active campaign, detect
    stationary clusters, and export the results to Excel.
    """
    print("=" * 70)
    print("  STATIONARY CLUSTER EXPORT — Google My Maps")
    print(f"  Campaign : {CAMPAIGN}")
    print(f"  Variant  : {_VARIANT_TAG}")
    print("=" * 70)
    print(f"\n  Input files  : {DATA_DIR if CAMPAIGN == 'standard' else TUKTUK_DATA_DIR}")
    print(f"  Output file  : {OUTPUT_FILE}")

    # ── 1. Load all sensor data ───────────────────────────────────────────────
    print("\n[1/3] Loading sensor CSV files ...")
    frames = [load_sensor(p, s) for s, p in SENSOR_FILES.items()]
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. "
            "Please check the paths defined in SENSOR_FILES."
        )

    data = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(data):,} measurements from "
          f"{', '.join(s.upper() for s in sorted(set(data['sensor'])))}")

    # ── Exclude points beyond the outer ring boundary ─────────────────────────
    # Compute distances on the full dataset and drop points > MAX_DISTANCE_KM.
    # These fall outside all defined analysis rings and must not appear in the
    # cluster table or the ring totals.
    data['distance_km'] = haversine_km(
        FCUL_LAT, FCUL_LON,
        data['lat'].values, data['lon'].values,
    )
    mask_in_range  = data['distance_km'] < MAX_DISTANCE_KM
    n_out_of_range = int((~mask_in_range).sum())
    if n_out_of_range:
        print(f"  [INFO] {n_out_of_range:,} points beyond {MAX_DISTANCE_KM} km "
              f"excluded (outside all defined rings).")
    data = data[mask_in_range].reset_index(drop=True)
    print(f"  Points within rings: {len(data):,}")

    # ── 2. Detect stationary clusters ────────────────────────────────────────
    print(f"\n[2/3] Detecting stationary clusters "
          f"(threshold ≥ {STATIONARY_THRESHOLD} measurements at the same spot) ...")

    lat_r      = data['lat'].round(COORD_PRECISION)
    lon_r      = data['lon'].round(COORD_PRECISION)
    coord_key  = lat_r.astype(str) + ',' + lon_r.astype(str)
    counts     = coord_key.value_counts()
    stationary = counts[counts >= STATIONARY_THRESHOLD]

    print(f"  {len(stationary)} stationary clusters found")

    # Build cluster table
    # 'distance_km' was computed and filtered (< MAX_DISTANCE_KM) above
    rows = []
    for idx, (key, n_meas) in enumerate(stationary.items(), start=1):
        lat_s, lon_s = key.split(',')
        lat_v = float(lat_s)
        lon_v = float(lon_s)

        # Use the mean distance of all points in the cluster (already filtered)
        mask  = coord_key == key
        dist  = float(data.loc[mask, 'distance_km'].mean())
        label = ring_label(dist)

        # Sensors contributing to this cluster
        mask            = coord_key == key
        sensors_present = ', '.join(
            sorted(data.loc[mask, 'sensor'].unique())
        ).upper()

        # Google Maps direct link
        gmaps_url = f"https://www.google.com/maps?q={lat_v},{lon_v}"

        rows.append({
            'Name'            : f"Cluster_{idx:03d}",
            'Latitude'        : lat_v,
            'Longitude'       : lon_v,
            'Measurements'    : int(n_meas),
            'Distance_FCUL_km': round(dist, 2),
            'Distance_Ring'   : label,
            'Sensors'         : sensors_present,
            'Google_Maps_Link': gmaps_url,
            'Description'     : (
                f"{int(n_meas)} stationary measurements | "
                f"{dist:.1f} km from FCUL | "
                f"Sensors: {sensors_present}"
            ),
        })

    df_c = (pd.DataFrame(rows)
              .sort_values('Measurements', ascending=False)
              .reset_index(drop=True))

    # ── Compute total measurements per ring (all data, not just clusters) ─────
    # 'distance_km' is already computed and filtered on the data DataFrame.
    # This is used in Sheet 3 to show what fraction of each ring's data
    # consists of stationary cluster measurements.
    data['ring_label'] = data['distance_km'].apply(ring_label)
    ring_totals: dict[str, int] = {
        ring: int((data['ring_label'] == ring).sum())
        for ring in RING_LABELS
    }

    # ── 3. Write Excel ────────────────────────────────────────────────────────
    print("\n[3/3] Writing Excel file ...")

    wb = Workbook()

    # Shared styles
    hdr_font  = Font(name='Arial', bold=True, color='FFFFFF', size=11)
    hdr_fill  = PatternFill('solid', start_color='2E4057')
    ctr_align = Alignment(horizontal='center', vertical='center')
    thin      = Side(style='thin', color='CCCCCC')
    cell_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style_header(ws, col_names):
        """Append the styled header row to `ws` and freeze it."""
        ws.append(col_names)
        for c in range(1, len(col_names) + 1):
            cell           = ws.cell(row=1, column=c)
            cell.font      = hdr_font
            cell.fill      = hdr_fill
            cell.alignment = ctr_align
        ws.freeze_panes = 'A2'
        ws.row_dimensions[1].height = 20

    # ── Sheet 1: Google My Maps import ───────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Google_MyMaps_Import"

    maps_cols = ['Name', 'Latitude', 'Longitude', 'Description']
    style_header(ws1, maps_cols)

    for _, row in df_c.iterrows():
        ws1.append([
            row['Name'],
            row['Latitude'],
            row['Longitude'],
            row['Description'],
        ])

    ws1.column_dimensions['A'].width = 18
    ws1.column_dimensions['B'].width = 14
    ws1.column_dimensions['C'].width = 14
    ws1.column_dimensions['D'].width = 70

    # ── Sheet 2: Full details ─────────────────────────────────────────────────
    ws2 = wb.create_sheet("Full_Details")

    full_cols = [
        'Name', 'Latitude', 'Longitude', 'Measurements',
        'Distance_FCUL_km', 'Distance_Ring', 'Sensors',
        'Google_Maps_Link', 'Description',
    ]
    style_header(ws2, full_cols)
    ws2.auto_filter.ref = f"A1:{get_column_letter(len(full_cols))}1"

    for _, row in df_c.iterrows():
        excel_row = ws2.max_row + 1
        ring      = row['Distance_Ring']
        row_fill  = PatternFill('solid',
                                start_color=RING_FILLS.get(ring, 'FFFFFF'))
        values = [
            row['Name'], row['Latitude'], row['Longitude'],
            row['Measurements'], row['Distance_FCUL_km'],
            row['Distance_Ring'], row['Sensors'],
            row['Google_Maps_Link'], row['Description'],
        ]
        for c_idx, val in enumerate(values, 1):
            cell           = ws2.cell(row=excel_row, column=c_idx, value=val)
            cell.fill      = row_fill
            cell.border    = cell_border
            cell.font      = Font(name='Arial', size=10)
            cell.alignment = Alignment(
                horizontal='center', vertical='center',
                wrap_text=(c_idx in [8, 9])
            )

    for col, w in zip(['A','B','C','D','E','F','G','H','I'],
                      [18, 14, 14, 16, 20, 14, 14, 55, 65]):
        ws2.column_dimensions[col].width = w

    # ── Sheet 3: Summary by ring ──────────────────────────────────────────────
    ws3 = wb.create_sheet("Summary_by_Ring")

    sum_cols = [
        'Distance_Ring',
        'Nb_Clusters',
        'Stationary_Measurements',
        'Total_Measurements_in_Ring',
        'Cluster_Pollution_Pct (%)',
        'Pct_of_All_Stationary (%)',
    ]
    style_header(ws3, sum_cols)

    total_meas     = df_c['Measurements'].sum()
    total_all_meas = len(data)

    for ring in RING_LABELS:
        sub        = df_c[df_c['Distance_Ring'] == ring]
        n          = len(sub)
        stat_meas  = int(sub['Measurements'].sum())
        ring_total = ring_totals.get(ring, 0)
        pollution  = round(stat_meas / ring_total * 100, 1) if ring_total else 0.0
        pct_of_all = round(stat_meas / total_meas * 100, 1) if total_meas else 0.0
        ws3.append([ring, n, stat_meas, ring_total, pollution, pct_of_all])

        r        = ws3.max_row
        row_fill = PatternFill('solid',
                               start_color=RING_FILLS.get(ring, 'FFFFFF'))
        for c in range(1, len(sum_cols) + 1):
            cell           = ws3.cell(row=r, column=c)
            cell.fill      = row_fill
            cell.font      = Font(name='Arial', size=10)
            cell.alignment = Alignment(horizontal='center')

    # Total row
    total_stat = int(total_meas)
    ws3.append([
        'TOTAL',
        len(df_c),
        total_stat,
        total_all_meas,
        round(total_stat / total_all_meas * 100, 1) if total_all_meas else 0.0,
        100.0,
    ])
    r = ws3.max_row
    for c in range(1, len(sum_cols) + 1):
        cell           = ws3.cell(row=r, column=c)
        cell.font      = Font(name='Arial', bold=True, size=10)
        cell.fill      = PatternFill('solid', start_color='D9D9D9')
        cell.alignment = Alignment(horizontal='center')

    for col, w in zip(['A','B','C','D','E','F'],
                      [16, 14, 26, 28, 24, 26]):
        ws3.column_dimensions[col].width = w

    # ── Save ──────────────────────────────────────────────────────────────────
    wb.save(OUTPUT_FILE)
    print(f"  {len(df_c)} clusters | {int(total_meas):,} stationary measurements")

    print("\n" + "=" * 70)
    print("  CLUSTER SUMMARY BY DISTANCE RING")
    print("=" * 70)
    print(f"  {'Ring':>10}  {'Clusters':>8}  {'Stationary':>10}  "
          f"{'Ring Total':>10}  {'Pollution':>10}")
    for ring in RING_LABELS:
        sub        = df_c[df_c['Distance_Ring'] == ring]
        stat_meas  = int(sub['Measurements'].sum())
        ring_total = ring_totals.get(ring, 0)
        pollution  = stat_meas / ring_total * 100 if ring_total else 0.0
        print(f"  {ring:>10}  {len(sub):>8}  {stat_meas:>10,}  "
              f"{ring_total:>10,}  {pollution:>9.1f}%")
    total_stat = int(total_meas)
    overall_pollution = total_stat / total_all_meas * 100 if total_all_meas else 0.0
    print(f"  {'TOTAL':>10}  {len(df_c):>8}  {total_stat:>10,}  "
          f"{total_all_meas:>10,}  {overall_pollution:>9.1f}%")
    print(f"\n  File saved → {OUTPUT_FILE}")
    print("\n  To import into Google My Maps:")
    print("    1. Go to https://www.google.com/maps/d/")
    print("    2. Create new map → Add layer → Import")
    print("    3. Upload this file → select sheet 'Google_MyMaps_Import'")
    print("    4. Use Latitude / Longitude columns for location")
    print("    5. Use 'Name' as marker title")
    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()