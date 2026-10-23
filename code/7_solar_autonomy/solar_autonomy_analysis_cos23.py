"""
================================================================================
Solar Autonomy Analysis — Monthly Summary & Land-Use (COS2023) Statistics
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

COS2023 counterpart of solar_autonomy_analysis_clc.py: same monthly table,
land-use table and four figures, with the finer COS2023 nomenclature (DGT)
instead of CORINE Land Cover:

  solar_autonomy_analysis_clc.py : 44 CLC classes, European coverage,
                                   column land_use_clcv
  this script                    : 93 COS2023 level-4 classes, Portugal only,
                                   columns cos23_n4_code / cos23_n4_label /
                                   cos23_n1_label (COS2023 join)

COS2023 splits the single CLC class "Continuous urban fabric" into more than
10 sub-types (vertical / horizontal residential, industry, commerce, parking
areas, road network, ...), for a finer analysis of intra-urban shading.

Outputs
-------
  OUTPUT 1 — CSV: monthly summary table
  OUTPUT 2 — CSV: COS2023 land-use statistics table (up to 93 rows)
  OUTPUT 3 — Four figures
      Fig 1 : Top-10 COS2023 level-4 classes by measurement count
      Fig 2 : Dual-axis loss rate + mean captured fraction by class
      Fig 3 : Relative autonomy losses (%) by class
      Fig 4 : Moving / Parked proportion by class (>= MIN_ZONE_POINTS records)
      Classes with fewer than MIN_FIG_ZONE_POINTS records are left out of
      Figs 2-3.

Urban / Rural for COS2023: level-1 code "1" (Artificial surfaces) is Urban,
all other level-1 branches (2–9) are Rural.

Physical model
--------------
  E_solar   [kWh/kWp] = rad_calib [W/m²] / 1000 × Δt [h]
  Autonomy  [km/kWp]  = E_solar / CVE    (CVE = 16.6 kWh/100 km, EEA /
                                          Weiss et al., 2024 — EU average BEV)
  loss_irr  [kWh/kWp] = max(0, GHI − rad_calib) / 1000 × Δt
  loss_auto [km/kWp]  = loss_irr / CVE
  frac_captured [-]   = rad_calib / ghi   (0 when either term is zero)

Movement classification
-----------------------
  A GPS transition between consecutive points >= THRESHOLD_M metres is
  "Moving", otherwise "Parked" (pairs across a gap > MAX_GAP_MINUTES or a date
  change count as Parked).

Monthly table
-------------
  Time complement per active day: unrecorded_h = 24 − parked_h − moving_h
  (night, acquisition gaps and any other period without data), averaged over
  the active days of the month. Months with fewer than MIN_MEASUREMENTS
  records stay in the table with their count, but all computed columns are
  set to NaN. This threshold does not apply to the land-use table.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', '<SENSOR> Rad calib', 'ghi', 'sun_elev',
             'cos23_n4_code', 'cos23_n4_label', 'cos23_n1_label'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             saa_cos23_monthly_table, saa_cos23_landuse_table
                 (_<CAMPAIGN>_<variant_tag>.csv)
             saa_cos23_fig1_top10_classes, saa_cos23_fig2_loss_frac,
             saa_cos23_fig3_loss_pct, saa_cos23_fig4_moving_parked
                 (_<CAMPAIGN>_<variant_tag>.png)

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.2
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── VIPV shared modules (code/common) ─────────────────────────────────────────

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR_STANDARD = STANDARD_DIR
DATA_DIR_TUKTUK   = TUKTUK_DIR
if CAMPAIGN == "standard":
    DATA_DIR = DATA_DIR_STANDARD
elif CAMPAIGN == "tuktuk":
    DATA_DIR = DATA_DIR_TUKTUK
else:
    raise ValueError(f"Unknown CAMPAIGN '{CAMPAIGN}'.")

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output directory ---------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "7_solar_autonomy" / "saa_cos23_outputs"

# --- Display figures during execution? ----------------------------------------
SHOW_FIGURES: bool = False

# --- Physical / vehicle parameters -------------------------------------------

# Average Battery Electric Vehicle energy consumption
# Source: European Environment Agency / Weiss et al., 2024 — EU average BEV
CVE_KWH_PER_100KM: float = 16.6
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0

# Measurement time resolution
DELTA_T_H: float = 1.0 / 60.0   # 1 minute in hours

# --- Movement threshold -------------------------------------------------------
# GPS transitions ≥ THRESHOLD_M metres → classified as Moving
THRESHOLD_M:     float = 30.0    # metres
R_EARTH_KM:      float = 6371.0  # mean Earth radius [km]
MAX_GAP_MINUTES: int   = 2

# --- Reliability thresholds --------------------------------------------------
# Monthly table: months with fewer than MIN_MEASUREMENTS records have their
# computed columns masked to NaN (row is kept, count is shown).
MIN_MEASUREMENTS: int = 600

# Fig 4 (Moving/Parked): COS2023 classes with fewer records are excluded.
MIN_ZONE_POINTS: int = 600

# Figs 2 & 3: minimum records to include a class in the loss-rate and
# relative-loss figures (suppresses artefacts from sparse classes).
MIN_FIG_ZONE_POINTS: int = 600

# ==============================================================================
# COS2023 URBAN / RURAL CLASSIFICATION
# ==============================================================================
# Level-1 code "1" = "Artificial surfaces" → Urban
# All other level-1 branches (2–9)         → Rural
# Applied row by row from the cos23_n1_label / cos23_n4_code columns.

def _urban_rural_from_level1(level1_label: str) -> str:
    """Return 'Urban' for Artificial surfaces, 'Rural' for all other branches."""
    if level1_label == "Artificial surfaces":
        return "Urban"
    return "Rural"


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

# Sensor lists per campaign.
SENSORS: list[str] = ['J30', 'J31', 'J32', 'J33']   # adjust to match deployed sensors

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _resolve_input_files(
    campaign: str,
    data_dir: Path,
    sensors: list[str],
) -> list[tuple[str, Path]]:
    """
    Return a list of (sensor_name, file_path) pairs for the selected campaign.
    Both campaigns provide one CSV file per Solar Jinie sensor.

    Parameters
    ----------
    campaign : str
        "standard" or "tuktuk".
    data_dir : Path
        Directory holding the per-sensor files for the selected campaign
        (already resolved to DATA_DIR_STANDARD or DATA_DIR_TUKTUK).
    sensors : list[str]
        Sensor identifiers (UPPER case) for the selected campaign.

    Returns
    -------
    list of (sensor_name: str, file_path: Path)
    """
    if campaign == "standard":
        sep = f"_{FILE_SUFFIX}" if FILE_SUFFIX else ""
        return [
            (s, data_dir / f"{s.lower()}_satellite_landuse_v2{sep}.csv")
            for s in sensors
        ]
    elif campaign == "tuktuk":
        sep = f"_{TUKTUK_FILE_SUFFIX}" if TUKTUK_FILE_SUFFIX else ""
        return [
            (s, data_dir / f"{s.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv")
            for s in sensors
        ]
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign (used in
    output filenames and figure annotations)."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output sub-directory for this campaign / variant."""
    return base_dir / campaign / variant_tag


def _build_output_path(out_dir: Path, base_name: str,
                       campaign: str, variant_tag: str,
                       extension: str) -> Path:
    """Build a full output file path embedding the active mode tags."""
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolve all paths at module level
INPUT_FILES: list[tuple[str, Path]] = _resolve_input_files(
    CAMPAIGN, DATA_DIR, SENSORS
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)

OUT_CSV_MONTHLY  = _build_output_path(OUTPUT_DIR, 'saa_cos23_monthly_table',
                                      CAMPAIGN, _VARIANT_TAG, 'csv')
OUT_CSV_LANDUSE  = _build_output_path(OUTPUT_DIR, 'saa_cos23_landuse_table',
                                      CAMPAIGN, _VARIANT_TAG, 'csv')
OUT_FIG1         = _build_output_path(OUTPUT_DIR, 'saa_cos23_fig1_top10_classes',
                                      CAMPAIGN, _VARIANT_TAG, 'png')
OUT_FIG2         = _build_output_path(OUTPUT_DIR, 'saa_cos23_fig2_loss_frac',
                                      CAMPAIGN, _VARIANT_TAG, 'png')
OUT_FIG3         = _build_output_path(OUTPUT_DIR, 'saa_cos23_fig3_loss_pct',
                                      CAMPAIGN, _VARIANT_TAG, 'png')
OUT_FIG4         = _build_output_path(OUTPUT_DIR, 'saa_cos23_fig4_moving_parked',
                                      CAMPAIGN, _VARIANT_TAG, 'png')


# ==============================================================================
# HELPER — HAVERSINE DISTANCE  (identical to SAA)
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2, R=R_EARTH_KM,
                 timestamps=None, max_gap_minutes=MAX_GAP_MINUTES):
    """
    Compute the great-circle distance in metres between consecutive GPS points.

    Parameters
    ----------
    lat1, lon1 : array-like
        Coordinates of the departure point (decimal degrees, WGS-84).
    lat2, lon2 : array-like
        Coordinates of the arrival point (decimal degrees, WGS-84).
    R : float
        Earth radius in kilometres (default: 6371.0 — IAU 2012).
    timestamps : array-like of datetime64, optional
        Full timestamp array (length N). When provided, pairs that cross a
        day boundary or a gap > max_gap_minutes are zeroed out.
        When None, all distances are returned as computed.
    max_gap_minutes : float
        Maximum allowed gap in minutes between consecutive rows.

    Returns
    -------
    numpy.ndarray of float, shape (N-1,)
        Great-circle distances in metres.
        Boundary pairs are returned as 0.0 when timestamps is provided.
    """
    lat1 = np.asarray(lat1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    phi1    = np.radians(lat1)
    phi2    = np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(np.asarray(lon2, dtype=float)
                         - np.asarray(lon1, dtype=float))
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    distances = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * 1000.0

    if timestamps is not None:
        s         = pd.Series(np.asarray(timestamps))
        gap_s     = s.diff().dt.total_seconds().fillna(0).values
        dates     = s.dt.date.values
        is_boundary = (
            (gap_s[1:] > max_gap_minutes * 60)
            | (dates[1:] != dates[:-1])
        )
        distances[is_boundary] = 0.0

    return distances


# ==============================================================================
# DATA LOADING
# ==============================================================================

def _consolidate_latlon_rad(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """
    Consolidate sensor-specific lat / lon / rad_calib columns into generic ones.

    Parameters
    ----------
    df : pd.DataFrame
        Combined join DataFrame (all sensors).
    sensors : list[str]
        List of sensor identifiers in UPPER case (e.g. ['J30', 'J31', ...]).

    Returns
    -------
    pd.DataFrame
        df enriched with 'lat', 'lon', 'rad_calib' columns.
        Rows for which no sensor column is found are left as NaN.
    """
    df["lat"]       = np.nan
    df["lon"]       = np.nan
    df["rad_calib"] = np.nan

    for s in sensors:
        tag  = s.upper()
        mask = df["sensor"].str.upper() == tag
        for src, dst in [
            (f"{tag} lat",       "lat"),
            (f"{tag} lon",       "lon"),
            (f"{tag} Rad calib", "rad_calib"),
        ]:
            if src in df.columns:
                df.loc[mask, dst] = df.loc[mask, src]

    return df


def _load_one_file(sensor_name: str, file_path: Path,
                   all_sensors: list[str]) -> pd.DataFrame | None:
    """
    Load one COS2023-enriched CSV file.

    Parameters
    ----------
    sensor_name : str
        Sensor identifier for this file (e.g. 'J33'). Used only to populate
        the 'sensor' column when it is not already present in the file.
    file_path : Path
        Full path to the CSV file.
    all_sensors : list[str]
        Full sensor list for the campaign (UPPER case).  Passed to
        _consolidate_latlon_rad() to find the correct prefixed columns.

    Returns
    -------
    pd.DataFrame or None
        Loaded, renamed DataFrame; None if the file does not exist.
    """
    if not file_path.exists():
        print(f"  [WARNING] File not found, skipping: {file_path}")
        return None

    print(f"  Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"  {len(df):,} rows read")

    # ── Timestamp ─────────────────────────────────────────────────────────────
    date_col = "Human readable date (Europe/London)"
    if date_col not in df.columns:
        raise KeyError(
            f"Expected timestamp column '{date_col}' not found in:\n  {file_path}\n"
            "Ensure normalize_tuktuk.py has been run for tuk-tuk files."
        )
    df["datetime"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # ── 'sensor' column ───────────────────────────────────────────────────────
    # Add it from the known sensor identifier if the file doesn't already
    # carry one (applies to both campaigns — files are per sensor).
    if "sensor" not in df.columns:
        df["sensor"] = sensor_name.lower()

    # ── Consolidate lat / lon / rad_calib from prefixed columns ───────────────
    df = _consolidate_latlon_rad(df, all_sensors)

    # ── GHI / sun_elev ────────────────────────────────────────────────────────
    for col in ["ghi", "sun_elev"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in: {file_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["lat", "lon", "rad_calib"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["lat", "lon", "rad_calib", "ghi", "sun_elev"])

    # ── COS2023 class columns ─────────────────────────────────────────────────
    for col in ["cos23_n4_code", "cos23_n4_label", "cos23_n1_label"]:
        if col not in df.columns:
            raise KeyError(
                f"Required COS2023 column '{col}' not found in: {file_path}\n"
                "Ensure join_cos_standard.py (standard) or join_cos23_tuktuk.py "
                "(tuktuk) has been run before this script."
            )
    df["cos23_n4_code"]  = df["cos23_n4_code"].fillna("Unknown / missing")
    df["cos23_n4_label"] = df["cos23_n4_label"].fillna("Unknown / missing")
    df["cos23_n1_label"] = df["cos23_n1_label"].fillna("Unknown / missing")

    return df


def load_combined_cos23(input_files: list[tuple[str, Path]],
                        sensors: list[str]) -> pd.DataFrame:
    """
    Load and concatenate all COS2023-enriched input files, then prepare the
    combined DataFrame for analysis.

    After loading and concatenating, the function:
      1. Derives Urban / Rural from cos23_n1_label.
      2. Adds calendar columns (month, date) in local time (Europe/Lisbon).
      3. Computes per-sensor Haversine distances with overnight-gap zeroing.
      4. Classifies Moving / Parked (threshold: THRESHOLD_M metres).
      5. Computes per-minute energy, loss, gain, fraction, and autonomy.
      6. Applies daytime and valid-irradiance filters.

    Parameters
    ----------
    input_files : list of (sensor_name: str, file_path: Path)
        Files to load, as returned by _resolve_input_files().
    sensors : list[str]
        Full sensor list for the campaign (passed to _consolidate_latlon_rad).

    Returns
    -------
    pd.DataFrame
        Filtered and enriched DataFrame ready for aggregation.
    """
    frames: list[pd.DataFrame] = []
    for sensor_name, file_path in input_files:
        frame = _load_one_file(sensor_name, file_path, sensors)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No input file could be loaded. "
            "Check DATA_DIR_STANDARD / DATA_DIR_TUKTUK, CAMPAIGN, and "
            "FILE_SUFFIX / TUKTUK_FILE_SUFFIX."
        )

    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(df):,} rows combined across all files")

    # ── Urban / Rural ─────────────────────────────────────────────────────────
    df["urban_rural"] = df["cos23_n1_label"].apply(_urban_rural_from_level1)

    # ── Calendar columns (local time) ─────────────────────────────────────────
    local       = df["datetime"].dt.tz_convert("Europe/Lisbon")
    df["month"] = local.dt.month
    df["date"]  = local.dt.date

    # ── Sort chronologically, per sensor ──────────────────────────────────────
    df = df.sort_values(["sensor", "datetime"]).reset_index(drop=True)

    # ── Haversine movement classification (per sensor, with overnight guard) ──
    df["dist_m"]    = np.nan
    df["is_moving"] = False

    for s in df["sensor"].unique():
        mask = df["sensor"] == s
        idx  = df.index[mask]
        sub  = df.loc[mask]
        if len(sub) > 1:
            dists = haversine_vectorized(
                sub["lat"].values[:-1], sub["lon"].values[:-1],
                sub["lat"].values[1:],  sub["lon"].values[1:],
                timestamps=sub["datetime"].values,
            )
            df.loc[idx[:-1], "dist_m"] = dists

    df["is_moving"] = (df["dist_m"] >= THRESHOLD_M).fillna(False)

    # ── Per-minute irradiance and autonomy quantities ─────────────────────────
    diff_W = df["ghi"] - df["rad_calib"]

    df["e_solar_kWh"]   = df["rad_calib"] / 1000.0 * DELTA_T_H
    df["loss_irr_kWh"]  = (diff_W / 1000.0 * DELTA_T_H).clip(lower=0.0)
    df["gain_irr_kWh"]  = ((-diff_W) / 1000.0 * DELTA_T_H).clip(lower=0.0)

    df["auto_solar_km"] = df["e_solar_kWh"] / CVE_KWH_PER_KM
    df["loss_auto_km"]  = df["loss_irr_kWh"] / CVE_KWH_PER_KM
    df["gain_auto_km"]  = df["gain_irr_kWh"] / CVE_KWH_PER_KM

    mask_frac = (df["ghi"] > 0) & (df["rad_calib"] > 0)
    df["frac_captured"] = 0.0
    df.loc[mask_frac, "frac_captured"] = (
        df.loc[mask_frac, "rad_calib"] / df.loc[mask_frac, "ghi"]
    )
    print(f"  {len(df):,} rows retained after daytime + irradiance filters")
    return df


# ==============================================================================
# OUTPUT 1 — MONTHLY TABLE
# ==============================================================================

def build_monthly_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-minute data into a monthly summary table.

    Columns
    -------
    month, month_name,
    n_measurements,
    energy_loss_kWh_m2, energy_gain_kWh_m2, energy_diff_kWh_m2,
    parked_h_per_day,   moving_h_per_day,   unrecorded_h_per_day,
    auto_solar_km_per_day, auto_loss_km_per_day, auto_diff_km_per_day,
    distance_total_km,
    distance_{sensor}_km  (one column per sensor in SENSORS)

    All computed columns are NaN for months with n_measurements < MIN_MEASUREMENTS.

    Parameters
    ----------
    data : pd.DataFrame
        Combined, filtered dataset.

    Returns
    -------
    pd.DataFrame
        12 rows (all calendar months represented).
    """
    rows = []

    for m in range(1, 13):
        sub = data[data["month"] == m]
        n   = len(sub)

        row: dict = {
            "month":          m,
            "month_name":     MONTH_NAMES[m - 1],
            "n_measurements": n,
        }

        computed_cols = [
            "energy_loss_kWh_m2", "energy_gain_kWh_m2", "energy_diff_kWh_m2",
            "parked_h_per_day",   "moving_h_per_day",   "unrecorded_h_per_day",
            "auto_solar_km_per_day", "auto_loss_km_per_day", "auto_diff_km_per_day",
            "distance_total_km",
        ] + [f"distance_{s.upper()}_km" for s in SENSORS]

        if n < MIN_MEASUREMENTS:
            for col in computed_cols:
                row[col] = np.nan
            rows.append(row)
            continue

        # ── Active days — time complement ─────────────────────────────────────
        daily_times = (
            sub.groupby("date", as_index=False)
               .agg(
                   parked_min  = ("is_moving", lambda x: (x == False).sum()),
                   moving_min  = ("is_moving", lambda x: (x == True ).sum()),
               )
        )
        daily_times["unrecorded_h"] = (
            24.0
            - daily_times["parked_min"] / 60.0
            - daily_times["moving_min"] / 60.0
        )
        n_active_days = len(daily_times)

        row["parked_h_per_day"]     = (daily_times["parked_min"].sum() / 60.0
                                       / n_active_days)
        row["moving_h_per_day"]     = (daily_times["moving_min"].sum() / 60.0
                                       / n_active_days)
        row["unrecorded_h_per_day"] = daily_times["unrecorded_h"].mean()

        # ── Energy ────────────────────────────────────────────────────────────
        row["energy_loss_kWh_m2"] = sub["loss_irr_kWh"].sum()
        row["energy_gain_kWh_m2"] = sub["gain_irr_kWh"].sum()
        row["energy_diff_kWh_m2"] = (row["energy_gain_kWh_m2"]
                                     - row["energy_loss_kWh_m2"])

        # ── Autonomy (daily average over sensor-days) ─────────────────────────
        daily_auto = (
            sub.groupby(["sensor", "date"], as_index=False)
               .agg(
                   auto_solar = ("auto_solar_km", "sum"),
                   auto_loss  = ("loss_auto_km",  "sum"),
                   auto_gain  = ("gain_auto_km",  "sum"),
               )
        )
        row["auto_solar_km_per_day"] = daily_auto["auto_solar"].mean()
        row["auto_loss_km_per_day"]  = daily_auto["auto_loss"].mean()
        row["auto_diff_km_per_day"]  = (daily_auto["auto_gain"].mean()
                                        - daily_auto["auto_loss"].mean())

        # ── Distance (Moving transitions only) ────────────────────────────────
        # dist_m is in metres; divide by 1000 for the *_km output columns.
        moving_sub = sub[sub["is_moving"]]
        row["distance_total_km"] = moving_sub["dist_m"].sum() / 1000.0
        for s in SENSORS:
            row[f"distance_{s.upper()}_km"] = (
                moving_sub.loc[
                    moving_sub["sensor"].str.upper() == s.upper(), "dist_m"
                ].sum() / 1000.0
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# OUTPUT 2 — COS2023 LAND-USE TABLE
# ==============================================================================

def build_cos23_landuse_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-minute data by COS2023 level-4 class.

    Unlike the CLC table (which scaffolds all 44 classes), only classes
    actually observed in the data are included.  This keeps the table compact
    and avoids meaningless rows for the many COS2023 classes that never appear
    in the sensor footprint.

    Columns
    -------
    cos23_n4_code, cos23_n4_label, cos23_n1_label, urban_rural,
    n_points,                   (number of 1-minute filtered records)
    duration_h,                 (n_points / 60)
    pct_of_total,               (% of all filtered records)
    auto_loss_rate_km_kWp_h,    (sum of per-minute losses / duration_h)
    auto_loss_pct,              (loss_auto / (auto_solar + loss_auto) × 100)
    mean_frac_captured_pct,     (mean rad_calib/ghi × 100)
    pct_moving,                 (% of records classified as Moving)
    pct_parked                  (% of records classified as Parked)

    Parameters
    ----------
    data : pd.DataFrame
        Combined, filtered dataset with cos23_n4_code and cos23_n4_label.

    Returns
    -------
    pd.DataFrame
        One row per observed COS2023 class, sorted by loss rate descending.
    """
    total_n = len(data)
    rows    = []

    grouped = data.groupby(
        ["cos23_n4_code", "cos23_n4_label", "cos23_n1_label", "urban_rural"],
        observed=True,
    )

    for (code, label, lvl1_label, ur), sub in grouped:
        n          = len(sub)
        duration_h = n / 60.0
        pct_total  = (n / total_n * 100.0) if total_n > 0 else 0.0

        loss_rate = (sub["loss_auto_km"].sum() / duration_h
                     if duration_h > 0 else np.nan)
        mean_frac = sub["frac_captured"].mean() * 100.0
        pct_mov   = sub["is_moving"].mean() * 100.0
        pct_par   = (1.0 - sub["is_moving"].mean()) * 100.0

        total_loss = sub["loss_auto_km"].sum()
        total_auto = sub["auto_solar_km"].sum()
        denom      = total_auto + total_loss
        auto_loss_pct = (total_loss / denom * 100.0
                         if denom > 0 else np.nan)

        rows.append({
            "cos23_n4_code":            code,
            "cos23_n4_label":           label,
            "cos23_n1_label":           lvl1_label,
            "urban_rural":              ur,
            "n_points":                 n,
            "duration_h":               round(duration_h,   4),
            "pct_of_total":             round(pct_total,    4),
            "auto_loss_rate_km_kWp_h":  round(loss_rate,    6),
            "auto_loss_pct":            (round(auto_loss_pct, 4)
                                         if not np.isnan(auto_loss_pct) else np.nan),
            "mean_frac_captured_pct":   round(mean_frac,    4),
            "pct_moving":               round(pct_mov,      2),
            "pct_parked":               round(pct_par,      2),
        })

    df_lu = pd.DataFrame(rows)
    return df_lu.sort_values(
        "auto_loss_rate_km_kWp_h", ascending=False, na_position="last"
    ).reset_index(drop=True)


# ==============================================================================
# HELPERS — FIGURES
# ==============================================================================

# Project-wide palette
COLOR_URBAN      = "#C0392B"   # Urban zones
COLOR_RURAL      = "#2E7D32"   # Rural zones
COLOR_MOVING     = "#1565C0"   # vehicle in motion
COLOR_PARKED     = "#E65100"   # vehicle stationary
COLOR_URBAN_EDGE = "#C0392B"
COLOR_RURAL_EDGE = "#2E7D32"


def _tag() -> str:
    """Return a compact variant tag for figure annotations."""
    return f"[{CAMPAIGN} / {_VARIANT_TAG}]"


def _save_and_show(fig: plt.Figure, path: Path) -> None:
    """Save figure to disk at 300 DPI and optionally display it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Figure saved → {path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def _bar_colors(df: pd.DataFrame) -> list[str]:
    """Return one bar colour per row based on Urban / Rural classification."""
    return [COLOR_URBAN if r == "Urban" else COLOR_RURAL
            for r in df["urban_rural"]]


# ==============================================================================
# FIGURE 1 — Top-10 COS2023 classes by measurement count
# ==============================================================================

def plot_fig1_top10_classes(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 1 — Top-10 COS2023 level-4 classes by measurement count.

    A horizontal bar chart coloured by Urban / Rural, with a mini pie chart
    inset showing the Urban / Rural split across the top-10 classes.

    Bar labels show the percentage of total measurement time for each class.

    Parameters
    ----------
    lu       : COS2023 land-use table (output of build_cos23_landuse_table).
    out_path : destination PNG path.
    """
    plot_df = lu[lu["n_points"] > 0].nlargest(10, "n_points").copy()

    if plot_df.empty:
        print("  [WARNING] Fig 1: no data to plot.")
        return

    colors  = _bar_colors(plot_df)
    y_pos   = np.arange(len(plot_df))

    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(15, 5),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    ax_bar.barh(y_pos, plot_df["n_points"].values,
                color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    for i, (n, pct) in enumerate(
        zip(plot_df["n_points"].values, plot_df["pct_of_total"].values)
    ):
        ax_bar.text(n * 1.005, y_pos[i],
                    f"{pct:.1f}%", va="center", ha="left",
                    fontsize=7.5, color="#333333")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(plot_df["cos23_n4_label"].values, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Number of 1-minute records", fontsize=10)
    ax_bar.set_title("Top-10 COS2023 Classes by Measurement Count",
                     fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.legend(
        handles=[mpatches.Patch(color=COLOR_URBAN, label="Urban"),
                 mpatches.Patch(color=COLOR_RURAL, label="Rural")],
        fontsize=8.5, loc="lower right"
    )

    # Mini pie: Urban vs Rural share in top-10
    n_urban = plot_df.loc[plot_df["urban_rural"] == "Urban", "n_points"].sum()
    n_rural = plot_df.loc[plot_df["urban_rural"] == "Rural", "n_points"].sum()
    if n_urban + n_rural > 0:
        ax_pie.pie(
            [n_urban, n_rural],
            labels=["Urban", "Rural"],
            colors=[COLOR_URBAN, COLOR_RURAL],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 8.5},
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        ax_pie.set_title("Urban / Rural split\n(top-10 classes)",
                         fontsize=9, fontweight="bold")
    else:
        ax_pie.axis("off")

    fig.suptitle(_tag(), fontsize=8, y=1.01, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIGURE 2 — Dual-axis loss rate + mean fraction captured
# ==============================================================================

def plot_fig2_loss_frac(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 2 — Dual-axis chart: autonomy loss rate and mean fraction captured.

    Left axis  (bars) : auto_loss_rate_km_kWp_h  [km/kWp/h]
    Right axis (line) : mean_frac_captured_pct   [%]

    Only classes with at least MIN_FIG_ZONE_POINTS records are shown.

    Parameters
    ----------
    lu       : COS2023 land-use table.
    out_path : destination PNG path.
    """
    plot_df = lu[
        (lu["n_points"] >= MIN_FIG_ZONE_POINTS) &
        lu["auto_loss_rate_km_kWp_h"].notna()
    ].copy()

    n_excluded = ((lu["n_points"] > 0) &
                  (lu["n_points"] < MIN_FIG_ZONE_POINTS)).sum()

    if plot_df.empty:
        print("  [WARNING] Fig 2: no data to plot.")
        return

    bar_colors = _bar_colors(plot_df)
    n  = len(plot_df)
    y  = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(13, max(6, n * 0.42)))
    ax2 = ax1.twiny()

    ax1.barh(y, plot_df["auto_loss_rate_km_kWp_h"].values,
             color=bar_colors, edgecolor="white", linewidth=0.4,
             height=0.65, zorder=3)

    frac_vals = plot_df["mean_frac_captured_pct"].values
    ax2.plot(frac_vals, y, color="#1a6faf", marker="o",
             markersize=4, linewidth=1.6, zorder=4)
    ax2.axvline(100.0, color="#1a6faf", linewidth=0.8, linestyle=":",
                alpha=0.6, zorder=2)

    ax1.set_yticks(y)
    ax1.set_yticklabels(plot_df["cos23_n4_label"].values, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Autonomy Loss Rate [km/kWp/h]", fontsize=10,
                   fontweight="bold", color="#333333")
    ax1.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax1.spines[["top"]].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_xlabel(
        "Mean Captured-Radiation Fraction [%]\n"
        "(> 100 % = sensor irradiance exceeds satellite GHI — physically possible)",
        fontsize=9, fontweight="bold", color="#1a6faf"
    )
    ax2.tick_params(axis="x", colors="#1a6faf")
    ax2.spines["top"].set_color("#1a6faf")
    ax2.spines[["right", "left", "bottom"]].set_visible(False)

    handles = [
        mpatches.Patch(color=COLOR_URBAN, label="Urban (loss rate)"),
        mpatches.Patch(color=COLOR_RURAL, label="Rural (loss rate)"),
        plt.Line2D([0], [0], color="#1a6faf", marker="o",
                   markersize=5, label="Mean fraction captured (%)"),
    ]
    ax1.legend(handles=handles, fontsize=8.5, loc="lower right")
    ax1.set_title(
        "Solar Autonomy Loss Rate & Mean Captured-Radiation Fraction\n"
        "by COS2023 Level-4 Class",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.text(
        0.5, -0.01,
        f"{_tag()}  —  classes with ≥ {MIN_FIG_ZONE_POINTS:,} records"
        f"  ({n_excluded} excluded)",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIGURE 3 — Relative autonomy losses (%) by COS2023 class
# ==============================================================================

def plot_fig3_loss_pct(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 3 — Relative autonomy losses (%) by COS2023 level-4 class.

    Horizontal bar chart ordered by loss % descending.  Bar annotations show
    the class's share of total measurement time.

    Relative autonomy loss formula (per class):
        loss_pct = sum(loss_auto) / (sum(auto_solar) + sum(loss_auto)) × 100

    Only classes with at least MIN_FIG_ZONE_POINTS records are shown.

    Parameters
    ----------
    lu       : COS2023 land-use table.
    out_path : destination PNG path.
    """
    plot_df = lu[
        (lu["n_points"] >= MIN_FIG_ZONE_POINTS) &
        lu["auto_loss_pct"].notna()
    ].copy()

    n_excluded = ((lu["n_points"] > 0) &
                  (lu["n_points"] < MIN_FIG_ZONE_POINTS)).sum()

    if plot_df.empty:
        print("  [WARNING] Fig 3: no data to plot.")
        return

    plot_df = plot_df.sort_values("auto_loss_pct", ascending=False)
    bar_colors = _bar_colors(plot_df)

    n  = len(plot_df)
    y  = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.42)))

    ax.barh(y, plot_df["auto_loss_pct"].values,
            color=bar_colors, edgecolor="white", linewidth=0.4,
            height=0.75, zorder=3)

    x_max = plot_df["auto_loss_pct"].max()
    for i, (lp, pct) in enumerate(
        zip(plot_df["auto_loss_pct"].values, plot_df["pct_of_total"].values)
    ):
        ax.text(lp + x_max * 0.008, y[i],
                f"{pct:.1f}%", va="center", ha="left",
                fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["cos23_n4_label"].values, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(
        r"Relative Autonomy Loss [%]  "
        r"$= \frac{\Sigma\,loss\_auto}{\Sigma\,auto\_solar + \Sigma\,loss\_auto} \times 100$",
        fontsize=10, fontweight="bold"
    )
    ax.set_title(
        "Relative Solar Autonomy Losses by COS2023 Level-4 Class",
        fontsize=12, fontweight="bold", pad=10
    )
    fig.text(
        0.5, -0.01,
        f"{_tag()}  —  classes with ≥ {MIN_FIG_ZONE_POINTS:,} records"
        f"  ({n_excluded} excluded)  —  bar labels = % of total measurement time",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    ax.legend(
        handles=[mpatches.Patch(color=COLOR_URBAN, label="Urban"),
                 mpatches.Patch(color=COLOR_RURAL, label="Rural")],
        fontsize=9, loc="lower right"
    )
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIGURE 4 — Moving / Parked proportion by COS2023 class
# ==============================================================================

def plot_fig4_moving_parked(lu: pd.DataFrame, out_path: Path) -> None:
    """
    Figure 4 — Moving / Parked proportion for each COS2023 class with
    sufficient data.

    Each class is shown as a horizontal 100 % stacked bar split into:
      - Moving (blue)   : % of 1-minute records classified as moving
      - Parked (orange) : % of 1-minute records classified as parked

    Only classes with at least MIN_ZONE_POINTS records are displayed.
    Classes are ordered by Moving % descending.
    A vertical dashed line marks the dataset-wide weighted mean Moving %.
    A thin coloured strip on the left edge of each bar codes Urban / Rural.

    Parameters
    ----------
    lu       : COS2023 land-use table.
    out_path : destination PNG path.
    """
    plot_df = lu[
        (lu["n_points"] >= MIN_ZONE_POINTS) &
        lu["pct_moving"].notna() &
        lu["pct_parked"].notna()
    ].copy()

    if plot_df.empty:
        print(
            f"  [WARNING] Fig 4: no COS2023 class has ≥ {MIN_ZONE_POINTS} "
            "records. Lower MIN_ZONE_POINTS or check the dataset."
        )
        return

    plot_df = plot_df.sort_values("pct_moving", ascending=True).reset_index(drop=True)
    n = len(plot_df)
    y = np.arange(n)

    total_n  = plot_df["n_points"].sum()
    mean_mov = (plot_df["pct_moving"] * plot_df["n_points"]).sum() / total_n

    fig_h = max(5, n * 0.42 + 1.8)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    # Stacked bars — Moving (left) then Parked (right)
    ax.barh(y, plot_df["pct_moving"].values,
            height=0.70, color=COLOR_MOVING, edgecolor="white",
            linewidth=0.5, label="Moving", zorder=3)
    ax.barh(y, plot_df["pct_parked"].values,
            left=plot_df["pct_moving"].values,
            height=0.70, color=COLOR_PARKED, edgecolor="white",
            linewidth=0.5, label="Parked", zorder=3)

    # Urban / Rural indicator: thin coloured strip on the left edge
    edge_colors = [COLOR_URBAN_EDGE if r == "Urban" else COLOR_RURAL_EDGE
                   for r in plot_df["urban_rural"]]
    for i, ec in enumerate(edge_colors):
        ax.barh(y[i], 0.6, height=0.72, color=ec, zorder=4)

    # Bar annotations
    MIN_LABEL_WIDTH = 6.0  # % — hide text if segment too narrow

    for i, (mov, par, n_pts) in enumerate(
        zip(plot_df["pct_moving"].values,
            plot_df["pct_parked"].values,
            plot_df["n_points"].values)
    ):
        if mov >= MIN_LABEL_WIDTH:
            ax.text(mov / 2.0, y[i], f"{mov:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=5)
        if par >= MIN_LABEL_WIDTH:
            ax.text(mov + par / 2.0, y[i], f"{par:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=5)
        ax.text(101.5, y[i], f"n = {n_pts:,}",
                ha="left", va="center", fontsize=7.0, color="#555555")

    # Dataset-wide mean reference line
    ax.axvline(mean_mov, color="#333333", linewidth=1.2, linestyle="--",
               zorder=5, label=f"Dataset mean Moving ({mean_mov:.1f}%)")
    ax.text(mean_mov + 0.4, n - 0.1,
            f"Mean: {mean_mov:.1f}%", fontsize=8, color="#333333", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["cos23_n4_label"].values, fontsize=7.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Proportion of 1-minute records [%]",
                  fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=COLOR_MOVING,     label="Moving"),
        mpatches.Patch(color=COLOR_PARKED,     label="Parked"),
        mpatches.Patch(color=COLOR_URBAN_EDGE, label="Urban zone"),
        mpatches.Patch(color=COLOR_RURAL_EDGE, label="Rural zone"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.2,
                   linestyle="--", label=f"Dataset mean Moving ({mean_mov:.1f}%)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right",
              framealpha=0.85)
    ax.set_title(
        "Moving / Parked Proportion by COS2023 Land-Use Class",
        fontsize=12, fontweight="bold", pad=12
    )
    n_zones_excluded = int((lu["n_points"] > 0).sum() - len(plot_df))
    fig.text(
        0.5, -0.01,
        f"{_tag()}  —  classes with ≥ {MIN_ZONE_POINTS:,} records"
        f"  ({n_zones_excluded} excluded for insufficient data)"
        f"  —  ordered by Moving % ↑",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Full pipeline:
      1. Load and process the combined COS2023 join file.
      2. Build and save the monthly summary table (Output 1).
      3. Build and save the COS2023 land-use statistics table (Output 2).
      4. Generate and save the four figures (Output 3).
    """
    print("═" * 70)
    print("  SOLAR AUTONOMY ANALYSIS — COS2023 LAND-USE STATISTICS")
    print("═" * 70)
    print(f"\n  CAMPAIGN           : {CAMPAIGN}")
    if CAMPAIGN == "tuktuk":
        print(f"  TUKTUK_FILE_SUFFIX : {TUKTUK_FILE_SUFFIX}")
        for s, p in INPUT_FILES:
            status = "OK" if p.exists() else "NOT FOUND"
            print(f"    [{status}] {p.name}")
    else:
        print(f"  FILE_SUFFIX        : {FILE_SUFFIX}")
        for s, p in INPUT_FILES:
            status = "OK" if p.exists() else "NOT FOUND"
            print(f"    [{status}] {p.name}")
    print(f"  CVE                : {CVE_KWH_PER_100KM} kWh/100 km")
    print(f"  THRESHOLD_M        : {THRESHOLD_M:.0f} m  (parked/moving)")
    print(f"  MIN_MEASUREMENTS   : {MIN_MEASUREMENTS}  (monthly table mask)")
    print(f"  MIN_ZONE_POINTS    : {MIN_ZONE_POINTS}  (Fig 4 class filter)")
    print(f"  MIN_FIG_ZONE_POINTS: {MIN_FIG_ZONE_POINTS}  (Figs 2 & 3 sparse-class filter)")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading COS2023 input file(s) ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_combined_cos23(INPUT_FILES, SENSORS)

    cos23_classes = data["cos23_n4_label"].nunique()
    print(f"  {cos23_classes} distinct COS2023 level-4 classes observed")
    print(f"  Sensors present: {sorted(data['sensor'].unique())}")
    print(f"  Date range: {data['date'].min()} → {data['date'].max()}")

    # ── 2. Monthly table ──────────────────────────────────────────────────────
    print("\n[2/4] Building monthly summary table ...")
    monthly = build_monthly_table(data)

    masked = monthly[monthly["energy_loss_kWh_m2"].isna()]["month_name"].tolist()
    if masked:
        print(f"  [INFO] Months with < {MIN_MEASUREMENTS} records (columns masked): "
              f"{', '.join(masked)}")

    monthly.to_csv(OUT_CSV_MONTHLY, index=False, float_format="%.6f")
    print(f"  CSV saved → {OUT_CSV_MONTHLY}")

    print(f"\n  {'Month':<6} {'N':>8}  {'Loss (kWh)':>12}  "
          f"{'Auto (km/kWp/d)':>16}  {'Dist (km)':>10}")
    print(f"  {'─'*6} {'─'*8}  {'─'*12}  {'─'*16}  {'─'*10}")
    for _, r in monthly.iterrows():
        n    = f"{int(r['n_measurements']):>8,}"
        loss = (f"{r['energy_loss_kWh_m2']:>12.4f}"
                if pd.notna(r["energy_loss_kWh_m2"]) else "         N/A")
        auto = (f"{r['auto_solar_km_per_day']:>16.3f}"
                if pd.notna(r["auto_solar_km_per_day"]) else "             N/A")
        dist = (f"{r['distance_total_km']:>10.2f}"
                if pd.notna(r["distance_total_km"]) else "       N/A")
        print(f"  {r['month_name']:<6} {n}  {loss}  {auto}  {dist}")

    # ── 3. COS2023 land-use table ─────────────────────────────────────────────
    print("\n[3/4] Building COS2023 land-use statistics table ...")
    landuse = build_cos23_landuse_table(data)

    landuse.to_csv(OUT_CSV_LANDUSE, index=False, float_format="%.6f")
    print(f"  CSV saved → {OUT_CSV_LANDUSE}")
    print(f"  {len(landuse)} COS2023 classes observed in the data")

    # Top-5 summary in console
    print("\n  Top-5 classes by measurement count:")
    top5 = landuse.nlargest(5, "n_points")
    for _, r in top5.iterrows():
        print(f"    {r['cos23_n4_label'][:55]:<55} "
              f"  n={r['n_points']:>8,}  loss={r['auto_loss_pct']:.1f}%")

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures ...")

    print("  Fig 1 — Top-10 COS2023 classes by measurement count ...")
    plot_fig1_top10_classes(landuse, OUT_FIG1)

    print("  Fig 2 — Dual-axis loss rate + fraction captured ...")
    plot_fig2_loss_frac(landuse, OUT_FIG2)

    print("  Fig 3 — Relative autonomy losses (%) by COS2023 class ...")
    plot_fig3_loss_pct(landuse, OUT_FIG3)

    print("  Fig 4 — Moving / Parked proportion by COS2023 class ...")
    plot_fig4_moving_parked(landuse, OUT_FIG4)

    print("\n" + "═" * 70)
    print("  Done.")
    print("═" * 70)


if __name__ == "__main__":
    main()