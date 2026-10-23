"""
================================================================================
Diffuse Fraction vs. Clearness Index  —  k_d / k_t  Diagram and Monthly k_t
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Produces:

  Figure 1  k_d vs k_t scatter diagram (equivalent to Fig. 1 of Russo et al.
            2022 / Perez-Astudillo et al. 2018), all sensors pooled, with the
            quality-control flags below highlighted.
  Figure 2  Monthly mean clearness index k_t (bars, ±1 std, IQR), for
            comparison with published climatological values.
  CSV       Monthly k_t statistics (mean, std, median, q25, q75, count).

Physical definitions
--------------------
  k_t  = GHI / GHI_extra          (clearness index, dimensionless [0, 1])
  k_d  = DHI / GHI                (diffuse fraction, dimensionless [0, 1])
  GHI_extra  : extraterrestrial horizontal irradiance  [W/m²]  — column
               'ghi_extra' in the VIPV CSVs, computed as
               Gsc × E0(doy) × sin(sun_elev).

Anomaly flags applied to the scatter diagram
---------------------------------------------
GHI, DHI, and GHI_extra are CAMS Heliosat satellite model outputs, not
independent sensor measurements.  CAMS components are self-consistent by
construction (GHI = BHI + DHI is enforced internally), so the instrument-
consistency checks from Perez-Astudillo et al. (2018) — Flags 3, 4, 9, 10,
12, 13, 17 — are meaningless on CAMS columns and are NOT applied here.
Applying them would silently remove valid CAMS data without physical
justification.

Two flags are retained because they test the physical plausibility of the
(k_t, k_d) pair itself, independently of whether the values come from a
sensor or a model:

  Flag 14 : k_t < 0.2  AND  k_d < 0.9
      Very low global irradiance combined with a low diffuse fraction is
      physically improbable: when total irradiance is near zero the sky is
      overcast and k_d should approach 1.  This combination in CAMS data
      may indicate a model anomaly (e.g. topographic shading artefact).

  Flag 15 : k_t > 0.5  AND  k_d > 0.8
      Moderately clear conditions combined with an anomalously high diffuse
      fraction is also improbable: a significant direct beam should suppress
      k_d well below 0.8.  In a pyranometer network this pattern signals
      tracker failure; in CAMS data it indicates a modelling inconsistency.

Additionally, for variants without 'hq' in their suffix, the two sensor-level
checks of clean_data.py are reproduced on rad_calib to flag physically
impossible sensor readings:

  Check A : rad_calib > Gsc × sin(sun_elev)^1.2 × 1.50 + 100  [W/m²]
      Adapted from Perez-Astudillo et al. (2018), Flag 03.
      Applied to the sensor measurement only (not to CAMS columns).

  Check B : kt_prime = (rad_calib / ghi_extra) / f(air_mass) > 1
      Modified clearness index > 1 is physically impossible.
      Adapted from Perez-Astudillo et al. (2018), Flag 17.
      Applied to the sensor measurement only (not to CAMS columns).

On the '_hq' variants, Checks A and B were already applied by clean_data.py
(Step 2) and are not recomputed (APPLY_SENSOR_FLAGS = False).

Flag colour code on the scatter diagram:
  Blue  — no anomaly detected
  Red   — Flag 14 or 15 raised  (CAMS model anomaly)
  Green — Check A or B raised   (sensor reading anomaly, _cleaned only)
Points flagged by Flag 14 or 15 are excluded from the monthly k_t statistics.

Terminology (project definitions)
---------------------------------
  k_t = clearness index  = GHI / GHI_extra   (used here)
  k_d = diffuse fraction = DHI / GHI         (used here)
  k_c = clear-sky index  = GHI / GHI_clear   (not used here)

Input / output
--------------
    input  : clean_data.py outputs (FILE_SUFFIX / TUKTUK_FILE_SUFFIX)
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> Rad calib',
             'ghi', 'ghi_extra', 'dhi', 'sun_elev', '<SENSOR> lat', '<SENSOR> lon'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             dri_kd_kt_scatter_<CAMPAIGN>_<variant_tag>.png   (Figure 1)
             dri_monthly_kt_<CAMPAIGN>_<variant_tag>.png      (Figure 2)
             dri_monthly_kt_<CAMPAIGN>_<variant_tag>.csv      (monthly statistics)

References
----------
  Perez-Astudillo D., Bachour D., Martin-Pomares L. (2018).
      Improved quality control protocols on solar radiation measurements.
      Solar Energy 169, 425–433.  DOI: 10.1016/j.solener.2018.05.028

  Russo P., Laguarda A., Abal G., Piccioli I. (2022).
      Performance of the REST2 model for 1-minute clear-sky solar irradiance
      with MERRA-2 atmospheric inputs.
      IX Congresso Brasileiro de Energia Solar, Florianópolis.

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.0
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
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py variant, e.g. "sd_cleaned", "sd_cleaned_lisbon")
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output base directory (sub-folder per campaign / variant created) --------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "3_irradiance" / "kd_kt_outputs"

# --- Display figures during execution? ----------------------------------------
SHOW_FIGURES: bool = False

# --- Global figure quality settings -------------------------------------------
FIGURE_DPI:       int = 300
FIGURE_FONT:      str = 'DejaVu Sans'
FIGURE_FONT_SIZE: int = 9

plt.rcParams.update({
    'figure.dpi':           FIGURE_DPI,
    'savefig.dpi':          FIGURE_DPI,
    'font.family':          FIGURE_FONT,
    'font.size':            FIGURE_FONT_SIZE,
    'axes.linewidth':       0.7,
    'axes.labelsize':       10,
    'axes.titlesize':       11,
    'legend.fontsize':      8.5,
    'xtick.major.width':    0.7,
    'ytick.major.width':    0.7,
    'xtick.minor.width':    0.4,
    'ytick.minor.width':    0.4,
    'xtick.labelsize':      8.5,
    'ytick.labelsize':      8.5,
})

# --- QC / physics constants ---------------------------------------------------
# Solar constant [W/m²]  (WMO value, same as clean_data.py SOLAR_CONSTANT_W_M2)
GSC: float = 1361.0

# Minimum solar elevation for valid daytime data [degrees]
MIN_SUN_ELEV: float = 10.0

# --- Sensor-level Checks A / B on rad_calib -----------------------------------
# Already applied by clean_data.py (Step 2) on the '_hq' variants: suppressed
# automatically when the active variant tag contains 'hq'.
def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


_VARIANT_TAG: str = _variant_tag()
APPLY_SENSOR_FLAGS: bool = 'hq' not in _VARIANT_TAG

# --- Sensor column mappings ---------------------------------------------------
SENSOR_COLS: dict[str, dict[str, str]] = {
    'J30': {'date': 'Human readable date (Europe/London)',
            'rad':  'J30 Rad calib', 'ghi': 'ghi',
            'ghi_extra': 'ghi_extra', 'dhi': 'dhi',
            'elev': 'sun_elev', 'lat': 'J30 lat', 'lon': 'J30 lon'},
    'J31': {'date': 'Human readable date (Europe/London)',
            'rad':  'J31 Rad calib', 'ghi': 'ghi',
            'ghi_extra': 'ghi_extra', 'dhi': 'dhi',
            'elev': 'sun_elev', 'lat': 'J31 lat', 'lon': 'J31 lon'},
    'J32': {'date': 'Human readable date (Europe/London)',
            'rad':  'J32 Rad calib', 'ghi': 'ghi',
            'ghi_extra': 'ghi_extra', 'dhi': 'dhi',
            'elev': 'sun_elev', 'lat': 'J32 lat', 'lon': 'J32 lon'},
    'J33': {'date': 'Human readable date (Europe/London)',
            'rad':  'J33 Rad calib', 'ghi': 'ghi',
            'ghi_extra': 'ghi_extra', 'dhi': 'dhi',
            'elev': 'sun_elev', 'lat': 'J33 lat', 'lon': 'J33 lon'},
}

# ==============================================================================
# FILENAME RESOLUTION  (pipeline convention)
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor names to their CSV input paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    (after normalize_tuktuk.py, tuk-tuk column names already match the
    standard convention — e.g. 'J31 Rad calib' — so SENSOR_COLS above is
    shared unchanged between both campaigns.)

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
        {'J30': Path(...), 'J31': Path(...), ...}
    """
    sensors = ['J30', 'J31', 'J32', 'J33']
    if campaign == "standard":
        return {
            sensor: data_dir / f"{sensor.lower()}_satellite_landuse_v2_{suffix}.csv"
            for sensor in sensors
        }
    elif campaign == "tuktuk":
        return {
            sensor: tuktuk_data_dir / f"{sensor.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
            for sensor in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """
    Build and return the output directory path, embedding the campaign and
    active variant tag so outputs from different campaigns/variants never
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
        Full output directory path (created on disk if it does not yet exist).
    """
    out = base_dir / campaign / variant_tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_output_path(out_dir: Path, base_name: str,
                       campaign: str, variant_tag: str, extension: str) -> Path:
    """
    Build an output file path that embeds the active campaign and variant tag.

    Parameters
    ----------
    out_dir : Path
        Output directory (already resolved).
    base_name : str
        Descriptive part of the filename (e.g. 'dri_kd_kt_scatter').
    campaign : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().
    extension : str
        File extension without leading dot (e.g. 'png').

    Returns
    -------
    Path
        Complete file path, e.g.
        out_dir / 'dri_kd_kt_scatter_standard_sd_cleaned.png'.
    """
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolve all paths at startup
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
)
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)

OUTPUT_SCATTER = _build_output_path(OUTPUT_DIR, 'dri_kd_kt_scatter', CAMPAIGN, _VARIANT_TAG, 'png')
OUTPUT_MONTHLY = _build_output_path(OUTPUT_DIR, 'dri_monthly_kt',    CAMPAIGN, _VARIANT_TAG, 'png')
OUTPUT_CSV     = _build_output_path(OUTPUT_DIR, 'dri_monthly_kt',    CAMPAIGN, _VARIANT_TAG, 'csv')


# ==============================================================================
# SECTION 1  —  DATA LOADING AND PREPARATION
# ==============================================================================

def load_sensor(sensor: str, csv_path: Path) -> pd.DataFrame | None:
    """
    Load and pre-process one sensor CSV.

    Steps
    -----
    1. Read the CSV with low_memory=False.
    2. Rename sensor-specific columns to generic names for uniform processing.
    3. Parse the datetime column to UTC-aware timestamps.
    4. Cast numeric columns and drop non-parseable rows.
    5. Apply the minimum solar-elevation filter (daytime only).
    6. Require strictly positive GHI and GHI_extra (needed for k_t).
    7. Tag the sensor name.

    Parameters
    ----------
    sensor : str
        Sensor identifier, e.g. 'J31'.
    csv_path : Path
        Full path to the cleaned CSV file.

    Returns
    -------
    pd.DataFrame or None
        Pre-processed DataFrame, or None if the file is not found.
    """
    if not csv_path.exists():
        print(f'  [WARNING] File not found, skipping: {csv_path}')
        return None

    print(f'\nLoading sensor {sensor}: {csv_path.name}')
    df = pd.read_csv(csv_path, low_memory=False)
    print(f'  → {len(df):,} rows loaded')

    cols = SENSOR_COLS[sensor]

    # Rename to generic column names for uniform downstream processing
    rename_map = {
        cols['date']:      'datetime',
        cols['rad']:       'rad_calib',
        cols['ghi']:       'ghi',
        cols['ghi_extra']: 'ghi_extra',
        cols['dhi']:       'dhi',
        cols['elev']:      'sun_elev',
        cols['lat']:       'lat',
        cols['lon']:       'lon',
    }
    df = df.rename(columns=rename_map)

    needed = ['datetime', 'rad_calib', 'ghi', 'ghi_extra', 'dhi', 'sun_elev', 'lat', 'lon']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f'  [WARNING] Missing columns {missing}, skipping {sensor}')
        return None

    df = df[needed].copy()

    # Parse timestamps
    df['datetime'] = pd.to_datetime(
        df['datetime'], utc=True, errors='coerce'
    )
    df = df.dropna(subset=['datetime'])
    df['month'] = df['datetime'].dt.month

    # Cast numeric columns
    for col in ['rad_calib', 'ghi', 'ghi_extra', 'dhi', 'sun_elev', 'lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in any required numeric column
    df = df.dropna(subset=['rad_calib', 'ghi', 'ghi_extra', 'dhi', 'sun_elev'])

    # Daytime filter: strictly positive sun elevation and GHI/GHI_extra
    df = df[df['sun_elev']   >= MIN_SUN_ELEV]
    df = df[df['ghi']        >  0.0]
    df = df[df['ghi_extra']  >  0.0]

    df['sensor'] = sensor
    print(f'  → {len(df):,} valid daytime rows retained')
    return df


# ==============================================================================
# SECTION 2  —  QC FLAGS  (Perez-Astudillo et al., 2018)
# ==============================================================================

def compute_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute anomaly flags for the k_d / k_t scatter diagram.

    Design rationale
    ----------------
    GHI, DHI, and GHI_extra are CAMS Heliosat satellite model outputs.
    CAMS components are self-consistent by construction, so instrument-
    consistency checks from Perez-Astudillo et al. (2018) that target
    measurement artefacts of physical pyranometers (Flags 3, 4, 9, 10,
    12, 13, 17) are NOT meaningful on CAMS columns and are not applied.

    Flags applied to CAMS columns (k_t / k_d pair plausibility)
    ------------------------------------------------------------
    Flag 14 : k_t < 0.2  AND  k_d < 0.9
        Very low global irradiance combined with a low diffuse fraction is
        physically improbable — under near-zero irradiance the sky is
        overcast and k_d should approach 1.  This combination in CAMS data
        indicates a model anomaly (e.g. topographic shading artefact).

    Flag 15 : k_t > 0.5  AND  k_d > 0.8
        A moderately clear atmosphere combined with an anomalously high
        diffuse fraction is improbable — a non-negligible direct beam
        should suppress k_d well below 0.8.  This indicates a CAMS
        modelling inconsistency.

    Sensor-level checks applied to rad_calib (_cleaned only)
    ---------------------------------------------------------
    On the '_hq' variants these checks were already applied by
    clean_data.py (Step 2) and are suppressed here (APPLY_SENSOR_FLAGS =
    False).

    Check A : rad_calib > Gsc × sin(sun_elev)^1.2 × 1.50 + 100  [W/m²]
        Physically impossible upper bound.
        Adapted from Perez-Astudillo et al. (2018), Flag 03.

    Check B : kt_prime = (rad_calib / ghi_extra) / f(air_mass) > 1
        Modified clearness index > 1 is physically impossible.
        Adapted from Perez-Astudillo et al. (2018), Flag 17.

    Output columns
    --------------
    flag14         : bool — CAMS k_t / k_d pair anomaly (type 14)
    flag15         : bool — CAMS k_t / k_d pair anomaly (type 15)
    flag_sensor    : bool — sensor rad_calib anomaly (Check A or B);
                            always False when APPLY_SENSOR_FLAGS is False
    cams_anomaly   : bool — flag14 | flag15  (used to exclude from monthly k_t)
    sensor_anomaly : bool — alias for flag_sensor (for plot clarity)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ghi, ghi_extra, dhi, rad_calib,
        sun_elev, k_t, k_d.

    Returns
    -------
    pd.DataFrame
        Copy of df with flag columns added.
    """
    kt  = df['k_t']
    kd  = df['k_d']

    # ── Flag 14 : low k_t / low k_d  (CAMS model anomaly) ───────────────────
    flag14 = (kt < 0.2) & (kd < 0.9)

    # ── Flag 15 : high k_t / high k_d  (CAMS model anomaly) ─────────────────
    flag15 = (kt > 0.5) & (kd > 0.8)

    # ── Sensor Check A + B  (rad_calib anomaly, _cleaned only) ───────────────
    if APPLY_SENSOR_FLAGS:
        sin_elev   = np.sin(np.radians(df['sun_elev'].clip(lower=0.0)))
        upper_bound = GSC * (sin_elev.clip(lower=0.0) ** 1.2) * 1.50 + 100.0
        check_A    = df['rad_calib'] > upper_bound

        sin_elev_safe = sin_elev.clip(lower=0.01)
        air_mass      = np.clip(1.0 / sin_elev_safe, 1.0, 40.0)
        kt_prime_denom = 1.031 * np.exp(-1.4 / (0.9 + 9.4 / air_mass)) + 0.1
        kt_sensor  = df['rad_calib'] / df['ghi_extra'].clip(lower=1e-6)
        kt_prime   = kt_sensor / np.clip(kt_prime_denom, 1e-6, None)
        check_B    = kt_prime > 1.0

        flag_sensor = check_A | check_B
    else:
        flag_sensor = pd.Series(False, index=df.index)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    df = df.copy()
    df['flag14']        = flag14
    df['flag15']        = flag15
    df['flag_sensor']   = flag_sensor
    df['cams_anomaly']  = flag14 | flag15        # excludes from monthly k_t
    df['sensor_anomaly'] = flag_sensor

    n14  = flag14.sum()
    n15  = flag15.sum()
    n_sen = flag_sensor.sum()
    print(f'  Anomaly flags — '
          f'Flag14 (CAMS): {n14:,}  |  '
          f'Flag15 (CAMS): {n15:,}  |  '
          f'Sensor (rad_calib): {n_sen:,}'
          + ('' if APPLY_SENSOR_FLAGS else '  [suppressed — hq variant]'))

    return df


# ==============================================================================
# SECTION 3  —  FIGURE 1  :  k_d  vs  k_t  scatter  (Russo et al. style)
# ==============================================================================

def plot_kd_kt_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot the diffuse fraction k_d vs. clearness index k_t scatter diagram.

    Colour scheme
    -------------
    Blue  (density histogram) — no anomaly detected.
    Red   (scatter overlay)   — CAMS model anomaly: Flag 14 (low k_t / low k_d)
                                or Flag 15 (high k_t / high k_d).
    Green (scatter overlay)   — sensor reading anomaly: rad_calib fails Check A
                                or Check B.  Only shown for the _cleaned variant;
                                suppressed on _cleaned_hq / _cleaned_hq_clearsky
                                because clean_data.py has already removed these
                                rows (APPLY_SENSOR_FLAGS = False).

    The hatched rectangles mark the Flag 14 and Flag 15 anomaly zones.

    Parameters
    ----------
    df : pd.DataFrame
        Pooled data for all sensors.  Must contain 'k_t', 'k_d',
        'cams_anomaly', and 'sensor_anomaly' columns.
    out_path : Path
        Destination PNG file path.
    """
    # Three mutually exclusive categories
    # Priority: sensor anomaly > CAMS anomaly > clean
    mask_sensor = df['sensor_anomaly']
    mask_cams   = df['cams_anomaly'] & ~mask_sensor
    mask_clean  = ~df['cams_anomaly'] & ~mask_sensor

    df_clean  = df[mask_clean]
    df_cams   = df[mask_cams]
    df_sensor = df[mask_sensor]

    total    = len(df)
    n_clean  = len(df_clean)
    n_cams   = len(df_cams)
    n_sensor = len(df_sensor)
    pct_cams   = 100.0 * n_cams   / max(total, 1)
    pct_sensor = 100.0 * n_sensor / max(total, 1)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # ── Density histogram for clean points ────────────────────────────────────
    if n_clean > 0:
        h = ax.hist2d(
            df_clean['k_t'].values,
            df_clean['k_d'].values,
            bins=120,
            range=[[0.0, 1.2], [0.0, 1.2]],
            norm=LogNorm(vmin=1),
            cmap='Blues',
            zorder=2,
        )
        cb = fig.colorbar(h[3], ax=ax, pad=0.02, shrink=0.85)
        cb.set_label('Count per bin  (log scale)', fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)

    # ── CAMS anomaly points (Flag 14 / 15) — red ─────────────────────────────
    if n_cams > 0:
        ax.scatter(
            df_cams['k_t'].values,
            df_cams['k_d'].values,
            s=2.0, c='#e74c3c', alpha=0.40,
            linewidths=0, zorder=3, rasterized=True,
            label=f'CAMS anomaly — Flag 14/15  (n = {n_cams:,},  {pct_cams:.1f}%)',
        )

    # ── Sensor anomaly points (Check A / B) — green ───────────────────────────
    if n_sensor > 0 and APPLY_SENSOR_FLAGS:
        ax.scatter(
            df_sensor['k_t'].values,
            df_sensor['k_d'].values,
            s=2.0, c='#27ae60', alpha=0.50,
            linewidths=0, zorder=4, rasterized=True,
            label=f'Sensor anomaly — Check A/B  (n = {n_sensor:,},  {pct_sensor:.1f}%)',
        )

    # ── Anomaly zone rectangles (Flags 14 and 15) ─────────────────────────────
    # Flag 14: k_t < 0.2  AND  k_d < 0.9  →  lower-left region
    ax.add_patch(plt.Rectangle(
        (0.0, 0.0), 0.2, 0.9,
        linewidth=1.0, edgecolor='#c0392b', facecolor='#c0392b',
        alpha=0.07, hatch='///', zorder=1,
    ))
    # Flag 15: k_t > 0.5  AND  k_d > 0.8  →  upper-right region
    ax.add_patch(plt.Rectangle(
        (0.5, 0.8), 0.7, 0.4,
        linewidth=1.0, edgecolor='#c0392b', facecolor='#c0392b',
        alpha=0.07, hatch='///', zorder=1,
    ))

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel(r'Clearness index  $k_t$ = GHI$_{\rm CAMS}$ / GHI$_{\rm extra}$',
                  fontsize=10)
    ax.set_ylabel(r'Diffuse fraction  $k_d$ = DHI$_{\rm CAMS}$ / GHI$_{\rm CAMS}$',
                  fontsize=10)
    ax.set_title(
        r'Diffuse Fraction $k_d$ vs. Clearness Index $k_t$',
        fontsize=11, fontweight='bold', pad=32,
    )
    ax.text(
        0.5, 1.01,
        f'All sensors pooled  —  {total:,} daytime measurements  '
        f'(GHI, DHI: CAMS Heliosat satellite model)',
        transform=ax.transAxes,
        fontsize=7.5, style='italic', color='#555555',
        ha='center', va='bottom',
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which='major', linestyle='--', alpha=0.30, zorder=0)
    ax.grid(which='minor', linestyle=':',  alpha=0.15, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    proxy_clean  = mpatches.Patch(color='#2980b9', alpha=0.7,
                                  label=f'No anomaly  (n = {n_clean:,})  — density')
    proxy_cams   = mpatches.Patch(color='#e74c3c', alpha=0.7,
                                  label=f'CAMS anomaly — Flag 14/15  '
                                        f'(n = {n_cams:,},  {pct_cams:.1f}%)')
    proxy_r14    = mpatches.Patch(facecolor='#c0392b', alpha=0.12,
                                  hatch='///', edgecolor='#c0392b',
                                  label='Flag 14 zone  ($k_t$ < 0.2, $k_d$ < 0.9)')
    proxy_r15    = mpatches.Patch(facecolor='#c0392b', alpha=0.12,
                                  hatch='///', edgecolor='#c0392b',
                                  label='Flag 15 zone  ($k_t$ > 0.5, $k_d$ > 0.8)')

    legend_handles = [proxy_clean, proxy_cams, proxy_r14, proxy_r15]

    if APPLY_SENSOR_FLAGS:
        proxy_sensor = mpatches.Patch(color='#27ae60', alpha=0.7,
                                      label=f'Sensor anomaly — Check A/B  '
                                            f'(n = {n_sensor:,},  {pct_sensor:.1f}%)')
        legend_handles.insert(2, proxy_sensor)

    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=8.0,
        framealpha=0.92,
        edgecolor='#cccccc',
        fancybox=False,
    )

    # ── Footnote ──────────────────────────────────────────────────────────────
    sensor_note = (
        'Sensor Check A/B (rad_calib): adapted from Perez-Astudillo et al. (2018), '
        'Flags 03 & 17.'
        if APPLY_SENSOR_FLAGS
        else 'Sensor Check A/B suppressed — already applied by clean_data.py (hq variant).'
    )
    fig.text(
        0.01, 0.005,
        'CAMS anomaly flags: Flag 14 (low $k_t$, low $k_d$) and Flag 15 '
        '(high $k_t$, high $k_d$)  —  '
        + sensor_note,
        fontsize=6.5, style='italic', color='#888888',
        ha='left', va='bottom',
    )

    fig.tight_layout(rect=[0, 0.030, 1, 1])
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f'  Figure saved: {out_path}')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ==============================================================================
# SECTION 4  —  FIGURE 2  :  Monthly mean k_t  (bar chart)
# ==============================================================================

def plot_monthly_kt(df_good: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Plot monthly mean clearness index k_t as a bar chart with ±1 std bands.

    Only QC-passed data are used.  The bar height represents the monthly
    arithmetic mean of k_t; error bars show ±1 standard deviation.
    The number of valid 1-minute measurements per month is annotated above
    each bar.

    Parameters
    ----------
    df_good : pd.DataFrame
        QC-passed subset of the pooled data, with 'k_t' and 'month' columns.
    out_path : Path
        Destination PNG file path.

    Returns
    -------
    pd.DataFrame
        Monthly statistics table (mean, std, median, q25, q75, count).
    """
    MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Compute statistics per calendar month
    grp = df_good.groupby('month')['k_t']
    stats = pd.DataFrame({
        'month':  range(1, 13),
        'mean':   [grp.get_group(m).mean()   if m in grp.groups else np.nan for m in range(1, 13)],
        'std':    [grp.get_group(m).std()    if m in grp.groups else np.nan for m in range(1, 13)],
        'median': [grp.get_group(m).median() if m in grp.groups else np.nan for m in range(1, 13)],
        'q25':    [grp.get_group(m).quantile(0.25) if m in grp.groups else np.nan for m in range(1, 13)],
        'q75':    [grp.get_group(m).quantile(0.75) if m in grp.groups else np.nan for m in range(1, 13)],
        'count':  [grp.get_group(m).count()  if m in grp.groups else 0       for m in range(1, 13)],
    })
    stats['month_name'] = MONTH_NAMES

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    x     = np.arange(1, 13)
    color = '#2980b9'

    ax.bar(
        x, stats['mean'],
        color=color, alpha=0.75, width=0.6,
        zorder=3, label='Monthly mean $k_t$',
    )

    # Error bars: ±1 std
    ax.errorbar(
        x, stats['mean'],
        yerr=stats['std'],
        fmt='none',
        ecolor='#1a5276',
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
        label='±1 std deviation',
    )

    # IQR shading (Q25–Q75) as light overlay
    ax.bar(
        x, stats['q75'] - stats['q25'],
        bottom=stats['q25'],
        color='#aed6f1', alpha=0.50, width=0.6,
        zorder=2, label='IQR (Q25–Q75)',
    )

    # Annotate count above each bar
    for i, row in stats.iterrows():
        if row['count'] > 0:
            y_top = row['mean'] + row['std'] + 0.02
            ax.text(
                row['month'], min(y_top, 1.02),
                f"{int(row['count']):,}",
                ha='center', va='bottom',
                fontsize=6.5, color='#333333', rotation=90,
            )

    # Reference lines
    ax.axhline(stats['mean'].mean(), color='#e74c3c', linewidth=1.0,
               linestyle='--', zorder=5,
               label=f"Annual mean = {stats['mean'].mean():.3f}")
    ax.axhline(1.0, color='#888888', linewidth=0.6, linestyle=':', zorder=1)

    ax.set_xlim(0.35, 12.65)
    ax.set_ylim(0.0, 1.10)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel(r'Clearness index  $k_t$ = GHI / GHI$_{\rm extra}$',
                  fontsize=10)
    ax.set_title(
        'Monthly Mean Clearness Index $k_t$',
        fontsize=11, fontweight='bold', pad=32,
    )
    ax.text(
        0.5, 1.01,
        f'All sensors pooled  —  CAMS anomalies excluded  '
        f'({len(df_good):,} measurements)',
        transform=ax.transAxes,
        fontsize=7.5, style='italic', color='#555555',
        ha='center', va='bottom',
    )

    ax.grid(axis='y', which='major', linestyle='--', alpha=0.30, zorder=0)
    ax.grid(axis='y', which='minor', linestyle=':',  alpha=0.15, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.92,
              edgecolor='#cccccc', fancybox=False)

    fig.text(
        0.01, 0.005,
        'Numbers above bars = count of valid 1-min measurements per month.  '
        'CAMS anomalies (Flag 14/15) excluded.  '
        r'$k_t$ = GHI$_{\rm CAMS}$ / GHI$_{\rm extra}$.',
        fontsize=6.5, style='italic', color='#888888',
        ha='left', va='bottom',
    )

    fig.tight_layout(rect=[0, 0.025, 1, 1])
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f'  Figure saved: {out_path}')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

    return stats


# ==============================================================================
# SECTION 5  —  CSV EXPORT
# ==============================================================================

def save_monthly_csv(stats: pd.DataFrame, out_path: Path) -> None:
    """
    Save the monthly k_t statistics to CSV.

    Columns exported:
        month, month_name, mean, std, median, q25, q75, count

    Values are rounded to 6 decimal places for reproducibility.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of plot_monthly_kt().
    out_path : Path
        Destination CSV file path.
    """
    export = stats[['month', 'month_name', 'mean', 'std',
                    'median', 'q25', 'q75', 'count']].copy()
    for col in ['mean', 'std', 'median', 'q25', 'q75']:
        export[col] = export[col].round(6)

    export.to_csv(out_path, index=False)
    print(f'  CSV saved:    {out_path}')


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Run the full k_d / k_t analysis pipeline:
      1. Load and pool data from all four sensors.
      2. Compute k_t (CAMS) and k_d (CAMS).
      3. Compute anomaly flags (Flag 14/15 on CAMS; Check A/B on rad_calib
         for _cleaned only).
      4. Produce the k_d vs. k_t scatter figure (colour-coded by anomaly type).
      5. Produce the monthly k_t bar chart and CSV (CAMS anomalies excluded).
    """
    print('\n' + '=' * 70)
    print('  k_d / k_t Analysis  —  VIPV Research, FCUL')
    print(f'  CAMPAIGN     : {CAMPAIGN}')
    print(f'  Data variant : {_VARIANT_TAG}')
    print('=' * 70)

    # ── 1. Load and pool all sensors ──────────────────────────────────────────
    all_frames: list[pd.DataFrame] = []

    for sensor, csv_path in SENSOR_FILES.items():
        df_s = load_sensor(sensor, csv_path)
        if df_s is not None:
            all_frames.append(df_s)

    if not all_frames:
        raise RuntimeError(
            'No valid data found. Check DATA_DIR/TUKTUK_DATA_DIR, CAMPAIGN, '
            'and FILE_SUFFIX/TUKTUK_FILE_SUFFIX.'
        )

    data = pd.concat(all_frames, ignore_index=True)
    print(f'\nTotal valid daytime measurements across all sensors: {len(data):,}')

    # ── 2. Compute k_t and k_d ────────────────────────────────────────────────
    data['k_t'] = (data['ghi'] / data['ghi_extra']).clip(0.0, 1.2)
    data['k_d'] = (data['dhi'] / data['ghi']).clip(0.0, 1.2)

    print(f'\nk_t : mean = {data["k_t"].mean():.3f},  std = {data["k_t"].std():.3f}')
    print(f'k_d : mean = {data["k_d"].mean():.3f},  std = {data["k_d"].std():.3f}')

    # ── 3. Compute anomaly flags ──────────────────────────────────────────────
    print('\nComputing anomaly flags...')
    data = compute_qc_flags(data)

    # ── 4. Scatter figure  (all data, anomaly colour-coded) ──────────────────
    print('\nGenerating Figure 1 — k_d vs k_t scatter...')
    plot_kd_kt_scatter(data, OUTPUT_SCATTER)

    # ── 5. Monthly k_t  (CAMS-anomaly-free data only) ────────────────────────
    # Rows flagged by Flag 14 or 15 (cams_anomaly) are excluded from the
    # monthly statistics.  Sensor anomaly rows (flag_sensor) are kept because
    # k_t is computed from CAMS columns, not from rad_calib.
    data_good = data[~data['cams_anomaly']].copy()
    print(f'\nData for monthly k_t statistics '
          f'(CAMS anomaly excluded): {len(data_good):,} rows')

    print('\nGenerating Figure 2 — Monthly k_t bar chart...')
    stats = plot_monthly_kt(data_good, OUTPUT_MONTHLY)

    save_monthly_csv(stats, OUTPUT_CSV)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  Monthly k_t summary  (CAMS anomalies excluded):')
    print('=' * 70)
    print(f"  {'Month':<6} {'Mean':>8} {'Std':>8} {'Median':>8} "
          f"{'Q25':>8} {'Q75':>8} {'Count':>10}")
    print('  ' + '-' * 62)
    for _, row in stats.iterrows():
        if row['count'] > 0:
            print(
                f"  {row['month_name']:<6} "
                f"{row['mean']:8.4f} {row['std']:8.4f} {row['median']:8.4f} "
                f"{row['q25']:8.4f} {row['q75']:8.4f} {int(row['count']):>10,}"
            )
    print('=' * 70)
    print('  Analysis complete.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()