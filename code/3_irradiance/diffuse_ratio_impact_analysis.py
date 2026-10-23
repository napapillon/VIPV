"""
================================================================================
Diffuse Fraction Impact on Solar Autonomy and Irradiance Losses
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Investigates how the diffuse fraction of solar irradiance (DHI / GHI) affects
solar autonomy and irradiance losses across nine measurement conditions:

  1. All data        — full dataset pooled
  2. Parked          — vehicle stationary (Haversine distance < THRESHOLD_M)
  3. Moving          — vehicle in motion  (Haversine distance ≥ THRESHOLD_M)
  4. Urban           — CLC land-use classified as Urban
  5. Rural           — CLC land-use classified as Rural
  6. Parked + Urban  — stationary in an urban environment
  7. Moving + Urban  — in motion in an urban environment
  8. Parked + Rural  — stationary in a rural environment
  9. Moving + Rural  — in motion in a rural environment

Physical context
----------------
GHI = BHI + DHI          (global = beam horizontal + diffuse horizontal)
BHI = DNI × cos(θ)       (beam horizontal = direct normal × cosine of SZA)
Diffuse fraction  k_d = DHI / GHI   (dimensionless, range [0, 1])

Only k_d is used here (project definitions: k_t = GHI / GHI_extra in
kd_kt_analysis.py; k_c = GHI / GHI_clear in the clear-sky filter of
clean_data.py, Step 3).

Under clear-sky conditions (k_d → 0) the direct beam dominates; shading by
vehicle structures or obstacles is most penalising.  Under overcast conditions
(k_d → 1) the irradiance is quasi-isotropic and harder to obstruct
geometrically, so relative losses decrease even if absolute GHI is low.

Method
------
1. Load cleaned 1-minute data for all sensors (J30–J33).
2. Compute per-minute quantities (same definitions as the solar-autonomy
   scripts):
       E_solar    = rad_calib / 1000 × Δt               [kWh/kWp]
       loss_irr   = max(0, GHI − rad_calib) × Δt        [Wh/m²]
       auto_solar = E_solar / CVE                        [km/kWp/min]
       loss_rel   = loss_irr / (loss_irr + E_solar×1000) [%]
3. Derive  k_d = DHI / GHI  (rows where GHI ≤ 0 excluded).
4. Classify each minute into N_BINS equal-width k_d bins.
5. For each condition and metric compute: mean, std, median, q25, q75, count.
6. Produce 18 publication-quality figures (2 statistical modes × 3 metrics
   × 3 groupings):

   Mode A — Mean ± std
       Central line  : arithmetic mean per bin
       Shaded band   : mean ± 1 standard deviation
       File suffix   : _mean
       Interpretation: consistent with solar radiation literature (Perez et al.)
                       Sensitive to outliers; may exceed physical bounds on
                       skewed metrics such as relative losses.

   Mode B — Median + IQR
       Central line  : median (50th percentile) per bin
       Shaded band   : interquartile range (Q25–Q75)
       File suffix   : _median
       Interpretation: robust to outliers and asymmetric distributions; always
                       bounded within the physical data range.

   Grouping A — Movement : All data / Parked / Moving
   Grouping B — Territory: All data / Urban  / Rural
   Grouping C — Crossed  : All data / Parked+Urban / Moving+Urban /
                                      Parked+Rural / Moving+Rural

   The shaded band can be toggled off via SHOW_BAND = False to obtain
   clean line-only figures.
   Bins with count < MIN_BIN_OBS are plotted with a hollow marker to
   flag reduced statistical reliability without hiding the data.
   Regime boundaries (k_d = 0.3 and 0.7) are annotated inside the plot area.

Data quality in CSV
-------------------
Each exported row carries a 'data_quality' column:
    'high'     : count ≥ 1000
    'moderate' : 100 ≤ count < 1000
    'low'      : count < 100
All five statistics (mean, std, median, q25, q75) are exported for every bin
regardless of the active PLOT_MODE, so the CSV is always complete.

Input / output
--------------
    input  : clean_data.py outputs (FILE_SUFFIX / TUKTUK_FILE_SUFFIX); the
             '_sd_cleaned' files are already free of invalid land_use_clcv,
             rad_calib = 0 and pre-deployment rooftop rows
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             dri_<metric>_<grouping>_<mode>_<CAMPAIGN>_<variant_tag>.png
             dri_kd_distribution[_movement|_territory]_<CAMPAIGN>_<variant_tag>.png
             dri_binned_statistics_<CAMPAIGN>_<variant_tag>.csv
             dri_condition_overview_<CAMPAIGN>_<variant_tag>.csv

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 4.0
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
import matplotlib.ticker as ticker
import matplotlib.transforms as _mt

# ── VIPV shared reference (code/common) ───────────────────────────────────────
from clc_mapping import CLC_URBAN_RURAL  # noqa: E402

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

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
#   "sd_cleaned"                      → full date-filtered dataset
#   "sd_cleaned_lisbon"               → Lisbon-only subset
#   "sd_cleaned_lisbon_september2022" → Lisbon, September 2022
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# --- Output directory (sub-folder per campaign / variant created) -------------
OUTPUT_BASE_DIR = OUTPUT_ROOT / "3_irradiance" / "dri_outputs"

# --- Display figures during execution? ----------------------------------------
SHOW_FIGURES: bool = False

# --- Show shaded uncertainty band on figures? ---------------------------------
# True  → draw the uncertainty band (±1σ for mean mode, IQR for median mode).
# False → lines and markers only; cleaner output for presentations or overlays.
SHOW_BAND: bool = True

# --- Statistical modes to generate --------------------------------------------
# Each entry produces a full set of 9 figures (3 metrics × 3 groupings).
# Both modes share the same CSV export (all statistics are always computed).
#
#   'mean'   → central line = arithmetic mean, band = mean ± 1 std deviation
#   'median' → central line = median (Q50),   band = IQR (Q25–Q75)
#
# To generate only one mode, remove the other entry from the list.
PLOT_MODES: list[str] = ['mean', 'median']

# --- Global figure quality settings ------------------------------------------
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
    'axes.titlesize':       10,
    'legend.fontsize':      8.5,
    'legend.title_fontsize':8.0,
    'xtick.major.width':    0.7,
    'ytick.major.width':    0.7,
    'xtick.minor.width':    0.4,
    'ytick.minor.width':    0.4,
    'xtick.labelsize':      8.5,
    'ytick.labelsize':      8.5,
    'lines.linewidth':      1.6,
})

# --- Sensor column mappings (same for every variant and both campaigns) -------
SENSOR_COLS: dict[str, dict[str, str]] = {

    'J30': {'date': 'Human readable date (Europe/London)',
            'rad':  'J30 Rad calib', 'ghi': 'ghi', 'dhi': 'dhi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J30 lat',       'lon':  'J30 lon'},
    'J31': {'date': 'Human readable date (Europe/London)',
            'rad':  'J31 Rad calib', 'ghi': 'ghi', 'dhi': 'dhi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J31 lat',       'lon':  'J31 lon'},
    'J32': {'date': 'Human readable date (Europe/London)',
            'rad':  'J32 Rad calib', 'ghi': 'ghi', 'dhi': 'dhi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J32 lat',       'lon':  'J32 lon'},
    'J33': {'date': 'Human readable date (Europe/London)',
            'rad':  'J33 Rad calib', 'ghi': 'ghi', 'dhi': 'dhi',
            'elev': 'sun_elev',      'land': 'land_use_clcv',
            'lat':  'J33 lat',       'lon':  'J33 lon'},
}

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================
# Input and output paths follow from CAMPAIGN and the input suffix.

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

    This ensures that output files from different campaigns/data variants
    are uniquely named and never overwrite each other even if placed in the
    same folder.

    Parameters
    ----------
    out_dir     : Path
        Output directory (already resolved).
    base_name   : str
        Descriptive part of the filename (e.g. 'dri_autonomy_movement').
    campaign    : str
        Active CAMPAIGN value.
    variant_tag : str
        Active variant tag, from _variant_tag().
    extension   : str
        File extension without leading dot (e.g. 'png', 'csv').

    Returns
    -------
    Path
        Complete file path, e.g.
        out_dir / 'dri_autonomy_movement_standard_sd_cleaned.png'.
    """
    return out_dir / f"{base_name}_{campaign}_{variant_tag}.{extension}"


# Resolve all paths at startup
_VARIANT_TAG: str = _variant_tag()
SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _VARIANT_TAG
)
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Output paths (figures + CSVs) -------------------------------------------
# Naming convention: dri_<description>_<CAMPAIGN>_<variant_tag>.<ext>
OUTPUT_CSV      = _build_output_path(OUTPUT_DIR, 'dri_binned_statistics',  CAMPAIGN, _VARIANT_TAG, 'csv')
OUTPUT_OVERVIEW = _build_output_path(OUTPUT_DIR, 'dri_condition_overview', CAMPAIGN, _VARIANT_TAG, 'csv')

# ==============================================================================
# PHYSICAL / VEHICLE PARAMETERS
# ==============================================================================

# --- Vehicle consumption and time step ----------------------------------------
CVE_KWH_PER_100KM: float = 16.6
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0
DELTA_T_H:         float = 1.0 / 60.0

# --- Timezone ----------------------------------------------------------------
LOCAL_TZ:          str   = 'Europe/Lisbon'

# --- Movement classification -------------------------------------------------
THRESHOLD_M: float = 30.0
R_EARTH_KM:  float = 6371.0
MAX_GAP_MINUTES: int = 2

# --- Binning -----------------------------------------------------------------
N_BINS:      int        = 10
BIN_EDGES:   np.ndarray = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_CENTRES: np.ndarray = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# Bins below MIN_BIN_OBS are drawn with hollow markers (data kept in CSV).
MIN_BIN_OBS: int = 30

# --- Data quality thresholds -------------------------------------------------
QUALITY_LOW:      int = 100
QUALITY_MODERATE: int = 1000

# ==============================================================================
# CONDITION DEFINITIONS
# Three groupings, each a list of condition dicts.
# "All data" is always the first entry in each group (grey reference).
# ==============================================================================

# Colour palette — chosen for distinctiveness and colour-blind friendliness.
_C = {
    'all':    '#333333',   # dark grey  — reference
    'park':   '#1565C0',   # blue
    'move':   '#C62828',   # red
    'urban':  '#E65100',   # orange
    'rural':  '#2E7D32',   # green
    'pu':     '#6A1B9A',   # purple
    'mu':     '#AD1457',   # pink-red
    'pr':     '#00838F',   # teal
    'mr':     '#558B2F',   # olive
}

GROUPS: dict[str, list[dict]] = {
    'movement': [
        {'label': 'All data', 'color': _C['all'],  'ls': '-',  'lw': 2.0,
         'filter': lambda df: df},
        {'label': 'Parked',   'color': _C['park'], 'ls': '-',  'lw': 1.6,
         'filter': lambda df: df[df['moving_status'] == 'Parked']},
        {'label': 'Moving',   'color': _C['move'], 'ls': '-',  'lw': 1.6,
         'filter': lambda df: df[df['moving_status'] == 'Moving']},
    ],
    'territory': [
        {'label': 'All data', 'color': _C['all'],   'ls': '-',  'lw': 2.0,
         'filter': lambda df: df},
        {'label': 'Urban',    'color': _C['urban'],  'ls': '-',  'lw': 1.6,
         'filter': lambda df: df[df['clc_group'] == 'Urban']},
        {'label': 'Rural',    'color': _C['rural'],  'ls': '-',  'lw': 1.6,
         'filter': lambda df: df[df['clc_group'] == 'Rural']},
    ],
    'crossed': [
        {'label': 'All data',       'color': _C['all'], 'ls': '-',  'lw': 2.0,
         'filter': lambda df: df},
        {'label': 'Parked + Urban', 'color': _C['pu'],  'ls': '--', 'lw': 1.4,
         'filter': lambda df: df[(df['moving_status'] == 'Parked') &
                                  (df['clc_group'] == 'Urban')]},
        {'label': 'Moving + Urban', 'color': _C['mu'],  'ls': '--', 'lw': 1.4,
         'filter': lambda df: df[(df['moving_status'] == 'Moving') &
                                  (df['clc_group'] == 'Urban')]},
        {'label': 'Parked + Rural', 'color': _C['pr'],  'ls': ':',  'lw': 1.4,
         'filter': lambda df: df[(df['moving_status'] == 'Parked') &
                                  (df['clc_group'] == 'Rural')]},
        {'label': 'Moving + Rural', 'color': _C['mr'],  'ls': ':',  'lw': 1.4,
         'filter': lambda df: df[(df['moving_status'] == 'Moving') &
                                  (df['clc_group'] == 'Rural')]},
    ],
}

# Flat list for CSV export — all unique conditions across the three groups
_seen: set[str] = set()
ALL_CONDITIONS: list[dict] = []
for grp in GROUPS.values():
    for c in grp:
        if c['label'] not in _seen:
            _seen.add(c['label'])
            ALL_CONDITIONS.append(c)

# ==============================================================================
# METRIC DEFINITIONS
# ==============================================================================

METRICS: list[dict] = [
    {
        'col':        'auto_solar_km',
        'ylabel':     r'Solar Autonomy [km kW$_p^{-1}$ min$^{-1}$]',
        'title':      'Impact of Diffuse Fraction on Solar Autonomy',
        'pct':        False,
        'key':        'autonomy',
    },
    {
        'col':        'loss_irr_Wh',
        'ylabel':     r'Irradiance Losses [Wh m$^{-2}$ min$^{-1}$]',
        'title':      'Impact of Diffuse Fraction on Absolute Irradiance Losses',
        'pct':        False,
        'key':        'loss_abs',
    },
    {
        'col':        'loss_rel_pct',
        'ylabel':     'Irradiance Losses [%]',
        'title':      'Impact of Diffuse Fraction on Relative Irradiance Losses',
        'pct':        True,
        'key':        'loss_rel',
    },
]

# Group display names for panel subtitles
GROUP_TITLES: dict[str, str] = {
    'movement':  'Movement',
    'territory': 'Territory',
    'crossed':   'Movement × Territory',
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2, R=R_EARTH_KM,
                         timestamps=None, max_gap_minutes=MAX_GAP_MINUTES):
    """
    Compute the great-circle distance in metres between consecutive GPS points.

    Uses the Haversine formula — numerically stable for short intra-urban
    distances typical of vehicle trajectory data.

    Parameters
    ----------
    lat1, lon1 : array-like
        Coordinates of the departure point (decimal degrees, WGS-84).
        Typically df[lat_col].values[:-1].
    lat2, lon2 : array-like
        Coordinates of the arrival point (decimal degrees, WGS-84).
        Typically df[lat_col].values[1:].
    R : float
        Earth radius in kilometres (default: 6371.0 — IAU 2012).
    timestamps : array-like of datetime64, optional
        Full timestamp array of length N (i.e. df['datetime'].values).
        When provided, pairs that cross a segment boundary are zeroed:
          - time gap > max_gap_minutes  (overnight stop, sensor restart)
          - calendar-date change        (explicit cross-day guard)
        When None (default), all distances are returned as computed — safe
        for point-to-fixed-reference uses (distance rings, FCUL proximity).
    max_gap_minutes : float
        Maximum allowed gap in minutes between consecutive rows.
        Defaults to MAX_GAP_MINUTES (defined in the CONFIG section).

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
        # gap_s[i+1] = elapsed seconds between row i and row i+1;
        # distances[i] covers that same pair -> compare gap_s[1:]
        is_boundary = (
            (gap_s[1:] > max_gap_minutes * 60)
            | (dates[1:] != dates[:-1])
        )
        distances[is_boundary] = 0.0

    return distances


def bin_statistics(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Compute descriptive statistics of value_col within each k_d bin.

    Statistics returned
    -------------------
    mean   : arithmetic mean  (kept for reference and downstream analysis)
    median : 50th percentile  (central line shown in figures)
    q25    : 25th percentile  (lower edge of IQR band)
    q75    : 75th percentile  (upper edge of IQR band)
    std    : standard deviation (kept for reference / CSV export)
    count  : number of observations in the bin

    Bins with count < MIN_BIN_OBS have mean/median/q25/q75/std set to NaN
    (plotted as hollow markers); count is always preserved for the CSV
    quality flag.

    Parameters
    ----------
    df         : DataFrame with 'k_d_bin' and the target column.
    value_col  : Column to aggregate.

    Returns
    -------
    pd.DataFrame indexed 0 … N_BINS-1 with the columns listed above.
    """
    grp   = df.groupby('k_d_bin')[value_col]
    stats = pd.DataFrame({
        'mean':   grp.mean(),
        'median': grp.median(),
        'q25':    grp.quantile(0.25),
        'q75':    grp.quantile(0.75),
        'std':    grp.std(),
        'count':  grp.count(),
    }).reindex(range(N_BINS))
    stats['count'] = stats['count'].fillna(0).astype(int)
    mask = stats['count'] < MIN_BIN_OBS
    stats.loc[mask, ['mean', 'median', 'q25', 'q75', 'std']] = np.nan
    return stats


def quality_flag(count: int) -> str:
    """Return a data quality label based on observation count."""
    if count < QUALITY_LOW:
        return 'low'
    if count < QUALITY_MODERATE:
        return 'moderate'
    return 'high'


def annotate_regimes(ax: plt.Axes) -> None:
    """
    Annotate sky-condition regimes inside the plot area.

    Two thin vertical dotted lines mark k_d = 0.3 and k_d = 0.7.
    'Clear sky', 'Mixed conditions', and 'Overcast' are placed using a
    blended transform (x in data space, y in axes fraction) so they are
    always at the same relative height inside the Axes regardless of the
    y-axis scale or tight_layout adjustments.

    Parameters
    ----------
    ax : target Axes
    """
    import matplotlib.transforms as mtransforms

    for kd in [0.3, 0.7]:
        ax.axvline(kd, color='#bbbbbb', linewidth=0.8, linestyle=':', zorder=1)

    blended  = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    label_kw = dict(transform=blended, fontsize=7.5, color='#777777',
                    ha='center', va='top', style='italic', zorder=6)
    ax.text(0.15, 0.97, 'Clear sky',        **label_kw)
    ax.text(0.50, 0.97, 'Mixed conditions', **label_kw)
    ax.text(0.85, 0.97, 'Overcast',         **label_kw)


# ==============================================================================
# 1.  LOAD AND POOL DATA FROM ALL SENSORS
# ==============================================================================
print('=' * 70)
print('  Diffuse Fraction Impact Analysis')
print(f'  CAMPAIGN     : {CAMPAIGN}')
print(f'  Data variant : {_VARIANT_TAG}')
print('=' * 70)

all_frames: list[pd.DataFrame] = []

for sensor, csv_path in SENSOR_FILES.items():
    cols = SENSOR_COLS[sensor]

    if not csv_path.exists():
        print(f'  [WARNING] File not found, skipping : {csv_path}')
        continue

    print(f'\nLoading sensor {sensor} : {csv_path.name}')
    df = pd.read_csv(csv_path, low_memory=False)
    print(f'  → {len(df):,} rows loaded')

    rename_map = {
        cols['date']: 'datetime',
        cols['rad']:  'rad_calib',
        cols['ghi']:  'ghi',
        cols['dhi']:  'dhi',
        cols['elev']: 'sun_elev',
        cols['land']: 'land_use_clcv',
        cols['lat']:  'lat',
        cols['lon']:  'lon',
    }
    df = df.rename(columns=rename_map)

    needed = ['datetime', 'rad_calib', 'ghi', 'dhi', 'sun_elev',
              'land_use_clcv', 'lat', 'lon']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f'  [WARNING] Missing columns {missing}, skipping {sensor}')
        continue
    df = df[needed].copy()

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df['datetime'] = df['datetime'].dt.tz_convert(LOCAL_TZ)
    df = df.dropna(subset=['datetime'])

    for col in ['rad_calib', 'ghi', 'dhi', 'sun_elev', 'land_use_clcv', 'lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_before = len(df)
    df = df.dropna(subset=['rad_calib', 'ghi', 'dhi']).reset_index(drop=True)
    print(f'  → {len(df):,} valid rows after filtering  '
          f'({n_before - len(df):,} excluded)')

    df['sensor'] = sensor
    all_frames.append(df)

if not all_frames:
    raise RuntimeError('No valid data found. Check DATA_DIR and file names.')

data = pd.concat(all_frames, ignore_index=True)
print(f'\nTotal valid measurements across all sensors : {len(data):,}')

# ==============================================================================
# 2.  COMPUTE PER-MINUTE QUANTITIES AND DIFFUSE FRACTION
# ==============================================================================
print('\nComputing per-minute quantities...')

data['E_solar_kWh']   = data['rad_calib'] / 1000.0 * DELTA_T_H
data['loss_irr_Wh']   = np.maximum(0.0, data['ghi'] - data['rad_calib']) * DELTA_T_H
data['auto_solar_km'] = data['E_solar_kWh'] / CVE_KWH_PER_KM

total_irr = data['loss_irr_Wh'] + data['E_solar_kWh'] * 1000.0
data['loss_rel_pct'] = np.where(
    total_irr > 0,
    data['loss_irr_Wh'] / total_irr * 100.0,
    np.nan,
)

data['k_d'] = (data['dhi'] / data['ghi']).clip(0.0, 1.0)
data['k_d_bin'] = pd.cut(
    data['k_d'],
    bins=BIN_EDGES,
    labels=range(N_BINS),
    include_lowest=True,
).astype('Int64')

data['clc_group'] = data['land_use_clcv'].map(CLC_URBAN_RURAL).fillna('Unknown')
print(f'  k_d : mean = {data["k_d"].mean():.3f},  std = {data["k_d"].std():.3f}')

# ==============================================================================
# 3.  COMPUTE MOVEMENT STATUS
# ==============================================================================
print('\nClassifying movement status per sensor...')

move_frames: list[pd.DataFrame] = []

for sensor, csv_path in SENSOR_FILES.items():
    cols_s = SENSOR_COLS[sensor]
    if not csv_path.exists():
        continue

    df_s = pd.read_csv(
        csv_path, low_memory=False,
        usecols=[cols_s['date'], cols_s['lat'], cols_s['lon']],
    )
    df_s = df_s.rename(columns={
        cols_s['date']: 'datetime',
        cols_s['lat']:  'lat',
        cols_s['lon']:  'lon',
    })
    df_s['datetime'] = pd.to_datetime(df_s['datetime'], utc=True, errors='coerce')
    df_s['datetime'] = df_s['datetime'].dt.tz_convert(LOCAL_TZ)
    df_s = df_s.dropna(subset=['datetime', 'lat', 'lon'])
    for col in ['lat', 'lon']:
        df_s[col] = pd.to_numeric(df_s[col], errors='coerce')
    df_s = (df_s.dropna(subset=['lat', 'lon'])
                .sort_values('datetime')
                .reset_index(drop=True))

    dist = haversine_vectorized(
        df_s['lat'].values[:-1], df_s['lon'].values[:-1],
        df_s['lat'].values[1:],  df_s['lon'].values[1:],
        timestamps=df_s['datetime'].values,
    )
    df_s = df_s.iloc[:-1].copy()
    df_s['moving_status'] = np.where(dist >= THRESHOLD_M, 'Moving', 'Parked')
    df_s['sensor']        = sensor
    move_frames.append(df_s[['datetime', 'sensor', 'moving_status']])

move_df = pd.concat(move_frames, ignore_index=True)
data    = data.merge(move_df, on=['datetime', 'sensor'], how='left')

print(f'  Parked : {(data["moving_status"] == "Parked").sum():,}  |  '
      f'Moving : {(data["moving_status"] == "Moving").sum():,}')

data_ws = data.dropna(subset=['moving_status']).copy()

# ==============================================================================
# 4.  DATASET OVERVIEW PER CONDITION — printed to console AND saved to CSV
# ==============================================================================
overview_rows: list[dict] = []
print('\n' + '─' * 70)
print(f"  {'Condition':<25} {'N obs':>10}  {'mean k_d':>9}  "
      f"{'mean auto [km/kWp/min]':>24}  {'mean loss [%]':>14}")
print('─' * 70)
for cond in ALL_CONDITIONS:
    sub = cond['filter'](data_ws)
    if len(sub) == 0:
        print(f"  {cond['label']:<25} {'—':>10}")
        overview_rows.append({
            'condition': cond['label'], 'n_obs': 0,
            'mean_k_d': np.nan, 'mean_auto_km_kWp_min': np.nan,
            'mean_loss_rel_pct': np.nan,
        })
        continue
    row = {
        'condition':            cond['label'],
        'n_obs':                len(sub),
        'mean_k_d':             round(sub['k_d'].mean(), 4),
        'mean_auto_km_kWp_min': round(sub['auto_solar_km'].mean(), 6),
        'mean_loss_rel_pct':    round(sub['loss_rel_pct'].mean(), 2),
    }
    overview_rows.append(row)
    print(f"  {row['condition']:<25} {row['n_obs']:>10,}  "
          f"{row['mean_k_d']:>9.3f}  "
          f"{row['mean_auto_km_kWp_min']:>24.6f}  "
          f"{row['mean_loss_rel_pct']:>13.1f}%")
print('─' * 70)

pd.DataFrame(overview_rows).to_csv(OUTPUT_OVERVIEW, index=False)
print(f'  Condition overview saved : {OUTPUT_OVERVIEW.name}')

# ==============================================================================
# 5.  PLOT FUNCTION — single-panel, publication-quality
# ==============================================================================

def make_panel(ax: plt.Axes,
               conditions:     list[dict],
               target_col:     str,
               ylabel:         str,
               base_title:     str,
               panel_title:    str,
               yformat_pct:    bool,
               subtitle_note:  str,
               all_stats_list: list[dict],
               mode:           str = 'median') -> None:
    """
    Draw one publication-quality panel on `ax`.

    Two statistical modes are supported, selected via the `mode` parameter
    (and globally via PLOT_MODES in the configuration section):

    'mean'
        Central line : arithmetic mean per k_d bin.
        Shaded band  : mean ± 1 standard deviation (when SHOW_BAND = True).
        Use case     : consistent with solar radiation literature conventions.
                       Note that ±1σ may exceed physical bounds on skewed or
                       bounded metrics (e.g. relative losses in [0, 100 %]).

    'median'
        Central line : median (50th percentile) per k_d bin.
        Shaded band  : interquartile range Q25–Q75 (when SHOW_BAND = True).
        Use case     : robust to outliers and asymmetric distributions; the
                       band always stays within the actual data range.

    The shaded band can be disabled entirely by setting SHOW_BAND = False in
    the configuration section, yielding clean line-only figures.

    Other design choices
    --------------------
    - Bins with count < MIN_BIN_OBS are shown with hollow markers to flag
      reduced statistical reliability without discarding the data.
    - Regime annotations ('Clear sky', 'Mixed conditions', 'Overcast') are
      placed as italic text at the top of the plot area using a blended
      transform so they never conflict with axis labels.
    - Vertical dotted lines at k_d = 0.3 and k_d = 0.7 mark regime boundaries.
    - A single legend is placed in the upper-right corner.

    Parameters
    ----------
    ax             : target Axes
    conditions     : list of condition dicts for this panel
    target_col     : column in data_ws to aggregate
    ylabel         : y-axis label string
    base_title     : main figure title (metric name, bold, top line)
    panel_title    : grouping subtitle, second line (e.g. 'Movement × Territory')
    yformat_pct    : if True, format y-axis ticks as percentages
    subtitle_note  : small-print note shown as third line (italic)
    all_stats_list : list to which CSV rows are appended (in-place)
    mode           : 'mean' or 'median' — selects the statistical estimator
    """
    if mode not in ('mean', 'median'):
        raise ValueError(f"mode must be 'mean' or 'median', got {mode!r}")

    ycentre_all:  list[np.ndarray] = []
    yupper_bound: list[np.ndarray] = []

    # ── First pass: collect central values and upper band edge for y-range ────
    for cond in conditions:
        sub   = cond['filter'](data_ws)
        stats = bin_statistics(sub, target_col)
        if mode == 'mean':
            yc  = stats['mean'].values.astype(float)
            yhi = stats['mean'].values.astype(float) + stats['std'].values.astype(float)
        else:
            yc  = stats['median'].values.astype(float)
            yhi = stats['q75'].values.astype(float)
        ycentre_all.append(yc)
        yupper_bound.append(yhi)

    # Y-axis ceiling based on the upper band edge (or centre when band is off)
    all_upper = []
    for yc, yhi in zip(ycentre_all, yupper_bound):
        top = yhi if SHOW_BAND else yc
        top = np.where(~np.isnan(top), top, np.where(~np.isnan(yc), yc, np.nan))
        all_upper.extend(top[~np.isnan(top)].tolist())
    ymax_data = max(all_upper) if all_upper else 1.0
    y_top     = ymax_data * 1.22   # 22 % headroom for regime labels and legend

    # ── Second pass: draw ─────────────────────────────────────────────────────
    for cond, ycentre in zip(conditions, ycentre_all):
        sub    = cond['filter'](data_ws)
        stats  = bin_statistics(sub, target_col)
        counts = stats['count'].values.astype(int)
        ymean  = stats['mean'].values.astype(float)
        ystd   = stats['std'].values.astype(float)
        ymed   = stats['median'].values.astype(float)
        yq25   = stats['q25'].values.astype(float)
        yq75   = stats['q75'].values.astype(float)

        valid = ~np.isnan(ycentre)
        x_v   = BIN_CENTRES[valid]
        y_v   = ycentre[valid]

        # ── Shaded band (optional) ────────────────────────────────────────────
        if SHOW_BAND:
            if mode == 'mean':
                std_v  = np.where(~np.isnan(ystd[valid]), ystd[valid], 0.0)
                y_lo   = y_v - std_v
                y_hi   = y_v + std_v
                _alpha = 0.12
            else:
                y_lo   = np.where(~np.isnan(yq25[valid]), yq25[valid], y_v)
                y_hi   = np.where(~np.isnan(yq75[valid]), yq75[valid], y_v)
                _alpha = 0.18
            ax.fill_between(
                x_v, y_lo, y_hi,
                color=cond['color'], alpha=_alpha, linewidth=0, zorder=2,
            )

        # ── Central line ──────────────────────────────────────────────────────
        ax.plot(
            x_v, y_v,
            color=cond['color'], linestyle=cond['ls'],
            linewidth=cond['lw'], zorder=4,
            label=cond['label'],
        )

        # ── Markers on valid bins ─────────────────────────────────────────────
        ax.plot(
            x_v, y_v,
            linestyle='none', marker='o', markersize=5.0,
            color=cond['color'], zorder=5,
        )

        # ── Append CSV rows (all statistics, regardless of mode) ──────────────
        for i in range(N_BINS):
            cnt = int(counts[i])
            all_stats_list.append({
                'condition':    cond['label'],
                'metric':       target_col,
                'bin_centre':   round(BIN_CENTRES[i], 3),
                'bin_low':      round(BIN_EDGES[i],   1),
                'bin_high':     round(BIN_EDGES[i+1], 1),
                'mean':         round(float(ymean[i]), 6) if not np.isnan(ymean[i]) else np.nan,
                'std':          round(float(ystd[i]),  6) if not np.isnan(ystd[i])  else np.nan,
                'median':       round(float(ymed[i]),  6) if not np.isnan(ymed[i])  else np.nan,
                'q25':          round(float(yq25[i]),  6) if not np.isnan(yq25[i])  else np.nan,
                'q75':          round(float(yq75[i]),  6) if not np.isnan(yq75[i])  else np.nan,
                'count':        cnt,
                'data_quality': quality_flag(cnt),
            })

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_xlabel(r'Diffuse Fraction  $k_d$ = DHI / GHI', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    # Title hierarchy (top to bottom, all inside figure space):
    #   Line 1 — main metric title   : bold, via ax.set_title (matplotlib reserves
    #                                   space for it automatically, no overlap risk)
    #   Line 2 — panel/grouping name : normal weight, just above the axes frame
    #   Line 3 — subtitle note       : italic small print, just above the axes frame
    # set_title handles line 1; lines 2–3 use ax.text with a blended transform
    # so their y position is expressed in axes-fraction coordinates and never
    # collides with the axes content.
    ax.set_title(base_title, fontsize=11, fontweight='bold', pad=32)
    ax.text(0.5, 1.08, panel_title,
            transform=ax.transAxes,
            fontsize=9, fontweight='normal', color='#222222',
            ha='center', va='bottom')
    ax.text(0.5, 1.02, subtitle_note,
            transform=ax.transAxes,
            fontsize=7.5, style='italic', color='#555555',
            ha='center', va='bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='both', which='major', linestyle='--', alpha=0.30, zorder=0)
    ax.grid(axis='both', which='minor', linestyle=':', alpha=0.15, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    if yformat_pct:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f'{v:.0f}%')
        )

    # Regime annotations (blended transform, always at axes top)
    annotate_regimes(ax)

    # Legend — upper-right, anchored just below the regime annotation labels
    # (which sit at axes y = 0.97). Using bbox_to_anchor guarantees the legend
    # never overlaps 'Overcast' regardless of content height.
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.0, 0.93),
        bbox_transform=ax.transAxes,
        framealpha=0.92,
        edgecolor='#cccccc',
        fancybox=False,
        handlelength=2.2,
        handleheight=1.0,
        borderpad=0.7,
        labelspacing=0.45,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """
    Save a figure to disk and optionally display it.

    The figure is always saved at FIGURE_DPI for publication quality.
    Whether it is shown interactively depends on the SHOW_FIGURES flag
    set in the configuration section.

    Parameters
    ----------
    fig  : matplotlib Figure — the figure to save.
    path : Path             — destination file path.
    """
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f'  Saved : {path.name}')
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ==============================================================================
# 6.  GENERATE FIGURES  (PLOT_MODES × 3 metrics × 3 groupings)
# ==============================================================================
# One full set of 9 figures per statistical mode of PLOT_MODES; file names
# embed the mode:
#   dri_<metric>_<grouping>_<mode>_<CAMPAIGN>_<variant_tag>.png
# ==============================================================================
print('\nGenerating figures...')

all_stats: list[dict] = []
total_figures: int    = 0

# Band description inserted in the subtitle depends on mode and SHOW_BAND
_BAND_LABEL: dict[str, str] = {
    'mean':   'mean \u00b11\u03c3',   # mean ±1σ  (unicode, no LaTeX)
    'median': 'median + IQR',
}

for plot_mode in PLOT_MODES:

    if SHOW_BAND:
        _note = (
            f'All sensors pooled \u2014 CVE = {CVE_KWH_PER_100KM} kWh/100 km'
            f' \u2014 bin width = {1/N_BINS:.1f} \u2014 {_BAND_LABEL[plot_mode]}'
        )
    else:
        _note = (
            f'All sensors pooled \u2014 CVE = {CVE_KWH_PER_100KM} kWh/100 km'
            f' \u2014 bin width = {1/N_BINS:.1f} \u2014 {plot_mode}, no band'
        )

    print(f'\n  Mode : {plot_mode}')

    for metric in METRICS:
        col        = metric['col']
        ylabel     = metric['ylabel']
        base_title = metric['title']
        yformat    = metric['pct']
        mkey       = metric['key']

        for gkey, conditions in GROUPS.items():
            panel_title = GROUP_TITLES[gkey]

            # Embed mode key before campaign/variant tag so filenames are self-describing
            output_path = _build_output_path(
                OUTPUT_DIR,
                f'dri_{mkey}_{gkey}_{plot_mode}',
                CAMPAIGN, _VARIANT_TAG, 'png',
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            stats_this: list[dict] = []

            make_panel(
                ax             = ax,
                conditions     = conditions,
                target_col     = col,
                ylabel         = ylabel,
                base_title     = base_title,
                panel_title    = panel_title,
                yformat_pct    = yformat,
                subtitle_note  = _note,
                all_stats_list = stats_this,
                mode           = plot_mode,
            )

            # CSV rows are only appended once (first mode) to avoid duplicates,
            # since both modes produce identical statistics (only the rendering
            # differs).  The CSV always contains all five statistics.
            if plot_mode == PLOT_MODES[0]:
                all_stats.extend(stats_this)

            save_figure(fig, output_path)
            total_figures += 1

print(f'\n  {total_figures} figures generated '
      f'({len(PLOT_MODES)} mode(s) × {len(METRICS)} metrics × {len(GROUPS)} groupings).')

# ==============================================================================
# 7.  k_d DISTRIBUTION FIGURES
# ==============================================================================
# Histograms of k_d over all pooled measurements, one step curve per
# condition (All data filled lightly as reference). They show whether the
# campaign sampled a broad range of sky conditions or was dominated by
# clear-sky or overcast episodes — context for the binned statistics.
#
#   A) Absolute counts  — all five conditions on one plot
#   B) Relative density — Movement grouping : All data / Parked / Moving
#   C) Relative density — Territory grouping: All data / Urban  / Rural
#
# "Relative distribution" means each condition is normalised independently
# so that its histogram integrates to 1 (density = count / (n × bin_width)).
# This allows direct shape comparison between conditions of very different
# sizes (e.g. Moving n≈36k vs Parked n≈120k) without one curve dominating.
# ==============================================================================
print('\nGenerating k_d distribution figures...')

KD_HIST_BINS: int = 40   # number of histogram bins across [0, 1]

bin_edges_kd   = np.linspace(0.0, 1.0, KD_HIST_BINS + 1)
bin_width_kd   = bin_edges_kd[1] - bin_edges_kd[0]
bin_centres_kd = 0.5 * (bin_edges_kd[:-1] + bin_edges_kd[1:])


def _plot_kd_dist(conditions: list[dict],
                  relative:   bool,
                  subtitle:   str,
                  out_path:   Path) -> None:
    """
    Draw and save one k_d distribution figure.

    Parameters
    ----------
    conditions : list of dicts with keys label, color, lw, ls, fill, filter
    relative   : if True normalise each histogram to a probability density
                 (area = 1); if False show raw counts
    subtitle   : italic small-print line shown below the main title
    out_path   : destination file path
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for cond in conditions:
        sub    = cond['filter'](data_ws)
        counts, _ = np.histogram(sub['k_d'].dropna(), bins=bin_edges_kd)

        if relative:
            # Normalise: density = count / (n_total × bin_width)
            # so that sum(density × bin_width) = 1
            y_vals = counts / (len(sub) * bin_width_kd)
        else:
            y_vals = counts.astype(float)

        if cond['fill']:
            ax.fill_between(bin_centres_kd, y_vals,
                            color=cond['color'], alpha=0.10, zorder=1)

        ax.step(
            np.append(bin_edges_kd[:-1], bin_edges_kd[-1]),
            np.append(y_vals, y_vals[-1]),
            where='post',
            color=cond['color'], linewidth=cond['lw'], linestyle=cond['ls'],
            label=f"{cond['label']}  (n = {len(sub):,})",
            zorder=3,
        )

    # Regime boundary lines
    for kd_ref in [0.3, 0.7]:
        ax.axvline(kd_ref, color='#bbbbbb', linewidth=0.8,
                   linestyle=':', zorder=2)

    # Regime annotations
    _bl = _mt.blended_transform_factory(ax.transData, ax.transAxes)
    _lkw = dict(transform=_bl, fontsize=7.5, color='#777777',
                ha='center', va='top', style='italic')
    ax.text(0.15, 0.97, 'Clear sky',        **_lkw)
    ax.text(0.50, 0.97, 'Mixed conditions', **_lkw)
    ax.text(0.85, 0.97, 'Overcast',         **_lkw)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Diffuse Fraction  $k_d$ = DHI / GHI', fontsize=10)
    ax.set_ylabel(
        r'Probability density  [bin$^{-1}$]' if relative
        else 'Number of measurements',
        fontsize=10,
    )
    ax.set_title(
        r'Distribution of Diffuse Fraction $k_d$',
        fontsize=11, fontweight='bold', pad=32,
    )
    ax.text(0.5, 1.02, subtitle,
            transform=ax.transAxes,
            fontsize=7.5, style='italic', color='#555555',
            ha='center', va='bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='both', which='major', linestyle='--', alpha=0.30, zorder=0)
    ax.grid(axis='both', which='minor', linestyle=':', alpha=0.15, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.88),
        bbox_transform=ax.transAxes,
        framealpha=0.92, edgecolor='#cccccc',
        fancybox=False, handlelength=2.0,
        fontsize=8.5,
        ncols=2,
    )

    save_figure(fig, out_path)


# ── Condition sets ─────────────────────────────────────────────────────────────
_conds_all_five: list[dict] = [
    {'label': 'All data', 'color': _C['all'],   'lw': 2.2, 'ls': '-',
     'fill': True,  'filter': lambda df: df},
    {'label': 'Parked',   'color': _C['park'],  'lw': 1.6, 'ls': '-',
     'fill': False, 'filter': lambda df: df[df['moving_status'] == 'Parked']},
    {'label': 'Moving',   'color': _C['move'],  'lw': 1.6, 'ls': '-',
     'fill': False, 'filter': lambda df: df[df['moving_status'] == 'Moving']},
    {'label': 'Urban',    'color': _C['urban'], 'lw': 1.4, 'ls': '--',
     'fill': False, 'filter': lambda df: df[df['clc_group'] == 'Urban']},
    {'label': 'Rural',    'color': _C['rural'], 'lw': 1.4, 'ls': '--',
     'fill': False, 'filter': lambda df: df[df['clc_group'] == 'Rural']},
]

_conds_movement: list[dict] = [
    {'label': 'All data', 'color': _C['all'],  'lw': 2.2, 'ls': '-',
     'fill': True,  'filter': lambda df: df},
    {'label': 'Parked',   'color': _C['park'], 'lw': 1.6, 'ls': '-',
     'fill': False, 'filter': lambda df: df[df['moving_status'] == 'Parked']},
    {'label': 'Moving',   'color': _C['move'], 'lw': 1.6, 'ls': '-',
     'fill': False, 'filter': lambda df: df[df['moving_status'] == 'Moving']},
]

_conds_territory: list[dict] = [
    {'label': 'All data', 'color': _C['all'],   'lw': 2.2, 'ls': '-',
     'fill': True,  'filter': lambda df: df},
    {'label': 'Urban',    'color': _C['urban'], 'lw': 1.6, 'ls': '--',
     'fill': False, 'filter': lambda df: df[df['clc_group'] == 'Urban']},
    {'label': 'Rural',    'color': _C['rural'], 'lw': 1.6, 'ls': '--',
     'fill': False, 'filter': lambda df: df[df['clc_group'] == 'Rural']},
]

_sub_abs = (f'All sensors pooled \u2014 {len(data_ws):,} measurements '
            f'\u2014 bin width = {bin_width_kd:.2f} \u2014 absolute counts')
_sub_mov = (f'All sensors pooled \u2014 {len(data_ws):,} measurements '
            f'\u2014 bin width = {bin_width_kd:.2f} \u2014 Movement \u2014 relative density')
_sub_ter = (f'All sensors pooled \u2014 {len(data_ws):,} measurements '
            f'\u2014 bin width = {bin_width_kd:.2f} \u2014 Territory \u2014 relative density')

# A) Absolute counts — all five conditions
_plot_kd_dist(
    conditions = _conds_all_five,
    relative   = False,
    subtitle   = _sub_abs,
    out_path   = _build_output_path(OUTPUT_DIR, 'dri_kd_distribution', CAMPAIGN, _VARIANT_TAG, 'png'),
)

# B) Relative density — Movement grouping
_plot_kd_dist(
    conditions = _conds_movement,
    relative   = True,
    subtitle   = _sub_mov,
    out_path   = _build_output_path(OUTPUT_DIR, 'dri_kd_distribution_movement', CAMPAIGN, _VARIANT_TAG, 'png'),
)

# C) Relative density — Territory grouping
_plot_kd_dist(
    conditions = _conds_territory,
    relative   = True,
    subtitle   = _sub_ter,
    out_path   = _build_output_path(OUTPUT_DIR, 'dri_kd_distribution_territory', CAMPAIGN, _VARIANT_TAG, 'png'),
)

print('  3 k_d distribution figures saved.')


# ==============================================================================
# 8.  CSV EXPORT
# ==============================================================================
# One CSV is produced regardless of how many modes were generated.
# All five statistics (mean, std, median, q25, q75) are always exported.
# ==============================================================================
df_stats = pd.DataFrame(all_stats)

# De-duplicate: same (condition, metric, bin) may appear from multiple groups
df_stats = df_stats.drop_duplicates(
    subset=['condition', 'metric', 'bin_centre']
).sort_values(['metric', 'condition', 'bin_centre']).reset_index(drop=True)

for col in ['mean', 'std', 'median', 'q25', 'q75']:
    df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce').round(6)

df_stats.to_csv(OUTPUT_CSV, index=False)
print(f'\nBinned statistics CSV saved : {OUTPUT_CSV}')
print('\nDone.')