"""
================================================================================
Data Cleaning Pipeline
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Cleans the joined sensor files of either campaign and writes up to four
nested quality levels per sensor. All outputs keep the input column
structure, so downstream scripts switch between them through FILE_SUFFIX
only.

--------------------------------------------------------------------------------
Campaigns (CAMPAIGN)
--------------------------------------------------------------------------------
  'standard' : Solar Jinie vehicle campaign (2021–2023).
               Input  : <sensor_id>_satellite_landuse_v2_cams_cos_clc_clcr_bh_sw.csv
               (raw file after join_cams / join_cos / join_clc / join_clc_raster
               / join_bh / join_sw _standard.py, which only append columns)
  'tuktuk'   : tuk-tuk campaign (2026).
               Input  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm.csv
               (decode_tuktuk.py, tuk-tuk join chain, normalize_tuktuk.py)

Sensors whose input file does not exist are skipped.

--------------------------------------------------------------------------------
Pipeline steps
--------------------------------------------------------------------------------
  Step 0 (optional) : drop rows before START_DATE                 -> '_sd' prefix
  Step 1            : drop invalid / incomplete / sentinel rows   -> _cleaned
  Step 2 (optional) : radiometric and geometric QC                -> _cleaned_hq
  Step 3 (optional) : clear-sky filter, ghi / ghi_clear >= 0.95   -> _cleaned_clearsky
                      (applied to _cleaned and to _cleaned_hq)       _cleaned_hq_clearsky

With ENABLE_START_DATE_FILTER = True, '_sd' precedes every suffix
(_sd_cleaned, _sd_cleaned_hq, ...) and the date-filtered base is also saved
as <stem>_sd.csv. _sd_cleaned is the recommended baseline for the standard
campaign: J30 and J31 stayed on the FCUL rooftop until 2022-06-14
(sensor_deployment_diagnostic.py).

Output file names: <stem><'_sd'><suffix>.csv, written next to the input,
with <stem> = <sensor_id>_satellite_landuse_v2 (standard) or
<sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm (tuk-tuk).

--------------------------------------------------------------------------------
References
--------------------------------------------------------------------------------
Long C.N., Dutton E.G. (2002). BSRN Global Network recommended QC tests, V2.0.
Perez-Astudillo D. et al. (2018). Solar Energy 169, 425–433.
    DOI: 10.1016/j.solener.2018.05.028

Requirements
------------
    pip install numpy pandas

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 8.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = 'tuktuk'   # 'standard' | 'tuktuk'
if CAMPAIGN not in ('standard', 'tuktuk'):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input / output directory (one per campaign) ------------------------------
_DIRS = {
    'standard': STANDARD_DIR,
    'tuktuk':   TUKTUK_DIR,
}
DATA_DIR: Path = _DIRS[CAMPAIGN]

# --- Input file per sensor (missing files are skipped) ------------------------
_SENSORS = {
    'standard': {
        'j30': DATA_DIR / 'j30_satellite_landuse_v2_cams_cos_clc_clcr_bh_sw.csv',
        'j31': DATA_DIR / 'j31_satellite_landuse_v2_cams_cos_clc_clcr_bh_sw.csv',
        'j32': DATA_DIR / 'j32_satellite_landuse_v2_cams_cos_clc_clcr_bh_sw.csv',
        'j33': DATA_DIR / 'j33_satellite_landuse_v2_cams_cos_clc_clcr_bh_sw.csv',
    },
    'tuktuk': {
        'j30': DATA_DIR / 'j30_tuktuk_ghi_cams_cos_clc_bh_sw_norm.csv',
        'j31': DATA_DIR / 'j31_tuktuk_ghi_cams_cos_clc_bh_sw_norm.csv',
        'j32': DATA_DIR / 'j32_tuktuk_ghi_cams_cos_clc_bh_sw_norm.csv',
        'j33': DATA_DIR / 'j33_tuktuk_ghi_cams_cos_clc_bh_sw_norm.csv',
    },
}
SENSORS: dict[str, Path] = _SENSORS[CAMPAIGN]

# --- Land-cover column used by the Step 1 range check -------------------------
# CLC vector join (join_clc_standard.py / join_clc_tuktuk.py), the land-cover
# reference of both campaigns. The raw 'land_use' column is kept but not used.
LAND_USE_COL: str = 'land_use_clcv'

# --- Step 0: start-date filter ------------------------------------------------
# Excludes the pre-deployment phase (sensors on the FCUL rooftop).
#   standard : J30 / J31 on the rooftop until 2022-06-14
#   tuktuk   : first field day 2026-06-29
ENABLE_START_DATE_FILTER: bool = True

_START_DATES = {
    'standard': '2022-06-15',
    'tuktuk':   '2026-06-29',
}
START_DATE: str = _START_DATES[CAMPAIGN]

# --- Step 2: high-quality filters ---------------------------------------------
# 2a  sun_elev > MIN_SUN_ELEV_DEG  (high air mass and cosine error at low sun)
# 2b  ghi > MIN_GHI_W_M2           (poor signal-to-noise near darkness)
# 2c  rad_calib QC, Perez-Astudillo et al. (2018):
#       Check A  physical upper bound (Flag 03)
#       Check B  modified clearness index kt' > 1 (Flag 17)
ENABLE_HQ_FILTER:    bool  = True
MIN_SUN_ELEV_DEG:    float = 10.0    # [°]
MIN_GHI_W_M2:        float = 25.0    # [W/m²]
SOLAR_CONSTANT_W_M2: float = 1361.0  # [W/m²]  WMO value

# --- Step 3: clear-sky filter -------------------------------------------------
# Keeps minutes with ghi_clear > 0 and ghi / ghi_clear >= CLEAR_SKY_FRACTION.
# Heliosat-4 sets ghi = ghi_clear (cloud index = 1) for pixels it classifies
# as cloud-free, so the filter keeps the minutes CAMS declares cloud-free.
# Under clear sky, any deficit of rad_calib below ghi is local shading.
ENABLE_CLEAR_SKY_FILTER: bool  = True
CLEAR_SKY_FRACTION:      float = 0.95

# =============================================================================
# HELPERS
# =============================================================================

def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    """Save df to path and print a one-line confirmation."""
    df.to_csv(path, index=False)
    print(f'    -> Saved ({label}) : {path}  [{len(df):,} rows]')


def _out(sensor_id: str, suffix: str, variant_prefix: str) -> Path:
    """Build the output path for a given sensor, variant prefix, and suffix."""
    if CAMPAIGN == 'standard':
        stem = f'{sensor_id}_satellite_landuse_v2'
    elif CAMPAIGN == 'tuktuk':
        stem = f'{sensor_id}_tuktuk_ghi_cams_cos_clc_bh_sw_norm'
    else:
        raise ValueError(f"Unknown campaign: '{CAMPAIGN}'")
    filename = f'{stem}{variant_prefix}{suffix}.csv'
    return DATA_DIR / filename


# =============================================================================
# STEP 0 — Filter by start date  (optional)
# =============================================================================

def step0_filter_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows recorded before START_DATE (inclusive lower bound).
    Skipped when ENABLE_START_DATE_FILTER is False.
    """
    if not ENABLE_START_DATE_FILTER:
        print('    [Step 0] Date filter DISABLED — full dataset retained')
        return df

    initial_count = len(df)
    date_col   = 'Human readable date (Europe/London)'
    timestamps = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    threshold  = pd.Timestamp(START_DATE, tz='UTC')
    df         = df[timestamps >= threshold].copy()

    removed = initial_count - len(df)
    print(
        f'    [Step 0] {initial_count:>7,} -> {len(df):>7,} rows  '
        f'({removed:>6,} removed — before {START_DATE})'
    )
    return df


# =============================================================================
# STEP 1 — Remove invalid and incomplete rows
# =============================================================================

def step1_remove_invalid_rows(df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
    """
    Remove rows that are invalid, incomplete, or carry sentinel values.

    Rules applied in order
    ----------------------
    1. Rename sensor-specific columns to generic names for uniform processing.
    2. Drop spurious 'Unnamed' index columns introduced by double CSV exports.
    3. Drop rows with missing values in any required column.
    4. Drop rows where rad_raw = 0
    5. Drop rows where rad_calib = 0
    6. Drop rows where rad_calib < −4 [W/m²] (BSRN floor, Long & Dutton 2002).
    7. Drop rows where LAND_USE_COL is outside [1, 44]  (valid CLC range)
    8. Restore original sensor-specific column names for downstream scripts.

    Note: land_use_str (and land_use_str_clcv) are excluded from
    required_columns; they are derived/display fields, not filtering keys.
    """
    initial_count = len(df)
    sid = sensor_id.upper()

    rename_map = {
        f'{sid} lat':       'lat',
        f'{sid} lon':       'lon',
        f'{sid} Rad raw':   'rad_raw',
        f'{sid} Rad calib': 'rad_calib',
    }
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    required_columns = [
        'lat', 'lon',
        'ghi', 'ghi_clear', 'ghi_extra',
        'bhi', 'dhi', 'dni',
        'sun_elev', 'sun_azim',
        LAND_USE_COL,
    ]
    df = df.dropna(subset=required_columns)
    df = df[df['rad_raw']   != 0.0]
    df = df[df['rad_calib'] != 0.0]
    df = df[df['rad_calib'] >= -4.0]
    df = df[(df[LAND_USE_COL] >= 1) & (df[LAND_USE_COL] <= 44)]

    removed = initial_count - len(df)
    print(
        f'    [Step 1] {initial_count:>7,} -> {len(df):>7,} rows  '
        f'({removed:>6,} removed — invalid / incomplete / out-of-range)'
    )
    df = df.rename(columns={v: k for k, v in rename_map.items()})
    return df


# =============================================================================
# STEP 2 — High-quality filters  (optional)
# =============================================================================

def step2_hq_filter(df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
    """
    Apply high-quality radiometric and geometric filters (→ _cleaned_hq).

    Filter 2a — Minimum solar elevation (MIN_SUN_ELEV_DEG, default 10°).
    Filter 2b — Minimum GHI (MIN_GHI_W_M2, default 25 W/m²).
    Filter 2c — QC Check A: physically impossible upper bound on rad_calib.
                  rad_calib > Gsc · sin(sun_elev)^1.2 · 1.50 + 100  [W/m²]
                  Perez-Astudillo et al. (2018), Flag 03.
             — QC Check B: modified clearness index kt_prime > 1.
                  kt_prime = (rad_calib / ghi_extra)
                             / (1.031·exp(−1.4/(0.9+9.4/m)) + 0.1)
                  with m = 1/sin(sun_elev)  (Kasten & Young air mass).
                  Perez-Astudillo et al. (2018), Flag 17.

    References
    ----------
    Perez-Astudillo D. et al. (2018). Solar Energy 169, 425–433.
    DOI: 10.1016/j.solener.2018.05.028
    """
    if not ENABLE_HQ_FILTER:
        print('    [Step 2] HQ filter DISABLED — _cleaned_hq not produced')
        return df

    initial_count = len(df)
    sid = sensor_id.upper()
    print('    [Step 2] Applying high-quality filters ...')

    sun_elev  = pd.to_numeric(df['sun_elev'],         errors='coerce').fillna(0.0)
    ghi       = pd.to_numeric(df['ghi'],              errors='coerce').fillna(0.0)
    rad_calib = pd.to_numeric(df[f'{sid} Rad calib'], errors='coerce').fillna(0.0)
    ghi_extra = pd.to_numeric(df['ghi_extra'],        errors='coerce').fillna(0.0)

    # 2a — solar elevation
    mask_elev = sun_elev > MIN_SUN_ELEV_DEG
    print(f'           2a sun_elev > {MIN_SUN_ELEV_DEG}°    : {int((~mask_elev).sum()):>6,} rows removed')

    # 2b — GHI floor
    mask_ghi = ghi > MIN_GHI_W_M2
    print(f'           2b ghi > {MIN_GHI_W_M2} W/m²  : {int((~mask_ghi).sum()):>6,} rows removed')

    # 2c — Check A: upper bound
    sin_elev    = np.sin(np.radians(sun_elev.clip(lower=0.0)))
    upper_bound = SOLAR_CONSTANT_W_M2 * sin_elev.clip(lower=0.0) ** 1.2 * 1.50 + 100.0
    check_A     = rad_calib > upper_bound
    print(f'           2c Check A (upper bound)   : {int(check_A.sum()):>6,} rows removed')

    # 2c — Check B: kt_prime > 1
    kt_sensor      = rad_calib / ghi_extra.clip(lower=1e-6)
    air_mass       = np.clip(
        np.where(sin_elev > 0.01, 1.0 / sin_elev.clip(lower=0.01), 40.0),
        1.0, 40.0,
    )
    kt_prime_denom = 1.031 * np.exp(-1.4 / (0.9 + 9.4 / air_mass)) + 0.1
    kt_prime       = kt_sensor / np.clip(kt_prime_denom, 1e-6, None)
    check_B        = kt_prime > 1.0
    print(f'           2c Check B (kt_prime > 1)  : {int(check_B.sum()):>6,} rows removed')

    keep    = mask_elev & mask_ghi & (~check_A) & (~check_B)
    df_hq   = df[keep].copy()
    removed = initial_count - len(df_hq)
    print(
        f'    [Step 2] {initial_count:>7,} -> {len(df_hq):>7,} rows  '
        f'({removed:>6,} removed — HQ filters combined, '
        f'{100.0 * removed / max(initial_count, 1):.1f} % of input)'
    )
    return df_hq


# =============================================================================
# STEP 3 — Clear-sky filter  (optional)
# =============================================================================

def step3_filter_clear_sky(
    df: pd.DataFrame,
    source_label: str,
) -> pd.DataFrame:
    """
    Retain only minutes where CAMS indicates near-clear-sky conditions.

    Criterion: ghi_clear > 0  AND  ghi / ghi_clear ≥ CLEAR_SKY_FRACTION

    Called independently on _cleaned and _cleaned_hq to produce
    _cleaned_clearsky and _cleaned_hq_clearsky.

    Parameters
    ----------
    source_label : str
        Short label of the input level, e.g. '_cleaned' or '_cleaned_hq'.
        Used only for console output.
    """
    initial_count = len(df)

    ghi           = pd.to_numeric(df['ghi'],       errors='coerce')
    ghi_clear     = pd.to_numeric(df['ghi_clear'], errors='coerce')
    valid_denom   = ghi_clear > 0.0
    transmittance = ghi / ghi_clear.clip(lower=1e-6)
    mask          = valid_denom & (transmittance >= CLEAR_SKY_FRACTION)

    df_clear = df[mask].copy()
    removed  = initial_count - len(df_clear)
    pct      = 100.0 * len(df_clear) / max(initial_count, 1)
    print(
        f'    [Step 3] {source_label} : '
        f'{initial_count:>7,} -> {len(df_clear):>7,} rows  '
        f'({removed:>6,} removed — transmittance < {CLEAR_SKY_FRACTION:.2f} '
        f'or ghi_clear = 0 ;  {pct:.1f} % retained)'
    )
    return df_clear


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Run the full cleaning pipeline for all sensors defined in SENSORS.
    All outputs share the same column structure and are interchangeable in
    downstream scripts by changing the FILE_SUFFIX constant.
    """
    date_label     = f'ENABLED  (>= {START_DATE})' if ENABLE_START_DATE_FILTER else 'DISABLED'
    hq_label       = (
        f'ENABLED  (sun_elev > {MIN_SUN_ELEV_DEG}°, ghi > {MIN_GHI_W_M2} W/m², QC A/B)'
        if ENABLE_HQ_FILTER else 'DISABLED'
    )
    clearsky_label = (
        f'ENABLED  (ghi / ghi_clear >= {CLEAR_SKY_FRACTION:.2f})'
        if ENABLE_CLEAR_SKY_FILTER else 'DISABLED'
    )

    print()
    print('═' * 70)
    print(f'  VIPV Data Cleaning Pipeline [campaign : {CAMPAIGN}]')
    print(f'  Step 0  date filter      : {date_label}')
    print(f'  Step 2  HQ filter        : {hq_label}')
    print(f'  Step 3  clear-sky filter : {clearsky_label}')
    print('═' * 70)

    # When the date filter is active, '_sd' is injected so all outputs
    # form a distinct family: _sd_cleaned, _sd_cleaned_hq, etc.
    variant_prefix = '_sd' if ENABLE_START_DATE_FILTER else ''

    for sensor_id, input_path in SENSORS.items():
        print(f'\n  ── Sensor : {sensor_id.upper()} ──')

        if not input_path.exists():
            print(f'  [WARNING] File not found, skipping : {input_path}')
            continue

        print(f'  Input  : {input_path}')
        df_raw = pd.read_csv(input_path, low_memory=False)
        print(f'  Loaded : {len(df_raw):>7,} rows x {df_raw.shape[1]} columns')

        # Step 0 — optional date filter; the date-filtered base is saved too.
        df_raw = step0_filter_by_date(df_raw.copy())
        if ENABLE_START_DATE_FILTER:
            _save_csv(df_raw, _out(sensor_id, '', variant_prefix), '_sd base')

        # Step 1 — remove invalid / incomplete rows → _cleaned
        df_cleaned = step1_remove_invalid_rows(df_raw.copy(), sensor_id)
        _save_csv(df_cleaned, _out(sensor_id, '_cleaned', variant_prefix), f'{variant_prefix}_cleaned')

        # Step 2 — HQ filters → _cleaned_hq  (optional)
        df_hq = step2_hq_filter(df_cleaned.copy(), sensor_id)
        if ENABLE_HQ_FILTER:
            _save_csv(df_hq, _out(sensor_id, '_cleaned_hq', variant_prefix), f'{variant_prefix}_cleaned_hq')

        # Step 3 — clear-sky filter  (optional)
        if ENABLE_CLEAR_SKY_FILTER:
            df_cs = step3_filter_clear_sky(df_cleaned.copy(), f'{variant_prefix}_cleaned')
            if not df_cs.empty:
                _save_csv(df_cs, _out(sensor_id, '_cleaned_clearsky', variant_prefix), f'{variant_prefix}_cleaned_clearsky')
            else:
                print(f'    [Step 3] WARNING: no clear-sky rows in {variant_prefix}_cleaned for {sensor_id.upper()} — file not written')

            if ENABLE_HQ_FILTER:
                df_hq_cs = step3_filter_clear_sky(df_hq.copy(), f'{variant_prefix}_cleaned_hq')
                if not df_hq_cs.empty:
                    _save_csv(df_hq_cs, _out(sensor_id, '_cleaned_hq_clearsky', variant_prefix), f'{variant_prefix}_cleaned_hq_clearsky')
                else:
                    print(f'    [Step 3] WARNING: no clear-sky rows in {variant_prefix}_cleaned_hq for {sensor_id.upper()} — file not written')
        else:
            print('    [Step 3] Clear-sky filter DISABLED — no clearsky files produced')

    print()
    print('═' * 70)
    print('  Pipeline complete.  All sensors processed.')
    print('═' * 70)
    print()


if __name__ == '__main__':
    main()
