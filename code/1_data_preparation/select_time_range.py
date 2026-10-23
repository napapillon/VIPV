"""
================================================================================
Temporal Range Selector
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Exports, for each sensor, the rows of a cleaned variant that fall inside a
time window. Works for both campaigns (CAMPAIGN = 'standard' | 'tuktuk').

--------------------------------------------------------------------------------
Selection modes (RANGE_MODE)
--------------------------------------------------------------------------------
  'date'     : [START_DATE, END_DATE], full days, both inclusive
               e.g. '2023-07-01' -> '2023-07-31'
  'datetime' : [START_DATETIME, END_DATETIME], minute precision, inclusive
               e.g. '2023-07-01 08:00' -> '2023-07-01 18:00'
Boundaries are compared in UTC.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <stem>_<INPUT_SUFFIX>.csv
    output : <stem>_<INPUT_SUFFIX>_<label>.csv

<stem> follows CAMPAIGN, as in clean_data.py:
    'standard' : <sensor_id>_satellite_landuse_v2
    'tuktuk'   : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm
Both campaigns share the column names used here ('<SENSOR> lat' / '<SENSOR> lon',
'Human readable date (Europe/London)') after normalize_tuktuk.py.

<label> is CUSTOM_LABEL, or built from the boundaries when CUSTOM_LABEL is
empty ('20220901_20220930', '20230601T0800_20230601T1800'). Outputs are
written next to the inputs; downstream scripts select them with
FILE_SUFFIX = '<INPUT_SUFFIX>_<label>'. Sensors whose input file is not found
are skipped; empty subsets are not written.

Requirements
------------
    pip install pandas

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = 'standard'   # 'standard' | 'tuktuk'
if CAMPAIGN not in ('standard', 'tuktuk'):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input / output directory (one per campaign) ------------------------------
_DIRS = {
    'standard': STANDARD_DIR,
    'tuktuk':   TUKTUK_DIR,
}
DATA_DIR: Path = _DIRS[CAMPAIGN]
OUTPUT_DIR: Path = DATA_DIR

# --- File stem per campaign (stem of the clean_data.py outputs) ---------------
_STEMS = {
    'standard': '{sensor_id}_satellite_landuse_v2',
    'tuktuk':   '{sensor_id}_tuktuk_ghi_cams_cos_clc_bh_sw_norm',
}
FILE_STEM: str = _STEMS[CAMPAIGN]

# --- Input suffix (any variant of clean_data.py / filter_lisbon.py) ----------
#   e.g. 'sd_cleaned', 'sd_cleaned_hq', 'sd_cleaned_lisbon',
#        'sd_cleaned_hq_clearsky_lisbon' (drop 'sd_' without the start-date filter)
INPUT_SUFFIX: str = 'cleaned_lisbon'

# --- Timestamp column ---------------------------------------------------------
DATE_COL: str = 'Human readable date (Europe/London)'

# --- Selection mode: 'date' | 'datetime' --------------------------------------
RANGE_MODE: str = 'date'

# --- 'date' mode: 'YYYY-MM-DD', both inclusive --------------------------------
START_DATE: str = '2022-09-01'
END_DATE  : str = '2022-09-30'

# --- 'datetime' mode: 'YYYY-MM-DD HH:MM', both inclusive ----------------------
START_DATETIME: str = '2023-06-01 08:00'
END_DATETIME  : str = '2023-06-01 18:00'

# --- Output label ('' -> built from the boundaries) ---------------------------
CUSTOM_LABEL: str = 'september2022'

# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(data_dir: Path, suffix: str) -> list[Path]:
    """
    Build the list of input CSV paths for sensors J30–J33.

    The naming convention followed by the VIPV pipeline is:
        <stem>_<suffix>.csv   (FILE_STEM, set by CAMPAIGN)

    Parameters
    ----------
    data_dir : Path
        Directory containing the CSV files.
    suffix : str
        INPUT_SUFFIX value set in the configuration section.

    Returns
    -------
    list[Path]
        One Path per sensor (j30 → j33), whether or not the file exists.
    """
    return [
        data_dir / f'{FILE_STEM.format(sensor_id=sid)}_{suffix}.csv'
        for sid in ['j30', 'j31', 'j32', 'j33']
    ]


def _build_output_path(output_dir: Path, input_path: Path,
                       input_suffix: str, label: str) -> Path:
    """
    Build the output CSV path for one sensor.

    The output suffix appends the time-range label to the input suffix, so
    the result integrates seamlessly into the downstream FILE_SUFFIX
    convention.

    Examples (CAMPAIGN = 'standard')
    --------------------------------
    input_suffix='cleaned',        label='september2022'
        → j31_satellite_landuse_v2_cleaned_september2022.csv

    input_suffix='cleaned_lisbon', label='september2022'
        → j31_satellite_landuse_v2_cleaned_lisbon_september2022.csv

    input_suffix='cleaned_hq',     label='20230601_20230630'
        → j31_satellite_landuse_v2_cleaned_hq_20230601_20230630.csv

    Parameters
    ----------
    output_dir : Path
        Directory to write the output file into.
    input_path : Path
        Path of the input CSV file (used to extract the sensor stem).
    input_suffix : str
        INPUT_SUFFIX value set in the configuration section.
    label : str
        Time-range label (CUSTOM_LABEL or auto-generated).

    Returns
    -------
    Path
        Full output file path.
    """
    # Derive the base stem by stripping the input suffix from the filename stem.
    # e.g. 'j31_satellite_landuse_v2_cleaned_lisbon' → 'j31_satellite_landuse_v2'
    stem      = input_path.stem                          # full stem without .csv
    base_stem = stem[: len(stem) - len(f'_{input_suffix}')]
    out_name  = f'{base_stem}_{input_suffix}_{label}.csv'
    return output_dir / out_name


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _build_auto_label(mode: str,
                      start_date: str, end_date: str,
                      start_dt: str, end_dt: str) -> str:
    """
    Build a compact label from the date/datetime boundaries.

    Examples
    --------
    mode='date'     → '20230601_20230630'
    mode='datetime' → '20230601T0800_20230601T1800'

    Parameters
    ----------
    mode : str
        'date' or 'datetime'.
    start_date, end_date : str
        ISO date strings used in 'date' mode.
    start_dt, end_dt : str
        ISO datetime strings used in 'datetime' mode.

    Returns
    -------
    str
        Compact label string.
    """
    if mode == 'date':
        s = start_date.replace('-', '')
        e = end_date.replace('-', '')
        return f'{s}_{e}'
    else:   # 'datetime'
        s = start_dt.replace('-', '').replace(' ', 'T').replace(':', '')
        e = end_dt.replace('-', '').replace(' ', 'T').replace(':', '')
        return f'{s}_{e}'


def _parse_datetime_column(series: pd.Series, date_col: str) -> pd.Series:
    """
    Parse the datetime column to timezone-naive UTC timestamps.

    The column holds ISO strings with a UTC offset (e.g.
    '2023-06-01 10:30:00+00:00'). They are converted to UTC and made
    timezone-naive so that they compare directly with the plain boundary
    strings of the configuration.

    Parameters
    ----------
    series : pd.Series
        Raw string series read from the CSV.
    date_col : str
        Column name, used only in the warning message.

    Returns
    -------
    pd.Series
        Parsed, timezone-naive datetime series (NaT for unparseable rows).
    """
    parsed = pd.to_datetime(series, utc=True, errors='coerce')
    parsed = parsed.dt.tz_convert(None)    # convert to UTC then drop tz info → naive UTC
    n_nat  = parsed.isna().sum()
    if n_nat > 0:
        print(f'    [WARNING] {n_nat:,} rows could not be parsed in column '
              f"'{date_col}' and will be excluded from the subset.")
    return parsed


def filter_by_date_range(df: pd.DataFrame, dt_series: pd.Series,
                         start_date: str, end_date: str) -> pd.DataFrame:
    """
    Keep rows whose date falls within [start_date, end_date] (both inclusive).

    Full-day boundaries are applied: the filter keeps everything from
    00:00:00 on start_date to 23:59:59 on end_date.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dt_series : pd.Series
        Parsed datetime series (timezone-naive) aligned with df.
    start_date, end_date : str
        ISO 8601 date strings ('YYYY-MM-DD'), both inclusive.

    Returns
    -------
    pd.DataFrame
        Filtered subset.
    """
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date) + pd.Timedelta('1D') - pd.Timedelta('1s')
    mask    = (dt_series >= t_start) & (dt_series <= t_end)
    return df.loc[mask].copy()


def filter_by_datetime_range(df: pd.DataFrame, dt_series: pd.Series,
                             start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Keep rows whose timestamp falls within [start_dt, end_dt] (both inclusive).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dt_series : pd.Series
        Parsed datetime series (timezone-naive) aligned with df.
    start_dt, end_dt : str
        ISO 8601 datetime strings ('YYYY-MM-DD HH:MM'), both inclusive.

    Returns
    -------
    pd.DataFrame
        Filtered subset.
    """
    t_start = pd.Timestamp(start_dt)
    t_end   = pd.Timestamp(end_dt)
    mask    = (dt_series >= t_start) & (dt_series <= t_end)
    return df.loc[mask].copy()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the temporal range filter for all four sensors."""

    # ── Validate mode ─────────────────────────────────────────────────────────
    if RANGE_MODE not in ('date', 'datetime'):
        raise ValueError(
            f"RANGE_MODE must be 'date' or 'datetime', got '{RANGE_MODE}'."
        )

    # ── Build time-range label ────────────────────────────────────────────────
    label = CUSTOM_LABEL.strip() if CUSTOM_LABEL.strip() else _build_auto_label(
        RANGE_MODE, START_DATE, END_DATE, START_DATETIME, END_DATETIME
    )

    # ── Resolve input file list from suffix ───────────────────────────────────
    sensor_files  = _build_sensor_files(DATA_DIR, INPUT_SUFFIX)
    output_suffix = f'{INPUT_SUFFIX}_{label}'

    # ── Print configuration header ────────────────────────────────────────────
    print('=' * 72)
    print('  Temporal Range Selector  —  VIPV / FCUL  —  v2.0')
    print('=' * 72)
    print(f'  Campaign       : {CAMPAIGN}')
    print(f'  Input suffix   : {INPUT_SUFFIX}')
    print(f'  Output suffix  : {output_suffix}')
    print(f'  Mode           : {RANGE_MODE}')
    if RANGE_MODE == 'date':
        print(f'  Start date     : {START_DATE}  (inclusive, 00:00:00)')
        print(f'  End date       : {END_DATE}    (inclusive, 23:59:59)')
    else:
        print(f'  Start datetime : {START_DATETIME}  (inclusive)')
        print(f'  End datetime   : {END_DATETIME}    (inclusive)')
    print(f'  Label          : {label}')
    print(f'  Output dir     : {OUTPUT_DIR}')
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each sensor ───────────────────────────────────────────────────
    for csv_path in sensor_files:
        print(f'Processing : {csv_path.name}')

        # ── File existence check ───────────────────────────────────────────────
        if not csv_path.exists():
            print('  [WARNING] File not found — skipping.\n')
            continue

        # ── Load ───────────────────────────────────────────────────────────────
        df      = pd.read_csv(csv_path, low_memory=False)
        n_total = len(df)
        print(f'  Rows loaded  : {n_total:,}')

        # ── Validate datetime column ───────────────────────────────────────────
        if DATE_COL not in df.columns:
            print(f"  [WARNING] Column '{DATE_COL}' not found — skipping.\n")
            continue

        # ── Parse datetime ─────────────────────────────────────────────────────
        dt_series = _parse_datetime_column(df[DATE_COL], DATE_COL)

        # ── Apply temporal filter ──────────────────────────────────────────────
        if RANGE_MODE == 'date':
            subset = filter_by_date_range(df, dt_series, START_DATE, END_DATE)
        else:
            subset = filter_by_datetime_range(
                df, dt_series, START_DATETIME, END_DATETIME
            )

        n_kept    = len(subset)
        n_removed = n_total - n_kept
        pct_kept  = n_kept / n_total * 100 if n_total > 0 else 0.0
        print(f'  Rows kept    : {n_kept:,}  ({pct_kept:.1f}% of input, '
              f'{n_removed:,} removed)')

        if n_kept == 0:
            print('  [WARNING] No rows match the selected range — '
                  'output file not written.\n')
            continue

        # ── Build output path and export ──────────────────────────────────────
        out_path = _build_output_path(OUTPUT_DIR, csv_path, INPUT_SUFFIX, label)
        subset.to_csv(out_path, index=False)
        print(f'  Output file  : {out_path.name}')
        print(f'  Saved to     : {out_path}\n')

    print('=' * 72)
    print('  Done.')
    print()
    print('  Next step — set FILE_SUFFIX in downstream scripts to:')
    print(f"      FILE_SUFFIX = '{output_suffix}'")
    print('=' * 72)


if __name__ == '__main__':
    main()
