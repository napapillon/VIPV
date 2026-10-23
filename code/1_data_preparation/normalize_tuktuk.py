"""
================================================================================
Column Normalisation — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Renames the generic columns of a joined tuk-tuk file to the sensor-prefixed
format of the standard Solar Jinie campaign (J30–J33), so that clean_data.py
and all downstream analysis scripts read both campaigns the same way. One
sensor per run (SENSOR_ID).

--------------------------------------------------------------------------------
Column mapping
--------------------------------------------------------------------------------
    timestamp   ->  'Human readable date (Europe/London)'
    lat         ->  '<SENSOR_ID> lat'
    lon         ->  '<SENSOR_ID> lon'
    A0          ->  '<SENSOR_ID> Rad raw'
    rad_calib   ->  '<SENSOR_ID> Rad calib'

All other columns are kept unchanged; spurious 'Unnamed' index columns are
dropped. Timestamps are written in UTC as '2026-07-01 08:00:00+00:00', which
the pipeline parses with pd.to_datetime(col, utc=True). Unparseable
timestamps become empty (NaT) and are reported.

SENSOR_ID accepts any 'J<digits>' identifier (J29, J30, ..., J33).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : INPUT_FILE, e.g. <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw.csv
    output : <input stem>_norm.csv, written next to the input

Tuk-tuk join chain:
    decode_tuktuk.py          ->  <sensor_id>_tuktuk.csv
    join_ghi_fcul_tuktuk.py   ->  _ghi
    join_cams_tuktuk.py       ->  _ghi_cams
    join_cos23_tuktuk.py      ->  _ghi_cams_cos
    join_clc_tuktuk.py        ->  _ghi_cams_cos_clc
    join_bh_tuktuk.py         ->  _ghi_cams_cos_clc_bh
    join_sw_tuktuk.py         ->  _ghi_cams_cos_clc_bh_sw
    normalize_tuktuk.py       ->  _ghi_cams_cos_clc_bh_sw_norm   <- this script
                                                                ->  clean_data.py

Requirements
------------
    pip install pandas

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import re
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input file (output of join_sw_tuktuk.py) ---------------------------------
INPUT_FILE: Path = TUKTUK_DIR / "j33_tuktuk_ghi_cams_cos_clc_bh_sw.csv"

# --- Sensor identifier of INPUT_FILE: 'J29', 'J30', 'J31', 'J32', 'J33', ... ---
SENSOR_ID: str = "J33"

# --- Source column names (decode_tuktuk.py format) ----------------------------
COL_TIMESTAMP: str = "timestamp"
COL_LAT:       str = "lat"
COL_LON:       str = "lon"
COL_A0:        str = "A0"
COL_RAD_CALIB: str = "rad_calib"

# --- Pipeline datetime column name (shared by all pipeline scripts) -----------
PIPELINE_DATE_COL: str = "Human readable date (Europe/London)"


# ==============================================================================
# HELPERS
# ==============================================================================

def _validate_sensor_id(sensor_id: str) -> str:
    """
    Validate and normalise the sensor identifier.

    Parameters
    ----------
    sensor_id : str
        Raw value from the configuration, e.g. 'j33' or 'J33'.

    Returns
    -------
    str
        Uppercase identifier, e.g. 'J33'.

    Raises
    ------
    ValueError
        If the identifier does not match the pattern J<digits>.
    """
    sid = sensor_id.strip().upper()
    if not re.fullmatch(r"J\d{2,}", sid):
        raise ValueError(
            f"Invalid SENSOR_ID '{sensor_id}'. "
            "Expected format: 'J29', 'J30', 'J31', 'J32', 'J33', etc."
        )
    return sid


def _build_rename_map(sid: str) -> dict[str, str]:
    """
    Build the source -> pipeline column mapping for one sensor.

    Parameters
    ----------
    sid : str
        Uppercase sensor identifier, e.g. 'J33'.

    Returns
    -------
    dict[str, str]
        Source column name -> pipeline column name.
    """
    return {
        COL_TIMESTAMP: PIPELINE_DATE_COL,
        COL_LAT:       f"{sid} lat",
        COL_LON:       f"{sid} lon",
        COL_A0:        f"{sid} Rad raw",
        COL_RAD_CALIB: f"{sid} Rad calib",
    }


def _build_output_path(input_file: Path) -> Path:
    """
    Build the output path by appending '_norm' to the input stem.

    Example
    -------
    j33_tuktuk_ghi_cams_cos_clc_bh_sw.csv
        -> j33_tuktuk_ghi_cams_cos_clc_bh_sw_norm.csv

    Parameters
    ----------
    input_file : Path
        Path to the input CSV file.

    Returns
    -------
    Path
        Output path in the same directory as the input file.
    """
    return input_file.with_stem(input_file.stem + "_norm")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the tuk-tuk CSV, rename its columns to the pipeline format, and export."""
    print("=" * 70)
    print("  COLUMN NORMALISATION — tuk-tuk campaign")
    print("=" * 70)

    # ── Validate sensor ID ────────────────────────────────────────────────────
    try:
        sid = _validate_sensor_id(SENSOR_ID)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    output_path = _build_output_path(INPUT_FILE)

    print(f"\n  Sensor      : {sid}")
    print(f"  Input       : {INPUT_FILE}")
    print(f"  Output      : {output_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    if not INPUT_FILE.is_file():
        print(f"\n[ERROR] Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print("\nLoading ...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    print(f"  {len(df):,} rows  x  {len(df.columns)} columns")

    # ── Check required source columns ─────────────────────────────────────────
    rename_map = _build_rename_map(sid)
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        print(
            f"\n[ERROR] Missing required columns: {missing}\n"
            "Check that decode_tuktuk.py and all join scripts have been run."
        )
        sys.exit(1)

    # ── Format timestamp (UTC, explicit offset) ───────────────────────────────
    ts = pd.to_datetime(df[COL_TIMESTAMP], utc=True, errors="coerce")
    n_bad = ts.isna().sum()
    if n_bad:
        print(f"  [WARNING] {n_bad:,} timestamp(s) could not be parsed -> NaT")
    df[COL_TIMESTAMP] = ts.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

    # ── Rename ────────────────────────────────────────────────────────────────
    print("\nRenaming columns:")
    for src, tgt in rename_map.items():
        print(f"  '{src}'  ->  '{tgt}'")
    df = df.rename(columns=rename_map)

    # ── Write ─────────────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"  Written : {output_path}  [{len(df):,} rows]")
    print(f"  Columns : {df.columns.tolist()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
