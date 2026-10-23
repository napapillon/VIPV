"""
================================================================================
FCUL Pyranometer GHI Join — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Appends the ground-measured global horizontal irradiance of the FCUL rooftop
pyranometer to every tuk-tuk measurement, by linear interpolation in time.
The tuk-tuk campaign stays within Lisbon, so the FCUL station is a local
ground reference alongside the CAMS satellite values (join_cams_tuktuk.py).

--------------------------------------------------------------------------------
Column added
--------------------------------------------------------------------------------
    ghi_fcul : FCUL pyranometer GHI interpolated at the measurement time [W/m²]
               NaN outside the time range of the pyranometer files

Negative night-time values (instrument offset) are kept; clean_data.py
applies the threshold. Measurement rows sharing the same timestamp
(retransmitted packets, different db_timestamp) are reduced to the first.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : <sensor_id>_tuktuk<INPUT_SUFFIX>.csv
    output : <sensor_id>_tuktuk<INPUT_SUFFIX>_ghi.csv

Tuk-tuk join chain:
    decode_tuktuk.py         ->  <sensor_id>_tuktuk.csv
    join_ghi_fcul_tuktuk.py  ->  _ghi                          <- this script
    join_cams_tuktuk.py      ->  _ghi_cams
    join_cos23_tuktuk.py     ->  _ghi_cams_cos
    join_clc_tuktuk.py       ->  _ghi_cams_cos_clc
    join_bh_tuktuk.py        ->  _ghi_cams_cos_clc_bh
    join_sw_tuktuk.py        ->  _ghi_cams_cos_clc_bh_sw
    normalize_tuktuk.py      ->  _ghi_cams_cos_clc_bh_sw_norm   ->  clean_data.py

--------------------------------------------------------------------------------
Pyranometer file format
--------------------------------------------------------------------------------
Excel export of the FCUL station (FCUL_DIR in config.py), 1-minute data, two
columns, second row = units (skipped):
    TIMESTAMP, GMT  |  Avg_GHI
    ISO datetime    |  W/m²

Requirements
------------
    pip install pandas numpy openpyxl

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TUKTUK_DIR, FCUL_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = TUKTUK_DIR

# --- Sensors to process (missing files are skipped) ---------------------------
SENSOR_IDS: list[str] = ["j29", "j30", "j31", "j32", "j33"]

# --- Input suffix (pipeline stage of the input files) -------------------------
#   ""         -> <sensor_id>_tuktuk.csv          (decoded)
INPUT_SUFFIX: str = ""

# --- Output suffix ------------------------------------------------------------
OUTPUT_SUFFIX: str = "_ghi"

# --- FCUL pyranometer file(s), in chronological order (concatenated) ----------
SOLAR_FILES: list[str] = [
    str(FCUL_DIR / "solar_data_for_Jinie33_full_mission_tuktuk.xlsx"),
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_solar_data(filepaths: list[str]) -> pd.DataFrame:
    """
    Load and concatenate one or more FCUL pyranometer Excel files.

    Expects two columns per file (units header row skipped):
        TIMESTAMP, GMT  →  parsed as UTC datetime
        Avg_GHI         →  W/m², numeric

    Duplicate timestamps (identical ghi values, known station artifact) are
    dropped, keeping the first occurrence. The result is sorted chronologically.

    Parameters
    ----------
    filepaths : list of str
        Paths to the Excel files, in chronological order.

    Returns
    -------
    pd.DataFrame
        Columns: ghi_fcul, indexed by UTC-aware datetime.
    """
    frames = []
    for fp in filepaths:
        df = pd.read_excel(
            fp,
            skiprows=1,
            names=["timestamp", "ghi_fcul"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["ghi_fcul"]  = pd.to_numeric(df["ghi_fcul"], errors="coerce")
        frames.append(df)
        print(f"    Loaded {fp}: {len(df)} rows")

    solar = pd.concat(frames, ignore_index=True)

    n_before = len(solar)
    solar = solar.drop_duplicates(subset=["timestamp"], keep="first")
    n_dupes = n_before - len(solar)
    if n_dupes > 0:
        print(f"    Removed {n_dupes} duplicate timestamp rows")

    solar = solar.sort_values("timestamp").set_index("timestamp")
    print(f"    Solar data range : {solar.index.min()}  →  {solar.index.max()}")
    print(f"    Valid rows       : {solar['ghi_fcul'].notna().sum()}")
    return solar


def interpolate_ghi(tuktuk_timestamps: pd.DatetimeIndex,
                    solar_df: pd.DataFrame) -> np.ndarray:
    """
    Linearly interpolate ghi_fcul from the pyranometer time series onto
    tuk-tuk measurement timestamps.

    Timestamps outside the solar data range return NaN (no extrapolation).

    Parameters
    ----------
    tuktuk_timestamps : pd.DatetimeIndex
        UTC-aware timestamps of the tuk-tuk measurements.
    solar_df : pd.DataFrame
        Pyranometer data with UTC DatetimeIndex and column 'ghi_fcul'.

    Returns
    -------
    np.ndarray
        Interpolated ghi_fcul values aligned to tuktuk_timestamps.
    """
    all_times       = tuktuk_timestamps.union(solar_df.index)
    solar_reindexed = solar_df["ghi_fcul"].reindex(all_times)
    solar_interp    = solar_reindexed.interpolate(method="time", limit_area="inside")
    return solar_interp.reindex(tuktuk_timestamps).values


def process_sensor(sensor_id: str, solar: pd.DataFrame) -> None:
    """
    Join ghi_fcul onto one sensor's tuk-tuk dataset and write output CSV.

    Parameters
    ----------
    sensor_id : str
        Sensor identifier (e.g. 'j33').
    solar : pd.DataFrame
        Pre-loaded pyranometer data (shared across all sensors).
    """
    input_file  = DATA_DIR / f"{sensor_id}_tuktuk{INPUT_SUFFIX}.csv"
    output_file = DATA_DIR / f"{sensor_id}_tuktuk{INPUT_SUFFIX}{OUTPUT_SUFFIX}.csv"

    print(f"\n{'─' * 60}")
    print(f"  Sensor : {sensor_id.upper()}")
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")
    print(f"{'─' * 60}")

    if not input_file.exists():
        print("  [SKIP] Input file not found.")
        return

    # Load tuk-tuk data
    df = pd.read_csv(input_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Remove retransmissions: packets sent multiple times to the database share
    # the same measurement timestamp but have different db_timestamp values.
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    n_retrans = n_before - len(df)
    if n_retrans > 0:
        print(f"  Removed {n_retrans} retransmitted duplicates")

    print(f"  {len(df)} unique measurements")
    print(f"  Range : {df['timestamp'].min()}  →  {df['timestamp'].max()}")

    # Interpolate
    tuktuk_ts      = pd.DatetimeIndex(df["timestamp"])
    df["ghi_fcul"] = interpolate_ghi(tuktuk_ts, solar)

    n_ok  = df["ghi_fcul"].notna().sum()
    n_nan = df["ghi_fcul"].isna().sum()
    print(f"  ghi_fcul interpolated : {n_ok} values")
    print(f"  NaN (outside range)   : {n_nan} values")
    if n_ok > 0:
        print(f"  ghi_fcul range        : {df['ghi_fcul'].min():.2f} – {df['ghi_fcul'].max():.2f} W/m²")

    df.to_csv(output_file, index=False)
    print(f"  Written → {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("  JOIN GHI FCUL — tuk-tuk campaign")
    print("=" * 60)

    # Load solar data once — shared across all sensors
    print("\nLoading FCUL pyranometer data ...")
    solar = load_solar_data(SOLAR_FILES)

    for sensor_id in SENSOR_IDS:
        process_sensor(sensor_id, solar)

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
