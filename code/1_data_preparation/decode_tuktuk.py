"""
================================================================================
Raw Export Decoder — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Decodes the raw Solar Jinie database exports of the tuk-tuk campaign into one
structured CSV per sensor. Sensors are processed independently; measurements
are never mixed across devices.

--------------------------------------------------------------------------------
Raw format
--------------------------------------------------------------------------------
Each row of a raw export holds a JSON payload ('data' field) with
delta-encoded arrays of usually 5 measurements. The first element is an
absolute value; the following ones are offsets from that first value, not
cumulative:

    val[0]   = absolute measurement
    val[i>0] = val[0] + delta[i]

--------------------------------------------------------------------------------
Columns written
--------------------------------------------------------------------------------
    timestamp     : UNIX time 't' converted to UTC datetime
    lat, lon      : GPS position [decimal degrees], rounded to 6 decimals
    A0            : raw irradiance sensor output [W/m²]
    rad_calib     : calibrated irradiance, slope x A0 + intercept [W/m²]
    voltage_V     : battery voltage, 'v' / 1000 [V]
    temp_C        : temperature, 'temp' / 100 [°C]
    s_raw, A1_raw : undocumented fields 's' and 'A1', kept as-is
    device        : 'Jinie ID' of the raw export (e.g. 'Solar-Jinie33')
    db_timestamp  : 'published_at', database storage time (UTC)

Rows are sorted by timestamp. Dropped: measurements without GPS or timestamp,
measurements outside [CAMPAIGN_START, CAMPAIGN_END] (corrupt packets), and
packets whose JSON cannot be parsed or has no lat/lon.

--------------------------------------------------------------------------------
Calibration
--------------------------------------------------------------------------------
rad_calib = slope x A0 + intercept, with one coefficient pair per sensor
(CALIB). The coefficients are to be updated after the post-campaign
calibration session; until then slope = 1 and intercept = 0 (rad_calib = A0).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : INPUT_FILES[<sensor_id>], raw export in DATA_DIR
    output : <sensor_id>_tuktuk.csv, in DATA_DIR

Tuk-tuk join chain:
    decode_tuktuk.py          ->  <sensor_id>_tuktuk.csv        <- this script
    join_ghi_fcul_tuktuk.py   ->  _ghi
    join_cams_tuktuk.py       ->  _ghi_cams
    join_cos23_tuktuk.py      ->  _ghi_cams_cos
    join_clc_tuktuk.py        ->  _ghi_cams_cos_clc
    join_bh_tuktuk.py         ->  _ghi_cams_cos_clc_bh
    join_sw_tuktuk.py         ->  _ghi_cams_cos_clc_bh_sw
    normalize_tuktuk.py       ->  _ghi_cams_cos_clc_bh_sw_norm   ->  clean_data.py

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
import datetime
import json
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

# --- Input / output directory -------------------------------------------------
DATA_DIR: Path = TUKTUK_DIR

# --- Raw export per sensor (missing files are skipped) ------------------------
INPUT_FILES: dict[str, str] = {
    "j29": "Jinie29_raw_tuktuk.csv",
    "j30": "Jinie30_raw_tuktuk.csv",
    "j31": "Jinie31_raw_tuktuk.csv",
    "j32": "Jinie32_raw_tuktuk.csv",
    "j33": "Jinie33_full_mission_tuktuk.csv",
}

# --- Output suffix: <sensor_id><OUTPUT_SUFFIX>.csv ----------------------------
OUTPUT_SUFFIX: str = "_tuktuk"

# --- Plausible timestamp window (UTC); measurements outside are dropped -------
CAMPAIGN_START = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
CAMPAIGN_END   = datetime.datetime(2027, 1, 1, tzinfo=datetime.timezone.utc)

# --- Calibration per sensor: rad_calib = slope x A0 + intercept ---------------
# Placeholder values (rad_calib = A0) until the post-campaign calibration.
CALIB: dict[str, dict[str, float]] = {
    "j29": {"slope": 1.0, "intercept": 0.0},
    "j30": {"slope": 1.0, "intercept": 0.0},
    "j31": {"slope": 1.0, "intercept": 0.0},
    "j32": {"slope": 1.0, "intercept": 0.0},
    "j33": {"slope": 1.0, "intercept": 0.0},
}


# ==============================================================================
# DECODING
# ==============================================================================

def decode_offsets(base_value, deltas):
    """
    Reconstruct absolute values from a delta-encoded array.

    The first element is the absolute base value; each following element is
    an offset from the base (not cumulative):
        result[0] = base_value
        result[i] = base_value + deltas[i]   for i >= 1

    Parameters
    ----------
    base_value : float | int | None
        Absolute value at position 0.
    deltas : list
        Full array as stored in the JSON (first element = base, rest = offsets).

    Returns
    -------
    list of float | None
        Reconstructed absolute values; None where an input is None.
    """
    result = []
    for i, d in enumerate(deltas):
        if i == 0:
            result.append(base_value)
        else:
            if base_value is None or d is None:
                result.append(None)
            else:
                result.append(base_value + d)
    return result


def decode_packet(json_str, device_id, db_timestamp_str, slope, intercept):
    """
    Decode one raw JSON payload into a list of measurement dicts.

    Parameters
    ----------
    json_str : str
        'data' field of the raw CSV row.
    device_id : str
        'Jinie ID' field (e.g. 'Solar-Jinie33').
    db_timestamp_str : str
        'published_at' field (database storage time, ISO 8601 UTC).
    slope : float
        Calibration slope (rad_calib = slope x A0 + intercept).
    intercept : float
        Calibration intercept [W/m²].

    Returns
    -------
    list of dict
        One dict per valid measurement of the packet; empty if the packet
        cannot be decoded.
    """
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []

    # Packets without lat/lon keys cannot be located
    if "lat" not in data or "lon" not in data:
        return []

    n = len(data["t"])

    timestamps = decode_offsets(data["t"][0],    data["t"])
    lats       = decode_offsets(data["lat"][0],  data["lat"])
    lons       = decode_offsets(data["lon"][0],  data["lon"])
    a0_vals    = decode_offsets(data["A0"][0],   data["A0"])
    v_vals     = decode_offsets(data["v"][0],    data["v"])
    temp_vals  = decode_offsets(data["temp"][0], data["temp"])
    s_vals     = decode_offsets(data["s"][0],    data["s"])
    a1_vals    = decode_offsets(data["A1"][0],   data["A1"])

    db_ts = pd.to_datetime(db_timestamp_str, utc=True)

    rows = []
    for i in range(n):
        lat = lats[i]
        lon = lons[i]
        if lat is None or lon is None:
            continue

        unix_ts = timestamps[i]
        if unix_ts is None:
            continue
        utc_dt = datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc)

        # Implausible timestamps (e.g. a corrupt 1999 packet)
        if not (CAMPAIGN_START <= utc_dt <= CAMPAIGN_END):
            continue

        a0 = a0_vals[i]
        rad_calib = round(slope * a0 + intercept, 4) if a0 is not None else None

        rows.append({
            "timestamp"    : utc_dt,
            "lat"          : round(lat, 6),
            "lon"          : round(lon, 6),
            "A0"           : a0,
            "rad_calib"    : rad_calib,
            "voltage_V"    : round(v_vals[i] / 1000, 4) if v_vals[i] is not None else None,
            "temp_C"       : round(temp_vals[i] / 100, 2) if temp_vals[i] is not None else None,
            "s_raw"        : s_vals[i],
            "A1_raw"       : a1_vals[i],
            "device"       : device_id,
            "db_timestamp" : db_ts,
        })

    return rows


# ==============================================================================
# PER-SENSOR PROCESSING
# ==============================================================================

def process_sensor(sensor_id: str, input_path: Path, output_path: Path,
                   slope: float, intercept: float) -> None:
    """
    Decode all packets of one sensor and write the output CSV.

    Parameters
    ----------
    sensor_id : str
        Sensor identifier (e.g. 'j33'), used for logging only.
    input_path : Path
        Raw export of this sensor.
    output_path : Path
        Decoded output CSV.
    slope, intercept : float
        Calibration coefficients of this sensor.
    """
    print(f"\n{'─' * 60}")
    print(f"  Sensor : {sensor_id.upper()}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Calib  : rad_calib = {slope} × A0 + {intercept}")
    print(f"{'─' * 60}")

    if not input_path.exists():
        print("  [SKIP] Input file not found.")
        return

    df_raw = pd.read_csv(input_path)
    print(f"  {len(df_raw)} packets found in raw file")

    all_rows = []
    skipped_packets = 0

    for _, row in df_raw.iterrows():
        measurements = decode_packet(
            json_str=row["data"],
            device_id=row["Jinie ID"],
            db_timestamp_str=row["published_at"],
            slope=slope,
            intercept=intercept,
        )
        if not measurements:
            skipped_packets += 1
        all_rows.extend(measurements)

    if not all_rows:
        print("  [SKIP] No valid measurements decoded.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Packets skipped        : {skipped_packets}  (no lat/lon or corrupt JSON)")
    print(f"  Measurements output    : {len(df)}")
    print(f"  Date range (UTC)       : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    print(f"  Lat range              : {df['lat'].min():.4f} – {df['lat'].max():.4f}")
    print(f"  Lon range              : {df['lon'].min():.4f} – {df['lon'].max():.4f}")
    print(f"  A0 range (W/m²)        : {df['A0'].min()} – {df['A0'].max()}")
    print(f"  rad_calib range (W/m²) : {df['rad_calib'].min():.1f} – {df['rad_calib'].max():.1f}")
    print(f"  Voltage range (V)      : {df['voltage_V'].min():.3f} – {df['voltage_V'].max():.3f}")
    print(f"  Temperature range (°C) : {df['temp_C'].min():.1f} – {df['temp_C'].max():.1f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Written → {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Decode the raw export of every sensor listed in INPUT_FILES."""
    print("=" * 60)
    print("  DECODE TUK-TUK — Solar Jinie raw export decoder")
    print("=" * 60)

    for sensor_id, input_filename in INPUT_FILES.items():
        input_path  = DATA_DIR / input_filename
        output_path = DATA_DIR / f"{sensor_id}{OUTPUT_SUFFIX}.csv"
        calib       = CALIB.get(sensor_id, {"slope": 1.0, "intercept": 0.0})
        process_sensor(
            sensor_id   = sensor_id,
            input_path  = input_path,
            output_path = output_path,
            slope       = calib["slope"],
            intercept   = calib["intercept"],
        )

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
