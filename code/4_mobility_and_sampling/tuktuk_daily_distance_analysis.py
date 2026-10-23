"""
================================================================================
Daily Distance Analysis — Tuk-tuk Campaign
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Computes the distance travelled each calendar day by each tuk-tuk sensor
from consecutive GPS positions (Haversine great-circle distance).

Processing rules
----------------
- One row = one GPS measurement (normally one per minute).
- A distance is computed between two consecutive valid GPS points and
  assigned to the calendar day (LOCAL_TZ) of the destination point.
- Excluded transitions: crossing midnight, time gap above MAX_GAP_MINUTES
  (missing data is not counted as travelled distance), invalid or missing
  coordinates, and speed above MAX_PLAUSIBLE_SPEED_KMH (GPS jumps; None
  disables this filter).
- With INCLUDE_ZERO_DISTANCE_DAYS, days without any valid transition
  between the first and last observed dates are kept with 0 km.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : tuk-tuk clean_data.py outputs, selected by TUKTUK_FILE_SUFFIX
             <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             for each sensor of SENSOR_IDS (missing files are skipped)
    output : OUTPUT_BASE_DIR / tuktuk / <suffix> /
             distance_transitions_<SENSOR>_tuktuk_<suffix>.csv   (per sensor)
             daily_distance_<SENSOR>_tuktuk_<suffix>.csv / .png  (per sensor)
             distance_summary_<SENSOR>_tuktuk_<suffix>.csv       (per sensor)
             daily_distance_all_sensors_long_tuktuk_<suffix>.csv
             daily_distance_all_sensors_wide_tuktuk_<suffix>.csv
             distance_summary_all_sensors_tuktuk_<suffix>.csv
             daily_distance_all_sensors_tuktuk_<suffix>.png

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.0
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Project configuration (paths) ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign (this script is dedicated to the tuk-tuk campaign) --------------
CAMPAIGN: str = "tuktuk"

# --- Sensors / vehicles -------------------------------------------------------
SENSOR_IDS: list[str] = ["J29", "J33"]

# --- Input --------------------------------------------------------------------
TUKTUK_DATA_DIR: Path = TUKTUK_DIR
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"

# Latitude / longitude columns per sensor
SENSOR_COLS = {
    "J29": {"lat": "J29 lat", "lon": "J29 lon"},
    "J33": {"lat": "J33 lat", "lon": "J33 lon"},
}

DATE_COL: str = "Human readable date (Europe/London)"
LOCAL_TZ: str = "Europe/Lisbon"

# --- Method -------------------------------------------------------------------
# Mean Earth radius [km], as in vehicle_movement_analysis.py
R_EARTH_KM: float = 6371.0

# A normal transition is about one minute. A maximum of two minutes tolerates
# small timestamp irregularities while rejecting larger missing-data gaps.
MAX_GAP_MINUTES: float = 2.0

# Optional GPS quality control. Set to None to disable.
# Tuk-tuks cannot realistically sustain very high road speeds; 120 km/h is a
# deliberately permissive ceiling intended only to remove obvious GPS jumps.
MAX_PLAUSIBLE_SPEED_KMH: float | None = 120.0

# If True, dates with no valid travelled transition between the first and last
# observed date are included with a distance of zero.
INCLUDE_ZERO_DISTANCE_DAYS: bool = True

# --- Outputs ------------------------------------------------------------------
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "4_mobility_and_sampling" / "tuktuk_daily_distance_analysis"
OUTPUT_DPI: int = 300
SHOW_FIGURES: bool = False

# ==============================================================================
# FILENAME AND OUTPUT RESOLUTION
# ==============================================================================


def build_sensor_files(
    data_dir: Path, suffix: str, sensor_ids: list[str]
) -> dict[str, Path]:
    """Return the expected input CSV path for every tuk-tuk sensor."""
    return {
        sensor_id: data_dir
        / f"{sensor_id.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm_{suffix}.csv"
        for sensor_id in sensor_ids
    }


def build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Keep outputs from different campaigns/variants separate."""
    return base_dir / campaign / variant_tag


def build_output_filename(base_name: str, extension: str = "csv") -> str:
    """Embed campaign and active data variant in an output filename."""
    return f"{base_name}_{CAMPAIGN}_{TUKTUK_FILE_SUFFIX}.{extension}"


SENSOR_FILES: dict[str, Path] = build_sensor_files(
    TUKTUK_DATA_DIR, TUKTUK_FILE_SUFFIX, SENSOR_IDS
)
OUTPUT_DIR: Path = build_output_dir(
    OUTPUT_BASE_DIR, CAMPAIGN, TUKTUK_FILE_SUFFIX
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def show_figure(fig) -> None:
    """Display a figure interactively when SHOW_FIGURES is True."""
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def haversine_vectorized(lat1, lon1, lat2, lon2, radius_km=R_EARTH_KM):
    """Calculate vectorized great-circle distances in metres.

    Parameters may be pandas Series or NumPy arrays, expressed in degrees.
    """
    lat1_rad = np.radians(np.asarray(lat1, dtype=float))
    lon1_rad = np.radians(np.asarray(lon1, dtype=float))
    lat2_rad = np.radians(np.asarray(lat2, dtype=float))
    lon2_rad = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    # Clipping protects arcsin from tiny floating-point excursions above 1.
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * 1000.0 * c


def parse_local_timestamps(values: pd.Series) -> pd.Series:
    """Parse timestamps and normalize timezone-aware values to Lisbon local time.

    Naive timestamps are assumed to already represent local time, matching the
    meaning of the configured source column.
    """
    parsed = pd.to_datetime(values, errors="coerce")

    # A homogeneous timezone-aware Series exposes .dt.tz.
    try:
        if parsed.dt.tz is not None:
            return parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    # Handles mixed-offset strings robustly by parsing through UTC.
    if parsed.dtype == "object":
        parsed_utc = pd.to_datetime(values, errors="coerce", utc=True)
        return parsed_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    return parsed


def validate_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.Series:
    """Return True for finite coordinates inside valid geographic bounds."""
    return (
        df[lat_col].notna()
        & df[lon_col].notna()
        & np.isfinite(df[lat_col])
        & np.isfinite(df[lon_col])
        & df[lat_col].between(-90.0, 90.0)
        & df[lon_col].between(-180.0, 180.0)
    )


def load_sensor_data(
    sensor_name: str, file_path: Path, lat_col: str, lon_col: str
) -> pd.DataFrame:
    """Load, validate, sort, and prepare one tuk-tuk sensor CSV."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found for {sensor_name}: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    required = [DATE_COL, lat_col, lon_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"{sensor_name}: missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    data = df[required].copy()
    data.rename(columns={lat_col: "latitude", lon_col: "longitude"}, inplace=True)
    data["timestamp"] = parse_local_timestamps(data[DATE_COL])
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")

    valid_coord = validate_coordinates(data, "latitude", "longitude")
    data = data.loc[data["timestamp"].notna() & valid_coord].copy()

    # Stable sorting preserves original order for duplicate timestamps.
    data.sort_values("timestamp", kind="mergesort", inplace=True)
    data.drop_duplicates(subset="timestamp", keep="first", inplace=True)
    data.reset_index(drop=True, inplace=True)
    data["sensor_id"] = sensor_name
    data["date"] = data["timestamp"].dt.normalize()

    if data.empty:
        raise ValueError(f"{sensor_name}: no valid timestamp/GPS row remains.")

    return data


def compute_transitions(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate and validate every consecutive GPS transition."""
    transitions = data.copy()
    transitions["previous_timestamp"] = transitions["timestamp"].shift(1)
    transitions["previous_latitude"] = transitions["latitude"].shift(1)
    transitions["previous_longitude"] = transitions["longitude"].shift(1)
    transitions["gap_minutes"] = (
        transitions["timestamp"] - transitions["previous_timestamp"]
    ).dt.total_seconds() / 60.0

    transitions["distance_m_raw"] = haversine_vectorized(
        transitions["previous_latitude"],
        transitions["previous_longitude"],
        transitions["latitude"],
        transitions["longitude"],
    )

    positive_gap = transitions["gap_minutes"] > 0
    same_day = (
        transitions["timestamp"].dt.normalize()
        == transitions["previous_timestamp"].dt.normalize()
    )
    acceptable_gap = transitions["gap_minutes"] <= MAX_GAP_MINUTES

    transitions["speed_kmh"] = np.where(
        positive_gap,
        (transitions["distance_m_raw"] / 1000.0)
        / (transitions["gap_minutes"] / 60.0),
        np.nan,
    )

    plausible_speed = pd.Series(True, index=transitions.index)
    if MAX_PLAUSIBLE_SPEED_KMH is not None:
        plausible_speed = transitions["speed_kmh"] <= MAX_PLAUSIBLE_SPEED_KMH

    transitions["valid_transition"] = (
        transitions["previous_timestamp"].notna()
        & positive_gap
        & same_day
        & acceptable_gap
        & plausible_speed
    )

    transitions["exclusion_reason"] = ""
    transitions.loc[transitions["previous_timestamp"].isna(), "exclusion_reason"] = (
        "first measurement"
    )
    transitions.loc[
        transitions["previous_timestamp"].notna() & ~positive_gap,
        "exclusion_reason",
    ] = "non-positive time gap"
    transitions.loc[positive_gap & ~same_day, "exclusion_reason"] = "crosses midnight"
    transitions.loc[
        positive_gap & same_day & ~acceptable_gap, "exclusion_reason"
    ] = "time gap too large"
    if MAX_PLAUSIBLE_SPEED_KMH is not None:
        transitions.loc[
            positive_gap & same_day & acceptable_gap & ~plausible_speed,
            "exclusion_reason",
        ] = "implausible speed"

    transitions["distance_m"] = transitions["distance_m_raw"].where(
        transitions["valid_transition"]
    )
    transitions["distance_km"] = transitions["distance_m"] / 1000.0
    return transitions


def aggregate_daily_distance(
    sensor_name: str, data: pd.DataFrame, transitions: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate valid transition distances by local calendar day."""
    daily = (
        transitions.groupby("date", as_index=False)
        .agg(
            distance_m=("distance_m", "sum"),
            valid_transitions=("valid_transition", "sum"),
            measured_rows=("timestamp", "size"),
            first_measurement=("timestamp", "min"),
            last_measurement=("timestamp", "max"),
        )
        .sort_values("date")
    )

    excluded_per_day = (
        transitions.loc[transitions["previous_timestamp"].notna()]
        .assign(excluded=lambda x: ~x["valid_transition"])
        .groupby("date")["excluded"]
        .sum()
    )
    daily["excluded_transitions"] = daily["date"].map(excluded_per_day).fillna(0)

    if INCLUDE_ZERO_DISTANCE_DAYS:
        full_dates = pd.DataFrame(
            {"date": pd.date_range(data["date"].min(), data["date"].max(), freq="D")}
        )
        daily = full_dates.merge(daily, on="date", how="left")
        numeric_cols = [
            "distance_m",
            "valid_transitions",
            "measured_rows",
            "excluded_transitions",
        ]
        daily[numeric_cols] = daily[numeric_cols].fillna(0)

    daily.insert(0, "sensor_id", sensor_name)
    daily["distance_km"] = daily["distance_m"] / 1000.0
    daily["valid_transitions"] = daily["valid_transitions"].astype(int)
    daily["measured_rows"] = daily["measured_rows"].astype(int)
    daily["excluded_transitions"] = daily["excluded_transitions"].astype(int)

    return daily[
        [
            "sensor_id",
            "date",
            "distance_m",
            "distance_km",
            "valid_transitions",
            "excluded_transitions",
            "measured_rows",
            "first_measurement",
            "last_measurement",
        ]
    ]


def build_vehicle_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Build one compact whole-period summary per vehicle."""
    active = daily["distance_km"] > 0
    return pd.DataFrame(
        {
            "sensor_id": [daily["sensor_id"].iloc[0]],
            "first_date": [daily["date"].min()],
            "last_date": [daily["date"].max()],
            "calendar_days": [len(daily)],
            "active_days": [int(active.sum())],
            "total_distance_km": [daily["distance_km"].sum()],
            "mean_km_per_calendar_day": [daily["distance_km"].mean()],
            "mean_km_per_active_day": [
                daily.loc[active, "distance_km"].mean() if active.any() else 0.0
            ],
            "maximum_daily_distance_km": [daily["distance_km"].max()],
            "date_of_maximum": [
                daily.loc[daily["distance_km"].idxmax(), "date"]
            ],
            "valid_transitions": [daily["valid_transitions"].sum()],
            "excluded_transitions": [daily["excluded_transitions"].sum()],
        }
    )


def plot_sensor_daily_distance(sensor_name: str, daily: pd.DataFrame) -> None:
    """Save a daily-distance line chart for one sensor."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily["date"], daily["distance_km"], linewidth=1.3)
    ax.set_title(f"Daily distance travelled — {sensor_name} — tuk-tuk campaign")
    ax.set_xlabel("Date")
    ax.set_ylabel("Distance travelled (km/day)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = OUTPUT_DIR / build_output_filename(
        f"daily_distance_{sensor_name}", "png"
    )
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    show_figure(fig)


def plot_combined_daily_distance(combined_daily: pd.DataFrame) -> None:
    """Save a chart comparing all tuk-tuk sensors."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for sensor_name, sensor_daily in combined_daily.groupby("sensor_id"):
        ax.plot(
            sensor_daily["date"],
            sensor_daily["distance_km"],
            linewidth=1.3,
            label=sensor_name,
        )
    ax.set_title("Daily distance travelled — tuk-tuk campaign")
    ax.set_xlabel("Date")
    ax.set_ylabel("Distance travelled (km/day)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Vehicle / sensor")
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = OUTPUT_DIR / build_output_filename(
        "daily_distance_all_sensors", "png"
    )
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    show_figure(fig)


def print_sensor_summary(sensor_name: str, daily: pd.DataFrame) -> None:
    """Print principal results for one vehicle."""
    active_days = int((daily["distance_km"] > 0).sum())
    total_km = daily["distance_km"].sum()
    print(f"\n{sensor_name}")
    print("-" * 50)
    print(f"Period                    : {daily['date'].min():%Y-%m-%d} to "
          f"{daily['date'].max():%Y-%m-%d}")
    print(f"Calendar days             : {len(daily):,}")
    print(f"Days with distance > 0    : {active_days:,}")
    print(f"Total distance            : {total_km:,.3f} km")
    print(f"Mean per calendar day     : {daily['distance_km'].mean():,.3f} km")
    if active_days:
        print(f"Mean per active day       : "
              f"{daily.loc[daily['distance_km'] > 0, 'distance_km'].mean():,.3f} km")
    print(f"Valid transitions         : {daily['valid_transitions'].sum():,}")
    print(f"Excluded transitions      : {daily['excluded_transitions'].sum():,}")


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    """Compute and export the daily distances of every available sensor."""
    print("=" * 72)
    print("DAILY DISTANCE ANALYSIS — TUK-TUK CAMPAIGN")
    print("=" * 72)
    print(f"Campaign       : {CAMPAIGN}")
    print(f"Data variant   : {TUKTUK_FILE_SUFFIX}")
    print(f"Sensors        : {', '.join(SENSOR_IDS)}")
    print(f"Maximum gap    : {MAX_GAP_MINUTES} minute(s)")
    print(f"Speed ceiling  : {MAX_PLAUSIBLE_SPEED_KMH} km/h")
    print(f"Output folder  : {OUTPUT_DIR}\n")

    all_daily: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    errors: list[str] = []

    for sensor_name, file_path in SENSOR_FILES.items():
        print(f"Processing {sensor_name}: {file_path.name}")
        try:
            cols = SENSOR_COLS[sensor_name]
            data = load_sensor_data(
                sensor_name, file_path, cols["lat"], cols["lon"]
            )
            transitions = compute_transitions(data)
            daily = aggregate_daily_distance(sensor_name, data, transitions)
            summary = build_vehicle_summary(daily)

            transition_columns = [
                "sensor_id",
                "timestamp",
                "previous_timestamp",
                "date",
                "latitude",
                "longitude",
                "previous_latitude",
                "previous_longitude",
                "gap_minutes",
                "distance_m_raw",
                "speed_kmh",
                "valid_transition",
                "exclusion_reason",
                "distance_m",
                "distance_km",
            ]
            transitions[transition_columns].to_csv(
                OUTPUT_DIR
                / build_output_filename(f"distance_transitions_{sensor_name}"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )
            daily.to_csv(
                OUTPUT_DIR / build_output_filename(f"daily_distance_{sensor_name}"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )
            summary.to_csv(
                OUTPUT_DIR / build_output_filename(f"distance_summary_{sensor_name}"),
                index=False,
                date_format="%Y-%m-%d",
            )

            plot_sensor_daily_distance(sensor_name, daily)
            print_sensor_summary(sensor_name, daily)
            all_daily.append(daily)
            all_summaries.append(summary)

        except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as exc:
            message = f"{sensor_name}: {exc}"
            errors.append(message)
            print(f"WARNING — {message}")

    if not all_daily:
        details = "\n".join(errors) if errors else "No sensor was processed."
        raise RuntimeError(f"No output could be produced.\n{details}")

    combined_daily = pd.concat(all_daily, ignore_index=True)
    combined_summary = pd.concat(all_summaries, ignore_index=True)

    combined_daily.to_csv(
        OUTPUT_DIR / build_output_filename("daily_distance_all_sensors_long"),
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    combined_summary.to_csv(
        OUTPUT_DIR / build_output_filename("distance_summary_all_sensors"),
        index=False,
        date_format="%Y-%m-%d",
    )

    wide = combined_daily.pivot(
        index="date", columns="sensor_id", values="distance_km"
    ).sort_index()
    wide.columns = [f"{sensor_id}_distance_km" for sensor_id in wide.columns]
    wide.reset_index().to_csv(
        OUTPUT_DIR / build_output_filename("daily_distance_all_sensors_wide"),
        index=False,
        date_format="%Y-%m-%d",
    )

    plot_combined_daily_distance(combined_daily)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Successfully processed: {len(all_daily)} sensor(s)")
    print(f"Output folder        : {OUTPUT_DIR}")
    if errors:
        print("Warnings:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
