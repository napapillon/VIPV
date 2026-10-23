"""
================================================================================
Building Height vs. VIPV Autonomy Losses — Base Relationship, SVF/H-W
Theoretical Overlay, and Confound Cross-Diagnostics
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Correlates building height (urban-canyon proxy, from the building-height
join) with VIPV autonomy losses, then tests whether the pattern observed
across building-height bins is consistent with pure street-canyon GEOMETRY,
or more likely reflects a temporal / weather confound (or a street-width
effect not captured by building height alone).

The per-bin energy-weighted loss is computed by a single function,
compute_bin_stats() (with _pooled_ratio_pct()), used by every figure, so
that the formula cannot drift between sections. RUN_SVF_OVERLAY,
RUN_CONFOUND_DIAGNOSTICS and RUN_SUNELEV_MECHANISM_DIAGNOSTIC switch the
optional sections on and off. The geographic scope is set by the input
suffix (e.g. "sd_cleaned_lisbon" vs "sd_cleaned").

Method
------
Autonomy / loss metrics, energy-weighted aggregation (ratio of sums):
    E_solar_kWh   = rad_calib [W/m²] / 1000 × Δt_h
    loss_irr_kWh  = max(0, GHI − rad_calib) / 1000 × Δt_h
    auto_solar_km = E_solar_kWh  / CVE_kWh_per_km
    loss_auto_km  = loss_irr_kWh / CVE_kWh_per_km
    loss_pct (per bin) = Σ loss_auto_km / (Σ auto_solar_km + Σ loss_auto_km) × 100
Confidence intervals: bootstrap (N_BOOTSTRAP resamples, CI_PCT %).
Parked / Moving: Haversine distance to the next GPS point of the same
sensor, Moving if >= THRESHOLD_M (last row of each sensor dropped).

No building nearby
------------------
A non-trivial fraction of points have NO building within the 50 m buffer —
mostly wide roads, parks and open land, where "building height" is not
applicable (not "zero height"). These rows (n_buildings_nearby == 0) are
EXCLUDED rather than imputed with height = 0, which would conflate "no
building nearby" with "very low building" and bias the correlation. The
excluded fraction is reported in every figure footnote. This is specific
to this analysis, not a general data-quality filter: "no building nearby"
is meaningful information for other analyses, so clean_data.py keeps it.

No other filtering is applied here: land_use_clcv range, sun_elev and
rad_calib are handled upstream by clean_data.py — choose the input suffix
with the filtering level needed (e.g. 'sd_cleaned_hq').

Theoretical model (Fig 7 only)
------------------------------
Brito et al. (2024, "Effect of urban shadowing on the potential of
solar-powered vehicles", Prog. Photovolt. Res. Appl. 32(2):73-83) relate the
height-to-width ratio (H/W) of an infinite-canyon street cross-section to
the loss of ISOTROPIC DIFFUSE irradiance through the sky view factor (SVF):

    Loss_diffuse [%] = (1 - cos^2(mean_obstruction_angle)) x 100   (Eq. 2, middle of street)

(diffuse_loss_pct from brito_svf_model). Fig 7 uses a small set of ASSUMED
constant street widths (ASSUMED_STREET_WIDTHS_M), not the real per-point
width (see hw_ratio_real_analysis.py for that), giving a purely geometric,
monotonic-by-construction reference curve. It is a SHAPE / TREND
diagnostic, not a magnitude validation: the theoretical curve is
diffuse-only while the measured loss bundles diffuse and direct (beam)
shading, so the two are drawn on independent (twin) axes.

Figures produced
----------------
  Fig 1  : Scatter of building height vs per-point relative loss, with a
           binned mean trend line (bootstrap 95 % CI band).
  Fig 2  : Energy-weighted relative loss per building-height bin, with
           bootstrap 95 % CI error bars.
  Fig 2B : Fig 2 PLUS (a) a sliding-window trend line (same formula,
           stepped every ROLLING_STEP_M, top axis in metres) showing whether
           the pattern survives outside the fixed 6 m bins, and (b) the
           absolute autonomy (metres lost per measurement on each bar, total
           km intact / lost in a reference box).
  Fig 3  : Histogram of mean_building_height_m with the bin edges, to show
           the sample density behind Fig 1/2.
  Fig 4  : Relative loss by height bin, Urban vs Rural.
  Fig 5  : Relative loss by height bin, Parked vs Moving.
  Fig 6  : Absolute (km lost, mean per point) vs relative (%) loss by height
           bin, twin axis.
  Fig 7  : [RUN_SVF_OVERLAY] Measured energy-weighted loss (bars) vs
           theoretical diffuse SVF loss for the assumed street widths (lines,
           twin axis) — Brito et al. (2024) Eq. 1-3.
  Fig 8  : [RUN_CONFOUND_DIAGNOSTICS] 2x2 panels — measured loss, building
           density proxy, clear-sky index (kc_clear), mean sun elevation, by
           height bin.
  Fig 9  : [RUN_CONFOUND_DIAGNOSTICS] Seasonal composition (% points per
           meteorological season) by height bin.
  Fig 10 : [RUN_SUNELEV_MECHANISM_DIAGNOSTIC] Solar-elevation distribution
           per height bin (density + empirical CDF): is the sun_elev
           difference across bins (Fig 8, panel 4) a hard truncation
           (optical selection) or a smooth shift (sampling confound)?
  Fig 11 : [RUN_SUNELEV_MECHANISM_DIAGNOSTIC] Hour-of-day composition per
           height bin (stacked bars) — the within-day counterpart of Fig 9.
  Fig 12 : [RUN_SUNELEV_MECHANISM_DIAGNOSTIC] Relative loss vs solar-
           elevation band, one curve per height bin: overlapping curves mean
           height carries no information once sun_elev is controlled (pure
           confound); taller-building bins above shorter ones at the SAME
           sun_elev indicate a genuine geometric signal.

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', '<SENSOR> Rad calib', 'ghi', 'ghi_clear'
             (optional), 'sun_elev', 'land_use_clcv',
             'mean_building_height_m', 'max_building_height_m',
             'n_buildings_nearby'
    output : OUTPUT_BASE_DIR / <CAMPAIGN> / <variant_tag> /
             bh_fig1_scatter_trend, bh_fig2_bins_bar, bh_fig2b_bins_bar_trend,
             bh_fig3_height_histogram, bh_fig4_urban_rural,
             bh_fig5_parked_moving, bh_fig6_absolute_vs_relative,
             bh_fig7_svf_overlay, bh_fig8_confound_panels,
             bh_fig9_confound_season, bh_fig10_sunelev_distribution,
             bh_fig11_hour_composition, bh_fig12_loss_vs_sunelev
                 (_<CAMPAIGN>_<variant_tag>.png)
             bh_bin_summary, bh_urban_rural_summary, bh_parked_moving_summary,
             bh_confound_diagnostics, bh_season_composition,
             bh_hour_composition, bh_loss_by_height_and_sunelev
                 (_<CAMPAIGN>_<variant_tag>.csv)
             Figures and tables of a disabled optional section are skipped.

Requirements
------------
    pip install pandas numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 2.3
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT, STANDARD_DIR, TUKTUK_DIR  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from clc_mapping import CLC_URBAN_RURAL
from brito_svf_model import diffuse_loss_pct


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Campaign selector --------------------------------------------------------
CAMPAIGN: str = "standard"   # "standard" | "tuktuk"
if CAMPAIGN not in ("standard", "tuktuk"):
    raise ValueError(f"CAMPAIGN must be 'standard' or 'tuktuk', got '{CAMPAIGN}'.")

# --- Input directories (one per campaign) -------------------------------------
DATA_DIR: Path = STANDARD_DIR            # CAMPAIGN = "standard"
TUKTUK_DATA_DIR: Path = TUKTUK_DIR       # CAMPAIGN = "tuktuk"

# --- Input suffix (clean_data.py / filter / time-range variant) ---------------
FILE_SUFFIX: str = "sd_cleaned"          # standard campaign
TUKTUK_FILE_SUFFIX: str = "sd_cleaned"   # tuk-tuk campaign

# Base output directory. A subdirectory named after CAMPAIGN / variant tag is
# created automatically.
OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "building_height_outputs"

# --- Optional sections --------------------------------------------------------
RUN_SVF_OVERLAY: bool = True                    # Fig 7 (theoretical SVF vs measured)
RUN_CONFOUND_DIAGNOSTICS: bool = True           # Fig 8-9 (weather/season proxies)
RUN_SUNELEV_MECHANISM_DIAGNOSTIC: bool = True   # Fig 10-12 (optical selection vs sampling confound)

# --- Physical / vehicle parameters --------------------------------------------
CVE_KWH_PER_100KM: float = 16.6
CVE_KWH_PER_KM: float = CVE_KWH_PER_100KM / 100.0
DELTA_T_H: float = 1.0 / 60.0

# No filtering on land_use_clcv, sun_elev or rad_calib here: clean_data.py
# applies it upstream (choose the input suffix accordingly).

# --- Movement classification (Parked/Moving, Fig 5) ---------------------------
THRESHOLD_M: float = 30.0     # Haversine distance separating Parked from Moving
R_EARTH_KM: float = 6371.0

# --- Building-height bins (metres) --------------------------------------------
HEIGHT_BIN_EDGES: list[float] = [0, 6, 12, 18, 24, 30, 100]
HEIGHT_BIN_LABELS: list[str] = [
    "0-6 m\n(~1-2 floors)", "6-12 m\n(~2-4 floors)", "12-18 m\n(~4-6 floors)",
    "18-24 m\n(~6-8 floors)", "24-30 m\n(~8-10 floors)", "30+ m\n(10+ floors)",
]
MIN_BIN_POINTS: int = 60

# --- Bootstrap confidence intervals -------------------------------------------
N_BOOTSTRAP: int = 300
CI_PCT: float = 95.0
_BOOT_SEED: int = 2026

# --- Continuous rolling trend (Fig 2B only) -----------------------------------
# Sliding-window version of the SAME Method A (pooled ratio-of-sums) formula
# used by compute_bin_stats(), stepped finer than the fixed HEIGHT_BIN_EDGES
# grid, to show whether the pattern seen across the six fixed bins survives
# outside them. Window width matches the fixed bin width (6 m) so the two are
# comparable in granularity — only the placement differs (fixed/non-
# overlapping vs. sliding/overlapping).
ROLLING_HALF_WIDTH_M: float = 3.0          # window = [centre-3, centre+3) => 6 m wide
ROLLING_STEP_M: float = 1.0                # distance between successive window centres
MIN_ROLLING_POINTS: int = MIN_BIN_POINTS   # same reliability threshold as the fixed bins
COLOR_ROLLING_TREND: str = "#212121"       # dark neutral, distinct from the YlOrRd bar fill

# --- Assumed constant street widths (Fig 7 only) ------------------------------
ASSUMED_STREET_WIDTHS_M: list[float] = [20.0, 15.0, 10.0]
WIDTH_LINE_STYLES: dict[float, dict] = {
    20.0: dict(color="#9e9e9e", linestyle="--"),
    15.0: dict(color="#616161", linestyle="--"),
    10.0: dict(color="#212121", linestyle="--"),
}

# --- Seasonal composition (Fig 9 only) ----------------------------------------
SEASON_MAP: dict[int, str] = {
    12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
    3: "Spring (MAM)", 4: "Spring (MAM)", 5: "Spring (MAM)",
    6: "Summer (JJA)", 7: "Summer (JJA)", 8: "Summer (JJA)",
    9: "Autumn (SON)", 10: "Autumn (SON)", 11: "Autumn (SON)",
}
SEASON_ORDER: list[str] = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]
SEASON_COLORS: dict[str, str] = {
    "Winter (DJF)": "#5C6BC0", "Spring (MAM)": "#66BB6A",
    "Summer (JJA)": "#FFCA28", "Autumn (SON)": "#FB8C00",
}

# --- Solar-elevation mechanism diagnostic (Fig 10-12 only) --------------------
# Hour-of-day buckets. Note: load_combined_data() parses 'datetime' as UTC and
# does not convert it, so these hours are UTC (Lisbon = UTC+0 / UTC+1).
HOUR_BINS: list[tuple[int, int, str]] = [
    (6, 9, "06-09h"), (9, 12, "09-12h"), (12, 15, "12-15h"),
    (15, 18, "15-18h"), (18, 21, "18-21h"),
]
HOUR_COLORS: dict[str, str] = {
    "06-09h": "#1A237E", "09-12h": "#1E88E5", "12-15h": "#FFC107",
    "15-18h": "#FB8C00", "18-21h": "#6A1B9A",
}
# Solar-elevation bands for the mechanism-separation figure (Fig 12).
SUNELEV_BAND_EDGES: list[float] = [0, 15, 30, 45, 60, 90]
SUNELEV_BAND_LABELS: list[str] = ["0-15", "15-30", "30-45", "45-60", "60-90"]
MIN_CELL_POINTS: int = 150  # per (height_bin x sunelev_band) cell — finer grain than MIN_BIN_POINTS
# Sequential colour scale for height bins in Fig 12 (light = short, dark = tall) —
# deliberately distinct from the red (loss) / grey (theoretical SVF) colours
# used elsewhere, so this reads as "height" specifically.
HEIGHT_BIN_CMAP = plt.cm.Blues

# --- Colours ------------------------------------------------------------------
# Project-wide categorical palette (Urban/Rural/Moving/Parked), shared with
# every other script that splits by these conditions.
COLOR_URBAN: str = "#C0392B"
COLOR_RURAL: str = "#2E7D32"
COLOR_MOVING: str = "#1565C0"
COLOR_PARKED: str = "#E65100"
COLOR_MEASURED_BAR: str = "#E05C5C"
# Distinct, non-semantic colours for the confound diagnostic (Fig 8) —
# intentionally different from the categorical palette above so they read
# as a separate diagnostic axis, not a fifth category.
COLOR_DENSITY: str = "#00897B"
COLOR_CSI: str = "#FF8F00"
COLOR_SUNELEV: str = "#3949AB"

DPI: int = 300
SHOW_FIGURES: bool = False


# ==============================================================================
# FILENAME RESOLUTION
# ==============================================================================

def _build_sensor_files(campaign: str, data_dir: Path, tuktuk_data_dir: Path,
                        suffix: str) -> dict[str, Path]:
    """
    Build the dict mapping sensor identifiers to their CSV input paths.

    Standard campaign : <sensor_id>_satellite_landuse_v2_<suffix>.csv
    Tuk-tuk campaign  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
    """
    sensors = ["J30", "J31", "J32", "J33"]
    if campaign == "standard":
        return {
            s: data_dir / f"{s.lower()}_satellite_landuse_v2_{suffix}.csv"
            for s in sensors
        }
    elif campaign == "tuktuk":
        sep = f"_{suffix}" if suffix else ""
        return {
            s: tuktuk_data_dir / f"{s.lower()}_tuktuk_ghi_cams_cos_clc_bh_sw_norm{sep}.csv"
            for s in sensors
        }
    else:
        raise ValueError(f"Unknown CAMPAIGN '{campaign}'.")


def _variant_tag() -> str:
    """Return the active variant tag for the current campaign."""
    return FILE_SUFFIX if CAMPAIGN == "standard" else TUKTUK_FILE_SUFFIX


def _build_output_dir(base_dir: Path, campaign: str, variant_tag: str) -> Path:
    """Return the output directory, embedding campaign + variant tag."""
    return base_dir / campaign / variant_tag


SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR: Path = _build_output_dir(OUTPUT_BASE_DIR, CAMPAIGN, _VARIANT_TAG)


def _out_csv(name: str) -> Path:
    """Return the CSV output path of a table, embedding campaign + variant tag."""
    return OUTPUT_DIR / f"{name}_{CAMPAIGN}_{_VARIANT_TAG}.csv"


def _out_png(name: str) -> Path:
    """Return the PNG output path of a figure, embedding campaign + variant tag."""
    return OUTPUT_DIR / f"{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"


# ==============================================================================
# HAVERSINE — movement classification
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2, R: float = R_EARTH_KM):
    """Great-circle distance [m] between consecutive GPS points."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a)) * 1000.0


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_combined_data(sensor_files: dict[str, Path]) -> tuple[pd.DataFrame, int]:
    """
    Load, consolidate, and enrich per-sensor CSV files.

    Steps per sensor: read required columns, rename sensor-prefixed columns
    to generic names, parse/sort by datetime, classify Parked/Moving via
    Haversine distance to the next point (last row per sensor dropped, no
    forward neighbour), tag sensor. Frames are then concatenated, CLC
    urban/rural is derived, points with no building nearby are excluded,
    and autonomy/loss columns are computed.

    Returns
    -------
    tuple[pd.DataFrame, int]
        (enriched dataframe, number of rows excluded for having no
        building within the buffer radius)
    """
    date_col = "Human readable date (Europe/London)"
    frames: list[pd.DataFrame] = []

    for sensor, csv_path in sensor_files.items():
        tag = sensor.upper()
        if not csv_path.exists():
            print(f"  [WARNING] Not found, skipping : {csv_path}")
            continue

        needed = [
            date_col, f"{tag} lat", f"{tag} lon", f"{tag} Rad calib",
            "ghi", "ghi_clear", "sun_elev", "land_use_clcv",
            "mean_building_height_m", "max_building_height_m", "n_buildings_nearby",
        ]
        df = pd.read_csv(csv_path, low_memory=False, usecols=lambda c: c in needed)
        # ghi_clear is optional (only used by the confound diagnostics section);
        # every other column is mandatory.
        mandatory = [c for c in needed if c != "ghi_clear"]
        missing = [c for c in mandatory if c not in df.columns]
        if missing:
            print(f"  [WARNING] {tag}: missing columns {missing} — skipping.")
            continue
        if "ghi_clear" not in df.columns:
            df["ghi_clear"] = np.nan

        df = df.rename(columns={
            date_col: "datetime",
            f"{tag} lat": "lat", f"{tag} lon": "lon",
            f"{tag} Rad calib": "rad_calib",
        })
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        for col in ("lat", "lon", "rad_calib", "ghi", "ghi_clear", "sun_elev",
                    "land_use_clcv", "mean_building_height_m",
                    "max_building_height_m", "n_buildings_nearby"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["datetime", "lat", "lon", "rad_calib", "ghi", "sun_elev"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # --- Movement classification (Parked/Moving) ---------------------------
        if len(df) >= 2:
            dist = haversine_vectorized(
                df["lat"].values[:-1], df["lon"].values[:-1],
                df["lat"].values[1:], df["lon"].values[1:],
            )
            df = df.iloc[:-1].copy()
            df["moving_status"] = np.where(dist >= THRESHOLD_M, "Moving", "Parked")
        else:
            df["moving_status"] = pd.NA

        df["sensor"] = sensor
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. Check DATA_DIR/TUKTUK_DATA_DIR, "
            "CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX. Note this script "
            "requires cos23/CLC/BH join columns to already be present "
            "(join_cos_standard.py, join_clc_standard.py, join_bh_standard.py "
            "or their tuk-tuk equivalents)."
        )

    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(df):,} valid rows loaded across {df['sensor'].nunique()} sensor(s)")

    df["clc_group"] = df["land_use_clcv"].map(CLC_URBAN_RURAL).fillna("Unknown")

    n_before_bldg = len(df)
    df = df[df["n_buildings_nearby"] > 0].copy()
    n_excluded_bldg = n_before_bldg - len(df)
    print(f"  {len(df):,} rows retained with >=1 building within range "
          f"({n_excluded_bldg:,} excluded — no building nearby, height not "
          f"applicable, see module docstring)")

    diff_w = df["ghi"] - df["rad_calib"]
    df["e_solar_kWh"] = df["rad_calib"] / 1000.0 * DELTA_T_H
    df["loss_irr_kWh"] = (diff_w / 1000.0 * DELTA_T_H).clip(lower=0.0)
    df["auto_solar_km"] = df["e_solar_kWh"] / CVE_KWH_PER_KM
    df["loss_auto_km"] = df["loss_irr_kWh"] / CVE_KWH_PER_KM

    denom = df["auto_solar_km"] + df["loss_auto_km"]
    df["loss_pct_point"] = np.where(denom > 0, df["loss_auto_km"] / denom * 100.0, np.nan)

    return df.reset_index(drop=True), n_excluded_bldg


def assign_height_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add the height_bin column (HEIGHT_BIN_EDGES, left-closed bins)."""
    df = df.copy()
    df["height_bin"] = pd.cut(
        df["mean_building_height_m"], bins=HEIGHT_BIN_EDGES,
        labels=HEIGHT_BIN_LABELS, right=False,
    )
    return df


def _save_and_show(fig, out_path: Path) -> None:
    """Save the figure at DPI, optionally show it, then close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  saved: {out_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ==============================================================================
# CORE AGGREGATION — single source of truth for the per-bin loss formula
# ==============================================================================

def _pooled_ratio_pct(loss_sum: float, solar_sum: float) -> float:
    """
    Method A (Pooled Ratio-of-Sums) — SAL project convention:

        loss_pct = loss_sum / (solar_sum + loss_sum) x 100

    THE single formula for energy-weighted relative loss used everywhere
    in this script: fixed-bin stats (compute_bin_stats), every bootstrap
    resample (bootstrap_energy_ratio_ci), and the continuous rolling
    trend (compute_rolling_height_trend). Do not reimplement it elsewhere
    in the file, so that the sections cannot drift apart.
    """
    denom = loss_sum + solar_sum
    return (loss_sum / denom * 100.0) if denom > 0 else np.nan


def bootstrap_energy_ratio_ci(
    loss_vals: np.ndarray, solar_vals: np.ndarray,
    n_boot: int = N_BOOTSTRAP, ci_pct: float = CI_PCT,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the energy-weighted relative-loss
    ratio loss_sum / (loss_sum + solar_sum) * 100, resampling rows with
    replacement.

    Returns (ci_lo, ci_hi) as percentages, or (nan, nan) if fewer than 2
    points are available.
    """
    n = len(loss_vals)
    if n < 2:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(_BOOT_SEED)
    loss_vals = np.asarray(loss_vals, dtype=float)
    solar_vals = np.asarray(solar_vals, dtype=float)

    boot_vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_vals[i] = _pooled_ratio_pct(loss_vals[idx].sum(), solar_vals[idx].sum())

    lo_pct, hi_pct = (100.0 - ci_pct) / 2.0, 100.0 - (100.0 - ci_pct) / 2.0
    return (float(np.nanpercentile(boot_vals, lo_pct)),
            float(np.nanpercentile(boot_vals, hi_pct)))


def compute_bin_stats(
    df: pd.DataFrame, group_col: str = "height_bin",
    labels: list[str] | None = None, with_ci: bool = True,
) -> pd.DataFrame:
    """
    Per-bin summary — THE single source of truth for the energy-weighted
    relative-loss formula used everywhere in this script (Fig 1/2/4/5/6/7):

        loss_pct = sum(loss_auto_km) / (sum(auto_solar_km) + sum(loss_auto_km)) x 100

    Also reports the per-point mean-of-ratios (a secondary sensitivity
    check, not to be used in place of the formula above), the
    absolute loss (total and mean per point, in km), and — if with_ci —
    a bootstrap 95% CI on loss_pct.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain loss_auto_km, auto_solar_km, loss_pct_point, and the
        grouping column.
    group_col : str
        Column to group by (e.g. 'height_bin', or 'clc_group', or
        'moving_status').
    labels : list[str] or None
        If given, bins are output in this order (missing labels get an
        all-NaN row); otherwise the observed group values are used as-is.
    with_ci : bool
        Whether to compute the (comparatively slow) bootstrap CI.

    Returns
    -------
    pd.DataFrame
        Columns: <group_col>, n, mean_height, loss_pct, loss_pct_ci_lo,
        loss_pct_ci_hi, mean_of_ratios_pct, total_loss_auto_km,
        total_auto_solar_km, mean_loss_auto_km_per_point.
    """
    rng = np.random.default_rng(_BOOT_SEED)
    group_values = labels if labels is not None else sorted(df[group_col].dropna().unique())

    rows = []
    for label in group_values:
        sub = df[df[group_col] == label]
        n = len(sub)
        if n == 0:
            rows.append({
                group_col: label, "n": 0, "mean_height": np.nan,
                "loss_pct": np.nan, "loss_pct_ci_lo": np.nan, "loss_pct_ci_hi": np.nan,
                "mean_of_ratios_pct": np.nan,
                "total_loss_auto_km": np.nan, "total_auto_solar_km": np.nan,
                "mean_loss_auto_km_per_point": np.nan,
            })
            continue

        loss_sum = sub["loss_auto_km"].sum()
        solar_sum = sub["auto_solar_km"].sum()
        loss_pct = _pooled_ratio_pct(loss_sum, solar_sum)

        ci_lo, ci_hi = (np.nan, np.nan)
        if with_ci and n >= 2:
            ci_lo, ci_hi = bootstrap_energy_ratio_ci(
                sub["loss_auto_km"].values, sub["auto_solar_km"].values, rng=rng,
            )

        rows.append({
            group_col: label, "n": n,
            "mean_height": sub["mean_building_height_m"].mean(),
            "loss_pct": loss_pct, "loss_pct_ci_lo": ci_lo, "loss_pct_ci_hi": ci_hi,
            "mean_of_ratios_pct": sub["loss_pct_point"].mean(),
            "total_loss_auto_km": loss_sum,
            "total_auto_solar_km": solar_sum,
            "mean_loss_auto_km_per_point": sub["loss_auto_km"].mean(),
        })

    return pd.DataFrame(rows)


def compute_bin_stats_split(
    df: pd.DataFrame, split_col: str, split_values: list[str],
    with_ci: bool = True,
) -> pd.DataFrame:
    """
    Height-bin stats computed SEPARATELY within each value of `split_col`
    (e.g. Urban/Rural, or Parked/Moving) — used for Fig 4 and Fig 5.

    Returns a long-format DataFrame with an extra `split_col` column,
    concatenating compute_bin_stats() for each split value.
    """
    frames = []
    for val in split_values:
        sub = df[df[split_col] == val]
        stats = compute_bin_stats(sub, "height_bin", HEIGHT_BIN_LABELS, with_ci=with_ci)
        stats[split_col] = val
        frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def compute_rolling_height_trend(
    df: pd.DataFrame,
    half_width_m: float = ROLLING_HALF_WIDTH_M,
    step_m: float = ROLLING_STEP_M,
    min_points: int = MIN_ROLLING_POINTS,
) -> pd.DataFrame:
    """
    Continuous sliding-window version of the SAME Method A formula used by
    compute_bin_stats() (pooled ratio-of-sums, via _pooled_ratio_pct) —
    stepped every `step_m` metres instead of snapping to the fixed
    HEIGHT_BIN_EDGES grid. Purpose (Fig 2B): show whether the pattern
    across the six fixed-width bins is an artefact of where the bin edges
    happen to fall, or a genuine trend that survives at finer resolution.

    Each window is [centre - half_width_m, centre + half_width_m), i.e.
    2 x half_width_m wide (6 m by default — matching the fixed bin width,
    so the two are comparable in granularity; only the placement differs:
    fixed/non-overlapping vs. sliding/overlapping).

    No confidence interval is computed here — kept deliberately
    lightweight. Fig 2/2B's bars already carry the bootstrap CI at the
    fixed bin resolution; this line is a SHAPE diagnostic layered on
    top, not a second set of inferential estimates.

    Returns
    -------
    pd.DataFrame
        Columns: height_m (window centre), loss_pct, n. Windows with
        fewer than `min_points` rows are omitted entirely — this
        naturally truncates the line in the sparse tail rather than
        plotting noise there.
    """
    heights = df["mean_building_height_m"].to_numpy()
    loss = df["loss_auto_km"].to_numpy()
    solar = df["auto_solar_km"].to_numpy()

    lo = float(np.nanmin(heights)) + half_width_m
    hi = float(np.nanmax(heights)) - half_width_m
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return pd.DataFrame(columns=["height_m", "loss_pct", "n"])

    rows = []
    for c in np.arange(lo, hi + step_m, step_m):
        mask = (heights >= c - half_width_m) & (heights < c + half_width_m)
        n = int(mask.sum())
        if n < min_points:
            continue
        rows.append({
            "height_m": float(c),
            "loss_pct": _pooled_ratio_pct(loss[mask].sum(), solar[mask].sum()),
            "n": n,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# FIG 1 — Scatter + binned trend (with bootstrap CI band)
# ==============================================================================

def plot_fig1_scatter_trend(df: pd.DataFrame, bin_stats: pd.DataFrame,
                             n_excluded_bldg: int, out_path: Path) -> None:
    """Fig 1 — per-point loss vs building height, with the binned trend and CI band."""
    valid_bins = bin_stats[bin_stats["n"] >= MIN_BIN_POINTS].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["mean_building_height_m"], df["loss_pct_point"],
               s=4, alpha=0.06, color="#1565C0", linewidths=0,
               rasterized=True, label="Individual measurements")

    if not valid_bins.empty:
        ax.fill_between(
            valid_bins["mean_height"], valid_bins["loss_pct_ci_lo"],
            valid_bins["loss_pct_ci_hi"], color="#F4A800", alpha=0.25,
            label=f"{CI_PCT:.0f}% bootstrap CI",
        )
        ax.plot(valid_bins["mean_height"], valid_bins["loss_pct"],
                color="#F4A800", linewidth=2.5, marker="o", markersize=7,
                zorder=5, label="Binned mean (energy-weighted trend)")
        for _, row in valid_bins.iterrows():
            ax.annotate(f"n={int(row['n']):,}",
                        (row["mean_height"], row["loss_pct"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7.5, color="#555555")

    ax.set_xlabel("Mean building height nearby [m]  (50 m radius)",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("Relative autonomy loss [%]", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Autonomy Loss vs. Nearby Building Height",
                 fontsize=12, fontweight="bold", pad=20)
    ax.text(0.5, 1.04,
            "Raw points are individual 1-minute measurements (noisy by "
            "construction) — the orange line and band are the actual trend",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
    fig.text(0.5, -0.02,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  points with no building within 50 m excluded "
             f"({n_excluded_bldg:,} pts)  —  bin threshold >= {MIN_BIN_POINTS} pts  —  "
             f"band = {CI_PCT:.0f}% bootstrap CI ({N_BOOTSTRAP} resamples)",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 2 — Bar comparison across height bins (with bootstrap CI error bars)
# ==============================================================================

def plot_fig2_bins_bar(bin_stats: pd.DataFrame, n_excluded_bldg: int,
                        out_path: Path) -> None:
    """Fig 2 — energy-weighted relative loss per height bin, with bootstrap CI."""
    plot_df = bin_stats[bin_stats["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    n_excl_bins = len(bin_stats) - len(plot_df)
    n_excl_pts = int(bin_stats.loc[bin_stats["n"] < MIN_BIN_POINTS, "n"].sum())

    if plot_df.empty:
        print("  [WARNING] No bins have enough points to plot Fig 2.")
        return

    colors = plt.get_cmap("YlOrRd")(np.linspace(0.3, 0.9, len(plot_df)))
    yerr = np.vstack([
        (plot_df["loss_pct"] - plot_df["loss_pct_ci_lo"]).clip(lower=0).values,
        (plot_df["loss_pct_ci_hi"] - plot_df["loss_pct"]).clip(lower=0).values,
    ])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["loss_pct"], color=colors, edgecolor="#333333",
           linewidth=0.5, yerr=yerr, capsize=4,
           error_kw=dict(ecolor="#333333", linewidth=1.0))
    # Anchor labels above the CI whisker (loss_pct_ci_hi), not the bar top
    # (loss_pct) alone — a fixed loss_pct + 1.0 offset let the error-bar cap
    # cut straight through the label whenever the CI was wider than ~1pp
    # (fixed 2026-08; visible e.g. on the 24-30 m and 30+ m bins).
    for xi, (loss_pct, ci_hi, n) in enumerate(
            zip(plot_df["loss_pct"], plot_df["loss_pct_ci_hi"], plot_df["n"])):
        ax.text(xi, ci_hi + 1.5, f"{loss_pct:.1f}%\n(n={n:,})",
                ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_ylim(0, plot_df["loss_pct_ci_hi"].max() * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["height_bin"], fontsize=8.5)
    ax.set_ylabel("Relative Autonomy Loss [%]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean building height nearby (50 m radius)",
                  fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Relative Autonomy Loss by Building-Height Bin",
                 fontsize=12, fontweight="bold", pad=20)
    fig.text(0.5, -0.04,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  no building within 50 m excluded "
             f"({n_excluded_bldg:,} pts)  —  bin threshold >= {MIN_BIN_POINTS} pts"
             + (f"  —  {n_excl_bins} bin(s) excluded ({n_excl_pts} pts)" if n_excl_bins else "")
             + f"  —  error bars = {CI_PCT:.0f}% bootstrap CI",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 2B — Fig 2 + continuous rolling trend + absolute-autonomy reference
# ==============================================================================

def plot_fig2b_bins_bar_trend(df: pd.DataFrame, bin_stats: pd.DataFrame,
                               n_excluded_bldg: int, out_path: Path) -> None:
    """
    Same bar chart as Fig 2 (energy-weighted relative loss per fixed
    height bin, Method A, bootstrap 95% CI), with two additions:

      1. A continuous sliding-window trend line (compute_rolling_height_
         trend — SAME Method A formula, no re-implementation) plotted on
         a secondary real-metres top axis, so the reader can see whether
         the bar pattern survives outside the fixed 6 m bin edges.
      2. An explicit absolute-autonomy reference: each bar is annotated
         with the mean loss per measurement in metres, and a boxed
         annotation reports the total intact + lost solar-driving
         autonomy (km) the whole plot is computed from — so "X% loss" is
         anchored to a concrete quantity, not read as a bare percentage.

    Caveat (stated in the footnote too): the top (continuous, metres) and
    bottom (categorical, fixed-width bin) axes align well for bins 1-5
    (all exactly 6 m wide) but NOT for the open-ended 30+ m bin — for
    that region, read the top axis for the trend line's true height
    value, not its horizontal position relative to the last bar.
    """
    plot_df = bin_stats[bin_stats["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    n_excl_bins = len(bin_stats) - len(plot_df)
    n_excl_pts = int(bin_stats.loc[bin_stats["n"] < MIN_BIN_POINTS, "n"].sum())

    if plot_df.empty:
        print("  [WARNING] No bins have enough points to plot Fig 2B.")
        return

    colors = plt.get_cmap("YlOrRd")(np.linspace(0.3, 0.9, len(plot_df)))
    yerr = np.vstack([
        (plot_df["loss_pct"] - plot_df["loss_pct_ci_lo"]).clip(lower=0).values,
        (plot_df["loss_pct_ci_hi"] - plot_df["loss_pct"]).clip(lower=0).values,
    ])

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["loss_pct"], color=colors, edgecolor="#333333",
           linewidth=0.5, yerr=yerr, capsize=4, zorder=3,
           error_kw=dict(ecolor="#333333", linewidth=1.0),
           label="Fixed bin, energy-weighted (Method A) + 95% bootstrap CI")

    # --- Continuous rolling trend geometry, computed BEFORE the bar labels ----
    # (needed up front so each label can be anchored above the trend line too,
    # not just above its own bar — see label placement below).
    rolling = compute_rolling_height_trend(df)
    ax_top = ax.twiny()
    edges = HEIGHT_BIN_EDGES
    # Bottom axis is categorical (bar index); align the top axis to the real
    # height range spanned by the PLOTTED bins (first plotted bin's lower
    # edge -> last plotted bin's upper edge) so both axes share the same
    # visual left/right extent.
    plotted_labels = list(plot_df["height_bin"])
    first_idx = HEIGHT_BIN_LABELS.index(plotted_labels[0])
    last_idx = HEIGHT_BIN_LABELS.index(plotted_labels[-1])
    height_lo, height_hi = edges[first_idx], edges[last_idx + 1]
    # Cap the display range if the open-ended 30+ m bin is included, so the
    # sparse long tail doesn't compress the informative part of the line.
    height_hi_display = min(height_hi, max(45.0, edges[last_idx]))

    # Labels are anchored above whichever is higher, the bar's own CI whisker
    # or the rolling trend line's GLOBAL peak — otherwise (fixed 2026-08) a
    # fixed loss_pct + 1.0 offset let both the CI cap and the trend line cut
    # straight through the label whenever either rose more than ~1pp above
    # the bar. A per-bin local window was tried first but the top (real
    # metres) / bottom (categorical bin) axes only align exactly for the
    # 6 m-wide bins — near the open-ended 30+ m bin a local window can
    # under-count the line's height at that on-screen position, so the
    # global peak is used instead for a guaranteed, simple clearance.
    rolling_global_max = rolling["loss_pct"].max() if not rolling.empty else np.nan
    label_pad = 2.0
    for xi, (loss_pct, ci_hi, n, m_loss) in enumerate(zip(
            plot_df["loss_pct"], plot_df["loss_pct_ci_hi"], plot_df["n"],
            plot_df["mean_loss_auto_km_per_point"])):
        clearance = ci_hi if pd.isna(rolling_global_max) else max(ci_hi, rolling_global_max)
        ax.text(xi, clearance + label_pad,
                f"{loss_pct:.1f}%\n(n={n:,})\n\u2248{m_loss * 1000:.0f} m/pt",
                ha="center", va="bottom", fontsize=7.5, color="#333333", zorder=7)

    ax.set_xlim(-0.5, len(plot_df) - 0.5)
    ax_top.set_xlim(height_lo, height_hi_display)

    if not rolling.empty:
        rplot = rolling[(rolling["height_m"] >= height_lo) &
                         (rolling["height_m"] <= height_hi_display)]
        ax_top.plot(rplot["height_m"], rplot["loss_pct"],
                    color=COLOR_ROLLING_TREND, linewidth=2.2, zorder=6,
                    label=f"Continuous rolling trend "
                          f"({2 * ROLLING_HALF_WIDTH_M:.0f} m window, "
                          f"{ROLLING_STEP_M:.0f} m step, same Method A formula)")
    ax_top.set_xlabel("Mean building height nearby [m]  (continuous, top axis)",
                       fontsize=8.5, color="#555555")
    ax_top.tick_params(axis="x", labelsize=7.5, colors="#555555")

    # Extra headroom: the 3-line bar annotations (%, n, m/pt) are taller than
    # Fig 2's 2-line version and can otherwise collide between neighbouring
    # bars when the trend is steep.
    top_candidates = [plot_df["loss_pct_ci_hi"].max()]
    if not rolling.empty:
        top_candidates.append(rolling["loss_pct"].max())
    y_top = max(v for v in top_candidates if pd.notna(v)) * 1.35
    ax.set_ylim(0, y_top)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["height_bin"], fontsize=8.5)
    ax.set_ylabel("Relative Autonomy Loss [%]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean building height nearby (50 m radius) — fixed bins, bottom axis",
                  fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax_top.spines[["right"]].set_visible(False)
    ax.set_title("Relative Autonomy Loss by Building-Height Bin — with Continuous Trend",
                 fontsize=12, fontweight="bold", pad=30)

    # --- Absolute-autonomy reference box ---------------------------------------
    total_loss_all = df["loss_auto_km"].sum()
    total_solar_all = df["auto_solar_km"].sum()
    n_all = len(df)
    ref_text = (
        f"Reference scale — this plot's underlying data totals "
        f"{total_solar_all:,.0f} km intact + {total_loss_all:,.0f} km lost "
        f"solar-driving autonomy across {n_all:,} 1-min measurements "
        f"(\u2248{total_solar_all / n_all * 1000:.0f} m intact / "
        f"{total_loss_all / n_all * 1000:.0f} m lost, on average per measurement)."
    )
    ax.text(0.5, 1.15, ref_text, transform=ax.transAxes, ha="center",
            fontsize=7.5, color="#555555", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#CCCCCC"))

    bar_handles, bar_labels = ax.get_legend_handles_labels()
    line_handles, line_labels = ax_top.get_legend_handles_labels()
    ax.legend(bar_handles + line_handles, bar_labels + line_labels,
              fontsize=7.5, loc="upper left", framealpha=0.9)

    fig.text(
        0.5, -0.05,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  no building within 50 m excluded "
        f"({n_excluded_bldg:,} pts)  —  bin threshold >= {MIN_BIN_POINTS} pts"
        + (f"  —  {n_excl_bins} bin(s) excluded ({n_excl_pts} pts)" if n_excl_bins else "")
        + f"  —  bars: {CI_PCT:.0f}% bootstrap CI  —  line: same Method A formula, "
          f"no CI (shape diagnostic only)  —  top axis capped at "
          f"{height_hi_display:.0f} m for readability (30+ m bin is open-ended)",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 3 — Building-height distribution histogram
# ==============================================================================

def plot_fig3_height_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 3 — histogram of mean_building_height_m with the bin edges."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(df["mean_building_height_m"], bins=60, color="#1565C0",
            alpha=0.75, edgecolor="white", linewidth=0.3)
    for edge in HEIGHT_BIN_EDGES[1:-1]:
        ax.axvline(edge, color="#E05C5C", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Mean building height nearby [m]  (50 m radius)",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of measurements", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Distribution of Nearby Building Height",
                 fontsize=12, fontweight="bold", pad=16)
    fig.text(0.5, -0.02,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  dashed red lines = height-bin edges "
             f"used in Fig 1/2/4/5/6/7  —  points with no building within 50 m "
             f"already excluded from this histogram",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 4 / FIG 5 — Grouped bars by condition split (Urban/Rural, Parked/Moving)
# ==============================================================================

def plot_fig_split_bar(split_stats: pd.DataFrame, split_col: str,
                        split_values: list[str], split_colors: dict[str, str],
                        title: str, out_path: Path) -> None:
    """Generic grouped-bar plotter shared by Fig 4 (Urban/Rural) and Fig 5
    (Parked/Moving) — same per-bin statistic (compute_bin_stats_split),
    different grouping column."""
    valid = split_stats[split_stats["n"] >= MIN_BIN_POINTS]
    bins_present = [b for b in HEIGHT_BIN_LABELS if b in valid["height_bin"].unique()]
    if not bins_present:
        print(f"  [WARNING] No bins have enough points to plot {title}.")
        return

    x = np.arange(len(bins_present))
    n_groups = len(split_values)
    w = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for i, val in enumerate(split_values):
        sub = split_stats[split_stats[split_col] == val].set_index("height_bin")
        sub = sub.reindex(bins_present)
        offset = (i - (n_groups - 1) / 2) * w
        ax.bar(x + offset, sub["loss_pct"].values, w,
               label=f"{val} (n={int(sub['n'].sum()):,})",
               color=split_colors[val], alpha=0.85,
               edgecolor="white", linewidth=0.6)
        for xi, row in zip(x + offset, sub.itertuples()):
            if pd.notna(row.loss_pct) and row.n >= MIN_BIN_POINTS:
                ax.text(xi, row.loss_pct + 0.8, f"{row.loss_pct:.1f}%",
                        ha="center", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([b.split("\n")[0] for b in bins_present], fontsize=9)
    ax.set_xlabel("Mean building height nearby (50 m radius)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Relative Autonomy Loss [%]", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=16)
    fig.text(0.5, -0.02,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin/group threshold >= {MIN_BIN_POINTS} pts "
             "(bars below threshold omitted)",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 6 — Absolute vs relative loss, twin axis
# ==============================================================================

def plot_fig6_absolute_vs_relative(bin_stats: pd.DataFrame, out_path: Path) -> None:
    """Fig 6 — absolute (km) vs relative (%) loss per height bin, twin axis."""
    plot_df = bin_stats[bin_stats["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    if plot_df.empty:
        print("  [WARNING] No bins have enough points to plot Fig 6.")
        return

    x = np.arange(len(plot_df))
    w = 0.38

    fig, ax1 = plt.subplots(figsize=(9.5, 6))
    ax2 = ax1.twinx()

    bars_rel = ax1.bar(x - w / 2, plot_df["loss_pct"], w, color=COLOR_MEASURED_BAR,
                        alpha=0.85, edgecolor="white", linewidth=0.6,
                        label="Relative loss [%] (energy-weighted)")
    bars_abs = ax2.bar(x + w / 2, plot_df["mean_loss_auto_km_per_point"], w,
                        color="#424242", alpha=0.85, edgecolor="white", linewidth=0.6,
                        label="Absolute loss [km lost / measurement, mean]")

    ax1.set_xticks(x)
    ax1.set_xticklabels([b.split("\n")[0] for b in plot_df["height_bin"]], fontsize=9)
    ax1.set_xlabel("Mean building height nearby (50 m radius)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Relative loss [%]", fontsize=10, fontweight="bold", color=COLOR_MEASURED_BAR)
    ax1.tick_params(axis="y", labelcolor=COLOR_MEASURED_BAR)
    ax2.set_ylabel("Absolute loss [km / measurement]", fontsize=10, fontweight="bold", color="#424242")
    ax2.tick_params(axis="y", labelcolor="#424242")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    handles = [bars_rel, bars_abs]
    ax1.legend(handles, [h.get_label() for h in handles], fontsize=9, loc="upper left", framealpha=0.9)
    ax1.set_title("Absolute vs. Relative Autonomy Loss by Building-Height Bin",
                  fontsize=12, fontweight="bold", pad=16)
    fig.text(0.5, -0.02,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
             "relative loss can stay flat while absolute loss rises with exposure time, "
             "or vice versa — compare both before drawing conclusions from either alone",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# FIG 7 — SVF theoretical overlay  (RUN_SVF_OVERLAY)
# ==============================================================================

# ==============================================================================
# FIG 7 — SVF theoretical overlay  (RUN_SVF_OVERLAY)
# ==============================================================================
# Diffuse-loss physics (obstruction angle, SVF) now lives in brito_svf_model.py
# (single source of truth shared with hw_ratio_impact_analysis.py and
# hw_ratio_real_analysis.py — see that module's docstring for why this
# extraction was done). That module's diffuse_loss_pct(hw, edge) takes a
# single H/W ratio array, unlike the local function this replaces (which
# took H and a scalar W separately and divided internally) — the call site
# below computes H/W explicitly before calling it.


def plot_fig7_svf_overlay(bin_stats: pd.DataFrame, n_total: int,
                           n_excluded_bldg: int, out_path: Path) -> None:
    """Fig 7 — measured loss per bin vs theoretical diffuse SVF loss (assumed widths)."""
    plot_df = bin_stats[(bin_stats["n"] >= MIN_BIN_POINTS) & bin_stats["loss_pct"].notna()]
    plot_df = plot_df.reset_index(drop=True)
    if plot_df.empty:
        print("  [WARNING] No bins have enough points to plot Fig 7.")
        return

    x = np.arange(len(plot_df))
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, plot_df["loss_pct"].values, width=0.55, color=COLOR_MEASURED_BAR,
                    alpha=0.80, edgecolor="white", linewidth=0.8, zorder=3,
                    label="Measured relative loss (energy-weighted)")
    # Anchor above whichever is higher, the bar or the diamond sensitivity
    # marker (mean_of_ratios_pct) — a fixed bar-height + 1.0 offset let the
    # diamond sit right on top of the label whenever the ratio exceeded the
    # bar by less than ~1.5pp (fixed 2026-08; visible e.g. on the 12-18 m
    # and 24-30 m bins).
    for xi, (n, v, ratio) in enumerate(
            zip(plot_df["n"], plot_df["loss_pct"], plot_df["mean_of_ratios_pct"])):
        clearance = v if pd.isna(ratio) else max(v, ratio)
        ax1.text(xi, clearance + 1.5, f"n={int(n):,}", ha="center", va="bottom",
                  fontsize=7.5, color="#555555", zorder=7)

    sens = ax1.scatter(x, plot_df["mean_of_ratios_pct"].values, marker="D", s=45,
                        color="#212121", zorder=6,
                        label="Mean of per-point ratios (sensitivity check)")

    ax1.set_xticks(x)
    ax1.set_xticklabels([b.split("\n")[0] for b in plot_df["height_bin"]], fontsize=10)
    ax1.set_xlabel("Mean building height within 50 m buffer", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Measured relative shading loss [%]\n(diffuse + direct, all-sky)",
                   fontsize=10, fontweight="bold", color=COLOR_MEASURED_BAR)
    ax1.tick_params(axis="y", labelcolor=COLOR_MEASURED_BAR)
    ax1.set_ylim(0, max(60.0, plot_df["loss_pct"].max() * 1.25))
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines[["top"]].set_visible(False)

    H_bin = plot_df["mean_height"].values
    theo_lines = []
    for W in ASSUMED_STREET_WIDTHS_M:
        loss_theo = diffuse_loss_pct(H_bin / W, edge=False)
        style = WIDTH_LINE_STYLES[W]
        line, = ax2.plot(x, loss_theo, marker="o", markersize=5, linewidth=1.8,
                          label=f"Theoretical SVF loss, W={W:.0f} m (driving)",
                          zorder=4, **style)
        theo_lines.append(line)

    ax2.set_ylabel("Theoretical isotropic-diffuse loss [%]\n(SVF, middle of street, Brito et al. 2024)",
                   fontsize=10, fontweight="bold", color="#424242")
    ax2.tick_params(axis="y", labelcolor="#424242")
    ax2.spines[["top"]].set_visible(False)

    handles = [bars, sens] + theo_lines
    ax1.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="upper left", framealpha=0.9)
    ax1.set_title("Building Height vs. Shading Loss — Geometry vs. Measurement\n"
                  "Does the measured pattern survive a purely geometric, monotonic-by-construction model?",
                  fontsize=11.5, fontweight="bold", pad=14)

    width_str = ", ".join(f"{w:.0f} m" for w in ASSUMED_STREET_WIDTHS_M)
    fig.text(0.5, -0.04,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
             f"{n_excluded_bldg:,}/{n_total:,} pts excluded (no building within 50 m)  —  "
             f"assumed street widths: {width_str}  —  "
             f"theoretical curve is DIFFUSE-ONLY (shape comparison, not magnitude)  —  "
             f"measured loss_pct = loss_auto / (auto_solar + loss_auto) x 100",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# CONFOUND DIAGNOSTICS  (RUN_CONFOUND_DIAGNOSTICS)
# ==============================================================================

def compute_confound_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Adds _month, _season, _kc_clear columns for the confound diagnostics."""
    df = df.copy()
    df["_month"] = df["datetime"].dt.month
    df["_season"] = df["_month"].map(SEASON_MAP)
    if "ghi_clear" not in df.columns or df["ghi_clear"].isna().all():
        print("  [warn] 'ghi_clear' unavailable — kc_clear proxy will be all-NaN.")
        df["_kc_clear"] = np.nan
    else:
        ghi_clear_safe = df["ghi_clear"].replace(0, np.nan)
        df["_kc_clear"] = df["ghi"] / ghi_clear_safe
    return df


def compute_bin_confound_diagnostics(df: pd.DataFrame, bin_stats: pd.DataFrame) -> pd.DataFrame:
    """Per height bin: building density, max height, kc_clear, sun elevation and measured loss."""
    rows = []
    for label in HEIGHT_BIN_LABELS:
        sub = df[df["height_bin"] == label]
        n = len(sub)
        if n == 0:
            continue
        rows.append(dict(
            height_bin=label, n=n,
            n_buildings_nearby_mean=sub["n_buildings_nearby"].mean(),
            max_building_height_mean=sub["max_building_height_m"].mean(),
            kc_clear_mean=sub["_kc_clear"].mean(),
            kc_clear_median=sub["_kc_clear"].median(),
            sun_elev_mean=sub["sun_elev"].mean(),
        ))
    diag = pd.DataFrame(rows)
    diag = diag.merge(bin_stats[["height_bin", "loss_pct"]]
                       .rename(columns={"loss_pct": "measured_loss_pct"}),
                       on="height_bin", how="left")
    return diag


def compute_season_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Per height bin: share of points (%) in each meteorological season."""
    rows = []
    for label in HEIGHT_BIN_LABELS:
        sub = df[df["height_bin"] == label]
        n = len(sub)
        if n == 0:
            continue
        counts = sub["_season"].value_counts(normalize=True) * 100.0
        row = {"height_bin": label, "n": n}
        for s in SEASON_ORDER:
            row[s] = counts.get(s, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fig8_confound_panels(diag: pd.DataFrame, n_total: int,
                               n_excluded_bldg: int, out_path: Path) -> None:
    """Fig 8 — 2x2 confound panels (loss, density, clear-sky index, sun elevation)."""
    plot_df = diag[diag["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    if plot_df.empty:
        print("  [WARNING] No bins have enough points to plot Fig 8.")
        return

    x = np.arange(len(plot_df))
    labels_short = [lbl.split("\n")[0] for lbl in plot_df["height_bin"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.bar(x, plot_df["measured_loss_pct"], color=COLOR_MEASURED_BAR, edgecolor="white", linewidth=0.8)
    for xi, v in zip(x, plot_df["measured_loss_pct"]):
        ax.text(xi, v + 1.0, f"{v:.1f}%", ha="center", fontsize=8, color="#333333")
    ax.set_title("Measured relative loss (reference)", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Loss [%]", fontsize=9.5)

    ax = axes[0, 1]
    ax.bar(x, plot_df["n_buildings_nearby_mean"], color=COLOR_DENSITY, edgecolor="white", linewidth=0.8)
    ax.set_title("Building density proxy\n(mean n_buildings_nearby, 50 m buffer)",
                 fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Mean buildings nearby [-]", fontsize=9.5)

    ax = axes[1, 0]
    ax.bar(x, plot_df["kc_clear_mean"], color=COLOR_CSI, edgecolor="white", linewidth=0.8)
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
    ax.set_title("Mean clear-sky index  (GHI / GHI_clear)", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("kc_clear [-]", fontsize=9.5)
    ax.set_ylim(0, max(1.15, plot_df["kc_clear_mean"].max() * 1.1))

    ax = axes[1, 1]
    ax.bar(x, plot_df["sun_elev_mean"], color=COLOR_SUNELEV, edgecolor="white", linewidth=0.8)
    ax.set_title("Mean solar elevation", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Sun elevation [deg]", fontsize=9.5)

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Candidate Confounds by Building-Height Bin\n"
                 "Does any proxy track the measured-loss pattern better than height alone?",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
             f"{n_excluded_bldg:,}/{n_total:,} pts excluded (no building within 50 m)  —  "
             f"kc_clear = GHI / GHI_clear (CAMS)",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


def plot_fig9_season_composition(season_df: pd.DataFrame, n_total: int,
                                  n_excluded_bldg: int, out_path: Path) -> None:
    """Fig 9 — seasonal composition of each height bin (stacked bars)."""
    plot_df = season_df[season_df["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    if plot_df.empty:
        print("  [WARNING] No bins have enough points to plot Fig 9.")
        return

    x = np.arange(len(plot_df))
    labels_short = [lbl.split("\n")[0] for lbl in plot_df["height_bin"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(plot_df))
    for s in SEASON_ORDER:
        vals = plot_df[s].values
        ax.bar(x, vals, bottom=bottom, color=SEASON_COLORS[s], label=s,
               edgecolor="white", linewidth=0.6)
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=10)
    ax.set_ylabel("Share of measurements [%]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean building height within 50 m buffer", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, loc="upper center", ncol=4, framealpha=0.9, bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Seasonal Composition by Building-Height Bin", fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.26,
             f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
             f"{n_excluded_bldg:,}/{n_total:,} pts excluded (no building within 50 m)  —  "
             f"meteorological seasons (DJF/MAM/JJA/SON)",
             ha="center", fontsize=7.5, color="#888888", style="italic")
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# SOLAR-ELEVATION MECHANISM DIAGNOSTIC  (RUN_SUNELEV_MECHANISM_DIAGNOSTIC)
# ==============================================================================
# Direct empirical test of two competing explanations for why mean solar
# elevation differs systematically across building-height bins (visible in
# Fig 8, panel 4): ~34 deg at 0-6 m down to ~26.5 deg at 30+ m.
#
# (A) OPTICAL SELECTION: a narrow canyon (high H/W) has a higher obstruction
#     angle theta = atan(H/W), which "filters" low sun-elevation periods
#     more severely (more of them are shaded), so narrow-canyon bins end up
#     with a systematically different sun_elev distribution.
# (B) TEMPORAL / SAMPLING CONFOUND: the vehicles simply happened to be near
#     tall buildings at different times of day/year than near low
#     buildings (routes, schedules), independent of any optical effect.
#
# Why these are distinguishable: optical obstruction in this dataset does
# NOT remove rows — a shaded minute still gets recorded (low rad_calib,
# high loss), it is not dropped from the CSV. So:
#   - Hypothesis (A), if real, would only be able to shift the marginal
#     sun_elev distribution by physically EXCLUDING low-elevation rows —
#     which would show up as TRUNCATION (a sharp cut at the low end) of the
#     per-bin distribution, at the low end specifically (an "is the sun
#     above the obstruction" gate cannot remove high-elevation rows). That
#     predicts HIGHER, not lower, mean sun_elev in narrow-canyon bins —
#     opposite in sign to what is observed.
#   - Hypothesis (B) predicts no such cut — just a smooth shift of the
#     whole distribution (different month/hour mix), since nothing removes
#     any row.
#
# Fig 10 : sun_elev distribution shape per height bin (density + CDF) —
#          tests truncation vs. shift directly.
# Fig 11 : hour-of-day composition per height bin (stacked bars) — extends
#          Fig 9's season composition to within-day timing.
# Fig 12 : measured relative loss vs. solar-elevation band, one curve per
#          height bin — the practically decisive figure. Overlapping
#          curves = height carries no information once sun_elev is
#          controlled (pure confound). Taller-building bins sitting
#          systematically above shorter-building bins at the SAME sun_elev
#          = genuine geometric (shading) signal, independent of the
#          sampling-mean question.

def _height_bin_colors(n: int) -> np.ndarray:
    """Return *n* sequential colours (light = short, dark = tall) for the height bins."""
    return HEIGHT_BIN_CMAP(np.linspace(0.35, 0.95, n))


def add_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add _hour and _hour_bucket from the parsed 'datetime' column (UTC)."""
    df = df.copy()
    df["_hour"] = df["datetime"].dt.hour

    def _bucket(h):
        if pd.isna(h):
            return np.nan
        for lo, hi, label in HOUR_BINS:
            if lo <= h < hi:
                return label
        return "other"

    df["_hour_bucket"] = df["_hour"].apply(_bucket)
    return df


def assign_sunelev_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add the sunelev_band column (SUNELEV_BAND_EDGES)."""
    df = df.copy()
    df["sunelev_band"] = pd.cut(
        df["sun_elev"], bins=SUNELEV_BAND_EDGES, labels=SUNELEV_BAND_LABELS,
        right=False, include_lowest=True,
    )
    return df


def _energy_weighted_loss_pct(sub: pd.DataFrame) -> float:
    """Same formula as compute_bin_stats(), applied to an arbitrary subset
    (here, a height_bin x sunelev_band cell)."""
    auto_sum = sub["auto_solar_km"].sum()
    loss_sum = sub["loss_auto_km"].sum()
    denom = auto_sum + loss_sum
    return (loss_sum / denom * 100.0) if denom > 0 else np.nan


def compute_hour_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Per height bin: share of points (%) in each hour-of-day bucket."""
    rows = []
    for label in HEIGHT_BIN_LABELS:
        sub = df[df["height_bin"] == label]
        n = len(sub)
        if n == 0:
            continue
        counts = sub["_hour_bucket"].value_counts(normalize=True) * 100.0
        row = {"height_bin": label, "n": n}
        for _, _, hb in HOUR_BINS:
            row[hb] = counts.get(hb, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_loss_by_height_and_sunelev(df: pd.DataFrame) -> pd.DataFrame:
    """Energy-weighted loss per (height bin x sun-elevation band) cell, NaN below MIN_CELL_POINTS."""
    rows = []
    for hlabel in HEIGHT_BIN_LABELS:
        for slabel in SUNELEV_BAND_LABELS:
            sub = df[(df["height_bin"] == hlabel) & (df["sunelev_band"] == slabel)]
            n = len(sub)
            loss_pct = _energy_weighted_loss_pct(sub) if n >= MIN_CELL_POINTS else np.nan
            rows.append(dict(height_bin=hlabel, sunelev_band=slabel, n=n, loss_pct=loss_pct))
    return pd.DataFrame(rows)


def plot_fig10_sunelev_distribution(df: pd.DataFrame, plot_bins: list[str],
                                     n_total: int, n_excluded_bldg: int,
                                     out_path: Path) -> None:
    """Fig 10 — sun-elevation distribution per height bin (density + empirical CDF)."""
    colors = _height_bin_colors(len(plot_bins))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for color, hlabel in zip(colors, plot_bins):
        vals = df.loc[df["height_bin"] == hlabel, "sun_elev"].dropna().values
        short = hlabel.split("\n")[0]
        if len(vals) == 0:
            continue
        counts, edges = np.histogram(vals, bins=40, range=(0, 90), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax1.plot(centers, counts, color=color, linewidth=2, label=f"{short} (n={len(vals):,})")

        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax2.plot(sorted_vals, cdf, color=color, linewidth=2, label=short)

    ax1.set_xlabel("Solar elevation [deg]", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Probability density [deg$^{-1}$]", fontsize=10, fontweight="bold")
    ax1.set_title("Distribution shape", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(alpha=0.3, linestyle="--")

    ax2.set_xlabel("Solar elevation [deg]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cumulative fraction", fontsize=10, fontweight="bold")
    ax2.set_title("Empirical CDF — truncation vs. shift", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.text(
        0.03, 0.97,
        "Hard truncation (Hyp. A): curves start at\n"
        "different x but share the SAME shape\n"
        "above the cut.\n\n"
        "Smooth shift (Hyp. B): curves are roughly\n"
        "parallel translations across the full range.",
        transform=ax2.transAxes, fontsize=7.8, va="top", color="#444444",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    fig.suptitle(
        "Solar Elevation Distribution by Building-Height Bin\n"
        "Is the low-elevation tail truncated (optical selection) or just shifted (sampling)?",
        fontsize=12.5, fontweight="bold", y=1.04,
    )
    fig.text(
        0.5, -0.03,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
        f"{n_excluded_bldg:,}/{n_total:,} pts excluded (no building within 50 m)",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


def plot_fig11_hour_composition(hour_df: pd.DataFrame, n_total: int,
                                 n_excluded_bldg: int, out_path: Path) -> None:
    """Fig 11 — hour-of-day composition of each height bin (stacked bars)."""
    plot_df = hour_df[hour_df["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    if plot_df.empty:
        print("  [WARNING] No bins have enough points to plot Fig 11.")
        return

    x = np.arange(len(plot_df))
    labels_short = [lbl.split("\n")[0] for lbl in plot_df["height_bin"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(plot_df))
    for _, _, hb in HOUR_BINS:
        vals = plot_df[hb].values
        ax.bar(x, vals, bottom=bottom, color=HOUR_COLORS[hb], label=hb,
               edgecolor="white", linewidth=0.6)
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=10)
    ax.set_ylabel("Share of measurements [%]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean building height within 50 m buffer", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, loc="upper center", ncol=5, framealpha=0.9, bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Hour-of-Day Composition by Building-Height Bin", fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5, -0.26,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
        f"{n_excluded_bldg:,}/{n_total:,} pts excluded (no building within 50 m)  —  local time",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


def plot_fig12_loss_vs_sunelev(cell_df: pd.DataFrame, plot_bins: list[str],
                                n_total: int, n_excluded_bldg: int,
                                out_path: Path) -> None:
    """Fig 12 — relative loss vs sun-elevation band, one curve per height bin."""
    colors = _height_bin_colors(len(plot_bins))
    x = np.arange(len(SUNELEV_BAND_LABELS))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for color, hlabel in zip(colors, plot_bins):
        sub = (cell_df[cell_df["height_bin"] == hlabel]
               .set_index("sunelev_band").reindex(SUNELEV_BAND_LABELS))
        short = hlabel.split("\n")[0]
        ax.plot(x, sub["loss_pct"].values, marker="o", markersize=6,
                linewidth=2, color=color, label=short)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl} deg" for lbl in SUNELEV_BAND_LABELS], fontsize=10)
    ax.set_xlabel("Solar elevation band", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Measured relative loss [%]  (energy-weighted)", fontsize=10.5, fontweight="bold")
    ax.legend(title="Building height", fontsize=8.5, title_fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Shading Loss vs. Solar Elevation, by Building-Height Bin\n"
        "Overlapping curves = height adds no information once sun elevation is controlled",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.text(
        0.5, -0.05,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  cell threshold >= {MIN_CELL_POINTS} pts (gaps = below threshold)  —  "
        f"{n_excluded_bldg:,}/{n_total:,} pts excluded (no building within 50 m)  —  "
        f"loss_pct = loss_auto / (auto_solar + loss_auto) x 100",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_and_show(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the base analysis, then the optional SVF, confound and sun-elevation sections."""
    print("=" * 78)
    print("  BUILDING HEIGHT vs. AUTONOMY LOSSES  (+ SVF overlay, + confound diag.)")
    print(f"  CAMPAIGN     : {CAMPAIGN}")
    print(f"  Data variant : {_VARIANT_TAG}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print("=" * 78)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/7] Loading sensor data...")
    df, n_excluded_bldg = load_combined_data(SENSOR_FILES)
    n_total = len(df) + n_excluded_bldg
    df = assign_height_bins(df)

    print("\n[2/7] Computing per-height-bin statistics (bootstrap CI)...")
    bin_stats = compute_bin_stats(df, "height_bin", HEIGHT_BIN_LABELS)
    print(bin_stats.to_string(index=False))
    bin_stats.round(3).to_csv(_out_csv("bh_bin_summary"), index=False)
    print(f"  Saved : {_out_csv('bh_bin_summary')}")

    print("\n[3/7] Figures 1-3 (base relationship + histogram)...")
    plot_fig1_scatter_trend(df, bin_stats, n_excluded_bldg, _out_png("bh_fig1_scatter_trend"))
    plot_fig2_bins_bar(bin_stats, n_excluded_bldg, _out_png("bh_fig2_bins_bar"))
    plot_fig2b_bins_bar_trend(df, bin_stats, n_excluded_bldg, _out_png("bh_fig2b_bins_bar_trend"))
    plot_fig3_height_histogram(df, _out_png("bh_fig3_height_histogram"))

    print("\n[4/7] Figures 4-6 (Urban/Rural, Parked/Moving, absolute vs relative)...")
    ur_stats = compute_bin_stats_split(df, "clc_group", ["Urban", "Rural"])
    ur_stats.round(3).to_csv(_out_csv("bh_urban_rural_summary"), index=False)
    print(f"  Saved : {_out_csv('bh_urban_rural_summary')}")
    n_rural = int((df["clc_group"] == "Rural").sum())
    print(f"  Rural points with a building nearby: {n_rural:,} "
          f"({n_rural / len(df) * 100:.1f}% of retained points)")
    plot_fig_split_bar(
        ur_stats, "clc_group", ["Urban", "Rural"],
        {"Urban": COLOR_URBAN, "Rural": COLOR_RURAL},
        "Relative Autonomy Loss by Building-Height Bin — Urban vs Rural",
        _out_png("bh_fig4_urban_rural"),
    )

    pm_stats = compute_bin_stats_split(df, "moving_status", ["Parked", "Moving"])
    pm_stats.round(3).to_csv(_out_csv("bh_parked_moving_summary"), index=False)
    print(f"  Saved : {_out_csv('bh_parked_moving_summary')}")
    plot_fig_split_bar(
        pm_stats, "moving_status", ["Parked", "Moving"],
        {"Parked": COLOR_PARKED, "Moving": COLOR_MOVING},
        "Relative Autonomy Loss by Building-Height Bin — Parked vs Moving",
        _out_png("bh_fig5_parked_moving"),
    )

    plot_fig6_absolute_vs_relative(bin_stats, _out_png("bh_fig6_absolute_vs_relative"))

    if RUN_SVF_OVERLAY:
        print("\n[5/7] Figure 7 — SVF theoretical overlay (RUN_SVF_OVERLAY=True)...")
        plot_fig7_svf_overlay(bin_stats, n_total, n_excluded_bldg, _out_png("bh_fig7_svf_overlay"))
    else:
        print("\n[5/7] Skipping SVF overlay (RUN_SVF_OVERLAY=False).")

    if RUN_CONFOUND_DIAGNOSTICS:
        print("\n[6/7] Figures 8-9 — confound diagnostics (RUN_CONFOUND_DIAGNOSTICS=True)...")
        df_conf = compute_confound_proxies(df)
        diag = compute_bin_confound_diagnostics(df_conf, bin_stats)
        season_df = compute_season_composition(df_conf)

        print("\n  Per-bin confound diagnostics:")
        print(diag.to_string(index=False))
        diag.round(3).to_csv(_out_csv("bh_confound_diagnostics"), index=False)
        print(f"  Saved : {_out_csv('bh_confound_diagnostics')}")

        print("\n  Seasonal composition [%]:")
        print(season_df.to_string(index=False))
        season_df.round(2).to_csv(_out_csv("bh_season_composition"), index=False)
        print(f"  Saved : {_out_csv('bh_season_composition')}")

        plot_df = diag[diag["n"] >= MIN_BIN_POINTS]
        print(f"\n  Directional correlations across bins (n={len(plot_df)} bins, "
              f"indicative only — too few points for formal significance):")
        for col in ["n_buildings_nearby_mean", "max_building_height_mean",
                    "kc_clear_mean", "sun_elev_mean"]:
            r = plot_df["measured_loss_pct"].corr(plot_df[col])
            print(f"    corr(measured_loss_pct, {col:28s}) = {r:+.2f}")

        plot_fig8_confound_panels(diag, n_total, n_excluded_bldg, _out_png("bh_fig8_confound_panels"))
        plot_fig9_season_composition(season_df, n_total, n_excluded_bldg, _out_png("bh_fig9_confound_season"))
    else:
        print("\n[6/7] Skipping confound diagnostics (RUN_CONFOUND_DIAGNOSTICS=False).")

    if RUN_SUNELEV_MECHANISM_DIAGNOSTIC:
        print("\n[7/7] Figures 10-12 — solar-elevation mechanism diagnostic "
              "(RUN_SUNELEV_MECHANISM_DIAGNOSTIC=True)...")
        df_mech = add_hour_column(df)
        df_mech = assign_sunelev_bands(df_mech)

        plot_bins = [
            lbl for lbl in HEIGHT_BIN_LABELS
            if (df_mech["height_bin"] == lbl).sum() >= MIN_BIN_POINTS
        ]

        plot_fig10_sunelev_distribution(
            df_mech, plot_bins, n_total, n_excluded_bldg,
            _out_png("bh_fig10_sunelev_distribution"),
        )

        hour_df = compute_hour_composition(df_mech)
        print("\n  Hour-of-day composition by height bin [%]:")
        print(hour_df.to_string(index=False))
        hour_df.round(2).to_csv(_out_csv("bh_hour_composition"), index=False)
        print(f"  Saved : {_out_csv('bh_hour_composition')}")
        plot_fig11_hour_composition(
            hour_df, n_total, n_excluded_bldg, _out_png("bh_fig11_hour_composition"),
        )

        cell_df = compute_loss_by_height_and_sunelev(df_mech)
        pivot_loss = (cell_df.pivot(index="height_bin", columns="sunelev_band", values="loss_pct")
                      .reindex(index=HEIGHT_BIN_LABELS, columns=SUNELEV_BAND_LABELS))
        pivot_n = (cell_df.pivot(index="height_bin", columns="sunelev_band", values="n")
                   .reindex(index=HEIGHT_BIN_LABELS, columns=SUNELEV_BAND_LABELS))
        print("\n  Energy-weighted loss_pct, by height bin x solar-elevation band:")
        print(pivot_loss.to_string())
        print("\n  Cell sample sizes (n):")
        print(pivot_n.to_string())
        spread = pivot_loss.max(axis=0) - pivot_loss.min(axis=0)
        print("\n  Spread across height bins per solar-elevation band (max - min loss_pct):")
        print(spread.to_string())
        cell_df.round(3).to_csv(_out_csv("bh_loss_by_height_and_sunelev"), index=False)
        print(f"  Saved : {_out_csv('bh_loss_by_height_and_sunelev')}")

        plot_fig12_loss_vs_sunelev(
            cell_df, plot_bins, n_total, n_excluded_bldg,
            _out_png("bh_fig12_loss_vs_sunelev"),
        )
    else:
        print("\n[7/7] Skipping solar-elevation mechanism diagnostic "
              "(RUN_SUNELEV_MECHANISM_DIAGNOSTIC=False).")

    print("\nDone.")


if __name__ == "__main__":
    main()