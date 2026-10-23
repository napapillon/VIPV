"""
================================================================================
Real (Data-Driven) H/W Ratio Analysis — Urban Canyon vs Measured Losses
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Empirical counterpart of hw_ratio_impact_analysis.py (theoretical). Computes
a REAL per-point urban-canyon H/W ratio for every measurement:

  H — mean nearby building height (mean_building_height_m, 50 m buffer).
  W — real street width (street_width_m, width_source, edge_match_dist_m).

Six analyses:

  1 — Real H/W vs measured autonomy loss, binned (same formula as
      compute_bin_stats() of building_height_vs_losses.py), with the
      theoretical Brito diffuse-loss curve as a backdrop.
  2 — Measured per-bin loss vs theoretical total loss (Brito model at the
      mean H/W of each bin); a 1:1 line shows agreement.
  3 — Spatial map: H/W and relative autonomy loss side by side, over an OSM
      basemap (contextily).
  4 — [RUN_CONTINUOUS_SVF_ANALYSIS] Continuous per-point theoretical SVF
      loss, quantile-binned, with a linear regression (slope / R² / p).
      Analysis 4b: H/W composition (mean + p10-p90) per SVF bin.
  5-6 — [RUN_KC_CLEAR_STRATIFICATION, requires 4] SVF correlation stratified
      by kc_clear = ghi / ghi_clear (terciles), on the same global SVF bin
      edges. Analysis 6 adds a point-level OLS interaction regression
      (loss_pct_point ~ svf_c * kc_c).
      Caveat: M5 of hw_ratio_impact_analysis.py shows that clear-sky loss
      can EXCEED overcast loss at high H/W (direct blockage dominates) —
      read a stratum difference against that, not against a "diffuse-only"
      intuition.

Method
------
  - Loss metrics: same energy-weighted ratio of sums as the rest of the
    pipeline, loss_pct = Σ loss_auto_km / (Σ auto_solar_km + Σ loss_auto_km)
    × 100, with loss_irr = max(0, GHI − rad_calib).
  - Points with no building within 50 m (n_buildings_nearby == 0) are
    excluded.
  - W = street_width_m from the street-width join (sw_mapping,
    join_sw_standard.py / join_sw_tuktuk.py) — no assumed constants.
  - Points with unusable width are EXCLUDED, not forced to a value:
    width_source == 'unmatched_too_far', or street_width_m outside
    [W_MIN_M, W_MAX_M]. Every exclusion reason is counted and reported in
    the figure footnotes.
  - The Brito et al. (2024) model curves come from the shared module
    brito_svf_model (also used by hw_ratio_impact_analysis.py and
    building_height_vs_losses.py).

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : clean_data.py outputs (or a filter / time-range subset), one
             file per sensor, selected by FILE_SUFFIX / TUKTUK_FILE_SUFFIX
             standard : <sensor_id>_satellite_landuse_v2_<suffix>.csv
             tuk-tuk  : <sensor_id>_tuktuk_ghi_cams_cos_clc_bh_sw_norm_<suffix>.csv
             columns: 'Human readable date (Europe/London)', '<SENSOR> lat',
             '<SENSOR> lon', '<SENSOR> Rad calib', 'ghi', 'ghi_clear'
             (optional, Analyses 5-6), 'sun_elev', 'mean_building_height_m',
             'max_building_height_m', 'n_buildings_nearby',
             'street_width_m', 'width_source', 'edge_match_dist_m'
    output : OUTPUT_ROOT / 6_urban_geometry / hw_ratio_real_outputs /
             <CAMPAIGN> / <variant_tag> /
             hw_real_a1_correlation, hw_real_a2_validation,
             hw_real_a3_spatial_map, hw_real_a4_continuous_svf_correlation,
             hw_real_a4b_hw_composition, hw_real_a5_kc_clear_stratified,
             hw_real_a6_interaction_effect  (_<CAMPAIGN>_<variant_tag>.png)

Requirements
------------
    pip install pandas numpy matplotlib pyproj scipy statsmodels contextily

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 4.2
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
from pyproj import Transformer
from scipy import stats
import statsmodels.formula.api as smf

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from brito_svf_model import diffuse_loss_pct, annual_mean_direct_loss_pct_ew


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

# --- Base output directory ----------------------------------------------------
_OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "hw_ratio_real_outputs"

# --- Physical / vehicle parameters (identical to the rest of the pipeline) ----
CVE_KWH_PER_100KM: float = 16.6
CVE_KWH_PER_KM:    float = CVE_KWH_PER_100KM / 100.0
DELTA_T_H:         float = 1.0 / 60.0

# --- Street-width plausibility filter -----------------------------------------
W_MIN_M: float = 3.0
W_MAX_M: float = 60.0

# --- H/W binning (for Analyses 1 & 2) ----------------------------------------
HW_BIN_EDGES:  list[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0]
HW_BIN_LABELS: list[str] = [
    "0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0",
    "1.0–1.5", "1.5–2.0", "2.0–3.0", "3.0+",
]
MIN_BIN_POINTS: int = 60

# --- Theory backdrop parameters (Analysis 1 & 2) -----------------------------
DIFFUSE_FRACTION_THEORY: float = 0.45
LATITUDE_DEG: float = 38.76

# --- Optional sections --------------------------------------------------------
RUN_CONTINUOUS_SVF_ANALYSIS: bool = True   # Analysis 4 (quantile-binned regression + H/W composition)
RUN_KC_CLEAR_STRATIFICATION: bool = True   # Analysis 5-6 (sky-condition stratification + interaction)
                                            # requires RUN_CONTINUOUS_SVF_ANALYSIS = True

# --- Continuous SVF correlation (Analysis 4) ----------------------------------
SVF_N_BINS: int = 8       # quantile bins of the per-point theoretical SVF loss
CHUNK_SIZE: int = 10_000  # rows per chunk when evaluating diffuse_loss_pct
                          # (bounds peak memory: the trapezoidal integral inside
                          # brito_svf_model creates an N x n_samples temporary
                          # array, fine for a chunk but not for a full point cloud)
COLOR_RAW_SCATTER:   str = "#1565C0"
COLOR_BINNED_TREND:  str = "#F4A800"
COLOR_REGRESSION:    str = "#555555"

# --- Sky-condition stratification (Analysis 5-6) ------------------------------
# Same kc_clear convention as sal_kc_elevation_diagnostic.py / sal_weather_
# imbalance_diagnostic.py: kc_clear = ghi / ghi_clear, clipped to [0, 1.3].
KC_TERCILE_LABELS: list[str] = ["Low kc (cloudy)", "Mid kc", "High kc (clear)"]
KC_TERCILE_COLORS: dict[str, str] = {
    "Low kc (cloudy)": "#90A4AE", "Mid kc": "#FFB300", "High kc (clear)": "#E05C5C",
}

# --- Figure conventions -------------------------------------------------------
SHOW_FIGURES: bool = False
OUTPUT_DPI:   int  = 300
COLOR_MEASURED_BAR: str = "#E05C5C"
COLOR_THEORY:       str = "#212121"
CMAP_HW:    str = "viridis"
CMAP_LOSS:  str = "YlOrRd"


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


SENSOR_FILES: dict[str, Path] = _build_sensor_files(
    CAMPAIGN, DATA_DIR, TUKTUK_DATA_DIR, _variant_tag()
)
_VARIANT_TAG: str = _variant_tag()
OUTPUT_DIR: Path = _OUTPUT_BASE_DIR / CAMPAIGN / _VARIANT_TAG


def _out_png(name: str) -> Path:
    """Return the PNG output path of a figure, embedding campaign + variant tag."""
    return OUTPUT_DIR / f"{name}_{CAMPAIGN}_{_VARIANT_TAG}.png"


# ==============================================================================
# WGS-84 -> WEB MERCATOR  (Analysis 3 basemap)
# ==============================================================================

def wgs84_to_webmercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproject WGS-84 (EPSG:4326) lon/lat arrays to Web Mercator (EPSG:3857)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return x, y


# ==============================================================================
# 1.  LOAD + PREPARE MEASUREMENT DATA
# ==============================================================================

def load_combined_data(sensor_files: dict[str, Path]) -> tuple[pd.DataFrame, int]:
    """
    Load, consolidate, and enrich per-sensor CSV files.

    cos23/CLC/BH/SW join columns already coexist in the same per-sensor file
    (added upstream by join_cos_standard.py / join_clc_standard.py /
    join_bh_standard.py / join_sw_standard.py, or their tuk-tuk equivalents,
    before clean_data.py) — no cross-file alignment is needed, and W
    (street_width_m) is read as a plain column exactly like H
    (mean_building_height_m).

    Returns
    -------
    df : pd.DataFrame
        Valid measurement rows with lat/lon, rad_calib, ghi, building
        height, street width, and the per-point autonomy/loss columns.
    n_excluded_bldg : int
        Number of rows dropped for having no building within the 50 m buffer.
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
            "ghi", "ghi_clear", "sun_elev", "mean_building_height_m",
            "max_building_height_m", "n_buildings_nearby",
            "street_width_m", "width_source", "edge_match_dist_m",
        ]
        df = pd.read_csv(csv_path, low_memory=False, usecols=lambda c: c in needed)
        # ghi_clear is optional (only used by the kc_clear stratification,
        # Analysis 5-6); every other column is mandatory.
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
        for col in ("lat", "lon", "rad_calib", "ghi", "ghi_clear", "sun_elev",
                    "mean_building_height_m", "max_building_height_m",
                    "n_buildings_nearby", "street_width_m", "edge_match_dist_m"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["lat", "lon", "rad_calib", "ghi", "sun_elev"])
        df["sensor"] = sensor
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No sensor CSV files could be loaded. Check DATA_DIR/TUKTUK_DATA_DIR, "
            "CAMPAIGN, and FILE_SUFFIX/TUKTUK_FILE_SUFFIX. Note this script "
            "requires BOTH the BH join (join_bh_standard.py / join_bh_tuktuk.py) "
            "AND the SW join (join_sw_standard.py / join_sw_tuktuk.py) columns "
            "to already be present."
        )

    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(df):,} valid rows loaded across {df['sensor'].nunique()} sensor(s)")

    n_before_bldg = len(df)
    df = df[df["n_buildings_nearby"] > 0].copy()
    n_excluded_bldg = n_before_bldg - len(df)
    df = df.reset_index(drop=True)
    print(f"  {len(df):,} rows with >=1 building within 50 m "
          f"({n_excluded_bldg:,} excluded — no building nearby)")

    diff_w = df["ghi"] - df["rad_calib"]
    df["e_solar_kWh"]   = df["rad_calib"] / 1000.0 * DELTA_T_H
    df["loss_irr_kWh"]  = (diff_w / 1000.0 * DELTA_T_H).clip(lower=0.0)
    df["auto_solar_km"] = df["e_solar_kWh"] / CVE_KWH_PER_KM
    df["loss_auto_km"]  = df["loss_irr_kWh"] / CVE_KWH_PER_KM
    denom = df["auto_solar_km"] + df["loss_auto_km"]
    df["loss_pct_point"] = np.where(denom > 0,
                                    df["loss_auto_km"] / denom * 100.0, np.nan)
    return df, n_excluded_bldg


# ==============================================================================
# 2.  H/W RATIO  (W = street_width_m, already joined upstream)
# ==============================================================================

def add_hw_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute H/W = mean_building_height_m / street_width_m and flag the
    reason for exclusion when a point does not yield a usable ratio.

    Adds:
        hw_ratio      — H/W [-], NaN when excluded
        width_status  — 'ok', 'unmatched_too_far' (from the upstream SW
                         join — no OSM edge within its MAX_MATCH_DISTANCE_M),
                         or 'implausible_W' (street_width_m outside
                         [W_MIN_M, W_MAX_M] — a local sanity check on top
                         of the already-validated upstream value).
    """
    df = df.copy()
    df["hw_ratio"] = np.nan
    df["width_status"] = "ok"

    unmatched = df["street_width_m"].isna()
    if "width_source" in df.columns:
        unmatched = unmatched | (df["width_source"] == "unmatched_too_far")
    df.loc[unmatched, "width_status"] = "unmatched_too_far"

    implausible = (~unmatched) & (
        (df["street_width_m"] < W_MIN_M) | (df["street_width_m"] > W_MAX_M)
    )
    df.loc[implausible, "width_status"] = "implausible_W"

    valid = df["width_status"] == "ok"
    df.loc[valid, "hw_ratio"] = (
        df.loc[valid, "mean_building_height_m"] / df.loc[valid, "street_width_m"]
    )
    return df


def assign_hw_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add the hw_bin column (HW_BIN_EDGES, left-closed bins)."""
    df = df.copy()
    df["hw_bin"] = pd.cut(
        df["hw_ratio"], bins=HW_BIN_EDGES, labels=HW_BIN_LABELS, right=False,
    )
    return df


def compute_bin_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Energy-weighted relative loss per H/W bin — identical formula to
    building_height_vs_losses.py's compute_bin_stats():

        loss_pct = sum(loss_auto_km) / (sum(auto_solar_km) + sum(loss_auto_km)) x 100
    """
    rows = []
    for label in HW_BIN_LABELS:
        sub = df[df["hw_bin"] == label]
        n = len(sub)
        if n == 0:
            rows.append(dict(hw_bin=label, n=0, mean_hw=np.nan,
                             loss_pct=np.nan, loss_pct_point_mean=np.nan))
            continue
        auto_sum = sub["auto_solar_km"].sum()
        loss_sum = sub["loss_auto_km"].sum()
        denom = auto_sum + loss_sum
        rows.append(dict(
            hw_bin=label,
            n=n,
            mean_hw=sub["hw_ratio"].mean(),
            loss_pct=(loss_sum / denom * 100.0) if denom > 0 else np.nan,
            loss_pct_point_mean=sub["loss_pct_point"].mean(),
        ))
    return pd.DataFrame(rows)


# ==============================================================================
# 3.  CONTINUOUS SVF CORRELATION  (RUN_CONTINUOUS_SVF_ANALYSIS — Analysis 4)
# ==============================================================================
# Analyses 1-2 bin points by H/W into FIXED-WIDTH bins (HW_BIN_EDGES), which
# necessarily mixes points with very different real H/W ratios within the
# same bin (a tall building on a wide avenue vs. a short building on a
# narrow alley can land in the same fixed bin despite very different Sky
# View Factors). This section removes that binning artefact by computing
# each point's THEORETICAL SVF diffuse loss directly from its own real H
# and W (Brito et al. 2024 Eq. 2, via brito_svf_model.diffuse_loss_pct),
# then grouping points into QUANTILE bins of that continuous value instead
# — quantile bins have no natural "metres" edges the way H/W does, so equal
# population per bin is the natural choice. A linear regression across bins
# (measured vs. theoretical) reports slope / R² / p-value as the main
# quantitative result, complementing Analyses 1-2's fixed-bin comparison.

def compute_point_theoretical_svf_loss(
    hw_values: np.ndarray, chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """
    Point-level evaluation of diffuse_loss_pct(hw), processed in chunks.

    brito_svf_model's diffuse_loss_pct broadcasts each point against ~2001
    azimuth samples internally (fine for a handful of bins, but an
    N x 2001 array for a full point cloud of N in the hundreds of thousands
    would be a multi-GB temporary array) — chunking bounds peak memory to
    chunk_size x 2001 without approximating or altering the formula itself.
    """
    hw_values = np.asarray(hw_values, dtype=float)
    out = np.full(len(hw_values), np.nan)
    for start in range(0, len(hw_values), chunk_size):
        end = start + chunk_size
        chunk = hw_values[start:end]
        valid = ~np.isnan(chunk)
        result = np.full(len(chunk), np.nan)
        if valid.any():
            result[valid] = diffuse_loss_pct(chunk[valid], edge=False)
        out[start:end] = result
    return out


def assign_svf_quantile_bins(df: pd.DataFrame, n_bins: int = SVF_N_BINS) -> pd.DataFrame:
    """Add the svf_bin column: quantile bins of the per-point theoretical SVF loss."""
    df = df.copy()
    df["svf_bin"] = pd.qcut(
        df["theoretical_svf_loss_pct"], q=n_bins, duplicates="drop",
    )
    return df


def compute_svf_bin_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Energy-weighted loss per SVF quantile bin — same formula as
    compute_bin_loss(), grouped by the continuous 'svf_bin' column instead
    of the fixed-width 'hw_bin'."""
    rows = []
    for interval, sub in df.groupby("svf_bin", observed=True):
        n = len(sub)
        auto_sum = sub["auto_solar_km"].sum()
        loss_sum = sub["loss_auto_km"].sum()
        denom = auto_sum + loss_sum
        measured_loss_pct = (loss_sum / denom * 100.0) if denom > 0 else np.nan
        rows.append(dict(
            svf_bin=str(interval), n=n,
            mean_theoretical_pct=sub["theoretical_svf_loss_pct"].mean(),
            mean_H=sub["mean_building_height_m"].mean(),
            mean_W=sub["street_width_m"].mean(),
            measured_loss_pct=measured_loss_pct,
            mean_of_ratios_pct=sub["loss_pct_point"].mean(),
        ))
    return pd.DataFrame(rows).sort_values("mean_theoretical_pct").reset_index(drop=True)


def compute_hw_bin_composition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per SVF quantile bin: mean + p10-p90 spread of H and W separately, to
    check whether the SVF-bin ordering is driven by height, width, or
    both — compute_svf_bin_stats() already reports mean_H/mean_W but with
    no spread, which hides whether a bin is a fairly homogeneous street
    type or a mix of very different (H, W) pairs landing on a similar
    theoretical value.
    """
    rows = []
    for interval, sub in df.groupby("svf_bin", observed=True):
        rows.append(dict(
            svf_bin=str(interval), n=len(sub),
            mean_theoretical_pct=sub["theoretical_svf_loss_pct"].mean(),
            mean_H=sub["mean_building_height_m"].mean(),
            p10_H=sub["mean_building_height_m"].quantile(0.10),
            p90_H=sub["mean_building_height_m"].quantile(0.90),
            mean_W=sub["street_width_m"].mean(),
            p10_W=sub["street_width_m"].quantile(0.10),
            p90_W=sub["street_width_m"].quantile(0.90),
        ))
    return pd.DataFrame(rows).sort_values("mean_theoretical_pct").reset_index(drop=True)


# ==============================================================================
# 4.  SKY-CONDITION STRATIFICATION  (RUN_KC_CLEAR_STRATIFICATION — Analysis 5-6)
# ==============================================================================
# Stratifies the continuous SVF correlation (Analysis 4) by the clear-sky
# index kc_clear = ghi / ghi_clear, to test whether measured loss tracks the
# theoretical isotropic-diffuse SVF model more cleanly under overcast
# conditions (diffuse-dominant, independent of sun position) than under
# clear sky (direct-beam-dominant, sun-position-dependent).
#
# Caveat (read before concluding "SVF confirmed" from a cleaner cloudy-sky
# fit): hw_ratio_impact_analysis.py's own theoretical model (M5) shows that
# at high H/W, clear-sky loss can EXCEED overcast loss (direct blockage
# dominates) — so "cloudy = always the cleaner SVF signal" is not
# guaranteed to hold uniformly across the full SVF range; a stratum
# difference should be read against that theoretical expectation, not
# against a flat "diffuse-only" intuition.
#
# Bin edges for the theoretical SVF value are computed ONCE on the full
# (unstratified) dataset (so all three kc_clear strata are compared on the
# SAME geometry ranges), then each stratum is aggregated within those fixed
# bins using the identical energy-weighted loss formula used everywhere
# else in this pipeline.

def add_kc_clear(df: pd.DataFrame) -> pd.DataFrame:
    """kc_clear = ghi / ghi_clear, clipped to [0, 1.3] — identical
    convention to sal_kc_elevation_diagnostic.py / sal_weather_imbalance_
    diagnostic.py."""
    df = df.copy()
    df["kc_clear"] = np.where(df["ghi_clear"] > 0, df["ghi"] / df["ghi_clear"], np.nan)
    df["kc_clear"] = df["kc_clear"].clip(lower=0, upper=1.3)
    return df


def compute_global_svf_bin_edges(df: pd.DataFrame, n_bins: int = SVF_N_BINS) -> np.ndarray:
    """Bin edges computed ONCE on the full dataset, so every kc_clear
    stratum is later aggregated on the SAME theoretical-SVF ranges —
    otherwise a per-stratum pd.qcut would silently compare different
    geometry ranges across panels."""
    _, edges = pd.qcut(df["theoretical_svf_loss_pct"], q=n_bins, retbins=True, duplicates="drop")
    return edges


def assign_fixed_svf_bins(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Add the svf_bin_fixed column, binned on the given (global) edges."""
    df = df.copy()
    df["svf_bin_fixed"] = pd.cut(df["theoretical_svf_loss_pct"], bins=edges, include_lowest=True)
    return df


def compute_bin_stats_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Same energy-weighted formula as compute_svf_bin_stats(), applied to
    a fixed-edge 'svf_bin_fixed' column (reproduced here rather than
    shared, so an empty per-stratum subset never raises inside a shared
    groupby)."""
    rows = []
    for interval, sub in df.groupby("svf_bin_fixed", observed=True):
        n = len(sub)
        if n == 0:
            continue
        auto_sum = sub["auto_solar_km"].sum()
        loss_sum = sub["loss_auto_km"].sum()
        denom = auto_sum + loss_sum
        measured_loss_pct = (loss_sum / denom * 100.0) if denom > 0 else np.nan
        rows.append(dict(
            svf_bin=str(interval), n=n,
            mean_theoretical_pct=sub["theoretical_svf_loss_pct"].mean(),
            measured_loss_pct=measured_loss_pct,
        ))
    return pd.DataFrame(rows).sort_values("mean_theoretical_pct").reset_index(drop=True)


def fit_svf_kc_interaction(df: pd.DataFrame):
    """
    Fit loss_pct_point ~ svf_c * kc_c (mean-centred), a point-level OLS
    interaction regression testing whether the SVF slope depends on sky
    condition — directly, at the point level (tens of thousands of rows),
    complementing the bin-level per-stratum regressions (Analysis 5, which
    only has SVF_N_BINS points per stratum and therefore very low
    statistical power).

    Both continuous predictors are mean-centred before building the
    interaction term (standard practice — keeps the main-effect
    coefficients interpretable as "effect at the sample mean of the other
    variable" rather than at the arbitrary zero of the raw scale, and
    avoids the collinearity an uncentred product term would add).

    Reading the interaction coefficient
    --------------------------------------
    A significant POSITIVE interaction means the SVF slope gets STEEPER as
    kc_clear increases (clear sky) — consistent with hw_ratio_impact_
    analysis.py's own theoretical result that direct-beam blockage adds on
    top of the diffuse loss under clear sky, especially at high H/W. A
    significant NEGATIVE interaction would mean the SVF slope is steeper
    under cloudy conditions — the naive "isotropic diffuse only"
    expectation. A non-significant interaction means this dataset cannot
    distinguish the two regimes' slopes at this sample size.

    Caveat: this uses loss_pct_point (per-point, equal-weight ratio), the
    same secondary/sensitivity statistic used elsewhere in this pipeline —
    not the primary energy-weighted metric, because OLS needs one
    dependent value per row. The bin-level, energy-weighted figure
    (Analysis 5) remains the primary result; this regression is a higher-
    power SIGNIFICANCE TEST of the slope-interaction question, not a
    replacement magnitude estimate.

    Returns
    -------
    (fitted_model, svf_mean, kc_mean)
    """
    d = df.dropna(subset=["loss_pct_point", "theoretical_svf_loss_pct", "kc_clear"]).copy()
    svf_mean = d["theoretical_svf_loss_pct"].mean()
    kc_mean = d["kc_clear"].mean()
    d["svf_c"] = d["theoretical_svf_loss_pct"] - svf_mean
    d["kc_c"] = d["kc_clear"] - kc_mean
    model = smf.ols("loss_pct_point ~ svf_c * kc_c", data=d).fit()
    return model, svf_mean, kc_mean


def print_interaction_verdict(model) -> None:
    """Print the svf_c:kc_c interaction coefficient, its CI and p-value, with a verdict."""
    coef = model.params.get("svf_c:kc_c", np.nan)
    pval = model.pvalues.get("svf_c:kc_c", np.nan)
    ci_low, ci_high = model.conf_int().loc["svf_c:kc_c"]
    print(f"\nInteraction term (svf_c:kc_c): coef={coef:.4f}, "
          f"95% CI=[{ci_low:.4f}, {ci_high:.4f}], p={pval:.4g}")
    if pval < 0.05:
        direction = ("STEEPER under clear sky (positive)" if coef > 0
                     else "STEEPER under cloudy sky (negative)")
        print(f"  -> Significant: the SVF slope is {direction}.")
    else:
        print("  -> Not significant at this sample size: cannot distinguish "
              "the two regimes' slopes.")


def print_width_diagnostics(df: pd.DataFrame) -> None:
    """Console summary of the H/W distribution and exclusion breakdown."""
    print("\n" + "─" * 72)
    print("  PER-POINT WIDTH / H-W DIAGNOSTICS")
    print("─" * 72)
    total = len(df)
    print(f"  Total points processed        : {total:,}")
    for status, grp in df.groupby("width_status"):
        print(f"    {status:<18}: {len(grp):>7,}  ({len(grp) / total * 100:5.1f}%)")

    ok = df[df["width_status"] == "ok"]
    if not ok.empty:
        print(f"\n  Valid-canyon points           : {len(ok):,}")
        print(f"  Street width W [m]   — median : {ok['street_width_m'].median():.1f}  "
              f"(IQR {ok['street_width_m'].quantile(.25):.1f}–{ok['street_width_m'].quantile(.75):.1f})")
        print(f"  Building height H [m] — median: {ok['mean_building_height_m'].median():.1f}")
        print(f"  H/W ratio            — median : {ok['hw_ratio'].median():.2f}  "
              f"(IQR {ok['hw_ratio'].quantile(.25):.2f}–{ok['hw_ratio'].quantile(.75):.2f})")
        if "width_source" in ok.columns:
            print("\n  Street-width source (valid points, from the street-width join):")
            for src, grp in ok.groupby("width_source"):
                print(f"    {src:<15}: {len(grp):>7,}  "
                      f"({len(grp) / len(ok) * 100:5.1f}%)  "
                      f"median W={grp['street_width_m'].median():.1f} m")
    print("─" * 72)


def _exclusion_summary(df: pd.DataFrame) -> str:
    """One-line breakdown of width_status counts for figure footnotes."""
    counts = df["width_status"].value_counts()
    total = len(df)
    parts = []
    for status in ["ok", "unmatched_too_far", "implausible_W"]:
        c = int(counts.get(status, 0))
        if c > 0:
            parts.append(f"{status}={c:,} ({c / total * 100:.0f}%)")
    return "  |  ".join(parts)


# ==============================================================================
# 3.  THEORY BACKDROP  (brito_svf_model.py — shared, always available)
# ==============================================================================

def _theory_curves(hw_values: np.ndarray) -> dict[str, np.ndarray]:
    """
    Diffuse + total theoretical loss backdrop for Analysis 1/2, from the
    shared brito_svf_model.py module.
    """
    lat_rad = np.radians(LATITUDE_DEG)
    diffuse = diffuse_loss_pct(hw_values, edge=False)
    bhi = annual_mean_direct_loss_pct_ew(hw_values, lat_rad, edge=False, n_doy=24)
    total = (DIFFUSE_FRACTION_THEORY * diffuse
             + (1.0 - DIFFUSE_FRACTION_THEORY) * bhi)
    return {"diffuse": diffuse, "total": total}


# ==============================================================================
# 4.  FIGURES
# ==============================================================================

def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    """Save the figure at OUTPUT_DPI, then show or close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    print(f"  Figure saved -> {out_path}  ({OUTPUT_DPI} dpi)")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def fig_analysis1_correlation(df: pd.DataFrame, agg: pd.DataFrame,
                               out_path: Path) -> None:
    """ANALYSIS 1 — Measured relative loss vs. real H/W bin (energy-weighted)."""
    plot_df = agg[(agg["n"] >= MIN_BIN_POINTS) & agg["loss_pct"].notna()].copy()
    plot_df = plot_df.reset_index(drop=True)
    if plot_df.empty:
        print("  [WARN] Analysis 1: no bins meet the threshold; skipping.")
        return

    x = np.arange(len(plot_df))
    fig, ax1 = plt.subplots(figsize=(11, 6.2))

    ax1.bar(
        x, plot_df["loss_pct"].values, width=0.6,
        color=COLOR_MEASURED_BAR, alpha=0.85, edgecolor="white", linewidth=0.8,
        zorder=3, label="Measured relative loss (energy-weighted)",
    )
    for xi, (n, v) in enumerate(zip(plot_df["n"], plot_df["loss_pct"])):
        ax1.text(xi, v + 0.8, f"{v:.1f}%\n(n={int(n):,})", ha="center",
                 va="bottom", fontsize=7.5, color="#555555")

    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["hw_bin"], fontsize=9.5)
    ax1.set_xlabel("Real urban-canyon H/W ratio  (street_width_m, OSM-derived)",
                   fontsize=11, fontweight="bold")
    ax1.set_ylabel("Measured relative autonomy loss  [%]",
                   fontsize=11, fontweight="bold", color=COLOR_MEASURED_BAR)
    ax1.tick_params(axis="y", labelcolor=COLOR_MEASURED_BAR)
    ax1.set_ylim(0, max(60.0, plot_df["loss_pct"].max() * 1.3))
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines[["top"]].set_visible(False)

    theory = _theory_curves(plot_df["mean_hw"].values)
    ax2 = ax1.twinx()
    ax2.plot(x, theory["diffuse"], color=COLOR_THEORY, linewidth=1.8,
             marker="o", markersize=5, linestyle="--",
             label="Theoretical diffuse loss (Brito, backdrop)")
    ax2.plot(x, theory["total"], color="#7B1FA2", linewidth=1.8,
             marker="s", markersize=5, linestyle=":",
             label=f"Theoretical total loss (f_d={DIFFUSE_FRACTION_THEORY})")
    ax2.set_ylabel("Theoretical loss  [%]  (backdrop, not a fit)",
                   fontsize=10, color="#424242")
    ax2.tick_params(axis="y", labelcolor="#424242")
    ax2.set_ylim(0, 100)
    ax2.spines[["top"]].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)

    ax1.set_title(
        "Analysis 1 — Measured Autonomy Loss vs. Real Canyon H/W Ratio",
        fontsize=12.5, fontweight="bold", pad=12,
    )
    n_ok = int((df["width_status"] == "ok").sum())
    fig.text(
        0.5, -0.04,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  Real H/W = mean building height (50 m) / "
        f"street_width_m (street-width join)  —  {n_ok:,} points with a "
        f"valid canyon  —  bin threshold >= {MIN_BIN_POINTS} pts  —  "
        f"loss_pct = loss_auto / (auto_solar + loss_auto) × 100\n"
        f"Exclusions: {_exclusion_summary(df)}",
        ha="center", fontsize=7, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_analysis2_validation(df: pd.DataFrame, agg: pd.DataFrame,
                             out_path: Path) -> None:
    """ANALYSIS 2 — Measured per-bin loss vs. theoretical total loss (1:1)."""
    plot_df = agg[(agg["n"] >= MIN_BIN_POINTS) & agg["loss_pct"].notna()].copy()
    plot_df = plot_df.reset_index(drop=True)
    if plot_df.empty:
        print("  [WARN] Analysis 2: no bins meet the threshold; skipping.")
        return

    theory = _theory_curves(plot_df["mean_hw"].values)
    measured = plot_df["loss_pct"].values
    predicted = theory["total"]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    sc = ax.scatter(predicted, measured, s=90, c=plot_df["mean_hw"].values,
                    cmap=CMAP_HW, edgecolor="#333333", linewidth=0.7, zorder=4)
    for xi, yi, lbl in zip(predicted, measured, plot_df["hw_bin"]):
        ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color="#333333")

    lim = max(predicted.max(), measured.max()) * 1.1
    ax.plot([0, lim], [0, lim], color="#999999", linestyle="--",
            linewidth=1.2, label="1:1 (perfect agreement)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"Theoretical total loss  [%]  (Brito, f_d={DIFFUSE_FRACTION_THEORY})",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("Measured relative autonomy loss  [%]",
                  fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Bin mean H/W", fontsize=9)
    ax.set_title("Analysis 2 — Measured vs. Theoretical Loss\n"
                 "(each point = one H/W bin)",
                 fontsize=12.5, fontweight="bold")
    fig.text(
        0.5, -0.02,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  Points above the 1:1 line: measured loss "
        "exceeds theory (real streets shadier than the idealised E–W canyon, or "
        "extra cloud/movement effects). Below: theory over-predicts.",
        ha="center", fontsize=7, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_analysis3_spatial_map(df: pd.DataFrame, out_path: Path) -> None:
    """ANALYSIS 3 — Two-panel spatial map: H/W (left) and loss (right)."""
    valid = df[df["width_status"] == "ok"].copy()
    if valid.empty:
        print("  [WARN] Analysis 3: no valid-canyon points; skipping.")
        return

    x_merc, y_merc = wgs84_to_webmercator(valid["lon"].values, valid["lat"].values)

    try:
        import contextily as cx
        have_basemap = True
    except ImportError:
        have_basemap = False
        print("  [info] contextily not installed; drawing without basemap.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    hw_clip = np.clip(valid["hw_ratio"].values, 0, np.nanpercentile(valid["hw_ratio"], 98))
    pa = axes[0].scatter(x_merc, y_merc, c=hw_clip,
                         cmap=CMAP_HW, s=10, alpha=0.7, linewidths=0)
    axes[0].set_title("Real canyon H/W ratio", fontsize=12, fontweight="bold")
    cba = fig.colorbar(pa, ax=axes[0], shrink=0.8)
    cba.set_label("H/W (clipped at 98th pct)", fontsize=9)

    loss_clip = np.clip(valid["loss_pct_point"].values, 0,
                        np.nanpercentile(valid["loss_pct_point"], 98))
    pb = axes[1].scatter(x_merc, y_merc, c=loss_clip,
                         cmap=CMAP_LOSS, s=10, alpha=0.7, linewidths=0)
    axes[1].set_title("Relative autonomy loss [%]", fontsize=12, fontweight="bold")
    cbb = fig.colorbar(pb, ax=axes[1], shrink=0.8)
    cbb.set_label("Loss [%] (clipped at 98th pct)", fontsize=9)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        if have_basemap:
            try:
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                               crs="EPSG:3857", attribution_size=5)
            except Exception as exc:
                print(f"  [info] basemap fetch failed ({exc!s}); plain background.")

    fig.suptitle(
        "Analysis 3 — Spatial Distribution of Canyon H/W and Autonomy Loss",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    n_ok = len(valid)
    fig.text(
        0.5, 0.02,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  {n_ok:,} measurement points with a valid "
        f"street-width match  —  left: H/W ratio  —  right: relative autonomy loss  —  "
        f"compare where tight canyons (left) coincide with high losses (right).",
        ha="center", fontsize=8, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_analysis4_continuous_svf_correlation(
    df: pd.DataFrame, agg: pd.DataFrame, out_path: Path,
) -> None:
    """ANALYSIS 4 — Continuous per-point SVF vs. measured loss, quantile bins + regression."""
    valid_bins = agg[agg["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    if len(valid_bins) < 2:
        print("  [WARN] Analysis 4: fewer than 2 bins have enough points; skipping.")
        return

    reg = stats.linregress(valid_bins["mean_theoretical_pct"], valid_bins["measured_loss_pct"])
    valid_pts = df.dropna(subset=["theoretical_svf_loss_pct", "loss_pct_point"])
    pearson_r_point, _ = (
        stats.pearsonr(valid_pts["theoretical_svf_loss_pct"], valid_pts["loss_pct_point"])
        if len(valid_pts) > 2 else (np.nan, np.nan)
    )

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(
        df["theoretical_svf_loss_pct"], df["loss_pct_point"],
        s=4, alpha=0.05, color=COLOR_RAW_SCATTER, linewidths=0,
        rasterized=True, label="Individual measurements",
    )
    ax.plot(
        valid_bins["mean_theoretical_pct"], valid_bins["measured_loss_pct"],
        color=COLOR_BINNED_TREND, linewidth=2.5, marker="o", markersize=7,
        zorder=5, label="Binned mean (energy-weighted)",
    )
    for _, row in valid_bins.iterrows():
        ax.annotate(f"n={int(row['n']):,}",
                    (row["mean_theoretical_pct"], row["measured_loss_pct"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color="#555555")

    x_fit = np.linspace(valid_bins["mean_theoretical_pct"].min(),
                         valid_bins["mean_theoretical_pct"].max(), 50)
    y_fit = reg.intercept + reg.slope * x_fit
    ax.plot(x_fit, y_fit, linestyle="--", linewidth=1.5, color=COLOR_REGRESSION,
            zorder=4, label=f"Linear fit across bins (R²={reg.rvalue**2:.2f})")

    ax.set_xlabel(
        "Theoretical isotropic-diffuse SVF loss [%]\n(real H & real W per point, Brito et al. 2024 Eq. 2)",
        fontsize=10, fontweight="bold",
    )
    ax.set_ylabel("Measured relative autonomy loss [%]\n(diffuse + direct, all-sky)",
                  fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Analysis 4 — Autonomy Loss vs. Continuous SVF Loss (Quantile Bins)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.text(
        0.5, 1.04,
        f"{SVF_N_BINS} quantile bins of the combined H/W metric — "
        "raw points are individual 1-minute measurements (noisy by construction)",
        transform=ax.transAxes, ha="center", fontsize=8, color="#555555",
    )
    fig.text(
        0.5, -0.05,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
        f"slope={reg.slope:.2f}, intercept={reg.intercept:.1f}, "
        f"R²={reg.rvalue**2:.2f}, p={reg.pvalue:.3f} (across bins)  —  "
        f"point-level Pearson r={pearson_r_point:.2f} (noisy, context only)  —  "
        f"theoretical curve is DIFFUSE-ONLY (shape comparison, not magnitude)",
        ha="center", fontsize=7, color="#888888", style="italic", wrap=True,
    )
    plt.tight_layout()
    _save_fig(fig, out_path)

    print("\nRegression across bins (measured_loss_pct ~ mean_theoretical_pct):")
    print(f"  slope     = {reg.slope:.3f}")
    print(f"  intercept = {reg.intercept:.3f}")
    print(f"  R^2       = {reg.rvalue**2:.3f}")
    print(f"  p-value   = {reg.pvalue:.4f}")
    print(f"\nPoint-level Pearson r (noisy, for context only): {pearson_r_point:.3f}")


def fig_analysis4b_hw_composition(comp: pd.DataFrame, out_path: Path) -> None:
    """ANALYSIS 4b — H vs. W composition (mean + p10-p90) per SVF quantile bin."""
    valid = comp[comp["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
    if len(valid) < 2:
        print("  [WARN] Analysis 4b: fewer than 2 bins have enough points; skipping.")
        return

    color_H, color_W = "#C0392B", "#2E7D32"
    x = np.arange(len(valid))

    # NOTE: with a highly concentrated distribution (e.g. many points sharing
    # the same 'highway_default' fallback width), the mean can sit outside
    # [p10, p90] (a skewed tail pulls the mean past a quantile that is itself
    # still at the mode) — clip to 0 so matplotlib's errorbar (which rejects
    # negative yerr) never raises on real data.
    err_H_lo = (valid["mean_H"] - valid["p10_H"]).clip(lower=0)
    err_H_hi = (valid["p90_H"] - valid["mean_H"]).clip(lower=0)
    err_W_lo = (valid["mean_W"] - valid["p10_W"]).clip(lower=0)
    err_W_hi = (valid["p90_W"] - valid["mean_W"]).clip(lower=0)

    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    ax2 = ax1.twinx()

    line_H = ax1.errorbar(
        x, valid["mean_H"],
        yerr=[err_H_lo, err_H_hi],
        color=color_H, marker="o", markersize=6, linewidth=2, capsize=4,
        label="Mean building height H (p10-p90)",
    )
    line_W = ax2.errorbar(
        x, valid["mean_W"],
        yerr=[err_W_lo, err_W_hi],
        color=color_W, marker="s", markersize=6, linewidth=2, capsize=4, linestyle="--",
        label="Mean street width W (p10-p90)",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"bin {i + 1}\n({v:.0f}%)" for i, v in enumerate(valid["mean_theoretical_pct"])],
        fontsize=9,
    )
    ax1.set_xlabel("SVF quantile bin (mean theoretical loss %)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean building height H [m]", fontsize=10, fontweight="bold", color=color_H)
    ax2.set_ylabel("Mean street width W [m]", fontsize=10, fontweight="bold", color=color_W)
    ax1.tick_params(axis="y", labelcolor=color_H)
    ax2.tick_params(axis="y", labelcolor=color_W)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    handles = [line_H, line_W]
    ax1.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="upper left")
    ax1.set_title("Analysis 4b — Building Height vs. Street Width Composition per SVF Bin",
                  fontsize=12, fontweight="bold", pad=20)
    fig.text(
        0.5, -0.04,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts  —  "
        "error bars show p10-p90 spread within each bin  —  "
        "shows whether H, W, or both drive the SVF-bin ordering",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_analysis5_kc_clear_stratified(
    strat_aggs: dict[str, pd.DataFrame], n_by_stratum: dict[str, int], out_path: Path,
) -> None:
    """ANALYSIS 5 — SVF correlation stratified by kc_clear tercile (sky condition)."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    reg_stats = {}
    for label in KC_TERCILE_LABELS:
        agg = strat_aggs[label]
        valid = agg[agg["n"] >= MIN_BIN_POINTS].reset_index(drop=True)
        if len(valid) < 2:
            print(f"  [WARNING] '{label}': fewer than 2 bins have enough points — skipped.")
            continue

        reg = stats.linregress(valid["mean_theoretical_pct"], valid["measured_loss_pct"])
        reg_stats[label] = reg
        color = KC_TERCILE_COLORS[label]

        ax.plot(
            valid["mean_theoretical_pct"], valid["measured_loss_pct"],
            color=color, linewidth=2.2, marker="o", markersize=6, zorder=5,
            label=f"{label}  (n={n_by_stratum[label]:,}, R²={reg.rvalue**2:.2f})",
        )
        x_fit = np.linspace(valid["mean_theoretical_pct"].min(),
                             valid["mean_theoretical_pct"].max(), 50)
        ax.plot(x_fit, reg.intercept + reg.slope * x_fit,
                color=color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=4)

    ax.set_xlabel(
        "Theoretical isotropic-diffuse SVF loss [%]\n(real H & real W per point, same bins across strata)",
        fontsize=10, fontweight="bold",
    )
    ax.set_ylabel("Measured relative autonomy loss [%]\n(energy-weighted per bin)",
                  fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Analysis 5 — Autonomy Loss vs. Theoretical SVF Loss, Stratified by Sky Condition",
                 fontsize=12, fontweight="bold", pad=20)
    ax.text(
        0.5, 1.04,
        "kc_clear terciles (same convention as sal_kc_elevation_diagnostic.py) — "
        "same SVF bin edges used across all three strata",
        transform=ax.transAxes, ha="center", fontsize=8, color="#555555",
    )

    footnote_parts = [f"[{CAMPAIGN} / {_VARIANT_TAG}]  bin threshold >= {MIN_BIN_POINTS} pts"]
    for label in KC_TERCILE_LABELS:
        if label in reg_stats:
            r = reg_stats[label]
            footnote_parts.append(f"{label}: slope={r.slope:.2f}, p={r.pvalue:.3f}")
    fig.text(0.5, -0.05, "  —  ".join(footnote_parts),
              ha="center", fontsize=7, color="#888888", style="italic", wrap=True)

    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_analysis6_interaction_effect(
    model, svf_mean: float, kc_mean: float, df: pd.DataFrame, out_path: Path,
) -> None:
    """ANALYSIS 6 — Predicted loss vs. theoretical SVF loss, at three
    representative kc_clear values (p10/p50/p90) — visualises the fitted
    SVF x sky-condition interaction directly, using the full point-level
    sample rather than the SVF_N_BINS-per-stratum aggregates."""
    kc_p10, kc_p50, kc_p90 = df["kc_clear"].quantile([0.10, 0.50, 0.90])
    svf_grid = np.linspace(
        df["theoretical_svf_loss_pct"].quantile(0.02),
        df["theoretical_svf_loss_pct"].quantile(0.98), 50,
    )

    fig, ax = plt.subplots(figsize=(9, 6.5))
    reps = [
        ("kc_clear p10 (cloudiest)", kc_p10, KC_TERCILE_COLORS["Low kc (cloudy)"]),
        ("kc_clear p50 (median)", kc_p50, KC_TERCILE_COLORS["Mid kc"]),
        ("kc_clear p90 (clearest)", kc_p90, KC_TERCILE_COLORS["High kc (clear)"]),
    ]
    for label, kc_val, color in reps:
        pred_df = pd.DataFrame({"svf_c": svf_grid - svf_mean, "kc_c": kc_val - kc_mean})
        preds = model.predict(pred_df)
        ax.plot(svf_grid, preds, color=color, linewidth=2.4,
                label=f"{label} (kc={kc_val:.2f})")

    ax.set_xlabel("Theoretical isotropic-diffuse SVF loss [%]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted per-point relative loss [%]\n(OLS interaction fit)",
                  fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Analysis 6 — Fitted SVF x Sky-Condition Interaction",
                 fontsize=12, fontweight="bold", pad=20)

    coef = model.params.get("svf_c:kc_c", np.nan)
    pval = model.pvalues.get("svf_c:kc_c", np.nan)
    fig.text(
        0.5, -0.04,
        f"[{CAMPAIGN} / {_VARIANT_TAG}]  loss_pct_point ~ svf_c * kc_c (mean-centred)  —  "
        f"interaction coef={coef:.4f}, p={pval:.4g}  —  "
        f"n={int(model.nobs):,}  —  R²={model.rsquared:.3f}",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Load the data, compute H/W and run Analyses 1–6."""
    print("=" * 78)
    print("  REAL H/W RATIO ANALYSIS — VIPV Urban Canyon")
    print(f"  CAMPAIGN     : {CAMPAIGN}")
    print(f"  Data variant : {_VARIANT_TAG}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print("=" * 78)

    print("\n[1/6] Loading and preparing measurement data "
          "(H and W both already joined upstream) ...")
    df, n_excluded_bldg = load_combined_data(SENSOR_FILES)

    print("\n[2/6] Computing H/W ratio and assigning bins ...")
    df = add_hw_ratio(df)
    df = assign_hw_bins(df)
    print_width_diagnostics(df)

    print("\n[3/6] Aggregating measured loss per H/W bin (Analyses 1-2-3) ...")
    agg = compute_bin_loss(df)
    print(agg.to_string(index=False))

    print("\n[4/6] Generating figures (Analyses 1-2-3) ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_analysis1_correlation(df, agg, _out_png("hw_real_a1_correlation"))
    fig_analysis2_validation(df, agg, _out_png("hw_real_a2_validation"))
    fig_analysis3_spatial_map(df, _out_png("hw_real_a3_spatial_map"))

    if RUN_CONTINUOUS_SVF_ANALYSIS:
        print("\n[5/6] Analysis 4 — continuous per-point SVF correlation "
              "(RUN_CONTINUOUS_SVF_ANALYSIS=True) ...")
        valid = df[df["width_status"] == "ok"].copy()
        valid["theoretical_svf_loss_pct"] = compute_point_theoretical_svf_loss(
            valid["hw_ratio"].values,
        )
        valid = assign_svf_quantile_bins(valid)
        svf_agg = compute_svf_bin_stats(valid)
        print(svf_agg.to_string(index=False))

        fig_analysis4_continuous_svf_correlation(
            valid, svf_agg, _out_png("hw_real_a4_continuous_svf_correlation"),
        )

        comp = compute_hw_bin_composition(valid)
        print("\n  H/W composition per SVF bin:")
        print(comp.to_string(index=False))
        fig_analysis4b_hw_composition(comp, _out_png("hw_real_a4b_hw_composition"))
    else:
        print("\n[5/6] Skipping continuous SVF analysis "
              "(RUN_CONTINUOUS_SVF_ANALYSIS=False).")
        valid = None

    if RUN_CONTINUOUS_SVF_ANALYSIS and RUN_KC_CLEAR_STRATIFICATION:
        print("\n[6/6] Analyses 5-6 — sky-condition (kc_clear) stratification "
              "(RUN_KC_CLEAR_STRATIFICATION=True) ...")
        valid = add_kc_clear(valid)
        print("\nkc_clear distribution check:")
        print(valid["kc_clear"].describe())
        pct_at_cap = (valid["kc_clear"] >= 0.999).mean() * 100
        print(f"  fraction with kc_clear >= 0.999 : {pct_at_cap:.1f}%")
        n_before_kc = len(valid)
        valid = valid.dropna(subset=["kc_clear"]).reset_index(drop=True)
        print(f"  {len(valid):,} rows with valid kc_clear "
              f"({n_before_kc - len(valid):,} excluded — ghi_clear <= 0 or missing)")

        if len(valid) < 3 * MIN_BIN_POINTS:
            print("  [WARN] Too few rows with valid kc_clear for a 3-way "
                  "stratification — skipping Analyses 5-6.")
        else:
            valid["kc_tercile"] = pd.qcut(
                valid["kc_clear"].rank(method="first"), 3, labels=KC_TERCILE_LABELS,
            )

            edges = compute_global_svf_bin_edges(valid)
            valid = assign_fixed_svf_bins(valid, edges)

            strat_aggs = {}
            n_by_stratum = {}
            for label in KC_TERCILE_LABELS:
                sub = valid[valid["kc_tercile"] == label]
                n_by_stratum[label] = len(sub)
                strat_agg = compute_bin_stats_fixed(sub)
                strat_aggs[label] = strat_agg
                print(f"\n  -- {label} (n={len(sub):,}) --")
                print(strat_agg.to_string(index=False))

            fig_analysis5_kc_clear_stratified(
                strat_aggs, n_by_stratum, _out_png("hw_real_a5_kc_clear_stratified"),
            )

            print(f"\n  Point-level SVF x kc_clear interaction regression "
                  f"(n={len(valid):,}) ...")
            model, svf_mean, kc_mean = fit_svf_kc_interaction(valid)
            print(model.summary())
            print_interaction_verdict(model)
            fig_analysis6_interaction_effect(
                model, svf_mean, kc_mean, valid,
                _out_png("hw_real_a6_interaction_effect"),
            )
    elif RUN_KC_CLEAR_STRATIFICATION and not RUN_CONTINUOUS_SVF_ANALYSIS:
        print("\n[6/6] RUN_KC_CLEAR_STRATIFICATION requires "
              "RUN_CONTINUOUS_SVF_ANALYSIS=True — skipping Analyses 5-6.")
    else:
        print("\n[6/6] Skipping sky-condition stratification "
              "(RUN_KC_CLEAR_STRATIFICATION=False).")

    print("\n" + "=" * 78)
    print("  Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()