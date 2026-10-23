"""
================================================================================
Theoretical H/W Ratio Sweep — Brito et al. (2024) Urban-Canyon Model
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Theoretical sweep of the Height-to-Width (H/W) ratio of an urban street
canyon over [HW_MIN, HW_MAX], with the analytical model of Brito et al.
(2024), at a configurable latitude (default: Lisbon, 38.76° N). No measured
data is used — the empirical counterpart is hw_ratio_real_analysis.py, which
can overlay the annual-mean curves of this model as a theoretical backdrop.

The physics (mean_obstruction_angle_rad, svf_from_hw, diffuse_loss_pct,
direct_loss_pct_ew_street, ...) is in the shared module brito_svf_model,
also used by building_height_vs_losses.py and hw_ratio_real_analysis.py; its
self-tests are run first. This script only holds the sweep configuration,
the figures and the console driver.

Metrics computed (all as a function of H/W)
-------------------------------------------
  M1 - Diffuse loss [%]: Loss_DHI = (1 - cos^2(alpha_bar)) x 100 [Eq. 2/4].
       Middle and edge positions; edge asymptote -> 50 % as H/W -> inf.
  M2 - Sky View Factor [-]: SVF_middle = cos^2(alpha_bar_m) [Eq. 2];
       SVF_edge = 0.5*cos^2(alpha_bar_e) + 0.5 [Eq. 4]. The higher SVF at the
       edge applies only to isotropic diffuse radiation, not to direct beam.
  M3 - Mean obstruction angle alpha_bar [deg]: azimuth-averaged, both
       positions (Eq. 3 vs Eq. 5); both converge to 90 deg as H/W -> inf.
  M4 - Direct (beam) loss [%]: Loss_BHI = 1 - I(H/W)/I(H=0) [Eq. 9], E-W
       street (worst-case orientation). Seasonal curves + DOY-averaged
       annual mean; the annual mean is the one comparable to Brito Fig. 7.
  M5 - Total (GHI-proxy) loss [%]: Loss_total = f_d.Loss_DHI + (1-f_d).Loss_BHI,
       for the DIFFUSE_FRACTIONS values, annual-mean BHI. Past the BHI
       threshold, clear-sky loss can exceed overcast loss (direct blockage
       dominates).
  M6 - Theoretical relative autonomy loss [%]: same quantity as M5 in the
       VIPV pipeline metric. Two panels: seasonal sensitivity (middle,
       f_d = 0.45) and position sensitivity (middle vs edge, three f_d).

Reference
---------
  Brito, M. C., et al. (2024). "Effect of urban shadowing on the potential
  of solar-powered vehicles." Progress in Photovoltaics, 32(2), 73-83.
  https://doi.org/10.1002/pip.3737

--------------------------------------------------------------------------------
Input / output
--------------------------------------------------------------------------------
    input  : none (purely theoretical). FILE_SUFFIX only names the output
             sub-directory, so that the figures sit next to the
             hw_ratio_real_analysis.py results of the same data variant.
    output : OUTPUT_ROOT / 6_urban_geometry / hw_ratio_outputs / <FILE_SUFFIX> /
             hw_m1_diffuse_loss.png, hw_m2_sky_view_factor.png,
             hw_m3_obstruction_angle.png, hw_m4_direct_loss_seasonal.png,
             hw_m5_total_loss_fd_sensitivity.png, hw_m6_autonomy_loss.png

Requirements
------------
    pip install numpy matplotlib

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 3.0
================================================================================
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Project configuration (paths, shared modules) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_ROOT  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── VIPV shared modules (code/common) ─────────────────────────────────────────
from brito_svf_model import (
    mean_obstruction_angle_rad,
    svf_from_hw,
    diffuse_loss_pct,
    solar_declination_rad,
    sunrise_hour_angle_rad,
    incidence_angle_rad,
    sun_azimuth_rad,
    obstruction_angle_at_azimuth_rad,
    direct_loss_pct_ew_street,
    annual_mean_direct_loss_pct_ew,
    total_loss_pct,
    _run_assertions,
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Output variant ----------------------------------------------------------
# No sensor data is loaded: FILE_SUFFIX only names the output sub-directory
# (same variant as the hw_ratio_real_analysis.py run it goes with).
FILE_SUFFIX: str = "sd_cleaned"

_OUTPUT_BASE_DIR: Path = OUTPUT_ROOT / "6_urban_geometry" / "hw_ratio_outputs"
OUTPUT_BASE_DIR: Path = _OUTPUT_BASE_DIR / FILE_SUFFIX

# Latitude of the sweep [degrees N]. Default: Lisbon.
LATITUDE_DEG: float = 38.76

# H/W sweep range
HW_MIN: float = 0.0
HW_MAX: float = 8.0
HW_N_POINTS: int = 800          # resolution of the continuous curves

# H/W reference values — vertical dotted lines annotated on each figure
# (Lisbon typical from Brito et al. 2024; dense urban and canyon for scale)
HW_REFERENCE_VALUES: dict[str, float] = {
    "Lisbon\ntypical": 0.7,
    "Dense\nurban":    1.5,
    "Deep\ncanyon":    3.0,
}

# Representative diffuse fraction f_d = DHI/GHI for the total-loss figures
DIFFUSE_FRACTIONS: list[float] = [0.25, 0.45, 0.65]   # clear / mixed / overcast

# Day-of-year for the three seasonal direct-loss overlays (M4)
SEASONS: dict[str, int] = {
    "Summer solstice (21 Jun)":  172,
    "Equinox (21 Mar)":           80,
    "Winter solstice (21 Dec)":  355,
}
SEASON_COLORS: dict[str, str] = {
    "Summer solstice (21 Jun)":  "#E65100",
    "Equinox (21 Mar)":          "#1565C0",
    "Winter solstice (21 Dec)":  "#2E7D32",
}

# Number of DOY samples for the annual-mean BHI computation
N_DOY_ANNUAL: int = 365

# Number of angular integration samples (trapezoidal rule)
N_GAMMA_SAMPLES: int = 2001
N_OMEGA_SAMPLES: int = 2001

# Pipeline figure conventions
SHOW_FIGURES: bool = False
OUTPUT_DPI:   int  = 300

# Colour palette (consistent with VIPV pipeline conventions)
COLOR_MIDDLE: str = "#C0392B"   # red — middle of street
COLOR_EDGE:   str = "#1565C0"   # blue — edge of street

# ==============================================================================
# DERIVED CONSTANTS
# ==============================================================================

def _build_output_path(out_dir: Path, base_name: str, extension: str) -> Path:
    """Return the output path <out_dir>/<base_name>.<extension>."""
    return out_dir / f"{base_name}.{extension}"


OUTPUT_DIR: Path = OUTPUT_BASE_DIR
LAT_RAD: float = np.radians(LATITUDE_DEG)
HW_ARRAY: np.ndarray = np.linspace(HW_MIN, HW_MAX, HW_N_POINTS)


# ==============================================================================
# PLOT HELPERS
# ==============================================================================

def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    """Save the figure at OUTPUT_DPI, then show or close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    print(f"\n  Figure saved -> {out_path}  ({OUTPUT_DPI} dpi)")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def _add_hw_references(ax: plt.Axes, y_top: float,
                        color: str = "#999999",
                        text_y_frac: float = 0.97) -> None:
    """Vertical dotted reference lines with clean multi-line labels."""
    for label, hw_val in HW_REFERENCE_VALUES.items():
        ax.axvline(hw_val, color=color, linewidth=0.9, linestyle=":",
                   alpha=0.8, zorder=1)
        ax.text(hw_val + 0.06, y_top * text_y_frac, label,
                fontsize=6.5, color=color, va="top", rotation=90,
                linespacing=1.3)


def _annotate_point(ax: plt.Axes, hw_val: float, y_val: float,
                     text: str, color: str, dx: float = 0.1,
                     dy: float = 2.5) -> None:
    """Small numeric annotation at a specific (H/W, y) point."""
    ax.annotate(
        text,
        xy=(hw_val, y_val), xytext=(hw_val + dx, y_val + dy),
        fontsize=7, color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
    )


# ==============================================================================
# FIGURES
# ==============================================================================

def fig_m1_diffuse_loss(hw: np.ndarray, out_path: Path) -> None:
    """
    M1 — Isotropic diffuse irradiance loss [%] vs H/W.
    Both positions. Asymptote at 50 % for edge annotated.
    Numeric values at each reference H/W marker.
    """
    loss_mid  = diffuse_loss_pct(hw, edge=False)
    loss_edge = diffuse_loss_pct(hw, edge=True)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(hw, loss_mid,  color=COLOR_MIDDLE, linewidth=2.2,
            label="Middle of street  (Loss → 100 % as H/W → ∞)")
    ax.plot(hw, loss_edge, color=COLOR_EDGE,   linewidth=2.2,
            linestyle="--",
            label="Edge of street  (Loss → 50 % as H/W → ∞)")

    ax.axhline(50.0, color=COLOR_EDGE, linewidth=0.9, linestyle=":",
               alpha=0.6, zorder=1)
    ax.text(HW_MAX - 0.1, 52.0, "Asymptote: 50 %\n(near wall fully blocks\none hemisphere)",
            fontsize=6.5, color=COLOR_EDGE, ha="right", va="bottom", linespacing=1.3)

    _add_hw_references(ax, y_top=100.0)

    for hw_ref in HW_REFERENCE_VALUES.values():
        idx = np.argmin(np.abs(hw - hw_ref))
        vm = loss_mid[idx];  ve = loss_edge[idx]
        ax.annotate(f"{vm:.0f}%", xy=(hw_ref, vm),
                    xytext=(hw_ref + 0.25, vm - 6),
                    fontsize=6.5, color=COLOR_MIDDLE,
                    arrowprops=dict(arrowstyle="-", color=COLOR_MIDDLE, lw=0.6))
        ax.annotate(f"{ve:.0f}%", xy=(hw_ref, ve),
                    xytext=(hw_ref + 0.25, ve + 5),
                    fontsize=6.5, color=COLOR_EDGE,
                    arrowprops=dict(arrowstyle="-", color=COLOR_EDGE, lw=0.6))

    ax.set_xlim(HW_MIN, HW_MAX)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Height-to-Width ratio  H/W  [–]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Diffuse irradiance loss  [%]",    fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title("M1 — Isotropic Diffuse Irradiance Loss vs. Canyon H/W Ratio",
                 fontsize=12, fontweight="bold")
    fig.text(
        0.5, -0.03,
        "Brito et al. (2024) Eq. 2 (middle) / Eq. 4 (edge)  —  "
        "Loss_DHI = (1 − cos²(ᾱ)) × 100  —  "
        "latitude-independent (isotropic diffuse assumption)  —  "
        "dotted vertical lines: Lisbon typical / dense urban / deep canyon H/W",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_m2_svf(hw: np.ndarray, out_path: Path) -> None:
    """
    M2 — Sky View Factor (SVF) vs H/W.
    Numeric values at reference H/W. Asymptotes annotated.
    Interpretive note: SVF_edge > SVF_middle is for diffuse-only.
    """
    svf_mid  = svf_from_hw(hw, edge=False)
    svf_edge = svf_from_hw(hw, edge=True)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(hw, svf_mid,  color=COLOR_MIDDLE, linewidth=2.2,
            label="Middle of street  (SVF → 0 as H/W → ∞)")
    ax.plot(hw, svf_edge, color=COLOR_EDGE,   linewidth=2.2,
            linestyle="--",
            label="Edge of street  (SVF → 0.5 as H/W → ∞)")

    ax.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.axhline(0.5, color=COLOR_EDGE, linewidth=0.9, linestyle=":",
               alpha=0.6, zorder=1)
    ax.text(HW_MAX - 0.1, 0.52, "Asymptote: SVF = 0.5",
            fontsize=6.5, color=COLOR_EDGE, ha="right", va="bottom")
    ax.text(HW_MAX - 0.1, 0.03, "Asymptote: SVF → 0",
            fontsize=6.5, color=COLOR_MIDDLE, ha="right", va="bottom")

    _add_hw_references(ax, y_top=1.0)

    for hw_ref in HW_REFERENCE_VALUES.values():
        idx = np.argmin(np.abs(hw - hw_ref))
        vm = svf_mid[idx];  ve = svf_edge[idx]
        ax.annotate(f"{vm:.2f}", xy=(hw_ref, vm),
                    xytext=(hw_ref + 0.28, vm - 0.06),
                    fontsize=6.5, color=COLOR_MIDDLE,
                    arrowprops=dict(arrowstyle="-", color=COLOR_MIDDLE, lw=0.6))
        ax.annotate(f"{ve:.2f}", xy=(hw_ref, ve),
                    xytext=(hw_ref + 0.28, ve + 0.05),
                    fontsize=6.5, color=COLOR_EDGE,
                    arrowprops=dict(arrowstyle="-", color=COLOR_EDGE, lw=0.6))

    ax.set_xlim(HW_MIN, HW_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Height-to-Width ratio  H/W  [–]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sky View Factor  SVF  [–]",        fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title("M2 — Sky View Factor (SVF) vs. Canyon H/W Ratio",
                 fontsize=12, fontweight="bold")
    ax.text(0.60, 0.55,
            "Note: SVF_edge > SVF_middle only for\n"
            "isotropic diffuse radiation. For direct\n"
            "(beam) radiation the edge position is\n"
            "more severely blocked by the near wall.",
            transform=ax.transAxes, fontsize=7, color="#555555",
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#CCCCCC", alpha=0.9))
    fig.text(
        0.5, -0.03,
        "SVF = cos²(ᾱ_m) (middle)  /  SVF = 0.5·cos²(ᾱ_e) + 0.5 (edge)  —  "
        "SVF = 1: fully open sky  —  "
        "dotted vertical lines: reference H/W values",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_m3_obstruction_angle(hw: np.ndarray, out_path: Path) -> None:
    """
    M3 — Mean obstruction angle ᾱ [°] vs H/W.
    Clarifying note on why middle > edge. Asymptote annotations.
    """
    alpha_mid  = np.degrees(mean_obstruction_angle_rad(hw, edge=False))
    alpha_edge = np.degrees(mean_obstruction_angle_rad(hw, edge=True))

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(hw, alpha_mid,  color=COLOR_MIDDLE, linewidth=2.2,
            label="Middle of street  (ᾱ → 90° as H/W → ∞)")
    ax.plot(hw, alpha_edge, color=COLOR_EDGE,   linewidth=2.2,
            linestyle="--",
            label="Edge of street  (ᾱ_e encodes far-wall only;  near-wall\n"
                  "contribution is in the +0.5 term of SVF_edge)")

    for hw_ref in HW_REFERENCE_VALUES.values():
        idx = np.argmin(np.abs(hw - hw_ref))
        vm = alpha_mid[idx];  ve = alpha_edge[idx]
        ax.annotate(f"{vm:.1f}°", xy=(hw_ref, vm),
                    xytext=(hw_ref + 0.28, vm + 3),
                    fontsize=6.5, color=COLOR_MIDDLE,
                    arrowprops=dict(arrowstyle="-", color=COLOR_MIDDLE, lw=0.6))
        ax.annotate(f"{ve:.1f}°", xy=(hw_ref, ve),
                    xytext=(hw_ref + 0.28, ve - 5),
                    fontsize=6.5, color=COLOR_EDGE,
                    arrowprops=dict(arrowstyle="-", color=COLOR_EDGE, lw=0.6))

    _add_hw_references(ax, y_top=90.0)

    ax.set_xlim(HW_MIN, HW_MAX)
    ax.set_ylim(0, 90)
    ax.set_xlabel("Height-to-Width ratio  H/W  [–]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean obstruction angle  ᾱ  [°]",  fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title("M3 — Mean Obstruction Angle vs. Canyon H/W Ratio",
                 fontsize=12, fontweight="bold")
    ax.text(0.62, 0.30,
            "ᾱ_middle > ᾱ_edge because the middle\n"
            "formula uses 2H/W (building W/2 away)\n"
            "vs. H/W for the far wall at the edge.\n"
            "The near wall at the edge (π/2 block)\n"
            "is captured separately in SVF_edge.",
            transform=ax.transAxes, fontsize=7, color="#555555",
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#CCCCCC", alpha=0.9))
    fig.text(
        0.5, -0.03,
        "Brito et al. (2024) Eq. 3 (middle) / Eq. 5 (edge)  —  "
        "ᾱ_m = (1/π) ∫ atan(2H/W · cosγ) dγ  over γ ∈ [–π/2, π/2]  —  "
        "dotted vertical lines: reference H/W values",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_m4_direct_loss(hw: np.ndarray, annual_bhi_mid: np.ndarray,
                        out_path: Path) -> None:
    """
    M4 — Direct (beam) irradiance loss [%] vs H/W.

    Shows the annual-mean curve (primary, thick) alongside three individual-
    day curves (seasonal, thin) as pedagogical overlays. The critical H/W
    threshold (= tan(β_noon)/2) for each seasonal day is annotated.

    Annual-mean curve is the appropriate comparison to Brito Fig. 7 (solid
    lines). Individual-day curves illustrate the sharp geometric transitions
    that the annual average smooths out.
    """
    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.plot(hw, annual_bhi_mid, color="#333333", linewidth=2.8,
            label="Annual mean  (DOY-weighted, comparable to Brito Fig. 7)",
            zorder=5)

    for season_label, doy in SEASONS.items():
        print(f"    {season_label} (DOY {doy}) ...", end="", flush=True)
        loss = direct_loss_pct_ew_street(hw, doy, LAT_RAD, edge=False)
        color = SEASON_COLORS[season_label]
        ax.plot(hw, loss, color=color, linewidth=1.3, alpha=0.7,
                linestyle="--", label=f"{season_label}  (single day)")

        decl = solar_declination_rad(doy)
        noon_elev_deg = 90.0 - abs(np.degrees(LAT_RAD) - np.degrees(decl))
        if noon_elev_deg > 1.0:
            hw_thresh = np.tan(np.radians(noon_elev_deg)) / 2.0
            if HW_MIN < hw_thresh < HW_MAX:
                ax.axvline(hw_thresh, color=color, linewidth=0.8,
                           linestyle=":", alpha=0.5, zorder=1)
                ax.text(hw_thresh + 0.06, 102.5,
                        f"H/W*={hw_thresh:.2f}\n(β_noon={noon_elev_deg:.0f}°)",
                        fontsize=6.0, color=color, va="bottom", linespacing=1.2)
        print(" done")

    _add_hw_references(ax, y_top=100.0, text_y_frac=0.82)

    ax.set_xlim(HW_MIN, HW_MAX)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Height-to-Width ratio  H/W  [–]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Direct irradiance loss  [%]",      fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title("M4 — Direct (Beam) Irradiance Loss vs. Canyon H/W Ratio\n"
                 "Annual Mean + Individual Seasonal Days  (E–W street, Lisbon)",
                 fontsize=12, fontweight="bold")
    ax.text(0.52, 0.22,
            "Single-day curves show sharp transitions (0 % → 100 %)\n"
            "at H/W* = tan(β_noon)/2, where β_noon is noon solar elevation.\n"
            "These are physically real for a perfect E–W infinite canyon\n"
            "but are smoothed by the annual average (thick curve).",
            transform=ax.transAxes, fontsize=7, color="#555555",
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#CCCCCC", alpha=0.9))
    fig.text(
        0.5, -0.03,
        "Brito et al. (2024) Eq. 9-10  —  E–W street (maximum direct loss)  —  "
        f"φ = {LATITUDE_DEG:.2f}° N  —  middle of street  —  uniform DNI assumed  —  "
        "dotted vertical lines: reference H/W values",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_m5_total_loss(hw: np.ndarray, annual_bhi_mid: np.ndarray,
                       annual_bhi_edge: np.ndarray, out_path: Path) -> None:
    """
    M5 — Total irradiance loss [%] vs H/W, using the annual-mean BHI.

    The counter-intuitive ordering (clear sky > overcast at high H/W) is
    explicitly annotated on the figure.
    """
    fd_colors = ["#2E7D32", "#1565C0", "#C0392B"]   # low / mid / high f_d
    fd_labels = [
        f"f_d = {fd:.2f}  ({'clear sky' if fd < 0.35 else 'mixed' if fd < 0.55 else 'overcast'})"
        for fd in DIFFUSE_FRACTIONS
    ]

    fig, ax = plt.subplots(figsize=(9, 5.8))

    losses = []
    for fd, color, label in zip(DIFFUSE_FRACTIONS, fd_colors, fd_labels):
        loss = total_loss_pct(hw, annual_bhi_mid, fd, edge=False)
        losses.append(loss)
        ax.plot(hw, loss, color=color, linewidth=2.2, label=label)

    cross_idx = np.where(np.diff(np.sign(losses[0] - losses[1])))[0]
    if len(cross_idx) > 0:
        hw_cross = hw[cross_idx[0]]
        ax.axvline(hw_cross, color="#999999", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.annotate(
            f"H/W ≈ {hw_cross:.2f}\nClear-sky loss\nexceeds overcast loss\n(direct blockage\ndominates)",
            xy=(hw_cross, losses[0][cross_idx[0]]),
            xytext=(hw_cross + 0.5, 45),
            fontsize=6.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
        )

    _add_hw_references(ax, y_top=100.0)

    ax.set_xlim(HW_MIN, HW_MAX)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Height-to-Width ratio  H/W  [–]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total irradiance loss  [%]",       fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(
        "M5 — Total Irradiance Loss vs. Canyon H/W Ratio\n"
        "Sensitivity to Sky Condition  (f_d = DHI/GHI)  —  Annual-Mean BHI",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, -0.03,
        "Loss_total = f_d · Loss_DHI + (1 − f_d) · Loss_BHI  —  "
        "BHI loss: annual mean (DOY-weighted)  —  "
        f"middle of street, φ = {LATITUDE_DEG:.2f}° N, E–W street  —  "
        "dotted vertical lines: reference H/W values",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


def fig_m6_autonomy_loss(hw: np.ndarray,
                          annual_bhi_mid: np.ndarray,
                          annual_bhi_edge: np.ndarray,
                          out_path: Path) -> None:
    """
    M6 — Theoretical relative autonomy loss [%] vs H/W.

    Left panel  — Seasonal sensitivity (middle of street, f_d = 0.45).
      Shows three individual-day total-loss curves AND the annual-mean
      curve so the reader can gauge how representative each day is.

    Right panel — Position sensitivity (middle vs. edge, annual-mean BHI).
      Three f_d values encoded via LINE WIDTH (thin=0.25, medium=0.45,
      thick=0.65) rather than line style, to limit visual clutter: 2 colours
      (red=middle, blue=edge) x 3 widths = 6 distinguishable series.
    """
    fd_typical = 0.45
    colors_season = list(SEASON_COLORS.values())
    labels_season = list(SEASON_COLORS.keys())
    lw_map = {0.25: 1.1, 0.45: 2.0, 0.65: 3.2}   # thin / medium / thick

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    print("  Computing M6 autonomy loss curves ...")

    # ── Left panel: seasonal sensitivity ─────────────────────────────────────
    ax = axes[0]
    ann_loss = total_loss_pct(hw, annual_bhi_mid, fd_typical, edge=False)
    ax.plot(hw, ann_loss, color="#333333", linewidth=2.8,
            label=f"Annual mean (f_d = {fd_typical:.2f})", zorder=5)

    for doy, color, lbl in zip(SEASONS.values(), colors_season, labels_season):
        decl = solar_declination_rad(doy)
        omega_s = sunrise_hour_angle_rad(LAT_RAD, decl)
        omega = np.linspace(-omega_s, omega_s, N_OMEGA_SAMPLES)
        theta = incidence_angle_rad(LAT_RAD, decl, omega)
        psi   = sun_azimuth_rad(LAT_RAD, decl, theta, omega)
        cos_theta = np.maximum(np.cos(theta), 0.0)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        I_total = trapz(cos_theta, omega)
        day_bhi = np.empty(len(hw))
        for i, hw_val in enumerate(hw):
            alpha_sun = obstruction_angle_at_azimuth_rad(hw_val, psi, edge=False)
            visible   = (np.pi / 2.0 - theta) > alpha_sun
            I_vis     = trapz(np.where(visible, cos_theta, 0.0), omega)
            day_bhi[i] = (1.0 - I_vis / I_total) * 100.0 if I_total > 0 else 0.0
        day_loss = total_loss_pct(hw, day_bhi, fd_typical, edge=False)
        ax.plot(hw, day_loss, color=color, linewidth=1.3, linestyle="--",
                alpha=0.7, label=f"{lbl}  (single day)")

    _add_hw_references(ax, y_top=100.0)
    ax.set_xlim(HW_MIN, HW_MAX)
    ax.set_ylim(0, 100)
    ax.set_xlabel("H/W  [–]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Theoretical relative autonomy loss  [%]",
                  fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(f"Seasonal sensitivity\n(middle of street, f_d = {fd_typical:.2f})",
                 fontsize=11, fontweight="bold")

    # ── Right panel: position sensitivity ────────────────────────────────────
    ax = axes[1]
    legend_handles = []
    for fd in DIFFUSE_FRACTIONS:
        lw = lw_map[fd]
        lm = total_loss_pct(hw, annual_bhi_mid,  fd, edge=False)
        le = total_loss_pct(hw, annual_bhi_edge, fd, edge=True)
        lh_m, = ax.plot(hw, lm, color=COLOR_MIDDLE, linewidth=lw,
                         label=f"Middle, f_d={fd:.2f}")
        lh_e, = ax.plot(hw, le, color=COLOR_EDGE,   linewidth=lw,
                         label=f"Edge,   f_d={fd:.2f}")
        legend_handles.extend([lh_m, lh_e])

    _add_hw_references(ax, y_top=100.0)
    ax.set_xlim(HW_MIN, HW_MAX)
    ax.set_xlabel("H/W  [–]", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title("Position sensitivity\n"
                 "(annual-mean BHI; line width = f_d: thin=0.25, medium=0.45, thick=0.65)",
                 fontsize=11, fontweight="bold")
    ax.text(0.37, 0.18,
            "red = middle,  blue = edge\n"
            "thicker line = higher f_d (more overcast)",
            transform=ax.transAxes, fontsize=7.5, color="#555555",
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#CCCCCC", alpha=0.9))

    fig.suptitle(
        "M6 — Theoretical Relative Autonomy Loss vs. Canyon H/W Ratio\n"
        "Annual-Mean BHI Base",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.03,
        "Loss_total = f_d · Loss_DHI + (1 − f_d) · Loss_BHI  —  "
        "E–W street orientation  —  "
        f"φ = {LATITUDE_DEG:.2f}° N  —  "
        "red = middle of street, blue = edge  —  "
        "dotted vertical lines: reference H/W values",
        ha="center", fontsize=7.5, color="#888888", style="italic",
    )
    plt.tight_layout()
    _save_fig(fig, out_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the model self-tests, compute the annual-mean BHI loss and draw figures M1–M6."""
    print("=" * 78)
    print("  H/W RATIO IMPACT ANALYSIS — VIPV Urban Canyon")
    print(f"  Latitude        : {LATITUDE_DEG}° N")
    print(f"  H/W sweep       : [{HW_MIN}, {HW_MAX}]  ({HW_N_POINTS} points)")
    print(f"  Output dir      : {OUTPUT_DIR}")
    print("=" * 78)

    print("\n[0/7] Validating shared physics model (brito_svf_model.py) ...")
    _run_assertions()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[Pre] Computing annual-mean BHI loss  "
          f"({N_DOY_ANNUAL} DOY × {len(HW_ARRAY)} H/W points — may take ~1 min) ...")
    annual_bhi_mid  = annual_mean_direct_loss_pct_ew(HW_ARRAY, LAT_RAD,
                                                      edge=False,
                                                      n_doy=N_DOY_ANNUAL,
                                                      n_omega=N_OMEGA_SAMPLES)
    annual_bhi_edge = annual_mean_direct_loss_pct_ew(HW_ARRAY, LAT_RAD,
                                                      edge=True,
                                                      n_doy=N_DOY_ANNUAL,
                                                      n_omega=N_OMEGA_SAMPLES)
    print("  Annual-mean BHI computed.")

    print("\n[1/6] M1 — Diffuse irradiance loss ...")
    fig_m1_diffuse_loss(
        HW_ARRAY,
        _build_output_path(OUTPUT_DIR, "hw_m1_diffuse_loss", "png"),
    )

    print("\n[2/6] M2 — Sky View Factor ...")
    fig_m2_svf(
        HW_ARRAY,
        _build_output_path(OUTPUT_DIR, "hw_m2_sky_view_factor", "png"),
    )

    print("\n[3/6] M3 — Mean obstruction angle ...")
    fig_m3_obstruction_angle(
        HW_ARRAY,
        _build_output_path(OUTPUT_DIR, "hw_m3_obstruction_angle", "png"),
    )

    print("\n[4/6] M4 — Direct irradiance loss (annual mean + seasonal) ...")
    fig_m4_direct_loss(
        HW_ARRAY, annual_bhi_mid,
        _build_output_path(OUTPUT_DIR, "hw_m4_direct_loss_seasonal", "png"),
    )

    print("\n[5/6] M5 — Total irradiance loss (annual-mean BHI, fd sensitivity) ...")
    fig_m5_total_loss(
        HW_ARRAY, annual_bhi_mid, annual_bhi_edge,
        _build_output_path(OUTPUT_DIR, "hw_m5_total_loss_fd_sensitivity", "png"),
    )

    print("\n[6/6] M6 — Theoretical relative autonomy loss ...")
    fig_m6_autonomy_loss(
        HW_ARRAY, annual_bhi_mid, annual_bhi_edge,
        _build_output_path(OUTPUT_DIR, "hw_m6_autonomy_loss", "png"),
    )

    print("\n" + "=" * 78)
    print("  Done.  6 figures saved.")
    print("=" * 78)


if __name__ == "__main__":
    main()