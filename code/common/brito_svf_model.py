"""
================================================================================
Brito et al. (2024) Urban-Canyon Model — Shared Module
================================================================================
Context : Vehicle-Integrated Photovoltaics (VIPV) Research
          Faculty of Sciences, University of Lisbon (FCUL)

Single implementation of the analytical urban-canyon irradiance-loss model of:

  Brito, M. C., et al. (2024). "Effect of urban shadowing on the potential
  of solar-powered vehicles." Progress in Photovoltaics, 32(2), 73-83.
  https://doi.org/10.1002/pip.3737

Used by building_height_vs_losses.py, hw_ratio_impact_analysis.py and
hw_ratio_real_analysis.py, which all import the model from here so that
their results stay consistent.

--------------------------------------------------------------------------------
Functions
--------------------------------------------------------------------------------
Diffuse (isotropic) component — Eq. 1-4:
    mean_obstruction_angle_rad(H_over_W, edge=False)
    svf_from_hw(H_over_W, edge=False)
    diffuse_loss_pct(H_over_W, edge=False)

Direct (beam) component, E-W street (worst case at mid-latitudes) — Eq. 6-11:
    solar_declination_rad(doy)
    sunrise_hour_angle_rad(lat_rad, decl_rad)
    incidence_angle_rad(lat_rad, decl_rad, omega)
    sun_azimuth_rad(lat_rad, decl_rad, theta_rad, omega)
    obstruction_angle_at_azimuth_rad(H_over_W, gamma_rad, edge=False)
    direct_loss_pct_ew_street(hw_array, doy, lat_rad, edge=False)
    annual_mean_direct_loss_pct_ew(hw_array, lat_rad, edge=False, n_doy=365)

Combined:
    total_loss_pct(hw_array, bhi_loss_array, diffuse_fraction, edge=False)

--------------------------------------------------------------------------------
Geometry convention
--------------------------------------------------------------------------------
  H/W        : height-to-width ratio of the street canyon (dimensionless).
  edge=False : MIDDLE of the street — symmetric two-wall view.
               SVF -> 1, Loss_DHI -> 0   as H/W -> 0
               SVF -> 0, Loss_DHI -> 100 as H/W -> infinity
  edge=True  : EDGE of the street — the near wall blocks one hemisphere,
               the far wall contributes the rest.
               SVF -> 1,   Loss_DHI -> 0  as H/W -> 0
               SVF -> 0.5, Loss_DHI -> 50 as H/W -> infinity

--------------------------------------------------------------------------------
Direct-loss model
--------------------------------------------------------------------------------
Infinite E-W street canyon (worst-case orientation at mid-latitudes: the sun
crosses the street axis near solar noon). direct_loss_pct_ew_street()
integrates DNI*cos(theta) (DNI = 1, purely geometric) over the daylight hours
of one day of year. Single-day curves show a sharp H/W threshold
(tan(noon elevation) / 2) where the noon sun switches from visible to
blocked. Brito et al. Fig. 7 uses annual-mean irradiance, which smooths this
discontinuity: use annual_mean_direct_loss_pct_ew() (each day weighted by its
clear-sky irradiance) for any comparison with the published curves or with
multi-day measurements.

Self-test
---------
    python brito_svf_model.py

runs the physical-invariant suite A1-A13: SVF / loss limits at H/W = 0 and
H/W -> infinity, monotonicity, SVF + Loss_DHI/100 = 1, solar declination at
equinox (~0 deg) and summer solstice (~+23.45 deg), the E-W canyon H/W
threshold at the latitude of Lisbon, the annual-mean direct loss in %, and
no direct loss at the street edge without buildings.

Requirements
------------
    pip install numpy

================================================================================
Authors  : See AUTHORS file
Date     : 2026
Version  : 1.1
================================================================================
"""

from __future__ import annotations

import numpy as np

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Default angular-integration resolution (trapezoidal rule). Callers may
# override via the n_samples / n_omega parameters where exposed.
N_GAMMA_SAMPLES: int = 2001
N_OMEGA_SAMPLES: int = 2001
N_DOY_ANNUAL: int = 365

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ==============================================================================
# DIFFUSE (ISOTROPIC) COMPONENT — Brito et al. 2024, Eq. 1-4
# ==============================================================================

def mean_obstruction_angle_rad(
    hw: np.ndarray, edge: bool = False, n_samples: int = N_GAMMA_SAMPLES,
) -> np.ndarray:
    """
    Azimuth-averaged obstruction angle alpha_bar [rad] for an infinite
    street canyon.

    Parameters
    ----------
    hw : array_like
        Height-to-width ratio H/W (dimensionless, >= 0).
    edge : bool
        False -> middle of street (Eq. 1a / Eq. 3):
            integrand = atan(2H/W * cos(gamma)), gamma in [-pi/2, pi/2]
        True  -> edge of street (Eq. 1b / Eq. 5):
            integrand = atan(H/W * cos(gamma))  (far-wall contribution
            only; the near wall is folded into the +0.5 term of
            svf_from_hw(edge=True) rather than into alpha_bar itself).
    n_samples : int
        Number of azimuth samples for the trapezoidal integration.

    Why alpha_middle > alpha_edge at every H/W
    --------------------------------------------
    The middle formula uses 2H/W (the building is W/2 away on each side);
    the edge formula uses H/W (the distant building is W away). The extra
    factor of 2 makes the middle integrand uniformly larger, hence a
    larger alpha_bar. The NEAR wall at the edge is handled separately via
    the +0.5 term in svf_from_hw(edge=True), not through alpha_bar itself.

    Returns
    -------
    np.ndarray
        alpha_bar [rad], same shape as hw.
    """
    gamma = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_samples)
    cos_gamma = np.cos(gamma)
    hw = np.atleast_1d(np.asarray(hw, dtype=float))
    coeff = hw if edge else 2.0 * hw
    integrand = np.arctan(coeff[:, None] * cos_gamma[None, :])
    return _trapz(integrand, gamma, axis=1) / np.pi


def svf_from_hw(hw: np.ndarray, edge: bool = False) -> np.ndarray:
    """
    Sky View Factor (SVF, dimensionless in [0, 1]) as a function of H/W.

        SVF_middle = cos^2(alpha_bar_m)             [Eq. 2] -> 0 as H/W -> inf
        SVF_edge   = 0.5*cos^2(alpha_bar_e) + 0.5    [Eq. 4] -> 0.5 as H/W -> inf
                     (near wall always blocks one hemisphere; far wall
                     blocks progressively less as H/W grows)
    """
    alpha = mean_obstruction_angle_rad(hw, edge=edge)
    if edge:
        return 0.5 * np.cos(alpha) ** 2 + 0.5
    return np.cos(alpha) ** 2


def diffuse_loss_pct(hw: np.ndarray, edge: bool = False) -> np.ndarray:
    """Isotropic-diffuse irradiance loss [%] = (1 - SVF) x 100."""
    return (1.0 - svf_from_hw(hw, edge=edge)) * 100.0


# ==============================================================================
# DIRECT (BEAM) COMPONENT, E-W STREET — Brito et al. 2024, Eq. 6-11
# ==============================================================================

def solar_declination_rad(doy: int) -> float:
    """
    Solar declination delta [rad] for day-of-year `doy` (Spencer 1971).

    The Spencer formula outputs radians (range ~= +/-0.4097 rad
    = +/-23.45 deg): do NOT apply np.radians() to the result.
    """
    B = 2.0 * np.pi * (doy - 1) / 365.0
    return (
        0.006918
        - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
        - 0.002697 * np.cos(3 * B) + 0.001480 * np.sin(3 * B)
    )


def sunrise_hour_angle_rad(lat_rad: float, decl_rad: float) -> float:
    """Sunrise / sunset hour angle omega_s [rad] (Brito Eq. 8)."""
    arg = -np.tan(lat_rad) * np.tan(decl_rad)
    return np.arccos(np.clip(arg, -1.0, 1.0))


def incidence_angle_rad(lat_rad: float, decl_rad: float,
                        omega: np.ndarray) -> np.ndarray:
    """
    Solar incidence angle theta on a horizontal surface [rad] (Brito Eq. 7).

        cos(theta) = sin(delta)*sin(phi) + cos(delta)*cos(phi)*cos(omega)
    """
    cos_theta = (np.sin(decl_rad) * np.sin(lat_rad)
                 + np.cos(decl_rad) * np.cos(lat_rad) * np.cos(omega))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def sun_azimuth_rad(lat_rad: float, decl_rad: float,
                    theta_rad: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Solar azimuth angle psi [rad], measured from south (Brito Eq. 11).

        psi = acos((cos(theta)*sin(phi) - sin(delta)) / (sin(theta)*cos(phi)))

    Convention: psi = 0 due south. Morning (omega <= 0): psi in [0, pi];
    afternoon (omega > 0): 2*pi - psi, i.e. -psi modulo 2*pi.
    """
    sin_theta = np.sin(theta_rad)
    safe_sin = np.where(np.abs(sin_theta) < 1e-9, 1e-9, sin_theta)
    arg = ((np.cos(theta_rad) * np.sin(lat_rad) - np.sin(decl_rad))
           / (safe_sin * np.cos(lat_rad)))
    psi = np.arccos(np.clip(arg, -1.0, 1.0))
    return np.where(omega > 0, 2.0 * np.pi - psi, psi)


def obstruction_angle_at_azimuth_rad(hw: float, gamma: np.ndarray,
                                     edge: bool = False) -> np.ndarray:
    """
    Obstruction angle alpha(gamma) [rad] at solar azimuth gamma
    (Brito Eq. 1a-b), gamma measured from the canyon-wall normal
    (= south, for an E-W street).

    gamma is first wrapped to (-pi, pi], so that the afternoon azimuths
    returned by sun_azimuth_rad() (2*pi - psi) are handled like the morning
    ones in the |gamma| test of the edge case.
    """
    gamma = np.mod(np.asarray(gamma, dtype=float) + np.pi, 2.0 * np.pi) - np.pi
    if edge:
        # Near wall: full block (pi/2) for |gamma| > pi/2 (behind the
        # vehicle). Far wall: atan(H/W * |cos(gamma)|) for |gamma| <= pi/2.
        return np.where(
            np.abs(gamma) <= np.pi / 2.0,
            np.arctan(hw * np.abs(np.cos(gamma))),
            np.pi / 2.0,
        )
    return np.arctan(2.0 * hw * np.abs(np.cos(gamma)))


def direct_loss_pct_ew_street(
    hw_array: np.ndarray, doy: int, lat_rad: float,
    n_omega: int = N_OMEGA_SAMPLES, edge: bool = False,
) -> np.ndarray:
    """
    Geometric direct (beam) irradiance loss [%] for an E-W infinite canyon,
    for ONE day of year.

    For each H/W value, integrates DNI*cos(theta) over daylight hours,
    counting only the hours where solar elevation exceeds the obstruction
    angle at the actual solar azimuth (Brito Eq. 9-10). DNI = 1 (uniform)
    gives a purely geometric loss factor.

    E-W orientation is the WORST CASE for direct losses at mid-latitudes:
    the sun crosses roughly perpendicular to the street at transit,
    maximising time spent below the obstruction angle. N-S orientation is
    the best case (sun travels along the canyon axis, obstruction angle
    along that axis is 0).

    Sharp H/W threshold
    ---------------------
    For a given latitude and day, a critical H/W* = tan(elevation_noon)/2
    (middle) exists below which the noon sun clears the canyon, and above
    which it is permanently blocked — a near-discontinuous jump in
    individual-day curves. Use annual_mean_direct_loss_pct_ew() for any
    comparison that should be smooth / comparable to published annual-mean
    figures.

    Returns
    -------
    np.ndarray
        Loss_BHI [%], same length as hw_array.
    """
    decl = solar_declination_rad(doy)
    omega_s = sunrise_hour_angle_rad(lat_rad, decl)
    omega = np.linspace(-omega_s, omega_s, n_omega)
    theta = incidence_angle_rad(lat_rad, decl, omega)
    psi = sun_azimuth_rad(lat_rad, decl, theta, omega)
    gamma_sun = psi   # azimuth from south = canyon normal for an E-W street
    cos_theta = np.maximum(np.cos(theta), 0.0)
    I_total = _trapz(cos_theta, omega)
    if I_total <= 0:
        return np.zeros(len(hw_array))

    losses = np.empty(len(hw_array))
    for i, hw in enumerate(hw_array):
        alpha_sun = obstruction_angle_at_azimuth_rad(hw, gamma_sun, edge=edge)
        sun_elev = np.pi / 2.0 - theta
        visible = sun_elev > alpha_sun
        I_visible = _trapz(np.where(visible, cos_theta, 0.0), omega)
        losses[i] = (1.0 - I_visible / I_total) * 100.0
    return losses


def annual_mean_direct_loss_pct_ew(
    hw_array: np.ndarray, lat_rad: float, edge: bool = False,
    n_doy: int = N_DOY_ANNUAL, n_omega: int = N_OMEGA_SAMPLES,
) -> np.ndarray:
    """
    Annual-mean geometric direct (beam) irradiance loss [%] for an E-W
    canyon — averages direct_loss_pct_ew_street() over `n_doy` equally-
    spaced days, weighting each day by its total clear-sky BHI (roughly
    proportional to day length and solar altitude). This is the quantity
    comparable to Brito et al. (2024) Fig. 7's solid lines, which use
    annual-mean irradiance rather than individual days.

    The loop is O(n_doy x len(hw_array) x n_omega), the expensive part of
    the model (~1 minute for n_doy = 365, n_omega = 2001 and ~800 H/W
    values): compute it once and reuse it across figures (as
    hw_ratio_impact_analysis.py does for M4/M5/M6).

    Returns
    -------
    np.ndarray
        Annual-mean Loss_BHI [%], same length as hw_array.
    """
    doys = np.round(np.linspace(1, 365, n_doy)).astype(int)
    weighted_loss = np.zeros(len(hw_array))
    total_weight = 0.0

    for doy in doys:
        decl = solar_declination_rad(int(doy))
        omega_s = sunrise_hour_angle_rad(lat_rad, decl)
        omega = np.linspace(-omega_s, omega_s, n_omega)
        theta = incidence_angle_rad(lat_rad, decl, omega)
        cos_theta = np.maximum(np.cos(theta), 0.0)
        I_total = _trapz(cos_theta, omega)
        if I_total <= 0:
            continue
        psi = sun_azimuth_rad(lat_rad, decl, theta, omega)
        gamma_sun = psi
        for i, hw in enumerate(hw_array):
            alpha_sun = obstruction_angle_at_azimuth_rad(hw, gamma_sun, edge=edge)
            visible = (np.pi / 2.0 - theta) > alpha_sun
            I_vis = _trapz(np.where(visible, cos_theta, 0.0), omega)
            weighted_loss[i] += (1.0 - I_vis / I_total) * 100.0 * I_total
        total_weight += I_total

    return weighted_loss / total_weight if total_weight > 0 else weighted_loss


# ==============================================================================
# COMBINED DIFFUSE + DIRECT
# ==============================================================================

def total_loss_pct(
    hw: np.ndarray, bhi_loss: np.ndarray, diffuse_fraction: float,
    edge: bool = False,
) -> np.ndarray:
    """
    Total irradiance loss [%]:
        Loss_total = f_d * Loss_DHI + (1 - f_d) * Loss_BHI

    Parameters
    ----------
    hw : np.ndarray
        H/W array.
    bhi_loss : np.ndarray
        Pre-computed BHI loss array (same shape as hw) — either a single
        day's direct_loss_pct_ew_street() output or (strongly recommended
        for anything compared against Brito Fig. 7 or real measured data)
        annual_mean_direct_loss_pct_ew()'s output.
    diffuse_fraction : float
        f_d = DHI / GHI. 0 -> beam-dominated (clear sky); 1 -> fully overcast.
    edge : bool
        Street position for the DHI component (must match the position
        used to compute bhi_loss, for a consistent combined estimate).
    """
    loss_dhi = diffuse_loss_pct(hw, edge=edge)
    return diffuse_fraction * loss_dhi + (1.0 - diffuse_fraction) * bhi_loss


# ==============================================================================
# SELF-TEST  (named assertion suite — A1 to A13)
# ==============================================================================

def _run_assertions() -> None:
    """Physical-invariant assertion suite A1-A13 (run after any change to
    this module: `python brito_svf_model.py`)."""
    lat_rad_lisbon = np.radians(38.76)

    # A1: SVF = 1 when H/W = 0 (open sky)
    svf_open = svf_from_hw(np.array([0.0]), edge=False)[0]
    assert abs(svf_open - 1.0) < 1e-6, \
        f"A1 FAIL: SVF(H/W=0) should be 1.0, got {svf_open:.6f}"
    print("  A1 PASS: SVF(H/W=0) = 1.0 (open sky)")

    # A2: diffuse loss = 0 when H/W = 0
    loss_open = diffuse_loss_pct(np.array([0.0]), edge=False)[0]
    assert abs(loss_open) < 1e-6, \
        f"A2 FAIL: Loss_DHI(H/W=0) should be 0.0, got {loss_open:.6f}"
    print("  A2 PASS: Loss_DHI(H/W=0) = 0 %")

    # A3: diffuse loss (middle) -> 100 % as H/W -> infinity
    loss_inf = diffuse_loss_pct(np.array([1e6]), edge=False)[0]
    assert loss_inf > 99.9, \
        f"A3 FAIL: Loss_DHI(H/W->inf) should approach 100 %, got {loss_inf:.4f}"
    print(f"  A3 PASS: Loss_DHI(H/W=1e6, middle) = {loss_inf:.4f} %  (-> 100 %)")

    # A4: diffuse loss (edge) -> 50 % as H/W -> infinity
    loss_edge_inf = diffuse_loss_pct(np.array([1e6]), edge=True)[0]
    assert abs(loss_edge_inf - 50.0) < 0.01, \
        f"A4 FAIL: Loss_DHI(H/W->inf, edge) should -> 50 %, got {loss_edge_inf:.4f}"
    print(f"  A4 PASS: Loss_DHI(H/W=1e6, edge) = {loss_edge_inf:.4f} %  (-> 50 %)")

    # A5: diffuse loss (middle) monotonically non-decreasing in H/W
    hw_test = np.linspace(0.01, 8.0, 200)
    loss_mid = diffuse_loss_pct(hw_test, edge=False)
    assert np.all(np.diff(loss_mid) >= 0), \
        "A5 FAIL: Loss_DHI(middle) must be monotonically non-decreasing in H/W"
    print("  A5 PASS: Loss_DHI(middle) is monotonically non-decreasing in H/W")

    # A6: SVF_middle + Loss_DHI_middle/100 = 1.0 (identity by construction)
    hw_check = np.array([0.5, 1.0, 2.0, 4.0])
    svf_c = svf_from_hw(hw_check, edge=False)
    loss_c = diffuse_loss_pct(hw_check, edge=False)
    resid = np.abs(svf_c + loss_c / 100.0 - 1.0)
    assert np.all(resid < 1e-10), \
        f"A6 FAIL: SVF + Loss_DHI/100 != 1 (max residual {resid.max():.2e})"
    print(f"  A6 PASS: SVF + Loss_DHI/100 = 1.0  (max residual {resid.max():.2e})")

    # A7: direct loss = 0 when H/W = 0
    loss_dir_open = direct_loss_pct_ew_street(np.array([0.0]), 80, lat_rad_lisbon)[0]
    assert abs(loss_dir_open) < 1e-3, \
        f"A7 FAIL: Loss_BHI(H/W=0) should be ~0, got {loss_dir_open:.4f}"
    print(f"  A7 PASS: Loss_BHI(H/W=0) = {loss_dir_open:.4f} %  (~0)")

    # A8: direct loss > 0 for H/W = 0.7 at winter solstice, Lisbon
    loss_win = direct_loss_pct_ew_street(np.array([0.7]), 355, lat_rad_lisbon)[0]
    assert loss_win > 50.0, \
        f"A8 FAIL: Loss_BHI(H/W=0.7, winter) should be > 50 %, got {loss_win:.1f}"
    print(f"  A8 PASS: Loss_BHI(H/W=0.7, winter) = {loss_win:.1f} %  (> 50 %)")

    # A9: Spencer declination at equinox (DOY 80) is ~0
    decl_eq = np.degrees(solar_declination_rad(80))
    assert abs(decl_eq) < 0.5, \
        f"A9 FAIL: declination at DOY=80 should be ~0 deg, got {decl_eq:.3f} deg"
    print(f"  A9 PASS: declination at DOY=80 = {decl_eq:.3f} deg  (~0 deg)")

    # A10: Spencer declination at summer solstice (DOY 172) is ~+23.45 deg
    decl_sum = np.degrees(solar_declination_rad(172))
    assert 22.0 < decl_sum < 25.0, \
        f"A10 FAIL: declination at DOY=172 should be ~23.45 deg, got {decl_sum:.3f} deg"
    print(f"  A10 PASS: declination at DOY=172 = {decl_sum:.3f} deg  (~+23.45 deg)")

    # A11: H/W critical threshold for E-W canyon, equinox, Lisbon
    #      Below threshold (H/W=0.50): Loss_BHI ~= 0; above (H/W=0.70): >> 50 %
    loss_below = direct_loss_pct_ew_street(np.array([0.50]), 80, lat_rad_lisbon)[0]
    loss_above = direct_loss_pct_ew_street(np.array([0.70]), 80, lat_rad_lisbon)[0]
    assert loss_below < 1.0, \
        f"A11a FAIL: Loss_BHI(H/W=0.5, equinox) should be ~0 %, got {loss_below:.1f}%"
    assert loss_above > 50.0, \
        f"A11b FAIL: Loss_BHI(H/W=0.7, equinox) should be > 50 %, got {loss_above:.1f}%"
    print(f"  A11 PASS: H/W threshold (Lisbon equinox, E-W): "
          f"H/W=0.50->{loss_below:.0f}%, H/W=0.70->{loss_above:.0f}%")

    # A12: annual-mean direct loss is in % and consistent with single days
    hw_a = np.array([0.0, 0.7, 50.0])
    ann = annual_mean_direct_loss_pct_ew(hw_a, lat_rad_lisbon, n_doy=24, n_omega=501)
    assert abs(ann[0]) < 1e-3 and 10.0 < ann[1] < 100.0 and ann[2] > 99.0, \
        f"A12 FAIL: annual Loss_BHI should be in % (0 / 10-100 / ~100), got {ann}"
    print(f"  A12 PASS: annual Loss_BHI [%] at H/W = 0 / 0.7 / 50: "
          f"{ann[0]:.1f} / {ann[1]:.1f} / {ann[2]:.1f}")

    # A13: edge of street, no buildings -> no direct loss, morning and afternoon
    loss_edge_open = direct_loss_pct_ew_street(np.array([0.0]), 80, lat_rad_lisbon,
                                               edge=True)[0]
    assert abs(loss_edge_open) < 1e-3, \
        f"A13 FAIL: Loss_BHI(H/W=0, edge) should be ~0, got {loss_edge_open:.2f}"
    print(f"  A13 PASS: Loss_BHI(H/W=0, edge) = {loss_edge_open:.4f} %  (~0)")

    print("\n  All assertions passed.")


if __name__ == "__main__":
    print("=" * 70)
    print("  brito_svf_model.py -- self-test")
    print("=" * 70)
    _run_assertions()