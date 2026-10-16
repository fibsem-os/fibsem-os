"""The bounded fit core for the correlation point fits (FIB-1218).

Synthetic spots and peaks with known answers, plus two real z profiles. No Qt.
"""

import numpy as np
import pytest

from fibsem.correlation.fitting import (
    FitOutcome,
    Polarity,
    fit_peak_1d,
    fit_spot_2d,
)


def _spot(
    cx,
    cy,
    *,
    size=61,
    sigma=3.0,
    amplitude=80.0,
    polarity=Polarity.DARK,
    noise=2.0,
    seed=0,
    sx=None,
    sy=None,
):
    yy, xx = np.indices((size, size), dtype=float)
    sx = sx or sigma
    sy = sy or sigma
    g = np.exp(-((xx - cx) ** 2) / (2 * sx**2) - ((yy - cy) ** 2) / (2 * sy**2))
    sign = -1.0 if polarity is Polarity.DARK else 1.0
    rng = np.random.default_rng(seed)
    return 150.0 + sign * amplitude * g + rng.normal(0, noise, (size, size))


def _peak(
    c, *, n=30, sigma=2.0, amplitude=50.0, polarity=Polarity.BRIGHT, noise=0.5, seed=0
):
    t = np.arange(n, dtype=float)
    sign = -1.0 if polarity is Polarity.DARK else 1.0
    rng = np.random.default_rng(seed)
    return (
        100.0
        + sign * amplitude * np.exp(-((t - c) ** 2) / (2 * sigma**2))
        + (rng.normal(0, noise, n))
    )


# --- 2D: finds the spot, either polarity, from anywhere in the window ---------


@pytest.mark.parametrize("polarity", [Polarity.DARK, Polarity.BRIGHT])
@pytest.mark.parametrize("dx, dy", [(0, 0), (4, -3), (-8, 6), (10, 10)])
def test_spot_found_from_the_window_extreme(polarity, dx, dy):
    img = _spot(30 + dx, 30 + dy, polarity=polarity)
    r = fit_spot_2d(img, 30, 30, 15, polarity, start="extreme")
    assert r.outcome is FitOutcome.FITTED, r.reason
    assert r.position == pytest.approx((30 + dx, 30 + dy), abs=0.5)
    assert r.sigma == pytest.approx((3.0, 3.0), abs=0.5)
    assert r.contrast_to_noise > 10


def test_spot_from_the_click_when_the_click_is_on_it():
    img = _spot(31.4, 28.7)
    r = fit_spot_2d(img, 31, 29, 15, Polarity.DARK)  # start="center"
    assert r.ok and r.position == pytest.approx((31.4, 28.7), abs=0.3)


def test_an_elliptical_spot_keeps_both_widths():
    img = _spot(30, 30, sx=2.0, sy=5.0)
    r = fit_spot_2d(img, 30, 30, 15, Polarity.DARK)
    assert r.ok and r.sigma == pytest.approx((2.0, 5.0), abs=0.4)


def test_the_wrong_polarity_finds_nothing():
    """A dark burn fitted as bright must not come back as a fit: the amplitude
    is bounded to the asked-for sign, so it fits noise and says so."""
    img = _spot(30, 30, polarity=Polarity.DARK)
    r = fit_spot_2d(img, 30, 30, 15, Polarity.BRIGHT, start="center")
    assert r.outcome is not FitOutcome.FITTED
    assert r.reason


def test_noise_alone_is_not_a_spot():
    rng = np.random.default_rng(3)
    img = 150.0 + rng.normal(0, 2.0, (61, 61))
    r = fit_spot_2d(img, 30, 30, 15, Polarity.DARK, start="extreme")
    assert r.outcome is FitOutcome.FAILED
    assert "contrast" in r.reason or "narrow" in r.reason or "wide" in r.reason


def test_a_flat_window_is_not_a_spot():
    r = fit_spot_2d(np.full((61, 61), 150.0), 30, 30, 15, Polarity.DARK)
    assert r.outcome is FitOutcome.FAILED


def test_a_gradient_is_too_wide_to_be_a_spot():
    """The FIB trench: a dark region running off the side of the window."""
    img = np.tile(np.linspace(200, 20, 61), (61, 1))
    r = fit_spot_2d(img, 30, 30, 15, Polarity.DARK, start="extreme")
    assert r.outcome is not FitOutcome.FITTED


def test_a_burn_with_a_bright_halo_is_a_burn():
    """METEOR FIB burns are a dark core in a bright ring. The model has no
    ring, so it leaves a large residual; contrast is judged against the
    window's pixel noise, not that residual, or real burns read as no contrast
    (10 of 94 saved picks did, with the residual; this one read 1.5x)."""
    yy, xx = np.indices((61, 61), dtype=float)
    r2 = (xx - 30) ** 2 + (yy - 30) ** 2
    ring = 250.0 * np.exp(-((np.sqrt(r2) - 6.0) ** 2) / (2 * 3.0**2))
    core = 70.0 * np.exp(-r2 / (2 * 2.0**2))
    img = 100.0 + ring - core + np.random.default_rng(1).normal(0, 2.0, (61, 61))
    r = fit_spot_2d(img, 31, 29, 15, Polarity.DARK)
    assert r.ok, r.reason
    assert r.position == pytest.approx((30, 30), abs=0.5)
    assert r.residual > r.noise  # the ring is in the residual, not the noise


# --- 2D: the window and the edge rule ------------------------------------------


@pytest.mark.parametrize("cx, cy", [(2, 30), (30, 2), (58, 58)])
def test_the_window_is_clamped_to_the_image(cx, cy):
    """A click near an edge: no wrap, and the answer comes back absolute."""
    img = _spot(cx, cy)
    r = fit_spot_2d(img, cx, cy, 15, Polarity.DARK)
    (x0, x1), (y0, y1) = r.window
    assert 0 <= x0 <= x1 <= 60 and 0 <= y0 <= y1 <= 60
    if r.position is not None:
        assert 0 <= r.position[0] <= 60 and 0 <= r.position[1] <= 60


def test_a_spot_past_the_window_edge_is_found_by_recentring_once():
    """The spot sits just outside a small window: the fit lands on the edge,
    the window moves once, and the spot is then inside."""
    img = _spot(41, 30)
    r = fit_spot_2d(img, 30, 30, 8, Polarity.DARK, start="extreme")
    assert r.ok, r.reason
    assert r.recentred
    assert r.position == pytest.approx((41, 30), abs=0.5)


def test_recentring_does_not_go_looking_somewhere_else():
    """Re-centring confirms a feature at the window's edge. When the first fit
    found nothing, it is not re-centred; and an answer far outside the first
    window is a different feature, not a confirmation."""
    img = _spot(52, 30, sigma=2.0)  # a spot well outside a window at x 15-45
    rng = np.random.default_rng(5)
    img[15:46, 15:46] = 150.0 + rng.normal(0, 2.0, (31, 31))
    r = fit_spot_2d(img, 30, 30, 15, Polarity.DARK, start="center")
    assert not r.ok
    if r.position is not None:
        assert r.position[0] < 52 - 4


def test_without_recentring_an_edge_fit_is_reported_not_returned():
    img = _spot(41, 30)
    r = fit_spot_2d(img, 30, 30, 8, Polarity.DARK, start="extreme", recentre=False)
    assert r.outcome is FitOutcome.AT_EDGE
    assert "window edge" in r.reason


def test_a_spot_at_the_image_edge_cannot_be_confirmed():
    img = _spot(0.5, 30)
    r = fit_spot_2d(img, 3, 30, 10, Polarity.DARK, start="extreme")
    assert r.outcome in (FitOutcome.AT_EDGE, FitOutcome.FAILED)
    assert not r.ok


def test_a_window_too_small_to_fit_fails_with_a_reason():
    r = fit_spot_2d(_spot(30, 30), 30, 30, 0, Polarity.DARK)
    assert r.outcome is FitOutcome.FAILED and "too small" in r.reason


def test_a_point_outside_the_image_fails_with_a_reason():
    r = fit_spot_2d(_spot(30, 30), -40, 30, 15, Polarity.DARK)
    assert r.outcome is FitOutcome.FAILED and "too small" in r.reason


# --- 1D --------------------------------------------------------------------------


@pytest.mark.parametrize("polarity", [Polarity.DARK, Polarity.BRIGHT])
@pytest.mark.parametrize("c", [8.0, 12.3, 18.6])
def test_peak_found(polarity, c):
    r = fit_peak_1d(_peak(c, polarity=polarity), 14, 10, polarity)
    assert r.ok, r.reason
    assert r.position[0] == pytest.approx(c, abs=0.3)


def test_a_peak_past_the_window_is_found_by_recentring_once():
    """Window 4-16, peak at 18: only its rising side is in the window."""
    r = fit_peak_1d(_peak(18.0), 10, 6, Polarity.BRIGHT)
    assert r.ok and r.recentred, r.reason
    assert r.position[0] == pytest.approx(18.0, abs=0.3)


def test_a_rising_profile_has_no_peak_to_find():
    """A profile that only rises to the end of the data: the best fit is at
    the edge, and the edge is the end of the stack, so it can't be confirmed."""
    profile = np.linspace(0, 100, 21) ** 1.5
    r = fit_peak_1d(profile, 10, 10, Polarity.BRIGHT)
    assert r.outcome is FitOutcome.AT_EDGE
    assert "end of the data" in r.reason


def test_the_answer_is_always_inside_the_profile():
    """Today's z fit can return a plane outside the stack, which then indexes
    from the far end or raises IndexError. Every answer here is an index."""
    rng = np.random.default_rng(7)
    for seed in range(30):
        profile = rng.normal(100, 5, 21) + np.linspace(0, rng.normal(0, 30), 21)
        for polarity in Polarity:
            r = fit_peak_1d(profile, rng.integers(0, 21), 8, polarity)
            if r.position is not None:
                assert 0 <= r.position[0] <= 20


# --- real z profiles: the FM target on one lamella (Arctis, 21 planes) ---------
# The mean of an 11 x 11 px box through the clicked target in each plane, for
# both channels. The target is a dark dip at plane ~10.9 in Channel-01 and a
# bright peak at ~7.8 in Channel-02. Today's fit, which assumes bright, returns
# z 2.06 in Channel-01 and the app shows "Fit ok".

CH01 = [
    643.2,
    645.1,
    642.8,
    641.0,
    643.6,
    620.6,
    581.5,
    540.6,
    502.2,
    473.9,
    463.0,
    467.6,
    478.7,
    495.1,
    521.5,
    555.0,
    580.5,
    606.2,
    622.0,
    633.8,
    641.6,
]
CH02 = [
    842.5,
    892.5,
    940.8,
    1023.0,
    1139.2,
    1251.7,
    1318.9,
    1372.7,
    1331.8,
    1339.8,
    1257.8,
    1208.9,
    1083.9,
    1010.7,
    933.3,
    841.2,
    815.3,
    799.6,
    749.4,
    745.6,
    709.8,
]


def test_real_target_bright_in_the_channel_that_shows_it_bright():
    r = fit_peak_1d(CH02, 8, 10, Polarity.BRIGHT)
    assert r.ok, r.reason
    assert r.position[0] == pytest.approx(7.8, abs=0.6)


def test_real_target_dark_in_the_channel_that_shows_it_dark():
    r = fit_peak_1d(CH01, 8, 10, Polarity.DARK)
    assert r.ok, r.reason
    assert r.position[0] == pytest.approx(10.9, abs=0.6)


def test_real_target_fitted_bright_where_it_is_dark_is_not_a_fit():
    """Today's answer here is z 2.06, shown as "Fit ok"."""
    r = fit_peak_1d(CH01, 8, 10, Polarity.BRIGHT)
    assert not r.ok
    assert r.reason


# --- deterministic --------------------------------------------------------------


def test_repeated_calls_give_identical_answers():
    """Today's xy fit gave two answers 0.07 px apart over eight identical calls
    on one real point. The same input must give the same answer, exactly."""
    img = _spot(33.3, 27.9, noise=6.0)
    first = fit_spot_2d(img, 30, 30, 15, Polarity.DARK, start="extreme")
    for _ in range(20):
        assert fit_spot_2d(img, 30, 30, 15, Polarity.DARK, start="extreme") == first
    profile = _peak(11.4, noise=3.0)
    first1 = fit_peak_1d(profile, 10, 8, Polarity.BRIGHT)
    for _ in range(20):
        assert fit_peak_1d(profile, 10, 8, Polarity.BRIGHT) == first1


def test_a_peak_near_the_end_of_the_stack_counts_when_it_turns_over():
    """A peak 5 planes from the end, 40% down by the last plane, is a fit; a
    profile that only rises to the end, or turns over by a sliver, is not."""
    t = np.arange(33, dtype=float)
    turns = 300.0 + 1400.0 * np.exp(-((t - 27.0) ** 2) / (2 * 4.0**2))
    r = fit_peak_1d(turns, 16, 33, Polarity.BRIGHT)
    assert r.ok, r.reason
    assert r.position[0] == pytest.approx(27.0, abs=0.5)

    rising = 300.0 + 1400.0 * np.exp(-((t - 40.0) ** 2) / (2 * 6.0**2))
    assert fit_peak_1d(rising, 16, 33, Polarity.BRIGHT).outcome is FitOutcome.AT_EDGE

    r = fit_peak_1d(CH01, 8, 21, Polarity.BRIGHT)  # the real target, fitted bright
    assert not r.ok
