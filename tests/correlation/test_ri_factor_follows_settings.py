"""The depth-correction factor is ζ of the parameters on screen (FIB-1234).

Applying a protocol's RI settings set the boxes with their signals blocked and
never recomputed, so ζ stayed at the widget's own defaults: on a real run the
boxes described 1.567 while 1.502 was used. Two default sets disagreed (NA 0.8 /
n 1.4 in the config, 0.75 / 1.35 in the widget), the tilt lock for pre-correction
kept the 15° factor under a box reading 0°, and the run recorded the factor but
not what it was computed from.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/correlation/test_ri_factor_follows_settings.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.config import CorrelationConfig, RISettings
from fibsem.correlation.refractive_index import _LUT_PATH, lookup_zeta
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointType,
    PointXYZ,
)
from fibsem.ui.correlation.widgets.refractive_index_widget import _DEFAULTS

# These compare the factor with the real table; CI has no table (FIB-637).
requires_lut = pytest.mark.skipif(
    not _LUT_PATH.exists(), reason="real refractive-index LUT not present"
)


def _zeta(params) -> float:
    return lookup_zeta(
        params.tilt_deg, params.depth_um, params.NA, params.n2, params.wavelength_um
    )


@pytest.fixture
def tab(qapp):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    w = CorrelationTabWidget()
    yield w
    w.close()


def test_there_is_one_default_set():
    ri = RISettings()
    assert (ri.tilt_deg, ri.depth_um, ri.na, ri.n2, ri.wavelength_um) == (
        _DEFAULTS.tilt_deg,
        _DEFAULTS.depth_um,
        _DEFAULTS.NA,
        _DEFAULTS.n2,
        _DEFAULTS.wavelength_um,
    )


@requires_lut
def test_a_protocols_settings_set_the_factor_they_show(tab):
    """No FM stack, so nothing else recomputes: the settings alone must."""
    tab.set_correlation_config(
        CorrelationConfig(ri=RISettings(na=0.75, n2=1.4, wavelength_um=0.635))
    )
    ri = tab._ri_tab._ri_widget
    assert ri.get_factor() == pytest.approx(_zeta(ri.get_params()), abs=1e-3)
    assert ri.get_factor() == pytest.approx(1.567, abs=1e-3)  # was 1.502


@requires_lut
def test_the_tilt_lock_gives_the_tilt_0_factor(tab):
    ri = tab._ri_tab._ri_widget
    ri.set_tilt_locked(True)
    assert ri.get_params().tilt_deg == 0.0
    assert ri.get_factor() == pytest.approx(_zeta(ri.get_params()), abs=1e-3)
    ri.set_tilt_locked(False)
    assert ri.get_factor() == pytest.approx(_zeta(_DEFAULTS), abs=1e-3)


@requires_lut
def test_placing_the_fm_surface_arms_the_tilt_0_factor_and_records_why(tab):
    tab.set_correlation_config(
        CorrelationConfig(ri=RISettings(na=0.75, n2=1.4, wavelength_um=0.635))
    )
    tab._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)

    ri = tab._ri_tab._ri_widget
    params = ri.get_params()
    assert params.tilt_deg == 0.0
    assert tab._ri_pre_correction_factor == pytest.approx(_zeta(params), abs=1e-3)
    record = tab.data.ri_pre_correction_params
    assert record == {
        "tilt_deg": 0.0,
        "depth_um": params.depth_um,
        "na": params.NA,
        "n2": params.n2,
        "wavelength_um": params.wavelength_um,
        "computed": True,
    }


def test_an_entered_factor_is_armed_as_entered_and_says_so(tab):
    """A restored or typed factor is not ζ of the boxes; the record says so."""
    tab._ri_tab._ri_widget.set_factor(1.42)
    tab._on_canvas_add_requested(1.0, 2.0, PointType.SURFACE_FM)
    assert tab._ri_pre_correction_factor == pytest.approx(1.42)
    assert tab.data.ri_pre_correction_params["computed"] is False


def test_the_record_travels_with_the_run_and_goes_with_the_surface(tab):
    fm_surface = Coordinate(PointXYZ(1.0, 2.0, 3.0), PointType.SURFACE_FM)
    params = {"tilt_deg": 0.0, "na": 0.75, "computed": True}
    data = CorrelationInputData(
        fm_surface_coordinate=fm_surface,
        ri_pre_correction_factor=1.5,
        ri_pre_correction_params=params,
    )
    back = CorrelationInputData.from_dict(data.to_dict())
    assert back.ri_pre_correction_params == params

    tab.set_data(back)
    assert tab.data.ri_pre_correction_params == params
    tab._clear_pre_correction_factor()
    assert tab.data.ri_pre_correction_params is None
