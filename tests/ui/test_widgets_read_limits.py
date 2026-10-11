"""The widgets offer the beam's own resolutions and ranges, and keep the standard
ones where the beam reports none."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.devices.beam import (  # noqa: E402
    DEFAULT_DWELL_TIME_LIMITS,
    DEFAULT_HFW_LIMITS,
    STANDARD_RESOLUTIONS,
)
from fibsem.devices.core import ParameterMetadata  # noqa: E402
from fibsem.structures import BeamType, ImageSettings, RangeLimit  # noqa: E402
from fibsem.ui.utils import beam_limits  # noqa: E402
from fibsem.ui.widgets.beam_settings_widget import (
    FibsemBeamSettingsWidget,  # noqa: E402
)
from fibsem.ui.widgets.fibsem_overview_settings_widget import (  # noqa: E402
    FibsemOverviewSettingsWidget,
)
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget  # noqa: E402

_app = QApplication.instance() or QApplication([])

E, I = BeamType.ELECTRON, BeamType.ION
RESOLUTIONS = [(512, 442), (1024, 884), (2048, 1768)]


def _demo():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    return microscope


def _report(microscope, beam_type, name, **metadata):
    """The beam reports ``metadata`` for ``name``, as a driver's metadata_ would."""
    microscope.beams[beam_type].parameters[name].metadata = ParameterMetadata(
        **metadata
    )


def _items(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def _range(spinbox):
    return spinbox.minimum(), spinbox.maximum()


def test_beam_limits_reads_the_device():
    microscope = _demo()
    assert beam_limits(microscope, "scan_rotation", E) == RangeLimit(0.0, 2 * math.pi)
    assert beam_limits(microscope, "not_a_parameter", E) is None
    _report(microscope, E, "hfw")
    assert beam_limits(microscope, "hfw", E) == DEFAULT_HFW_LIMITS
    # no such beam, or no microscope: the range every beam declares
    microscope.beams = {E: microscope.beams[E]}
    assert beam_limits(microscope, "hfw", I) == DEFAULT_HFW_LIMITS
    assert beam_limits(None, "dwell_time", I) == DEFAULT_DWELL_TIME_LIMITS


def test_beam_settings_follow_the_beam():
    microscope = _demo()
    _report(microscope, E, "resolution", choices=RESOLUTIONS)
    _report(microscope, E, "hfw", limits=RangeLimit(1e-6, 2e-3))
    _report(microscope, E, "dwell_time", limits=RangeLimit(25e-9, 500e-6))
    widget = FibsemBeamSettingsWidget(microscope=microscope, beam_type=E)
    widget.populate_beam_combos()

    assert _items(widget.resolution_combo) == RESOLUTIONS
    assert _range(widget.hfw_spinbox) == pytest.approx((1, 2000))
    assert _range(widget.dwell_time_spinbox) == pytest.approx((0.025, 500))
    assert _range(widget.scan_rotation_spinbox) == pytest.approx((0, 360))


def test_beam_settings_keep_the_standard_values_when_the_beam_reports_none():
    microscope = _demo()
    _report(microscope, E, "resolution")
    _report(microscope, E, "dwell_time")
    widget = FibsemBeamSettingsWidget(microscope=microscope, beam_type=E)
    widget.populate_beam_combos()

    assert _items(widget.resolution_combo) == list(STANDARD_RESOLUTIONS)
    # the range every beam declares, 1 ns to 1 ms
    assert _range(widget.dwell_time_spinbox) == pytest.approx((0.001, 1000))


def test_image_settings_offer_what_the_beam_acquires():
    microscope = _demo()
    _report(microscope, I, "resolution", choices=RESOLUTIONS)
    _report(microscope, I, "hfw", limits=RangeLimit(1e-6, 900e-6))
    widget = ImageSettingsWidget()
    standard_dwell = _range(widget.dwell_time_spinbox)
    widget.use_beam(microscope, I)

    assert _items(widget.resolution_combo) == RESOLUTIONS
    assert _range(widget.hfw_spinbox) == pytest.approx((1, 900))
    # the beam reports no dwell limits: the standard range stays
    assert _range(widget.dwell_time_spinbox) == standard_dwell


def test_image_settings_keep_a_resolution_the_beam_does_not_list():
    """A protocol's resolution is shown as it is, not snapped to a neighbour."""
    microscope = _demo()
    _report(microscope, I, "resolution", choices=RESOLUTIONS)
    widget = ImageSettingsWidget()
    widget.use_beam(microscope, I)

    widget.update_from_settings(ImageSettings(resolution=(3072, 2048)))

    assert widget.get_settings().resolution == (3072, 2048)


def test_the_overview_offers_the_standard_resolutions_and_keeps_a_saved_one():
    """Square tiles are no longer offered, but an overview saved with one keeps it."""
    widget = FibsemOverviewSettingsWidget()
    assert _items(widget.combo_resolution) == list(STANDARD_RESOLUTIONS)

    settings = widget.get_settings()
    settings.image_settings.resolution = (1024, 1024)
    widget.update_from_settings(settings)

    assert widget.get_settings().image_settings.resolution == (1024, 1024)


def test_a_listed_resolution_is_found_when_built_elsewhere():
    """A resolution read from a file is an equal tuple, not the combo's own object."""
    widget = ImageSettingsWidget()
    count = widget.resolution_combo.count()
    widget.update_from_settings(ImageSettings(resolution=tuple([768, 512])))

    assert widget.resolution_combo.count() == count
    assert widget.get_settings().resolution == (768, 512)


def test_the_overview_offers_the_selected_beams_resolutions():
    microscope = _demo()
    _report(microscope, I, "resolution", choices=RESOLUTIONS)
    widget = FibsemOverviewSettingsWidget()
    widget.set_microscope(microscope)

    widget.combo_beam.set_value(I)
    assert _items(widget.combo_resolution) == RESOLUTIONS
    # the electron beam's 1536x1024 is not the ion beam's: the nearest one is
    assert widget.get_settings().image_settings.resolution == (1024, 884)

    settings = widget.get_settings()
    settings.image_settings.beam_type = E
    settings.image_settings.resolution = (768, 512)
    widget.update_from_settings(settings)
    assert _items(widget.combo_resolution) == list(STANDARD_RESOLUTIONS)
    assert widget.get_settings().image_settings.resolution == (768, 512)


def test_the_overview_bounds_field_of_view_and_dwell_by_the_selected_beam():
    microscope = _demo()
    _report(microscope, I, "hfw", limits=RangeLimit(1e-6, 900e-6))
    widget = FibsemOverviewSettingsWidget()
    # no microscope: the range every beam declares
    assert _range(widget.spin_hfw) == pytest.approx((0.001, 10000), abs=0.01)

    widget.set_microscope(microscope)
    widget.combo_beam.set_value(I)
    assert _range(widget.spin_hfw) == pytest.approx((1, 900))
