"""Device parameters as form metadata: the declared display hint, with what the
instrument reports laid over it, and the beam and detector panels built from it."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication, QDoubleSpinBox

from fibsem import utils
from fibsem.constants import DEGREE_SYMBOL
from fibsem.devices.beam import Beam
from fibsem.structures import BeamType, ImageSettings, RangeLimit
from fibsem.ui.widgets.beam_settings_widget import FibsemBeamSettingsWidget
from fibsem.ui.widgets.detector_settings_widget import FibsemDetectorSettingsWidget
from fibsem.ui.widgets.form_builder import (
    configure_spinbox,
    parameter_field_metadata,
)

_app = QApplication.instance() or QApplication([])

E = BeamType.ELECTRON


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


def _report(microscope, name, **metadata):
    """Have the electron beam report ``metadata`` for ``name``."""
    parameter = microscope.beams[E].parameters[name]
    parameter.metadata = replace(parameter.metadata, **metadata)


def test_a_declared_parameter_reads_its_hint_and_its_declared_limits():
    metadata = parameter_field_metadata(Beam.hfw)

    assert metadata["label"] == "Field of View"
    assert metadata["scale"] == 1e6
    assert metadata["unit"] == "m"
    # the declared 1 nm to 10 mm, in µm
    assert (metadata["minimum"], metadata["maximum"]) == pytest.approx((1e-3, 1e4))


def test_the_reported_limits_and_choices_win(microscope):
    _report(microscope, "hfw", limits=RangeLimit(2e-6, 900e-6))
    _report(microscope, "resolution", choices=[(1024, 884)])

    hfw = parameter_field_metadata(microscope.beams[E].parameters["hfw"])
    resolution = parameter_field_metadata(microscope.beams[E].parameters["resolution"])

    assert (hfw["minimum"], hfw["maximum"]) == pytest.approx((2, 900))
    assert resolution["items"] == [(1024, 884)]


def test_a_point_takes_its_fields_limits():
    metadata = parameter_field_metadata(Beam.shift, "x")

    assert (metadata["minimum"], metadata["maximum"]) == pytest.approx((-50, 50))
    assert metadata["advanced"] is True


def test_a_parameter_with_no_hint_is_labelled_from_its_name():
    metadata = parameter_field_metadata(Beam.angular_correction)

    assert metadata["label"] == "Angular correction"
    assert metadata["unit"] == "rad"
    assert metadata["tooltip"] == Beam.angular_correction.doc


def test_a_spinbox_takes_the_shown_unit_step_and_range():
    spinbox = QDoubleSpinBox()
    configure_spinbox(spinbox, parameter_field_metadata(Beam.scan_rotation))

    assert spinbox.suffix() == f" {DEGREE_SYMBOL}"
    assert spinbox.singleStep() == 180
    assert spinbox.decimals() == 0
    assert (spinbox.minimum(), spinbox.maximum()) == pytest.approx((0, 360))


def test_the_beam_panel_is_labelled_and_scaled_by_the_beam(microscope):
    widget = FibsemBeamSettingsWidget(microscope=microscope, beam_type=E)
    widget.populate_beam_combos()

    assert widget.hfw_label.text() == "Field of View"
    assert widget.hfw_spinbox.suffix() == " µm"
    assert widget.working_distance_spinbox.suffix() == " mm"
    assert widget.dwell_time_spinbox.suffix() == " µs"
    assert widget.shift_label.text() == "Shift X / Y"
    assert widget.shift_x_spinbox.suffix() == " µm"
    assert widget.stigmation_x_spinbox.suffix() == ""

    widget.hfw_spinbox.setValue(150)
    widget.scan_rotation_spinbox.setValue(180)
    settings = widget.get_settings()
    assert settings.hfw == pytest.approx(150e-6)
    assert settings.scan_rotation == pytest.approx(3.14159265)


def test_the_detector_panel_is_labelled_by_the_beam(microscope):
    widget = FibsemDetectorSettingsWidget(microscope=microscope, beam_type=E)

    assert widget.brightness_label.text() == "Brightness"
    assert widget.contrast_label.text() == "Contrast"
    assert widget.type_label.text() == "Detector Type"


def test_the_image_settings_are_labelled_and_scaled_by_their_fields(microscope):
    from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget

    widget = ImageSettingsWidget(show_advanced=True)

    assert widget.hfw_label.text() == "Field of View"
    assert widget.hfw_spinbox.suffix() == " µm"
    assert widget.dwell_time_spinbox.suffix() == " µs"
    assert widget.line_integration_label.text() == "Line Integration"
    assert (
        widget.frame_integration_spinbox.minimum(),
        widget.frame_integration_spinbox.maximum(),
    ) == (1, 512)
    assert widget.get_settings().hfw == ImageSettings().hfw
