"""Device parameters as form fields: the declared display hint, with what the
instrument reports in place of the declared limits and choices, and the beam and
detector panels built from it."""

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
from fibsem.ui.widgets.custom_widgets import ValueSpinBox, unit_suffix
from fibsem.ui.widgets.detector_settings_widget import FibsemDetectorSettingsWidget
from fibsem.ui.widgets.form_builder import (
    bound_spec,
    configure_spinbox,
    parameter_spec,
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
    spec = parameter_spec(Beam.hfw)

    assert spec.label == "Field of View"
    assert spec.scale == 1e6
    assert spec.unit == "m"
    # the declared 1 nm to 10 mm, in µm
    assert spec.shown_range == pytest.approx((1e-3, 1e4))


def test_the_reported_limits_and_choices_win(microscope):
    _report(microscope, "hfw", limits=RangeLimit(2e-6, 900e-6))
    _report(microscope, "resolution", choices=[(1024, 884)])

    hfw = parameter_spec(microscope.beams[E].parameters["hfw"])
    resolution = parameter_spec(microscope.beams[E].parameters["resolution"])

    assert hfw.shown_range == pytest.approx((2, 900))
    assert resolution.choices == ((1024, 884),)


def test_a_point_takes_its_fields_limits():
    spec = parameter_spec(Beam.shift, "x")

    assert spec.shown_range == pytest.approx((-50, 50))
    assert spec.advanced is True


def test_a_parameter_with_no_hint_is_labelled_from_its_name():
    spec = parameter_spec(Beam.angular_correction)

    assert spec.label == "Angular correction"
    assert spec.unit == "rad"
    assert spec.doc == Beam.angular_correction.doc


def test_a_spinbox_takes_the_shown_unit_step_and_range():
    spinbox = QDoubleSpinBox()
    configure_spinbox(spinbox, parameter_spec(Beam.scan_rotation))

    # a degree sign sits against its number ("90°"); other units follow a space
    assert spinbox.suffix() == DEGREE_SYMBOL
    assert spinbox.singleStep() == 180
    assert spinbox.decimals() == 0
    assert (spinbox.minimum(), spinbox.maximum()) == pytest.approx((0, 360))


def test_a_unit_suffix_follows_a_space_except_a_degree_sign():
    assert unit_suffix("µm") == " µm"
    assert unit_suffix(DEGREE_SYMBOL) == DEGREE_SYMBOL
    assert unit_suffix(None) == ""
    assert ValueSpinBox(suffix=DEGREE_SYMBOL).suffix() == DEGREE_SYMBOL


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


# -- a bound field takes its device's limits -------------------------------------


def test_a_bound_field_takes_what_its_device_reports(microscope):
    hfw = ImageSettings.params["hfw"]
    reported = microscope.beams[E].parameters["hfw"].limits
    assert bound_spec(hfw, {"beam": microscope.beams[E]}).limits == reported


def test_without_its_device_a_bound_field_takes_what_the_device_declares():
    hfw = ImageSettings.params["hfw"]
    assert bound_spec(hfw).limits == Beam.hfw.limits


def test_a_device_that_cannot_answer_gives_what_it_declares():
    class Unreachable:
        @property
        def parameters(self):
            raise ConnectionError("the device's computer is gone")

    hfw = ImageSettings.params["hfw"]
    assert bound_spec(hfw, {"beam": Unreachable()}).limits == Beam.hfw.limits


def test_a_field_bound_to_nothing_comes_back_as_it_is():
    autocontrast = ImageSettings.params["autocontrast"]
    assert bound_spec(autocontrast, {"beam": None}) is autocontrast
