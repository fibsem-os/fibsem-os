"""``StagePositionWidget`` -- the stage readout and entry form, on its own.

The widget is cheap to build: a microscope and nothing else. No host, no view
controller, no image widget. That is most of the point of extracting it, and it is what
lets these tests do something ``tests/ui/test_stage_position_readout.py`` cannot --
build the form against a *chosen* microscope configuration and pin both sides of the
compustage branch, rather than branching on whichever configuration the machine
happens to default to.

Two configurations, both shipped in ``fibsem/config``:

* ``microscope-configuration.yaml`` -- not a compustage. Rotation offered, millimetre
  arrow steps, tilt range -10 to 90 degrees.
* ``sim-arctis-configuration.yaml`` -- a compustage. Rotation hidden, micrometre arrow
  steps, tilt range -195 to 15 degrees.

The tilt ranges differ between them, which is what makes the range tests worth running
twice: a form that ignored the stage and hard-coded a range would still pass against
one configuration.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_stage_position_widget.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QWheelEvent  # noqa: E402

from fibsem import config as cfg  # noqa: E402
from fibsem import constants, utils  # noqa: E402
from fibsem.structures import FibsemStagePosition  # noqa: E402
from fibsem.ui.widgets.stage_position_widget import StagePositionWidget  # noqa: E402

FLAT_STAGE = "microscope-configuration.yaml"
COMPUSTAGE = "sim-arctis-configuration.yaml"

# Off-origin on every axis, and asymmetric, so a transposed pair or a dropped
# conversion cannot coincide with the right answer. Inside both stages' limits.
SAMPLE = FibsemStagePosition(
    x=120e-6,
    y=-45e-6,
    z=310e-6,
    r=np.radians(37.0),
    t=np.radians(-8.0),
    coordinate_system="RAW",
)

_sessions = {}


def _microscope(configuration: str):
    """One Demo microscope per configuration for the whole file -- setup_session is
    slow, and neither of these tests mutates the stage."""
    if configuration not in _sessions:
        _sessions[configuration] = utils.setup_session(
            config_path=os.path.join(cfg.CONFIG_PATH, configuration),
            setup_logging=False,
        )[0]
    return _sessions[configuration]


@pytest.fixture(params=[FLAT_STAGE, COMPUSTAGE], ids=["flat-stage", "compustage"])
def widget(request, qapp):
    """The form, against each shipped stage kind in turn."""
    microscope = _microscope(request.param)
    form = StagePositionWidget(microscope=microscope)
    yield form
    form.deleteLater()


@pytest.fixture
def compustage_widget(qapp):
    microscope = _microscope(COMPUSTAGE)
    form = StagePositionWidget(microscope=microscope)
    yield form
    form.deleteLater()


@pytest.fixture
def flat_widget(qapp):
    microscope = _microscope(FLAT_STAGE)
    form = StagePositionWidget(microscope=microscope)
    yield form
    form.deleteLater()


def _wheel(delta: int = 120) -> QWheelEvent:
    return QWheelEvent(
        QPointF(5.0, 5.0),
        QPointF(5.0, 5.0),
        QPoint(0, delta),
        QPoint(0, delta),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )


# --- the conversions, on both stage kinds ------------------------------------


def test_it_shows_millimetres_and_degrees(widget):
    widget.set_position(SAMPLE)

    assert widget.spinbox_x.value() == pytest.approx(0.12, abs=1e-9)
    assert widget.spinbox_y.value() == pytest.approx(-0.045, abs=1e-9)
    assert widget.spinbox_z.value() == pytest.approx(0.31, abs=1e-9)
    assert widget.spinbox_rotation.value() == pytest.approx(37.0, abs=1e-3)
    assert widget.spinbox_tilt.value() == pytest.approx(-8.0, abs=1e-3)


def test_it_reads_back_in_si_units(widget):
    """The way out undoes the way in. A conversion applied on one side only survives
    each half being checked alone, but not this."""
    widget.set_position(SAMPLE)
    read = widget.get_position()

    assert read.x == pytest.approx(SAMPLE.x, abs=1e-9)
    assert read.y == pytest.approx(SAMPLE.y, abs=1e-9)
    assert read.z == pytest.approx(SAMPLE.z, abs=1e-9)
    assert read.r == pytest.approx(SAMPLE.r, abs=1e-5)
    assert read.t == pytest.approx(SAMPLE.t, abs=1e-5)


def test_it_reads_raw_coordinates(widget):
    assert widget.get_position().coordinate_system == "RAW"


def test_the_units_are_on_the_boxes(widget):
    """The suffix is the only thing telling the operator which unit to type in."""
    assert widget.spinbox_x.suffix() == " mm"
    assert widget.spinbox_y.suffix() == " mm"
    assert widget.spinbox_z.suffix() == " mm"
    assert widget.spinbox_rotation.suffix() == constants.DEGREE_SYMBOL
    assert widget.spinbox_tilt.suffix() == constants.DEGREE_SYMBOL


def test_translation_is_shown_to_ten_nanometres(widget):
    for spinbox in (widget.spinbox_x, widget.spinbox_y, widget.spinbox_z):
        assert spinbox.decimals() == 5


# --- the ranges come from the stage ------------------------------------------


def test_translation_ranges_are_the_stage_limits_in_millimetres(widget):
    limits = widget.microscope._stage.limits

    for axis, spinbox in (
        ("x", widget.spinbox_x),
        ("y", widget.spinbox_y),
        ("z", widget.spinbox_z),
    ):
        assert spinbox.minimum() == pytest.approx(
            limits[axis].min * constants.SI_TO_MILLI
        )
        assert spinbox.maximum() == pytest.approx(
            limits[axis].max * constants.SI_TO_MILLI
        )


def test_the_tilt_range_is_taken_in_degrees_already(widget):
    """The stage reports x/y/z in metres but t in degrees, and the form must not
    convert the one that needs no converting."""
    tilt = widget.microscope._stage.limits["t"]

    assert widget.spinbox_tilt.minimum() == pytest.approx(tilt.min)
    assert widget.spinbox_tilt.maximum() == pytest.approx(tilt.max)


def test_the_two_stages_really_do_disagree_about_tilt():
    """Guards the two tests above from being run twice against the same numbers -- if
    the fixtures ever resolved to one configuration, the range tests would still pass
    and prove nothing."""
    flat = _microscope(FLAT_STAGE)._stage.limits["t"]
    compu = _microscope(COMPUSTAGE)._stage.limits["t"]

    assert (flat.min, flat.max) != (compu.min, compu.max)


# --- the compustage branch, both sides ---------------------------------------


def test_a_compustage_steps_in_micrometres(compustage_widget):
    for spinbox in (
        compustage_widget.spinbox_x,
        compustage_widget.spinbox_y,
        compustage_widget.spinbox_z,
    ):
        assert spinbox.singleStep() == pytest.approx(1e-6 * constants.SI_TO_MILLI)


def test_any_other_stage_steps_in_millimetres(flat_widget):
    for spinbox in (
        flat_widget.spinbox_x,
        flat_widget.spinbox_y,
        flat_widget.spinbox_z,
    ):
        assert spinbox.singleStep() == pytest.approx(0.001)


def test_a_compustage_is_not_offered_rotation(compustage_widget):
    """Label as well as box -- hiding only the box leaves a caption over the tilt row."""
    assert not compustage_widget.spinbox_rotation.isVisibleTo(compustage_widget)
    assert not compustage_widget.label_rotation.isVisibleTo(compustage_widget)


def test_any_other_stage_is(flat_widget):
    assert flat_widget.spinbox_rotation.isVisibleTo(flat_widget)
    assert flat_widget.label_rotation.isVisibleTo(flat_widget)


# --- the guards --------------------------------------------------------------


def test_a_passing_scroll_does_not_retype_a_box(qapp, widget):
    """Scrolling the panel used to change whichever box the pointer crossed."""
    widget.set_position(SAMPLE)

    for axis, spinbox in widget._spinboxes().items():
        before = spinbox.value()
        # sendEvent, not spinbox.wheelEvent: the guard is an event filter, so it only
        # sees the event on the way through dispatch. Calling the handler direct walks
        # past it and would pass with the guard removed.
        qapp.sendEvent(spinbox, _wheel())
        assert spinbox.value() == before, f"{axis} changed on a wheel event"


# --- it asks, it does not read -----------------------------------------------


def test_refresh_asks_the_host_rather_than_the_stage(widget, monkeypatch):
    """The form never calls the microscope on a UI event. Reading the stage is a device
    call and the host owns those, so the button emits and stops."""
    asked = []
    reads = []
    widget.refresh_requested.connect(lambda: asked.append(True))
    monkeypatch.setattr(
        widget.microscope,
        "get_stage_position",
        lambda *a, **k: reads.append(True),
        raising=False,
    )

    widget.btn_refresh.click()

    assert asked == [True]
    assert reads == [], "the form read the stage instead of asking for a refresh"
