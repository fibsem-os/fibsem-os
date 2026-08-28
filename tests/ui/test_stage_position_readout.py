"""What the stage position form on the Movement tab does, before it is extracted.

The form is the five spin boxes in rows 0-4 of ``FibsemMovementWidget``'s stage panel
plus the refresh button in the panel header. It converts in both directions -- metres to
millimetres, radians to degrees on the way in, and back again on the way out -- and takes
its ranges from the stage rather than from the widget. None of that is asserted anywhere
today: the existing movement tests all drive the half that *moves*.

These pin the conversions, the ranges and the two compustage adjustments so that
splitting the form out of the widget is a refactor with a net under it rather than a
rewrite that happens to still construct. They are deliberately written against the
public surface a host uses -- ``update_ui`` and ``get_position_from_ui`` -- so they keep
their meaning when the form moves behind a delegating facade.

Two of them branch on ``stage_is_compustage``. The default microscope configuration is
user state and is not the same on every machine, so a test that assumed either answer
would pass here and fail elsewhere; each asserts the behaviour for the configuration it
actually got, and the branch it did not take is covered from the other side.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_stage_position_readout.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QWheelEvent  # noqa: E402

# Same construction seam the rest of tests/ui uses: a host with a view controller, a
# real image widget and a Demo microscope. The movement widget asserts on its parent,
# so there is no lighter way to build the real thing.
sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_viewer_less_widgets import (  # noqa: E402
    _CanvasHost,
    _image_widget,
    _movement_widget,
)

from fibsem import constants  # noqa: E402
from fibsem.structures import FibsemStagePosition  # noqa: E402

# Off-origin on every axis, and asymmetric, so a transposed pair or a dropped
# conversion cannot coincide with the right answer. Inside the Demo stage limits.
SAMPLE = FibsemStagePosition(
    x=120e-6,
    y=-45e-6,
    z=310e-6,
    r=np.radians(37.0),
    t=np.radians(-12.0),
    coordinate_system="RAW",
)


@pytest.fixture
def movement(qapp):
    host = _CanvasHost()
    _image_widget(host)
    widget = _movement_widget(host)
    yield widget
    host.deleteLater()


def _boxes(movement) -> dict:
    return {
        "x": movement.doubleSpinBox_movement_stage_x,
        "y": movement.doubleSpinBox_movement_stage_y,
        "z": movement.doubleSpinBox_movement_stage_z,
        "r": movement.doubleSpinBox_movement_stage_rotation,
        "t": movement.doubleSpinBox_movement_stage_tilt,
    }


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


# --- the conversions ---------------------------------------------------------


def test_the_form_shows_millimetres_and_degrees(movement):
    """Metres and radians go in; the operator reads millimetres and degrees."""
    movement.update_ui(stage_position=SAMPLE)
    boxes = _boxes(movement)

    assert boxes["x"].value() == pytest.approx(0.12, abs=1e-9)
    assert boxes["y"].value() == pytest.approx(-0.045, abs=1e-9)
    assert boxes["z"].value() == pytest.approx(0.31, abs=1e-9)
    assert boxes["r"].value() == pytest.approx(37.0, abs=1e-3)
    assert boxes["t"].value() == pytest.approx(-12.0, abs=1e-3)


def test_the_form_reads_back_in_si_units(movement):
    """And the way out undoes the way in, on every axis at once.

    The round trip is the assertion that matters: a conversion applied on one side
    only survives each half being checked separately, but not this.
    """
    movement.update_ui(stage_position=SAMPLE)
    read = movement.get_position_from_ui()

    assert read.x == pytest.approx(SAMPLE.x, abs=1e-9)
    assert read.y == pytest.approx(SAMPLE.y, abs=1e-9)
    assert read.z == pytest.approx(SAMPLE.z, abs=1e-9)
    assert read.r == pytest.approx(SAMPLE.r, abs=1e-5)
    assert read.t == pytest.approx(SAMPLE.t, abs=1e-5)


def test_the_form_reads_raw_coordinates(movement):
    """RAW, not the linked coordinate system -- the entry boxes mean stage axes."""
    assert movement.get_position_from_ui().coordinate_system == "RAW"


def test_the_units_are_on_the_boxes(movement):
    """The suffix is the only thing telling the operator which unit to type in."""
    boxes = _boxes(movement)
    assert boxes["x"].suffix() == " mm"
    assert boxes["y"].suffix() == " mm"
    assert boxes["z"].suffix() == " mm"
    # Set twice during construction -- with a leading space in the layout, without one
    # when the connections are made. The second write is the one that survives.
    assert boxes["r"].suffix() == constants.DEGREE_SYMBOL
    assert boxes["t"].suffix() == constants.DEGREE_SYMBOL


def test_translation_is_shown_to_ten_nanometres(movement):
    """Five decimals of a millimetre. Fewer and a stable_move would round to nothing."""
    boxes = _boxes(movement)
    assert boxes["x"].decimals() == 5
    assert boxes["y"].decimals() == 5
    assert boxes["z"].decimals() == 5


# --- the ranges come from the stage, not from the widget ---------------------


def test_translation_ranges_are_the_stage_limits_in_millimetres(movement):
    limits = movement.microscope._stage.limits
    boxes = _boxes(movement)

    for axis in ("x", "y", "z"):
        assert boxes[axis].minimum() == pytest.approx(
            limits[axis].min * constants.SI_TO_MILLI
        )
        assert boxes[axis].maximum() == pytest.approx(
            limits[axis].max * constants.SI_TO_MILLI
        )


def test_the_tilt_range_is_taken_in_degrees_already(movement):
    """The stage reports x/y/z in metres but t in degrees, and the form must not
    convert the one that needs no converting."""
    tilt = movement.microscope._stage.limits["t"]
    box = _boxes(movement)["t"]

    assert box.minimum() == pytest.approx(tilt.min)
    assert box.maximum() == pytest.approx(tilt.max)


# --- the compustage adjustments ----------------------------------------------


def test_the_step_size_suits_the_stage(movement):
    """A compustage works in micrometres, so its arrows step by 1 um rather than 1 um
    times a thousand. Everything else keeps the default."""
    boxes = _boxes(movement)
    expected = (
        1e-6 * constants.SI_TO_MILLI
        if movement.microscope.stage_is_compustage
        else 0.001
    )

    for axis in ("x", "y", "z"):
        assert boxes[axis].singleStep() == pytest.approx(expected)


def test_rotation_is_offered_only_where_it_exists(movement):
    """A compustage does not rotate, so both the label and the box go -- not just the
    box, which would leave a stranded caption."""
    shown = not movement.microscope.stage_is_compustage

    assert movement.doubleSpinBox_movement_stage_rotation.isVisibleTo(movement) is shown
    assert movement.label_movement_stage_rotation.isVisibleTo(movement) is shown


# --- the guards --------------------------------------------------------------


def test_a_passing_scroll_does_not_move_the_stage_position(qapp, movement):
    """Scrolling the panel used to retype whichever box the pointer crossed. Every box
    on the form is guarded, not only the ones that were reported."""
    movement.update_ui(stage_position=SAMPLE)

    for axis, box in _boxes(movement).items():
        before = box.value()
        # sendEvent, not box.wheelEvent: the guard is an event filter, so it only sees
        # the event on the way through dispatch. Calling the handler direct walks past
        # it and would pass with the guard removed.
        qapp.sendEvent(box, _wheel())
        assert box.value() == before, f"{axis} changed on a wheel event"


def test_the_refresh_button_asks_the_stage_where_it_is(movement, monkeypatch):
    """The one place the form is allowed to read the device: a button the operator
    pressed. Nothing else on this half polls."""
    asked = []

    def _get_stage_position():
        asked.append(True)
        return SAMPLE

    monkeypatch.setattr(
        movement.microscope, "get_stage_position", _get_stage_position, raising=False
    )
    movement.btn_refresh_stage.click()

    assert asked, "refresh did not read the stage"
    assert _boxes(movement)["x"].value() == pytest.approx(0.12, abs=1e-9)
