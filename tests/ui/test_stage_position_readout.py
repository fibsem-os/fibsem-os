"""What the stage position form on the Movement tab does, before it is extracted.

The form is the five spin boxes and the refresh button that make up
``StagePositionWidget``, reached through the Movement tab that hosts it. It converts in
both directions -- metres to millimetres, radians to degrees on the way in, and back
again on the way out -- and takes its ranges from the stage rather than from the widget.

**These are the equivalence tests for the extraction.** They were written against the
monolithic widget, before the form was a separate class, and they still drive the tab's
own surface -- ``update_ui`` and ``get_position_from_ui``, which is what a host calls --
rather than the form directly. That is what makes them evidence that composing the form
changed nothing, and it is why they stay here rather than being folded into
``tests/ui/test_stage_position_widget.py``, which tests the form on its own terms.

Where a test needs a control rather than a value it reaches through
``movement.position_widget``. It used to reach for aliases the tab kept during the swap;
those are gone.

**The default microscope configuration is user state and is not the same on every
machine**, and it reaches these tests twice over.

Once through ``stage_is_compustage``: two tests branch on it rather than assuming an
answer, and the branch not taken is covered directly in
``tests/ui/test_stage_position_widget.py``, where the form is cheap enough to build
against a chosen configuration.

And once through the *ranges*, which is the subtler of the two. A spin box clamps to its
range, so a sample position outside the configured stage limits comes back as the limit
and the failure reads as a broken conversion rather than an out-of-range fixture. That
is exactly what happened: ``sim-arctis`` tilts to -195 degrees and the shipped default
only to -10, so a -12 degree sample passed locally and came back as -10.0 in CI.
``SAMPLE`` therefore sits inside the intersection of the shipped configurations, and
``test_the_sample_is_inside_this_stages_limits`` says so out loud -- so a future
configuration that narrows further fails by name instead of by clamping.

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
# conversion cannot coincide with the right answer.
#
# Inside every shipped configuration's stage limits, which is narrower than it looks:
# z starts at 0 on the default configuration and tilt only reaches -10 there, against
# -195 on sim-arctis. A spin box clamps silently, so a sample outside the range does
# not fail as "out of range" -- it fails as a wrong conversion.
SAMPLE = FibsemStagePosition(
    x=120e-6,
    y=-45e-6,
    z=310e-6,
    r=np.radians(37.0),
    t=np.radians(-8.0),
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
        "x": movement.position_widget.spinbox_x,
        "y": movement.position_widget.spinbox_y,
        "z": movement.position_widget.spinbox_z,
        "r": movement.position_widget.spinbox_rotation,
        "t": movement.position_widget.spinbox_tilt,
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


def test_the_sample_is_inside_this_stages_limits(movement):
    """The fixture, not the widget. A spin box clamps silently, so a SAMPLE outside the
    configured range would make every conversion test above fail as though a unit had
    been dropped. This one names the real cause instead.

    It bites because the shipped configurations disagree: tilt reaches -195 degrees on
    sim-arctis and only -10 on the default, and z starts at 0 there.
    """
    limits = movement.microscope._stage.limits

    for axis, value_mm in (
        ("x", SAMPLE.x * constants.SI_TO_MILLI),
        ("y", SAMPLE.y * constants.SI_TO_MILLI),
        ("z", SAMPLE.z * constants.SI_TO_MILLI),
    ):
        low = limits[axis].min * constants.SI_TO_MILLI
        high = limits[axis].max * constants.SI_TO_MILLI
        assert low <= value_mm <= high, f"SAMPLE.{axis} is outside {low}..{high} mm"

    tilt_deg = np.degrees(SAMPLE.t)
    assert limits["t"].min <= tilt_deg <= limits["t"].max, (
        f"SAMPLE.t is outside {limits['t'].min}..{limits['t'].max} deg"
    )


def test_the_form_shows_millimetres_and_degrees(movement):
    """Metres and radians go in; the operator reads millimetres and degrees."""
    movement.update_ui(stage_position=SAMPLE)
    boxes = _boxes(movement)

    assert boxes["x"].value() == pytest.approx(0.12, abs=1e-9)
    assert boxes["y"].value() == pytest.approx(-0.045, abs=1e-9)
    assert boxes["z"].value() == pytest.approx(0.31, abs=1e-9)
    assert boxes["r"].value() == pytest.approx(37.0, abs=1e-3)
    assert boxes["t"].value() == pytest.approx(-8.0, abs=1e-3)


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

    form = movement.position_widget
    assert form.spinbox_rotation.isVisibleTo(form) is shown
    assert form.label_rotation.isVisibleTo(form) is shown


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
    movement.position_widget.btn_refresh.click()

    assert asked, "refresh did not read the stage"
    assert _boxes(movement)["x"].value() == pytest.approx(0.12, abs=1e-9)
