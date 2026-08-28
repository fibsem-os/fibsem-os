"""Three parts of the Movement tab that nothing exercised.

Traced the existing suite -- ``test_movement_progress``, ``test_movement_worker_bodies``,
``test_canvas_double_click_guards``, ``test_movement_widget_vertical_gate`` and
``test_viewer_less_widgets`` -- against ``FibsemMovementWidget`` statement by statement.
Every method ran except these:

* ``move_to_position``            0 of 3 statements
* ``_update_milling_angle``       0 of 5
* ``update_ui_after_movement``    6 of 14

``move_to_position`` is the notable one. It is the *public* entry point: the Move button,
both minimaps, the lamella list, the saved-position list and AutoLamella's pose move all
land here, six call sites in four files. What was untested is precisely the branch those
six do not share -- reading the form when no position is passed, which is what the Move
button does and nothing else does.

``update_ui_after_movement`` chooses what to re-acquire once the stage has arrived, from
the user's preferences. Its early return was covered; the four preference branches were
not, so a wrong pairing would have re-acquired the wrong beam after every move.

These pin behaviour that stands on its own, and they will keep their meaning if the
control half is later extracted -- like ``test_stage_position_readout``, they drive the
tab's own surface rather than an internal one.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_movement_control_surface.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

# Same construction seam the rest of tests/ui uses: a host with a view controller, a
# real image widget and a Demo microscope.
sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_viewer_less_widgets import (  # noqa: E402
    _CanvasHost,
    _image_widget,
    _movement_widget,
)

from fibsem import config as cfg  # noqa: E402
from fibsem import constants  # noqa: E402
from fibsem.structures import FibsemStagePosition  # noqa: E402

# Inside every shipped configuration's stage limits -- z starts at 0 on the default
# configuration and tilt only reaches -10 there. A spin box clamps silently, so a value
# outside the range would fail as a wrong readout rather than as a bad fixture.
TYPED = FibsemStagePosition(
    x=80e-6,
    y=-25e-6,
    z=140e-6,
    r=np.radians(12.0),
    t=np.radians(-4.0),
    coordinate_system="RAW",
)
ASKED = FibsemStagePosition(
    x=-60e-6,
    y=35e-6,
    z=220e-6,
    r=np.radians(-19.0),
    t=np.radians(6.0),
    coordinate_system="RAW",
)


@pytest.fixture
def movement(qapp):
    host = _CanvasHost()
    _image_widget(host)
    widget = _movement_widget(host)
    yield widget
    widget._teardown_connections()
    host.deleteLater()


@pytest.fixture
def moves(movement, monkeypatch):
    """Record the target of every absolute move, without one happening."""
    seen = []
    monkeypatch.setattr(
        movement.microscope,
        "safe_absolute_stage_movement",
        lambda stage_position, *a, **k: seen.append(stage_position),
        raising=False,
    )
    return seen


def _settle(qapp, seen, tries: int = 40):
    """The move runs on a worker thread; pump until it lands."""
    for _ in range(tries):
        qapp.processEvents()
        if seen:
            return


# --- move_to_position: the surface six call sites use ------------------------


def test_a_named_position_is_the_one_moved_to(qapp, movement, moves):
    """What the five non-button callers do: hand over a position explicitly."""
    movement.move_to_position(ASKED)
    _settle(qapp, moves)

    assert moves, "no move was started"
    assert moves[0].x == pytest.approx(ASKED.x)
    assert moves[0].y == pytest.approx(ASKED.y)
    assert moves[0].z == pytest.approx(ASKED.z)


def test_no_position_means_the_one_typed_into_the_form(qapp, movement, moves):
    """The Move button's whole path, and the only branch the other five never take.

    Passing None must read the form -- not the stage, and not a remembered value.
    """
    movement.position_widget.set_position(TYPED)

    movement.move_to_position(None)
    _settle(qapp, moves)

    assert moves, "no move was started"
    assert moves[0].x == pytest.approx(TYPED.x, abs=1e-9)
    assert moves[0].y == pytest.approx(TYPED.y, abs=1e-9)
    assert moves[0].z == pytest.approx(TYPED.z, abs=1e-9)
    assert moves[0].r == pytest.approx(TYPED.r, abs=1e-5)
    assert moves[0].t == pytest.approx(TYPED.t, abs=1e-5)


def test_the_move_button_is_that_path(qapp, movement, moves):
    """And the button really is wired to it -- a disconnected button would leave every
    assertion above passing."""
    movement.position_widget.set_position(TYPED)

    movement.pushButton_move.click()
    _settle(qapp, moves)

    assert moves, "the Move button started no move"
    assert moves[0].x == pytest.approx(TYPED.x, abs=1e-9)


def test_the_buttons_go_down_before_the_stage_starts_moving(movement, monkeypatch):
    """Disabled on the GUI thread, before the worker starts -- not after it reports.
    A second click landing in that window is a second overlapping stage move."""
    enabled_when_the_move_began = []
    monkeypatch.setattr(
        movement.microscope,
        "safe_absolute_stage_movement",
        lambda *a, **k: enabled_when_the_move_began.append(
            movement.pushButton_move.isEnabled()
        ),
        raising=False,
    )

    movement.move_to_position(ASKED)

    assert not movement.pushButton_move.isEnabled(), (
        "the Move button was still live after the move was dispatched"
    )


# --- the milling angle -------------------------------------------------------


def test_typing_a_milling_angle_stores_it_on_the_microscope(movement):
    """The spin box is the only way to set this, and it writes through on edit rather
    than on a confirm button -- so the value in the box and the value on the microscope
    must not be able to disagree."""
    movement.doubleSpinBox_milling_angle.setValue(23.0)

    assert movement.microscope.system.stage.milling_angle == pytest.approx(23.0)


def test_the_milling_angle_tooltip_follows_the_value(movement):
    """The tooltip is where the resulting orientation is actually shown."""
    movement.doubleSpinBox_milling_angle.setValue(11.0)
    before = movement.pushButton_move_to_milling_angle.toolTip()

    movement.doubleSpinBox_milling_angle.setValue(31.0)
    after = movement.pushButton_move_to_milling_angle.toolTip()

    assert before != after, "the tooltip still describes the old milling angle"
    assert after == movement.microscope.get_orientation("MILLING").pretty_orientation


def test_setting_a_milling_angle_does_not_move_the_stage(movement, monkeypatch):
    """It stores a number. Editing the box must not re-pose the stage under the
    operator -- this is a settings write, not a movement action."""
    moved = []
    for name in (
        "safe_absolute_stage_movement",
        "move_to_orientation",
        "stable_move",
        "vertical_move",
    ):
        monkeypatch.setattr(
            movement.microscope, name, lambda *a, **k: moved.append(name), raising=False
        )

    movement.doubleSpinBox_milling_angle.setValue(29.0)

    assert moved == []


# --- what is re-acquired once the stage has arrived --------------------------


def _acquisitions(movement, monkeypatch) -> list:
    seen = []
    for name in ("acquire_reference_images", "acquire_sem_image", "acquire_fib_image"):
        monkeypatch.setattr(
            movement.image_widget,
            name,
            lambda *a, n=name, **k: seen.append(n),
            raising=False,
        )
    return seen


def _prefs(movement, monkeypatch, *, sem: bool, fib: bool) -> None:
    prefs = cfg.load_user_preferences()
    prefs.movement.acquire_sem_after_stage_movement = sem
    prefs.movement.acquire_fib_after_stage_movement = fib
    # The widget does `from fibsem import config as cfg`, so it holds this very module
    # object -- patching the attribute here is what it will read.
    monkeypatch.setattr(cfg, "load_user_preferences", lambda: prefs)


@pytest.mark.parametrize(
    ("sem", "fib", "expected"),
    [
        (True, True, ["acquire_reference_images"]),
        (True, False, ["acquire_sem_image"]),
        (False, True, ["acquire_fib_image"]),
        (False, False, []),
    ],
    ids=["both", "sem-only", "fib-only", "neither"],
)
def test_the_preferences_choose_what_is_retaken(
    movement, monkeypatch, sem, fib, expected
):
    """Both beams go through one call, not two -- and the single-beam cases must not
    swap, which is the failure an operator would read as the wrong detector."""
    _prefs(movement, monkeypatch, sem=sem, fib=fib)
    seen = _acquisitions(movement, monkeypatch)

    movement.update_ui_after_movement()

    assert seen == expected


def test_asking_for_no_retake_acquires_nothing(movement, monkeypatch):
    """`retake=False` is the caller saying it has already dealt with the images. It
    still refreshes the readout, or the form would show where the stage used to be."""
    _prefs(movement, monkeypatch, sem=True, fib=True)
    seen = _acquisitions(movement, monkeypatch)
    refreshed = []
    monkeypatch.setattr(
        movement, "update_ui", lambda *a, **k: refreshed.append(True), raising=False
    )

    movement.update_ui_after_movement(retake=False)

    assert seen == []
    assert refreshed, "the readout was left showing the old position"


def test_nothing_is_retaken_while_the_microscope_is_already_acquiring(
    movement, monkeypatch
):
    """Queuing a second acquisition onto a busy microscope is how a SEM frame ends up
    on the FIB canvas."""
    _prefs(movement, monkeypatch, sem=True, fib=True)
    seen = _acquisitions(movement, monkeypatch)
    monkeypatch.setattr(
        type(movement.microscope), "is_acquiring", property(lambda self: True)
    )

    movement.update_ui_after_movement()

    assert seen == []


# --- the readout still says where the stage is -------------------------------


def test_a_move_leaves_the_form_showing_where_the_stage_went(qapp, movement):
    """End to end through the public entry point, against a stage that really moves --
    the form is the operator's confirmation that the move landed.

    Deliberately does not use the `moves` fixture: with the movement stubbed out the
    stage never leaves the origin and this would compare zero against zero.
    """
    movement.move_to_position(ASKED)
    for _ in range(60):
        qapp.processEvents()
        if movement.pushButton_move.isEnabled():
            break

    stage = movement.microscope.get_stage_position()
    shown = movement.get_position_from_ui()

    assert stage.x == pytest.approx(ASKED.x, abs=1e-9), "the stage did not get there"
    assert shown.x == pytest.approx(stage.x, abs=1e-8), (
        f"form shows {shown.x * constants.SI_TO_MILLI:.5f} mm, "
        f"stage is at {stage.x * constants.SI_TO_MILLI:.5f} mm"
    )
    assert shown.z == pytest.approx(stage.z, abs=1e-8)
