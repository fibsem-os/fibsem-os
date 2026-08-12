"""The milling controls are locked while a mill is running (FIB-580).

Reported from the 2026-08-11 testing session as "properly disable the milling controls
when milling". Before this, `_update_button_states` governed only Run / Stop / Pause:
the stage and pattern editor stayed fully editable, and the FIB canvas still offered
"Move Patterns Here" over a lamella that was being milled.

The beam is not at risk -- `get_settings()` deep-copies before the worker starts, so the
run holds its own config. The damage is to the record: the supervised loop writes the
widget's config back as the task's when it finishes, so an edit made mid-mill lands in
the protocol as the position that was milled, when it is not.

Every test here has a not-milling counterpart. A lock test that only ever asserts "this
did nothing" passes just as well against a widget that does nothing at all.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem import utils
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemMillingSettings, Point
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

_app = QApplication.instance() or QApplication(sys.argv)


class _RunningThread:
    """Stands in for the FunctionWorker; `is_milling` asks it exactly this."""

    def is_alive(self) -> bool:
        return True


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


def _config_with_a_stage() -> FibsemMillingTaskConfig:
    """A config with one enabled stage.

    The default `FibsemMillingTaskConfig()` has none, and both guarded paths return
    early when there are no enabled stages -- so a widget built from the default makes
    every "nothing happened" assertion below pass for the wrong reason. The idle
    counterpart tests caught exactly that.
    """
    return FibsemMillingTaskConfig(
        stages=[
            FibsemMillingStage(
                name="test-stage",
                milling=FibsemMillingSettings(milling_current=2e-9),
                pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
            )
        ]
    )


@pytest.fixture()
def widget(microscope):
    w = MillingTaskViewerWidget(
        microscope=microscope,
        milling_task_config=_config_with_a_stage(),
        milling_enabled=True,
    )
    yield w
    w.deleteLater()


def _start_milling(widget) -> None:
    """Put the widget in the state a running mill leaves it in."""
    widget.milling_widget._milling_thread = _RunningThread()
    widget.milling_widget._update_button_states()


def _finish_milling(widget) -> None:
    """What `_milling_worker`'s finally block does."""
    widget.milling_widget._milling_thread = None
    widget.milling_widget._update_button_states()


# ── the editor ────────────────────────────────────────────────────────────────


def test_the_editor_is_editable_when_nothing_is_milling(widget):
    """The control for everything below: the panel is not simply always disabled."""
    assert widget.is_milling is False
    assert widget.config_widget.isEnabled() is True


def test_the_editor_locks_while_milling(widget):
    _start_milling(widget)

    assert widget.config_widget.isEnabled() is False
    # the stage list is the part that would rewrite a pattern, and it is a child, so
    # this also pins that disabling the panel actually reaches it
    assert widget.config_widget.milling_stages_widget.isEnabled() is False


def test_the_editor_unlocks_when_the_mill_finishes(widget):
    _start_milling(widget)
    _finish_milling(widget)

    assert widget.config_widget.isEnabled() is True


def test_the_lock_lands_at_start_not_at_the_first_progress_signal(widget, monkeypatch):
    """`run_milling` used to disable only the Run button and leave the rest until a
    progress signal arrived. Preparing the milling conditions takes seconds, and the
    editor was live for all of them."""
    import fibsem.ui.widgets.milling_widget as milling_widget_module

    class _StubWorker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass  # never actually mill

        def is_alive(self):
            return True

    monkeypatch.setattr(milling_widget_module, "FunctionWorker", _StubWorker)

    widget.milling_widget.run_milling(_config_with_a_stage())

    assert widget.config_widget.isEnabled() is False


# ── the canvas right-click ────────────────────────────────────────────────────


def _record_menu(widget, monkeypatch) -> list:
    opened = []
    monkeypatch.setattr(
        widget, "_show_reposition_menu", lambda *a, **k: opened.append(a)
    )
    return opened


def test_the_reposition_menu_opens_when_nothing_is_milling(widget, monkeypatch, fib_image):
    """The control: without this, the milling case below proves nothing."""
    widget.set_fib_image(fib_image)
    opened = _record_menu(widget, monkeypatch)

    widget._on_canvas_right_click(10.0, 10.0, None)

    assert opened, "right-click should offer the menu when idle"


def test_the_reposition_menu_stays_shut_while_milling(widget, monkeypatch, fib_image):
    """The canvas is not inside the editor, so disabling that panel does not reach it."""
    widget.set_fib_image(fib_image)
    opened = _record_menu(widget, monkeypatch)
    _start_milling(widget)

    widget._on_canvas_right_click(10.0, 10.0, None)

    assert opened == []


def test_moving_patterns_is_refused_while_milling(widget, fib_image):
    """Guarded at the mutation too: a menu already on screen when a run starts would
    otherwise still act on it."""
    widget.set_fib_image(fib_image)
    stages = widget.config_widget.milling_stages_widget.get_enabled_stages()
    assert stages, "the fixture must supply a stage, or this test proves nothing"
    before = Point(x=stages[0].pattern.point.x, y=stages[0].pattern.point.y)
    _start_milling(widget)

    widget._move_patterns(Point(x=1e-6, y=1e-6), move_all=True)

    after = widget.config_widget.milling_stages_widget.get_enabled_stages()[0].pattern.point
    assert (after.x, after.y) == (before.x, before.y)


def test_moving_patterns_still_works_when_idle(widget, fib_image):
    """The counterpart. Without it the test above passes against a broken mover."""
    widget.set_fib_image(fib_image)
    stages = widget.config_widget.milling_stages_widget.get_enabled_stages()
    assert stages, "the fixture must supply a stage, or this test proves nothing"
    before = Point(x=stages[0].pattern.point.x, y=stages[0].pattern.point.y)

    widget._move_patterns(Point(x=before.x + 5e-6, y=before.y + 5e-6), move_all=True)

    after = widget.config_widget.milling_stages_widget.get_enabled_stages()[0].pattern.point
    assert (after.x, after.y) != (before.x, before.y)


@pytest.fixture()
def fib_image():
    """512px at 200nm/px = a ~102um field.

    Sized in metres, not pixels: `_move_patterns` refuses a move that would put the
    pattern outside the image, and the 10x5um stage above does not fit in a field
    smaller than itself. A 64px/10nm image looks fine and silently makes every move a
    no-op, which reads exactly like the lock working.
    """
    import numpy as np

    from fibsem.structures import (
        BeamType,
        FibsemImage,
        FibsemImageMetadata,
        ImageSettings,
        MicroscopeState,
    )

    return FibsemImage(
        data=np.zeros((512, 512), dtype=np.uint8),
        metadata=FibsemImageMetadata(
            image_settings=ImageSettings(resolution=(512, 512), beam_type=BeamType.ION),
            pixel_size=Point(2e-7, 2e-7),
            microscope_state=MicroscopeState(),
        ),
    )


def test_the_lock_survives_the_workflow_enabling_this_widget(widget):
    """`update_milling_config_ui` calls setEnabled(True) on the *viewer* when it hands a
    config to the UI. Qt re-enables children on that call unless they were disabled in
    their own right -- which is why the lock is applied to `config_widget` directly
    rather than by disabling the panel through its parent.
    """
    _start_milling(widget)

    widget.setEnabled(True)  # what the workflow does

    assert widget.config_widget.isEnabled() is False
    _finish_milling(widget)
    assert widget.config_widget.isEnabled() is True
