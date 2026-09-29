"""Where the milling worker stops and the widget starts.

``_milling_worker`` runs a milling task off the GUI thread and then, in its ``finally``,
reaches back for the widget to re-enable its own controls. The call is marshalled --
``_update_button_states`` carries ``@ensure_main_thread`` -- so this is not a threading
bug. It is the operation and the UI update sharing a function, the same shape the
movement workers were split out of.

Two orderings are load-bearing and easy to lose when the call moves, so they are pinned
here rather than discovered later:

* the buttons unlock only once the mill is genuinely over, and
* they unlock *before* ``finished_milling_signal`` reaches its listeners. The
  coincidence viewer hangs ``_finalize_milling_ui`` off that signal precisely because it
  means "fully complete", and it should not run while the controls still say otherwise.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QEventLoop, QTimer

sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_milling_lockout import _config_with_a_stage  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ui.widgets import milling_widget as milling_widget_module  # noqa: E402
from fibsem.ui.widgets.milling_task_viewer_widget import (  # noqa: E402
    MillingTaskViewerWidget,
)

_microscope = None


def _session():
    global _microscope
    if _microscope is None:
        _microscope, _ = utils.setup_session(manufacturer="Demo")
    return _microscope


def _pump(ms: int = 100) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _run(condition, tries: int = 30) -> None:
    for _ in range(tries):
        _pump()
        if condition():
            return


@pytest.fixture
def milling(qapp):
    viewer = MillingTaskViewerWidget(
        microscope=_session(),
        milling_task_config=_config_with_a_stage(),
        milling_enabled=True,
    )
    yield viewer.milling_widget
    viewer.deleteLater()


class _Mill:
    """A milling task the test starts and ends on purpose.

    `run_milling_task` is replaced by something that blocks until `finish()`. A stub
    that returned immediately would be worse than useless here: the worker would be over
    before `run_milling` reached its own closing `_update_button_states()`, so that call
    would find `is_milling` already False and unlock the controls itself. Every
    assertion about the end-of-mill hand-off would then pass without the hand-off
    existing -- which is exactly what happened before this was written.
    """

    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self._release = threading.Event()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        self._release.wait(timeout=10)

    def begin(self, milling, config):
        milling.run_milling(config)
        assert self.started.wait(timeout=10), "the milling task never started"

    def finish(self):
        self._release.set()


@pytest.fixture
def mill(monkeypatch):
    gate = _Mill()
    monkeypatch.setattr(milling_widget_module, "run_milling_task", gate)
    yield gate
    gate.finish()


def _watch(milling, order=None) -> list:
    """Record each button update: the thread, `is_milling` at the time, and the value.

    Wraps the button's own `setEnabled` rather than `_update_button_states`. Replacing
    the method on the instance would shadow its `@ensure_main_thread`, so the recorder
    would run wherever the *caller* is -- which is exactly the thing under test, and the
    test would report the marshalling it removed.
    """
    seen = []
    button = milling.pushButton_run_milling
    original = button.setEnabled

    def _record(enabled):
        seen.append((threading.current_thread().name, milling.is_milling, enabled))
        if order is not None:
            order.append("buttons")
        original(enabled)

    button.setEnabled = _record
    return seen


# --- the hand-off at the end of a mill ---------------------------------------


def test_a_finished_mill_unlocks_the_controls(milling, mill):
    """The control for everything else here: a mill that ran leaves the widget usable."""
    mill.begin(milling, _config_with_a_stage())
    assert not milling.pushButton_run_milling.isEnabled(), (
        "precondition: the controls lock while milling"
    )

    mill.finish()
    _run(lambda: not milling.is_milling and milling.pushButton_run_milling.isEnabled())

    assert mill.calls, "the milling task never ran"
    assert not milling.is_milling
    assert milling.pushButton_run_milling.isEnabled(), "the controls stayed locked"


def test_the_controls_unlock_before_the_finished_signal_is_delivered(milling, mill):
    """`finished_milling_signal` means "fully complete" to its listeners.

    The coincidence viewer resets its whole panel on it. Delivering it while the
    controls still read "milling" would have the two halves of the UI disagreeing.
    """
    order = []
    seen = _watch(milling, order)
    milling.finished_milling_signal.connect(lambda: order.append("finished"))

    mill.begin(milling, _config_with_a_stage())
    mill.finish()
    _run(lambda: "finished" in order and milling.pushButton_run_milling.isEnabled())

    assert order.count("finished") == 1, order
    # `run_milling` locks the controls on the way in, so the first "buttons" entry is
    # that lock. What has to precede the signal is the *unlock*.
    unlocked = next(i for i, (_, _, enabled) in enumerate(seen) if enabled)
    assert (
        order.index("finished")
        > [i for i, label in enumerate(order) if label == "buttons"][unlocked]
    ), order


def test_the_buttons_are_updated_on_the_gui_thread(milling, mill):
    """Touching a widget off the GUI thread is a Qt violation.

    True today because `@ensure_main_thread` marshals the worker's call. It has to stay
    true however the call is arranged.
    """
    seen = _watch(milling)

    mill.begin(milling, _config_with_a_stage())
    mill.finish()
    _run(lambda: milling.pushButton_run_milling.isEnabled())

    assert seen, "the buttons were never updated"
    assert {thread for thread, _, _ in seen} == {"MainThread"}, seen


def test_the_mill_is_over_by_the_time_the_buttons_are_updated(milling, mill):
    """`_update_button_states` reads `is_milling`, so it has to run after the worker
    has let go of its thread -- otherwise it re-locks the controls it came to free."""
    seen = _watch(milling)

    mill.begin(milling, _config_with_a_stage())
    mill.finish()
    _run(lambda: milling.pushButton_run_milling.isEnabled())

    assert seen[-1][1] is False, f"the buttons were updated while still milling: {seen}"
    assert seen[-1][2] is True, f"the last update did not unlock the controls: {seen}"


# --- the rule the split establishes ------------------------------------------


def test_the_worker_body_does_not_touch_the_widget(milling, mill):
    """The body runs the task and nothing else.

    `@ensure_main_thread` makes the old direct call safe, so this is about coupling
    rather than correctness -- but it is the rule the movement workers now follow, and
    the reason this one is worth moving too.
    """
    touched = []
    milling._update_button_states = lambda: touched.append(True)

    mill.finish()  # the body runs in place here, so let it through
    milling._milling_worker(_session(), _config_with_a_stage())

    assert mill.calls, "the milling task never ran"
    assert touched == [], "the worker body updated the widget's buttons"
