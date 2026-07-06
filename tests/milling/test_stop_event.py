"""Milling honors the stop event during "Preparing" (reference imaging / alignment),
not only at stage boundaries.

Regression: a Stop issued mid-"Preparing" — especially during overtilt's
``multi_step_alignment_v2`` — was ignored until beam-milling started (or never, on
the last stage), because the stop signal was not threaded into the strategy/prep
path. The fix passes an explicit ``stop_event`` through ``strategy.run`` and the
task, checked at the prep checkpoints, raising ``MillingStoppedError`` to unwind the
whole task (so its ``finally`` still restores imaging conditions).

Run headless:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/milling/test_stop_event.py
"""
import threading
import types

import pytest

from fibsem import utils
from fibsem.milling import FibsemMillingStage
from fibsem.milling.base import MillingStoppedError, raise_if_stopped
from fibsem.milling.patterning.patterns2 import TrenchPattern
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingStrategy
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig


def _spy(obj, name):
    """Replace ``obj.name`` with a wrapper that records calls, returns the recorder."""
    calls = []
    orig = getattr(obj, name)

    def wrapper(*a, **k):
        calls.append((a, k))
        return orig(*a, **k)

    setattr(obj, name, wrapper)
    return calls


def _trench_stage(name: str) -> FibsemMillingStage:
    stage = FibsemMillingStage(
        name=name,
        pattern=TrenchPattern(
            width=10e-6, depth=5e-6, spacing=2e-6,
            upper_trench_height=5e-6, lower_trench_height=5e-6,
        ),
    )
    stage.strategy = OvertiltTrenchMillingStrategy()
    return stage


# ── the primitive ────────────────────────────────────────────────────────────

def test_raise_if_stopped():
    raise_if_stopped(None)                 # no-op when absent
    raise_if_stopped(threading.Event())    # no-op when clear
    ev = threading.Event(); ev.set()
    with pytest.raises(MillingStoppedError):
        raise_if_stopped(ev)


# ── strategies abort before the beam starts ──────────────────────────────────

def test_standard_strategy_aborts_before_milling():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    stage = FibsemMillingStage(name="std")
    milled = _spy(microscope, "run_milling")
    ev = threading.Event(); ev.set()
    with pytest.raises(MillingStoppedError):
        StandardMillingStrategy().run(microscope, stage, stop_event=ev)
    assert milled == []  # beam never started


def test_overtilt_strategy_aborts_before_milling():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    stage = _trench_stage("ot")
    milled = _spy(microscope, "run_milling")
    ev = threading.Event(); ev.set()
    with pytest.raises(MillingStoppedError):
        stage.strategy.run(microscope, stage, stop_event=ev)
    assert milled == []


def test_overtilt_threads_stop_event_into_alignment(monkeypatch):
    """The overtilt strategy must pass its stop_event into multi_step_alignment_v2
    (so a stop mid-alignment aborts between steps), and honor a stop that arrives
    *during* alignment — before the beam mills."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    stage = _trench_stage("ot")
    milled = _spy(microscope, "run_milling")
    ev = threading.Event()  # not set yet: prep runs, stop arrives mid-alignment

    captured = {}

    def fake_alignment(*args, **kwargs):
        captured["stop_event"] = kwargs.get("stop_event")
        ev.set()  # simulate the user clicking Stop during alignment

    monkeypatch.setattr(
        "fibsem.milling.strategy.overtilt.alignment.multi_step_alignment_v2",
        fake_alignment,
    )

    with pytest.raises(MillingStoppedError):
        stage.strategy.run(microscope, stage, stop_event=ev)

    assert captured["stop_event"] is ev  # wiring: strategy -> alignment
    assert milled == []                  # aborted after alignment, before the beam


# ── task-level: aborts and still cleans up ───────────────────────────────────

def test_task_aborts_and_restores_conditions(tmp_path):
    """A stop before/at the first checkpoint aborts the whole task without milling,
    and the task's finally still restores imaging conditions (finish_milling)."""
    microscope, _ = utils.setup_session(manufacturer="Demo")
    stage = FibsemMillingStage(name="std")
    config = FibsemMillingTaskConfig.from_stages(stages=[stage], name="task")
    config.acquisition.imaging.path = str(tmp_path)

    ev = threading.Event(); ev.set()
    parent_ui = types.SimpleNamespace(_milling_stop_event=ev)

    milled = _spy(microscope, "run_milling")
    finished = _spy(microscope, "finish_milling")

    task = FibsemMillingTask(microscope, config, parent_ui=parent_ui)
    task.run()  # MillingStoppedError is caught inside run(); must not propagate

    assert milled == []        # never milled
    assert len(finished) >= 1  # but cleanup (finish_milling) still ran


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
