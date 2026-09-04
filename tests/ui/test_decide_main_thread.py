"""Experiment.decide runs on the Qt main thread and blocks its caller.

The Review tab calls it from the GUI thread and the agent server from a
worker; both must land on the main thread because a confirm can end in Qt
work (a generative review adds lamellae, whose list rebuild is not safe off
the GUI thread). Without a QApplication it is a plain call -- covered in
tests/autolamella/test_proposals.py.
"""

import os
import threading
import time

import pytest
from psygnal.containers import EventedDict

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QThread  # noqa: E402

from fibsem.applications.autolamella.proposals import (  # noqa: E402
    MILLING_SETUP,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (  # noqa: E402
    SelectMillingPositionTaskConfig,
)
from fibsem.structures import FibsemStagePosition, MicroscopeState, Point  # noqa: E402

SETUP = "Setup Lamella Position"


def _experiment(tmp_path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        MicroscopeState(stage_position=FibsemStagePosition()),
        EventedDict({SETUP: SelectMillingPositionTaskConfig(task_name=SETUP)}),
    )
    exp.positions[0].proposals[SETUP] = Proposal(
        kind=MILLING_SETUP, values={"poi": Point(0.0, 0.0)}
    )
    return exp


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def test_decide_from_a_worker_lands_on_the_main_thread_and_blocks(qapp, tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    seen = {}
    exp.decided.connect(lambda *_: seen.setdefault("thread", QThread.currentThread()))

    def worker():
        seen["worker_thread"] = QThread.currentThread()
        seen["result"] = exp.decide(
            lamella.id,
            SETUP,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                author="agent:test",
                values={"poi": Point(1e-6, 0.0)},
            ),
        )
        seen["after"] = lamella.poi  # the worker sees the applied write on return

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    _spin_until(qapp, lambda: "result" in seen)
    t.join(5)

    assert seen["result"].applied is True
    assert seen["thread"] is qapp.thread(), "the write ran on the main thread"
    assert seen["worker_thread"] is not qapp.thread()
    assert seen["after"] == Point(1e-6, 0.0), "the caller blocked until it was applied"


def test_decide_on_the_main_thread_is_a_direct_call(qapp, tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(2e-6, 0.0)},
        ),
    )
    assert result.applied is True
    assert lamella.poi == Point(2e-6, 0.0)
