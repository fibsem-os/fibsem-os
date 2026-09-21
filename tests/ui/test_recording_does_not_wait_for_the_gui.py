"""Recording a question, and taking one back, must not wait for the GUI thread.

Both are called from the workflow thread at moments the main thread is not free
to run a call back: asking, with the task about to park on a future, and
withdrawing, while a run is unwinding from an abort the main thread may itself
be waiting on. Marshalling either with ``await_return=True`` stalls the caller
for superqt's one-second timeout and then raises ``TimeoutError`` -- out of an
abort path -- while the queued call still lands later, on a run that has moved
on. It is invisible without a ``QApplication``, where the marshal degrades to a
plain call and everything passes. That is why this lives under ``tests/ui``:
the application is the condition.

The main thread below deliberately does **not** spin the event loop while the
worker runs, and gives it well under that second. A call that needs the main
thread is still waiting when the time is up.
"""

import os
import threading

import pytest
from psygnal.containers import EventedDict

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.proposals import (  # noqa: E402
    POINT_OF_INTEREST,
    Proposal,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.structures import MicroscopeState, Point  # noqa: E402

TASK = "Mill Rough"


@pytest.fixture
def experiment(qapp, tmp_path):
    exp = Experiment(path=tmp_path, name="no-wait-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    return exp


def _proposal() -> Proposal:
    return Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={"task_id": "run-1", "proposer": "detection"},
    )


def _off_the_main_thread(call, timeout_s=0.5):
    """Run ``call`` on a worker while this thread sits still, and say whether
    it came back."""
    done = {}

    def target():
        done["result"] = call()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)  # no processEvents: the GUI is "busy"
    assert not thread.is_alive(), "it waited for the main thread"
    return done["result"]


def test_recording_a_question_returns_while_the_gui_is_busy(experiment):
    lamella = experiment.positions[0]

    recorded = _off_the_main_thread(
        lambda: experiment.ask_proposal(lamella.id, TASK, _proposal())
    )

    assert recorded
    assert lamella.proposals[TASK].asking


def test_taking_a_question_back_returns_while_the_gui_is_busy(experiment):
    lamella = experiment.positions[0]
    experiment.ask_proposal(lamella.id, TASK, _proposal())

    result = _off_the_main_thread(
        lambda: experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")
    )

    assert result.applied, result.reason
    assert lamella.proposals[TASK].withdrawn


def test_the_gui_still_hears_about_both_on_its_own_thread(experiment, qapp):
    """Not waiting is not the same as not telling: the notification is handed
    to the main thread and arrives when it next looks."""
    lamella = experiment.positions[0]
    heard = []
    main = threading.current_thread()
    experiment.asked.connect(
        lambda *_: heard.append(("asked", threading.current_thread() is main))
    )
    experiment.decided.connect(
        lambda *_: heard.append(("decided", threading.current_thread() is main))
    )

    _off_the_main_thread(lambda: experiment.ask_proposal(lamella.id, TASK, _proposal()))
    _off_the_main_thread(
        lambda: experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")
    )
    assert heard == [], "nothing is delivered until the main thread looks"

    qapp.processEvents()

    assert heard == [("asked", True), ("decided", True)]
