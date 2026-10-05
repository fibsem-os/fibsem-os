"""The app's event recorder watches the experiment the app has open, for its
questions and decisions (FIB-1034): whether the experiment was opened after
the microscope connected, or before it.

The real main window on Demo, with the recorder it builds itself.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402

from fibsem.applications.autolamella.proposals import (  # noqa: E402
    POINT_OF_INTEREST,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.structures import MicroscopeState, Point  # noqa: E402

TASK = "Setup Lamella Position"
RUN = "run-1"


@pytest.fixture(scope="module")
def window(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    yield win
    if win.autolamella_ui.microscope is not None:
        win.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


def _experiment(path, name):
    exp = Experiment(path=path, name=name)
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = RUN
    lamella.task_state.status = AutoLamellaTaskStatus.Completed
    return exp


def _decide(experiment):
    """The Review tab's decision on the lamella's point of interest."""
    lamella = experiment.positions[0]
    proposal = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={"task_id": RUN, "proposer": "current-poi"},
    )
    lamella.proposals[TASK] = [proposal]
    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0.0)},
            via="review",
            task_id=RUN,
            proposal_id=proposal.id,
        ),
    )
    assert result.applied, result.reason
    return proposal


def _decided(ui):
    return [
        e["payload"]["proposal_id"]
        for e in ui._event_recorder.buffer.events_since(0)["events"]
        if e["kind"] == "proposal_decided"
    ]


def test_the_experiment_opened_is_the_one_watched(window, tmp_path):
    ui = window.autolamella_ui
    first = _experiment(tmp_path, "first")
    second = _experiment(tmp_path, "second")

    ui._adopt_experiment(first)
    on_first = _decide(first)
    ui._adopt_experiment(second)
    on_second = _decide(second)
    _decide(first)  # no longer open: not the app's to record

    assert _decided(ui) == [on_first.id, on_second.id]


def test_an_experiment_open_before_the_microscope_connects_is_watched(window, tmp_path):
    ui = window.autolamella_ui
    experiment = _experiment(tmp_path, "open-first")
    ui._adopt_experiment(experiment)
    # The connect button toggles: disconnect, then connect again.
    ui.system_widget.connect_to_microscope()
    assert ui.microscope is None and ui._event_recorder is None
    ui.system_widget.connect_to_microscope()  # a new stream for a new connection
    assert ui.microscope is not None

    proposal = _decide(experiment)

    assert _decided(ui) == [proposal.id]
