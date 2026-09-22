"""A supervised Mill Undercut, the real task with the real model, answers
each of its detections in the Review tab (FIB-1045).

Four detections in one run, each asked through ``ask``: the run holds, the
Review tab is fronted with the question selected, the answer is a decision on
the record, and the stage moves by the corrected value. The milling session
between them is the prompt it has always been, and its Continue is the
operator's decision on the task's result, so the task ends Completed with
nothing left waiting.

Needs the ``ml`` extra and the checkpoint already in the Hugging Face cache;
skipped otherwise (nothing here downloads).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("segmentation_models_pytorch")

from psygnal.containers import EventedDict
from PyQt5.QtCore import QCoreApplication, QEvent

import fibsem.config as fibsem_config
from fibsem import conversions
from fibsem.applications.autolamella.proposals import (
    DETECTION,
    AuthorKind,
    Decision,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.status import HoldKind
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTask,
    MillUndercutTaskConfig,
)
from fibsem.structures import FibsemImage, Point

UNDERCUT = "Mill Undercut"
CHECKPOINT = "autolamella-waffle-20240107.pt"


@pytest.fixture
def checkpoint(monkeypatch) -> str:
    from huggingface_hub import try_to_load_from_cache

    import fibsem.segmentation.model as model_module
    import fibsem.segmentation.utils as seg_utils

    cached = try_to_load_from_cache(fibsem_config.HUGGINFACE_REPO, CHECKPOINT)
    if not isinstance(cached, str) or not os.path.exists(cached):
        pytest.skip(f"{CHECKPOINT} is not in the Hugging Face cache")

    def _cached(name: str) -> str:
        return name if os.path.exists(name) else cached

    monkeypatch.setattr(seg_utils, "download_checkpoint", _cached)
    monkeypatch.setattr(model_module, "download_checkpoint", _cached)
    return cached


@pytest.fixture
def window(qapp, tmp_path, checkpoint):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    ui = win.autolamella_ui
    experiment = Experiment(path=tmp_path, name="undercut-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=UNDERCUT, required=True, attention=Attention.supervised
                )
            ]
        )
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(),
        EventedDict(
            {UNDERCUT: MillUndercutTaskConfig(task_name=UNDERCUT, milling_angles=[25])}
        ),
    )
    lamella = experiment.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = ui.microscope.get_microscope_state()
    ui.experiment = experiment
    win._on_experiment_update()
    win._preferences.features.proposer_reviewer_workflow_enabled = True
    win._apply_review_visibility()
    win.tab_widget.setCurrentIndex(0)
    win._started = []

    yield win
    for manager, thread in win._started:
        manager.stop()
        _pump(qapp, lambda: not thread.is_alive())
    microscope = ui.microscope
    ui.disconnect_from_microscope()
    if microscope is not None:
        microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()


def _pump(qapp, predicate, timeout_s=60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _start(window):
    ui = window.autolamella_ui
    experiment = ui.experiment
    manager = TaskManager(microscope=ui.microscope, experiment=experiment, parent_ui=ui)
    manager.review_enabled = True
    lamella = experiment.positions[0]
    task = MillUndercutTask(
        microscope=ui.microscope,
        config=lamella.task_config[UNDERCUT],
        lamella=lamella,
        parent_ui=ui,
        task_manager=manager,
    )
    seen = {}

    def _target():
        try:
            task.run()
        except Exception as exc:  # noqa: BLE001 - run re-raises for the manager
            seen["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    window._started.append((manager, thread))
    return task, thread, seen


def _asking(lamella):
    """The detection the run is held on, if one is up."""
    proposal = lamella.proposal(UNDERCUT, DETECTION)
    return proposal if proposal is not None and proposal.asking else None


def test_each_detection_is_answered_in_the_review_tab_and_the_stage_follows(
    window, qapp
):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    microscope = ui.microscope
    moves = []
    real_stable, real_vertical = microscope.stable_move, microscope.vertical_move
    microscope.stable_move = lambda dx, dy, **kw: (
        moves.append(("stable", dx, dy)),
        real_stable(dx, dy, **kw),
    )[1]
    microscope.vertical_move = lambda dx, dy, **kw: (
        moves.append(("vertical", dx, dy)),
        real_vertical(dx=dx, dy=dy, **kw),
    )[1]
    task, thread, seen = _start(window)
    tab = window.review_tab

    # 1. the coincident alignment's electron detection, corrected by a drag
    assert _pump(qapp, lambda: _asking(lamella) is not None and ui.hold is not None), (
        seen,
        ui.hold,
    )
    first = _asking(lamella)
    assert ui.hold.kind is HoldKind.decision
    assert window.tab_widget.currentWidget() is tab, "fronted on the question"
    assert first.provenance["features"] == ["LamellaCentre"]
    assert first.provenance["reference_image"].endswith("_eb.tif")
    (feature,) = first.values["features"]
    moved = Point(feature["px"].x + 40, feature["px"].y - 25)
    image = FibsemImage.load(
        os.path.join(str(lamella.path), first.provenance["reference_image"])
    )
    expected = conversions.image_to_microscope_image_coordinates(
        moved, image.data, image.metadata.pixel_size.x
    )
    result = ui.experiment.decide(
        lamella.id,
        UNDERCUT,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": moved}]},
            proposal_id=first.id,
            via="review",
        ),
    )
    assert result.applied, result.reason

    # 2. the ion one, confirmed as it stands from the tab
    assert _pump(
        qapp, lambda: _asking(lamella) is not None and _asking(lamella) is not first
    ), seen
    assert moves and moves[0][0] == "stable"
    assert (moves[0][1], moves[0][2]) == pytest.approx((expected.x, expected.y)), (
        "the stage moved by the corrected point, not the model's"
    )
    second = _asking(lamella)
    assert first.current.outcome is DecisionOutcome.Confirmed
    assert first.delta()["features"]
    assert tab.select(lamella.id, UNDERCUT)
    tab.confirm_current()

    # 3. the undercut's own detection, then the milling session's Continue
    assert _pump(qapp, lambda: (_asking(lamella) or second) is not second), seen
    third = _asking(lamella)
    assert third.provenance["features"] == ["LamellaTopEdge"]
    assert tab.select(lamella.id, UNDERCUT)
    tab.confirm_current()
    assert _pump(qapp, lambda: "Run Milling" in ui.label_instructions.text()), (
        ui.label_instructions.text(),
        seen,
    )
    assert ui.hold is not None and ui.hold.kind is HoldKind.question, "a session"
    ui.pushButton_no.click()  # Continue, without milling

    # 4. the final alignment
    assert _pump(qapp, lambda: (_asking(lamella) or third) is not third), seen
    fourth = _asking(lamella)
    assert fourth.provenance["features"] == ["LamellaCentre"]
    assert tab.select(lamella.id, UNDERCUT)
    tab.confirm_current()

    assert _pump(qapp, lambda: not thread.is_alive()), "the task went on to its end"
    assert "error" not in seen, seen
    detections = [p for p in lamella.proposals[UNDERCUT] if p.kind == DETECTION]
    assert [p.id for p in detections] == [first.id, second.id, third.id, fourth.id]
    for proposal in detections:
        decision = proposal.current
        assert decision.outcome is DecisionOutcome.Confirmed
        assert decision.author.kind is AuthorKind.human
        assert decision.proposal_id == proposal.id, "each names its own"
    assert ui.hold is None
    assert lamella.task_state.status is AutoLamellaTaskStatus.Completed, (
        "the milling Continue was the decision on the result"
    )

    # the record survives the file
    ui.experiment.save()
    again = Experiment.load(Path(ui.experiment.path) / "experiment.yaml")
    saved = [p for p in again.positions[0].proposals[UNDERCUT] if p.kind == DETECTION]
    assert [p.current.outcome for p in saved] == [DecisionOutcome.Confirmed] * 4
    assert saved[0].delta()["features"]
