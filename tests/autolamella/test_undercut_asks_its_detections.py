"""The undercut asks its detections through ``detect``, on the record.

The real task, the real model: a supervised or automated Mill Undercut on the
Demo microscope with the waffle checkpoint, four detections in one run (two in
the coincident alignment, one per undercut, one to finish). Each is a
``detection`` question through ``ask``, whether the review preference is on
or off; headless nobody answers, so each goes on the record as proposed,
``Unreviewed``, on the image the model ran on, and the training data is
written as the Detection tab writes it.

Needs the ``ml`` extra and the checkpoint already in the Hugging Face cache;
skipped otherwise (nothing here downloads).
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

from tests.fixtures.microscopes import demo_microscope

pytest.importorskip("segmentation_models_pytorch")

import fibsem.config as fibsem_config
from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTask,
    MillUndercutTaskConfig,
)
from fibsem.structures import Point
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import DETECTION, Decision, DecisionOutcome
from fibsem.workflows.status import TaskStatus
from tests.protocols import planned, task_protocol

UNDERCUT = "Mill Undercut"
CHECKPOINT = "autolamella-waffle-20240107.pt"


@pytest.fixture
def checkpoint(monkeypatch) -> str:
    """The waffle checkpoint from the local Hugging Face cache, or skip. The
    task names the checkpoint by file name and ``download_checkpoint`` would
    go to the hub for it; routed to the cached file so nothing does."""
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


microscope = demo_microscope(scope="function")


def _experiment(tmp_path: Path, microscope, attention=Attention.automated):
    exp = Experiment(path=tmp_path, name="undercut-exp")
    exp.task_protocol = task_protocol(
        planned(name=UNDERCUT, required=True, attention=attention)
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        # one angle: the default milling config has one undercut stage
        EventedDict(
            {UNDERCUT: MillUndercutTaskConfig(task_name=UNDERCUT, milling_angles=[25])}
        ),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _task(microscope, exp) -> MillUndercutTask:
    manager = TaskManager(microscope=microscope, experiment=exp)
    lamella = exp.positions[0]
    return MillUndercutTask(
        microscope=microscope,
        config=lamella.task_config[UNDERCUT],
        lamella=lamella,
        task_manager=manager,
    )


def test_the_task_declares_the_detection():
    assert MillUndercutTask.asks == (DETECTION,)


def test_headless_every_detection_goes_on_the_record_unreviewed(
    microscope, tmp_path, checkpoint
):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    _task(microscope, exp).run()

    assert [p.kind for p in lamella.proposals_from(UNDERCUT)] == [DETECTION] * 4, (
        "two in the coincident alignment, one per undercut, one to finish"
    )
    detections = [p for p in lamella.proposals_from(UNDERCUT) if p.kind == DETECTION]
    assert [p.provenance["features"] for p in detections] == [
        ["LamellaCentre"],
        ["LamellaCentre"],
        ["LamellaTopEdge"],
        ["LamellaCentre"],
    ]
    for proposal in detections:
        assert proposal.unreviewed
        assert proposal.provenance["proposer"] == "autolamella-waffle-20240107"
        assert proposal.provenance["checkpoint"] == CHECKPOINT, "by name, as asked for"
        image = proposal.provenance["reference_image"]
        assert image.startswith("ml-") and not os.path.isabs(image), image
        assert os.path.exists(os.path.join(str(lamella.path), image)), (
            "the image the model ran on, in the lamella's folder"
        )
        (feature,) = proposal.values["features"]
        assert isinstance(feature["px"], Point)
        assert proposal.current.values == proposal.values, "used as proposed"
        stem = os.path.splitext(image)[0].rsplit("_", 1)[0]
        assert os.path.exists(
            os.path.join(fibsem_config.DATA_ML_PATH, f"{stem}.tif")
        ), "the training image the Detection tab would have written"
    assert detections[0].provenance["reference_image"].endswith("_eb.tif"), (
        "the first coincident detection is on the electron image"
    )
    assert lamella.task_state.status is TaskStatus.Completed


def test_a_rejected_detection_fails_the_task(
    microscope, tmp_path, checkpoint, monkeypatch
):
    """Reject in the Review tab means what it says: the task does not carry
    on with a detection somebody refused."""

    def rejected(self, *args, **kwargs):
        return Decision(
            outcome=DecisionOutcome.Rejected, actor="operator:op", reason="off target"
        )

    monkeypatch.setattr(AutoLamellaTask, "ask", rejected)
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    with pytest.raises(RuntimeError, match="rejected: off target"):
        _task(microscope, exp).run()

    assert lamella.task_state.status is TaskStatus.Failed
