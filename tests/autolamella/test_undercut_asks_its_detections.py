"""The undercut asks its detections through ``detect``, on the record.

The real task, the real model: a supervised or automated Mill Undercut on the
Demo microscope with the waffle checkpoint, four detections in one run (two in
the coincident alignment, one per undercut, one to finish). With the review
preference on each is a ``detection`` question through ``ask``; headless
nobody answers, so each goes on the record as proposed, ``Unreviewed``, on the
image the model ran on, and the training data is written as the Detection
tab writes it. With the preference off it is ``update_detection_ui`` exactly
as before.

Needs the ``ml`` extra and the checkpoint already in the Hugging Face cache;
skipped otherwise (nothing here downloads).
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

pytest.importorskip("segmentation_models_pytorch")

import fibsem.config as fibsem_config
from fibsem import utils
from fibsem.applications.autolamella.proposals import (
    DETECTION,
    TASK_RESULT,
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
from fibsem.applications.autolamella.workflows import core as C
from fibsem.applications.autolamella.workflows.tasks import base as B
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTask,
    MillUndercutTaskConfig,
)
from fibsem.structures import Point

UNDERCUT = "Mill Undercut"
CHECKPOINT = "autolamella-waffle-20240107.pt"
CONFIG = os.path.join(fibsem_config.CONFIG_PATH, "microscope-configuration.yaml")


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


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def _experiment(tmp_path: Path, microscope, attention=Attention.automated):
    exp = Experiment(path=tmp_path, name="undercut-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=UNDERCUT, required=True, attention=attention
                )
            ]
        )
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


def _task(microscope, exp, review_enabled=True) -> MillUndercutTask:
    manager = TaskManager(microscope=microscope, experiment=exp, parent_ui=None)
    manager.review_enabled = review_enabled
    lamella = exp.positions[0]
    return MillUndercutTask(
        microscope=microscope,
        config=lamella.task_config[UNDERCUT],
        lamella=lamella,
        parent_ui=None,
        task_manager=manager,
    )


def test_the_task_declares_the_detection():
    assert MillUndercutTask.questions == (DETECTION,)


def test_headless_every_detection_goes_on_the_record_unreviewed(
    microscope, tmp_path, checkpoint
):
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    _task(microscope, exp).run()

    assert [p.kind for p in lamella.proposals[UNDERCUT]] == [DETECTION] * 4 + [
        TASK_RESULT
    ], "two in the coincident alignment, one per undercut, one to finish"
    detections = [p for p in lamella.proposals[UNDERCUT] if p.kind == DETECTION]
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
    assert lamella.task_state.status is AutoLamellaTaskStatus.Completed


def test_with_the_preference_off_the_detection_tab_prompt_asks_as_before(
    microscope, tmp_path, checkpoint, monkeypatch
):
    calls = []
    real = B.update_detection_ui

    def counted(**kwargs):
        calls.append(kwargs["features"][0].name)
        return real(**kwargs)

    monkeypatch.setattr(B, "update_detection_ui", counted)
    monkeypatch.setattr(C, "update_detection_ui", counted)
    exp = _experiment(tmp_path, microscope, attention=Attention.supervised)
    lamella = exp.positions[0]

    _task(microscope, exp, review_enabled=False).run()

    assert calls == [
        "LamellaCentre",
        "LamellaCentre",
        "LamellaTopEdge",
        "LamellaCentre",
    ]
    assert [p.kind for p in lamella.proposals[UNDERCUT]] == [TASK_RESULT], (
        "no record of the detections"
    )


def test_a_rejected_detection_fails_the_task(
    microscope, tmp_path, checkpoint, monkeypatch
):
    """Reject in the Review tab means what it says: the task does not carry
    on with a detection somebody refused."""

    def rejected(self, *args, **kwargs):
        return Decision(
            outcome=DecisionOutcome.Rejected, author="human:op", reason="off target"
        )

    monkeypatch.setattr(AutoLamellaTask, "ask", rejected)
    exp = _experiment(tmp_path, microscope)
    lamella = exp.positions[0]

    with pytest.raises(RuntimeError, match="rejected: off target"):
        _task(microscope, exp).run()

    assert lamella.task_state.status is AutoLamellaTaskStatus.Failed
