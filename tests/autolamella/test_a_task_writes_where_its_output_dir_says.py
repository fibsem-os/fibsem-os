"""A task writes where its output folder says (FIB-1253).

``AutoLamellaTask.output_dir`` is the one place a task decides where its
reference images, alignment runs, autofocus runs and milling are saved. Pointed
somewhere other than the lamella's folder, they all follow it: real Setup and
Rough Milling runs on Demo, through the session's runner.

Not asserted: where the alignment reference lands. Tasks share it by name in
the lamella's folder; it is lamella state, overwritten as the lamella moves on
(FIB-1258). The output folder is the lamella's, so the two agree; a layout
that split them would have to save that file to the lamella's folder.

The alignment and autofocus wrappers also return what they measured, instead of
dropping it.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.alignment import AlignmentResult
from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    AutoLamellaTask,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)
from fibsem.autofunctions.autofocus import AutoFocusResult
from fibsem.structures import BeamType
from fibsem.workflows.status import TaskStatus
from fibsem.workflows.task import ALIGNMENT_REFERENCE_IMAGE_FILENAME
from tests.protocols import planned, task_protocol
from tests.run_state import run_lamellae

CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml")
SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=CONFIG, setup_logging=False
    )
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(microscope, tmp_path):
    rough = MillRoughTaskConfig(task_name=ROUGH)
    for milling in rough.milling.values():
        milling.alignment.enabled = True
    exp = Experiment(path=tmp_path / "experiment", name="output-dir")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = task_protocol(
        planned(name=SETUP, required=True), planned(name=ROUGH, required=True)
    )
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP, use_autofocus=True
                ),
                ROUGH: rough,
            }
        ),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _top_level(folder: Path):
    return {p.name for p in folder.iterdir()} if folder.exists() else set()


def _write_elsewhere(monkeypatch, folder: Path) -> None:
    monkeypatch.setattr(AutoLamellaTask, "output_dir", property(lambda _: str(folder)))


def test_everything_a_task_writes_follows_the_output_folder(
    microscope, experiment, tmp_path, monkeypatch
):
    elsewhere = tmp_path / "elsewhere"
    _write_elsewhere(monkeypatch, elsewhere)
    lamella = experiment.positions[0]

    run_lamellae(microscope, experiment, [SETUP, ROUGH])

    assert lamella.task_state.status is TaskStatus.Completed
    written = _top_level(elsewhere)
    assert {"autofunctions", "Milling"} <= written
    assert any(name.startswith(f"ref_{ROUGH}_final") for name in written)
    in_lamella = _top_level(lamella.path)
    assert not {"autofunctions", "Milling"} & in_lamella, in_lamella
    assert not any(name.startswith(f"ref_{ROUGH}_") for name in in_lamella)


def test_the_output_folder_is_the_run_s_own(microscope, experiment):
    """``<lamella>/<task name>/<YYYYMMDD-HHMMSS>/`` (FIB-1275): each task object
    is one run, with its own folder, so a re-run overwrites nothing."""
    from pathlib import Path

    lamella = experiment.positions[0]
    task = MillRoughTask(microscope, lamella.task_config[ROUGH], lamella)

    folder = Path(task.output_dir)
    assert folder.parent == Path(lamella.path) / ROUGH
    assert len(folder.name) == len("20261014-093012") and folder.name[8] == "-"
    assert task.config.imaging.path == task.output_dir  # its images follow
    assert not folder.exists(), "made by the first save, not before"

    os.makedirs(task.output_dir)  # the run saved something
    again = MillRoughTask(microscope, lamella.task_config[ROUGH], lamella)
    assert again.output_dir != task.output_dir, "a second run in the same second"


def test_a_task_name_a_folder_cannot_hold_is_made_safe(microscope, experiment):
    from fibsem.util.timestamps import now
    from fibsem.workflows.task import run_folder

    lamella = experiment.positions[0]
    folder = run_folder(lamella.path, 'Rough: 1/2 "fast"', now())

    assert os.path.basename(os.path.dirname(folder)) == "Rough- 1-2 -fast-"


def test_a_round_has_a_folder_of_its_own(tmp_path):
    from fibsem.util.timestamps import now
    from fibsem.workflows.task import run_folder

    folder = run_folder(tmp_path, "Mill", now(), iteration=3)
    assert folder == os.path.join(str(tmp_path), "Mill", "0003")

    os.makedirs(folder)  # a re-run of the same iteration
    assert run_folder(tmp_path, "Mill", now(), iteration=3) == folder + "-2"


def test_the_wrappers_return_what_they_measured(microscope, experiment):
    lamella = experiment.positions[0]
    run_lamellae(microscope, experiment, [SETUP])  # leaves the alignment reference
    task = MillRoughTask(microscope, lamella.task_config[ROUGH], lamella)

    assert isinstance(
        task._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME), AlignmentResult
    )
    assert task._align_reference_image("no-such-reference") is None
    assert isinstance(task._run_autofocus(BeamType.ELECTRON), AutoFocusResult)


def test_a_missing_reference_is_recorded_as_a_skipped_alignment(microscope, experiment):
    """The alignment the task asked for did not happen; the record says so
    beside the ones that did, instead of only a log line (FIB-1255)."""
    lamella = experiment.positions[0]
    task = MillRoughTask(microscope, lamella.task_config[ROUGH], lamella)
    recorded = []
    microscope.session.record_signal.connect(
        lambda kind, payload: recorded.append(payload) if kind == "alignment" else None
    )
    try:
        assert task._align_reference_image("no-such-reference") is None
    finally:
        microscope.session.record_signal.disconnect()

    (skipped,) = recorded
    assert skipped["status"] == "skipped"
    assert "no-such-reference" in skipped["reason"]
