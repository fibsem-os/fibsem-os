"""A volume position is edited through the experiment: each edit refuses with
why, changing nothing, or is made and saved."""

import pytest

from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.structures import (
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeState,
    Point,
)
from fibsem.workflows.status import TaskState, TaskStatus


def _state() -> MicroscopeState:
    return MicroscopeState(stage_position=FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))


@pytest.fixture
def aspen(experiment):
    return experiment.add_position("aspen", _state(), Point(0, 1e-6), iterations=3)


def _cut(position, iteration: int) -> None:
    position.task_history.append(
        TaskState(
            name="Volume FIB Milling",
            iteration=iteration,
            status=TaskStatus.Completed,
        )
    )


def _saved(experiment):
    return VolumeExperiment.load(experiment.path).get_position_by_name


def test_a_rename_moves_the_folder_and_is_saved(experiment, aspen):
    folder = experiment.position_path(aspen)
    folder.mkdir(parents=True)

    assert experiment.rename_position("aspen", " birch ") == ""

    assert aspen.name == "birch"
    assert not folder.exists() and experiment.position_path(aspen).exists()
    assert _saved(experiment)("birch") is not None


def test_a_rename_is_refused_with_why(experiment, aspen):
    experiment.add_position("birch", _state())

    assert "already a position" in experiment.rename_position("aspen", "birch")
    assert "needs a name" in experiment.rename_position("aspen", "  ")
    assert "No position" in experiment.rename_position("oak", "elm")
    _cut(aspen, 0)
    assert "keeps its name" in experiment.rename_position("aspen", "elm")
    assert aspen.name == "aspen"


def test_a_removed_position_keeps_its_folder(experiment, aspen):
    folder = experiment.position_path(aspen)
    folder.mkdir(parents=True)

    assert experiment.remove_position("aspen") == ""

    assert experiment.positions == [] and folder.exists()
    assert _saved(experiment)("aspen") is None


def test_iterations_never_drop_below_the_slices_cut(experiment, aspen):
    _cut(aspen, 0)
    _cut(aspen, 1)

    assert "already cut 2" in experiment.set_iterations("aspen", 1)
    assert "at least one" in experiment.set_iterations("aspen", 0)
    assert experiment.set_iterations("aspen", 5) == ""
    assert _saved(experiment)("aspen").iterations == 5


def test_the_point_of_interest_moves_only_before_the_first_slice(experiment, aspen):
    assert experiment.set_poi("aspen", Point(2e-6, 3e-6)) == ""
    assert _saved(experiment)("aspen").poi.x == pytest.approx(2e-6)

    _cut(aspen, 0)

    assert "has cut slices" in experiment.set_poi("aspen", Point(0, 0))
    assert aspen.poi.x == pytest.approx(2e-6)


def test_the_alignment_area_must_be_inside_the_frame(experiment, aspen):
    before = aspen.alignment_area

    assert "inside the frame" in experiment.set_alignment_area(
        "aspen", FibsemRectangle(0.8, 0.8, 0.5, 0.5)
    )
    assert aspen.alignment_area == before
    area = FibsemRectangle(0.1, 0.2, 0.3, 0.4)
    assert experiment.set_alignment_area("aspen", area) == ""
    assert _saved(experiment)("aspen").alignment_area == area
