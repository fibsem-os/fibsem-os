"""A volume position is edited through the experiment: each edit refuses with
why, changing nothing, or is made and saved."""

import pytest

from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.structures.settings import (
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


def test_a_new_positions_name_is_its_number_and_two_words(experiment):
    import re

    name = experiment.next_position_name()

    assert re.fullmatch(r"01-[a-z]+-[a-z]+", name)
    experiment.add_position(name, _state())
    assert experiment.next_position_name().startswith("02-")


def test_a_new_name_is_never_a_removed_positions_folder(experiment, monkeypatch):
    import fibsem.applications.volume.experiment as module

    words = iter(["brave-otter", "brave-otter", "calm-heron"])
    monkeypatch.setattr(module.petname, "generate", lambda n: next(words))
    first = experiment.add_position(experiment.next_position_name(), _state())
    experiment.position_path(first).mkdir(parents=True)
    experiment.remove_position(first.name)

    assert experiment.next_position_name() == "01-calm-heron"


def test_a_removed_position_keeps_its_folder(experiment, aspen):
    folder = experiment.position_path(aspen)
    folder.mkdir(parents=True)

    assert experiment.remove_position("aspen") == ""

    assert experiment.positions == [] and folder.exists()
    assert _saved(experiment)("aspen") is None


def test_iterations_never_drop_below_the_slices_cut(experiment, aspen):
    _cut(aspen, 0)
    _cut(aspen, 1)

    assert "already cut 2" in experiment.patch_position("aspen", iterations=1)
    assert "at least one" in experiment.patch_position("aspen", iterations=0)
    assert experiment.patch_position("aspen", iterations=5) == ""
    assert _saved(experiment)("aspen").iterations == 5


def test_the_point_of_interest_moves_only_before_the_first_slice(experiment, aspen):
    assert experiment.patch_position("aspen", poi=Point(2e-6, 3e-6)) == ""
    assert _saved(experiment)("aspen").poi.x == pytest.approx(2e-6)

    _cut(aspen, 0)

    assert "has cut slices" in experiment.patch_position("aspen", poi=Point(0, 0))
    assert aspen.poi.x == pytest.approx(2e-6)


def test_the_alignment_area_must_be_inside_the_frame(experiment, aspen):
    before = aspen.alignment_area

    assert "inside the frame" in experiment.patch_position(
        "aspen", alignment_area=FibsemRectangle(0.8, 0.8, 0.5, 0.5)
    )
    assert aspen.alignment_area == before
    area = FibsemRectangle(0.1, 0.2, 0.3, 0.4)
    assert experiment.patch_position("aspen", alignment_area=area) == ""
    assert _saved(experiment)("aspen").alignment_area == area


def test_a_patch_is_all_or_nothing(experiment, aspen):
    before = aspen.alignment_area

    why = experiment.patch_position(
        "aspen",
        iterations=7,
        alignment_area=FibsemRectangle(0.8, 0.8, 0.5, 0.5),
        description="near the crack",
    )

    assert "inside the frame" in why
    assert aspen.iterations == 3
    assert aspen.alignment_area == before and aspen.description == ""
    assert experiment.patch_position("aspen", description=" near the crack ") == ""
    assert _saved(experiment)("aspen").description == "near the crack"
