"""Tests for the grid workflow Phase 1: GridRecord, Experiment.grids, and the
GridTask lifecycle (pre_task -> _run -> post_task)."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from fibsem import utils
from fibsem.microscopes._stage import SampleGrid
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.workflows.tasks.grid_tasks import (
    GridTask,
    GridTaskConfig,
)


# --- helpers ---------------------------------------------------------------

@dataclass
class _NoOpGridConfig(GridTaskConfig):
    task_type: ClassVar[str] = "NOOP_GRID"
    display_name: ClassVar[str] = "No-op"


class _NoOpGridTask(GridTask):
    config_cls = _NoOpGridConfig

    def _run(self):
        pass


class _BoomGridTask(GridTask):
    config_cls = _NoOpGridConfig

    def _run(self):
        raise RuntimeError("boom")


@pytest.fixture
def demo_microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    return microscope


@pytest.fixture
def experiment(tmp_path):
    return Experiment.create(path=str(tmp_path), name="exp-grid-test")


# --- GridRecord ------------------------------------------------------------

def test_grid_record_round_trip():
    g = GridRecord(name="grid-aspen")
    g.task_state.name = "NOOP_GRID"
    g.task_state.status = AutoLamellaTaskStatus.Completed
    g.task_history.append(g.task_state)

    g2 = GridRecord.from_dict(g.to_dict())

    assert g2.name == "grid-aspen"
    assert g2._id == g._id
    assert g2.task_state.status is AutoLamellaTaskStatus.Completed
    assert g2.completed_tasks == ["NOOP_GRID"]
    assert g2.has_completed_task("NOOP_GRID")
    assert not g2.is_failure


def test_grid_record_is_failure():
    g = GridRecord(name="grid-x")
    assert not g.is_failure
    g.task_state.status = AutoLamellaTaskStatus.Failed
    assert g.is_failure


# --- Experiment.grids ------------------------------------------------------

def test_add_grid_and_lookup(experiment):
    experiment.add_grid(GridRecord(name="grid-aspen"))
    assert experiment.get_grid_by_name("grid-aspen") is not None
    assert experiment.get_grid_by_name("missing") is None


def test_add_grid_rejects_duplicate(experiment):
    experiment.add_grid(GridRecord(name="grid-aspen"))
    with pytest.raises(ValueError):
        experiment.add_grid(GridRecord(name="grid-aspen"))


def test_add_grid_type_checked(experiment):
    with pytest.raises(TypeError):
        experiment.add_grid("not-a-grid")


def test_grids_persist_and_reload(experiment):
    experiment.add_grid(GridRecord(name="grid-aspen"))
    experiment.add_grid(GridRecord(name="grid-birch"))
    experiment.save()

    loaded = Experiment.load(f"{experiment.path}/experiment.yaml")
    assert [g.name for g in loaded.grids] == ["grid-aspen", "grid-birch"]


def test_from_dict_back_compat_without_grids_key(experiment):
    # an experiment dict predating the grids field must still load
    ddict = experiment.to_dict()
    del ddict["grids"]
    restored = Experiment.from_dict(ddict)
    assert list(restored.grids) == []


# --- sync_grids_from_holder ------------------------------------------------

def test_sync_grids_from_holder_is_idempotent(demo_microscope, experiment):
    slot_names = list(demo_microscope._stage.holder.slots.keys())[:2]
    demo_microscope._stage.holder.slots[slot_names[0]].loaded_grid = SampleGrid(name="grid-aspen")
    demo_microscope._stage.holder.slots[slot_names[1]].loaded_grid = SampleGrid(name="grid-birch")

    experiment.sync_grids_from_holder(demo_microscope)
    experiment.sync_grids_from_holder(demo_microscope)  # must not duplicate

    assert [g.name for g in experiment.grids] == ["grid-aspen", "grid-birch"]


# --- GridTask lifecycle ----------------------------------------------------

def test_grid_task_lifecycle_success(demo_microscope, experiment):
    record = GridRecord(name="grid-aspen")
    experiment.add_grid(record)

    _NoOpGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                  record, experiment).run()

    assert record.task_state.status is AutoLamellaTaskStatus.Completed
    assert record.has_completed_task("NOOP_GRID")
    assert record.task_state.duration >= 0


def test_grid_task_lifecycle_failure_records_state(demo_microscope, experiment):
    record = GridRecord(name="grid-birch")
    experiment.add_grid(record)

    with pytest.raises(RuntimeError):
        _BoomGridTask(demo_microscope, _NoOpGridConfig(task_name="NOOP_GRID"),
                      record, experiment).run()

    assert record.task_state.status is AutoLamellaTaskStatus.Failed
    assert record.is_failure
    assert record.task_state.status_message == "boom"
