"""A new lamella carries the id of the grid it is on (FIB-941).

Resolved by name, the way the rest of the system finds a grid: with a loader the
one grid on the stage, on a fixed holder the calibrated slot the position falls
in, then the experiment's record of that name. On the constructor, so anything
grouping lamellae by grid reads it on ``inserted``. None whenever a step has no
answer; never a guess, never a record created as a side effect.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

import fibsem.config as fibsem_config
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.microscopes._stage import SampleGrid, SlotCalibration
from fibsem.structures import FibsemStagePosition


def _ui(monkeypatch, tmp_path, arctis: bool):
    widget = AutoLamellaUI(parent_ui=None)
    if arctis:
        config = os.path.join(
            os.path.dirname(fibsem_config.__file__),
            "config",
            "sim-arctis-configuration.yaml",
        )
        monkeypatch.setattr(
            widget.system_widget,
            "load_configuration",
            lambda configuration_name=None: config,
        )
    widget.system_widget.connect_to_microscope()
    experiment = Experiment(path=tmp_path, name="stamp")
    os.makedirs(str(experiment.path), exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    widget.experiment = experiment
    return widget


@pytest.fixture
def arctis(qapp, monkeypatch, tmp_path):
    widget = _ui(monkeypatch, tmp_path, arctis=True)
    assert widget.microscope._stage.loader is not None
    yield widget
    widget.experiment = None
    widget.microscope.disconnect()
    widget.close()


@pytest.fixture
def fixed(qapp, monkeypatch, tmp_path):
    widget = _ui(monkeypatch, tmp_path, arctis=False)
    microscope = widget.microscope
    microscope.stage_is_compustage = False
    from fibsem.microscopes._stage import _create_sample_stage

    microscope._stage = _create_sample_stage(microscope)
    assert microscope._stage.loader is None
    yield widget
    widget.experiment = None
    microscope.disconnect()
    widget.close()


class TestWithALoader:
    def test_the_loaded_grid_is_stamped_and_present_on_inserted(self, arctis):
        exp = arctis.experiment
        stage = arctis.microscope._stage
        exp.sync_grids_from_inventory(stage)
        stage.ensure_loaded("Grid-02")
        record = exp.get_grid_by_name("Grid-02")
        seen = []
        exp.positions.events.inserted.connect(
            lambda index, value: seen.append(value.grid_id)
        )
        lamella = arctis.add_new_lamella()
        assert lamella.grid_id == record.id
        assert seen == [record.id]  # already there when the listeners redraw
        assert exp.get_lamellae_for_grid(record) == [lamella]

    def test_an_explicit_grid_wins_over_the_loaded_one(self, arctis):
        """A lamella marked on a grid's overview belongs to that grid, whether or
        not it is on the stage: the caller's grid is taken as given."""
        exp = arctis.experiment
        stage = arctis.microscope._stage
        exp.sync_grids_from_inventory(stage)
        stage.ensure_loaded("Grid-02")
        other = exp.get_grid_by_name("Grid-01")
        assert arctis.add_new_lamella(grid_id=other.id).grid_id == other.id

    def test_nothing_loaded_leaves_it_none(self, arctis):
        exp = arctis.experiment
        exp.sync_grids_from_inventory(arctis.microscope._stage)
        assert arctis.microscope._stage.loaded_grids == []
        assert arctis.add_new_lamella().grid_id is None

    def test_a_grid_with_no_record_leaves_it_none_and_creates_none(self, arctis):
        exp = arctis.experiment
        arctis.microscope._stage.ensure_loaded("Grid-01")
        assert exp.grids == []
        assert arctis.add_new_lamella().grid_id is None
        assert exp.grids == []


class TestOnAFixedHolder:
    def _calibrate(self, stage, name, x, grid_name):
        slot = stage.holder.slots[name]
        slot.position = FibsemStagePosition(name=name, x=x, y=1e-3, z=4e-3, r=0, t=0)
        slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-14T10:00:00", "t")
        slot.loaded_grid = SampleGrid(name=grid_name)
        return slot

    def test_the_slot_the_position_falls_in_is_stamped(self, fixed):
        stage = fixed.microscope._stage
        self._calibrate(stage, "Slot-01", -4e-3, "grid-ash")
        self._calibrate(stage, "Slot-02", 4e-3, "grid-oak")
        exp = fixed.experiment
        exp.add_grid(GridRecord(name="grid-ash"))
        oak = exp.add_grid(GridRecord(name="grid-oak"))
        near_oak = FibsemStagePosition(x=4.3e-3, y=1.2e-3, z=4e-3, r=0, t=0)
        assert fixed.add_new_lamella(stage_position=near_oak).grid_id == oak.id
        far = FibsemStagePosition(x=0.0, y=1e-3, z=4e-3, r=0, t=0)
        assert fixed.add_new_lamella(stage_position=far).grid_id is None

    def test_an_uncalibrated_slot_never_matches(self, fixed):
        stage = fixed.microscope._stage
        slot = stage.holder.slots["Slot-01"]
        slot.position = None
        slot.loaded_grid = SampleGrid(name="grid-ash")
        fixed.experiment.add_grid(GridRecord(name="grid-ash"))
        at_origin = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0, t=0)
        assert fixed.add_new_lamella(stage_position=at_origin).grid_id is None
