"""GridRecord: the workflow's record of a grid, and its place on the Experiment.

A record is linked to the hardware by name and to lamellae by `Lamella.grid_id`;
it holds no slot, position or loaded state, so it stays valid when the grid moves
or leaves the magazine. Experiments written before records existed load with none.
"""

import yaml

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridQuality,
    GridRecord,
    Lamella,
)
from fibsem.microscopes._stage import DemoSampleLoader, SampleGrid, _create_sample_stage


def _experiment(tmp_path) -> Experiment:
    return Experiment(path=tmp_path, name="exp")


def _lamella(tmp_path, name="lam-01", grid_id=None) -> Lamella:
    return Lamella(path=tmp_path / name, number=1, petname=name, grid_id=grid_id)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class TestGridRecord:
    def test_defaults(self):
        grid = GridRecord(name="grid-aspen")
        assert grid.id
        assert grid.quality is GridQuality.UNASSESSED
        assert grid.task_history == []
        assert not grid.is_failure

    def test_round_trip(self):
        grid = GridRecord(name="grid-aspen", description="HeLa, batch B")
        grid.quality = GridQuality.GOOD
        done = AutoLamellaTaskState(
            name="overview_sem", status=AutoLamellaTaskStatus.Completed
        )
        grid.task_history.append(done)
        again = GridRecord.from_dict(yaml.safe_load(yaml.safe_dump(grid.to_dict())))
        assert again.id == grid.id
        assert again.name == "grid-aspen"
        assert again.description == "HeLa, batch B"
        assert again.quality is GridQuality.GOOD
        assert again.has_completed_task("overview_sem")
        assert again.created_at == grid.created_at

    def test_unknown_quality_reads_as_unassessed(self):
        again = GridRecord.from_dict({"name": "g", "quality": "SPLENDID"})
        assert again.quality is GridQuality.UNASSESSED

    def test_missing_id_gets_one(self):
        again = GridRecord.from_dict({"name": "g"})
        assert again.id

    def test_task_state_is_mutated_not_replaced(self):
        grid = GridRecord(name="g")
        state = grid.task_state
        state.status = AutoLamellaTaskStatus.Failed
        assert grid.is_failure
        assert grid.task_state is state


# ---------------------------------------------------------------------------
# On the experiment
# ---------------------------------------------------------------------------


class TestExperimentGrids:
    def test_new_experiment_has_no_grids(self, tmp_path):
        assert list(_experiment(tmp_path).grids) == []

    def test_add_and_lookup(self, tmp_path):
        exp = _experiment(tmp_path)
        grid = exp.add_grid(GridRecord(name="grid-aspen"))
        assert exp.get_grid_by_name("grid-aspen") is grid
        assert exp.get_grid_by_id(grid.id) is grid
        assert exp.get_grid_by_id(None) is None

    def test_names_are_unique(self, tmp_path):
        exp = _experiment(tmp_path)
        exp.add_grid(GridRecord(name="grid-aspen"))
        try:
            exp.add_grid(GridRecord(name="grid-aspen"))
        except ValueError:
            pass
        else:
            raise AssertionError("a duplicate grid name was accepted")

    def test_persists_and_old_files_load_with_none(self, tmp_path):
        exp = _experiment(tmp_path)
        grid = exp.add_grid(GridRecord(name="grid-aspen"))
        grid.quality = GridQuality.POOR
        data = exp.to_dict()
        again = Experiment.from_dict(yaml.safe_load(yaml.safe_dump(data)))
        assert [g.name for g in again.grids] == ["grid-aspen"]
        assert again.grids[0].quality is GridQuality.POOR
        assert again.grids[0].id == grid.id

        del data["grids"]  # an experiment written before grid records existed
        older = Experiment.from_dict(data)
        assert list(older.grids) == []

    def test_save_and_load_round_trip(self, tmp_path):
        exp = _experiment(tmp_path)
        exp.add_grid(GridRecord(name="grid-birch"))
        (tmp_path / "exp").mkdir()
        exp.save()
        loaded = Experiment.load(tmp_path / "exp" / "experiment.yaml")
        assert [g.name for g in loaded.grids] == ["grid-birch"]


class TestLamellaLink:
    def test_grid_id_round_trips_and_defaults_to_none(self, tmp_path):
        exp = _experiment(tmp_path)
        grid = exp.add_grid(GridRecord(name="grid-aspen"))
        lamella = _lamella(tmp_path, grid_id=grid.id)
        again = Lamella.from_dict(yaml.safe_load(yaml.safe_dump(lamella.to_dict())))
        assert again.grid_id == grid.id
        data = lamella.to_dict()
        del data["grid_id"]
        assert Lamella.from_dict(data).grid_id is None

    def test_grid_to_lamella_is_derived(self, tmp_path):
        exp = _experiment(tmp_path)
        aspen = exp.add_grid(GridRecord(name="grid-aspen"))
        birch = exp.add_grid(GridRecord(name="grid-birch"))
        exp.positions.append(_lamella(tmp_path, "a1", grid_id=aspen.id))
        exp.positions.append(_lamella(tmp_path, "a2", grid_id=aspen.id))
        exp.positions.append(_lamella(tmp_path, "b1", grid_id=birch.id))
        exp.positions.append(_lamella(tmp_path, "legacy"))
        assert [p.name for p in exp.get_lamellae_for_grid(aspen)] == ["a1", "a2"]
        assert exp.get_grid_for_lamella(exp.positions[2]) is birch
        assert exp.get_grid_for_lamella(exp.positions[3]) is None

    def test_removing_a_grid_orphans_its_lamellae(self, tmp_path):
        exp = _experiment(tmp_path)
        aspen = exp.add_grid(GridRecord(name="grid-aspen"))
        exp.positions.append(_lamella(tmp_path, "a1", grid_id=aspen.id))
        removed = exp.remove_grid("grid-aspen")
        assert removed is aspen
        assert list(exp.grids) == []
        assert len(exp.positions) == 1 and exp.positions[0].grid_id is None
        assert exp.remove_grid("grid-aspen") is None


# ---------------------------------------------------------------------------
# Sync from the inventory
# ---------------------------------------------------------------------------


class TestSyncFromInventory:
    def _compustage(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        microscope._stage = _create_sample_stage(microscope)
        microscope._stage.loader = DemoSampleLoader(
            microscope, occupied=(1, 3), names={3: "grid-cedar"}
        )
        return microscope

    def test_creates_a_record_per_present_grid(self, tmp_path):
        microscope = self._compustage()
        exp = _experiment(tmp_path)
        added = exp.sync_grids_from_inventory(microscope._stage)
        assert [g.name for g in added] == ["Grid-01", "grid-cedar"]
        assert [g.name for g in exp.grids] == ["Grid-01", "grid-cedar"]

    def test_is_idempotent_and_keeps_history(self, tmp_path):
        microscope = self._compustage()
        exp = _experiment(tmp_path)
        exp.sync_grids_from_inventory(microscope._stage)
        exp.get_grid_by_name("Grid-01").quality = GridQuality.GOOD
        assert exp.sync_grids_from_inventory(microscope._stage) == []
        assert exp.get_grid_by_name("Grid-01").quality is GridQuality.GOOD

    def test_a_record_exists_before_its_grid_is_in_the_beam(self, tmp_path):
        microscope = self._compustage()
        exp = _experiment(tmp_path)
        exp.sync_grids_from_inventory(microscope._stage)
        assert microscope._stage.loaded_grids == []
        assert exp.get_grid_by_name("grid-cedar") is not None

    def test_fixed_holder_syncs_named_slots(self, tmp_path):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        microscope._stage = _create_sample_stage(microscope)
        microscope._stage.assign_grid(
            "Slot-02", SampleGrid(name="grid-birch"), persist=False
        )
        exp = _experiment(tmp_path)
        exp.sync_grids_from_inventory(microscope._stage)
        assert [g.name for g in exp.grids] == ["grid-birch"]

    def test_a_grid_that_leaves_keeps_its_record(self, tmp_path):
        microscope = self._compustage()
        exp = _experiment(tmp_path)
        exp.sync_grids_from_inventory(microscope._stage)
        microscope._stage.loader.assign_grid(
            "Slot-01", None
        )  # taken out of the magazine
        exp.sync_grids_from_inventory(microscope._stage)
        assert exp.get_grid_by_name("Grid-01") is not None
        present = {e.name for e in microscope._stage.grid_inventory() if e.present}
        assert "Grid-01" not in present
