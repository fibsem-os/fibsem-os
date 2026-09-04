"""The loader magazine, the Demo autoloader, and ``Stage.grid_inventory()``.

Two hardware shapes, one query. On a compustage the inventory is the loader's
magazine and "loaded" is whichever grid sits in the holder's working slot; on a
fixed holder the inventory is the holder itself and every present grid is in the
beam. Nothing here is cached: every answer is re-derived from the slots.
"""

import pytest

from fibsem import utils
from fibsem.microscopes._stage import (
    DemoSampleLoader,
    GridExchangeError,
    GridSlot,
    SampleGrid,
    SampleGridLoader,
    _create_sample_stage,
)
from fibsem.structures import FibsemStagePosition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compustage_demo():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)
    return microscope


def _fixed_demo():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    return microscope


def _with_magazine(microscope, occupied=(1, 2, 5), names=None) -> DemoSampleLoader:
    loader = DemoSampleLoader(microscope, capacity=12, occupied=occupied, names=names)
    microscope._stage.loader = loader
    return loader


def _entry(microscope, slot_name):
    return next(
        e for e in microscope._stage.grid_inventory() if e.slot_name == slot_name
    )


# ---------------------------------------------------------------------------
# GridSlot: magazine slots have no position
# ---------------------------------------------------------------------------


class TestGridSlotPosition:
    def test_position_is_optional(self):
        slot = GridSlot(name="Slot-01", index=0)
        assert slot.position is None

    def test_roundtrip_without_position(self):
        slot = GridSlot(name="Slot-03", index=2, loaded_grid=SampleGrid(name="g"))
        again = GridSlot.from_dict(slot.to_dict())
        assert again.position is None
        assert again.loaded_grid.name == "g"

    def test_roundtrip_with_position_still_names_it(self):
        slot = GridSlot(
            name="Slot-01", index=0, position=FibsemStagePosition(x=1e-3, y=0, z=0)
        )
        again = GridSlot.from_dict(slot.to_dict())
        assert again.position.name == "Slot-01"


# ---------------------------------------------------------------------------
# The magazine
# ---------------------------------------------------------------------------


class TestMagazine:
    def test_capacity_slots_without_positions(self):
        microscope = _compustage_demo()
        loader = SampleGridLoader(microscope, capacity=4)
        assert list(loader.slots) == ["Slot-01", "Slot-02", "Slot-03", "Slot-04"]
        assert all(s.position is None for s in loader.slots.values())
        assert loader.loaded_magazine_slots == []

    def test_demo_occupancy_and_default_names(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope, occupied=(1, 2, 5), names={2: "grid-birch"})
        names = {s.name: s.loaded_grid.name for s in loader.loaded_magazine_slots}
        assert names == {
            "Slot-01": "Grid-01",
            "Slot-02": "grid-birch",
            "Slot-05": "Grid-05",
        }
        assert loader.slots["Slot-03"].loaded_grid is None

    def test_demo_names_accept_string_keys_as_yaml_gives_them(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope, occupied=(7,), names={"7": "grid-fir"})
        assert loader.slots["Slot-07"].loaded_grid.name == "grid-fir"

    def test_demo_occupied_outside_capacity_refused(self):
        microscope = _compustage_demo()
        with pytest.raises(ValueError):
            DemoSampleLoader(microscope, capacity=4, occupied=(5,))

    def test_run_inventory_reports_occupied_slots(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        assert [s.name for s in loader.run_inventory()] == [
            "Slot-01",
            "Slot-02",
            "Slot-05",
        ]

    def test_assign_and_find_grid(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.assign_grid("Slot-03", SampleGrid(name="grid-cedar"))
        assert loader.find_grid("grid-cedar") is loader.slots["Slot-03"]
        loader.assign_grid("Slot-03", None)
        assert loader.find_grid("grid-cedar") is None

    def test_assign_to_unknown_slot_raises(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        with pytest.raises(GridExchangeError):
            loader.assign_grid("Slot-99", SampleGrid(name="x"))


# ---------------------------------------------------------------------------
# Exchange: magazine <-> working slot
# ---------------------------------------------------------------------------


class TestExchange:
    def test_load_puts_the_same_grid_object_in_the_working_slot(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        grid = loader.load_grid("Slot-02")
        working = microscope._stage.holder.slots["Slot-01"]
        assert working.loaded_grid is grid
        # the magazine slot is the grid's home; it keeps the grid while it is loaded
        assert loader.slots["Slot-02"].loaded_grid is grid

    def test_load_another_grid_exchanges(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.load_grid("Slot-01")
        loader.load_grid("Slot-05")
        working = microscope._stage.holder.slots["Slot-01"]
        assert working.loaded_grid.name == "Grid-05"
        assert loader.slots["Slot-01"].loaded_grid.name == "Grid-01"  # still present

    def test_load_same_grid_is_a_noop(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.load_grid("Slot-02")
        loader.fail_next_exchange = True  # would raise if an exchange happened
        loader.load_grid("Slot-02")
        assert loader.fail_next_exchange is True

    def test_unload_clears_the_working_slot(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.load_grid("Slot-02")
        loader.unload_grid()
        assert microscope._stage.holder.slots["Slot-01"].loaded_grid is None
        assert loader.slots["Slot-02"].loaded_grid is not None

    def test_unload_when_empty_is_a_noop(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.fail_next_exchange = True
        loader.unload_grid()
        assert loader.fail_next_exchange is True

    def test_load_empty_slot_raises(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        with pytest.raises(GridExchangeError):
            loader.load_grid("Slot-03")

    def test_failed_exchange_raises_and_leaves_state_untouched(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.load_grid("Slot-01")
        loader.fail_next_exchange = True
        with pytest.raises(GridExchangeError):
            loader.load_grid("Slot-02")
        working = microscope._stage.holder.slots["Slot-01"]
        # the unload half of the exchange failed, so Grid-01 is still loaded
        assert working.loaded_grid.name == "Grid-01"
        assert loader.fail_next_exchange is False


# ---------------------------------------------------------------------------
# Stage.grid_inventory()
# ---------------------------------------------------------------------------


class TestInventoryWithLoader:
    def test_one_row_per_magazine_slot(self):
        microscope = _compustage_demo()
        _with_magazine(microscope)
        rows = microscope._stage.grid_inventory()
        assert len(rows) == 12
        assert {r.source for r in rows} == {"magazine"}
        assert [r.slot_name for r in rows[:3]] == ["Slot-01", "Slot-02", "Slot-03"]

    def test_present_and_in_beam_are_separate(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.load_grid("Slot-02")
        one, two, three = (_entry(microscope, f"Slot-0{i}") for i in (1, 2, 3))
        assert (one.present, one.loaded) == (True, False)
        assert (two.present, two.loaded) == (True, True)
        assert (three.present, three.loaded, three.name) == (False, False, None)

    def test_in_beam_follows_the_holder_not_a_cache(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.load_grid("Slot-02")
        # an operator hand-loads something else: the inventory must say so
        microscope._stage.holder.slots["Slot-01"].loaded_grid = loader.slots[
            "Slot-05"
        ].loaded_grid
        assert _entry(microscope, "Slot-02").loaded is False
        assert _entry(microscope, "Slot-05").loaded is True


class TestInventoryOnFixedHolder:
    def test_rows_are_holder_slots_and_present_means_in_beam(self):
        microscope = _fixed_demo()
        holder = microscope._stage.holder
        holder.slots["Slot-01"].loaded_grid = SampleGrid(name="grid-aspen")
        rows = microscope._stage.grid_inventory()
        assert len(rows) == len(holder.slots)
        assert {r.source for r in rows} == {"holder"}
        first = _entry(microscope, "Slot-01")
        assert (first.name, first.present, first.loaded) == ("grid-aspen", True, True)
        second = _entry(microscope, "Slot-02")
        assert (second.name, second.present, second.loaded) == (None, False, False)


# ---------------------------------------------------------------------------
# _create_sample_stage wires the right loader
# ---------------------------------------------------------------------------


class TestCreateSampleStage:
    def test_compustage_demo_builds_the_loader_from_sim_config(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = True
        # a copy, not a mutation: `system.sim` may be shared with other sessions
        microscope.system.sim = dict(
            microscope.system.sim,
            loader={"capacity": 6, "occupied": [1, 4], "names": {4: "grid-elm"}},
        )
        stage = _create_sample_stage(microscope)
        assert isinstance(stage.loader, DemoSampleLoader)
        assert stage.loader.capacity == 6
        assert [s.loaded_grid.name for s in stage.loader.loaded_magazine_slots] == [
            "Grid-01",
            "grid-elm",
        ]

    def test_compustage_demo_without_loader_block_has_an_empty_magazine(self):
        microscope = _compustage_demo()
        assert isinstance(microscope._stage.loader, DemoSampleLoader)
        assert microscope._stage.loader.loaded_magazine_slots == []

    def test_fixed_holder_has_no_loader(self):
        microscope = _fixed_demo()
        assert microscope._stage.loader is None


# ---------------------------------------------------------------------------
# Stage.ensure_loaded / unload: the one "make reachable" primitive
# ---------------------------------------------------------------------------


class TestEnsureLoadedWithLoader:
    def test_exchange_when_not_in_beam(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope, names={2: "grid-birch"})
        slot = microscope._stage.ensure_loaded("grid-birch")
        assert slot is loader.working_slot
        assert slot.loaded_grid.name == "grid-birch"
        assert [g.name for g in microscope._stage.loaded_grids] == ["grid-birch"]

    def test_noop_when_already_in_beam(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope, names={2: "grid-birch"})
        microscope._stage.ensure_loaded("grid-birch")
        loader.fail_next_exchange = True  # would raise if anything moved
        microscope._stage.ensure_loaded("grid-birch")
        assert loader.fail_next_exchange is True

    def test_unknown_grid_raises(self):
        microscope = _compustage_demo()
        _with_magazine(microscope)
        with pytest.raises(GridExchangeError, match="not in the magazine"):
            microscope._stage.ensure_loaded("grid-nowhere")

    def test_hardware_failure_propagates(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.fail_next_exchange = True
        with pytest.raises(GridExchangeError):
            microscope._stage.ensure_loaded("Grid-01")
        assert microscope._stage.loaded_grids == []

    def test_does_not_move_the_stage(self):
        microscope = _compustage_demo()
        _with_magazine(microscope)
        before = microscope.get_stage_position()
        microscope._stage.ensure_loaded("Grid-01")
        assert microscope.get_stage_position().is_close2(before, tol=1e-9)

    def test_unload_retracts(self):
        microscope = _compustage_demo()
        _with_magazine(microscope)
        microscope._stage.ensure_loaded("Grid-01")
        microscope._stage.unload()
        assert microscope._stage.loaded_grids == []


class TestEnsureLoadedOnFixedHolder:
    def test_present_grid_is_already_reachable(self):
        microscope = _fixed_demo()
        holder = microscope._stage.holder
        holder.slots["Slot-02"].loaded_grid = SampleGrid(name="grid-aspen")
        holder.slots["Slot-02"].position = FibsemStagePosition(
            name="Slot-02", x=3e-3, y=0.0, z=4e-3, r=0.0, t=0.0
        )
        before = microscope.get_stage_position()
        slot = microscope._stage.ensure_loaded("grid-aspen")
        assert slot is holder.slots["Slot-02"]
        assert microscope.get_stage_position().is_close2(before, tol=1e-9)

    def test_present_but_uncalibrated_slot_is_refused(self):
        microscope = _fixed_demo()
        microscope._stage.holder.slots["Slot-01"].loaded_grid = SampleGrid(name="g")
        with pytest.raises(GridExchangeError, match="Calibrate slot positions"):
            microscope._stage.ensure_loaded("g")

    def test_absent_grid_raises_with_a_manual_instruction(self):
        microscope = _fixed_demo()
        with pytest.raises(GridExchangeError, match="Place it in the holder"):
            microscope._stage.ensure_loaded("grid-aspen")

    def test_unload_is_a_noop(self):
        microscope = _fixed_demo()
        microscope._stage.holder.slots["Slot-01"].loaded_grid = SampleGrid(name="g")
        microscope._stage.unload()
        assert microscope._stage.holder.slots["Slot-01"].loaded_grid is not None

    def test_move_to_grid_resolves_the_slot(self):
        microscope = _fixed_demo()
        holder = microscope._stage.holder
        holder.slots["Slot-02"].loaded_grid = SampleGrid(name="grid-aspen")
        holder.slots["Slot-02"].position = FibsemStagePosition(
            name="Slot-02", x=3e-3, y=-2e-3, z=4e-3, r=0.0, t=0.0
        )
        microscope._stage.move_to_grid("grid-aspen")
        pos = microscope.get_stage_position()
        assert abs(pos.x - 3e-3) < 1e-9 and abs(pos.y + 2e-3) < 1e-9

    def test_move_to_grid_unknown_raises(self):
        microscope = _fixed_demo()
        with pytest.raises(ValueError):
            microscope._stage.move_to_grid("grid-nowhere")


# ---------------------------------------------------------------------------
# Stage.assign_grid / run_inventory: naming on either shape
# ---------------------------------------------------------------------------


class TestAssignGrid:
    def test_with_loader_names_the_magazine_slot(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        microscope._stage.assign_grid("Slot-03", SampleGrid(name="grid-cedar"))
        assert loader.slots["Slot-03"].loaded_grid.name == "grid-cedar"
        assert _entry(microscope, "Slot-03").present is True

    def test_on_fixed_holder_names_the_slot_and_saves_the_occupancy(
        self, tmp_path, monkeypatch
    ):
        import fibsem.microscopes._stage as stage_module

        path = tmp_path / "occupancy.yaml"
        monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_OCCUPANCY_PATH", str(path))
        microscope = _fixed_demo()
        microscope._stage.assign_grid("Slot-02", SampleGrid(name="grid-birch"))
        assert _entry(microscope, "Slot-02").name == "grid-birch"
        # the next session's holder picks it up; the calibration file is untouched
        again = _fixed_demo()
        assert again._stage.holder.slots["Slot-02"].loaded_grid.name == "grid-birch"

    def test_on_fixed_holder_can_skip_persisting(self, tmp_path, monkeypatch):
        import fibsem.microscopes._stage as stage_module

        path = tmp_path / "occupancy.yaml"
        monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_OCCUPANCY_PATH", str(path))
        microscope = _fixed_demo()
        microscope._stage.assign_grid("Slot-01", SampleGrid(name="g"), persist=False)
        assert not path.exists()

    def test_on_fixed_holder_clearing_persists_too(self, tmp_path, monkeypatch):
        import fibsem.microscopes._stage as stage_module

        path = tmp_path / "occupancy.yaml"
        monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_OCCUPANCY_PATH", str(path))
        microscope = _fixed_demo()
        microscope._stage.assign_grid("Slot-01", SampleGrid(name="g"))
        microscope._stage.assign_grid("Slot-01", None)
        assert _fixed_demo()._stage.holder.slots["Slot-01"].loaded_grid is None

    def test_unknown_holder_slot_raises(self):
        microscope = _fixed_demo()
        with pytest.raises(ValueError):
            microscope._stage.assign_grid(
                "Slot-99", SampleGrid(name="g"), persist=False
            )


class TestRunInventory:
    def test_with_loader_scans_and_returns_rows(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        scanned = []
        loader._scan_magazine = lambda: scanned.append(True)  # type: ignore[assignment]
        rows = microscope._stage.run_inventory()
        assert scanned == [True]
        assert [r.name for r in rows if r.present] == ["Grid-01", "Grid-02", "Grid-05"]

    def test_get_inventory_reads_without_a_scan(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        calls = []
        loader._read_magazine = lambda: calls.append("read")  # type: ignore[assignment]
        loader._scan_magazine = lambda: calls.append("scan")  # type: ignore[assignment]
        rows = microscope._stage.get_inventory()
        assert calls == ["read"]
        assert [r.name for r in rows if r.present] == ["Grid-01", "Grid-02", "Grid-05"]

    def test_on_fixed_holder_is_a_refresh(self):
        microscope = _fixed_demo()
        microscope._stage.holder.slots["Slot-01"].loaded_grid = SampleGrid(name="g")
        rows = microscope._stage.run_inventory()
        assert [r.name for r in rows if r.present] == ["g"]


# ---------------------------------------------------------------------------
# current_slot / current_grid before the position cache is filled
# ---------------------------------------------------------------------------


def test_current_grid_is_none_before_the_first_position_read():
    """A fresh connection has no cached stage position. Asking which grid the
    stage is at then means "not known", not an AttributeError."""
    microscope = _compustage_demo()
    _with_magazine(microscope)
    stage = microscope._stage
    stage.ensure_loaded("Grid-01")
    microscope._stage_position = None

    assert stage.current_slot is None
    assert stage.current_grid is None

    stage.position  # the first read fills the cache
    stage.move_to_slot("Slot-01")
    assert stage.current_grid is not None
    assert stage.current_grid.name == "Grid-01"


# ---------------------------------------------------------------------------
# The slot state: where the grid is, as one word
# ---------------------------------------------------------------------------


class TestSlotState:
    def test_a_magazine_reads_occupied_loaded_and_empty(self):
        from fibsem.microscopes._stage import GridSlotState

        microscope = _compustage_demo()
        _with_magazine(microscope, occupied=(1, 2))
        stage = microscope._stage
        stage.ensure_loaded("Grid-02")
        rows = {e.slot_name: e for e in stage.grid_inventory()}
        assert rows["Slot-01"].state is GridSlotState.OCCUPIED
        assert rows["Slot-02"].state is GridSlotState.LOADED
        assert rows["Slot-03"].state is GridSlotState.EMPTY
        # present and loaded are read off the state
        assert rows["Slot-01"].present and not rows["Slot-01"].loaded
        assert rows["Slot-02"].present and rows["Slot-02"].loaded
        assert not rows["Slot-03"].present

    def test_a_fixed_holder_slot_with_a_grid_is_loaded_outright(self):
        """Its grid is already in the holder: reaching it is a stage move."""
        from fibsem.microscopes._stage import GridSlotState

        microscope = _fixed_demo()
        stage = microscope._stage
        stage.assign_grid("Slot-01", SampleGrid(name="grid-ash"), persist=False)
        rows = {e.slot_name: e for e in stage.grid_inventory()}
        assert rows["Slot-01"].state is GridSlotState.LOADED
        assert rows["Slot-02"].state is GridSlotState.EMPTY
        assert GridSlotState.UNKNOWN not in {e.state for e in rows.values()}

    def test_the_demo_magazine_is_known_from_the_start(self):
        """An in-memory magazine has nothing to scan, so its slots are never
        UNKNOWN; the tests and the simulator rely on reading it before any
        inventory has run."""
        from fibsem.microscopes._stage import GridSlotState

        microscope = _compustage_demo()
        _with_magazine(microscope, occupied=(1,))
        states = {e.state for e in microscope._stage.grid_inventory()}
        assert states == {GridSlotState.OCCUPIED, GridSlotState.EMPTY}
