"""The loader magazine, the Demo autoloader, and ``Stage.grid_inventory()``.

Two hardware shapes, one query. On a compustage the inventory is the loader's
magazine and "in beam" is whichever grid sits in the holder's working slot; on a
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
    for slot in microscope._stage.holder.slots.values():
        slot.loaded_grid = None
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
        # the magazine slot is the grid's home; it keeps the grid while it is in the beam
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
        # the unload half of the exchange failed, so Grid-01 is still in the beam
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
        assert (one.present, one.in_beam) == (True, False)
        assert (two.present, two.in_beam) == (True, True)
        assert (three.present, three.in_beam, three.name) == (False, False, None)

    def test_in_beam_follows_the_holder_not_a_cache(self):
        microscope = _compustage_demo()
        loader = _with_magazine(microscope)
        loader.load_grid("Slot-02")
        # an operator hand-loads something else: the inventory must say so
        microscope._stage.holder.slots["Slot-01"].loaded_grid = loader.slots[
            "Slot-05"
        ].loaded_grid
        assert _entry(microscope, "Slot-02").in_beam is False
        assert _entry(microscope, "Slot-05").in_beam is True


class TestInventoryOnFixedHolder:
    def test_rows_are_holder_slots_and_present_means_in_beam(self):
        microscope = _fixed_demo()
        holder = microscope._stage.holder
        holder.slots["Slot-01"].loaded_grid = SampleGrid(name="grid-aspen")
        rows = microscope._stage.grid_inventory()
        assert len(rows) == len(holder.slots)
        assert {r.source for r in rows} == {"holder"}
        first = _entry(microscope, "Slot-01")
        assert (first.name, first.present, first.in_beam) == ("grid-aspen", True, True)
        second = _entry(microscope, "Slot-02")
        assert (second.name, second.present, second.in_beam) == (None, False, False)


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
