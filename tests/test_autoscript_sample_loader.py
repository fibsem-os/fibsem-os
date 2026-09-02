"""``AutoscriptSampleLoader`` against a fake of the AutoScript autoloader API.

The fake mirrors what operator code confirmed on an Arctis: ``get_slots(run_inventory)``
returns ``AutoloaderSlot``-like objects with a 1-based ``id``, a ``state`` in
``{"Unknown", "Occupied", "Empty"}`` and a ``sample_description``; ``load(id)``
blocks; ``unload()`` takes nothing; ``stage`` reports what is on the microscope.
Nothing here has run on hardware -- that is the Arctis bench issue.
"""

from typing import List, Optional

import pytest

from fibsem import utils
from fibsem.microscopes._stage import (
    GridExchangeError,
    SampleGrid,
    _create_sample_stage,
)
from fibsem.microscopes.autoscript import AutoscriptSampleLoader

# ---------------------------------------------------------------------------
# A fake autoloader, shaped like the vendor API
# ---------------------------------------------------------------------------


class FakeAutoloaderSlot:
    def __init__(self, id: int, state: str = "Empty", sample_description: str = ""):
        self.id = id
        self.state = state
        self.sample_description = sample_description


class FakeAutoloaderStage:
    def __init__(self) -> None:
        self.sample_description = ""
        self.state = "Empty"


class FakeAutoloader:
    """Twelve slots; loading moves a grid onto ``stage`` and empties its slot."""

    def __init__(self, occupied: Optional[dict] = None, scanned: bool = True) -> None:
        self.is_installed = True
        self.stage = FakeAutoloaderStage()
        self._slots: List[FakeAutoloaderSlot] = [
            FakeAutoloaderSlot(i, "Empty" if scanned else "Unknown")
            for i in range(1, 13)
        ]
        for number, name in (occupied or {}).items():
            slot = self._slots[number - 1]
            slot.state = "Occupied" if scanned else "Unknown"
            slot.sample_description = name
        self._pending_occupied = dict(occupied or {}) if not scanned else {}
        self.calls: List[tuple] = []
        self.fail_load = False
        self._loaded_from: Optional[int] = None

    def get_slots(self, run_inventory: bool) -> List[FakeAutoloaderSlot]:
        self.calls.append(("get_slots", run_inventory))
        if run_inventory:
            for slot in self._slots:
                if slot.state == "Unknown":
                    slot.state = (
                        "Occupied" if slot.id in self._pending_occupied else "Empty"
                    )
        return list(self._slots)

    def load(self, grid_id: int) -> None:
        self.calls.append(("load", grid_id))
        if self.fail_load:
            raise RuntimeError("Autoloader: gripper fault")
        slot = self._slots[grid_id - 1]
        self.stage.sample_description = slot.sample_description
        self.stage.state = "Occupied"
        slot.state = "Empty"  # what the hardware reports for a grid on the stage
        self._loaded_from = grid_id

    def unload(self) -> None:
        self.calls.append(("unload",))
        if self._loaded_from is not None:
            self._slots[self._loaded_from - 1].state = "Occupied"
        self.stage.sample_description = ""
        self.stage.state = "Empty"
        self._loaded_from = None


class FakeSpecimen:
    def __init__(self, autoloader: FakeAutoloader) -> None:
        self.autoloader = autoloader


class FakeConnection:
    def __init__(self, autoloader: FakeAutoloader) -> None:
        self.specimen = FakeSpecimen(autoloader)


def _microscope_with(autoloader: FakeAutoloader):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)
    microscope.connection = FakeConnection(autoloader)
    loader = AutoscriptSampleLoader(parent=microscope)
    microscope._stage.loader = loader
    return microscope, loader


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


class TestInventory:
    def test_mirrors_occupied_slots_and_names(self):
        hw = FakeAutoloader(occupied={1: "grid-cedar", 3: "", 5: "grid-elm"})
        microscope, loader = _microscope_with(hw)
        slots = loader.run_inventory()
        assert [s.name for s in slots] == ["Slot-01", "Slot-03", "Slot-05"]
        names = {s.name: s.loaded_grid.name for s in slots}
        # an occupied slot with a blank description gets the default name
        assert names == {
            "Slot-01": "grid-cedar",
            "Slot-03": "Grid-03",
            "Slot-05": "grid-elm",
        }
        assert loader.slots["Slot-02"].loaded_grid is None

    def test_uses_last_known_states_unless_all_unknown(self):
        hw = FakeAutoloader(occupied={2: "g"})
        _, loader = _microscope_with(hw)
        loader.run_inventory()
        assert hw.calls == [("get_slots", False)]

    def test_forces_a_scan_when_nothing_is_known(self):
        hw = FakeAutoloader(occupied={2: "g"}, scanned=False)
        _, loader = _microscope_with(hw)
        slots = loader.run_inventory()
        assert hw.calls == [("get_slots", False), ("get_slots", True)]
        assert [s.name for s in slots] == ["Slot-02"]

    def test_capacity_follows_the_hardware(self):
        hw = FakeAutoloader()
        hw._slots = hw._slots[:6]
        _, loader = _microscope_with(hw)
        loader.run_inventory()
        assert loader.capacity == 6
        assert list(loader.slots) == [f"Slot-{i:02d}" for i in range(1, 7)]

    def test_a_grid_already_on_the_stage_is_reported_in_beam(self):
        hw = FakeAutoloader(occupied={4: "grid-birch"})
        hw.load(4)  # someone loaded it from the vendor UI before we connected
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        assert [g.name for g in microscope._stage.loaded_grids] == ["grid-birch"]

    def test_inventory_rows_come_from_the_magazine(self):
        hw = FakeAutoloader(occupied={1: "a", 2: "b"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        rows = microscope._stage.grid_inventory()
        assert [r.name for r in rows if r.present] == ["a", "b"]
        assert all(r.source == "magazine" for r in rows)


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------


class TestExchange:
    def test_load_calls_the_hardware_by_slot_id_and_fills_the_working_slot(self):
        hw = FakeAutoloader(occupied={3: "grid-cedar"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        microscope._stage.ensure_loaded("grid-cedar")
        assert ("load", 3) in hw.calls
        assert microscope._stage.loaded_grids[0].name == "grid-cedar"
        # the magazine slot stays the grid's home, whatever the hardware reads
        assert loader.slots["Slot-03"].loaded_grid.name == "grid-cedar"

    def test_the_home_slot_survives_a_rescan_while_loaded(self):
        hw = FakeAutoloader(occupied={3: "grid-cedar", 4: "grid-elm"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        microscope._stage.ensure_loaded("grid-cedar")
        loader.run_inventory()  # hardware now reads slot 3 as Empty
        assert loader.slots["Slot-03"].loaded_grid.name == "grid-cedar"
        assert microscope._stage.grid_inventory()[2].in_beam is True

    def test_exchange_unloads_then_loads(self):
        hw = FakeAutoloader(occupied={1: "a", 2: "b"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        microscope._stage.ensure_loaded("a")
        microscope._stage.ensure_loaded("b")
        assert hw.calls[-2:] == [("unload",), ("load", 2)]
        assert microscope._stage.loaded_grids[0].name == "b"

    def test_unload_calls_the_hardware(self):
        hw = FakeAutoloader(occupied={1: "a"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        microscope._stage.ensure_loaded("a")
        microscope._stage.unload()
        assert hw.calls[-1] == ("unload",)
        assert microscope._stage.loaded_grids == []

    def test_hardware_failure_becomes_a_grid_exchange_error(self):
        hw = FakeAutoloader(occupied={1: "a"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        hw.fail_load = True
        with pytest.raises(GridExchangeError, match="gripper fault"):
            microscope._stage.ensure_loaded("a")
        assert microscope._stage.loaded_grids == []


# ---------------------------------------------------------------------------
# Naming writes back to the slot description
# ---------------------------------------------------------------------------


class TestNaming:
    def test_assign_grid_writes_the_slot_description(self):
        hw = FakeAutoloader(occupied={2: "Grid-02"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        microscope._stage.assign_grid("Slot-02", SampleGrid(name="grid-birch"))
        assert hw._slots[1].sample_description == "grid-birch"
        assert loader.find_grid("grid-birch") is loader.slots["Slot-02"]

    def test_clearing_a_slot_clears_the_description(self):
        hw = FakeAutoloader(occupied={2: "grid-birch"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        microscope._stage.assign_grid("Slot-02", None)
        assert hw._slots[1].sample_description == ""


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestIsInstalled:
    def test_reports_the_hardware_flag(self):
        hw = FakeAutoloader()
        _, loader = _microscope_with(hw)
        assert loader.is_installed is True
        hw.is_installed = False
        assert loader.is_installed is False

    def test_absent_device_reads_as_not_installed(self):
        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.connection = FakeConnection(FakeAutoloader())
        del microscope.connection.specimen.autoloader
        assert AutoscriptSampleLoader(parent=microscope).is_installed is False
