"""The AutoScript autoloader against a fake of the AutoScript autoloader API.

Every case runs twice: on the old grid loader (``AutoscriptSampleLoader`` in
``fibsem.microscopes.autoscript``) and on what ``ThermoMicroscope`` builds now, the
grid model (``DeviceSampleLoader``) over the ``sample_loader`` device
(``fibsem.devices.drivers.autoscript``), and needs the same answer from both.

The fake mirrors what operator code confirmed on an Arctis: ``get_slots(run_inventory)``
returns ``AutoloaderSlot``-like objects with a 1-based ``id``, a ``state`` in
``{"Unknown", "Occupied", "Empty"}`` and a ``sample_description``; ``load(id)``
blocks; ``unload()`` takes nothing; ``stage`` reports what is on the microscope.
Nothing here has run on hardware -- that is the Arctis bench issue.
"""

from typing import List, Optional

import pytest

from fibsem import utils
from fibsem.devices.drivers import autoscript as autoscript_devices
from fibsem.microscopes._stage import (
    DeviceSampleLoader,
    GridExchangeError,
    SampleGrid,
    _create_sample_stage,
)
from fibsem.microscopes.autoscript import AutoscriptSampleLoader
from fibsem.structures import DeviceEntry

_KIND = "old"


@pytest.fixture(autouse=True, params=["old", "device"])
def loader_kind(request):
    """Run each case on the old grid loader and on the device's grid model."""
    global _KIND
    if request.param == "device" and request.node.cls is TestIsInstalled:
        pytest.skip("the old loader's own installed check")
    _KIND = request.param
    yield request.param
    _KIND = "old"


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

    def __init__(
        self,
        occupied: Optional[dict] = None,
        scanned: bool = True,
        reports_loaded: bool = False,
    ) -> None:
        """``reports_loaded`` is AutoScript 4.14: the home slot of the grid on the
        stage reads ``Loaded``; before that it read ``Empty``."""
        self.is_installed = True
        self.reports_loaded = reports_loaded
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
        # The home slot of a grid on the stage: Loaded from 4.14, Empty before.
        slot.state = "Loaded" if self.reports_loaded else "Empty"
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
    microscope.stage_device.compustage = True
    microscope._stage = _create_sample_stage(microscope)
    microscope.connection = FakeConnection(autoloader)
    if _KIND == "old":
        loader = AutoscriptSampleLoader(parent=microscope)
    else:
        device = autoscript_devices.AutoscriptSampleLoader(microscope).connect()
        loader = DeviceSampleLoader(microscope, device)
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

    def test_get_inventory_reads_what_the_autoloader_knows(self):
        """A read is `get_slots(False)` and nothing else: instant, and only as
        fresh as the autoloader's own last scan -- unscanned slots stay UNKNOWN."""
        from fibsem.microscopes._stage import GridSlotState

        hw = FakeAutoloader(occupied={2: "g"}, scanned=False)
        microscope, loader = _microscope_with(hw)
        assert loader.get_inventory() == []
        assert hw.calls == [("get_slots", False)]
        rows = microscope._stage.grid_inventory()
        assert {r.state for r in rows} == {GridSlotState.UNKNOWN}

    def test_run_inventory_scans_the_magazine(self):
        """A scan is `get_slots(True)`: the slow one, the caller's explicit choice."""
        hw = FakeAutoloader(occupied={2: "g"}, scanned=False)
        _, loader = _microscope_with(hw)
        slots = loader.run_inventory()
        assert hw.calls == [("get_slots", True)]
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

    def test_slots_are_unknown_until_the_hardware_has_answered(self):
        """Before any inventory every magazine slot is UNKNOWN, and nothing is
        present: the UI says "run an inventory" rather than "empty". After a
        scan, a slot the autoloader still could not read stays UNKNOWN."""
        from fibsem.microscopes._stage import GridSlotState

        hw = FakeAutoloader(occupied={1: "a", 2: "b"})
        microscope, loader = _microscope_with(hw)
        rows = microscope._stage.grid_inventory()
        assert {r.state for r in rows} == {GridSlotState.UNKNOWN}
        assert not any(r.present for r in rows)

        hw._slots[4].state = "Unknown"  # one slot the autoloader has not read
        loader.get_inventory()  # a read keeps it that way; a scan would resolve it
        rows = {r.slot_name: r for r in microscope._stage.grid_inventory()}
        assert rows["Slot-01"].state is GridSlotState.OCCUPIED
        assert rows["Slot-03"].state is GridSlotState.EMPTY
        assert rows["Slot-05"].state is GridSlotState.UNKNOWN
        assert not rows["Slot-05"].present

    def test_a_loaded_slot_puts_its_grid_on_the_stage_on_a_fresh_connect(self):
        """AutoScript 4.14 (FIB-952): the home slot of the grid on the stage reads
        LOADED. On a fresh connect, with no memory of any exchange, that grid is
        present in its slot, loaded, and in our working slot, named from the slot
        description. States are matched case-blind, as the enum names arrive."""
        from fibsem.microscopes._stage import GridSlotState

        hw = FakeAutoloader(
            occupied={2: "grid-elm", 4: "grid-birch"}, reports_loaded=True
        )
        hw.load(4)  # loaded from the vendor UI before we connected
        hw._slots[1].state = "OCCUPIED"
        hw._slots[3].state = "LOADED"
        hw.stage.state = "OCCUPIED"
        microscope, loader = _microscope_with(hw)
        loader.get_inventory()
        rows = {r.name: r for r in microscope._stage.grid_inventory() if r.present}
        assert set(rows) == {"grid-elm", "grid-birch"}
        assert rows["grid-birch"].state is GridSlotState.LOADED
        assert rows["grid-elm"].state is GridSlotState.OCCUPIED
        assert [g.name for g in microscope._stage.loaded_grids] == ["grid-birch"]
        assert loader.working_slot.loaded_grid is loader.slots["Slot-04"].loaded_grid
        # Already the loaded grid: no exchange.
        before = len(hw.calls)
        microscope._stage.ensure_loaded("grid-birch")
        assert hw.calls[before:] == []

    def test_the_4_13_home_slot_reads_empty_and_memory_still_keeps_the_grid(self):
        """Before 4.14 the home slot reads Empty while its grid is out; a rescan
        keeps the grid there only because we remember loading it."""
        hw = FakeAutoloader(occupied={4: "grid-birch"})
        microscope, loader = _microscope_with(hw)
        loader.get_inventory()
        microscope._stage.ensure_loaded("grid-birch")
        assert hw._slots[3].state == "Empty"
        loader.get_inventory()
        assert loader.slots["Slot-04"].loaded_grid is not None
        assert [g.name for g in microscope._stage.loaded_grids] == ["grid-birch"]

    def test_the_raw_rows_are_logged_on_every_read(self, caplog):
        """A bench session run from the GUI still gets the hardware's own words
        from the log: slot id, state and description, and the stage."""
        import logging

        hw = FakeAutoloader(occupied={2: "grid-elm"})
        hw.load(2)
        _, loader = _microscope_with(hw)
        # setup_session reconfigures the root logger with force=True, which
        # detaches pytest's capture handler; put it back before the read.
        logging.getLogger().addHandler(caplog.handler)
        caplog.set_level(logging.INFO)
        loader.get_inventory()
        slots = [m for m in caplog.messages if m.startswith("Autoloader slots:")]
        stage = [m for m in caplog.messages if m.startswith("Autoloader stage:")]
        assert slots and "1=Empty" in slots[-1] and "2=Empty 'grid-elm'" in slots[-1]
        assert stage[-1] == "Autoloader stage: Occupied 'grid-elm'"

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

    def test_an_exchange_on_4_14_reads_loaded_then_occupied_again(self):
        from fibsem.microscopes._stage import GridSlotState

        hw = FakeAutoloader(occupied={1: "a", 2: "b"}, reports_loaded=True)
        microscope, loader = _microscope_with(hw)
        loader.get_inventory()
        stage = microscope._stage
        stage.ensure_loaded("a")
        loader.get_inventory()
        rows = {r.name: r.state for r in stage.grid_inventory() if r.present}
        assert rows == {"a": GridSlotState.LOADED, "b": GridSlotState.OCCUPIED}
        stage.ensure_loaded("b")  # unload a, load b
        loader.get_inventory()
        rows = {r.name: r.state for r in stage.grid_inventory() if r.present}
        assert rows == {"a": GridSlotState.OCCUPIED, "b": GridSlotState.LOADED}
        stage.unload()
        loader.get_inventory()
        rows = {r.name: r.state for r in stage.grid_inventory() if r.present}
        assert rows == {"a": GridSlotState.OCCUPIED, "b": GridSlotState.OCCUPIED}

    def test_the_home_slot_survives_a_rescan_while_loaded(self):
        hw = FakeAutoloader(occupied={3: "grid-cedar", 4: "grid-elm"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        microscope._stage.ensure_loaded("grid-cedar")
        loader.run_inventory()  # hardware now reads slot 3 as Empty
        assert loader.slots["Slot-03"].loaded_grid.name == "grid-cedar"
        assert microscope._stage.grid_inventory()[2].loaded is True

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

    def test_a_name_that_does_not_stick_is_an_error(self):
        """Each read hands back fresh slot records, and this one ignores the
        write: the next inventory read would undo the rename, so it is said now."""
        hw = FakeAutoloader(occupied={2: "Grid-02"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()
        real = hw.get_slots

        def fresh_copies(run_inventory):
            return [
                FakeAutoloaderSlot(s.id, s.state, s.sample_description)
                for s in real(run_inventory)
            ]

        hw.get_slots = fresh_copies
        with pytest.raises(GridExchangeError, match="did not stick"):
            microscope._stage.assign_grid("Slot-02", SampleGrid(name="grid-birch"))
        assert hw._slots[1].sample_description == "Grid-02"

    def test_a_refused_write_is_an_error(self):
        hw = FakeAutoloader(occupied={2: "Grid-02"})
        microscope, loader = _microscope_with(hw)
        loader.run_inventory()

        class ReadOnlySlot(FakeAutoloaderSlot):
            def __setattr__(self, key, value):
                if key == "sample_description" and hasattr(self, key):
                    raise PermissionError("sample_description is read-only")
                super().__setattr__(key, value)

        hw._slots[1] = ReadOnlySlot(2, "Occupied", "Grid-02")
        with pytest.raises(GridExchangeError, match="read-only"):
            microscope._stage.assign_grid("Slot-02", SampleGrid(name="grid-birch"))


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


# ---------------------------------------------------------------------------
# ThermoMicroscope builds the device when the autoloader is installed
# ---------------------------------------------------------------------------


def _thermo(autoloader: FakeAutoloader, devices=None):
    """A ThermoMicroscope with just what `_create_grid_loader` reads: the
    configuration, the connection and the devices map."""
    from fibsem.microscopes.autoscript import ThermoMicroscope

    demo, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    system = demo.system
    if devices is not None:
        system.other_devices = [DeviceEntry.from_dict(entry) for entry in devices]
    microscope = ThermoMicroscope.__new__(ThermoMicroscope)
    microscope.system = system
    microscope.connection = FakeConnection(autoloader)
    microscope._devices = {}
    return microscope


class TestThermoBuildsTheDevice:
    @pytest.fixture(autouse=True)
    def _once(self, loader_kind):
        if loader_kind == "old":
            pytest.skip("wiring, not a loader")

    def test_an_installed_autoloader_is_a_sample_loader_device(self):
        microscope = _thermo(FakeAutoloader(occupied={1: "a"}))
        loader = microscope._create_grid_loader()
        assert isinstance(loader, DeviceSampleLoader)
        device = microscope.devices["sample_loader"]
        assert isinstance(device, autoscript_devices.AutoscriptSampleLoader)
        assert loader.device is device
        # nothing is read at connect
        assert microscope.connection.specimen.autoloader.calls == []

    def test_a_rebuilt_sample_stage_keeps_the_device(self):
        microscope = _thermo(FakeAutoloader())
        first = microscope._create_grid_loader().device
        assert microscope._create_grid_loader().device is first

    def test_no_autoloader_no_loader(self):
        hw = FakeAutoloader()
        hw.is_installed = False
        microscope = _thermo(hw)
        assert microscope._create_grid_loader() is None
        assert "sample_loader" not in microscope.devices

    def test_switched_off_in_the_configuration_no_loader(self):
        hw = FakeAutoloader()
        microscope = _thermo(hw, devices=[{"name": "sample_loader", "enabled": False}])
        assert microscope._create_grid_loader() is None
        assert "sample_loader" not in microscope.devices
