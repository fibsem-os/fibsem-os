"""The SampleLoader device on the Demo driver, and the grid model over it.

The device reports numbered magazine slots, their states and descriptions, and does
the exchanges; ``DeviceSampleLoader`` turns that into grids. The parity tests run
one sequence of exchanges through the in-memory loader the simulator used before
(``_stage.DemoSampleLoader``) and through the device, and compare what the grid
model reports after each step.
"""

import threading

import pytest

from fibsem import utils
from fibsem.devices.core import Resources
from fibsem.devices.drivers.demo import DemoSampleLoader as DemoSampleLoaderDevice
from fibsem.devices.drivers.demo import build_demo_sample_loader
from fibsem.devices.sample_loader import (
    GridExchangeError,
    Magazine,
    MagazineSlot,
    MagazineSlotState,
    StageSample,
)
from fibsem.devices.stage import STAGE_RESOURCE
from fibsem.devices.wire import from_wire, to_wire
from fibsem.microscopes._stage import (
    DemoSampleLoader,
    DeviceSampleLoader,
    SampleGrid,
    _create_sample_stage,
)
from fibsem.microscopes.registry import BuildContext
from fibsem.structures import DeviceEntry


def _device(**kwargs) -> DemoSampleLoaderDevice:
    return DemoSampleLoaderDevice(**kwargs).connect()


def _states(device) -> list:
    return [s.state for s in device.magazine.get_value().slots]


# ---------------------------------------------------------------------------
# The device
# ---------------------------------------------------------------------------


class TestDevice:
    def test_reads_numbered_slots_with_states_and_descriptions(self):
        device = _device(capacity=4, occupied=(1, 3), names={3: "grid-elm"})
        assert device.capacity.get_value() == 4
        assert device.magazine.get_value() == Magazine(
            (
                MagazineSlot(1, MagazineSlotState.OCCUPIED, ""),
                MagazineSlot(2, MagazineSlotState.EMPTY),
                MagazineSlot(3, MagazineSlotState.OCCUPIED, "grid-elm"),
                MagazineSlot(4, MagazineSlotState.EMPTY),
            )
        )
        assert device.on_stage.get_value() == StageSample(present=False)

    def test_every_parameter_is_read_only(self):
        device = _device()
        assert sorted(device.parameters) == [
            "busy",
            "capacity",
            "exchange_time",
            "magazine",
            "on_stage",
        ]
        assert not any(p.settable for p in device.parameters.values())

    def test_load_puts_the_slots_grid_on_the_stage_and_unload_takes_it_back(self):
        device = _device(capacity=3, occupied=(2,), names={2: "grid-oak"})
        device.load(2)
        assert device.magazine.get_value().slot(2).state is MagazineSlotState.LOADED
        assert device.on_stage.get_value() == StageSample(True, "grid-oak")
        device.unload()
        assert device.magazine.get_value().slot(2).state is MagazineSlotState.OCCUPIED
        assert device.on_stage.get_value() == StageSample(present=False)

    def test_load_and_unload_read_nothing_back(self, monkeypatch):
        """AutoScript before 4.14 reads the home slot of the grid on the stage as
        empty, so a read straight after an exchange would lose that grid."""
        device = DemoSampleLoaderDevice(capacity=3, occupied=(2,))
        reads = []
        for name in ("read_magazine", "read_on_stage"):
            real = getattr(device, name)
            monkeypatch.setattr(
                device,
                name,
                lambda real=real, name=name: (reads.append(name), real())[1],
            )
        device.connect()
        device.load(2)
        device.unload()
        assert reads == []
        device.scan()  # a scan does read back
        assert sorted(reads) == ["read_magazine", "read_on_stage"]

    def test_loading_an_empty_slot_or_onto_an_occupied_stage_is_refused(self):
        device = _device(capacity=3, occupied=(1, 2))
        with pytest.raises(GridExchangeError, match="holds no grid"):
            device.load(3)
        device.load(1)
        with pytest.raises(GridExchangeError, match="unload it first"):
            device.load(2)

    def test_a_failed_exchange_changes_nothing(self):
        device = _device(capacity=2, occupied=(1,))
        device.fail_next_exchange = True
        with pytest.raises(GridExchangeError):
            device.load(1)
        assert device.on_stage.get_value().present is False
        assert device.busy.get_value() is False
        device.load(1)  # the failure is one-shot

    def test_an_unscanned_magazine_reads_unknown_until_a_scan(self):
        device = _device(capacity=3, occupied=(2,), start_unscanned=True)
        assert set(_states(device)) == {MagazineSlotState.UNKNOWN}
        assert set(_states(device)) == {
            MagazineSlotState.UNKNOWN
        }  # a read is not a scan
        found = device.scan()
        assert [s.state for s in found.slots] == [
            MagazineSlotState.EMPTY,
            MagazineSlotState.OCCUPIED,
            MagazineSlotState.EMPTY,
        ]

    def test_set_description_writes_and_reads_back(self):
        device = _device(capacity=2, occupied=(1,))
        assert device.set_description(1, "grid-ash") == MagazineSlot(
            1, MagazineSlotState.OCCUPIED, "grid-ash"
        )
        with pytest.raises(GridExchangeError, match="no slot 5"):
            device.set_description(5, "nowhere")

    def test_a_description_that_does_not_stick_is_an_error(self, monkeypatch):
        device = _device(capacity=2, occupied=(1,))
        monkeypatch.setattr(device, "_set_description", lambda slot, text: None)
        with pytest.raises(GridExchangeError, match="did not stick"):
            device.set_description(1, "grid-ash")

    def test_exchange_time_is_an_unload_and_a_load(self):
        assert _device(exchange_delay=5.0).exchange_time.get_value() == 10.0

    def test_busy_while_a_command_runs_and_signalled(self):
        device = _device(capacity=2, occupied=(1,))
        seen_inside = []
        load = device._load
        device._load = lambda slot: (seen_inside.append(device.busy.cached), load(slot))
        changes = []
        device.busy.changed.connect(changes.append)
        device.load(1)
        assert seen_inside == [True]
        assert device.busy.cached is False
        assert changes == [True, False]

    def test_load_and_unload_hold_the_stage_scan_does_not(self):
        resources = Resources(
            groups={"sample_loader": "sample_loader", STAGE_RESOURCE: STAGE_RESOURCE}
        )
        device = _device(capacity=2, occupied=(1,), resources=resources)

        def stage_free() -> bool:
            result = []

            def probe():
                lock = resources.lock(STAGE_RESOURCE)
                got = lock.acquire(blocking=False)
                if got:
                    lock.release()
                result.append(got)

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join()
            return result[0]

        observed = {}
        for name in ("_load", "_unload", "_scan"):
            original = getattr(device, name)

            def hook(*args, _name=name, _original=original):
                observed[_name] = stage_free()
                return _original(*args)

            setattr(device, name, hook)
        device.load(1)
        device.unload()
        device.scan()
        assert observed == {"_load": False, "_unload": False, "_scan": True}

    def test_the_magazine_and_stage_report_cross_the_wire(self):
        magazine = Magazine(
            (
                MagazineSlot(1, MagazineSlotState.LOADED, "grid-oak"),
                MagazineSlot(2, MagazineSlotState.UNKNOWN),
            )
        )
        assert from_wire(Magazine, to_wire(magazine)) == magazine
        sample = StageSample(True, "grid-oak")
        assert from_wire(StageSample, to_wire(sample)) == sample

    def test_vendor_state_names_are_read_case_blind(self):
        assert MagazineSlotState.from_name("Occupied") is MagazineSlotState.OCCUPIED
        assert MagazineSlotState.from_name("State.LOADED") is MagazineSlotState.LOADED
        assert MagazineSlotState.from_name("Docking") is MagazineSlotState.UNKNOWN


# ---------------------------------------------------------------------------
# The builder: entry keys, then the old sim.loader block
# ---------------------------------------------------------------------------


class TestBuilder:
    def _build(self, options, legacy=None):
        microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
        microscope.system.sim = dict(microscope.system.sim, loader=legacy or {})
        entry = DeviceEntry.from_dict(
            {"name": "sample_loader", "type": "sample_loader", **options}
        )
        return build_demo_sample_loader(entry, BuildContext(microscope=microscope))

    def test_entry_keys_build_the_magazine(self):
        device = self._build({"capacity": 6, "occupied": [2], "names": {2: "grid-elm"}})
        assert device.name == "sample_loader"
        assert device.capacity.get_value() == 6
        assert device.magazine.get_value().slot(2).description == "grid-elm"

    def test_the_old_sim_loader_block_fills_what_the_entry_leaves_out(self):
        device = self._build(
            {"capacity": 4},
            legacy={"capacity": 9, "occupied": [3], "exchange_delay": 2},
        )
        assert device.capacity.get_value() == 4  # the entry wins
        assert device.magazine.get_value().slot(3).state is MagazineSlotState.OCCUPIED
        assert device.exchange_time.get_value() == 4.0


# ---------------------------------------------------------------------------
# The grid model over the device, against the in-memory loader it replaces
# ---------------------------------------------------------------------------


def _arctis(loader_kind: str, **config):
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)
    if loader_kind == "old":
        loader = DemoSampleLoader(microscope, **config)
    else:
        device = _device(parent=microscope, **config)
        loader = DeviceSampleLoader(microscope, device, read_at_connect=True)
    microscope._stage.loader = loader
    return microscope


def _report(microscope) -> list:
    return [
        (e.slot_name, e.name, e.state.value) for e in microscope._stage.grid_inventory()
    ]


def _working(microscope):
    grid = microscope._stage.loader.working_slot.loaded_grid
    return grid.name if grid is not None else None


class TestGridModelParity:
    CONFIG = dict(capacity=6, occupied=(1, 2, 5), names={5: "grid-elm"})

    def _both(self, **config):
        return _arctis("old", **config), _arctis("device", **config)

    def test_inventory_matches_at_connect(self):
        for old, new in [self._both(**self.CONFIG)]:
            assert _report(new) == _report(old)

    def test_exchanges_match_step_by_step(self):
        old, new = self._both(**self.CONFIG)
        steps = [
            lambda m: m._stage.ensure_loaded("Grid-02"),
            lambda m: m._stage.ensure_loaded("grid-elm"),
            lambda m: m._stage.get_inventory(),
            lambda m: m._stage.unload(),
            lambda m: m._stage.run_inventory(),
            lambda m: m._stage.ensure_loaded("Grid-01"),
        ]
        for step in steps:
            step(old)
            step(new)
            assert _report(new) == _report(old)
            assert _working(new) == _working(old)

    def test_the_loaded_grid_is_one_object_in_its_home_and_the_working_slot(self):
        microscope = _arctis("device", **self.CONFIG)
        microscope._stage.ensure_loaded("grid-elm")
        microscope._stage.get_inventory()  # a read must not split the identity
        loader = microscope._stage.loader
        assert loader.working_slot.loaded_grid is loader.slots["Slot-05"].loaded_grid

    def test_unscanned_matches(self):
        old, new = self._both(capacity=4, occupied=(2,), start_unscanned=True)
        assert _report(new) == _report(old)
        old._stage.run_inventory()
        new._stage.run_inventory()
        assert _report(new) == _report(old)

    def test_a_failed_exchange_matches(self):
        old, new = self._both(**self.CONFIG)
        old._stage.loader.fail_next_exchange = True
        new._stage.loader.device.fail_next_exchange = True
        for microscope in (old, new):
            with pytest.raises(GridExchangeError):
                microscope._stage.ensure_loaded("Grid-01")
        assert _report(new) == _report(old)
        assert _working(new) is _working(old) is None

    def test_renaming_writes_the_slot_description(self):
        microscope = _arctis("device", **self.CONFIG)
        microscope._stage.assign_grid("Slot-01", SampleGrid(name="grid-ash"))
        device = microscope._stage.loader.device
        assert device.magazine.cached.slot(1).description == "grid-ash"
        microscope._stage.get_inventory()
        assert microscope._stage.loader.slots["Slot-01"].loaded_grid.name == "grid-ash"

    def test_exchange_seconds_is_the_devices(self):
        microscope = _arctis("device", exchange_delay=3.0)
        assert microscope._stage.loader.exchange_seconds == 6.0
