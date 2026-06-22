"""Unit tests for AutoscriptSampleLoader (maps specimen.autoloader → SampleGridLoader).

Uses a fake autoloader/connection so the API mapping can be checked without
AutoScript hardware.
"""

import types

import pytest

from fibsem.microscopes._stage import GridSlot, SampleGrid, SampleHolder
from fibsem.microscopes.autoscript import AutoscriptSampleLoader


class _Slot:
    def __init__(self, id, state, sample_description=None):
        self.id = id
        self.state = state
        self.sample_description = sample_description


class _FakeAutoloader:
    def __init__(self, last_known, scanned, stage):
        self._last_known = last_known
        self._scanned = scanned
        self.stage = stage
        self.is_installed = True
        self.loaded = []
        self.unloaded = 0
        self.get_slots_calls = []

    def get_slots(self, run_inventory):
        self.get_slots_calls.append(run_inventory)
        return self._scanned if run_inventory else self._last_known

    def load(self, slot_id):
        self.loaded.append(slot_id)

    def unload(self):
        self.unloaded += 1


def _holder():
    """A holder with one working slot, so load/unload can update occupancy."""
    return SampleHolder(capacity=1,
                        slots={"Slot-01": GridSlot(name="Slot-01", index=0)})


def _microscope(autoloader, holder):
    specimen = types.SimpleNamespace(autoloader=autoloader)
    stage = types.SimpleNamespace(holder=holder)
    return types.SimpleNamespace(
        connection=types.SimpleNamespace(specimen=specimen), _stage=stage)


def _loader(autoloader, capacity=4, holder=None):
    holder = holder if holder is not None else _holder()
    return AutoscriptSampleLoader(
        parent=_microscope(autoloader, holder), capacity=capacity)


# --- inventory --------------------------------------------------------------

def test_construction_does_not_query_hardware():
    al = _FakeAutoloader([_Slot(1, "Occupied", "grid-a")], [],
                         types.SimpleNamespace(state="Empty"))
    loader = _loader(al)
    assert al.get_slots_calls == []  # no inventory on init
    assert loader.loaded_magazine_slots == []  # magazine empty until scanned


def test_inventory_forces_scan_when_all_unknown():
    unknown = [_Slot(1, "Unknown"), _Slot(2, "Unknown")]
    scanned = [_Slot(1, "Occupied", "grid-a"), _Slot(2, "Empty")]
    al = _FakeAutoloader(unknown, scanned, types.SimpleNamespace(state="Empty"))
    loader = _loader(al)
    loader.run_inventory()
    # quick read all 'Unknown' → forced a physical scan
    assert al.get_slots_calls == [False, True]
    assert [s.loaded_grid.name for s in loader.loaded_magazine_slots] == ["grid-a"]


def test_inventory_uses_quick_read_when_states_known():
    known = [_Slot(1, "Occupied", "grid-a"), _Slot(2, "Occupied", "grid-b")]
    scanned = []  # would be wrong to use this
    al = _FakeAutoloader(known, scanned, types.SimpleNamespace(state="Empty"))
    loader = _loader(al)
    loader.run_inventory()
    assert al.get_slots_calls == [False]  # no forced scan
    assert sorted(s.loaded_grid.name for s in loader.loaded_magazine_slots) == \
        ["grid-a", "grid-b"]


def test_empty_sample_description_falls_back_to_slot_name():
    known = [_Slot(3, "Occupied", None)]  # occupied but no description
    al = _FakeAutoloader(known, [], types.SimpleNamespace(state="Empty"))
    loader = _loader(al)
    loader.run_inventory()
    assert loader.loaded_magazine_slots[0].loaded_grid.name == "Slot-03"


# --- load / unload ----------------------------------------------------------

def test_load_grid_loads_by_slot_id_and_updates_holder():
    known = [_Slot(1, "Occupied", "grid-a"), _Slot(2, "Occupied", "grid-b")]
    al = _FakeAutoloader(known, [], types.SimpleNamespace(state="Empty"))
    loader = _loader(al)
    loader.run_inventory()

    loader.load_grid("Slot-01", SampleGrid(name="grid-b"))

    assert al.loaded == [2]  # loaded by AutoloaderSlot.id
    holder = loader.parent._stage.holder
    # the holder working slot now reflects what was loaded
    assert holder.occupied_slots[0].loaded_grid.name == "grid-b"


def test_load_grid_unknown_grid_raises():
    al = _FakeAutoloader([_Slot(1, "Occupied", "grid-a")], [],
                         types.SimpleNamespace(state="Empty"))
    loader = _loader(al)
    loader.run_inventory()
    with pytest.raises(ValueError):
        loader.load_grid("Slot-01", SampleGrid(name="not-in-magazine"))


def test_unload_grid_calls_unload_and_clears_holder():
    al = _FakeAutoloader([_Slot(1, "Occupied", "grid-a")], [],
                         types.SimpleNamespace(state="Empty"))
    loader = _loader(al)
    loader.run_inventory()
    loader.load_grid("Slot-01", SampleGrid(name="grid-a"))

    loader.unload_grid("Slot-01")

    assert al.unloaded == 1
    assert loader.parent._stage.holder.occupied_slots == []  # holder cleared


def test_is_installed():
    al = _FakeAutoloader([], [], types.SimpleNamespace(state="Empty"))
    assert _loader(al).is_installed is True
