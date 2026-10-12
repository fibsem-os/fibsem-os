"""Today's grid model, pinned before it moves into a service.

`fibsem.microscopes._stage` holds the sample holder, the grid model over the sample
loader device, and the commands that make a grid reachable. It is about to become the
`SampleHandling` service (`microscope.sample_handling`). This file pins what it does on
the three kinds of system it serves, so the move can show it changed nothing:

* ``flat``: the Demo's default rotation stage, with the shipped holder (no slot
  calibrated) and no loader;
* ``flat-calibrated``: the same, with both holder slots calibrated, so the moves and
  ``ensure_loaded`` succeed;
* ``arctis``: the ``sim-arctis`` compustage, whose one-slot shuttle is fed by the
  simulated autoloader;
* ``arctis-no-loader``: the same compustage with its ``sample_loader`` entry switched
  off (``enabled: false``), whose grids are exchanged by hand.

Each case records the holder it built, the inventory, and what each command returns or
raises: naming a grid, making it reachable, moving to its slot, unloading, reading and
scanning the inventory, and the occupancy a new session restores. Alongside, the
commands that reached the stage and sample loader drivers, in order.

The expected values are in ``tests/fixtures/sample_handling_pins.json``, recorded from
main at the commit that added this file. A change that is meant to move one regenerates
the file with

    PYTHONPATH=. python tests/test_sample_handling_pins.py --write

and says in its pull request which values moved and why.
"""

import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemStagePosition, SampleGrid

PINS_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "sample_handling_pins.json"
)

CASES = ("flat", "flat-calibrated", "arctis", "arctis-no-loader")


# ---- the only lines the move to the service changes ----------------------------------


def _handling(microscope):
    """The grid model under test."""
    return microscope._stage


def _rebuild(microscope) -> None:
    """Build the grid model again, as a configuration change or a new session does."""
    microscope._create_sample_stage()


# ---- connecting each case ------------------------------------------------------------


def _isolate_holder_files() -> None:
    """As tests/conftest.py does, but for every case, here and under --write: a fresh
    copy of the shipped default holder, never this machine's, no old occupancy file
    to import, and an empty session state."""
    import fibsem.microscopes._stage as stage_module
    import fibsem.session_state as session_state
    from fibsem.structures import default_sample_holder

    directory = tempfile.mkdtemp(prefix="pins-holder-")
    path = os.path.join(directory, "sample-holder.yaml")
    default_sample_holder(pre_tilt=0.0).save(path)
    cfg.SAMPLE_HOLDER_CONFIGURATION_PATH = path
    stage_module.SAMPLE_HOLDER_CONFIGURATION_PATH = path
    cfg.SAMPLE_HOLDER_OCCUPANCY_PATH = os.path.join(directory, "occupancy.yaml")
    # The session state (where occupancy is saved) is keyed by the configuration's
    # file name, and every case's configuration has a shipped name.
    session_state.SESSION_STATE_DIRECTORY = os.path.join(directory, "session")


def _connect(case: str):
    _isolate_holder_files()
    if case.startswith("flat"):
        name, sim, loader = (
            "microscope-configuration.yaml",
            {"is_compustage": False},
            None,
        )
    else:
        name, sim = "sim-arctis-configuration.yaml", {}
        loader = case == "arctis"
    data = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, name))
    data["sim"] = {**(data.get("sim") or {}), **sim}
    if loader is False:
        for entry in data["hardware"]["devices"]:
            if entry.get("type") == "sample_loader":
                entry["enabled"] = False
    # A directory of its own, as `tests/fixtures/demo_stage.py` does: the microscope
    # keeps the path to write calibrations back to.
    path = os.path.join(tempfile.mkdtemp(prefix=f"pins-{case}-"), name)
    utils.save_yaml(path, data)
    microscope, _ = utils.setup_session(
        config_path=path, manufacturer="Demo", setup_logging=False
    )
    if case == "flat-calibrated":
        _calibrate(microscope)
    return microscope


def _calibrate(microscope) -> None:
    """Give every holder slot a trusted position, 2 mm apart in x about where the
    stage is at connect. The holder is the configuration's, so a rebuild keeps it."""
    from copy import deepcopy

    from fibsem.structures import SlotCalibration

    stage = microscope.system.stage
    here = microscope.get_stage_position()
    for slot in _handling(microscope).holder.slots.values():
        position = deepcopy(here)
        position.x = (slot.index - 0.5) * 2e-3
        position.y = 0.0
        position.name = slot.name
        slot.position = position
        slot.calibration = SlotCalibration.builtin(
            float(stage.shuttle_pre_tilt), float(stage.rotation_reference)
        )


# ---- recording -----------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, FibsemStagePosition):
        return {
            axis: None
            if getattr(value, axis) is None
            else round(float(getattr(value, axis)), 12)
            for axis in ("x", "y", "z", "r", "t")
        }
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "value") and hasattr(value, "name"):  # an Enum
        return value.value
    return value


def _record_drivers(microscope, calls: List[Any]) -> None:
    """Append every stage move and sample loader command that reaches a driver."""
    stage = microscope.stage

    def wrap(obj, name, label, args_of):
        original = getattr(obj, name)

        def wrapper(*args, **kwargs):
            calls.append([label, _plain(args_of(*args, **kwargs))])
            return original(*args, **kwargs)

        setattr(obj, name, wrapper)

    wrap(stage, "_move_absolute", "stage.move_absolute", lambda p, *a, **k: p)
    wrap(stage, "_move_relative", "stage.move_relative", lambda p, *a, **k: p)
    loader = microscope.devices.get("sample_loader")
    if loader is not None:
        wrap(loader, "_load", "loader.load", lambda slot, *a, **k: slot)
        wrap(loader, "_unload", "loader.unload", lambda *a, **k: None)
        wrap(loader, "_scan", "loader.scan", lambda *a, **k: None)
        wrap(
            loader,
            "_set_description",
            "loader.set_description",
            lambda slot, text, *a, **k: [slot, text],
        )


def _outcome(call: Callable[[], Any], shape: Callable[[Any], Any] = _plain) -> Dict:
    try:
        return {"returned": shape(call())}
    except Exception as e:  # what it raises is part of what is pinned
        return {"raised": type(e).__name__, "message": str(e)}


def _inventory(rows) -> List[Dict]:
    return [
        {
            "slot": r.slot_name,
            "index": r.index,
            "source": r.source,
            "name": r.name,
            "state": r.state.value,
            "present": r.present,
            "loaded": r.loaded,
        }
        for r in rows
    ]


def _holder(holder) -> Dict:
    return {
        "name": holder.name,
        "capacity": holder.capacity,
        "pre_tilt": _plain(holder.pre_tilt),
        "slots": [
            {
                "name": s.name,
                "index": s.index,
                "calibrated": bool(s.is_calibrated),
                "position": _plain(s.position),
                "grid": None if s.loaded_grid is None else s.loaded_grid.name,
            }
            for s in sorted(holder.slots.values(), key=lambda s: s.index)
        ],
    }


def _slot_name(slot) -> Any:
    return None if slot is None else slot.name


# ---- the measurement -----------------------------------------------------------------


def _measure(case: str) -> Dict[str, Any]:
    microscope = _connect(case)
    handling = _handling(microscope)
    calls: List[Any] = []
    _record_drivers(microscope, calls)
    result: Dict[str, Any] = {
        "has_loader": handling.loader is not None,
        "holder": _holder(handling.holder),
        "inventory": _inventory(handling.grid_inventory()),
    }

    def step(name: str, call: Callable[[], Any], shape=_plain) -> None:
        del calls[:]
        outcome = _outcome(call, shape)
        result[name] = {
            **outcome,
            "calls": list(calls),
            "inventory": _inventory(handling.grid_inventory()),
        }

    step("get_inventory", lambda: _inventory(handling.get_inventory()), lambda v: v)
    step("run_inventory", lambda: _inventory(handling.run_inventory()), lambda v: v)

    # Name a grid. With a loader the inventory slot is a magazine slot; without one,
    # a holder slot.
    first_slot = handling.grid_inventory()[0].slot_name
    step(
        "assign_grid",
        lambda: handling.assign_grid(
            first_slot, SampleGrid(name="Pinned"), persist=True
        ),
    )
    present = [r.name for r in handling.grid_inventory() if r.present]
    target = present[0] if present else "Pinned"

    step("ensure_loaded", lambda: handling.ensure_loaded(target), _slot_name)
    step("ensure_loaded_again", lambda: handling.ensure_loaded(target), _slot_name)
    step(
        "ensure_loaded_unknown",
        lambda: handling.ensure_loaded("No-Such-Grid"),
        _slot_name,
    )
    step(
        "move_to_slot",
        lambda: [
            _outcome(lambda n=s.name: handling.move_to_slot(n))
            for s in sorted(handling.holder.slots.values(), key=lambda s: s.index)
        ],
        lambda v: v,
    )
    step("move_to_grid", lambda: handling.move_to_grid(target))
    step(
        "after_moves",
        lambda: {
            "current_slot": _slot_name(handling.current_slot),
            "current_grid": None
            if handling.current_grid is None
            else handling.current_grid.name,
            "loaded_grids": [g.name for g in handling.loaded_grids],
            "slot_at_origin": _slot_name(
                handling.slot_at_position(FibsemStagePosition(x=0.0, y=0.0))
            ),
        },
        lambda v: v,
    )
    step("unload", lambda: handling.unload())

    # A new grid model, as a new session builds: what occupancy comes back.
    del calls[:]
    _rebuild(microscope)
    handling = _handling(microscope)
    result["rebuilt"] = {
        "restore_occupancy": _plain(handling.restore_occupancy()),
        "holder": _holder(handling.holder),
        "inventory": _inventory(handling.grid_inventory()),
    }
    return result


def _measure_all() -> Dict[str, Any]:
    return {case: _measure(case) for case in CASES}


@pytest.fixture(scope="module")
def pins() -> Dict[str, Any]:
    with open(PINS_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("case", CASES)
def test_the_grid_model_is_unchanged(case: str, pins: Dict[str, Any]):
    measured = json.loads(json.dumps(_measure(case)))
    assert measured == pins[case]


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit(
            "usage: PYTHONPATH=. python tests/test_sample_handling_pins.py --write"
        )
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    with open(PINS_PATH, "w") as f:
        json.dump(_measure_all(), f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {PINS_PATH}")
