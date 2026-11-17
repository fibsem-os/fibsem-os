"""What was done to the sample, kept in a file so it outlives the session.

A scene grows from its seed, so the grid itself needs no saving: only what a
session did to it. With `sim: sample: state_file:` set, every change (a milling
pass, a spot burn, wear from ion imaging, a coat) is written to that file, and a
session that connects with the same sample finds the file and carries on from it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np

from fibsem.drivers.demo.sim import faults as sim_faults
from fibsem.drivers.demo.sim.scene import MilledRegion, SampleScene
from fibsem.drivers.demo.sim.view import BeamCut
from fibsem.structures.settings import FibsemStagePosition

logger = logging.getLogger(__name__)

FORMAT = 1
# what makes two scenes the same sample: with any of these different the grid
# is a different one, and trenches saved from the other would sit nowhere
SAME_SAMPLE = (
    "seed",
    "preparation",
    "cell_type",
    "film",
    "grid_mesh",
    "grid_pitch",
    "extent",
)


def _plain(value: Any) -> Any:
    """A value as JSON holds it: arrays and tuples tagged, so they come back."""
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_plain(v) for v in value]}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _value(data: Any) -> Any:
    if isinstance(data, list):
        return [_value(v) for v in data]
    if isinstance(data, dict):
        if set(data) == {"__array__"}:
            return np.array(data["__array__"], dtype=float)
        if set(data) == {"__tuple__"}:
            return tuple(_value(v) for v in data["__tuple__"])
        return {k: _value(v) for k, v in data.items()}
    return data


def _region(region: MilledRegion) -> dict:
    cut = region.beam_cut
    return {
        "points": _plain(np.asarray(region.points)),
        "depth": float(region.depth),
        "halo": float(region.halo),
        "beam_cut": None
        if cut is None
        else {
            "polygon": _plain(np.asarray(cut.polygon)),
            "turn": float(cut.turn),
            "fs": float(cut.fs),
            "k": float(cut.k),
        },
        "face": _plain(region.face),
        "scan": bool(region.scan),
    }


def _milled(data: dict) -> MilledRegion:
    cut = data.get("beam_cut")
    return MilledRegion(
        points=_value(data["points"]),
        depth=data["depth"],
        halo=data["halo"],
        beam_cut=None
        if cut is None
        else BeamCut(
            polygon=_value(cut["polygon"]), turn=cut["turn"], fs=cut["fs"], k=cut["k"]
        ),
        face=_value(data.get("face")),
        scan=data.get("scan", False),
    )


def sample(scene: SampleScene) -> dict:
    """What makes this scene the sample it is (`SAME_SAMPLE`)."""
    return _plain({name: getattr(scene, name, None) for name in SAME_SAMPLE})


def dump(scene: SampleScene) -> dict:
    """Everything done to the sample so far, as JSON holds it."""
    reference = scene.reference_position
    index = {id(region): n for n, region in enumerate(scene.milled)}
    schedule = getattr(scene, "_fault_schedule", None)
    return {
        "format": FORMAT,
        "sample": sample(scene),
        "reference_position": None if reference is None else reference.to_dict(),
        "holder": _plain(scene._holder_signature),
        "gis_thickness": float(scene.gis_thickness),
        "sputter_thickness": float(scene.sputter_thickness),
        "milled": [_region(region) for region in scene.milled],
        "cuts": _plain(scene._cuts),
        "passes": _plain(scene.mill_passes),
        "damage_started": _plain(scene.damage_started),
        "exposures": [
            {
                "key": [int(v) for v in key],
                **_plain({k: v for k, v in seen.items() if k != "region"}),
                "region": index.get(id(seen.get("region"))),
            }
            for key, seen in scene.exposures.items()
        ],
        "faults": None
        if schedule is None
        else {
            "counts": [[k[0], k[1], n] for k, n in schedule.counts.items()],
            "fired": [bool(fault.fired) for fault in schedule.faults],
            "log": _plain(schedule.log),
        },
    }


def load(scene: SampleScene, data: dict) -> None:
    """Put a scene back as `dump` found one. The grid is the scene's own, from
    its seed; this is only what was done to it."""
    reference = data.get("reference_position")
    if reference is not None:
        scene.reference_position = FibsemStagePosition.from_dict(reference)
    scene.gis_thickness = float(data.get("gis_thickness", scene.gis_thickness))
    scene.sputter_thickness = float(
        data.get("sputter_thickness", scene.sputter_thickness)
    )
    scene.milled = [_milled(region) for region in data.get("milled", [])]
    scene._cuts = _value(data.get("cuts", []))
    scene.mill_passes = _value(data.get("passes", []))
    scene.damage_started = _value(data.get("damage_started"))
    scene.exposures = {}
    for seen in data.get("exposures", []):
        seen = dict(_value(seen))
        key = tuple(int(v) for v in seen.pop("key"))
        n = seen.get("region")
        seen["region"] = None if n is None else scene.milled[n]
        scene.exposures[key] = seen
    faults = data.get("faults")
    if faults is not None and scene.faults:
        schedule = sim_faults.Faults(scene.faults, scene.seed)
        schedule.counts = {(event, beam): n for event, beam, n in faults["counts"]}
        for fault, fired in zip(schedule.faults, faults["fired"]):
            fault.fired = bool(fired)
        schedule.log = _value(faults["log"])
        object.__setattr__(scene, "_fault_schedule", schedule)
    # anything drawn before now shows the sample as it was
    for cache in ("_solid_cache",):
        if getattr(scene, cache, None) is not None:
            getattr(scene, cache).clear()


def save(scene: SampleScene, path: str) -> None:
    """Write the sample's state to `path`, whole or not at all."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    partial = f"{path}.part"
    with open(partial, "w") as f:
        json.dump(dump(scene), f)
    os.replace(partial, path)


def read(scene: SampleScene, path: str) -> Optional[dict]:
    """The state saved at `path`, when it is this sample's. None when there is
    no file, it cannot be read, or it is another sample's (a different seed,
    say): those are logged, and the session starts on a fresh sample."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("Could not read the sample state at %s (%s).", path, e)
        return None
    if data.get("format") != FORMAT or data.get("sample") != sample(scene):
        logger.warning(
            "The sample state at %s is another sample's (%s, this one is %s): "
            "starting fresh, and the file is replaced at the first change.",
            path,
            data.get("sample"),
            sample(scene),
        )
        return None
    return data


def attach(scene: SampleScene) -> bool:
    """Keep the scene in its `state_file` from here on, and first put back what
    the file holds. True when a saved sample was restored."""
    path = scene.state_file
    if not path:
        return False
    path = os.path.expanduser(str(path))
    data = read(scene, path)
    if data is not None:
        load(scene, data)
        logger.info(
            "Sample restored from %s: %d milled regions, %d passes.",
            path,
            len(scene.milled),
            len(scene.mill_passes),
        )

    def grids_placed() -> None:
        # the holder's grids are laid out at the first image, which clears what
        # was milled: the saved sample was on those same grids, so it goes back
        if data is not None and _plain(scene._holder_signature) == data.get("holder"):
            keep = scene.reference_position
            load(scene, data)
            scene.reference_position = keep

    scene.on_grids_placed = grids_placed
    scene.on_change = lambda: save(scene, path)
    return data is not None
