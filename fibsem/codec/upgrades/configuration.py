"""A microscope configuration as older versions wrote it, in the current shape.

The file carries its ``version`` (`fibsem.structures.CONFIGURATION_VERSION`); a file with
none was written before versioning. What changed, oldest first:

- Before version 1 every block sat at the top level (``stage:``, ``electron:``,
  ``ion:``, ``fm:``, ``imaging:``), a column's defaults inside its block, the holders
  and pre-tilt inside ``stage:``. Version 1 put the blocks under ``hardware:``, the
  defaults under ``defaults:`` and the measured values under ``calibration:``.
- Version 1 kept the stage's devices under ``stage.devices`` (``FIBSEM``, ``FM``)
  with one shared ``stage.device_range``, and their orientations as
  ``acquisition_orientations``, where an empty list meant any. Version 2 lists the
  devices (``hardware.devices``), each with its own ``origin``,
  ``available_orientations`` and ``range``.
- A column said ``plasma: true/false`` beside ``plasma_gas``, and shipped files spelt
  no gas as the string ``None``. Version 2 has ``plasma_gas``, null for none.
- A detector was once spelt with bare names (``mode``, ``contrast``), and the stage once
  stored ``rotation_180``, which is now derived.

Where a file states a value in both an old and a new place, the new one wins, as the
readers had it. Presence, not the version, decides each step, so a file half way
between two shapes upgrades as it says. Keys this version does not know are left
where they are, for the reader to report.

`fibsem.utils.load_microscope_configuration` upgrades a file this way when it loads
it, and writes it back after keeping a copy of the old one.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

from fibsem.structures import (
    BEAMS_STAGE_DEVICE,
    CONFIGURATION_VERSION,
    CONFIGURED_DEVICES,
    DEFAULT_AVAILABLE_ORIENTATIONS,
    DEFAULT_STAGE_DEVICES,
    KNOWN_ORIENTATIONS,
    VERSION_1_DEVICE_RANGE,
    BeamSettings,
    DeviceEntry,
    device_axes_to_dict,
    stage_device_entry_name,
)

# What `calibration:` took from the stage block in version 1.
CALIBRATION_KEYS = (
    "holders",
    "active_holder",
    "shuttle_pre_tilt",
    "rotation_centre_correction",
)

# The detector keys of a beam block, in the spelling a configuration uses.
DETECTOR_KEYS = ("type", "mode", "brightness", "contrast")

# What `defaults.<column>` took from a column's block in version 1: what a beam's
# settings and detector read.
BEAM_DEFAULT_KEYS = (
    tuple(name for name in BeamSettings.__dataclass_fields__ if name != "beam_type")
    + ("current",)
    + tuple(f"detector_{name}" for name in DETECTOR_KEYS)
)

# The ways shipped files wrote "no plasma gas", read as None.
NO_GAS = ("", "none", "null")


def upgrade(config: Dict[str, Any]) -> Dict[str, Any]:
    """``config`` from any version, in the current shape."""
    config = copy.deepcopy(config or {})
    hardware = config.get("hardware")
    hardware = hardware if isinstance(hardware, dict) else {}
    defaults = _section(config, "defaults")
    calibration = _section(config, "calibration")

    entries = _device_list(config, hardware)
    stage = entries.get("stage")
    if stage is not None:
        for key in CALIBRATION_KEYS:
            if key in stage:
                calibration.setdefault(key, stage.pop(key))
        stage.pop("rotation_180", None)  # derived since FIB-834
        for position in _version_1_positions(stage):
            _merge(hardware["devices"], entries, position)
    for name in ("electron", "ion"):
        entry = entries.get(name)
        if entry is None:
            continue
        _detector_prefixed(entry)
        _plasma_gas(entry)
        column = _section(defaults, name)
        for key in BEAM_DEFAULT_KEYS:
            if key in entry:
                column.setdefault(key, entry.pop(key))
    imaging = config.pop("imaging", None)
    if isinstance(imaging, dict) and not defaults.get("imaging"):
        defaults["imaging"] = imaging

    if hardware:
        config["hardware"] = hardware
    for name in ("defaults", "calibration"):
        if not config.get(name):
            config.pop(name, None)
    # Never lowered: a file from a later version keeps saying so.
    if config.get("version", 0) < CONFIGURATION_VERSION:
        config["version"] = CONFIGURATION_VERSION
    return config


def _device_list(config: dict, hardware: dict) -> Dict[str, dict]:
    """The configured devices' blocks moved into ``hardware.devices``; the entries by
    name.

    A block at the top level, then one under ``hardware:`` over it, then an entry the
    list already has over both, key by key, as the reader had it. Only the devices
    that had blocks move: any other ``hardware.<name>`` block was never read, and is
    left for the reader to report.
    """
    listed = hardware.get("devices")
    listed = list(listed) if isinstance(listed, list) else []
    entries: Dict[str, dict] = {}
    for item in listed:
        if isinstance(item, dict):
            try:
                entries[DeviceEntry.from_dict(item).name] = item
            except ValueError:
                pass  # left as it is, for the reader to report
    blocks: Dict[str, dict] = {}
    for source in (config, hardware):
        for name in CONFIGURED_DEVICES:
            if isinstance(source.get(name), dict):
                blocks[name] = {**blocks.get(name, {}), **source.pop(name)}
    converted: List[dict] = []
    for name, block in blocks.items():
        _merge(converted, entries, {"name": name, **block})
    if converted or "devices" in hardware:
        hardware["devices"] = converted + listed
    return entries


def _merge(devices: List[dict], entries: Dict[str, dict], block: dict) -> None:
    """``block`` into the entry of its name, which wins key by key; a new entry,
    appended to ``devices``, when there is none."""
    name = block["name"]
    type_ = CONFIGURED_DEVICES.get(name, name)
    entry = entries.get(name)
    if entry is None:
        entries[name] = entry = {"type": type_, **block}
        devices.append(entry)
        return
    merged = {"type": type_, **block, **entry}
    entry.clear()
    entry.update(merged)


def _section(parent: dict, name: str) -> dict:
    section = parent.get(name)
    if not isinstance(section, dict):
        section = parent[name] = {}
    return section


def _detector_prefixed(block: dict) -> None:
    """Bare detector names as the prefixed ones; the prefixed one wins. Not ``type``,
    which is the entry's: a bare ``type`` was always read as the device's."""
    for name in DETECTOR_KEYS[1:]:
        if name in block:
            block.setdefault(f"detector_{name}", block.pop(name))


def _plasma_gas(entry: dict) -> None:
    """``plasma`` folded into ``plasma_gas``: ``false`` means no gas, whatever the gas
    key says, since the flag was the one the drivers consulted."""
    flag = entry.pop("plasma", None)
    gas = entry.get("plasma_gas")
    if flag is False or (gas is not None and str(gas).strip().lower() in NO_GAS):
        entry["plasma_gas"] = None
    if flag is True and entry.get("plasma_gas") is None:
        # The old flag without a gas: not a plasma column until the instrument names
        # its gas at connect (`FibsemMicroscope._read_plasma_source`).
        logging.info(
            "The configuration says `plasma: true` but names no plasma gas; the "
            "gas is read from the instrument at connect."
        )


def _version_1_positions(stage: dict) -> List[dict]:
    """The device entries for the positions a version 1 ``stage.devices`` declares.

    Each declared device gets the shared ``device_range`` (or the 20 mm version 1 used
    when the file stated none), so a site keeps the window it had. The beams are the
    stage's zero and have no entry; an origin for them away from zero is not kept,
    and is warned about.
    """
    declared = stage.pop("devices", None) or {}
    shared = stage.pop("device_range", None) or device_axes_to_dict(
        VERSION_1_DEVICE_RANGE
    )
    entries = []
    for key, device in declared.items():
        device = dict(device or {})
        if key == BEAMS_STAGE_DEVICE:
            origin = device.get("origin") or {}
            if any(origin.get(axis) for axis in origin):
                logging.warning(
                    f"stage.devices.{key}.origin {origin} is not kept: the beams are "
                    "the stage's zero."
                )
            continue
        name = stage_device_entry_name(key)
        if "acquisition_orientations" in device:
            # Version 1's empty list meant any orientation.
            orientations = device.pop("acquisition_orientations") or list(
                KNOWN_ORIENTATIONS
            )
            device.setdefault("available_orientations", orientations)
        device.setdefault("range", shared)
        entries.append({"name": name, **device})
    undeclared = [
        key
        for key in DEFAULT_STAGE_DEVICES
        if declared and key != BEAMS_STAGE_DEVICE and key not in declared
    ]
    for key in undeclared:
        logging.warning(
            f"stage.devices declares no {key}; it takes the default position "
            f"{DEFAULT_STAGE_DEVICES[key].to_dict()}, with orientations "
            f"{list(DEFAULT_AVAILABLE_ORIENTATIONS.get(stage_device_entry_name(key), ()))}."
        )
    return entries
