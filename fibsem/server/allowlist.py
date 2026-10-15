"""What the agent may never touch on the App Server, at any level.

Two separate questions (Patrick, 2026-10-14). The level says how the agent may
act right now: Watch reads, Assist asks, Drive acts. This list says whether it
may ever touch an entry: a blocked write or command is refused (403
``not_allowed``) at every level, an approved Assist request included, since a
request runs its actions through the same routes. Its value stays readable.

The test for a block is "never, whatever the level", like venting. Risky but
legitimate actions (an absolute stage move, a grid exchange, the beam voltage)
are allowed, and the level guards them.
"""

import os
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import Device
from fibsem.devices.fm import Objective
from fibsem.devices.manipulator import Manipulator
from fibsem.devices.stage import Stage

# Blocked by the kind of device, not its name: every beam, whatever it is called.
# "write" names parameters the agent may not set, "command" commands it may not
# run; ALL blocks every one.
ALL = "*"
_BLOCKED: Tuple[Tuple[type, Dict[str, Any]], ...] = (
    # Switching a column off; the app's image-to-stage geometry assumes its own
    # corrections (scan rotation is allowed); the source gas.
    (Beam, {"write": {"on", "angular_correction", "tilt_correction", "plasma_gas"}}),
    # Pumping and venting: the operator's, at the microscope. State and pressure
    # stay readable.
    (Chamber, {"write": ALL, "command": ALL}),
    # Homing, not for now; the stage link.
    (Stage, {"command": {"home", "link"}}),
    # Every move and the insert, stop included: blocked for now, to revisit.
    (Manipulator, {"write": ALL, "command": ALL}),
    # The safety limit the objective's moves are checked against.
    (Objective, {"write": {"limit_position"}}),
)

_WHY = {
    Beam: "a column setting the operator keeps",
    Chamber: "pumping and venting are the operator's, at the microscope",
    Stage: "the operator's, at the microscope",
    Manipulator: "the manipulator is the operator's for now",
    Objective: "the objective's safety limit",
}


def blocked(device: Device, kind: str, name: str) -> Optional[str]:
    """Why the agent may never use *name* (a parameter write when *kind* is
    "write", a command when "command") on *device*, or None when it may."""
    for cls, blocks in _BLOCKED:
        if isinstance(device, cls):
            names = blocks.get(kind, ())
            if names == ALL or name in names:
                return _WHY[cls]
    return None


def path_outside(
    kwargs: Mapping[str, Any], experiment_path: Callable[[], Optional[str]]
) -> Optional[str]:
    """Why a command's or query's path is refused, or None: the agent saves
    and reads inside the open experiment only, never anywhere the app can
    write. The path is a ``path`` argument (the focus sweep's, a milling
    task's folder), any ``*_path`` argument (the image a pattern preview draws
    on, an alignment's reference), or the ``path`` of settings that save (an
    image's ``image_settings``)."""
    paths = [
        value
        for name, value in kwargs.items()
        if name == "path" or name.endswith("_path")
    ] + [
        getattr(value, "path", None)
        for value in kwargs.values()
        if getattr(value, "save", False)
    ]
    paths = [str(path) for path in paths if path]
    if not paths:
        return None
    root = experiment_path()
    if root is None:
        return "no experiment is open; give paths inside one, or leave a save path out"
    root = os.path.realpath(root)
    for path in paths:
        target = os.path.realpath(path)
        if os.path.commonpath([root, target]) != root:
            return f"give a path inside the open experiment ({root}), not {path}"
    return None
