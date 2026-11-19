"""The App Server's blocklist: what the agent may never touch, at any level.

Two separate questions. The level says how the agent may
act right now: Watch reads, Assist asks, Drive acts. The blocklist names what
it may never touch: a blocked write or command is refused (403
``not_allowed``) at every level, an approved Assist request included, since a
request runs its actions through the same routes. Its value stays readable.
Everything not listed is allowed, and the level guards it.

The list is the ``operator_only`` marks next to the code they block
(`fibsem.devices.core.operator_only_reason`): a parameter's (``DeviceParameter(...,
operator_only=why)``), a command's (``@command(operator_only=why)``), or a whole kind of
device's (``operator_only = why``, as the chamber's). They sit on the base
classes, so every backend shares them and none can drop one.

The test for a mark is "never, whatever the level", like venting. Risky but
legitimate actions (an absolute stage move, a grid exchange, the beam voltage)
stay unmarked.
"""

import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

# Imported so their classes, and the marks on them, are known to `rules`.
import fibsem.devices.beam  # noqa: F401
import fibsem.devices.chamber  # noqa: F401
import fibsem.devices.fm  # noqa: F401
import fibsem.devices.gis  # noqa: F401
import fibsem.devices.manipulator  # noqa: F401
import fibsem.devices.sputter_coater  # noqa: F401
import fibsem.devices.stage  # noqa: F401
import fibsem.services.milling  # noqa: F401
import fibsem.services.stage_movement  # noqa: F401
from fibsem.devices.core import Device, DeviceParameter, operator_only_reason
from fibsem.services.core import Service


def _kinds() -> Iterator[type]:
    """Every device and service class, bases before the classes built on them."""
    seen = set()
    stack = [Device, Service]
    while stack:
        cls = stack.pop(0)
        if cls in seen:
            continue
        seen.add(cls)
        yield cls
        stack.extend(cls.__subclasses__())


def _inherited(cls: type, kind: str, name: str) -> bool:
    """Whether a class *cls* builds on already marks *name*."""
    return any(
        base is not object and _marked(base, kind, name) for base in cls.__mro__[1:]
    )


def _marked(cls: type, kind: str, name: str) -> bool:
    if kind == "class":
        return bool(vars(cls).get("operator_only"))
    attr = vars(cls).get(name)
    if kind == "write":
        return isinstance(attr, DeviceParameter) and bool(attr.operator_only)
    return bool(getattr(attr, "__operator_only__", None))


def _own_marks(cls: type) -> Dict[str, Dict[str, List[str]]]:
    """The marks *cls* adds itself, by reason: {why: {"write": [...], "command":
    [...], "class": [...]}}."""
    found: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    if _marked(cls, "class", "") and not _inherited(cls, "class", ""):
        found[vars(cls)["operator_only"]]["class"].append("")
    for name, attr in vars(cls).items():
        for kind in ("write", "command"):
            if _marked(cls, kind, name) and not _inherited(cls, kind, name):
                why = attr.operator_only if kind == "write" else attr.__operator_only__
                found[why][kind].append(name)
    return found


def _device_name(cls: type) -> str:
    first, *rest = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cls.__name__).split()
    return " ".join([first] + [word.lower() for word in rest])


def rules() -> List[Dict[str, str]]:
    """The blocklist in words, one row per kind of device and reason: what is
    blocked and why. The operator's Customisation dialog lists them."""
    rows = []
    for cls in _kinds():
        for why, marks in _own_marks(cls).items():
            if marks.get("class"):
                what = "every setting and command"
            else:
                parts = []
                if marks.get("write"):
                    parts.append("settings " + ", ".join(sorted(marks["write"])))
                if marks.get("command"):
                    parts.append("commands " + ", ".join(sorted(marks["command"])))
                what = "; ".join(parts)
            rows.append({"device": _device_name(cls), "blocked": what, "why": why})
    return rows


def blocked(device: Any, kind: str, name: str) -> Optional[str]:
    """Why the agent may never use *name* (a parameter write when *kind* is
    "write", a command when "command") on *device*, or None when it may."""
    return operator_only_reason(device, kind, name)


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


def in_experiment(
    kwargs: Mapping[str, Any], experiment_path: Callable[[], Optional[str]]
) -> Dict[str, Any]:
    """*kwargs* with each relative ``path`` or ``*_path`` read inside the open
    experiment: "overviews" is the experiment's overviews folder, never one
    beside the app. Unchanged with no experiment open."""
    root = experiment_path()
    kwargs = dict(kwargs)
    if root is None:
        return kwargs
    for name, value in kwargs.items():
        if (
            (name == "path" or name.endswith("_path"))
            and isinstance(value, str)
            and value
            and not os.path.isabs(value)
        ):
            kwargs[name] = os.path.join(root, value)
    return kwargs
