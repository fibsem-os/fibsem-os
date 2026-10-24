"""Every type a device or service serves crosses the wire and reads back as itself.

The types are gathered from the classes, not listed by hand: every parameter's type
and every command's and query's arguments and result, on every device and service
fibsem ships. A new one that the codec can't read fails here, until it reads, or is
named below as one that never travels as JSON.
"""

import enum
import importlib
import inspect
import pkgutil
import threading
import typing

import numpy as np
import pytest

from fibsem.autofunctions.autofocus import AutoFocusResult
from fibsem.autofunctions.brightness_contrast import BrightnessContrastResult
from fibsem.cancellation import AnyStopEvent
from fibsem.codec import decode
from fibsem.devices.core import _Controllable
from fibsem.devices.wire import (
    Frame,
    to_wire,
    type_from_path,
    type_hints,
    type_path,
)
from fibsem.imaging.spot import SpotBurnStatus
from fibsem.milling.progress import MillingProgressStatus
from fibsem.services.coating import CoatingStatus
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    GridSlot,
    GridSlotState,
    Point,
    SampleGrid,
)

# Sent as bytes (a TIFF, an npy), held in process (a stop event), or returned only,
# written by their own to_dict (a result assembled with its images).
NOT_JSON = {
    FibsemImage,
    np.ndarray,
    Frame,
    threading.Event,
    AnyStopEvent,
    AutoFocusResult,
    BrightnessContrastResult,
}

# A value for each served type that has no default.
SAMPLES = {
    "EmissionFilter": lambda cls: cls(name="GFP"),
    "MagazineSlot": lambda cls: cls(number=3, description="grid"),
    "SpotBurnProgress": lambda cls: cls(status=list(SpotBurnStatus)[0], total_points=4),
    "CoatingProgress": lambda cls: cls(
        status=list(CoatingStatus)[0], step="Depositing", remaining_time=12.0
    ),
    "MillingProgress": lambda cls: cls(status=list(MillingProgressStatus)[0]),
    "GridInventoryEntry": lambda cls: cls(
        slot_name="Slot-01",
        index=0,
        source="holder",
        name="grid-01",
        state=list(GridSlotState)[-1],
    ),
    "GridSlot": lambda cls: GridSlot(name="Slot-01", index=0),
    "SampleGrid": lambda cls: SampleGrid(name="grid-01"),
    "FibsemPatternSettings": lambda cls: FibsemRectangleSettings(
        width=1e-6, height=2e-6, depth=1e-6, centre_x=0.0, centre_y=1e-6
    ),
}

for _package in ("fibsem.devices", "fibsem.services", "fibsem.drivers"):
    _module = importlib.import_module(_package)
    for _info in pkgutil.walk_packages(_module.__path__, f"{_package}."):
        try:
            importlib.import_module(_info.name)
        except Exception:  # a vendor driver whose SDK isn't installed
            pass


def _classes():
    pending, found = list(_Controllable.__subclasses__()), set()
    while pending:
        cls = pending.pop()
        if cls not in found and cls.__module__.startswith("fibsem."):
            found.add(cls)
        pending.extend(cls.__subclasses__())
    return found


def _leaves(hint, into):
    if typing.get_origin(hint) is not None:
        for arg in typing.get_args(hint):
            if arg is not Ellipsis:
                _leaves(arg, into)
    elif isinstance(hint, type):
        into.add(hint)


def _result_hints():
    hints = []
    for cls in _classes():
        for _, method in inspect.getmembers(cls, inspect.isfunction):
            if hasattr(method, "__device_command__") or hasattr(
                method, "__device_query__"
            ):
                if "return" in type_hints(method):
                    hints.append(type_hints(method)["return"])
    return hints


def _served_types():
    types = set()
    for cls in _classes():
        for spec in cls.declared_parameters().values():
            _leaves(spec.type, types)
        for _, method in inspect.getmembers(cls, inspect.isfunction):
            if hasattr(method, "__device_command__") or hasattr(
                method, "__device_query__"
            ):
                for hint in type_hints(method).values():
                    _leaves(hint, types)
    plain = (bool, int, float, str, dict, list, tuple, type(None), typing.Any)
    return sorted((t for t in types if t not in plain), key=lambda t: t.__name__)


def _sample(cls):
    if cls.__name__ in SAMPLES:
        return SAMPLES[cls.__name__](cls)
    if issubclass(cls, enum.Enum):
        return list(cls)[-1]
    return cls()


SERVED = [t for t in _served_types() if t not in NOT_JSON]


def test_the_served_types_are_found():
    names = {t.__name__ for t in SERVED}
    assert {"FibsemStagePosition", "FibsemMillingStage", "BeamType"} <= names


@pytest.mark.parametrize("type_", SERVED, ids=lambda t: t.__name__)
def test_a_served_type_reads_back_as_itself(type_):
    value = _sample(type_)
    sent = to_wire(value)
    assert decode(type_, sent, strict=True) == value


def test_an_enum_crosses_by_name_where_it_says_so_and_reads_either_way():
    assert to_wire(BeamType.ION) == "ION"
    assert decode(BeamType, 2) is BeamType.ION


@pytest.mark.parametrize("type_", sorted(NOT_JSON, key=lambda t: t.__name__))
def test_a_type_that_never_travels_as_json_is_still_served(type_):
    """A stale entry here would hide a type that now has to read."""
    assert type_ in _served_types()


def test_every_served_result_type_reads_back_from_its_path():
    """A description names what each command returns by path; a client reads it."""
    for hint in _result_hints():
        path = type_path(hint)
        assert type_path(type_from_path(path)) == path


def test_every_served_parameter_type_reads_back_from_its_path():
    """A description names each parameter's type by path; a client reads it."""
    for cls in _classes():
        for spec in cls.declared_parameters().values():
            path = type_path(spec.type)
            assert type_path(type_from_path(path)) == path


def test_a_type_path_imports_only_fibsem():
    with pytest.raises(ValueError):
        type_from_path("os.system")
    with pytest.raises(ValueError):
        type_from_path("List[subprocess.Popen]")
    assert (
        type_from_path("List[fibsem.structures.settings.Point]") == typing.List[Point]
    )
