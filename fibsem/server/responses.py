"""What the App Server's and the Device Server's routes answer, as codec structures.

A route declares its class as ``response_model``, so ``/openapi.json`` describes
it, and returns ``answer(value)``: what ``fibsem.codec.encode`` writes, the one
encoder for files and the wire. A field typed ``Any`` is a value the device or
service declares itself (``type_path`` names its type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from fastapi.responses import JSONResponse

from fibsem.codec import encode, saved
from fibsem.control import InControl
from fibsem.structures import MicroscopeState


def answer(value: Any, type_: Any = None) -> JSONResponse:
    """*value* as a route's JSON answer, written as *type_* when that isn't its
    own class (a dict of descriptions)."""
    return JSONResponse(encode(value, type_))


@dataclass
class Status:
    status: str = "ok"


# --- The microscope as a whole ---


@dataclass
class Fitted:
    """What the instrument answered is fitted, at connect."""

    manipulator: bool = False


@dataclass
class MicroscopeInfo:
    manufacturer: str
    model: str
    serial_number: str
    software_version: str
    fitted: Fitted
    # The system configuration, in its file's shape.
    system: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateAnswer:
    microscope_state: MicroscopeState


@dataclass
class ControlView:
    in_control: Optional[InControl]
    yours: bool


@dataclass
class LeaseTaken:
    lease: str
    idle_s: float


@dataclass
class Released:
    released: bool


@dataclass
class LendEnded:
    ended: bool


# --- Devices and services ---


@dataclass
class Range:
    """A limit's ends; None is unbounded (JSON has no infinity)."""

    min: Optional[float] = None
    max: Optional[float] = None


# One range, or one per field of a composite value.
Limits = Union[Range, Dict[str, Range]]


@dataclass
class ParameterDescription:
    type: str
    unit: Optional[str]
    limits: Optional[Limits]
    choices: Optional[List[Any]]
    settable: bool
    # The value's type as its path, so a client reads it as that type.
    type_path: Optional[str] = None
    display: Optional[Dict[str, Any]] = None
    native_max: Optional[float] = None
    native_unit: Optional[str] = None
    # Whether the agent may set it; None where the server has no blocklist.
    agent: Optional[bool] = None


@dataclass
class QueryDescription:
    signature: str
    # JSON Schema of the arguments (`fibsem.schema`).
    arguments: Dict[str, Any]
    returns: Optional[str]
    available: bool


@dataclass
class CommandDescription(QueryDescription):
    agent: Optional[bool] = None
    # Answered at once with a run on the App Server.
    long: bool = False


@saved(keys={"class_": "class"})
@dataclass
class DeviceDescription:
    name: str
    class_: str
    driver: Optional[str]
    type: Optional[str]
    parameters: Dict[str, ParameterDescription]
    commands: Dict[str, CommandDescription]
    queries: Dict[str, QueryDescription]
    facts: Dict[str, Any]


@dataclass
class ValueAnswer:
    value: Any


@dataclass
class Values:
    values: Dict[str, Any]
    # The parameters this backend lacks.
    unavailable: List[str]


@dataclass
class MetadataAnswer:
    limits: Optional[Limits]
    choices: Optional[List[Any]]
    settable: bool
    native_max: Optional[float]
    native_unit: Optional[str]


@dataclass
class ResultAnswer:
    result: Any


@dataclass
class RunAnswer:
    """A long command's run (`fibsem.devices.runs.Run`)."""

    run_id: str
    service: str
    command: str
    started_at: str
    state: str
    ended_at: Optional[str] = None
    result: Any = None
    # How it failed: error_type and message, as a refusal says them.
    error: Optional[Dict[str, Any]] = None


@dataclass
class RunEnvelope:
    run: RunAnswer


@dataclass
class DeviceHealth:
    ok: bool
    detail: Optional[str] = None


@dataclass
class Health:
    ok: bool
    devices: Dict[str, DeviceHealth]
