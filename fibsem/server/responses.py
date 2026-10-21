"""What the App Server's and the Device Server's routes answer, as codec structures.

A route declares its class as ``response_model``, so ``/openapi.json`` describes
it, and returns ``answer(value)``: what ``fibsem.codec.encode`` writes, the one
encoder for files and the wire. A field typed ``Any`` is a value the device or
service declares itself (``type_path`` names its type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fibsem.ai.agent.recent import RecentRow
from fibsem.codec import encode, saved
from fibsem.control import InControl
from fibsem.structures import (
    FibsemImageMetadata,
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeState,
    Point,
)
from fibsem.workflows.outputs import RunFiles

if TYPE_CHECKING:
    from fastapi.responses import JSONResponse


def answer(value: Any, type_: Any = None, status_code: int = 200) -> JSONResponse:
    """*value* as a route's JSON answer, written as *type_* when that isn't its
    own class (a dict of descriptions)."""
    # Here, not at the top: the window imports these classes without the server.
    from fastapi.responses import JSONResponse

    return JSONResponse(encode(value, type_), status_code=status_code)


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
    # How it failed: error_type and message, as a refusal says them, and the
    # status that refusal answers (409 for a grid not in the inventory).
    error: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None


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


# --- The app's experiment and workflow (/app) ---
#
# A read answers ``available: false`` in a 200 when there is nothing to read
# (no experiment open, no run yet), and an item read names the items there are.


@dataclass
class ExperimentRef:
    name: str
    path: str
    num_items: int


@dataclass
class Asking:
    """A proposal a task is stopped on."""

    item_id: str
    item_name: str
    task_name: str
    proposal_id: str


@dataclass
class WorkflowState:
    running: bool
    current_task: Optional[str] = None
    current_item: Optional[str] = None
    queue_total: Optional[int] = None
    queue_version: Optional[int] = None
    asking: List[Asking] = field(default_factory=list)


@dataclass
class AppStatus:
    microscope_connected: bool
    experiment: Optional[ExperimentRef]
    workflow: WorkflowState


@dataclass
class QueueEntry:
    id: str
    item_name: str
    task_name: str
    status: str


@dataclass
class Queue:
    available: bool
    version: Optional[int] = None
    items: List[QueueEntry] = field(default_factory=list)


@dataclass
class SummaryRow:
    """One item of the experiment, as every app's summary has it. An app adds
    its own facts in a subclass."""

    experiment_name: str
    item_name: str
    item_id: str
    is_failure: bool
    last_completed_task: Optional[str] = None
    last_completed_at: Optional[str] = None


@dataclass
class ExperimentSummary:
    available: bool
    items: List[SummaryRow] = field(default_factory=list)


@dataclass
class HistoryRow:
    """One run of a task on an item, oldest first."""

    item_name: str
    item_id: str
    task_name: str
    task_id: str
    task_type: str
    task_status: str
    task_status_message: str
    iteration: Optional[int] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    completed_at: Optional[str] = None
    duration: Optional[float] = None


@dataclass
class TaskHistory:
    available: bool
    items: List[HistoryRow] = field(default_factory=list)


@dataclass
class ItemRead:
    """A read of one item: ``available: false`` with ``error`` and the
    ``item_names`` there are when it isn't there."""

    available: bool
    item_name: str
    error: Optional[str] = None
    item_names: Optional[List[str]] = None


@dataclass
class ItemDetail(ItemRead):
    """An item's durable facts. An app adds its own in a subclass."""

    id: Optional[str] = None
    description: Optional[str] = None
    # metres, in the milling coordinate system (+y up, origin centre)
    poi: Optional[Point] = None
    # fractions of the FIB frame, origin top-left
    alignment_area: Optional[FibsemRectangle] = None
    # where each pose puts the stage, None for a pose with no position
    poses: Dict[str, Optional[FibsemStagePosition]] = field(default_factory=dict)
    # the item document's version: patches echo it
    version: Optional[str] = None


@dataclass
class TaskOutputs(ItemRead):
    completed_tasks: List[str] = field(default_factory=list)
    # each run in order, with its existing files by role (basenames)
    tasks: List[RunFiles] = field(default_factory=list)


@dataclass
class PixelSize:
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class ImagePayload:
    """A JPEG preview with the scale facts to map coordinates onto it."""

    image_b64_jpeg: str
    # the preview's size
    width: int
    height: int
    # the source image's size
    full_width: Optional[int] = None
    full_height: Optional[int] = None
    beam_type: Optional[str] = None
    hfw: Optional[float] = None
    # metres per source pixel, and per preview pixel
    pixelsize: Optional[PixelSize] = None
    preview_pixelsize: Optional[float] = None
    acquired_at: Optional[str] = None


@dataclass
class DisplayImages:
    available: bool
    sem: Optional[ImagePayload] = None
    fib: Optional[ImagePayload] = None


@dataclass
class SavedImageMetadata:
    path: str
    metadata: FibsemImageMetadata


@dataclass
class SavedImagePosition:
    path: str
    stage_position: FibsemStagePosition


@dataclass
class PixelPoint:
    """Where a stage position falls on an image, in source pixels; ``reason``
    when the image cannot place it."""

    px: Optional[List[float]] = None
    inside: Optional[bool] = None
    reason: Optional[str] = None


@dataclass
class SavedImagePixels:
    path: str
    # the image's scale block (`fibsem.server.images.image_metadata`)
    image: Dict[str, Any]
    points: List[PixelPoint]


@dataclass
class FileEntry:
    name: str
    path: str
    kind: str
    size: Optional[int]
    modified: str
    # for a folder, where the images are under it
    images: Optional[int] = None


@dataclass
class FolderListing:
    root: str
    dir: str
    entries: List[FileEntry]


# --- The agent: its requests, and the operator's card (/app/agent) ---


@dataclass
class ActionView:
    """One tool call of a request, and the operator's lines for it."""

    tool: str
    arguments: Dict[str, Any]
    method: str
    path: str
    what: str = ""
    change: str = ""
    # every value, for "Show details": {columns, rows}
    details: Optional[Dict[str, Any]] = None


@dataclass
class AgentRequestView:
    request_id: str
    status: str
    reason: str
    actions: List[ActionView]
    created_at: float
    decided_by: Optional[str] = None
    decided_at: Optional[float] = None
    reject_reason: str = ""
    cancel_reason: str = ""
    # one per action run: tool, status_code, ok, and the route's answer
    results: List[Dict[str, Any]] = field(default_factory=list)
    expires_at: Optional[float] = None
    # what lapsing does, while it can lapse
    on_expiry: Optional[str] = None
    # the beam image the card draws the actions' values on
    image: Optional[str] = None
    # what the agent judged from (`fibsem.workflows.evidence`, its record)
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class RequestList:
    requests: List[AgentRequestView]


@dataclass
class LevelOption:
    level: str
    name: str
    word: str
    tip: str


@dataclass
class PictureInfo:
    caption: str
    image: bool
    version: int


@dataclass
class SuggestionCard:
    proposal_id: str
    proposal: str
    decision: str
    reason: str
    evidence: Optional[Dict[str, Any]] = None
    # what the picture is, when no evidence is cited
    no_evidence: Optional[str] = None


@dataclass
class Waiting:
    """The decision a run waits on, and who can make it."""

    proposal: str
    who: str


@dataclass
class ConnectAskView:
    """An agent on this PC asking to connect, as the card shows it."""

    id: str
    # as it names itself
    client: str
    # "this PC", or an address over SSH
    origin: str
    # the PC user it runs as
    user: str
    asked_at: float
    # what the allowed agents' list will call it
    name: str
    # what allowing it means at the level now, in the card's words
    if_allowed: str


@dataclass
class ConnectionDecided:
    id: str
    allowed: bool
    # the allowed agent's name in the list, None when denied
    name: Optional[str] = None


@dataclass
class AgentView:
    """The agent as the chip and card show it."""

    experiment: Optional[str]
    level: Optional[str]
    stopped: bool
    last_heard_s: Optional[float]
    client: Optional[str]
    # its session as it reported it, in the card's words
    session: Optional[str]
    request: Optional[AgentRequestView]
    suggestion: Optional[SuggestionCard]
    now: Optional[str]
    waiting: Optional[Waiting]
    pictures: Dict[str, PictureInfo]
    levels: List[LevelOption]
    # an agent on this PC asking to connect
    asking: Optional[ConnectAskView] = None


@dataclass
class ConnectText:
    # None when the sidecar isn't installed
    text: Optional[str]


@dataclass
class RecentRows:
    rows: List[RecentRow]


@dataclass
class Picture:
    image_b64_png: str
    caption: str
    version: int


@dataclass
class EvidenceView:
    preview: Optional[Dict[str, Any]]
    marks: Optional[Any]
    path: str


@dataclass
class MessageSent:
    id: str


@dataclass
class LevelSet:
    level: str


@dataclass
class AgentStopped:
    stopped: bool
    level: Optional[str]


@dataclass
class RequestDecided:
    id: str
    approved: bool


@dataclass
class SuggestionDecided:
    proposal_id: str
    applied: bool
