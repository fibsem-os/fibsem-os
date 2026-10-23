"""What the App Server's and the Device Server's routes answer, as codec structures.

A route declares its class as ``response_model``, so ``/openapi.json`` describes
it, and returns ``answer(value)``: what ``fibsem.codec.encode`` writes, the one
encoder for files and the wire. A field typed ``Any`` is a value the device or
service declares itself (``type_path`` names its type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar, Union

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
from fibsem.workflows.outputs import RunFiles, StackSlice

if TYPE_CHECKING:
    from fastapi.responses import JSONResponse


def answer(value: Any, type_: Any = None, status_code: int = 200) -> JSONResponse:
    """*value* as a route's JSON answer, written as *type_* when that isn't its
    own class (a dict of descriptions)."""
    # Here, not at the top: the window imports these classes without the server.
    from fastapi.responses import JSONResponse

    return JSONResponse(encode(value, type_), status_code=status_code)


def answer_class(returns: Any, answer_types: Dict[type, type]) -> Any:
    """The class an app answers for *returns*: its own subclass of a base class
    it maps (*answer_types*, `AppContext.answer_types`), and a generic answer
    of one (`ItemEdit`) with the app's classes in it."""
    if returns in answer_types:
        return answer_types[returns]
    parameters = getattr(returns, "__parameters__", ())
    if parameters:
        return returns[
            tuple(answer_types.get(p.__bound__, p.__bound__) for p in parameters)
        ]
    return returns


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
class ItemStack(ItemRead):
    """The images an item's runs recorded under one role, as a stack."""

    role: Optional[str] = None
    slices: List[StackSlice] = field(default_factory=list)


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


@dataclass
class RunSummary:
    available: bool
    # the last run's outcome table, a row per item; its columns are the run's
    items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProtocolTask:
    """A task's plan entry (``TaskPlan``); its config is looked up by name."""

    name: str
    attention: Optional[str] = None
    requires: List[str] = field(default_factory=list)
    required: Optional[bool] = None
    scheduled_at: Optional[str] = None


@dataclass
class ProtocolView:
    """The workflow's tasks in order. An app adds its own in a subclass."""

    available: bool
    name: Optional[str] = None
    tasks: List[ProtocolTask] = field(default_factory=list)


@dataclass
class LogRecord:
    time: str
    level: str
    # one line, or a record's lines with what was logged under it
    text: Optional[str] = None
    lines: Optional[List[str]] = None


@dataclass
class LogRead:
    available: bool
    records: List[LogRecord] = field(default_factory=list)
    since: Optional[str] = None
    until: Optional[str] = None
    level: Optional[str] = None
    truncated: Optional[bool] = None
    read_from: Optional[str] = None


@dataclass
class TaskConfigRead:
    """A task's config document, at the protocol's level or an item's, with the
    version a patch echoes; ``error`` and the ``task_names`` there are when
    there is no such task."""

    available: bool
    task_name: Optional[str] = None
    item_name: Optional[str] = None
    level: Optional[str] = None
    version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    task_names: Optional[List[str]] = None
    item_names: Optional[List[str]] = None
    applied: Optional[bool] = None


@dataclass
class ItemHistory:
    available: bool
    item_name: str
    # the filter shown (`history_rows.HistoryFilter`), the tasks run, and
    # how many runs before it filtered
    filter: Optional[Dict[str, Any]] = None
    tasks: List[str] = field(default_factory=list)
    total: Optional[int] = None
    # each run shown (`history_rows.run_record`), files relative to the experiment
    runs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    item_names: Optional[List[str]] = None


@dataclass
class GridSummary:
    name: str
    id: str
    description: str
    quality: str
    status: str
    is_failure: bool
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    last_completed_task: Optional[str] = None
    num_items: int = 0
    # each overview role's latest file (a basename)
    overviews: Dict[str, str] = field(default_factory=dict)


@dataclass
class Grids:
    available: bool
    items: List[GridSummary] = field(default_factory=list)


@dataclass
class GridDetail:
    """A grid's summary, every task run with its files, and its items; or
    ``available: false`` with the ``grid_names`` there are."""

    available: Optional[bool] = None
    grid_name: Optional[str] = None
    error: Optional[str] = None
    grid_names: Optional[List[str]] = None
    name: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None
    quality: Optional[str] = None
    status: Optional[str] = None
    is_failure: Optional[bool] = None
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    last_completed_task: Optional[str] = None
    num_items: Optional[int] = None
    overviews: Dict[str, str] = field(default_factory=dict)
    tasks: List[RunFiles] = field(default_factory=list)
    items: List[str] = field(default_factory=list)


@dataclass
class OverviewStagePosition:
    grid_name: str
    grid_id: str
    filename: str
    stage_position: FibsemStagePosition


@dataclass
class GridMarker:
    item_name: str
    # source-image pixels, origin top-left, +y down
    x: float
    y: float
    inside: bool
    is_completed: bool
    is_failure: bool
    last_completed_task: Optional[str] = None


@dataclass
class Unplaced:
    item_name: str
    reason: str


@dataclass
class GridMarkers:
    available: bool
    grid_name: Optional[str] = None
    filename: Optional[str] = None
    # the overview's scale block (`fibsem.server.images.image_metadata`)
    image: Optional[Dict[str, Any]] = None
    convention: Optional[str] = None
    markers: List[GridMarker] = field(default_factory=list)
    unplaced: List[Unplaced] = field(default_factory=list)
    linked: Optional[bool] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    grid_names: Optional[List[str]] = None
    filenames: Optional[List[str]] = None


@dataclass
class ProtocolFile:
    name: str
    path: str
    default: bool


@dataclass
class ProtocolFiles:
    folder: Optional[str]
    protocols: List[ProtocolFile] = field(default_factory=list)


@dataclass
class ExperimentEntry:
    name: str
    path: str
    # its folder under the experiments folder, as loads and reads take it
    relative_path: Optional[str]
    app: Optional[str]
    created_at: Optional[str]
    num_items: int
    available: bool
    instrument_model: Optional[str] = None
    instrument_serial: Optional[str] = None
    operator: Optional[str] = None
    software_version: Optional[str] = None


@dataclass
class RecentExperiments:
    items: List[ExperimentEntry]


@dataclass
class SavedExperiments:
    folder: str
    items: List[ExperimentEntry]


@dataclass
class FileRead:
    """One file of an experiment as its kind reads: a folder as its listing,
    text as text, an image as a preview, anything else as its entry."""

    path: Optional[str] = None
    name: Optional[str] = None
    kind: Optional[str] = None
    size: Optional[int] = None
    modified: Optional[str] = None
    images: Optional[int] = None
    # a folder
    root: Optional[str] = None
    dir: Optional[str] = None
    entries: Optional[List[FileEntry]] = None
    # text
    text: Optional[str] = None
    truncated: Optional[bool] = None
    # an image: its metadata and its preview (`fibsem.server.files.read`)
    summary: Optional[Dict[str, Any]] = None
    preview: Optional[Dict[str, Any]] = None


@dataclass
class Proposals:
    available: bool
    # each pending proposal (`Proposal.to_dict`) with its item, task, ids and
    # the reference image its values sit on
    proposals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Events:
    available: bool
    # each event record (`fibsem.workflows.events`)
    events: List[Dict[str, Any]] = field(default_factory=list)
    latest_seq: Optional[int] = None
    stream: Optional[str] = None
    oldest_available: Optional[int] = None


@dataclass
class EventHistory:
    available: bool
    events: List[Dict[str, Any]] = field(default_factory=list)
    # the offset to pass back as before for the page before this one
    start: Optional[int] = None


@dataclass
class Thickness:
    """A lamella's thickness read off a FIB image (`ThicknessResult.to_dict`,
    in nm and degrees), with the plot."""

    item_name: str
    image: str
    poi: Point
    window_um: List[float]
    window_from: str
    saved_to: Optional[str]
    pixel_size_nm: float
    region_px: List[int]
    continuity: Dict[str, Any]
    uncertainty_nm: float
    thickness_nm: Optional[Dict[str, Optional[float]]] = None
    thirds_nm: Optional[List[Optional[float]]] = None
    wedge: Optional[Dict[str, Optional[float]]] = None
    angle_deg: Optional[float] = None
    straightness: Optional[Dict[str, Optional[float]]] = None
    warnings: Optional[List[Dict[str, str]]] = None
    plot: Optional[Dict[str, str]] = None


@dataclass
class MarkersDrawn:
    image: str
    path: str
    items: List[str]
    plot: Dict[str, str]
    grid_name: Optional[str] = None
    linked: Optional[bool] = None
    reason: Optional[str] = None


@dataclass
class ImageDrawn:
    image: str
    # The picture saved beside the image: drawn/<image>.png.
    path: str
    # The marks as drawn: measured lines labelled with their length, each
    # place with its stage position when the image says where it was taken.
    marks: Optional[Dict[str, Any]]
    plot: Dict[str, str]


@dataclass
class ImagePairDrawn:
    before: str
    after: str
    path: str
    marks: Optional[Dict[str, Any]]
    plot: Dict[str, str]


@dataclass
class PlanStep:
    grid: str
    step: str


@dataclass
class GridPlan:
    available: bool
    valid: bool
    reason: Optional[str] = None
    screen_all: Optional[bool] = None
    task_names: Optional[List[str]] = None
    grid_names: Optional[List[str]] = None
    note: Optional[str] = None
    steps: List[PlanStep] = field(default_factory=list)
    # seconds of work, and each task's count and seconds
    estimate: Optional[Dict[str, Any]] = None


@dataclass
class Recorded:
    """A note, a message or evidence put on the record; for a message, its
    ``id``; for evidence, its record (`fibsem.workflows.evidence.save`)."""

    available: bool
    recorded: bool
    error: Optional[str] = None
    item_names: Optional[List[str]] = None
    evidence_id: Optional[str] = None
    image: Optional[str] = None
    uploaded: Optional[bool] = None
    id: Optional[str] = None
    source: Optional[str] = None
    marks: Optional[Any] = None
    item_name: Optional[str] = None
    created_at: Optional[str] = None
    by: Optional[str] = None
    channels: Optional[List[str]] = None


# --- Edits to the experiment and the run ---
#
# An edit answers ``applied`` and, when it changed a file, ``saved``. A
# refusal is an HTTP error (`app_routes._answer_or_refusal`); what an edit
# could not do but is not refused for is ``reason``.


@dataclass
class Edit:
    applied: bool = False
    available: Optional[bool] = None
    saved: Optional[bool] = None
    save_error: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Decided(Edit):
    # what a decision could not do while a task runs, and its kind
    running: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    synced_tasks: Optional[List[str]] = None


@dataclass
class Suggested(Edit):
    suggestion_id: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class RunStarted:
    started: bool
    available: Optional[bool] = None
    reason: Optional[str] = None
    # the names there are, when it named one that isn't
    task_names: Optional[List[str]] = None
    item_names: Optional[List[str]] = None
    grid_names: Optional[List[str]] = None
    screen_all: Optional[bool] = None
    # a grid run's steps in order
    plan: Optional[List[PlanStep]] = None


@dataclass
class Requeued:
    queued: bool
    available: Optional[bool] = None
    reason: Optional[str] = None
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    task_name: Optional[str] = None
    queue_version: Optional[int] = None


@dataclass
class QueueEdited(Edit):
    changed: Optional[bool] = None
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    task_name: Optional[str] = None
    action: Optional[str] = None
    queue_version: Optional[int] = None


@dataclass
class QueuedGridStep:
    item_id: str
    grid_name: str
    step: str


@dataclass
class GridTasksQueued:
    queued: bool
    reason: Optional[str] = None
    added: List[QueuedGridStep] = field(default_factory=list)
    queue_version: Optional[int] = None
    grid_names: Optional[List[str]] = None
    task_names: Optional[List[str]] = None


@dataclass
class RunningTask:
    item_name: str
    task_name: str


@dataclass
class Stopped:
    stopped: bool
    available: Optional[bool] = None
    reason: Optional[str] = None
    item_name: Optional[str] = None
    task_name: Optional[str] = None
    # the task that is running, when it was told to stop another
    running: Optional[RunningTask] = None


@dataclass
class TaskEdit(Edit):
    """An edit to the workflow's tasks: the task, and the tasks in order."""

    task_name: Optional[str] = None
    task_names: Optional[List[str]] = None
    supervise: Optional[bool] = None
    scheduled_at: Optional[str] = None
    invalid_value: Optional[str] = None
    task_type: Optional[str] = None
    requires: Optional[List[str]] = None
    required: Optional[bool] = None


@dataclass
class OptionsSet(Edit):
    turn_beams_off: Optional[bool] = None
    show_run_summary: Optional[bool] = None


@dataclass
class LamellaDefaultsSet(Edit):
    lamella_defaults: Optional[Dict[str, Any]] = None


@dataclass
class ProtocolSaved:
    saved: bool
    name: str
    path: str


@dataclass
class Change:
    path: str
    old: Any = None
    new: Any = None


@dataclass
class ConfigEdit(Edit):
    """A patch to a task config, or its milling stages reordered, against
    the version read: each value changed, and the new version."""

    task_name: Optional[str] = None
    item_name: Optional[str] = None
    changes: List[Change] = field(default_factory=list)
    version: Optional[str] = None
    # the items' configs that followed a protocol edit
    synced_tasks: Optional[List[str]] = None
    milling_key: Optional[str] = None
    order: Optional[List[str]] = None
    task_names: Optional[List[str]] = None
    # a correlation's run, under the item's folder
    run_path: Optional[str] = None


@dataclass
class GridEdit(Edit):
    grid_name: Optional[str] = None
    # a rename's old name
    was: Optional[str] = None
    added: Optional[List[str]] = None
    grid_names: Optional[List[str]] = None
    # the verdict and its reason (`GridQuality.to_dict`)
    quality: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    items_without_grid: Optional[List[str]] = None


# An app's own item detail, experiment summary and protocol view
# (`AppContext.answer_type`).
D = TypeVar("D", bound=ItemDetail)
S = TypeVar("S", bound=ExperimentSummary)
P = TypeVar("P", bound=ProtocolView)


@dataclass
class ItemEdit(Edit, Generic[D]):
    """An edit to one item; ``item`` is its detail after, when it answers it."""

    item_name: Optional[str] = None
    item: Optional[D] = None
    pose: Optional[str] = None
    stage_position: Optional[FibsemStagePosition] = None
    fluorescence_pose_followed: Optional[bool] = None
    # the objective's position, metres
    objective_position: Optional[float] = None
    # the items it was applied to, and those it skipped
    items: Optional[int] = None
    skipped: Optional[List[str]] = None
    # a patch: the fields it set, or each value it changed and the version after
    fields: Optional[List[str]] = None
    changes: Optional[List[Change]] = None
    version: Optional[str] = None
    synced_tasks: Optional[List[str]] = None


@dataclass
class ExperimentOpened(Generic[S]):
    applied: bool
    experiment: Optional[S] = None


@dataclass
class ProtocolLoaded(Generic[P]):
    applied: bool
    protocol: Optional[P] = None


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
    # the allowed agent that allowing this one unpairs: one per microscope
    replaces: Optional[str] = None


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
    # what it said it can do, as [name, value] rows
    facts: List[List[str]] = field(default_factory=list)
    # whether it has spent most of its budget: the session line warns
    budget_low: bool = False


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
    # Who marked it: agent or operator.
    by: str = "agent"
    # The fluorescence channels shown, for marks on a stack's projection.
    channels: Optional[List[str]] = None


@dataclass
class MessageSent:
    id: str
    # The image the operator marked, kept as evidence, when the message has one.
    evidence_id: Optional[str] = None


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
class AgentSession:
    # what the agent reported of its session, None before it has
    session: Optional[Dict[str, Any]] = None


@dataclass
class SuggestionDecided:
    proposal_id: str
    applied: bool
