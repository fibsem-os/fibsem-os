"""What the App Server's routes take: a dataclass per request body.

A route declares its body as one of these, so FastAPI reads it and
``/openapi.json`` describes requests as `responses` describes answers. Each
class carries ``STRICT``, so a key it doesn't have is refused, as the codec's
strict decode refuses one: on the wire, a key the route doesn't take is the
caller's mistake. Numbers sent as text read as numbers, as the codec reads them.

A body that doesn't read is answered as every refusal is, ``{error_type,
message}`` (`invalid_body`), with ``missing_field`` when only required fields
are missing. Checks a type can't say (a length, a pattern, text that isn't
blank) stay in the route, with their own messages.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import ConfigDict, Field, Strict

from fibsem.server.auth import refusal_error

# Unknown keys refused: set as each class's ``__pydantic_config__``.
STRICT = ConfigDict(extra="forbid")

# A boolean sent as true or false: 1 or "yes" is refused, not read as True.
Flag = Annotated[bool, Strict()]


def invalid_body(errors: Sequence[Mapping[str, Any]]) -> HTTPException:
    """FastAPI's validation errors as one refusal: each field by its path,
    what was wrong with it."""
    said = []
    for error in errors:
        loc = [str(part) for part in error.get("loc", ())]
        where = ".".join(loc[1:] if loc[:1] == ["body"] else loc)
        kind = error.get("type")
        if not where:
            said.append(
                "Send a JSON body."
                if kind == "missing"
                else "The body is a JSON object."
            )
        elif kind == "missing":
            said.append(f"{where} is required.")
        elif kind in ("extra_forbidden", "unexpected_keyword_argument"):
            said.append(f"{where} is not a field this route takes.")
        else:
            message = str(error.get("msg", "")).removeprefix("Value error, ")
            said.append(f"{where}: {message}.")
    missing = all(error.get("type") == "missing" for error in errors)
    return refusal_error(422, "missing_field" if missing else "invalid", " ".join(said))


def refuse_invalid_requests(app: FastAPI) -> None:
    """Answer a body, path or query of *app*'s that doesn't read as every
    refusal is (`invalid_body`), not as FastAPI's list of errors."""

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> Response:
        return await http_exception_handler(request, invalid_body(exc.errors()))


# --- the agent's own routes (`fibsem.server.agent`) ---------------------------


@dataclass
class MarkedEvidence:
    """An image the operator marked for the agent, sent with a message
    (`evidence.save`). ``keep`` keeps it as an example: True, or the
    knowledge section (``sample``, ``playbook``, ``instrument``)."""

    __pydantic_config__ = STRICT

    image: Optional[str] = None
    marks: Optional[Any] = None
    channels: Optional[List[Any]] = None
    answers: Optional[str] = None
    verdict: Optional[str] = None
    judged: Optional[List[Dict[str, Any]]] = None
    keep: Union[Flag, str, None] = None
    sample: Optional[str] = None


@dataclass
class MessageBody:
    """POST /app/agent/messages: the operator's message to the agent. ``text``
    may be empty when ``evidence`` says it."""

    __pydantic_config__ = STRICT

    text: str = ""
    evidence: Optional[MarkedEvidence] = None


@dataclass
class LevelBody:
    """PUT /app/agent/level."""

    __pydantic_config__ = STRICT

    level: str


@dataclass
class ReasonBody:
    """A decision with the operator's reason: denying a connection, rejecting
    a request."""

    __pydantic_config__ = STRICT

    reason: str = ""


@dataclass
class AgentRequestBody:
    """POST /app/agent/requests: the agent's actions, each ``{tool,
    arguments}``, and why, for the operator to approve at Assist."""

    __pydantic_config__ = STRICT

    actions: List[Dict[str, Any]]
    reason: str


@dataclass
class WithdrawBody:
    """POST /app/agent/requests/{id}/withdraw."""

    __pydantic_config__ = STRICT

    reason: str


@dataclass
class PlanSectionBody:
    """POST /app/agent/plan/section: one section; empty text removes it."""

    __pydantic_config__ = STRICT

    section: str
    text: str = ""


@dataclass
class PlanBody:
    """PUT /app/agent/plan: the whole plan, its sections under ## headings."""

    __pydantic_config__ = STRICT

    markdown: str


@dataclass
class TranscriptBody:
    """POST /app/agent/transcript: lines fibsem-agent printed."""

    __pydantic_config__ = STRICT

    session_id: str
    entries: List[Any] = field(default_factory=list)


@dataclass
class MessagesReadBody:
    """POST /app/agent/read: the operator's messages the agent has read."""

    __pydantic_config__ = STRICT

    ids: List[str]


@dataclass
class LibraryFileBody:
    """PUT /app/library/{kind}/{name}: the file's ``text``, or its ``data``
    for the library to write."""

    __pydantic_config__ = STRICT

    text: Optional[str] = None
    data: Optional[Any] = None


# --- the workflow: runs, the queue, proposals and stopping (`app_routes`) -----


@dataclass
class RunBody:
    """POST /app/workflow/start: the tasks to run, over the items named, or
    every item."""

    __pydantic_config__ = STRICT

    task_names: List[str]
    item_names: Optional[List[str]] = None


@dataclass
class RunPlanBody(RunBody):
    """POST /app/workflow/plan: a run's confirmation, or for adding to the
    running queue (``add``); computes, never acts."""

    run_next: Flag = False
    add: Flag = False


@dataclass
class GridPlanBody:
    """POST /app/workflow/grids/plan: the grid tasks, over the grids named or
    every recorded grid; ``screen_all`` inventories first and takes no
    grid_names."""

    __pydantic_config__ = STRICT

    task_names: List[str]
    grid_names: Optional[List[str]] = None
    screen_all: Flag = False


@dataclass
class GridRunBody(GridPlanBody):
    """POST /app/workflow/grids/start."""

    unload_at_end: Flag = False
    report_at_end: Flag = False


@dataclass
class DecideBody:
    """POST /app/workflow/decide: a decision on a proposal, by its id (or the
    run it came from, ``task_id``), as GET /app/workflow/proposals lists it."""

    __pydantic_config__ = STRICT

    item_id: str
    task_name: str
    outcome: str
    proposal_id: Optional[str] = None
    task_id: Optional[str] = None
    values: Optional[Dict[str, Any]] = None
    patch: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    reason: str = ""
    actor: str = ""


@dataclass
class SuggestBody:
    """POST /app/workflow/proposals/suggest: what the agent would decide, for
    the operator to approve."""

    __pydantic_config__ = STRICT

    item_id: str
    task_name: str
    proposal_id: str
    reason: str
    outcome: Optional[str] = None
    values: Optional[Dict[str, Any]] = None
    patch: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    actor: str = ""


@dataclass
class SupervisionBody:
    """POST /app/experiment/protocol/supervision."""

    __pydantic_config__ = STRICT

    task_name: str
    supervise: Flag


@dataclass
class RequeueBody:
    """POST /app/workflow/queue/requeue: ``front`` puts it first."""

    __pydantic_config__ = STRICT

    item_name: str
    task_name: str
    front: Flag = False


@dataclass
class QueueEditBody:
    """POST /app/workflow/queue/edit: ``action`` is remove, move_up,
    move_down or run_next."""

    __pydantic_config__ = STRICT

    item_name: str
    task_name: str
    action: str


@dataclass
class QueueGridsBody:
    """POST /app/workflow/queue/grids: grid tasks added to a running grid run."""

    __pydantic_config__ = STRICT

    task_names: List[str]
    grid_names: List[str]


@dataclass
class StopBody:
    """POST /app/workflow/stop: why, for the record (the agent must say)."""

    __pydantic_config__ = STRICT

    reason: str = ""


@dataclass
class StopTaskBody:
    """POST /app/workflow/stop_task: the task meant, so a late stop stops
    nothing else."""

    __pydantic_config__ = STRICT

    item_name: str
    task_name: str
    reason: str = ""


# --- configuration: protocols, item configs, experiments (`app_routes`) -------


@dataclass
class PatchBody:
    """A config edit: ``patch`` is ``{"<dotted path>": value}``, ``version``
    the one the config read answered, so an edit over a newer one is refused."""

    __pydantic_config__ = STRICT

    patch: Dict[str, Any]
    version: str


@dataclass
class ScheduleBody:
    """POST /app/workflow/schedule: ISO-8601, or null to clear."""

    __pydantic_config__ = STRICT

    task_name: str
    scheduled_at: Optional[str] = None


@dataclass
class WorkflowOptionsBody:
    """POST /app/workflow/options: only the options given change."""

    __pydantic_config__ = STRICT

    show_run_summary: Optional[Flag] = None
    turn_beams_off: Optional[Flag] = None


@dataclass
class NewTaskBody:
    """POST /app/experiment/protocol/tasks."""

    __pydantic_config__ = STRICT

    task_type: str
    task_name: str


@dataclass
class TaskOrderBody:
    """Every task, in the new order."""

    __pydantic_config__ = STRICT

    order: List[str]


@dataclass
class RequirementsBody:
    """POST .../tasks/{task_name}/requirements: the earlier tasks it needs,
    and whether it must run."""

    __pydantic_config__ = STRICT

    requires: Optional[List[str]] = None
    required: Optional[Flag] = None


@dataclass
class GridRequirementsBody:
    """POST /app/experiment/grid_protocol/tasks/{task_name}/requirements."""

    __pydantic_config__ = STRICT

    requires: List[str]


@dataclass
class SaveProtocolBody:
    """POST /app/experiment/protocol/save: a protocol folder's name."""

    __pydantic_config__ = STRICT

    name: str
    overwrite: Flag = False


@dataclass
class StageOrderBody:
    """A task's milling stages, by name, in their new sequence."""

    __pydantic_config__ = STRICT

    milling_key: str
    order: List[str]
    version: str


@dataclass
class ApplyProtocolBody:
    """POST .../items/{item_name}/apply_protocol: the tasks to reset, or
    every task the protocol defines."""

    __pydantic_config__ = STRICT

    task_names: Optional[List[str]] = None


@dataclass
class NewExperimentBody:
    """POST /app/experiment/create: a folder under the experiments folder."""

    __pydantic_config__ = STRICT

    name: str
    protocol: Optional[str] = None
    description: Optional[str] = None
    sample: Optional[str] = None


@dataclass
class ExperimentPathBody:
    """POST /app/experiment/load: a path from the recents or the experiments
    folder."""

    __pydantic_config__ = STRICT

    path: str


@dataclass
class ProtocolNameBody:
    """POST /app/experiment/protocol/load."""

    __pydantic_config__ = STRICT

    protocol: str


@dataclass
class NameBody:
    """A new name."""

    __pydantic_config__ = STRICT

    name: str


@dataclass
class GridQualityBody:
    """POST .../grids/{grid_name}/quality: GOOD, REWORK, FAILED, or
    UNASSESSED to clear."""

    __pydantic_config__ = STRICT

    verdict: str
    reason: str = ""


@dataclass
class GridNoteBody:
    """POST .../grids/{grid_name}/note: '' clears."""

    __pydantic_config__ = STRICT

    note: str


@dataclass
class PlaceBody:
    """Where an item is: a stage position, or a point on a grid overview
    (``{grid_name, filename, px | fraction}``)."""

    __pydantic_config__ = STRICT

    stage_position: Optional[Dict[str, Any]] = None
    overview: Optional[Dict[str, Any]] = None


@dataclass
class NewItemBody(PlaceBody):
    """POST /app/experiment/items: neither place is the stage now; the app
    names it unless ``name`` is given."""

    name: Optional[str] = None


@dataclass
class PoseBody:
    """POST .../items/{item_name}/move_to: a pose name from the item."""

    __pydantic_config__ = STRICT

    pose: str


@dataclass
class CorrelationRunBody:
    """POST .../items/{item_name}/correlation: a run folder from
    correlation_correlate."""

    __pydantic_config__ = STRICT

    run_path: str


# --- images, analysis, exports and the record (`app_routes`) ------------------

# A point on an image, [x, y] in its pixels.
Point = Tuple[float, float]
# A fluorescence stack's plane, 0 the first: a whole number, not text.
Plane = Annotated[int, Strict(), Field(ge=0)]


@dataclass
class ThicknessBody:
    """POST .../items/{item_name}/thickness: a saved image, and the window to
    measure in, else the whole lamella."""

    __pydantic_config__ = STRICT

    path: str
    width_um: Optional[float] = None
    height_um: Optional[float] = None
    centre: Optional[Point] = None


@dataclass
class MarkersBody:
    """POST /app/experiment/images/markers."""

    __pydantic_config__ = STRICT

    path: str
    grid_name: Optional[str] = None
    fov: Optional[Flag] = None
    current: Optional[Flag] = None


@dataclass(kw_only=True)
class DrawOptions:
    """How a saved image is drawn: an FM stack's channels and looks, a box
    to crop to ([x, y, width, height] in pixels)."""

    __pydantic_config__ = STRICT

    crosshair: Optional[Flag] = None
    metadata_bar: Optional[Flag] = None
    channels: Optional[List[str]] = None
    looks: Optional[Dict[str, Dict[str, Any]]] = None
    box: Optional[Tuple[float, float, float, float]] = None


@dataclass(kw_only=True)
class DrawBody(DrawOptions):
    """POST /app/experiment/images/drawn: a saved image with marks on it."""

    path: str
    marks: Optional[Any] = None
    plane: Optional[Plane] = None


@dataclass(kw_only=True)
class ImagePairBody(DrawOptions):
    """POST /app/experiment/images/pair: two saved images side by side."""

    before: str
    after: str
    marks: Optional[Any] = None
    before_plane: Optional[Plane] = None
    after_plane: Optional[Plane] = None
    share_looks: Optional[Flag] = None


@dataclass
class IntensityBody:
    """POST /app/experiment/analysis/intensity: points on a saved image."""

    __pydantic_config__ = STRICT

    path: str
    points: List[Point]
    planes: Optional[List[Plane]] = None
    radius_px: Optional[float] = None
    channel: Optional[str] = None
    reference: Optional[Plane] = None


@dataclass
class ProfileBody:
    """POST /app/experiment/analysis/profile: a line on a saved image."""

    __pydantic_config__ = STRICT

    path: str
    start: Point
    end: Point
    width_px: Optional[float] = None
    channel: Optional[str] = None
    plane: Optional[Plane] = None
    dip: Optional[Flag] = None


@dataclass
class MillingDrawnBody:
    """POST .../items/{item_name}/milling/drawn: the milling tasks to draw,
    the first in colour; ``background`` is related or none."""

    __pydantic_config__ = STRICT

    task_names: Union[str, List[str]]
    background: str = "related"
    image_path: Optional[str] = None
    scan_direction: Optional[Flag] = None
    alignment_area: Optional[Flag] = None
    metadata_bar: Optional[Flag] = None


@dataclass
class PixelsBody:
    """POST /app/experiment/images/pixels: stage positions ({x, y, z, r, t,
    coordinate_system}) as points on a saved image."""

    __pydantic_config__ = STRICT

    path: str
    stage_positions: List[Dict[str, Any]]


@dataclass
class ImageExportBody:
    """POST /app/experiment/export/image: the dialog's options, or its
    defaults."""

    __pydantic_config__ = STRICT

    path: str
    options: Optional[Dict[str, Any]] = None
    format: str = "png"


@dataclass
class AnimationExportBody:
    """POST /app/experiment/export/animation."""

    __pydantic_config__ = STRICT

    item_name: str
    options: Optional[Dict[str, Any]] = None
    format: str = "gif"


@dataclass
class NoteBody:
    """POST /app/experiment/notes: an observation for the record."""

    __pydantic_config__ = STRICT

    text: str
    item_name: Optional[str] = None


@dataclass
class OperatorMessageBody(NoteBody):
    """POST /app/experiment/messages: the agent's message to the operator,
    ``reply_to`` the message it answers."""

    reply_to: Optional[str] = None


@dataclass
class EvidenceBody:
    """POST /app/experiment/evidence: an experiment image (``image``) or one
    the agent made (``upload``, base64), with marks on it."""

    __pydantic_config__ = STRICT

    image: Optional[str] = None
    upload: Optional[str] = None
    marks: Optional[Any] = None
    source: Optional[str] = None
    item_name: Optional[str] = None


# --- knowledge and pairing ----------------------------------------------------


@dataclass
class KnowledgeBody:
    """POST /app/knowledge: an entry, new or (with ``id``) replacing one.
    ``images`` are paths in the open experiment."""

    __pydantic_config__ = STRICT

    title: str
    text: str = ""
    section: Optional[str] = None
    sample: Optional[str] = None
    playbook: Optional[str] = None
    lasts: Optional[str] = None
    remind: Optional[str] = None
    id: Optional[str] = None
    item: Optional[str] = None
    images: List[str] = field(default_factory=list)


@dataclass
class PairBody:
    """POST /pair: the code shown, and what to call this computer. Unknown
    keys are ignored: a client of another version pairs here too."""

    code: str = ""
    name: str = ""
    browser: str = ""


@dataclass
class ConnectBody:
    """POST /connect: an agent asking to connect, waiting up to ``wait_s``.
    Unknown keys are ignored, as for pairing."""

    wait_s: Optional[float] = None
    client: str = ""
    origin: str = ""
    user: str = ""


@dataclass
class OpenPairingBody:
    """POST /pairing: for the operator's app, or ``agent`` (``hub`` on a
    gateway, with ``grants``)."""

    __pydantic_config__ = STRICT

    role: str = "operator"
    grants: List[str] = field(default_factory=list)


@dataclass
class GrantsBody:
    """PATCH /pairing/computers/{id}: the computer's grants, replaced."""

    __pydantic_config__ = STRICT

    grants: List[str] = field(default_factory=list)


@dataclass
class SwitchBody:
    """A setting turned on or off."""

    __pydantic_config__ = STRICT

    on: Flag
