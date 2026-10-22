"""What a tool is (`ToolSpec`), the words its description shares, and the
pieces its route call is built from."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    method: str  # "GET" | "POST" | "PUT"
    path: str
    level: str  # the level the route needs: "watch" | "assist" | "drive"
    params: Dict[str, str] = field(default_factory=dict)  # name -> description
    # The params that may be left out; the rest are required. With params, the
    # tool's arguments: a request is checked against them at filing (FIB-1425),
    # and the sidecar's signatures are pinned to them (test_sidecar).
    optional: Tuple[str, ...] = ()
    # Which router serves it: "microscope" unless the App Server's devices
    # service is off (FIB-1035); "app" only when the server was built with an
    # app_context; "services" (the analysis services) and "base" always.
    # /capabilities reports which.
    router: str = "microscope"
    # The optional /app part that serves it (fibsem.server.parts); "" for the
    # core. /capabilities lists the parts the app serves.
    part: str = ""
    # Commands or changes the microscope, so takes control of it (FIB-1270).
    hardware: bool = False
    # The level note, where the level alone does not say it (request_create).
    note: str = ""
    # What a call of it says in an agent's request, as the operator approves it
    # (fibsem.util.wording.Line). A tool whose route is a service command with
    # a line of its own (command(line=...)) needs none.
    line: Optional[Callable[..., Dict[str, Any]]] = field(default=None, compare=False)
    # Its route call's (path values, body) from its arguments, where they are
    # not the route's path and body as they stand (route_call).
    call: Optional[
        Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Optional[dict]]]
    ] = field(default=None, compare=False)
    # The class its route answers, by its name in `fibsem.server.responses`
    # (`answer_class` writes it as the app's); "" for an image.
    returns: str = ""

    @property
    def answers(self) -> Any:
        """The class named by ``returns``, or None for an image."""
        if not self.returns:
            return None
        from fibsem.server import responses

        return getattr(responses, self.returns)

    @property
    def level_note(self) -> str:
        """What the tool needs, in the agent's terms (FIB-1035, FIB-1283)."""
        if self.note:
            return self.note
        if self.level == "watch":
            return "Works at Watch, Assist and Drive."
        if self.level == "assist":
            return "Works at Assist and Drive without asking; refused (403) at Watch."
        note = (
            "Needs the agent at Drive; at Assist, ask first: put the call in "
            "request_create and the operator approves it (a direct call is "
            "refused, 403 ask_first). Refused (403) at Watch."
        )
        if self.hardware:
            return (
                f"{note} Refused (409) while a run or script is in control, "
                "except while a run waits on a proposal: then one action at a "
                "time, and the run waits for it to finish."
            )
        return f"{note} The operator sets the level from the Agent chip in the app."

    @property
    def full_description(self) -> str:
        return f"{self.description} {self.level_note}"

    @property
    def title(self) -> str:
        """The name a client shows: "Stage: Move absolute" for stage_move_absolute."""
        if self.name in _TITLES:
            return _TITLES[self.name]
        group, _, rest = self.name.partition("_")
        return f"{GROUP_TITLES[group]}: {rest.replace('_', ' ').capitalize()}"

    @property
    def read_only(self) -> bool:
        """Changes nothing: a Watch tool other than the few that write a record."""
        return self.level == "watch" and self.name not in _WRITES_AT_WATCH

    @property
    def takes_evidence(self) -> bool:
        """Takes the optional ``evidence`` (`EVIDENCE_PARAM`): any tool that
        acts or adds to the record. The sidecar keeps it and the call cites it."""
        return not self.read_only

    @property
    def destructive(self) -> bool:
        """Changes or removes what is there, as MCP means it: every write but
        the few that only add. A client may ask before one."""
        return not self.read_only and self.name not in _ADDITIVE

    @property
    def idempotent(self) -> bool:
        """Calling it twice with the same arguments does no more than once."""
        return self.read_only or self.method == "PUT"


# The optional argument every acting tool takes (`ToolSpec.takes_evidence`):
# what the agent judged from, kept in the experiment (fibsem.workflows.evidence).
EVIDENCE_PARAM = (
    "optional: the image you judged from, kept in the experiment and shown to "
    "the operator as this call's picture; say what you saw in it in your reason. "
    "{image | file, marks?, source?, item_name?}: image, a path inside the "
    "experiment as the image tools answered it (the request's values are drawn "
    "on it); or file, an image you made yourself in your working folder (an overlay "
    "your code drew, a plot; PNG or JPEG, at most 4 MB, saved at most 2048 px), "
    "shown as it is, with source, the experiment image it was drawn from; "
    "marks, {points: [{x, y, label?}], areas: [{x, y, width, height, label?}]} "
    "in the image's pixels from its top left, labels naming what is where. "
    "Attach it whenever an image decided what you do; without it the operator "
    "sees 'No evidence'"
)

# Group first, then verb (plans: mcp-tool-naming). A new tool joins a group
# here, or the group is added on purpose; the title names it.
GROUP_TITLES: Dict[str, str] = {
    "server": "Server",
    "microscope": "Microscope",
    "device": "Device",
    "service": "Service",
    "stage": "Stage",
    "beam": "Beam",
    "fm": "FM",
    "milling": "Milling",
    "coating": "Coating",
    "alignment": "Alignment",
    "correlation": "Correlation",
    "app": "App",
    "experiment": "Experiment",
    "protocol": "Protocol",
    "item": "Item",
    "grid": "Grid",
    "workflow": "Workflow",
    "proposal": "Proposal",
    "request": "Request",
}

# Watch tools that write a record (a note, a request) rather than only read.
_WRITES_AT_WATCH = frozenset(
    {
        "experiment_add_note",
        "server_report_session",  # the agent's own session, on the record
        "request_create",
        "request_withdraw",
        "item_measure_thickness",  # saves its result beside the image
        "correlation_correlate",  # saves a run in the lamella's Correlation folder
        "experiment_render_markers",  # saves its picture beside the image
    }
)

# Writes that only add: notes, the agent's own requests, a new
# item or experiment. Every other write changes or removes something.
_ADDITIVE = _WRITES_AT_WATCH | {
    "proposal_suggest",
    "item_add",
    "experiment_create",
    "protocol_add_task",
    "workflow_queue_grids",
}

# An overview's centre_position parameter.
OVERVIEW_CENTRE = (
    "stage position dict (x, y, z, r, t, coordinate_system) to centre the grid "
    "on; the stage comes back to where it was (default where the stage is)"
)

# For a move that changes the pose (FIB-1497).
RETRACT_FIRST = " Retract the FM objective first."
# For the safe move, which crosses to the other device by itself (#2096).
RETRACT_FIRST_SAFE_MOVE = (
    " Retract the FM objective first, except for a position at the other "
    "device on an offset FM mount: that move crosses with the objective out."
)

# For an FM image: inserted is not in focus (FIB-1497).
NEAR_FOCUS = (
    " The objective must be near its focus, not only inserted: go to the "
    "item's objective position (item_move_to_objective_position) or run "
    "fm_auto_focus first."
)

# Titles the name, read out, gets wrong.
_TITLES: Dict[str, str] = {
    "beam_auto_brightness_contrast": "Beam: Auto brightness and contrast",
}


# --- From a tool call to its route call (ToolSpec.call) ---


def stage_position(a: Dict[str, Any], x="x", y="y", z="z", r="r", t="t") -> dict:
    return {
        "name": None,
        "x": float(a[x]) if x in a else 0.0,
        "y": float(a[y]) if y in a else 0.0,
        "z": float(a[z]) if z in a else 0.0,
        "r": float(a[r]) if r in a else 0.0,
        "t": float(a[t]) if t in a else 0.0,
        "coordinate_system": "RAW",
    }


def absolute_position(a: Dict[str, Any]) -> dict:
    missing = [axis for axis in "xyzrt" if axis not in a]
    if missing:
        raise KeyError(", ".join(missing))
    return stage_position(a)


def beam_kwargs(a: Dict[str, Any], **more: Any) -> dict:
    """An imaging command's kwargs: its beam_type, checked, and *more* when set."""
    beam = str(a.get("beam_type", "ELECTRON")).upper()
    if beam not in ("ELECTRON", "ION"):
        raise ValueError(f"beam_type must be ELECTRON or ION, not {beam!r}")
    if more.get("image_settings") is not None:
        more["image_settings"] = dict(more["image_settings"], beam_type=beam)
    return {"beam_type": beam, **{k: v for k, v in more.items() if v is not None}}


def command_body(
    kwargs: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    preview: bool = False,
) -> dict:
    """`fibsem.devices.wire.command_body`, imported when called, as the catalog
    imports everything."""
    from fibsem.devices.wire import command_body

    return command_body(kwargs, reason, preview)


def kwargs_body(**values: Any) -> dict:
    return command_body(values)
