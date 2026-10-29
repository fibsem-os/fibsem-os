"""What a tool is (`ToolSpec`), the words its description shares, and the
pieces its route call is built from."""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union


@dataclass(frozen=True)
class Arg:
    """A tool argument other than a plain string: its words, its type and, when
    it is optional and not None, its default."""

    doc: str
    type: Any = str
    default: Any = None


def arg(doc: str, type: Any = str, default: Any = None) -> Arg:
    return Arg(doc, type, default)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    method: str  # "GET" | "POST" | "PUT"
    path: str
    level: str  # the level the route needs: "watch" | "assist" | "drive"
    # name -> its description, a plain string argument; or its `arg`.
    params: Dict[str, Union[str, Arg]] = field(default_factory=dict)
    # The params that may be left out; the rest are required. With params, the
    # tool's arguments: a request is checked against them at filing, and the
    # sidecar's tools take them (`arguments`).
    optional: Tuple[str, ...] = ()
    # Which router serves it: "microscope" unless the App Server's devices
    # service is off; "app" only when the server was built with an
    # app_context; "services" (the analysis services) and "base" always.
    # /capabilities reports which.
    router: str = "microscope"
    # The optional /app part that serves it (fibsem.server.parts); "" for the
    # core. /capabilities lists the parts the app serves.
    part: str = ""
    # Commands or changes the microscope, so takes control of it.
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
    # The name the sidecar answers its route's ``result`` under; "" answers
    # the route's whole answer.
    result_key: str = ""

    @property
    def arguments(self) -> Dict[str, Arg]:
        """Each param as its `Arg`."""
        return {
            name: Arg(value) if isinstance(value, str) else value
            for name, value in self.params.items()
        }

    @property
    def signature(self) -> inspect.Signature:
        """The tool's arguments as a Python signature, from which the sidecar's
        MCP server derives the tool's input schema."""
        parameters = []
        for name, argument in self.arguments.items():
            if name not in self.optional:
                default, annotation = inspect.Parameter.empty, argument.type
            elif argument.default is None:
                default, annotation = None, Optional[argument.type]
            else:
                default, annotation = argument.default, argument.type
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return inspect.Signature(parameters)

    @property
    def answers(self) -> Any:
        """The class named by ``returns``, or None for an image."""
        if not self.returns:
            return None
        from fibsem.server import responses

        return getattr(responses, self.returns)

    @property
    def level_note(self) -> str:
        """What the tool needs, in the agent's terms."""
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


# Marks on an image, as evidence and the draw tools take them
# (fibsem.workflows.evidence._marks).
MARKS_PARAM = (
    "{points: [{x, y, shape?, label?}], areas: [{x, y, width, height, label?}], "
    "circles: [{x, y, radius, label?}], lines: [{x1, y1, x2, y2, arrow?, measure?, "
    "label?}]} in the image's full-resolution pixels from its top left: shape is "
    "cross (default) or target (a ring crosshair); arrow puts a head at x2, y2; "
    "measure labels the line with its length, as the ruler does. Labels (at most "
    "32 characters) name what is where; at most 50 marks. Marks read off a crop "
    "take crop: the cell's name (B3) or the crop's box [x, y, width, height], "
    "their x, y then as read off that crop's picture. On a fluorescence stack any mark takes "
    "z, the plane it is on (0 the first, as fm_points and poi_points of "
    "correlation_correlate count; a fitted fraction is drawn on the nearest plane; "
    "pictures and the viewer say plane z + 1): a "
    "picture shows that plane, and the mark's "
    "[x, y, z] goes to correlation as it is"
)

# How each layer of a picture is drawn, as the viewer's layers panel sets it
# (fibsem.imaging.looks).
LOOKS_PARAM = (
    "optional: how each layer is drawn, as the viewer's layers panel sets it: "
    "{layer: {clim: [low, high] in the image's own values (left out: auto, the "
    "1st and 99th percentiles), gamma (1 linear), opacity (0 to 1), color}}; a "
    "layer is a fluorescence channel by name, or SEM / FIB for a beam image"
)
BOX_PARAM = (
    "optional: draw only this region, [x, y, width, height] in the image's "
    "pixels (marks stay in the image's pixels); each fluorescence channel is "
    "contrasted to the region"
)

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
    f"marks, {MARKS_PARAM}; experiment_draw_image shows them drawn. "
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
    "analysis": "Analysis",
    "app": "App",
    "experiment": "Experiment",
    "protocol": "Protocol",
    "item": "Item",
    "grid": "Grid",
    "workflow": "Workflow",
    "proposal": "Proposal",
    "request": "Request",
    "knowledge": "Knowledge",
}

# Watch tools that write a record (a note, a request) rather than only read.
_WRITES_AT_WATCH = frozenset(
    {
        "experiment_add_note",
        "experiment_message_operator",
        "knowledge_write",  # the microscope's knowledge base, not the experiment
        "server_report_session",  # the agent's own session, on the record
        "server_set_constraints",
        "server_set_checklist",
        "request_create",
        "request_withdraw",
        "item_measure_thickness",  # saves its result beside the image
        "correlation_correlate",  # saves a run in the lamella's Correlation folder
        "correlation_refine_point",  # saves its fit figure beside the image
        "experiment_render_markers",  # saves its picture beside the image
        "experiment_draw_image",  # saves its picture beside the image
        "experiment_draw_pair",  # saves its picture beside the first image
        "item_draw_milling",  # saves its picture beside the image
        "analysis_measure_intensity",  # saves its figure beside the image
        "analysis_measure_profile",  # saves its figure beside the image
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

# For a move that changes the pose.
RETRACT_FIRST = " Retract the FM objective first."
# For the safe move, which crosses to the other device by itself.
RETRACT_FIRST_SAFE_MOVE = (
    " Retract the FM objective first, except for a position at the other "
    "device on an offset FM mount: that move crosses with the objective out."
)

# For an FM image: inserted is not in focus.
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
