"""The tool catalog: the single source of truth for what agents can do.

The sidecar (``fibsem-mcp``) builds its MCP tool list from this module, and a
contract test asserts every entry maps onto a real route on the server app —
so the two surfaces cannot drift apart. Pure data, no dependencies beyond the
standard library: importable everywhere, including the py3.8 floor.

Levels mirror the server's enforcement (fibsem/server/auth.py). The sidecar
registers every tool the server's mounted routers serve, whatever the agent's
level (FIB-1035): the level changes mid-session, and an MCP client may never
re-read its tool list, so the server's 403 is the gate and each description
names the level it needs.
"""

import math
import re
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
    # Which router serves it: "microscope" unless the App Server's devices
    # service is off (FIB-1035); "app" only when the server was built with an
    # app_context; "base" always. /capabilities reports which.
    router: str = "microscope"
    # Commands or changes the microscope, so takes control of it (FIB-1270).
    hardware: bool = False
    # The level note, where the level alone does not say it (request_actions).
    note: str = ""

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
            "request_actions and the operator approves it (a direct call is "
            "refused, 403 ask_first). Refused (403) at Watch."
        )
        if self.hardware:
            return (
                f"{note} Refused (409) while a run or script is in control of "
                "the microscope."
            )
        return f"{note} The operator sets the level from the Agent chip in AutoLamella."

    @property
    def full_description(self) -> str:
        return f"{self.description} {self.level_note}"


CATALOG: Tuple[ToolSpec, ...] = (
    # --- read ---
    ToolSpec(
        name="get_capabilities",
        description="Server capabilities: API version, which halves are served (routers.microscope, routers.app), and the agent's level (off / watch / assist / drive). Read it first: the level says what the other tools will do. At Assist, act by asking: request_actions.",
        method="GET",
        path="/capabilities",
        level="watch",
        router="base",
    ),
    ToolSpec(
        name="get_system_info",
        description="The microscope: manufacturer, model, serial number and software version, which optional hardware is fitted, and its system settings.",
        method="GET",
        path="/microscope",
        level="watch",
    ),
    ToolSpec(
        name="get_stage_position",
        description="Current stage position (x/y/z in metres, r/t in radians).",
        method="GET",
        path="/microscope/devices/stage/position",
        level="watch",
    ),
    ToolSpec(
        name="get_stage_orientation",
        description="Named stage orientation (SEM, FIB, MILLING, FM or NONE).",
        method="POST",
        path="/microscope/services/stage_movement/queries/stage_orientation",
        level="watch",
    ),
    ToolSpec(
        name="get_microscope_state",
        description="Full microscope state snapshot (beams, detectors, stage).",
        method="GET",
        path="/microscope/state",
        level="watch",
    ),
    ToolSpec(
        name="list_devices",
        description=(
            "Every device on the microscope (electron, ion, stage, manipulator, ...): "
            "its parameters with type, unit, limits, choices and whether settable, "
            "and its commands. agent: false marks a write or command you may never "
            "use, at any level (403 not_allowed); its value stays readable."
        ),
        method="GET",
        path="/microscope/devices",
        level="watch",
    ),
    ToolSpec(
        name="get_parameter",
        description="Read one device parameter live, e.g. device=ion parameter=current.",
        method="GET",
        path="/microscope/devices/{device}/{parameter}",
        level="watch",
        params={
            "device": "device name from list_devices",
            "parameter": "parameter name",
        },
    ),
    ToolSpec(
        name="get_milling_angle",
        description="Current milling angle in degrees: the angle between the ion beam and the sample surface at the stage's rotation and tilt (90 at the FIB pose).",
        method="POST",
        path="/microscope/services/stage_movement/queries/current_milling_angle",
        level="watch",
    ),
    ToolSpec(
        name="get_geometry",
        description="The instrument's geometry in one read: column tilts, pre-tilt, rotation reference and centre, the orientation table, device origins, and what they mean at the stage's pose now: which way raw +z goes in the chamber, the milling angle here and the pose for the stored one, and per beam the stage move a +1 µm stable move along each image axis makes at the current scan rotation. Read it first on a new instrument, then confirm once with one small move. Angles in degrees.",
        method="POST",
        path="/microscope/services/stage_movement/queries/geometry",
        level="watch",
    ),
    ToolSpec(
        name="get_milling_state",
        description="Milling state (IDLE, RUNNING, PAUSED, ...).",
        method="GET",
        path="/microscope/services/milling/state",
        level="watch",
    ),
    ToolSpec(
        name="estimate_milling_time",
        description="Estimated time in seconds for the currently drawn patterns.",
        method="POST",
        path="/microscope/services/milling/queries/estimate",
        level="watch",
    ),
    ToolSpec(
        name="get_pattern_preview",
        description=(
            "Milling stages drawn over a saved ion image, to see where they would "
            "mill before running them: an image you can look at, the saved image's "
            "shape scaled to 1024 px wide. stages is a list of FibsemMillingStage "
            "dicts (as run_stage takes); left out, the patterns prepare drew. "
            "image_path must be inside the open experiment. Moves nothing."
        ),
        method="POST",
        path="/microscope/services/milling/queries/pattern_preview",
        level="watch",
        params={
            "image_path": "a saved ion image inside the open experiment",
            "stages": "FibsemMillingStage dicts; left out, what prepare drew",
        },
    ),
    ToolSpec(
        name="stop_milling",
        description="Stop an in-progress mill, with the reason the operator reads.",
        method="POST",
        path="/microscope/services/milling/commands/stop",
        level="drive",
        params={"reason": "why you are stopping it, for the operator (required)"},
        note="A mill the agent started itself it can stop at Assist or Drive, never at Watch; anyone else's (the workflow's, the operator's) needs Drive, and at Assist goes in request_actions for the operator to approve. Never refused for someone else being in control of the microscope.",
    ),
    # --- hardware ---
    ToolSpec(
        name="acquire_image_preview",
        description=(
            "Acquire a fresh image and return an agent-sized JPEG preview with its "
            "scale (hfw, pixel size). With no image_settings the beam images as it "
            "is set now. image_settings set the beam first and stay set; a field "
            "left out takes its default (resolution [1536, 1024], dwell_time 1e-6, "
            "hfw 150e-6), not the beam's current value. Nothing is saved."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/acquire",
        level="drive",
        hardware=True,
        params={
            "beam_type": "ELECTRON or ION",
            "image_settings": 'optional: ImageSettings fields, e.g. {"hfw": 150e-6, "resolution": [1536, 1024], "dwell_time": 1e-6}',
        },
    ),
    ToolSpec(
        name="stable_move",
        description=(
            "Move the stage so a feature seen in a beam image comes to the centre "
            "of that image: the way to move to what you see. dx and dy are the "
            "feature's offset from the image centre in metres, x to the right and "
            "y up (from a preview: (pixel - width/2) * hfw/width, and y with the "
            "sign flipped, since pixel rows count down). The server turns the "
            "offset into stage x, y and z for the beam's viewing angle, the stage "
            "tilt and rotation and the scan rotation, so the sample slides along "
            "its own surface and stays at the same height: the feature stays in "
            "focus and, at the coincidence point, stays in the other beam's view "
            "too. Prefer it to move_stage_relative whenever the move comes from an "
            "image. It uses the stage and scan rotation as they are now, so an "
            "offset measured on an older image is only right if neither has "
            "changed since. Take a fresh image afterwards to check where the "
            "feature landed."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/stable_move",
        level="drive",
        hardware=True,
        params={
            "dx": "the feature's offset from the image centre along x, metres, right positive",
            "dy": "the feature's offset from the image centre along y, metres, up positive",
            "beam_type": "the beam whose image the offset was measured in: ELECTRON or ION",
        },
    ),
    ToolSpec(
        name="vertical_move",
        description=(
            "Restore coincidence: with a feature centred in one beam's image, move "
            "the stage along the chamber vertical so the same feature comes to the "
            "centre of the other beam's image; the first beam sees no change. "
            "beam_type names the view the offset was measured in: ION (the "
            "default) when the feature is centred in the electron image, ELECTRON "
            "when it is centred in the ion image. dy (and dx) are the feature's "
            "offset from that image's centre, in metres, x right and y up. "
            "Use it after a stable_move in the other beam; never a plain z move."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/vertical_move",
        level="drive",
        hardware=True,
        params={
            "dy": "the feature's offset from the image centre along y, metres, up positive",
            "dx": "the offset along x, metres, right positive (usually 0)",
            "beam_type": "the view the offset was measured in: ION (default) or ELECTRON",
            "relaxation": "fraction of the exact move to make, 0-1 (default 1); below 1 undershoots, to keep a repeated look-and-correct loop from overshooting",
        },
    ),
    ToolSpec(
        name="move_to_orientation",
        description=(
            "Move the stage to a named orientation (SEM, FIB, MILLING, and FM "
            "where fitted): its rotation and tilt, by the safe move "
            "(safe_move_stage_absolute). x, y and z stay, except that a half turn "
            "carries x and y round the rotation centre where the instrument "
            "reports one, so the same spot comes back under the beam."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/move_to_orientation",
        level="drive",
        hardware=True,
        params={"orientation": "SEM, FIB, MILLING or FM"},
    ),
    ToolSpec(
        name="safe_move_stage_absolute",
        description=(
            "Move the stage to an absolute position (metres and radians) safely: "
            "for a large rotation it tilts flat first, and it rotates before "
            "moving the other axes, so a tilted sample isn't swung round. Use "
            "it, not move_stage_absolute, to return to a saved position."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/safe_absolute_stage_movement",
        level="drive",
        hardware=True,
        params={
            "x": "x in metres",
            "y": "y in metres",
            "z": "z in metres",
            "r": "rotation in radians",
            "t": "tilt in radians",
        },
    ),
    ToolSpec(
        name="move_stage_relative",
        description=(
            "Move the stage axes by a raw offset (metres), in stage coordinates. "
            "It knows nothing of the image: at any tilt a y move also changes "
            "the sample's height, so the feature drifts out of focus and out of "
            "the other beam. To move to something seen in an image, use "
            "stable_move instead."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/move_relative",
        level="drive",
        hardware=True,
        params={
            "dx": "relative x in metres",
            "dy": "relative y in metres",
            "dz": "relative z in metres",
        },
    ),
    ToolSpec(
        name="move_stage_absolute",
        description=(
            "Move every stage axis straight to an absolute position (metres and "
            "radians), with no safety ordering. For a position with a different "
            "rotation or tilt, use safe_move_stage_absolute instead."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/move_absolute",
        level="drive",
        hardware=True,
        params={
            "x": "x in metres",
            "y": "y in metres",
            "z": "z in metres",
            "r": "rotation in radians",
            "t": "tilt in radians",
        },
    ),
    ToolSpec(
        name="move_to_milling_angle",
        description="Tilt the stage to a milling angle, in degrees.",
        method="POST",
        path="/microscope/services/stage_movement/commands/move_to_milling_angle",
        level="drive",
        hardware=True,
        params={
            "milling_angle_deg": "target milling angle in degrees",
            "rotation_deg": "stage rotation in degrees; the rotation reference if left out",
        },
    ),
    ToolSpec(
        name="set_parameter",
        description=(
            "Set one device parameter (SI units); the server checks it against the "
            "parameter's limits and choices and answers the value written."
        ),
        method="PUT",
        path="/microscope/devices/{device}/{parameter}",
        level="drive",
        hardware=True,
        params={
            "device": "device name from list_devices",
            "parameter": "parameter name",
            "value": "the new value",
        },
    ),
    ToolSpec(
        name="call_command",
        description="Run a device command listed by list_devices, e.g. device=ion command=blank.",
        method="POST",
        path="/microscope/devices/{device}/commands/{command}",
        level="drive",
        hardware=True,
        params={
            "device": "device name from list_devices",
            "command": "command name",
            "kwargs": "the command's arguments, by name",
            "reason": "for a stop (stop, stop_live, cancel): why, for the operator (required there)",
        },
    ),
    ToolSpec(
        name="list_services",
        description=(
            "Every service on the microscope (milling, stage_movement, spot_burn, "
            "sample_handling, ...): what the instrument does with its devices over "
            "time. Each lists its parameters (e.g. milling progress) and its commands "
            "with their signatures and whether this microscope has them; agent: "
            "false marks one you may never use."
        ),
        method="GET",
        path="/microscope/services",
        level="watch",
    ),
    ToolSpec(
        name="get_service_parameter",
        description=(
            "Read one service parameter live, e.g. service=milling parameter=progress."
        ),
        method="GET",
        path="/microscope/services/{service}/{parameter}",
        level="watch",
        params={
            "service": "service name from list_services",
            "parameter": "parameter name",
        },
    ),
    ToolSpec(
        name="set_service_parameter",
        description=(
            "Set one service parameter, e.g. service=stage_movement "
            "parameter=milling_angle (degrees); the server checks it against the "
            "parameter's limits and answers the value written."
        ),
        method="PUT",
        path="/microscope/services/{service}/{parameter}",
        level="drive",
        hardware=True,
        params={
            "service": "service name from list_services",
            "parameter": "parameter name",
            "value": "the new value",
        },
    ),
    ToolSpec(
        name="call_service_command",
        description=(
            "Run a service command listed by list_services, e.g. service=stage_movement "
            "command=move_to_orientation kwargs={'orientation': 'SEM'}. A long command "
            "(listed with long: true: milling run, run_stage and run_task, spot burn "
            "run) answers at once with "
            "its run (run_id, state: running) and runs on, in control of the "
            "microscope; wait_for_run says how it ended. command=stop from "
            "another call stops it: the agent's own run at Assist or Drive, "
            "anyone else's at Drive, never at Watch, always with a reason."
        ),
        method="POST",
        path="/microscope/services/{service}/commands/{command}",
        level="drive",
        hardware=True,
        params={
            "service": "service name from list_services",
            "command": "command name",
            "kwargs": "the command's arguments, by name",
            "reason": "for a stop (stop, stop_live, cancel): why, for the operator (required there)",
        },
    ),
    ToolSpec(
        name="wait_for_run",
        description=(
            "How a long command you started (milling run, spot burn run) is going: "
            "its run with state running, done, failed or cancelled, and when it has "
            "ended its result or error. Waits up to wait seconds (at most 60) for "
            "it to end, so call it again while it says running. With run_id, waits "
            "for that run; a later run on the service answers at once. The same end "
            "comes as a command_ended event on get_events."
        ),
        method="GET",
        path="/microscope/services/{service}/run",
        level="watch",
        params={
            "service": "the service the command ran on, e.g. milling",
            "run_id": "the run_id the command answered with",
            "wait": "seconds to wait for the end, at most 60 (default 60)",
        },
    ),
    ToolSpec(
        name="call_service_query",
        description=(
            "Ask a service query listed by list_services: it takes arguments, "
            "answers, and moves nothing, e.g. service=stage_movement "
            "query=project_stable_move kwargs={'dx': 1e-6, 'dy': 0, 'beam_type': 1, "
            "'base_position': {...}} for where a stable move from base_position "
            "would end (beam_type 1 is the electron beam, 2 the ion beam)."
        ),
        method="POST",
        path="/microscope/services/{service}/queries/{query}",
        level="watch",
        params={
            "service": "service name from list_services",
            "query": "query name",
            "kwargs": "the query's arguments, by name",
        },
    ),
    ToolSpec(
        name="autocontrast",
        description="Run autocontrast for a beam (changes detector settings).",
        method="POST",
        path="/microscope/services/imaging/commands/autocontrast",
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
    ),
    ToolSpec(
        name="auto_focus",
        description=(
            "Focus a beam: the instrument's own autofocus where the beam has one, "
            "else an image-based sweep of the working distance at the beam's field "
            "of view. Moves the working distance. Take a fresh image to check."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/auto_focus",
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
    ),
)


APP_TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_app_status",
        description="What the application is doing: experiment, running workflow, current task and item.",
        method="GET",
        path="/app/workflow/status",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_app_queue",
        description="The live work queue: id-anchored items with status, plus the mutation version.",
        method="GET",
        path="/app/workflow/queue",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_experiment_summary",
        description="Per-item summary of the open experiment.",
        method="GET",
        path="/app/experiment",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_task_history",
        description="Every task run so far: outcomes, durations, errors. Oldest first, by start time.",
        method="GET",
        path="/app/experiment/task_history",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_run_summary",
        description="The most recent workflow run's outcome table.",
        method="GET",
        path="/app/workflow/run_summary",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_log",
        description="The session's log (logfile.log), raw: one record per logged line, with any traceback after it. Read a window: a run's own with task_id (the id on its task events; a few seconds either side; INFO and above unless you give a level), or since/until; with none, the latest records. Times are the instrument's wall clock. level=WARNING keeps warnings and errors, which is usually what explains a failure. Over the lines cap the latest records are kept: truncated says older records in the window were dropped, so narrow the window or raise the level.",
        method="GET",
        path="/app/log",
        level="watch",
        router="app",
        params={
            "task_id": "a run's task id: read the log while that run was going",
            "since": "ISO time: records at or after it (instead of task_id)",
            "until": "ISO time: records at or before it (instead of task_id)",
            "level": "the lowest level to keep: DEBUG, INFO, WARNING, ERROR (default DEBUG, or INFO with task_id)",
            "lines": "the most records to return, the latest kept (default 200, max 2000)",
        },
    ),
    ToolSpec(
        name="get_protocol",
        description="The workflow definition with live supervision flags and schedules. Each task carries `attendance`: whether a person must be at the microscope while it runs and what waits for their decision afterwards, under its current attention.",
        method="GET",
        path="/app/experiment/protocol",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_item_detail",
        description="Everything durable about one item: status, failure flag, POI, alignment area, milling angle, and where its poses put the stage.",
        method="GET",
        path="/app/experiment/items/{item_name}",
        level="watch",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="get_task_outputs",
        description="The files an item's task runs produced (reference images by role). tasks has every run with its status; completed_tasks names only the runs that completed.",
        method="GET",
        path="/app/experiment/items/{item_name}/outputs",
        level="watch",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="get_item_output_image",
        description="One saved image of an item, as an image you can look at: a 768 px JPEG preview of a file get_task_outputs lists (stacks and channels max-projected). A preview, not the data. An unlisted name is refused with the valid ones.",
        method="GET",
        path="/app/experiment/items/{item_name}/outputs/{filename}",
        level="watch",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "filename": "a basename from get_task_outputs",
        },
    ),
    ToolSpec(
        name="get_grids",
        description="Every grid the experiment records: quality verdict, task status, completed tasks, linked item count, and the basename of its latest SEM/FIB/FM overview by role. Records only — not the live slot inventory.",
        method="GET",
        path="/app/experiment/grids",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_grid_detail",
        description="One grid: its summary plus every task run with the files it recorded by role, and the names of the items linked to it.",
        method="GET",
        path="/app/experiment/grids/{grid_name}",
        level="watch",
        router="app",
        params={"grid_name": "the grid name"},
    ),
    ToolSpec(
        name="get_grid_output_image",
        description="One saved overview (mosaic) of a grid, as an image you can look at: a 768 px JPEG preview of a file get_grids / get_grid_detail lists. A preview, not the data; get_grid_markers places the items on it. An unlisted name is refused with the valid ones.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}",
        level="watch",
        router="app",
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from get_grids / get_grid_detail",
        },
    ),
    ToolSpec(
        name="get_grid_markers",
        description="Where the experiment's items fall on one recorded overview, in source-image pixels with each item's status, plus the image's scale block (size, pixel size, field width, stage position). Computed server-side; never re-derive stage geometry. Items that cannot be placed are listed with the reason.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}/markers",
        level="watch",
        router="app",
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from get_grids / get_grid_detail",
        },
    ),
    ToolSpec(
        name="get_protocol_task_config",
        description="One task's protocol-level configuration document (the defaults new items copy), with a version token naming exactly this state of it. Full document, not a summary.",
        method="GET",
        path="/app/experiment/protocol/task_config/{task_name}",
        level="watch",
        router="app",
        params={"task_name": "the task name, as listed by get_protocol"},
    ),
    ToolSpec(
        name="get_item_task_config",
        description="One item's own copy of a task configuration (what its run actually executes — tasks may have rewritten it mid-run), with a version token naming exactly this state of it.",
        method="GET",
        path="/app/experiment/items/{item_name}/task_config/{task_name}",
        level="watch",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name, as listed by get_protocol",
        },
    ),
    ToolSpec(
        name="add_note",
        description="Put an observation on the record — the event stream and the experiment log. Use it for judgments worth keeping ('curtaining on 02's face, accepted anyway'); notes change nothing. Optionally name the item it concerns.",
        method="POST",
        path="/app/agent/notes",
        level="watch",  # Watch adds notes: they change nothing
        router="app",
        params={
            "text": "the note, at most 4000 characters",
            "item_name": "optional: the item this note concerns",
        },
    ),
    ToolSpec(
        name="request_actions",
        description="At Assist, ask the operator to approve one or more actions: each is {tool, arguments}, the tool you would call and the arguments you would give it, run in order once approved. Give the reason the operator reads before deciding. One request waits at a time (409 request_pending names it). Answers the request, status pending; follow it with get_request or the request_* events (request_decided, request_applied per action, request_finished). Approved actions run through their own routes, so each is checked and can be refused as a direct call would be; the first refusal stops the batch. A rejection carries the operator's reason. Changing the level or Stop agent cancels a waiting request. Reads need no request, nor does stopping your own mill (stop_milling): call those directly. Stopping anything else (stop_workflow, someone else's mill) goes in a request.",
        method="POST",
        path="/app/agent/requests",
        level="watch",
        router="app",
        params={
            "actions": "list of {tool, arguments}, at most 20",
            "reason": "why: what you saw and what the actions will do, for the operator",
        },
        note="Works at Assist only: at Drive make the calls yourself; refused (403 level) at Drive and Watch.",
    ),
    ToolSpec(
        name="get_request",
        description="One request you filed with request_actions: its status (pending, applying, applied, failed, rejected, cancelled), who decided it, the operator's reason on a reject, and each action's result or the route's error.",
        method="GET",
        path="/app/agent/requests/{request_id}",
        level="watch",
        router="app",
        params={"request_id": "the request_id request_actions answered"},
    ),
    ToolSpec(
        name="apply_protocol_to_item",
        description="Re-copy protocol task configs onto an existing item (protocol-level edits only reach items created after them; this brings an existing item up to date). Wholesale replace of that item's copies for the named tasks — omit task_names for all. Running tasks are refused.",
        method="POST",
        path="/app/experiment/items/{item_name}/apply_protocol",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_names": "optional list of task names; omitted = every task in the protocol",
        },
    ),
    ToolSpec(
        name="reorder_item_milling_stages",
        description="Reorder the milling stages inside one of an item's task configs ('run the stress relief first'). order must be exactly the current stage names in their new sequence — this can never add, drop, or duplicate a stage. Echo the version from get_item_task_config; stale is refused.",
        method="POST",
        path="/app/experiment/items/{item_name}/task_config/{task_name}/stages/reorder",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name",
            "milling_key": "which milling config within the task (e.g. mill_rough)",
            "order": "the current stage names in their new sequence",
            "version": "the version token from get_item_task_config",
        },
    ),
    ToolSpec(
        name="reorder_protocol_milling_stages",
        description="Reorder the milling stages inside a protocol-level task config. Same rules as the item form; reaches items created (or re-applied) after.",
        method="POST",
        path="/app/experiment/protocol/task_config/{task_name}/stages/reorder",
        level="drive",
        router="app",
        params={
            "task_name": "the task name",
            "milling_key": "which milling config within the task",
            "order": "the current stage names in their new sequence",
            "version": "the version token from get_protocol_task_config",
        },
    ),
    ToolSpec(
        name="set_task_schedule",
        description="Set (or clear) when a task may start: ISO-8601 timestamp (naive = instrument-local time), or null to clear. The workflow reads schedules at each task start, so a change takes effect at the next start; the schedule persists with the protocol.",
        method="POST",
        path="/app/workflow/schedule",
        level="drive",
        router="app",
        params={
            "task_name": "the task name, as listed by get_protocol",
            "scheduled_at": "ISO-8601 timestamp, or null to clear",
        },
    ),
    ToolSpec(
        name="set_workflow_options",
        description="Set whether the run summary dialog opens when a run ends (the workflow's \"Show summary after completion\"). Off still toasts each run's end with a Show summary link. Turn it off before starting runs yourself, so no dialog stands over the operator's window, and tell the operator. Persists with the protocol; get_protocol's options show the current value.",
        method="POST",
        path="/app/workflow/options",
        level="drive",
        router="app",
        params={
            "show_run_summary": "true to open the summary dialog at a run's end, false for the toast only",
        },
    ),
    ToolSpec(
        name="update_item_detail",
        description="Patch an item's own document with dotted-path edits: poi.x/poi.y (metres, milling frame), alignment_area.left/top/width/height (frame fractions), description, defect.verdict (UNASSESSED/GOOD/REWORK/FAILED) and defect.reason. milling_angle is read-only: it is an outcome of the Setup task, not an input. Echo the version from get_item_detail — a mismatch is refused (409 stale_config). Tasks re-record geometry at their own moments, so an edit here is what the NEXT run starts from, not a permanent override.",
        method="POST",
        path="/app/experiment/items/{item_name}",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from get_item_detail",
        },
    ),
    ToolSpec(
        name="update_protocol_task_config",
        description="Patch a task's protocol-level defaults (what new items copy) with dotted-path edits. Echo the version from get_protocol_task_config — a mismatch is refused (409 stale_config). Running tasks are never affected: they hold their item's copy; the edit reaches items created after.",
        method="POST",
        path="/app/experiment/protocol/task_config/{task_name}",
        level="drive",
        router="app",
        params={
            "task_name": "the task name, as listed by get_protocol",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from get_protocol_task_config",
        },
    ),
    ToolSpec(
        name="update_item_task_config",
        description="Patch an item's task config with dotted-path edits (e.g. {'milling.mill_rough.stages.0.pattern.depth': 2e-6}). Echo the version from get_item_task_config — a mismatch is refused (409 stale_config): re-read and re-patch. A task currently running for that item is refused (it already copied its config); pending tasks pick the change up at start. Values are validated against the live field's type and declared bounds; the response echoes every change old -> new.",
        method="POST",
        path="/app/experiment/items/{item_name}/task_config/{task_name}",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name, as listed by get_protocol",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from get_item_task_config",
        },
    ),
    ToolSpec(
        name="get_events",
        description="Events after a sequence number: milling/acquisition progress, stage moves, task lifecycle. Long-polls up to timeout seconds.",
        method="GET",
        path="/app/events",
        level="watch",
        router="app",
        params={
            "since": "return events with seq greater than this (0 for all held)",
            "timeout": "seconds to wait for new events before returning (max 30)",
        },
    ),
    ToolSpec(
        name="stop_workflow",
        description="Stop the running workflow — the same path as the GUI's Stop button; the mill halts too. Give the reason the operator reads.",
        method="POST",
        path="/app/workflow/stop",
        level="drive",
        router="app",
        params={"reason": "why you are stopping it, for the operator (required)"},
        note="Needs the agent at Drive: the workflow is never the agent's own. At Assist, put it in request_actions and the operator approves it (a direct call is refused, 403 ask_first); the operator can also press Stop. Refused (403) at Watch. Never refused for someone else being in control of the microscope.",
    ),
    ToolSpec(
        name="start_workflow",
        description="Start a workflow — the Run button, remotely. Names the tasks to run (from the protocol) and optionally the items; omitted items means all of them. Refused while a workflow is already running.",
        method="POST",
        path="/app/workflow/start",
        level="drive",
        router="app",
        params={
            "task_names": "the tasks to run, from the protocol's task names",
            "item_names": "the items (lamellae) to run them for; omit for all",
        },
    ),
    ToolSpec(
        name="plan_grid_workflow",
        description="What a grid run would execute — the preflight dialog's content: the (grid, step) sequence with each grid's load first, validated against the grid protocol and the recorded grids. screen_all plans over the known grids and says the run will use whatever the inventory finds. Computes only; reads no hardware.",
        method="POST",
        path="/app/workflow/grids/plan",
        level="watch",
        router="app",
        params={
            "task_names": "grid tasks to run, from the protocol's grid_tasks names",
            "grid_names": "grids to run them on; omit for every recorded grid",
            "screen_all": "true = inventory first, then every present grid (no grid_names)",
        },
    ),
    ToolSpec(
        name="start_grid_workflow",
        description="Start a grid run — the Grids view's Run button, remotely; with screen_all, its Screen all grids button (inventory, then every present grid). Refused while any workflow, lamella or grid, is running. Call plan_grid_workflow first and show the plan.",
        method="POST",
        path="/app/workflow/grids/start",
        level="drive",
        router="app",
        params={
            "task_names": "grid tasks to run, from the protocol's grid_tasks names",
            "grid_names": "grids to run them on; omit for every recorded grid",
            "screen_all": "true = inventory first, then every present grid (no grid_names)",
        },
    ),
    ToolSpec(
        name="set_task_supervision",
        description="Set whether a task asks for supervision, in the live protocol. Takes effect at the workflow's next prompt-or-proceed decision, mid-run — the same behaviour as the GUI's supervised/automated toggle.",
        method="POST",
        path="/app/agent/supervision",
        level="drive",
        router="app",
        params={
            "task_name": "the task to change, e.g. 'Rough Milling'",
            "supervise": "true to ask before acting, false to run automatically",
        },
    ),
    ToolSpec(
        name="requeue_task",
        description="Queue a task for an item (again) in the RUNNING workflow — 'run 03's fiducial again'. Re-running a completed pair is the queue's own re-run mechanism; the queue only exists during a run.",
        method="POST",
        path="/app/workflow/queue/requeue",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) to run the task for",
            "task_name": "the task to queue, from the protocol's task names",
            "front": "true to run it next instead of at the end (optional)",
        },
    ),
    ToolSpec(
        name="get_display_images",
        description="The SEM and FIB images the app's GUI is displaying right now, as previews — the display cache, not a new acquisition. Post-mill images appear here as soon as the GUI shows them, before they reach disk.",
        method="GET",
        path="/app/experiment/images",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_pending_prompt",
        description='The supervision question awaiting an answer, if any — message, options, and context images. A question that is also on the item\'s record says answer_via: "decide" and gives the item_id, task_name and proposal_id: it is decided with decide_review, like any other review, and answer_prompt will refuse it.',
        method="GET",
        path="/app/workflow/prompt",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="get_pending_reviews",
        description="Every proposal waiting for a decision (the Review tab's inbox, `reviews`): which item and task, its own id (proposal_id, which decide_review passes back) and the run it is from (task_id), whether the task that made it is stopped on the answer (holding_the_run: nothing else runs until it is decided, so decide those first), the proposed values (a point of interest in metres, milling frame), confidence and alternatives when the proposer gave any, provenance, its review mode, which tasks are waiting on the decision, and the reference image the values sit on as a JPEG preview. `to_check` lists proposals a producer applied itself (advise mode) that nobody has looked at yet: the run did not wait on them; acknowledge one with decide_review Confirmed and no values, which records the look and writes nothing. Empty lists when nothing is pending.",
        method="GET",
        path="/app/workflow/reviews",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="decide_review",
        description="Confirm or reject a proposal, exactly as the Review tab would. Pass the proposal_id get_pending_reviews gave for it: the decision is on that proposal, and is refused (409 stale_review) if it is no longer the one on the record -- the task re-ran, or asked again in the same run -- so re-read and decide the current one. (task_id, the run, is accepted in its place but cannot tell two questions from one run apart.) Confirming a pending proposal needs all its values (e.g. {'poi': {'x','y'}} metres, milling frame), as proposed or adjusted; they are written through, the consumer becomes runnable, and the delta against the proposal is computed, so pass the values you actually judge right. Confirming a to_check proposal (already applied) acknowledges it and must carry no values; to change an applied value, re-run the task. A wrong, missing or foreign value is refused (422 invalid_value) with nothing written. Rejected needs a reason and fails the task that was waiting on the decision, so nothing that requires it runs; the lamella itself is not marked defective. Refused (409) when the item has a task running -- stop it instead -- unless that task is the one stopped on this very proposal (holding_the_run), which a decision releases: Confirmed hands the values back and the task carries on, Rejected fails it. Also refused when there is no proposal for that item and task. The decision is recorded with the agent as author.",
        method="POST",
        path="/app/workflow/decide",
        level="drive",
        router="app",
        params={
            "item_id": "the item id from get_pending_reviews (by id, never name)",
            "task_name": "the task name from get_pending_reviews",
            "proposal_id": "the proposal_id from get_pending_reviews: the proposal you looked at",
            "task_id": "optional, and only in place of proposal_id: the run the proposal is from",
            "outcome": "Confirmed or Rejected",
            "values": "on Confirmed of a pending proposal: every value its kind carries, as proposed or adjusted; omitted (or empty) when acknowledging a to_check one",
            "reason": "required on Rejected",
            "author": "optional agent/model name, recorded as agent:<name>",
        },
    ),
    ToolSpec(
        name="answer_prompt",
        description='Answer the pending supervision question, exactly as clicking the matching button would. Not for a question whose prompt says answer_via: "decide" -- that one is refused here, with nothing clicked, and decided with decide_review. True = the positive option. Echo the question_id from get_pending_prompt; if that question is no longer pending the answer is refused (409 stale_prompt) — re-read and answer the current one. For EditAlignmentArea and PickPOI the answer may carry a value: adjusted geometry that is placed into the widget (the operator sees it land) and then accepted through the same click path. An out-of-bounds or wrong-shape value is refused (422 invalid_value) without clicking anything.',
        method="POST",
        path="/app/workflow/prompt/answer",
        level="drive",
        router="app",
        params={
            "response": "true for the positive option, false for the negative",
            "question_id": "the question_id from get_pending_prompt, naming the question being answered",
            "value": "optional adjusted geometry: {left, top, width, height} (fractions of the frame, origin top-left) for EditAlignmentArea, or {x, y} (metres, microscope image coordinates, origin centre, +y up) for PickPOI",
        },
    ),
    ToolSpec(
        name="suggest_answer",
        description="Suggest an answer to the pending supervision question, for the operator to take or not. Nothing is answered: the prompt shows 'Agent suggests: <the button>' with your reason under it, and the operator's click on it answers as theirs, recorded with you as the one who suggested it (prompt_suggested, then prompt_answered with suggested_by). Answering any other way ignores it; a new posting of the question drops it. Echo the question_id from get_pending_prompt (409 stale_prompt if it has gone). value is as for answer_prompt, and is placed in the widget only when the operator takes the suggestion. Not for a question whose prompt says answer_via: \"decide\" (409 not_suggestible): at Assist, decide it through request_actions.",
        method="POST",
        path="/app/workflow/prompt/suggest",
        level="assist",
        router="app",
        params={
            "response": "true for the positive option, false for the negative",
            "question_id": "the question_id from get_pending_prompt",
            "reason": "why: shown under the suggestion for the operator",
            "value": "optional geometry, as for answer_prompt",
        },
        note="Works at Assist and Drive; refused (403) at Watch. At Drive you may answer_prompt directly instead.",
    ),
    ToolSpec(
        name="list_recent_experiments",
        description="Recently opened experiments on this machine, without loading them.",
        method="GET",
        path="/app/experiment/recent",
        level="watch",
        router="app",
    ),
)

CATALOG = CATALOG + APP_TOOLS

# Every route the server serves is a tool above or here, with why it is not one
# (tests/server/test_catalog_discovery.py). A new route fails that test until
# someone decides. Path parameter names don't matter.
NOT_TOOLS: Dict[str, str] = {
    "GET /health": "Open liveness check, no token; the agent's first read is get_capabilities.",
    "GET /dashboard": "The operator's monitor page, HTML.",
    "GET /dashboard/log": "The operator's event log page, HTML; get_events is the agent's read.",
    "GET /app/events/history": "The event log page's read of the run before the live buffer; an agent reads a finished run from its experiment folder.",
    "GET /app/agent/requests": "The app's list of every request; the agent follows its own with get_request.",
    "GET /microscope/devices/{device}": "list_devices carries every device's description.",
    "GET /microscope/devices/{device}/{parameter}/metadata": "list_devices carries limits, choices and settable.",
    "GET /microscope/services/{service}": "list_services carries every service's description.",
    "GET /microscope/services/{service}/{parameter}/metadata": "list_services carries limits, choices and settable.",
    "GET /microscope/devices/{device}/values": "Not decided yet; get_parameter reads one value at a time.",
    "GET /microscope/services/{service}/values": "Not decided yet; get_service_parameter reads one value at a time.",
    "POST /microscope/devices/{device}/queries/{query}": "Not decided yet; services hold the questions an agent asks (call_service_query).",
    "PUT /microscope/state": "Not decided yet: restoring beams and stage is allowed, but has no tool.",
}


def tools_for_capabilities(capabilities: Dict) -> Tuple[ToolSpec, ...]:
    """Catalog entries to offer given a full /capabilities payload.

    Every tool whatever the agent's level: the level changes mid-session and
    the server's 403 is the gate (FIB-1035). The microscope's tools too while
    its routes are off: the operator can turn them on mid-session, and until
    then each answers 404 ``microscope_off`` saying so, where tools fixed at
    start would need the agent to reconnect. Only the app's tools depend on
    the routers: a server hosts an application or not for its whole life.
    """
    routers = dict(capabilities.get("routers", {}), base=True, microscope=True)
    return tuple(t for t in CATALOG if routers.get(t.router, False))


# --- From a tool call to its route call ---
# The sidecar makes each tool's call this way, and an approved request makes the
# same call, so a request takes the arguments the tool does.

_PATH_PARAM = re.compile(r"{(\w+)}")


def _position(a: Dict[str, Any], x="x", y="y", z="z", r="r", t="t") -> dict:
    return {
        "name": None,
        "x": float(a[x]) if x in a else 0.0,
        "y": float(a[y]) if y in a else 0.0,
        "z": float(a[z]) if z in a else 0.0,
        "r": float(a[r]) if r in a else 0.0,
        "t": float(a[t]) if t in a else 0.0,
        "coordinate_system": "RAW",
    }


def _absolute(a: Dict[str, Any]) -> dict:
    missing = [axis for axis in "xyzrt" if axis not in a]
    if missing:
        raise KeyError(", ".join(missing))
    return _position(a)


def _beam(a: Dict[str, Any], **more: Any) -> dict:
    """An imaging command's kwargs: its beam_type, checked, and *more* when set."""
    beam = str(a.get("beam_type", "ELECTRON")).upper()
    if beam not in ("ELECTRON", "ION"):
        raise ValueError(f"beam_type must be ELECTRON or ION, not {beam!r}")
    if more.get("image_settings") is not None:
        more["image_settings"] = dict(more["image_settings"], beam_type=beam)
    return {"beam_type": beam, **{k: v for k, v in more.items() if v is not None}}


def _kwargs(**values: Any) -> dict:
    return {"kwargs": values}


def _preview(kwargs: dict) -> dict:
    """A generic command's body: an image it returns comes back as the agent's
    answer (summary, saved path, preview), not as data it can't read."""
    return {"kwargs": kwargs, "answer": "preview"}


def _stop(body: dict, a: Dict[str, Any]) -> dict:
    """A command's body with the agent's reason, which every stop carries."""
    if a.get("reason") is not None:
        body["reason"] = str(a["reason"])
    return body


# Tools whose arguments are not their route's path and body as they stand: each
# gives (path values, body).
_CALLS: Dict[str, Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Optional[dict]]]] = {
    "acquire_image_preview": lambda a: (
        {},
        {
            "kwargs": _beam(a, image_settings=a.get("image_settings")),
            "answer": "preview",
        },
    ),
    "autocontrast": lambda a: ({}, _kwargs(**_beam(a))),
    "auto_focus": lambda a: ({}, _kwargs(**_beam(a))),
    # A service command or query with no arguments still sends its kwargs.
    "stop_milling": lambda a: ({}, _stop(_kwargs(), a)),
    "estimate_milling_time": lambda a: ({}, _kwargs()),
    "get_pattern_preview": lambda a: (
        {},
        _kwargs(image_path=str(a["image_path"]), stages=a.get("stages")),
    ),
    "get_stage_orientation": lambda a: ({}, _kwargs()),
    "get_milling_angle": lambda a: ({}, _kwargs()),
    "get_geometry": lambda a: ({}, _kwargs()),
    "stable_move": lambda a: (
        {},
        _kwargs(
            dx=float(a["dx"]),
            dy=float(a["dy"]),
            beam_type=str(a.get("beam_type", "ELECTRON")),
        ),
    ),
    "vertical_move": lambda a: (
        {},
        _kwargs(
            dy=float(a["dy"]),
            dx=float(a.get("dx", 0.0)),
            beam_type=str(a.get("beam_type") or "ION"),
            relaxation=float(a.get("relaxation") or 1.0),
        ),
    ),
    "move_to_orientation": lambda a: ({}, _kwargs(orientation=str(a["orientation"]))),
    "safe_move_stage_absolute": lambda a: ({}, _kwargs(stage_position=_absolute(a))),
    "move_stage_relative": lambda a: (
        {},
        _kwargs(delta=_position(a, x="dx", y="dy", z="dz", r="", t="")),
    ),
    "move_stage_absolute": lambda a: ({}, _kwargs(position=_absolute(a))),
    "move_to_milling_angle": lambda a: (
        {},
        _kwargs(
            milling_angle=float(a["milling_angle_deg"]),
            rotation=(
                None if a.get("rotation_deg") is None else float(a["rotation_deg"])
            ),
        ),
    ),
    "call_command": lambda a: (
        {"device": a.get("device"), "command": a.get("command")},
        _stop(_preview(dict(a.get("kwargs") or {})), a),
    ),
    "call_service_command": lambda a: (
        {"service": a.get("service"), "command": a.get("command")},
        _stop(_preview(dict(a.get("kwargs") or {})), a),
    ),
    "call_service_query": lambda a: (
        {"service": a.get("service"), "query": a.get("query")},
        _preview(dict(a.get("kwargs") or {})),
    ),
}


def route_call(
    spec: ToolSpec, arguments: Dict[str, Any]
) -> Tuple[str, str, Optional[dict]]:
    """The route call (method, path, body) a tool call stands for: the path's
    parameters from the arguments, the rest as the JSON body. Raises ValueError
    for a missing path parameter or a malformed argument."""
    args = dict(arguments)
    shape = _CALLS.get(spec.name)
    if shape is not None:
        try:
            args, body = shape(args)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{spec.name}: bad or missing argument {exc}") from None
    names = _PATH_PARAM.findall(spec.path)
    missing = [n for n in names if not isinstance(args.get(n), str) or not args[n]]
    if missing:
        raise ValueError(f"{spec.name} needs {', '.join(missing)} (strings)")
    path = spec.path
    for name in names:
        path = path.replace("{" + name + "}", args.pop(name))
    if shape is None:
        body = args
    return spec.method, path, body or None
