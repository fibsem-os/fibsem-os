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
    # The level note, where the level alone does not say it (request_create).
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
            "request_create and the operator approves it (a direct call is "
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
        name="server_get_capabilities",
        description="Server capabilities: API version, which halves are served (routers.microscope, routers.app), and the agent's level (off / watch / assist / drive). Read it first: the level says what the other tools will do. At Assist, act by asking: request_create.",
        method="GET",
        path="/capabilities",
        level="watch",
        router="base",
    ),
    ToolSpec(
        name="microscope_get_info",
        description="The microscope: manufacturer, model, serial number and software version, which optional hardware is fitted, and its system settings.",
        method="GET",
        path="/microscope",
        level="watch",
    ),
    ToolSpec(
        name="stage_get_position",
        description="Current stage position (x/y/z in metres, r/t in radians).",
        method="GET",
        path="/microscope/devices/stage/position",
        level="watch",
    ),
    ToolSpec(
        name="stage_get_orientation",
        description="Named stage orientation (SEM, FIB, MILLING, FM or NONE).",
        method="POST",
        path="/microscope/services/stage_movement/queries/stage_orientation",
        level="watch",
    ),
    ToolSpec(
        name="microscope_get_state",
        description="Full microscope state snapshot (beams, detectors, stage).",
        method="GET",
        path="/microscope/state",
        level="watch",
    ),
    ToolSpec(
        name="device_list",
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
        name="device_get_parameter",
        description="Read one device parameter live, e.g. device=ion parameter=current.",
        method="GET",
        path="/microscope/devices/{device}/{parameter}",
        level="watch",
        params={
            "device": "device name from device_list",
            "parameter": "parameter name",
        },
    ),
    ToolSpec(
        name="stage_get_milling_angle",
        description="Current milling angle in degrees: the angle between the ion beam and the sample surface at the stage's rotation and tilt (90 at the FIB pose).",
        method="POST",
        path="/microscope/services/stage_movement/queries/current_milling_angle",
        level="watch",
    ),
    ToolSpec(
        name="stage_get_geometry",
        description="The instrument's geometry in one read: column tilts, pre-tilt, rotation reference and centre, the orientation table, device origins, and what they mean at the stage's pose now: which way raw +z goes in the chamber, the milling angle here and the pose for the stored one, and per beam the stage move a +1 µm stable move along each image axis makes at the current scan rotation. Read it first on a new instrument, then confirm once with one small move. Angles in degrees.",
        method="POST",
        path="/microscope/services/stage_movement/queries/geometry",
        level="watch",
    ),
    ToolSpec(
        name="milling_get_state",
        description="Milling state (IDLE, RUNNING, PAUSED, ...).",
        method="GET",
        path="/microscope/services/milling/state",
        level="watch",
    ),
    ToolSpec(
        name="milling_estimate_time",
        description="Estimated time in seconds for the currently drawn patterns.",
        method="POST",
        path="/microscope/services/milling/queries/estimate",
        level="watch",
    ),
    ToolSpec(
        name="milling_preview_patterns",
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
        name="alignment_align_to_reference",
        description=(
            "Bring the view back onto a saved reference image: acquire with its "
            "settings, measure the shift and correct it, steps times, with the "
            "beam shift (fine, small range), the stage (stable move) or "
            "stage-vertical. reference_path must be inside the open experiment "
            "(a path from item_list_outputs or an image you saved). A long command: "
            "it answers with its run at once; service_wait_for_run alignment gives each "
            "step's shift, whether the methods agree on what is left, and the "
            "folder the run saved to. alignment_measure_shift first if you only want to know "
            "how far off the view is."
        ),
        method="POST",
        path="/microscope/services/alignment/commands/align",
        level="drive",
        hardware=True,
        params={
            "reference_path": "a saved image inside the open experiment",
            "steps": "how many measure-and-correct steps (default 3)",
            "subsystem": "beam-shift (default), stage or stage-vertical",
        },
    ),
    ToolSpec(
        name="alignment_measure_shift",
        description=(
            "The shift of one saved image from another, as alignment_align_to_reference "
            "measures it: metres and pixels, a score (higher is better) and "
            "whether it cleared the minimum. Both paths inside the open "
            "experiment. Reads the files and moves nothing."
        ),
        method="POST",
        path="/microscope/services/alignment/queries/measure_shift",
        level="watch",
        params={
            "reference_path": "the saved reference image",
            "image_path": "the saved image to measure against it",
        },
    ),
    ToolSpec(
        name="milling_stop",
        description="Stop an in-progress mill, with the reason the operator reads: the whole milling task, so no later stage starts. Refused 409 NothingToStop when nothing is milling.",
        method="POST",
        path="/microscope/services/milling/commands/stop",
        level="drive",
        params={"reason": "why you are stopping it, for the operator (required)"},
        note="A mill the agent started itself it can stop at Assist or Drive, never at Watch; anyone else's (the workflow's, the operator's) needs Drive, and at Assist goes in request_create for the operator to approve. Never refused for someone else being in control of the microscope.",
    ),
    # --- hardware ---
    ToolSpec(
        name="beam_acquire_image",
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
        name="stage_stable_move",
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
            "too. Prefer it to stage_move_relative whenever the move comes from an "
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
        name="stage_vertical_move",
        description=(
            "Restore coincidence: with a feature centred in one beam's image, move "
            "the stage along the chamber vertical so the same feature comes to the "
            "centre of the other beam's image; the first beam sees no change. "
            "beam_type names the view the offset was measured in: ION (the "
            "default) when the feature is centred in the electron image, ELECTRON "
            "when it is centred in the ion image. dy (and dx) are the feature's "
            "offset from that image's centre, in metres, x right and y up. "
            "Use it after a stage_stable_move in the other beam; never a plain z move."
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
        name="stage_move_to_orientation",
        description=(
            "Move the stage to a named orientation (SEM, FIB, MILLING, and FM "
            "where fitted): its rotation and tilt, by the safe move "
            "(stage_move_absolute). x, y and z stay, except that a half turn "
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
        name="stage_move_absolute",
        description=(
            "Move the stage to an absolute position (metres and radians) safely: "
            "for a large rotation it tilts flat first, and it rotates before "
            "moving the other axes, so a tilted sample isn't swung round. Use "
            "it to return to a saved position."
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
        name="stage_move_relative",
        description=(
            "Move the stage axes by a raw offset (metres), in stage coordinates. "
            "It knows nothing of the image: at any tilt a y move also changes "
            "the sample's height, so the feature drifts out of focus and out of "
            "the other beam. To move to something seen in an image, use "
            "stage_stable_move instead."
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
        name="stage_move_to_milling_angle",
        description=(
            "Tilt the stage to a milling angle, in degrees. The stage stays at its "
            "current rotation unless rotation_deg names another."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/move_to_milling_angle",
        level="drive",
        hardware=True,
        params={
            "milling_angle_deg": "target milling angle in degrees",
            "rotation_deg": "stage rotation in degrees; the current rotation if left out",
        },
    ),
    ToolSpec(
        name="device_set_parameter",
        description=(
            "Set one device parameter (SI units); the server checks it against the "
            "parameter's limits and choices and answers the value written."
        ),
        method="PUT",
        path="/microscope/devices/{device}/{parameter}",
        level="drive",
        hardware=True,
        params={
            "device": "device name from device_list",
            "parameter": "parameter name",
            "value": "the new value",
        },
    ),
    ToolSpec(
        name="device_call_command",
        description="Run a device command listed by device_list, e.g. device=ion command=blank.",
        method="POST",
        path="/microscope/devices/{device}/commands/{command}",
        level="drive",
        hardware=True,
        params={
            "device": "device name from device_list",
            "command": "command name",
            "kwargs": "the command's arguments, by name",
            "reason": "for a stop (stop, stop_live): why, for the operator (required there)",
        },
    ),
    ToolSpec(
        name="service_list",
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
        name="service_get_parameter",
        description=(
            "Read one service parameter live, e.g. service=milling parameter=progress."
        ),
        method="GET",
        path="/microscope/services/{service}/{parameter}",
        level="watch",
        params={
            "service": "service name from service_list",
            "parameter": "parameter name",
        },
    ),
    ToolSpec(
        name="service_set_parameter",
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
            "service": "service name from service_list",
            "parameter": "parameter name",
            "value": "the new value",
        },
    ),
    ToolSpec(
        name="service_call_command",
        description=(
            "Run a service command listed by service_list, e.g. service=stage_movement "
            "command=move_to_orientation kwargs={'orientation': 'SEM'}. A long command "
            "(listed with long: true: milling run, run_stage and run_task, spot burn "
            "run, alignment align, imaging fm_focus_sweep, tiled_overview, "
            "fm_tiled_overview and focus_stack) answers at once with "
            "its run (run_id, state: running) and runs on, in control of the "
            "microscope; service_wait_for_run says how it ended. command=stop from "
            "another call stops it: the agent's own run at Assist or Drive, "
            "anyone else's at Drive, never at Watch, always with a reason."
        ),
        method="POST",
        path="/microscope/services/{service}/commands/{command}",
        level="drive",
        hardware=True,
        params={
            "service": "service name from service_list",
            "command": "command name",
            "kwargs": "the command's arguments, by name",
            "reason": "for a stop (stop, stop_live): why, for the operator (required there)",
        },
    ),
    ToolSpec(
        name="service_wait_for_run",
        description=(
            "How a long command you started (a milling or spot burn run, an "
            "alignment, an FM auto-focus, an overview or a focus stack) is going: "
            "its run with state running, done, failed or cancelled, and when it has "
            "ended its result or error. Waits up to wait seconds (at most 60) for "
            "it to end, so call it again while it says running. With run_id, waits "
            "for that run; a later run on the service answers at once. The same end "
            "comes as a command_ended event on app_get_events."
        ),
        method="GET",
        path="/microscope/services/{service}/run",
        level="watch",
        params={
            "service": "required: the service the command ran on, e.g. milling, imaging or alignment",
            "run_id": "the run_id the command answered with",
            "wait": "seconds to wait for the end, at most 60 (default 60)",
        },
    ),
    ToolSpec(
        name="service_call_query",
        description=(
            "Ask a service query listed by service_list: it takes arguments, "
            "answers, and moves nothing, e.g. service=stage_movement "
            "query=project_stable_move kwargs={'dx': 1e-6, 'dy': 0, 'beam_type': 1, "
            "'base_position': {...}} for where a stable move from base_position "
            "would end (beam_type 1 is the electron beam, 2 the ion beam)."
        ),
        method="POST",
        path="/microscope/services/{service}/queries/{query}",
        level="watch",
        params={
            "service": "service name from service_list",
            "query": "query name",
            "kwargs": "the query's arguments, by name",
        },
    ),
    ToolSpec(
        name="beam_auto_contrast",
        description="Run the instrument's own autocontrast routine for a beam (changes detector settings). beam_auto_brightness_contrast is the experimental image-based one.",
        method="POST",
        path="/microscope/services/imaging/commands/autocontrast",
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
    ),
    ToolSpec(
        name="beam_auto_focus",
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
    ToolSpec(
        name="beam_acquire_overview",
        description=(
            "A tiled SEM or FIB overview centred where the stage is: tiles of "
            "image_settings (hfw is one tile's field of view, beam_type the beam) "
            "in nrows x ncols, stitched into one image. The stage moves to each "
            "tile and comes back after. path must be a folder inside the open "
            "experiment. beam_estimate_overview first: a big grid at a long dwell runs "
            "for hours. A long command: it answers with its run at once; "
            "service_wait_for_run imaging gives the stitched image (summary, path, "
            "preview), the tiles' paths and the layout (the layout check's warning, "
            "and whether register_tiles moved the tiles and why); "
            "service_call_command imaging stop, "
            "with a reason, ends it before the next tile."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/tiled_overview",
        level="drive",
        hardware=True,
        params={
            "settings": (
                "OverviewAcquisitionSettings dict: image_settings, nrows, ncols, "
                "overlap, autofocus_mode, autocontrast_mode, focus_stack_settings, "
                "register_tiles (experimental: place each tile where its picture "
                "says it is, not where the stage step put it; default false)"
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": "the file name, stamped with the time (default overview)",
        },
    ),
    ToolSpec(
        name="beam_estimate_overview",
        description=(
            "What beam_acquire_overview with these settings would do: how many tiles, "
            "the beam-on scan time in seconds and the field of view in metres. "
            "Not the run's length: the stage moves between tiles add to it, "
            "often more than the scanning. Moves nothing."
        ),
        method="POST",
        path="/microscope/services/imaging/queries/estimate_tiled_overview",
        level="watch",
        params={
            "settings": "OverviewAcquisitionSettings dict, as beam_acquire_overview"
        },
    ),
    ToolSpec(
        name="beam_acquire_focus_stack",
        description=(
            "One image focused in strips: the frame is split into n_steps "
            "horizontal strips, each focused (beam_auto_focus) and acquired on its own, "
            "and stacked into one image. For a tilted view where one focus "
            "can't hold the whole frame. Moves the working distance. path must "
            "be a folder inside the open experiment. A long command: it answers "
            "with its run at once; service_wait_for_run imaging gives the image "
            "(summary, path, preview); service_call_command imaging stop, with "
            "a reason, ends it before the next strip."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/focus_stack",
        level="drive",
        hardware=True,
        params={
            "image_settings": "ImageSettings dict: beam_type, hfw, resolution, dwell_time, filename",
            "path": "a folder inside the open experiment to save it in",
            "n_steps": "how many strips (default 3)",
            "auto_focus": "focus each strip first (default true)",
        },
    ),
    ToolSpec(
        name="fm_auto_focus",
        description=(
            "Focus the fluorescence microscope: sweep the objective, a coarse then "
            "a fine pass, scoring a frame at each step on the current channel, as "
            "the FM tab's auto-focus does, and leave it at the best position. "
            "Refused while the FM is in use or can't see the sample from this "
            "pose (move to the FM pose first). A long command: it answers with "
            "its run at once; service_wait_for_run imaging gives the best objective "
            "position and the scores; service_call_command imaging stop, with a "
            "reason, ends it."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/fm_focus_sweep",
        level="drive",
        hardware=True,
        params={
            "channel_name": "the channel to focus on; left out, the current one",
            "path": "a folder inside the open experiment to save the sweep to",
        },
    ),
    ToolSpec(
        name="fm_acquire_overview",
        description=(
            "A tiled fluorescence overview centred where the stage is: rows x cols "
            "camera fields in each of the channels, stitched into one mosaic. The "
            "stage moves to each tile, and it and the objective come back after. "
            "The objective must be in, at the FM pose; refused while the FM is in "
            "use. path must be a folder inside the open experiment. "
            "fm_estimate_overview first: tiles take seconds to minutes each. A "
            "long command: it answers with its run at once; service_wait_for_run imaging "
            "gives the mosaic's path and summary, the tiles' paths and the layout, "
            "as beam_acquire_overview's; "
            "service_call_command imaging stop, with a reason, ends it before the "
            "next tile."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/fm_tiled_overview",
        level="drive",
        hardware=True,
        params={
            "channels": (
                "list of ChannelSettings dicts: name, excitation_wavelength, "
                "emission_wavelength, power, exposure_time"
            ),
            "parameters": (
                "OverviewParameters dict: rows, cols, overlap, autofocus_mode "
                "(none, once, each_row, each_tile), use_zstack, register_tiles "
                "(experimental, as beam_acquire_overview's; default false)"
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": "the file name, stamped with the time (default overview)",
            "zparams": "ZParameters dict (zmin, zmax, zstep), with use_zstack",
        },
    ),
    ToolSpec(
        name="fm_estimate_overview",
        description=(
            "What fm_acquire_overview with these arguments would do: how many "
            "tiles and images, and how long it runs in seconds, stage and "
            "objective moves and focus sweeps counted. Moves nothing."
        ),
        method="POST",
        path="/microscope/services/imaging/queries/estimate_fm_tiled_overview",
        level="watch",
        params={
            "channels": "list of ChannelSettings dicts, as fm_acquire_overview",
            "parameters": "OverviewParameters dict, as fm_acquire_overview",
            "zparams": "ZParameters dict, as fm_acquire_overview",
        },
    ),
    ToolSpec(
        name="beam_auto_brightness_contrast",
        description=(
            "Experimental: set a beam's detector brightness and contrast from its "
            "own images, not the instrument's routine (beam_auto_contrast): at most 6 images of the centre of the "
            "field of view, starting from what last worked at this beam current. "
            "Answers the brightness and contrast it set, whether it converged, "
            "and each image's clipped fractions. Fails, with the detector at the "
            "best it found, when every image was mostly black, mostly white or "
            "flat; stop before aligning or milling on that beam. Changes detector "
            "settings."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/auto_brightness_contrast",
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
    ),
)


APP_TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="app_get_status",
        description="What the application is doing: experiment, running workflow, current task and item.",
        method="GET",
        path="/app/workflow/status",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="workflow_get_queue",
        description="The live work queue: id-anchored items with status, plus the mutation version.",
        method="GET",
        path="/app/workflow/queue",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="item_list",
        description="Per-item summary of the open experiment.",
        method="GET",
        path="/app/experiment",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="experiment_get_task_history",
        description="Every task run so far: outcomes, durations, errors. Oldest first, by start time.",
        method="GET",
        path="/app/experiment/task_history",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="workflow_get_run_summary",
        description="The most recent workflow run's outcome table.",
        method="GET",
        path="/app/workflow/run_summary",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="app_get_log",
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
        name="protocol_get",
        description="The workflow definition with live supervision flags and schedules. Each task carries `attendance`: whether a person must be at the microscope while it runs and what waits for their decision afterwards, under its current attention.",
        method="GET",
        path="/app/experiment/protocol",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="item_get",
        description="Everything durable about one item: status, failure flag, POI, alignment area, milling angle, and where its poses put the stage.",
        method="GET",
        path="/app/experiment/items/{item_name}",
        level="watch",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="item_list_outputs",
        description="The files an item's task runs produced (reference images by role). tasks has every run with its status; completed_tasks names only the runs that completed.",
        method="GET",
        path="/app/experiment/items/{item_name}/outputs",
        level="watch",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="item_get_output_image",
        description="One saved image of an item, as an image you can look at: a 768 px JPEG preview of a file item_list_outputs lists (stacks and channels max-projected). A preview, not the data. A beam image comes with its metadata block (pixel size, preview_pixelsize, stage position, beam, detector). An unlisted name is refused with the valid ones.",
        method="GET",
        path="/app/experiment/items/{item_name}/outputs/{filename}",
        level="watch",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "filename": "a basename from item_list_outputs",
        },
    ),
    ToolSpec(
        name="experiment_get_image_metadata",
        description="The full saved metadata of one image file in the experiment folder: the image settings, pixel size, microscope state (stage, both beams, detectors) and the item and task it was taken for. The image tools already carry a short block of this; use this for the rest. Paths outside the experiment are refused.",
        method="GET",
        path="/app/experiment/images/metadata",
        level="watch",
        router="app",
        params={
            "path": "an image file path inside the experiment folder, e.g. the path an image answer gives"
        },
    ),
    ToolSpec(
        name="grid_list",
        description="Every grid the experiment records: quality verdict, task status, completed tasks, linked item count, and the basename of its latest SEM/FIB/FM overview by role. Records only — not the live slot inventory.",
        method="GET",
        path="/app/experiment/grids",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="grid_get",
        description="One grid: its summary plus every task run with the files it recorded by role, and the names of the items linked to it.",
        method="GET",
        path="/app/experiment/grids/{grid_name}",
        level="watch",
        router="app",
        params={"grid_name": "the grid name"},
    ),
    ToolSpec(
        name="grid_get_output_image",
        description="One saved overview (mosaic) of a grid, as an image you can look at: a 768 px JPEG preview of a file grid_list / grid_get lists. A preview, not the data; grid_get_markers places the items on it. A beam image comes with its metadata block (pixel size, preview_pixelsize, stage position, beam, detector). An unlisted name is refused with the valid ones.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}",
        level="watch",
        router="app",
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from grid_list / grid_get",
        },
    ),
    ToolSpec(
        name="grid_get_stage_position",
        description="The stage position under a point of a recorded grid overview: the inverse of grid_get_markers, from the overview's own metadata, so it holds wherever the stage is now. Give x, y in source pixels (as the markers use), or fx, fy from 0 to 1 for a point read off a preview. Nothing moves.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}/stage_position",
        level="watch",
        router="app",
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from grid_list / grid_get",
            "x": "source-pixel x (with y)",
            "y": "source-pixel y, +y down (with x)",
            "fx": "fraction across the image, 0-1 (with fy)",
            "fy": "fraction down the image, 0-1 (with fx)",
        },
    ),
    ToolSpec(
        name="grid_get_markers",
        description="Where the experiment's items fall on one recorded overview, in source-image pixels with each item's status, plus the image's scale block (size, pixel size, field width, stage position). Computed server-side; never re-derive stage geometry. Items that cannot be placed are listed with the reason.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}/markers",
        level="watch",
        router="app",
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from grid_list / grid_get",
        },
    ),
    ToolSpec(
        name="protocol_get_task_config",
        description="One task's protocol-level configuration document (the defaults new items copy), with a version token naming exactly this state of it. Full document, not a summary.",
        method="GET",
        path="/app/experiment/protocol/task_config/{task_name}",
        level="watch",
        router="app",
        params={"task_name": "the task name, as listed by protocol_get"},
    ),
    ToolSpec(
        name="item_get_task_config",
        description="One item's own copy of a task configuration (what its run actually executes — tasks may have rewritten it mid-run), with a version token naming exactly this state of it.",
        method="GET",
        path="/app/experiment/items/{item_name}/task_config/{task_name}",
        level="watch",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name, as listed by protocol_get",
        },
    ),
    ToolSpec(
        name="experiment_add_note",
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
        name="request_create",
        description="At Assist, ask the operator to approve one or more actions: each is {tool, arguments}, the tool you would call and the arguments you would give it, run in order once approved. Give the reason the operator reads before deciding. One request waits at a time (409 request_pending names it): to change it, withdraw it (request_withdraw) and file again. Answers the request, status pending; follow it with request_get or the request_* events (request_decided, request_applied per action, request_finished). Approved actions run through their own routes, so each is checked and can be refused as a direct call would be; the first refusal stops the batch. A rejection carries the operator's reason. An action the agent may never use (agent: false in device_list) is refused at filing (403 not_allowed). Changing the level or Stop agent cancels a waiting request. Reads need no request, nor does stopping your own mill (milling_stop): call those directly. Stopping anything else (workflow_stop, someone else's mill) goes in a request.",
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
        name="request_withdraw",
        description='Withdraw a request you filed that is still waiting for the operator, with a reason: the plan changed, or the operator asked for something else. It ends cancelled, "withdrawn by the agent", and a new request can be filed. A request the operator has decided can no longer be withdrawn (409 not_pending).',
        method="POST",
        path="/app/agent/requests/{request_id}/withdraw",
        level="watch",
        router="app",
        params={
            "request_id": "the request_id request_create answered",
            "reason": "why you are withdrawing it, for the operator",
        },
        note="Works at any level the agent is on (a request only waits at Assist).",
    ),
    ToolSpec(
        name="request_get",
        description="One request you filed with request_create: its status (pending, applying, applied, failed, rejected, cancelled), who decided it, the operator's reason on a reject, why it was cancelled, and each action's result or the route's error.",
        method="GET",
        path="/app/agent/requests/{request_id}",
        level="watch",
        router="app",
        params={"request_id": "the request_id request_create answered"},
    ),
    ToolSpec(
        name="request_list",
        description="Every request filed since the app started, newest first, each as request_get gives it: the way back to a request that is still waiting after you reconnect, and to the ids request_get and request_withdraw take.",
        method="GET",
        path="/app/agent/requests",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="item_apply_protocol",
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
        name="item_add",
        description="Add a lamella, as the Lamella tab's Add does: at the stage position now (no where; refused while a workflow runs, since the run moves the stage), at stage_position, or under a point of a recorded grid overview (overview = {grid_name, filename, px: [x, y] source pixels or fraction: [x, y] 0-1}), which also puts it on that grid. Its poses, milling angle and grid are worked out as for a click, and it is saved. The app names it unless name is given. Answers with the new item.",
        method="POST",
        path="/app/experiment/items",
        level="drive",
        router="app",
        params={
            "name": "optional name; the app picks one when omitted",
            "stage_position": "optional {x, y, z, r, t} (metres, radians) to add at",
            "overview": "optional {grid_name, filename, px | fraction}: a point on a grid overview",
        },
    ),
    ToolSpec(
        name="item_remove",
        description="Remove a lamella from the experiment, as its row's Remove does. Its files stay on disk. Refused while a workflow runs.",
        method="DELETE",
        path="/app/experiment/items/{item_name}",
        level="drive",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="item_move_to_pose",
        description="Move the stage to one of a lamella's poses (MILLING, FLUORESCENCE, ... as item_get lists), through the safe move that goes flat before a large rotation. Answers with the stage position after.",
        method="POST",
        path="/app/experiment/items/{item_name}/move_to",
        level="drive",
        router="app",
        hardware=True,
        params={"item_name": "the item (lamella) name", "pose": "the pose name"},
    ),
    ToolSpec(
        name="item_set_pose",
        description="Set the stage position now as one of a lamella's poses, as the tab's Set pose does. Setting MILLING moves the lamella: its milling angle and fluorescence pose follow. Saved; answers with the item.",
        method="POST",
        path="/app/experiment/items/{item_name}/poses/{pose_name}",
        level="drive",
        router="app",
        params={"item_name": "the item (lamella) name", "pose_name": "the pose name"},
    ),
    ToolSpec(
        name="item_set_objective_position",
        description="Store the FM objective's position now on a lamella's fluorescence pose, as the tab's Use current does: where it is when inserted, else its focus position. Saved.",
        method="POST",
        path="/app/experiment/items/{item_name}/objective_position",
        level="drive",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="item_reorder_milling_stages",
        description="Reorder the milling stages inside one of an item's task configs ('run the stress relief first'). order must be exactly the current stage names in their new sequence — this can never add, drop, or duplicate a stage. Echo the version from item_get_task_config; stale is refused.",
        method="POST",
        path="/app/experiment/items/{item_name}/task_config/{task_name}/stages/reorder",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name",
            "milling_key": "which milling config within the task (e.g. mill_rough)",
            "order": "the current stage names in their new sequence",
            "version": "the version token from item_get_task_config",
        },
    ),
    ToolSpec(
        name="protocol_reorder_milling_stages",
        description="Reorder the milling stages inside a protocol-level task config. Same rules as the item form; reaches items created (or re-applied) after.",
        method="POST",
        path="/app/experiment/protocol/task_config/{task_name}/stages/reorder",
        level="drive",
        router="app",
        params={
            "task_name": "the task name",
            "milling_key": "which milling config within the task",
            "order": "the current stage names in their new sequence",
            "version": "the version token from protocol_get_task_config",
        },
    ),
    ToolSpec(
        name="protocol_set_schedule",
        description="Set (or clear) when a task may start: ISO-8601 timestamp (naive = instrument-local time), or null to clear. The workflow reads schedules at each task start, so a change takes effect at the next start; the schedule persists with the protocol.",
        method="POST",
        path="/app/workflow/schedule",
        level="drive",
        router="app",
        params={
            "task_name": "the task name, as listed by protocol_get",
            "scheduled_at": "ISO-8601 timestamp, or null to clear",
        },
    ),
    ToolSpec(
        name="workflow_set_options",
        description="Set whether the run summary dialog opens when a run ends (the workflow's \"Show summary after completion\"). Off still toasts each run's end with a Show summary link. Turn it off before starting runs yourself, so no dialog stands over the operator's window, and tell the operator. Persists with the protocol; protocol_get's options show the current value.",
        method="POST",
        path="/app/workflow/options",
        level="drive",
        router="app",
        params={
            "show_run_summary": "true to open the summary dialog at a run's end, false for the toast only",
        },
    ),
    ToolSpec(
        name="item_update",
        description="Patch an item's own document with dotted-path edits: poi.x/poi.y (metres, milling frame), alignment_area.left/top/width/height (frame fractions), description, defect.verdict (UNASSESSED/GOOD/REWORK/FAILED) and defect.reason. milling_angle is read-only: it is an outcome of the Setup task, not an input. Echo the version from item_get — a mismatch is refused (409 stale_config). Tasks re-record geometry at their own moments, so an edit here is what the NEXT run starts from, not a permanent override.",
        method="POST",
        path="/app/experiment/items/{item_name}",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from item_get",
        },
    ),
    ToolSpec(
        name="protocol_update_task_config",
        description="Patch a task's protocol-level defaults (what new items copy) with dotted-path edits. Echo the version from protocol_get_task_config — a mismatch is refused (409 stale_config). Running tasks are never affected: they hold their item's copy; the edit reaches items created after.",
        method="POST",
        path="/app/experiment/protocol/task_config/{task_name}",
        level="drive",
        router="app",
        params={
            "task_name": "the task name, as listed by protocol_get",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from protocol_get_task_config",
        },
    ),
    ToolSpec(
        name="item_update_task_config",
        description="Patch an item's task config with dotted-path edits (e.g. {'milling.mill_rough.stages.0.pattern.depth': 2e-6}). Echo the version from item_get_task_config — a mismatch is refused (409 stale_config): re-read and re-patch. A task currently running for that item is refused (it already copied its config); pending tasks pick the change up at start. Values are validated against the live field's type and declared bounds; the response echoes every change old -> new.",
        method="POST",
        path="/app/experiment/items/{item_name}/task_config/{task_name}",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name, as listed by protocol_get",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from item_get_task_config",
        },
    ),
    ToolSpec(
        name="app_get_events",
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
        name="workflow_stop",
        description="Stop the running workflow — the same path as the GUI's Stop button; the mill halts too. Give the reason the operator reads.",
        method="POST",
        path="/app/workflow/stop",
        level="drive",
        router="app",
        params={"reason": "why you are stopping it, for the operator (required)"},
        note="Needs the agent at Drive: the workflow is never the agent's own. At Assist, put it in request_create and the operator approves it (a direct call is refused, 403 ask_first); the operator can also press Stop. Refused (403) at Watch. Never refused for someone else being in control of the microscope.",
    ),
    ToolSpec(
        name="workflow_start",
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
        name="workflow_plan_grids",
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
        name="workflow_start_grids",
        description="Start a grid run — the Grids view's Run button, remotely; with screen_all, its Screen all grids button (inventory, then every present grid). Refused while any workflow, lamella or grid, is running. Call workflow_plan_grids first and show the plan.",
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
        name="protocol_set_supervision",
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
        name="workflow_requeue",
        description="Queue a task for an item (again) in the RUNNING workflow — 'run 03's fiducial again'. Re-running a completed pair is the queue's own re-run mechanism; the queue only exists during a run. A pair already waiting in the queue is refused (409 already_queued): read the queue (workflow_get_queue) before asking for a requeue.",
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
        name="app_get_display_images",
        description="The SEM and FIB images the app's GUI is displaying right now, as previews — the display cache, not a new acquisition. Post-mill images appear here as soon as the GUI shows them, before they reach disk.",
        method="GET",
        path="/app/experiment/images",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="prompt_get",
        description='The supervision question awaiting an answer, if any — message, options, and context images. A question that is also on the item\'s record says answer_via: "decide" and gives the item_id, task_name and proposal_id: it is decided with review_decide, like any other review, and prompt_answer will refuse it.',
        method="GET",
        path="/app/workflow/prompt",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="review_list",
        description="Every proposal waiting for a decision (the Review tab's inbox, `reviews`): which item and task, its own id (proposal_id, which review_decide passes back) and the run it is from (task_id), whether the task that made it is stopped on the answer (holding_the_run: nothing else runs until it is decided, so decide those first), the proposed values (a point of interest in metres, milling frame), confidence and alternatives when the proposer gave any, provenance, its review mode, which tasks are waiting on the decision, and the reference image the values sit on as a JPEG preview. `to_check` lists proposals a producer applied itself (advise mode) that nobody has looked at yet: the run did not wait on them; acknowledge one with review_decide Confirmed and no values, which records the look and writes nothing. Empty lists when nothing is pending.",
        method="GET",
        path="/app/workflow/reviews",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="review_decide",
        description="Confirm or reject a proposal, exactly as the Review tab would. Pass the proposal_id review_list gave for it: the decision is on that proposal, and is refused (409 stale_review) if it is no longer the one on the record -- the task re-ran, or asked again in the same run -- so re-read and decide the current one. (task_id, the run, is accepted in its place but cannot tell two questions from one run apart.) Confirming a pending proposal needs all its values (e.g. {'poi': {'x','y'}} metres, milling frame), as proposed or adjusted; they are written through, the consumer becomes runnable, and the delta against the proposal is computed, so pass the values you actually judge right. Confirming a to_check proposal (already applied) acknowledges it and must carry no values; to change an applied value, re-run the task. A wrong, missing or foreign value is refused (422 invalid_value) with nothing written. Rejected needs a reason and fails the task that was waiting on the decision, so nothing that requires it runs; the lamella itself is not marked defective. Refused (409) when the item has a task running -- stop it instead -- unless that task is the one stopped on this very proposal (holding_the_run), which a decision releases: Confirmed hands the values back and the task carries on, Rejected fails it. Also refused when there is no proposal for that item and task. The decision is recorded with the agent as author.",
        method="POST",
        path="/app/workflow/decide",
        level="drive",
        router="app",
        params={
            "item_id": "the item id from review_list (by id, never name)",
            "task_name": "the task name from review_list",
            "proposal_id": "the proposal_id from review_list: the proposal you looked at",
            "task_id": "optional, and only in place of proposal_id: the run the proposal is from",
            "outcome": "Confirmed or Rejected",
            "values": "on Confirmed of a pending proposal: every value its kind carries, as proposed or adjusted; omitted (or empty) when acknowledging a to_check one",
            "reason": "required on Rejected",
            "author": "optional agent/model name, recorded as agent:<name>",
        },
    ),
    ToolSpec(
        name="prompt_answer",
        description='Answer the pending supervision question, exactly as clicking the matching button would. Not for a question whose prompt says answer_via: "decide" -- that one is refused here, with nothing clicked, and decided with review_decide. True = the positive option. Echo the question_id from prompt_get; if that question is no longer pending the answer is refused (409 stale_prompt) — re-read and answer the current one. For EditAlignmentArea, PickPOI, ConfirmDetection and RunSpotBurn the answer may carry a value: the edit the operator could make in that widget, placed into it (the operator sees it land) and then answered through the same click path. For RunSpotBurn, yes burns the points as shown and asks again, no continues. An out-of-bounds or wrong-shape value is refused (422 invalid_value) without clicking anything.',
        method="POST",
        path="/app/workflow/prompt/answer",
        level="drive",
        router="app",
        params={
            "response": "true for the positive option, false for the negative",
            "question_id": "the question_id from prompt_get, naming the question being answered",
            "value": "optional edit: {left, top, width, height} (fractions of the frame, origin top-left) for EditAlignmentArea; {x, y} (metres, microscope image coordinates, origin centre, +y up) for PickPOI; {features: [{name, px: {x, y}}]} (source-image pixels, origin top-left, only the named features move) for ConfirmDetection; {points: [{x, y}]} (fractions of the frame, origin top-left; current and exposure stay the protocol's) for RunSpotBurn",
        },
    ),
    ToolSpec(
        name="prompt_suggest_answer",
        description="Suggest an answer to the pending supervision question, for the operator to take or not. Nothing is answered: the prompt shows 'Agent suggests: <the button>' with your reason under it, and the operator's click on it answers as theirs, recorded with you as the one who suggested it (prompt_suggested, then prompt_answered with suggested_by). Answering any other way ignores it; a new posting of the question drops it. Echo the question_id from prompt_get (409 stale_prompt if it has gone). value is as for prompt_answer, and is placed in the widget only when the operator takes the suggestion. Not for a question whose prompt says answer_via: \"decide\" (409 not_suggestible): at Assist, decide it through request_create.",
        method="POST",
        path="/app/workflow/prompt/suggest",
        level="assist",
        router="app",
        params={
            "response": "true for the positive option, false for the negative",
            "question_id": "the question_id from prompt_get",
            "reason": "why: shown under the suggestion for the operator",
            "value": "optional geometry, as for prompt_answer",
        },
        note="Works at Assist and Drive; refused (403) at Watch. At Drive you may prompt_answer directly instead.",
    ),
    ToolSpec(
        name="experiment_list_recent",
        description="Recently opened experiments on this machine, without loading them. experiment_load takes a path from here.",
        method="GET",
        path="/app/experiment/recent",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="protocol_list",
        description="The protocol files in the protocol folder, by name, with the default marked: what experiment_create and protocol_load take.",
        method="GET",
        path="/app/experiment/protocols",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="experiment_create",
        description="Create an experiment in the default experiments folder and open it in place of the current one, with a protocol from protocol_list (the default when omitted). An existing name is refused, never overwritten. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/create",
        level="drive",
        router="app",
        params={
            "name": "the new experiment's folder name",
            "protocol": "optional protocol name from protocol_list",
            "description": "optional one-line description",
        },
    ),
    ToolSpec(
        name="experiment_load",
        description="Open one of the recent experiments (a path from experiment_list_recent) in place of the current one. Any other experiment is the operator's to open. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/load",
        level="drive",
        router="app",
        params={"path": "a path from experiment_list_recent"},
    ),
    ToolSpec(
        name="protocol_load",
        description="Load a protocol from protocol_list into the open experiment, replacing its protocol, and save it beside the experiment. Existing items keep their task configs; item_apply_protocol updates one. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/protocol/load",
        level="drive",
        router="app",
        params={"protocol": "a protocol name from protocol_list"},
    ),
)

CATALOG = CATALOG + APP_TOOLS

# Every route the server serves is a tool above or here, with why it is not one
# (tests/server/test_catalog_discovery.py). A new route fails that test until
# someone decides. Path parameter names don't matter.
NOT_TOOLS: Dict[str, str] = {
    "GET /health": "Open liveness check, no token; the agent's first read is server_get_capabilities.",
    "GET /dashboard": "The operator's monitor page, HTML.",
    "GET /dashboard/log": "The operator's event log page, HTML; app_get_events is the agent's read.",
    "GET /app/events/history": "The event log page's read of the run before the live buffer; an agent reads a finished run from its experiment folder.",
    "GET /microscope/devices/{device}": "device_list carries every device's description.",
    "GET /microscope/devices/{device}/{parameter}/metadata": "device_list carries limits, choices and settable.",
    "GET /microscope/services/{service}": "service_list carries every service's description.",
    "GET /microscope/services/{service}/{parameter}/metadata": "service_list carries limits, choices and settable.",
    "GET /microscope/devices/{device}/values": "Not decided yet; device_get_parameter reads one value at a time.",
    "GET /microscope/services/{service}/values": "Not decided yet; service_get_parameter reads one value at a time.",
    "POST /microscope/devices/{device}/queries/{query}": "Not decided yet; services hold the questions an agent asks (service_call_query).",
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
    "beam_acquire_image": lambda a: (
        {},
        {
            "kwargs": _beam(a, image_settings=a.get("image_settings")),
            "answer": "preview",
        },
    ),
    "beam_auto_contrast": lambda a: ({}, _kwargs(**_beam(a))),
    "beam_auto_focus": lambda a: ({}, _kwargs(**_beam(a))),
    "fm_auto_focus": lambda a: (
        {},
        _kwargs(
            settings=(
                {"channel_name": str(a["channel_name"])}
                if a.get("channel_name")
                else None
            ),
            path=a.get("path"),
        ),
    ),
    "beam_acquire_overview": lambda a: (
        {},
        _kwargs(
            settings=dict(a["settings"]),
            path=str(a["path"]),
            name=str(a.get("name") or "overview"),
        ),
    ),
    "beam_estimate_overview": lambda a: ({}, _kwargs(settings=dict(a["settings"]))),
    "beam_acquire_focus_stack": lambda a: (
        {},
        _kwargs(
            image_settings=dict(a["image_settings"]),
            path=str(a["path"]),
            n_steps=int(a.get("n_steps") or 3),
            auto_focus=a.get("auto_focus", True),
        ),
    ),
    "fm_acquire_overview": lambda a: (
        {},
        _kwargs(
            channels=[dict(c) for c in a["channels"]],
            parameters=dict(a["parameters"]),
            path=str(a["path"]),
            name=str(a.get("name") or "overview"),
            zparams=a.get("zparams"),
        ),
    ),
    "fm_estimate_overview": lambda a: (
        {},
        _kwargs(
            channels=[dict(c) for c in a["channels"]],
            parameters=dict(a["parameters"]),
            zparams=a.get("zparams"),
        ),
    ),
    "beam_auto_brightness_contrast": lambda a: ({}, _kwargs(**_beam(a))),
    # A service command or query with no arguments still sends its kwargs.
    "milling_stop": lambda a: ({}, _stop(_kwargs(), a)),
    "milling_estimate_time": lambda a: ({}, _kwargs()),
    "alignment_align_to_reference": lambda a: (
        {},
        _kwargs(
            reference_path=str(a["reference_path"]),
            steps=int(a.get("steps") or 3),
            subsystem=str(a.get("subsystem") or "beam-shift"),
        ),
    ),
    "alignment_measure_shift": lambda a: (
        {},
        _kwargs(
            reference_path=str(a["reference_path"]), image_path=str(a["image_path"])
        ),
    ),
    "milling_preview_patterns": lambda a: (
        {},
        _kwargs(image_path=str(a["image_path"]), stages=a.get("stages")),
    ),
    "stage_get_orientation": lambda a: ({}, _kwargs()),
    "stage_get_milling_angle": lambda a: ({}, _kwargs()),
    "stage_get_geometry": lambda a: ({}, _kwargs()),
    "stage_stable_move": lambda a: (
        {},
        _kwargs(
            dx=float(a["dx"]),
            dy=float(a["dy"]),
            beam_type=str(a.get("beam_type", "ELECTRON")),
        ),
    ),
    "stage_vertical_move": lambda a: (
        {},
        _kwargs(
            dy=float(a["dy"]),
            dx=float(a.get("dx", 0.0)),
            beam_type=str(a.get("beam_type") or "ION"),
            relaxation=float(a.get("relaxation") or 1.0),
        ),
    ),
    "stage_move_to_orientation": lambda a: (
        {},
        _kwargs(orientation=str(a["orientation"])),
    ),
    "stage_move_absolute": lambda a: ({}, _kwargs(stage_position=_absolute(a))),
    "stage_move_relative": lambda a: (
        {},
        _kwargs(delta=_position(a, x="dx", y="dy", z="dz", r="", t="")),
    ),
    "stage_move_to_milling_angle": lambda a: (
        {},
        _kwargs(
            milling_angle=float(a["milling_angle_deg"]),
            rotation=(
                None if a.get("rotation_deg") is None else float(a["rotation_deg"])
            ),
            keep_rotation=True,
        ),
    ),
    "device_call_command": lambda a: (
        {"device": a.get("device"), "command": a.get("command")},
        _stop(_preview(dict(a.get("kwargs") or {})), a),
    ),
    "service_call_command": lambda a: (
        {"service": a.get("service"), "command": a.get("command")},
        _stop(_preview(dict(a.get("kwargs") or {})), a),
    ),
    "service_call_query": lambda a: (
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
