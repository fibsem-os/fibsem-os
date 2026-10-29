"""The microscope's tools: the server, the microscope, its devices and services, the stage, alignment and correlation."""

import os
from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from fibsem.ai.agent.catalog.spec import (
    RETRACT_FIRST,
    RETRACT_FIRST_SAFE_MOVE,
    ToolSpec,
    absolute_position,
    arg,
    command_body,
    kwargs_body,
    stage_position,
)
from fibsem.util import wording

if TYPE_CHECKING:
    from fibsem.structures.params import Parameter


_ALIGN_WITH = {
    "beam-shift": "the beam shift",
    "stage": "stage moves",
    "stage-vertical": "vertical stage moves",
}


def _parameter_spec(info: Any) -> Optional["Parameter"]:
    """A device parameter's listing (``describe_device``) as a ``Parameter``:
    its unit and how it is shown."""
    if not isinstance(info, dict):
        return None
    from fibsem.structures.params import Display, Parameter

    display = info.get("display")
    shown = None
    if isinstance(display, dict) and {"label", "scale", "unit"} & set(display):
        known = {f.name for f in dataclass_fields(Display)}
        try:
            shown = Display(**{k: v for k, v in display.items() if k in known})
        except TypeError:
            shown = None
    return Parameter(unit=info.get("unit"), display=shown)


def _set_parameter(
    a: Dict[str, Any], owner_key: str, path: str, get: Callable[[str], Any]
) -> Dict[str, Any]:
    owner, parameter = a.get(owner_key), str(a.get("parameter"))
    if parameter == "on" and isinstance(a.get("value"), bool):
        # Said as "Electron beam on", not "Electron · Set on to on".
        name = wording.words(owner) + (" beam" if owner in ("electron", "ion") else "")
        state = "on" if a["value"] else "off"
        return {
            "what": wording.words(owner).capitalize(),
            "change": f"Turn the {name} {state}",
        }
    spec = None
    listing = get(path.rsplit("/", 1)[0]) if "/" in path else None
    if isinstance(listing, dict):
        spec = _parameter_spec((listing.get("parameters") or {}).get(parameter))
    name = (spec.label if spec is not None else None) or wording.words(parameter)
    new = wording.display(a.get("value"), spec)
    current = get(path)
    if isinstance(current, dict) and "value" in current:
        change = f"Set {name} {wording.display(current['value'], spec)} → {new}"
    else:
        change = f"Set {name} to {new}"
    return {
        "what": wording.words(owner).capitalize() if owner else "",
        "change": change,
    }


def _stage_move_relative(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {"what": "Stage", "change": wording.stage_relative(a, ctx.stage_now())}


def _stage_move_absolute(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    change = f"Safe move to {wording.position_text(a)}"
    now = ctx.stage_now()
    if now:
        change += f"\nfrom {wording.position_text(now)}"
    return {"what": "Stage", "change": change}


def _stage_stable_move(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {
        "what": "Stage",
        "change": f"Stable move in {wording.beam_words(a.get('beam_type'))}: "
        f"{wording.offset(a.get('dx'), a.get('dy'))}",
    }


def _stage_fm_stable_move(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Stage",
        "change": "Stable move in the fluorescence image: "
        f"{wording.offset(a.get('dx'), a.get('dy'))}",
    }


def _stage_move_to_device(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    change = f"Move to the {a.get('device')} device"
    if a.get("orientation"):
        change += f", in the {str(a['orientation']).upper()} orientation"
    return {"what": "Stage", "change": change}


def _stage_save_position(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    change = f"Save the stage position now as {a.get('name')!r}"
    now = ctx.stage_now()
    if now:
        change += f": {wording.position_text(now)}"
    return {"what": "Saved positions", "change": change}


def _stage_move_to_saved_position(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Stage",
        "change": f"Safe move to the saved position {a.get('name')!r}",
    }


def _stage_remove_saved_position(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Saved positions",
        "change": f"Forget the saved position {a.get('name')!r}",
    }


def _stage_vertical_move(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    change = (
        "Vertical move, to centre a feature "
        f"{wording.offset(a.get('dx'), a.get('dy'))} in {wording.beam_words(a.get('beam_type'), 'ION')}"
    )
    try:
        relaxation = float(a["relaxation"])
    except (KeyError, TypeError, ValueError):
        relaxation = 1.0
    if relaxation < 1:
        change += f", {wording.number(relaxation * 100)}% of the way"
    return {"what": "Stage", "change": change}


def _stage_move_to_orientation(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Stage",
        "change": f"Move to the {str(a.get('orientation', '')).upper()} orientation",
    }


def _stage_move_to_milling_angle(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Stage",
        "change": f"Tilt to the milling angle, {wording.short(a.get('milling_angle_deg'))}°"
        + (
            f", rotation {wording.short(a['rotation_deg'])}°"
            if a.get("rotation_deg") is not None
            else ""
        ),
    }


def _alignment_align_to_reference(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    reference = os.path.basename(str(a.get("reference_path") or "")) or "a reference"
    steps = a.get("steps") or 3
    method = _ALIGN_WITH.get(str(a.get("subsystem") or "beam-shift"), "the beam shift")
    return {
        "what": "Alignment",
        "change": f"Align to {reference}: {wording.plural(int(steps), 'step')} with {method}",
    }


def _microscope_set_state(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Microscope",
        "change": "Set both beams and the stage to a saved state; the stage moves",
    }


def _alignment_check_coincidence(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Alignment",
        "change": "Take an SEM and a FIB image to measure coincidence (moves nothing)",
    }


def _device_set_parameter(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return _set_parameter(a, "device", ctx.path, ctx.get)


def _service_set_parameter(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return _set_parameter(a, "service", ctx.path, ctx.get)


TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="server_get_capabilities",
        description="Server capabilities: API version, which halves are served (routers.microscope, routers.app), and the agent's level (off / watch / assist / drive). Read it first: the level says what the other tools will do. At Assist, act by asking: request_create.",
        method="GET",
        path="/capabilities",
        level="watch",
        router="base",
    ),
    ToolSpec(
        name="server_report_session",
        returns="AgentSession",
        description="Tell the app about your session, so the operator sees who is working and toward what: your session id and model when you start, the goal you were given, and your budget and spend if your harness tracks them. Fields you leave out keep their last value; a new session_id starts a new session. fibsem-agent reports for you after every turn: under it, call this only with goal, when the operator explicitly gives you a new goal. Pass ended: true, with a reason, when you stop.",
        method="POST",
        path="/app/agent/session",
        level="watch",
        router="app",
        params={
            "session_id": "your session's id, as your harness names it",
            "model": "the model you run as",
            "goal": "what you were asked to do, in a sentence",
            "budget_usd": arg("what this session may spend, in US dollars", float),
            "spent_usd": arg("what it has spent so far", float),
            "turns": arg("how many turns it has taken", int),
            "ended": arg("true when the session ends", bool),
            "reason": "why it ended",
        },
        optional=(
            "session_id",
            "model",
            "goal",
            "budget_usd",
            "spent_usd",
            "turns",
            "ended",
            "reason",
        ),
    ),
    ToolSpec(
        name="server_set_constraints",
        returns="AgentSession",
        description='Keep the operator\'s constraints for this session where the Agent page shows them, beside the goal: limits the work must keep, in the operator\'s words, such as "finish by 07:00" or "don\'t change the milling currents". Give the whole list each time; an empty list clears it. A constraint with a time takes it as by: 07:00 (its next occurrence) or an ISO date and time. Under fibsem-agent you are woken 30 min before each time and at it.',
        method="POST",
        path="/app/agent/session",
        level="watch",
        router="app",
        params={
            "constraints": arg(
                "the whole list, each {text, by?}: text in the operator's words, by the time it must be done by",
                List[Dict[str, Any]],
            ),
        },
    ),
    ToolSpec(
        name="server_set_checklist",
        returns="AgentSession",
        description="Keep your checklist of the operator's instructions where the Agent page shows it: each item {text, state}, state open (the default), done when it is finished, or always for a standing rule that stays until the operator lifts it. Give the whole list each time, in the operator's words; it replaces the last one. Update it when an instruction arrives and when an item is done.",
        method="POST",
        path="/app/agent/session",
        level="watch",
        router="app",
        params={
            "checklist": arg(
                "the whole list, each {text, state}: state open, done or always",
                List[Dict[str, Any]],
            ),
        },
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
        result_key="orientation",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
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
        name="stage_get_current_milling_angle",
        description="Current milling angle in degrees: the angle between the ion beam and the sample surface at the stage's rotation and tilt (90 at the FIB pose).",
        method="POST",
        path="/microscope/services/stage_movement/queries/current_milling_angle",
        result_key="milling_angle_deg",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="stage_get_geometry",
        description="The instrument's geometry in one read: column tilts, pre-tilt, rotation reference and centre, the orientation table, device origins, and what they mean at the stage's pose now: which way raw +z goes in the chamber, the milling angle here and the pose for the stored one, and per beam the stage move a +1 µm stable move along each image axis makes at the current scan rotation. Read it first on a new instrument, then confirm once with one small move. Angles in degrees.",
        method="POST",
        path="/microscope/services/stage_movement/queries/geometry",
        result_key="geometry",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
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
        path="/microscope/services/alignment/commands/align_to_reference",
        result_key="run",
        level="drive",
        hardware=True,
        params={
            "reference_path": "a saved image inside the open experiment",
            "steps": arg(
                "how many measure-and-correct steps (default 3)", int, default=3
            ),
            "subsystem": arg(
                "beam-shift (default), stage or stage-vertical", default="beam-shift"
            ),
            "use_autofocus": arg(
                "true to auto focus before each acquisition (default false)",
                bool,
                default=False,
            ),
            "method": arg(
                "how the shift is measured: cross-correlation (default), phase-correlation or skimage-phase-correlation",
                default="cross-correlation",
            ),
            "validate": arg(
                "take a final image and check the methods agree on what is left (default true)",
                bool,
                default=True,
            ),
            "use_autocontrast": arg(
                "true to autocontrast before each acquisition (default false)",
                bool,
                default=False,
            ),
            "path": "a folder inside the open experiment to save the run to (default Alignment, next to the reference)",
        },
        optional=(
            "steps",
            "subsystem",
            "use_autofocus",
            "method",
            "validate",
            "use_autocontrast",
            "path",
        ),
        line=_alignment_align_to_reference,
        call=lambda a: (
            {},
            kwargs_body(
                reference_path=str(a["reference_path"]),
                steps=int(a.get("steps") or 3),
                subsystem=str(a.get("subsystem") or "beam-shift"),
                use_autofocus=bool(a.get("use_autofocus", False)),
                method=str(a.get("method") or "cross-correlation"),
                validate=bool(a.get("validate", True)),
                use_autocontrast=bool(a.get("use_autocontrast", False)),
                path=a.get("path"),
            ),
        ),
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
        result_key="shift",
        level="watch",
        params={
            "reference_path": "the saved reference image",
            "image_path": "the saved image to measure against it",
            "method": arg(
                "cross-correlation (default), phase-correlation or skimage-phase-correlation",
                default="cross-correlation",
            ),
        },
        optional=("method",),
        call=lambda a: (
            {},
            kwargs_body(
                reference_path=str(a["reference_path"]),
                image_path=str(a["image_path"]),
                method=str(a.get("method") or "cross-correlation"),
            ),
        ),
    ),
    ToolSpec(
        name="alignment_compare_methods",
        description=(
            "The shift of one saved image from another by every alignment method, "
            "how far apart the methods are in pixels, and whether they agree: the "
            "check alignment_align_to_reference validates with. A disagreement "
            "means the measurement is not to be trusted. Both paths inside the "
            "open experiment. Reads the files and moves nothing."
        ),
        method="POST",
        path="/microscope/services/alignment/queries/compare_methods",
        result_key="comparison",
        level="watch",
        params={
            "reference_path": "the saved reference image",
            "image_path": "the saved image to measure against it",
            "agreement_px": arg(
                "the most the methods may differ and agree, in pixels (default 2)",
                float,
                default=2.0,
            ),
        },
        optional=("agreement_px",),
        call=lambda a: (
            {},
            kwargs_body(
                reference_path=str(a["reference_path"]),
                image_path=str(a["image_path"]),
                agreement_px=float(a.get("agreement_px") or 2.0),
            ),
        ),
    ),
    ToolSpec(
        name="alignment_check_coincidence",
        description=(
            "EXPERIMENTAL, to try beside the by-hand loop: take an SEM and a FIB "
            "image here and find the target centred in one beam (beam, ELECTRON "
            "by default) in the other beam's view. The target is x, y metres "
            "from that beam's image centre, +y up; the feature you centred is "
            "0, 0 (the default). Autocontrasts each beam first unless "
            "autocontrast is false. Answers an overlay to look at (the patch and "
            "target beside the match: + where the target was found, x where it "
            "would be if coincident) and dy for stage_vertical_move with "
            "beam_type the OTHER beam, the height error dz, and dx: a "
            "height error cannot move x, so a steady dx is a beam shift or scan "
            "rotation, never a stage move. Also the score, how well two filter "
            "scales agree, how close a second match comes and warnings. From the "
            "FIB a height error shows ~3x larger in the SEM and can leave its "
            "frame: start from the SEM when it may be large. It moves "
            "nothing and refuses nothing: judge the overlay before moving. "
            "path saves the pair and the overlay, for alignment_measure_coincidence "
            "to replay."
        ),
        method="POST",
        path="/microscope/services/alignment/commands/check_coincidence",
        result_key="coincidence",
        level="drive",
        hardware=True,
        params={
            "x": arg(
                "target x, metres from that beam's image centre (default 0)",
                float,
                default=0.0,
            ),
            "y": arg(
                "target y, metres from that beam's image centre, +y up (default 0)",
                float,
                default=0.0,
            ),
            "hfw": arg(
                "field width of both images in metres (default 150e-6)",
                float,
                default=0.00015,
            ),
            "autocontrast": arg(
                "autocontrast each beam first (default true)", bool, default=True
            ),
            "path": "optional folder inside the open experiment to save into",
            "beam": arg(
                "the beam the target is centred in: ELECTRON (default) or ION",
                default="ELECTRON",
            ),
        },
        optional=("x", "y", "hfw", "autocontrast", "path", "beam"),
        line=_alignment_check_coincidence,
        call=lambda a: (
            {},
            command_body(
                {
                    "x": float(a.get("x") or 0.0),
                    "y": float(a.get("y") or 0.0),
                    "hfw": float(a.get("hfw") or 150e-6),
                    "autocontrast": a.get("autocontrast") is not False,
                    "path": a.get("path"),
                    "beam": str(a.get("beam") or "ELECTRON"),
                },
                preview=True,
            ),
        ),
    ),
    ToolSpec(
        name="alignment_measure_coincidence",
        description=(
            "EXPERIMENTAL: alignment_check_coincidence on a saved SEM and FIB "
            "pair of the same size and field, both inside the open experiment: "
            "replays a check, or measures images you took. Same answer: an "
            "overlay and dy, dz, dx, score and warnings. Reads the files and "
            "moves nothing."
        ),
        method="POST",
        path="/microscope/services/alignment/queries/measure_coincidence",
        result_key="coincidence",
        level="watch",
        params={
            "sem_path": "the saved SEM image",
            "fib_path": "the saved FIB image of the same place",
            "x": arg(
                "target x, metres from that beam's image centre (default 0)",
                float,
                default=0.0,
            ),
            "y": arg(
                "target y, metres from that beam's image centre, +y up (default 0)",
                float,
                default=0.0,
            ),
            "beam": arg(
                "the beam the target is centred in: ELECTRON (default) or ION",
                default="ELECTRON",
            ),
        },
        optional=("x", "y", "beam"),
        call=lambda a: (
            {},
            kwargs_body(
                sem_path=str(a["sem_path"]),
                fib_path=str(a["fib_path"]),
                x=float(a.get("x") or 0.0),
                y=float(a.get("y") or 0.0),
                beam=str(a.get("beam") or "ELECTRON"),
            ),
        ),
    ),
    ToolSpec(
        name="correlation_correlate",
        description=(
            "Correlate a saved FM stack to a saved FIB image, as the correlation "
            "tab does: fit the transform to fiducial pairs and map the points of "
            "interest into the FIB image. fib_points are (x, y) FIB pixels; "
            "fm_points the same fiducials, in the same order, as (x, y, z) FM "
            "pixels and slices (at least 4 pairs); poi_points the targets as "
            "(x, y, z) in the FM stack. Answers the RMS error (px, nm), each point "
            "of interest in FIB image pixels and metres from the image centre, the "
            "verdict (good, check, bad, with reasons) and what the fit started "
            "from. With lamella_path it starts from the experiment's previous runs "
            "and saves a new run in the lamella's Correlation folder, with an "
            "overlay to look at (overlay_path, experiment_read_file: o picked, "
            "x fitted, + targets). Refractive-index "
            "correction, as the tab's RI panel: give ONE surface point, "
            "fm_surface_point (x, y, z in the FM stack: corrects the targets' depth "
            "before the fit) or fib_surface_point (x, y in the FIB image: moves the "
            "first target after it). The factor is the lookup table's for the "
            "protocol's RI settings, with ri's values over them, or ri_factor; the "
            "answer's ri_correction says what was applied, and ri_note that none "
            "was. Paths inside the open "
            "experiment. Does not move the lamella's point of interest. Moves "
            "nothing."
        ),
        method="POST",
        path="/services/correlation/queries/correlate",
        router="services",
        level="watch",
        params={
            "fib_image_path": "the saved FIB image the fib_points are in",
            "fm_image_path": "the saved FM stack the fm_points and poi_points are in",
            "fib_points": arg("fiducials as [[x, y], ...] FIB pixels", list),
            "fm_points": arg(
                "the same fiducials as [[x, y, z], ...] FM pixels and slices", list
            ),
            "poi_points": arg("targets as [[x, y, z], ...] FM pixels and slices", list),
            "lamella_path": arg(
                "the lamella's folder, to start from previous runs and save the run there",
                default="",
            ),
            "fm_surface_point": arg(
                "[x, y, z] the sample surface in the FM stack: RI pre-correction",
                List[float],
            ),
            "fib_surface_point": arg(
                "[x, y] the sample surface in the FIB image: RI post-correction",
                List[float],
            ),
            "ri": arg(
                "RI settings over the protocol's: any of tilt_deg, depth_um, na, n2, wavelength_um",
                Dict[str, float],
            ),
            "ri_factor": arg(
                "the RI correction factor itself, instead of the lookup table's", float
            ),
        },
        optional=(
            "lamella_path",
            "fm_surface_point",
            "fib_surface_point",
            "ri",
            "ri_factor",
        ),
        call=lambda a: ({}, kwargs_body(**a)),
    ),
    ToolSpec(
        name="correlation_find_spot_burns",
        description=(
            "The spot-burn fiducials of a lamella, from its spot-burn task, as the "
            "correlation tab seeds its FIB fiducials: fractions of the image they "
            "were placed on and its field width, and with fib_image_path their "
            "pixels in that FIB image (refused, with the reason, when it is "
            "another field width). reason says why there are none. Reads the "
            "experiment and moves nothing."
        ),
        method="POST",
        path="/services/correlation/queries/spot_burns",
        router="services",
        level="watch",
        params={
            "lamella_path": "the lamella's folder",
            "fib_image_path": arg(
                "a saved FIB image to give the burns' pixels in", default=""
            ),
        },
        optional=("fib_image_path",),
        call=lambda a: ({}, kwargs_body(**a)),
    ),
    ToolSpec(
        name="correlation_refine_point",
        description=(
            "Move a picked point onto the feature under it, as the correlation "
            "tab's point fit does: a hole in a FIB image (beam fib, method Hole), "
            "or a reflection hole (Hole) or fluorescent target (Gaussian, the "
            "default) in a channel of an FM stack (beam fm). Answers the status "
            "(ok, unchanged, error), the fitted (x, y, z), how far it moved in "
            "pixels, and why it failed; for an FM fit, z_fit: the z Gaussian's "
            "amplitude above the background (a reflection hole's inverted) and "
            "its FWHM in planes. Saves the fit's figure (the z signal and fit, "
            "the XY crop with the picked and fitted points, as the tab's confirm "
            "dialog shows it) beside the image as drawn/<image>-fit-<x>-<y>.png "
            "and answers it as figure_path with a preview: look at it before "
            "you use the point. A proposal: nothing moves."
        ),
        method="POST",
        path="/services/correlation/queries/refine_point",
        result_key="point",
        router="services",
        level="watch",
        params={
            "image_path": "the saved FIB image or FM stack, inside the open experiment",
            "x": arg("the picked x, pixels", float),
            "y": arg("the picked y, pixels", float),
            "z": arg(
                "the picked slice, for an FM point, 0 the first (default the "
                "channel's sharpest plane; a Hole fit searches only the planes "
                "around it)",
                float,
            ),
            "beam": arg("fib or fm (default fib)", default="fib"),
            "method": arg(
                "Hole or Gaussian (default Hole for fib, Gaussian for fm)", default=""
            ),
            "channel": arg(
                "the FM channel, by name or index (default the first)",
                Union[str, int],
                default=0,
            ),
        },
        optional=("z", "beam", "method", "channel"),
        call=lambda a: ({}, kwargs_body(**a)),
    ),
    ToolSpec(
        name="correlation_predict",
        description=(
            "Map FM points through a saved correlation run (run_path from "
            "correlation_correlate, or a run folder in a lamella's Correlation "
            "folder): (x, y, z) FM pixels and slices to FIB image pixels and "
            "metres from the image centre. No refractive-index correction. Reads "
            "the run and moves nothing."
        ),
        method="POST",
        path="/services/correlation/queries/predict",
        result_key="prediction",
        router="services",
        level="watch",
        params={
            "run_path": "a correlation run folder inside the open experiment",
            "fm_points": arg("[[x, y, z], ...] FM pixels and slices", list),
        },
        call=lambda a: ({}, kwargs_body(**a)),
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
        result_key="position",
        level="drive",
        hardware=True,
        params={
            "dx": arg(
                "the feature's offset from the image centre along x, metres, right positive",
                float,
            ),
            "dy": arg(
                "the feature's offset from the image centre along y, metres, up positive",
                float,
            ),
            "beam_type": arg(
                "the beam whose image the offset was measured in: ELECTRON or ION",
                default="ELECTRON",
            ),
            "static_wd": arg(
                "true to put the electron working distance at its eucentric height after, rather than where it was (default false)",
                bool,
                default=False,
            ),
        },
        optional=("beam_type", "static_wd"),
        line=_stage_stable_move,
        call=lambda a: (
            {},
            kwargs_body(
                dx=float(a["dx"]),
                dy=float(a["dy"]),
                beam_type=str(a.get("beam_type", "ELECTRON")),
                static_wd=bool(a.get("static_wd", False)),
            ),
        ),
    ),
    ToolSpec(
        name="stage_fm_stable_move",
        description=(
            "Move the stage so a feature seen in the fluorescence image comes to "
            "the centre of that image: stage_stable_move for the FM view. dx and dy "
            "are the feature's offset from the image centre in metres, x to the "
            "right and y up, in the image as displayed. The server turns the offset "
            "into stage x, y and z for the camera's tilt and display transform, so "
            "the sample slides along its own surface and stays in the objective's "
            "focal plane. Insert the objective first. Take a fresh FM image "
            "afterwards to check where the feature landed."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/fm_stable_move",
        result_key="position",
        level="drive",
        hardware=True,
        params={
            "dx": arg(
                "the feature's offset from the image centre along x, metres, right positive",
                float,
            ),
            "dy": arg(
                "the feature's offset from the image centre along y, metres, up positive",
                float,
            ),
        },
        line=_stage_fm_stable_move,
        call=lambda a: ({}, kwargs_body(dx=float(a["dx"]), dy=float(a["dy"]))),
    ),
    ToolSpec(
        name="stage_get_saved_positions",
        description="The named stage positions saved with this microscope configuration, as the Saved Positions panel lists them (metres and radians).",
        method="POST",
        path="/microscope/services/stage_movement/queries/saved_positions",
        result_key="saved_positions",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="stage_save_position",
        description="Save the stage position now under name, as the Saved Positions panel's add and update do: a position already saved under that name is replaced. Kept with the microscope configuration.",
        method="POST",
        path="/microscope/services/stage_movement/commands/save_position",
        result_key="position",
        level="drive",
        params={"name": "the name to save it under"},
        line=_stage_save_position,
        call=lambda a: ({}, kwargs_body(name=str(a["name"]))),
    ),
    ToolSpec(
        name="stage_move_to_saved_position",
        description="Move the stage to a saved position by name (stage_get_saved_positions), through the safe absolute move that tilts flat before a large rotation."
        + RETRACT_FIRST_SAFE_MOVE,
        method="POST",
        path="/microscope/services/stage_movement/commands/move_to_saved_position",
        result_key="position",
        level="drive",
        hardware=True,
        params={"name": "the saved position's name"},
        line=_stage_move_to_saved_position,
        call=lambda a: ({}, kwargs_body(name=str(a["name"]))),
    ),
    ToolSpec(
        name="stage_remove_saved_position",
        description="Forget a saved position by name, as the Saved Positions panel's remove does.",
        method="POST",
        path="/microscope/services/stage_movement/commands/remove_saved_position",
        level="drive",
        params={"name": "the saved position's name"},
        line=_stage_remove_saved_position,
        call=lambda a: ({}, kwargs_body(name=str(a["name"]))),
    ),
    ToolSpec(
        name="stage_move_to_device",
        description=(
            "Travel the stage to another device on an instrument where the FM is "
            "beside the FIB-SEM rather than under it: FIBSEM or FM, as "
            "stage_get_geometry names them. It retracts the objective and re-poses "
            "on the way when it has to, so a rotation never happens under the "
            "objective. orientation names the pose to arrive in; left out, the "
            "pose carries across when the device can image from it, else the "
            "device's first imaging orientation. Refused when the stage is at no "
            "device. Read the position back after."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/move_to_device",
        level="drive",
        hardware=True,
        params={
            "device": "the device to travel to: FIBSEM or FM",
            "orientation": "the pose to arrive in (SEM, FIB, MILLING, FM); left out, as above",
        },
        optional=("orientation",),
        line=_stage_move_to_device,
        call=lambda a: (
            {},
            kwargs_body(
                device=str(a["device"]),
                orientation=(
                    None if not a.get("orientation") else str(a["orientation"]).upper()
                ),
            ),
        ),
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
        result_key="position",
        level="drive",
        hardware=True,
        params={
            "dy": arg(
                "the feature's offset from the image centre along y, metres, up positive",
                float,
            ),
            "dx": arg(
                "the offset along x, metres, right positive (usually 0)",
                float,
                default=0.0,
            ),
            "beam_type": "the view the offset was measured in: ION (default) or ELECTRON",
            "relaxation": arg(
                "fraction of the exact move to make, 0-1 (default 1); below 1 undershoots, to keep a repeated look-and-correct loop from overshooting",
                float,
            ),
        },
        optional=("dx", "beam_type", "relaxation"),
        line=_stage_vertical_move,
        call=lambda a: (
            {},
            kwargs_body(
                dy=float(a["dy"]),
                dx=float(a.get("dx", 0.0)),
                beam_type=str(a.get("beam_type") or "ION"),
                relaxation=float(a.get("relaxation") or 1.0),
            ),
        ),
    ),
    ToolSpec(
        name="stage_move_to_orientation",
        description=(
            "Move the stage to a named orientation (SEM, FIB, MILLING, and FM "
            "where fitted): its rotation and tilt, by the safe move "
            "(stage_move_absolute). x, y and z stay, except that a half turn "
            "carries x and y round the rotation centre where the instrument "
            "reports one, so the same spot comes back under the beam." + RETRACT_FIRST
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/move_to_orientation",
        result_key="position",
        level="drive",
        hardware=True,
        params={"orientation": "SEM, FIB, MILLING or FM"},
        line=_stage_move_to_orientation,
        call=lambda a: (
            {},
            kwargs_body(orientation=str(a["orientation"])),
        ),
    ),
    ToolSpec(
        name="stage_move_absolute",
        description=(
            "Move the stage to an absolute position (metres and radians) safely: "
            "for a large rotation it tilts flat first, and it rotates before "
            "moving the other axes, so a tilted sample isn't swung round. Use "
            "it to return to a saved position." + RETRACT_FIRST_SAFE_MOVE
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/safe_absolute_stage_movement",
        result_key="position",
        level="drive",
        hardware=True,
        params={
            "x": arg("x in metres", float),
            "y": arg("y in metres", float),
            "z": arg("z in metres", float),
            "r": arg("rotation in radians", float),
            "t": arg("tilt in radians", float),
        },
        line=_stage_move_absolute,
        call=lambda a: ({}, kwargs_body(stage_position=absolute_position(a))),
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
        result_key="position",
        level="drive",
        hardware=True,
        params={
            "dx": arg("relative x in metres", float, default=0.0),
            "dy": arg("relative y in metres", float, default=0.0),
            "dz": arg("relative z in metres", float, default=0.0),
        },
        optional=("dx", "dy", "dz"),
        line=_stage_move_relative,
        call=lambda a: (
            {},
            kwargs_body(delta=stage_position(a, x="dx", y="dy", z="dz", r="", t="")),
        ),
    ),
    ToolSpec(
        name="stage_move_to_milling_angle",
        description=(
            "Tilt the stage to a milling angle, in degrees. The stage stays at its "
            "current rotation unless rotation_deg names another." + RETRACT_FIRST
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/move_to_milling_angle",
        result_key="success",
        level="drive",
        hardware=True,
        params={
            "milling_angle_deg": arg("target milling angle in degrees", float),
            "rotation_deg": arg(
                "stage rotation in degrees; the current rotation if left out", float
            ),
        },
        optional=("rotation_deg",),
        line=_stage_move_to_milling_angle,
        call=lambda a: (
            {},
            kwargs_body(
                milling_angle=float(a["milling_angle_deg"]),
                rotation=(
                    None if a.get("rotation_deg") is None else float(a["rotation_deg"])
                ),
                keep_rotation=True,
            ),
        ),
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
            "value": arg("the new value", Any),
        },
        line=_device_set_parameter,
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
            "kwargs": arg("the command's arguments, by name", dict),
            "reason": "for a stop (stop, stop_live): why, for the operator (required there)",
        },
        optional=("kwargs", "reason"),
        call=lambda a: (
            {"device": a.get("device"), "command": a.get("command")},
            command_body(a.get("kwargs"), a.get("reason"), preview=True),
        ),
    ),
    ToolSpec(
        name="device_call_query",
        description="Ask a device query listed by device_list: it takes arguments, answers, and moves nothing, e.g. device=stage query=pose_table.",
        method="POST",
        path="/microscope/devices/{device}/queries/{query}",
        level="watch",
        params={
            "device": "device name from device_list",
            "query": "query name",
            "kwargs": arg("the query's arguments, by name", dict),
        },
        optional=("kwargs",),
        call=lambda a: (
            {"device": a.get("device"), "query": a.get("query")},
            command_body(a.get("kwargs"), preview=True),
        ),
    ),
    ToolSpec(
        name="microscope_set_state",
        description=(
            "Set the microscope to a saved state: both beams' settings and the "
            "stage position, e.g. the microscope_state of an image's metadata "
            "(experiment_get_image_metadata) to go back to where and how it was "
            "taken. The stage moves. microscope_get_state gives the shape."
        ),
        method="PUT",
        path="/microscope/state",
        level="drive",
        hardware=True,
        params={
            "microscope_state": arg(
                "a MicroscopeState dict, as microscope_get_state answers", dict
            )
        },
        line=_microscope_set_state,
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
            "value": arg("the new value", Any),
        },
        line=_service_set_parameter,
    ),
    ToolSpec(
        name="service_call_command",
        description=(
            "Run a service command listed by service_list, e.g. service=stage_movement "
            "command=move_to_orientation kwargs={'orientation': 'SEM'}. A long command "
            "(listed with long: true: milling run, run_stage and run_task, spot burn "
            "run, alignment align_to_reference, imaging fm_auto_focus, "
            "fm_acquire_image, acquire_overview, fm_acquire_overview and "
            "acquire_focus_stack) answers at once with "
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
            "kwargs": arg("the command's arguments, by name", dict),
            "reason": "for a stop (stop, stop_live): why, for the operator (required there)",
        },
        optional=("kwargs", "reason"),
        call=lambda a: (
            {"service": a.get("service"), "command": a.get("command")},
            command_body(a.get("kwargs"), a.get("reason"), preview=True),
        ),
    ),
    ToolSpec(
        name="service_wait_for_run",
        description=(
            "How a long command on a service is going (a mill, a spot burn, an "
            "alignment, an FM auto-focus, an overview or a focus stack), whoever "
            "started it: you, a workflow task or the operator. A workflow's "
            "milling task is a run on milling, its run_id the milling_run_id "
            "its milling events carry. Answers its run with state running, "
            "done, failed or cancelled, and when it has ended its result or "
            "error. Waits up to wait seconds (at most 60) for it to end, so call "
            "it again while it says running. Without run_id, the run going on, "
            "else the service's last; with run_id, waits for that run, and a "
            "later run on the service answers at once. The same end comes as a "
            "command_ended event on app_get_events."
        ),
        method="GET",
        path="/microscope/services/{service}/run",
        level="watch",
        params={
            "service": "required: the service the command ran on, e.g. milling, imaging or alignment",
            "run_id": "the run_id the command answered with, or a milling_run_id",
            "wait": arg(
                "seconds to wait for the end, at most 60 (default 60)",
                float,
                default=60.0,
            ),
        },
        optional=("run_id", "wait"),
    ),
    ToolSpec(
        name="service_call_query",
        description=(
            "Ask a service query listed by service_list: it takes arguments, "
            "answers, and moves nothing, e.g. service=stage_movement "
            "query=project_stable_move kwargs={'dx': 1e-6, 'dy': 0, "
            "'beam_type': 'ELECTRON', 'base_position': {...}} for where a stable "
            "move from base_position would end."
        ),
        method="POST",
        path="/microscope/services/{service}/queries/{query}",
        level="watch",
        params={
            "service": "service name from service_list",
            "query": "query name",
            "kwargs": arg("the query's arguments, by name", dict),
        },
        optional=("kwargs",),
        call=lambda a: (
            {"service": a.get("service"), "query": a.get("query")},
            command_body(a.get("kwargs"), preview=True),
        ),
    ),
)
