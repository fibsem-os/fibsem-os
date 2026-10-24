"""The imaging tools: SEM/FIB images, overviews and focus, and the FM's."""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fibsem.ai.agent.catalog.spec import (
    NEAR_FOCUS,
    OVERVIEW_CENTRE,
    ToolSpec,
    arg,
    beam_kwargs,
    command_body,
    kwargs_body,
)
from fibsem.util import wording

# What an overview covers, for both overview tools.
_OVERVIEW_RULE = (
    " An overview the operator asks for is the whole grid, from its centre "
    "(grid_move_to first) with the tiles the stage can't reach turned off "
    "(odd rows and columns lose the fewest: reach is judged by tile centre); a "
    "tile set (e.g. 3 x 3 at 500 um) is where the stage is, at the size given. "
    "One image is beam_acquire_image or fm_acquire_image, never a 1 x 1 overview."
)

# A fresh image's settings when the agent leaves one out: the tool's defaults,
# not the beam's current values (catalog: beam_acquire_image).
_IMAGE_DEFAULTS = {"hfw": 150e-6, "resolution": [1536, 1024], "dwell_time": 1e-6}


# An image's settings as the imaging panel labels them, in display units.
_IMAGE_LABELS = {
    "beam_type": "Beam",
    "hfw": "Field of view",
    "resolution": "Resolution",
    "dwell_time": "Dwell time",
    "autocontrast": "Autocontrast",
    "autogamma": "Auto gamma",
    "save": "Save",
    "path": "Saved in",
    "filename": "File name",
    "line_integration": "Line integration",
    "frame_integration": "Frame integration",
    "scan_interlacing": "Scan interlacing",
    "drift_correction": "Drift correction",
    "reduced_area": "Reduced area",
    "scan_rotation": "Scan rotation",
    "nrows": "Rows",
    "ncols": "Columns",
    "overlap": "Overlap",
}


# The beam's settings an image changes, by the beam device's parameter names.
_BEAM_PARAMETERS = ("hfw", "resolution", "dwell_time")


def _image_value(key: str, value: Any) -> str:
    if key == "hfw":
        return wording.distance(value)
    if key == "dwell_time":
        return wording.dwell(value)
    if key == "resolution":
        return wording.resolution(value)
    if key == "scan_rotation":
        return wording.degrees(value)
    if key == "beam_type":
        return wording.beam_short(value)
    if key == "overlap" and isinstance(value, (int, float)):
        return f"{wording.number(value * 100)}%"
    if key == "path":
        return wording.folder(value)
    return wording.short(value)


def _image_label(key: str) -> str:
    return _IMAGE_LABELS.get(key) or wording.words(key).capitalize()


def _image_settings(settings: Dict[str, Any]) -> str:
    """An image's field of view, resolution and dwell, as one phrase."""
    hfw = settings.get("hfw")
    resolution = settings.get("resolution")
    dwell = settings.get("dwell_time")
    return ", ".join(
        part
        for part in (
            f"{wording.distance(hfw)} wide" if hfw is not None else "",
            wording.resolution(resolution) if resolution is not None else "",
            f"{wording.dwell(dwell)} dwell" if dwell is not None else "",
        )
        if part
    )


def _image_rows(settings: Dict[str, Any], skip: Sequence[str] = ()) -> List[tuple]:
    """An image's settings as (label, value) rows, in display units."""
    return [
        (_image_label(key), _image_value(key, value))
        for key, value in settings.items()
        if key not in skip and value is not None
    ]


def _beam_device(a: Dict[str, Any]) -> str:
    return "ion" if str(a.get("beam_type", "")).upper() == "ION" else "electron"


def _beam_acquire_image(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    settings = a.get("image_settings")
    saved = f"\nSave it in {a['path']}" if a.get("path") else ""
    if not isinstance(settings, dict) or not settings:
        return {
            "what": wording.beam_name(a),
            "change": "Take an image, as the beam is set now" + saved,
        }
    settings = {**_IMAGE_DEFAULTS, **settings}
    beam = _beam_device(a)
    rows = []
    for key, value in settings.items():
        if key == "beam_type" or value is None:
            continue
        new = _image_value(key, value)
        now = (
            ctx.get(f"/microscope/devices/{beam}/{key}")
            if key in _BEAM_PARAMETERS
            else None
        )
        if isinstance(now, dict) and "value" in now:
            rows.append((_image_label(key), _image_value(key, now["value"]), new))
        else:
            rows.append((_image_label(key), "", new))
    return {
        "what": wording.beam_name(a),
        # The caveat on a line of its own: a line about it after the fact
        # (Recent) reads the first only.
        "change": f"Take an image: {_image_settings(settings)}\n"
        "These settings stay on the beam" + saved,
        "details": wording.table(("Setting", "Now", "New"), rows),
    }


_IMAGING = "/microscope/services/imaging/queries/"


def _beam_acquire_overview(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    settings = a.get("settings") if isinstance(a.get("settings"), dict) else {}
    image = settings.get("image_settings") or {}
    rows, cols = settings.get("nrows"), settings.get("ncols")
    beam = wording.beam_short(image.get("beam_type"))
    hfw = image.get("hfw")
    grid = f"{rows} × {cols} " if rows and cols else ""
    change = f"Take a {grid}{beam} overview{_tiles_on(rows, cols, settings.get('tile_mask'))}"
    size = _overview_size(settings)
    if hfw is not None and size:
        change += f", {wording.distance(hfw)} tiles ({size})"
    change += ", here"
    estimate = wording.result_of(
        ctx.get(_IMAGING + "estimate_overview", {"settings": settings})
    )
    if isinstance(estimate, dict) and estimate.get("scan_time"):
        change += f", scanning about {wording.seconds(float(estimate['scan_time']))}"
        change += " and the stage moves"
    table = _image_rows(
        {
            k: v
            for k, v in settings.items()
            if not isinstance(v, dict) and k != "tile_mask"
        }
    ) + _image_rows(image)
    if a.get("path"):
        table.append(
            (
                "Saved as",
                f"{a.get('name') or 'overview'} in {wording.folder(a['path'])}",
            )
        )
    return {
        "what": "Overview",
        "change": change,
        "details": wording.table(("Setting", "Value"), table),
    }


def _tiles_on(rows: Any, cols: Any, mask: Any) -> str:
    """ " with 15 of its 25 tiles" when *mask* turns some off; else empty."""
    try:
        total = int(rows) * int(cols)
        on = sum(1 for row in mask for enabled in row if enabled)
    except (TypeError, ValueError):
        return ""
    return f" with {on} of its {total} tiles" if on < total else ""


def _overview_size(settings: Dict[str, Any]) -> Optional[str]:
    """What an overview covers, as its estimate says: "about 420 × 280 µm"."""
    from fibsem.structures.settings import OverviewAcquisitionSettings

    try:
        parsed = OverviewAcquisitionSettings.from_dict(settings)
        across, down = float(parsed.total_fov_x), float(parsed.total_fov_y)
    except Exception:  # noqa: BLE001 - a partial settings dict says less
        return None
    width, height = wording.distance(across), wording.distance(down)
    if width.split()[-1] == height.split()[-1]:
        width = width.split()[0]
    return f"about {width} × {height}"


def _beam_acquire_focus_stack(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    image = a.get("image_settings") if isinstance(a.get("image_settings"), dict) else {}
    steps = a.get("n_steps") or 3
    change = f"Take an image focused in {steps} strips"
    if image.get("hfw") is not None:
        change += f", {wording.distance(image['hfw'])} wide"
    if a.get("auto_focus", True):
        change += ", each strip focused first"
    return {"what": wording.beam_name(image), "change": change}


def _beam_auto_focus(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    """The sweep's passes, as the operator reads them: "50 µm in 5 µm steps"."""
    if str(a.get("method") or "sweep").lower() == "instrument":
        return {
            "what": wording.beam_name(a),
            "change": "Autofocus with the instrument's routine",
        }
    settings = a.get("settings") if isinstance(a.get("settings"), dict) else {}
    passes = settings.get("passes")
    if not isinstance(passes, list) or not passes:
        from fibsem.autofunctions.autofocus import AutoFocusSettings

        passes = [vars(p) for p in AutoFocusSettings().passes]
    steps = [
        f"{wording.distance(p.get('search_range'))} in {wording.distance(p.get('step_size'))} steps"
        for p in passes
        if isinstance(p, dict) and p.get("enabled", True)
    ]
    change = "Focus by sweeping the working distance"
    if steps:
        change += ": " + ", then ".join(steps)
    if a.get("hfw") is not None:
        change += f", images {wording.distance(a['hfw'])} wide"
    return {"what": wording.beam_name(a), "change": change}


def _channel_setting(key: str, value: Any) -> str:
    """One ChannelSettings field as the FM panel shows it: wavelengths in nm
    (as stored), the emission filter by name, exposure in ms, power and gain as
    % of full."""
    if isinstance(value, dict) and "name" in value:
        return f"{wording.words(key)} {wording.short(value['name'])}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{wording.words(key)} {wording.short(value)}"
    if key.endswith("wavelength"):
        shown = f"{wording.number(number)} nm"
    elif key == "exposure_time":
        shown = f"{wording.number(number * 1e3)} ms"
    elif key in ("power", "gain"):
        shown = f"{wording.number(number * 100)}%"
    else:
        shown = wording.short(value)
    return f"{wording.words(key).replace('time', '').strip()} {shown}"


def _z_range(z: Dict[str, Any]) -> str:
    """A z-stack's range about the objective, "−10 µm to +10 µm in 1 µm steps
    (21 planes)"; "" when the call doesn't say."""
    try:
        zmin, zmax, zstep = float(z["zmin"]), float(z["zmax"]), float(z["zstep"])
    except (KeyError, TypeError, ValueError):
        return ""
    sign = "" if zmax < 0 else "+"
    text = f"{wording.minus(wording.distance(zmin))} to {sign}{wording.minus(wording.distance(zmax))}"
    text += f" in {wording.distance(zstep)} steps"
    if zstep > 0 and zmax > zmin:
        text += f" ({int(round((zmax - zmin) / zstep)) + 1} planes)"
    return text


def _fm_channels(a: Dict[str, Any], change: str) -> Dict[str, Any]:
    """*change* in the channels the call names, with their settings as details."""
    channels = [c for c in a.get("channels") or [] if isinstance(c, dict)]
    names = [str(c.get("name") or "channel") for c in channels]
    change += f" in {wording.plural(len(channels), 'channel')}"
    if names:
        change += f" ({wording.names_cut(names)})"
    change += ", here"
    table = [
        (
            str(c.get("name") or ""),
            ", ".join(_channel_setting(k, v) for k, v in c.items() if k != "name"),
        )
        for c in channels
    ]
    return {
        "what": "Fluorescence",
        "change": change,
        "details": wording.table(("Channel", "Settings"), table) if table else None,
    }


def _fm_acquire_overview(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    parameters = a.get("parameters") if isinstance(a.get("parameters"), dict) else {}
    rows, cols = parameters.get("rows"), parameters.get("cols")
    grid = f"{rows} × {cols} " if rows and cols else ""
    tiles = _tiles_on(rows, cols, parameters.get("tile_mask"))
    line = _fm_channels(a, f"Take a {grid}overview{tiles}")
    asked = {k: a[k] for k in ("channels", "parameters", "zparams") if a.get(k)}
    estimate = wording.result_of(ctx.get(_IMAGING + "fm_estimate_overview", asked))
    if isinstance(estimate, dict):
        # its whole run, the stage and objective moves counted
        line["change"] += wording.about(estimate.get("duration"))
    return line


def _fm_acquire_image(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    zparams = a.get("zparams") if isinstance(a.get("zparams"), dict) else None
    if not zparams:
        return _fm_channels(a, "Take an image")
    z = _z_range(zparams)
    return _fm_channels(a, f"Take a z-stack, {z}," if z else "Take a z-stack")


def _auto_focus_call(a: Dict[str, Any]) -> Tuple[Dict[str, Any], dict]:
    """beam_auto_focus: the imaging service's focus sweep, or with method
    instrument its auto_focus (the instrument's own where the beam has one)."""
    method = str(a.get("method") or "sweep").lower()
    if method == "instrument":
        if any(a.get(k) is not None for k in ("hfw", "settings", "path", "name")):
            raise ValueError("hfw, settings, path and name are for method sweep")
        return {"command": "auto_focus"}, kwargs_body(**beam_kwargs(a))
    if method != "sweep":
        raise ValueError(f"method must be sweep or instrument, not {method!r}")
    sweep = beam_kwargs(
        a,
        hfw=a.get("hfw"),
        settings=a.get("settings"),
        path=a.get("path"),
        name=a.get("name"),
    )
    return {"command": "focus_sweep"}, kwargs_body(**sweep)


def _beam_auto_contrast(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {"what": wording.beam_name(a), "change": "Autocontrast"}


def _beam_auto_brightness_contrast(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {"what": wording.beam_name(a), "change": "Auto brightness and contrast"}


def _fm_auto_focus(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    name = a.get("channel_name") or (a.get("settings") or {}).get("channel_name")
    on = f"on {name}" if name else "on the current channel"
    return {"what": "Fluorescence", "change": f"Focus the objective {on}"}


REDUCED_AREA = (
    "a rectangle of the frame to work on, {left, top, width, height} (0-1); "
    "left out, the whole frame"
)
FM_FOCUS_SETTINGS = (
    "AutoFocusSettings dict: passes [{search_range, step_size}] in metres of "
    "objective travel, each centred on the last one's best (default 50 µm in "
    "5 µm steps, then 10 µm in 1 µm steps), method (tenengrad, laplacian, "
    "sobel or variance), reduced_area {left, top, width, height} (0-1), "
    "channel_name"
)


def _fm_auto_focus_call(a: Dict[str, Any]) -> Tuple[Dict[str, Any], dict]:
    """fm_auto_focus: channel_name is the settings' own, given on its own."""
    settings = a.get("settings")
    if a.get("channel_name"):
        settings = dict(settings or {}, channel_name=str(a["channel_name"]))
    return {}, kwargs_body(
        settings=settings,
        path=a.get("path"),
        name=str(a.get("name") or "autofocus"),
        channel=a.get("channel"),
    )


def _beam_turn_on(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {"what": wording.beam_name(a), "change": "Turn it on"}


def _beam_turn_off(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {"what": wording.beam_name(a), "change": "Turn it off"}


def _fm_insert_objective(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    change = "Insert the objective"
    try:
        change += f", at {math.degrees(float(a['tilt'])):.0f}° tilt"
    except (KeyError, TypeError, ValueError):
        pass
    return {"what": "Fluorescence", "change": change}


def _fm_save_focus_position(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    change = "Save the objective position now as the focus position"
    now = ctx.get("/microscope/devices/objective/focus_position")
    if isinstance(now, dict) and isinstance(now.get("value"), (int, float)):
        change += f" (saved now: {now['value'] * 1e6:.1f} µm)"
    return {"what": "Fluorescence", "change": change}


def _fm_retract_objective(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {"what": "Fluorescence", "change": "Retract the objective"}


TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="beam_acquire_image",
        description=(
            "Acquire a fresh image and return an agent-sized JPEG preview with its "
            "scale (hfw, pixel size). With no image_settings the beam images as it "
            "is set now. image_settings set the beam first and stay set; a field "
            "left out takes its default (resolution [1536, 1024], dwell_time 1e-6, "
            "hfw 150e-6), not the beam's current value. Saved only given path: a "
            "folder inside the open experiment (relative to it, e.g. "
            '"02-nice-dassie/checks"), as <filename>_eb or _ib, filename '
            "defaulting to when it was taken; the answer names the saved file, "
            "for evidence and item_measure_thickness. Images what the beam is "
            "centred on: it moves nothing."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/acquire",
        level="drive",
        hardware=True,
        params={
            "beam_type": arg("ELECTRON or ION", default="ELECTRON"),
            "image_settings": arg(
                'optional: ImageSettings fields, e.g. {"hfw": 150e-6, "resolution": [1536, 1024], "dwell_time": 1e-6}',
                dict,
            ),
            "path": "optional folder inside the open experiment to save into",
            "filename": "optional file name, without the _eb/_ib suffix (default: the time)",
        },
        optional=("beam_type", "image_settings", "path", "filename"),
        line=_beam_acquire_image,
        call=lambda a: (
            {},
            command_body(
                beam_kwargs(
                    a,
                    image_settings=a.get("image_settings"),
                    path=a.get("path"),
                    filename=a.get("filename"),
                ),
                preview=True,
            ),
        ),
    ),
    ToolSpec(
        name="beam_auto_contrast",
        description="Run the instrument's own autocontrast routine for a beam (changes detector settings). beam_auto_brightness_contrast is the experimental image-based one.",
        method="POST",
        path="/microscope/services/imaging/commands/autocontrast",
        level="drive",
        hardware=True,
        params={
            "beam_type": arg("ELECTRON or ION", default="ELECTRON"),
            "reduced_area": arg(REDUCED_AREA, dict),
        },
        optional=("beam_type", "reduced_area"),
        line=_beam_auto_contrast,
        call=lambda a: (
            {},
            kwargs_body(**beam_kwargs(a, reduced_area=a.get("reduced_area"))),
        ),
    ),
    ToolSpec(
        name="beam_auto_focus",
        description=(
            "Focus a beam by sweeping its working distance and scoring an image "
            "at each step; it moves to the best one. Answers each step's working "
            "distance and score, the working distance it moved to, and at_edge "
            "when the best score was at an end of the range searched (widen "
            "search_range and run again). By default two passes, 50 µm in 5 µm "
            "steps then 10 µm in 1 µm steps, at the beam's field of view. Every "
            "step is an image: on the ION beam each one doses the sample, so "
            "sweep the FIB only when you need to, with as few steps as will do. "
            "method instrument runs the instrument's own autofocus instead. Where "
            "the working distance can't be set, the instrument's autofocus runs "
            "and the answer says so (focused_with: instrument)."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/{command}",
        result_key="result",
        level="drive",
        hardware=True,
        params={
            "beam_type": arg("ELECTRON or ION", default="ELECTRON"),
            "method": arg("sweep (default) or instrument", default="sweep"),
            "hfw": arg(
                "the probe images' field of view in metres (default the beam's)", float
            ),
            "settings": arg(
                "AutoFocusSettings dict, for the sweep: passes [{search_range, "
                "step_size}] in metres, method (tenengrad, laplacian, sobel or "
                "variance: how an image is scored), probe_resolution [w, h], "
                "probe_dwell_time (s), reduced_area {left, top, width, height} "
                "(0-1), use_autocontrast",
                dict,
            ),
            "path": "a folder inside the open experiment to save the sweep to, for the sweep",
            "name": "the sweep's file name (default autofocus)",
        },
        optional=("beam_type", "method", "hfw", "settings", "path", "name"),
        line=_beam_auto_focus,
        call=_auto_focus_call,
    ),
    ToolSpec(
        name="beam_acquire_overview",
        description=(
            "A tiled SEM or FIB overview centred on centre_position, or where the "
            "stage is: tiles of "
            "image_settings (hfw is one tile's field of view, beam_type the beam) "
            "in nrows x ncols, stitched into one image. The stage moves to each "
            "tile and comes back after."
            + _OVERVIEW_RULE
            + " path must be a folder inside the open "
            "experiment. experiment_render_markers draws the items on it. "
            "beam_estimate_overview first: a big grid at a long dwell runs "
            "for hours. A long command: it answers with its run at once; "
            "service_wait_for_run imaging gives the stitched image (summary, path, "
            "preview), the tiles' paths and the overview: the resolution used, tiles "
            "taken and skipped, how far the tiles' steps were off, whether "
            "register_tiles moved them (largest_shift, metres) and why, the warning, "
            "and the record file with every tile's planned and read-back position; "
            "service_call_command imaging stop, "
            "with a reason, ends it before the next tile."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/acquire_overview",
        result_key="run",
        level="drive",
        hardware=True,
        params={
            "settings": arg(
                "OverviewAcquisitionSettings dict: image_settings, nrows, ncols, "
                "overlap, autofocus_mode, autocontrast_mode, focus_stack_settings, "
                "register_tiles (experimental: place each tile where its picture "
                "says it is, not where the stage step put it, when no tile moves "
                "more than 15 µm; default false)",
                dict,
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": arg(
                "the file name, stamped with the time (default overview)",
                default="overview",
            ),
            "centre_position": arg(OVERVIEW_CENTRE, dict),
        },
        optional=("name", "centre_position"),
        line=_beam_acquire_overview,
        call=lambda a: (
            {},
            kwargs_body(
                settings=dict(a["settings"]),
                path=str(a["path"]),
                name=str(a.get("name") or "overview"),
                centre_position=a.get("centre_position"),
            ),
        ),
    ),
    ToolSpec(
        name="beam_estimate_overview",
        description=(
            "What beam_acquire_overview with these settings would do: how many tiles, "
            "the beam-on scan time in seconds, the field of view in metres, the "
            "resolution the beam scans the tiles at (requested_resolution when "
            "it is not the one asked), and the tiles the stage can't reach from "
            "where it is (unreachable, [row, col]: turn them off in tile_mask or "
            "the run is refused). Not the run's length: the stage moves between "
            "tiles add to it, often more than the scanning. Moves nothing."
        ),
        method="POST",
        path="/microscope/services/imaging/queries/estimate_overview",
        result_key="estimate",
        level="watch",
        params={
            "settings": arg(
                "OverviewAcquisitionSettings dict, as beam_acquire_overview", dict
            )
        },
        call=lambda a: ({}, kwargs_body(settings=dict(a["settings"]))),
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
        path="/microscope/services/imaging/commands/acquire_focus_stack",
        result_key="run",
        level="drive",
        hardware=True,
        params={
            "image_settings": arg(
                "ImageSettings dict: beam_type, hfw, resolution, dwell_time, filename",
                dict,
            ),
            "path": "a folder inside the open experiment to save it in",
            "n_steps": arg("how many strips (default 3)", int, default=3),
            "auto_focus": arg(
                "focus each strip first (default true)", bool, default=True
            ),
        },
        optional=("n_steps", "auto_focus"),
        line=_beam_acquire_focus_stack,
        call=lambda a: (
            {},
            kwargs_body(
                image_settings=dict(a["image_settings"]),
                path=str(a["path"]),
                n_steps=int(a.get("n_steps") or 3),
                auto_focus=a.get("auto_focus", True),
            ),
        ),
    ),
    ToolSpec(
        name="fm_auto_focus",
        description=(
            "Focus the fluorescence microscope: sweep the objective, a coarse then "
            "a fine pass, scoring a frame at each step on the current channel, as "
            "the FM tab's auto-focus does, and leave it at the best position. "
            "Refused while the FM is in use or can't see the sample from this "
            "pose (move to the FM pose first). Start from the item's objective "
            "position (item_move_to_objective_position) where it has one. A long command: it answers with "
            "its run at once; service_wait_for_run imaging gives the best objective "
            "position and the scores, or status no_focus when the scores were flat "
            "(the objective goes back where it started: move nearer focus or check "
            "the channel); service_call_command imaging stop, with a reason, ends it."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/fm_auto_focus",
        result_key="run",
        level="drive",
        hardware=True,
        params={
            "channel_name": "the channel to focus on; left out, the current one",
            "settings": arg(FM_FOCUS_SETTINGS, dict),
            "channel": arg(
                "ChannelSettings dict to focus in, as fm_acquire_image's channels "
                "take (excitation_wavelength, emission_filter, power, "
                "exposure_time); left out, the channel's own",
                dict,
            ),
            "path": "a folder inside the open experiment to save the sweep to",
            "name": arg(
                "the sweep's file name (default autofocus)", default="autofocus"
            ),
        },
        optional=("channel_name", "settings", "channel", "path", "name"),
        line=_fm_auto_focus,
        call=_fm_auto_focus_call,
    ),
    ToolSpec(
        name="beam_turn_on",
        description=(
            "Switch a beam on, only when you are sure it is off: read its on "
            "parameter first (false). The ion beam "
            "switches itself off after a while idle (30 minutes on some "
            "instruments), so check on before imaging or milling. Switching on "
            "can take a while (a minute or more): read on again, and let it "
            "settle before you image."
        ),
        method="POST",
        path="/microscope/devices/{device}/commands/turn_on",
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
        line=_beam_turn_on,
        call=lambda a: ({"device": _beam_device(a)}, kwargs_body()),
    ),
    ToolSpec(
        name="beam_turn_off",
        description=(
            "Switch a beam off. Only when you are sure: the work is done or the "
            "operator asked. Switching off and back on can each take a while, and a source "
            "switched back on needs time to settle (a plasma source restrikes). "
            "A run's turn_beams_off option switches both off when the run ends."
        ),
        method="POST",
        path="/microscope/devices/{device}/commands/turn_off",
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
        line=_beam_turn_off,
        call=lambda a: ({"device": _beam_device(a)}, kwargs_body()),
    ),
    ToolSpec(
        name="fm_insert_objective",
        description=(
            "Insert the FM objective over the sample, for FM imaging, as the FM "
            "tab's Insert does. tilt (radians) tilts the stage there first with "
            "the objective out, unless it is within half a degree already: the "
            "tab offers 0 and -pi (-180°). Left out, it inserts where the stage "
            "stands. A tilt is refused when the stage is not at the FM. Then go "
            "to focus (item_move_to_objective_position or fm_auto_focus)."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/insert_objective",
        level="drive",
        hardware=True,
        params={
            "tilt": arg("optional stage tilt to insert at, radians (0 or -pi)", float)
        },
        optional=("tilt",),
        line=_fm_insert_objective,
        call=lambda a: (
            {},
            kwargs_body(tilt=None if a.get("tilt") is None else float(a["tilt"])),
        ),
    ),
    ToolSpec(
        name="fm_save_focus_position",
        description=(
            "Save where the FM objective is now as its focus position, as the FM "
            "tab's Set Focus Position does: an FM overview set to start from the "
            "saved focus position starts there, and is refused while none is saved. "
            "Focus first (fm_auto_focus). Read it with device_get_parameter "
            "objective focus_position (null when none is saved)."
        ),
        method="POST",
        path="/microscope/devices/objective/commands/save_focus_position",
        level="drive",
        hardware=True,
        line=_fm_save_focus_position,
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="fm_retract_objective",
        description=(
            "Retract the FM objective away from the sample, as the FM tab's "
            "Retract does. Do it before any move that changes the pose. "
            "item_move_to_objective_position and the FM imaging tools need it "
            "inserted again."
        ),
        method="POST",
        path="/microscope/services/stage_movement/commands/retract_objective",
        level="drive",
        hardware=True,
        line=_fm_retract_objective,
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="fm_acquire_image",
        description=(
            "A fluorescence image where the stage and objective are, in each of "
            "the channels; with zparams, a z-stack about the objective's position, "
            "which it comes back to after. Refused while the FM is in use or "
            "can't image the sample from here."
            + NEAR_FOCUS
            + " path must be a folder inside "
            "the open experiment. A long command: it answers with its run at "
            "once; service_wait_for_run imaging gives the image's path and "
            "summary, and substituted when a channel's excitation wavelength "
            "isn't on the filter set and the nearest was used; "
            "service_call_command imaging stop, with a reason, ends it "
            "before the next frame and saves nothing."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/fm_acquire_image",
        result_key="run",
        level="drive",
        hardware=True,
        params={
            "channels": arg(
                "list of ChannelSettings dicts: name, excitation_wavelength (nm), "
                "emission_filter ({name}, a name from the filter set's choices), "
                "power (fraction of full, 0-1), exposure_time (s)",
                List[dict],
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": arg(
                "the file name, stamped with the time (default image)", default="image"
            ),
            "zparams": arg(
                "ZParameters dict (zmin, zmax, zstep, metres about the objective's "
                "position) for a z-stack; left out, one plane",
                dict,
            ),
        },
        optional=("name", "zparams"),
        line=_fm_acquire_image,
        call=lambda a: (
            {},
            kwargs_body(
                channels=[dict(c) for c in a["channels"]],
                path=str(a["path"]),
                name=str(a.get("name") or "image"),
                zparams=a.get("zparams"),
            ),
        ),
    ),
    ToolSpec(
        name="fm_acquire_overview",
        description=(
            "A tiled fluorescence overview centred on centre_position, or where the "
            "stage is: rows x cols "
            "camera fields in each of the channels, stitched into one mosaic. The "
            "stage moves to each tile, and it and the objective come back after. "
            "The objective must be in, at the FM pose; refused while the FM is in "
            "use."
            + NEAR_FOCUS
            + _OVERVIEW_RULE
            + " path must be a folder inside the open experiment. "
            "fm_estimate_overview first: tiles take seconds to minutes each. A "
            "long command: it answers with its run at once; service_wait_for_run imaging "
            "gives the mosaic's path and summary, the tiles' paths and the overview, "
            "as beam_acquire_overview's; "
            "service_call_command imaging stop, with a reason, ends it before the "
            "next tile."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/fm_acquire_overview",
        result_key="run",
        level="drive",
        hardware=True,
        params={
            "channels": arg(
                "list of ChannelSettings dicts: name, excitation_wavelength (nm), "
                "emission_filter ({name}, a name from the filter set's choices), "
                "power (fraction of full, 0-1), exposure_time (s)",
                List[dict],
            ),
            "parameters": arg(
                "OverviewParameters dict: rows, cols, overlap, autofocus_mode "
                "(none, once, each_row, each_tile), use_zstack, register_tiles "
                "(experimental, as beam_acquire_overview's; default false)",
                dict,
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": arg(
                "the file name, stamped with the time (default overview)",
                default="overview",
            ),
            "zparams": arg(
                "ZParameters dict (zmin, zmax, zstep), with use_zstack", dict
            ),
            "autofocus_settings": arg(
                "the sweep for parameters.autofocus_mode: " + FM_FOCUS_SETTINGS, dict
            ),
            "centre_position": arg(OVERVIEW_CENTRE, dict),
        },
        optional=("name", "zparams", "autofocus_settings", "centre_position"),
        line=_fm_acquire_overview,
        call=lambda a: (
            {},
            kwargs_body(
                channels=[dict(c) for c in a["channels"]],
                parameters=dict(a["parameters"]),
                path=str(a["path"]),
                name=str(a.get("name") or "overview"),
                zparams=a.get("zparams"),
                autofocus_settings=a.get("autofocus_settings"),
                centre_position=a.get("centre_position"),
            ),
        ),
    ),
    ToolSpec(
        name="fm_estimate_overview",
        description=(
            "What fm_acquire_overview with these arguments would do: how many "
            "tiles and images, and how long it runs in seconds, stage and "
            "objective moves and focus sweeps counted, and the tiles the stage "
            "can't reach from where it is (unreachable, [row, col]: turn them off "
            "in parameters.tile_mask). Moves nothing." + NEAR_FOCUS
        ),
        method="POST",
        path="/microscope/services/imaging/queries/fm_estimate_overview",
        result_key="estimate",
        level="watch",
        params={
            "channels": arg(
                "list of ChannelSettings dicts, as fm_acquire_overview", List[dict]
            ),
            "parameters": arg("OverviewParameters dict, as fm_acquire_overview", dict),
            "zparams": arg("ZParameters dict, as fm_acquire_overview", dict),
            "autofocus_settings": arg(
                "AutoFocusSettings dict, as fm_acquire_overview", dict
            ),
        },
        optional=("zparams", "autofocus_settings"),
        call=lambda a: (
            {},
            kwargs_body(
                channels=[dict(c) for c in a["channels"]],
                parameters=dict(a["parameters"]),
                zparams=a.get("zparams"),
                autofocus_settings=a.get("autofocus_settings"),
            ),
        ),
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
        result_key="result",
        level="drive",
        hardware=True,
        params={
            "beam_type": arg("ELECTRON or ION", default="ELECTRON"),
            "settings": arg(
                "BrightnessContrastSettings dict: max_images (default 6), "
                "max_clipped (fraction at either limit, default 0.01), "
                "target_centre and target_spread (0-1), probe_resolution [w, h], "
                "probe_dwell_time (s), reduced_area {left, top, width, height} "
                "(0-1), start_from_last (default true)",
                dict,
            ),
            "path": "a folder inside the open experiment to save the images to",
            "name": "the file name (default brightness_contrast)",
        },
        optional=("beam_type", "settings", "path", "name"),
        line=_beam_auto_brightness_contrast,
        call=lambda a: (
            {},
            kwargs_body(
                **beam_kwargs(
                    a,
                    settings=a.get("settings"),
                    path=a.get("path"),
                    name=a.get("name"),
                )
            ),
        ),
    ),
)
