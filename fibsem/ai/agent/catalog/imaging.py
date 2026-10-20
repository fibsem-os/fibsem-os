"""The imaging tools: SEM/FIB images, overviews and focus, and the FM's."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from fibsem.ai.agent.catalog.spec import (
    NEAR_FOCUS,
    OVERVIEW_CENTRE,
    ToolSpec,
    beam_kwargs,
    kwargs_body,
)
from fibsem.util import wording

# What an overview covers, for both overview tools.
_OVERVIEW_RULE = (
    " An overview the operator asks for is the whole grid, from its centre "
    "(grid_move_to first) with the tiles the stage can't reach turned off; a "
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


def _beam_acquire_image(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    settings = a.get("image_settings")
    if not isinstance(settings, dict) or not settings:
        return {
            "what": wording.beam_name(a),
            "change": "Take an image, as the beam is set now",
        }
    settings = {**_IMAGE_DEFAULTS, **settings}
    beam = "ion" if str(a.get("beam_type", "")).upper() == "ION" else "electron"
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
        "These settings stay on the beam",
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
    change = f"Take a {grid}{beam} overview"
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


def _overview_size(settings: Dict[str, Any]) -> Optional[str]:
    """What an overview covers, as its estimate says: "about 420 × 280 µm"."""
    from fibsem.structures import OverviewAcquisitionSettings

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
    line = _fm_channels(a, f"Take a {grid}overview")
    asked = {k: a[k] for k in ("channels", "parameters", "zparams") if a.get(k)}
    estimate = wording.result_of(ctx.get(_IMAGING + "fm_estimate_overview", asked))
    if isinstance(estimate, dict):
        # its whole run, the stage and objective moves counted (FIB-1492)
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
        if a.get("hfw") is not None or a.get("settings") is not None:
            raise ValueError("hfw and settings are for method sweep")
        return {"command": "auto_focus"}, kwargs_body(**beam_kwargs(a))
    if method != "sweep":
        raise ValueError(f"method must be sweep or instrument, not {method!r}")
    sweep = beam_kwargs(a, hfw=a.get("hfw"), settings=a.get("settings"))
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
    on = (
        f"on {a['channel_name']}" if a.get("channel_name") else "on the current channel"
    )
    return {"what": "Fluorescence", "change": f"Focus the objective {on}"}


TOOLS: Tuple[ToolSpec, ...] = (
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
        optional=("beam_type", "image_settings"),
        line=_beam_acquire_image,
        call=lambda a: (
            {},
            {
                "kwargs": beam_kwargs(a, image_settings=a.get("image_settings")),
                "answer": "preview",
            },
        ),
    ),
    ToolSpec(
        name="beam_auto_contrast",
        description="Run the instrument's own autocontrast routine for a beam (changes detector settings). beam_auto_brightness_contrast is the experimental image-based one.",
        method="POST",
        path="/microscope/services/imaging/commands/autocontrast",
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
        optional=("beam_type",),
        line=_beam_auto_contrast,
        call=lambda a: ({}, kwargs_body(**beam_kwargs(a))),
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
        level="drive",
        hardware=True,
        params={
            "beam_type": "ELECTRON or ION",
            "method": "sweep (default) or instrument",
            "hfw": "the probe images' field of view in metres (default the beam's)",
            "settings": (
                "AutoFocusSettings dict, for the sweep: passes [{search_range, "
                "step_size}] in metres, method (tenengrad, laplacian, sobel or "
                "variance: how an image is scored), probe_resolution [w, h], "
                "probe_dwell_time (s), reduced_area {left, top, width, height} "
                "(0-1), use_autocontrast"
            ),
        },
        optional=("beam_type", "method", "hfw", "settings"),
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
            "experiment. beam_estimate_overview first: a big grid at a long dwell runs "
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
        level="drive",
        hardware=True,
        params={
            "settings": (
                "OverviewAcquisitionSettings dict: image_settings, nrows, ncols, "
                "overlap, autofocus_mode, autocontrast_mode, focus_stack_settings, "
                "register_tiles (experimental: place each tile where its picture "
                "says it is, not where the stage step put it, when no tile moves "
                "more than 15 µm; default false)"
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": "the file name, stamped with the time (default overview)",
            "centre_position": OVERVIEW_CENTRE,
        },
        optional=("name", "centre_position"),
        line=_beam_acquire_overview,
        call=lambda a: (
            {},
            kwargs_body(
                settings=dict(a["settings"]),
                path=str(a["path"]),
                name=str(a.get("name") or "overview"),
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
        level="watch",
        params={
            "settings": "OverviewAcquisitionSettings dict, as beam_acquire_overview"
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
        level="drive",
        hardware=True,
        params={
            "image_settings": "ImageSettings dict: beam_type, hfw, resolution, dwell_time, filename",
            "path": "a folder inside the open experiment to save it in",
            "n_steps": "how many strips (default 3)",
            "auto_focus": "focus each strip first (default true)",
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
            "position and the scores; service_call_command imaging stop, with a "
            "reason, ends it."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/fm_auto_focus",
        level="drive",
        hardware=True,
        params={
            "channel_name": "the channel to focus on; left out, the current one",
            "path": "a folder inside the open experiment to save the sweep to",
        },
        optional=("channel_name", "path"),
        line=_fm_auto_focus,
        call=lambda a: (
            {},
            kwargs_body(
                settings=(
                    {"channel_name": str(a["channel_name"])}
                    if a.get("channel_name")
                    else None
                ),
                path=a.get("path"),
            ),
        ),
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
            "summary; service_call_command imaging stop, with a reason, ends it "
            "before the next frame and saves nothing."
        ),
        method="POST",
        path="/microscope/services/imaging/commands/fm_acquire_image",
        level="drive",
        hardware=True,
        params={
            "channels": (
                "list of ChannelSettings dicts: name, excitation_wavelength (nm), "
                "emission_filter ({name}, a name from the filter set's choices), "
                "power (fraction of full, 0-1), exposure_time (s)"
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": "the file name, stamped with the time (default image)",
            "zparams": (
                "ZParameters dict (zmin, zmax, zstep, metres about the objective's "
                "position) for a z-stack; left out, one plane"
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
        level="drive",
        hardware=True,
        params={
            "channels": (
                "list of ChannelSettings dicts: name, excitation_wavelength (nm), "
                "emission_filter ({name}, a name from the filter set's choices), "
                "power (fraction of full, 0-1), exposure_time (s)"
            ),
            "parameters": (
                "OverviewParameters dict: rows, cols, overlap, autofocus_mode "
                "(none, once, each_row, each_tile), use_zstack, register_tiles "
                "(experimental, as beam_acquire_overview's; default false)"
            ),
            "path": "a folder inside the open experiment to save it in",
            "name": "the file name, stamped with the time (default overview)",
            "zparams": "ZParameters dict (zmin, zmax, zstep), with use_zstack",
            "centre_position": OVERVIEW_CENTRE,
        },
        optional=("name", "zparams", "centre_position"),
        line=_fm_acquire_overview,
        call=lambda a: (
            {},
            kwargs_body(
                channels=[dict(c) for c in a["channels"]],
                parameters=dict(a["parameters"]),
                path=str(a["path"]),
                name=str(a.get("name") or "overview"),
                zparams=a.get("zparams"),
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
        level="watch",
        params={
            "channels": "list of ChannelSettings dicts, as fm_acquire_overview",
            "parameters": "OverviewParameters dict, as fm_acquire_overview",
            "zparams": "ZParameters dict, as fm_acquire_overview",
        },
        optional=("zparams",),
        call=lambda a: (
            {},
            kwargs_body(
                channels=[dict(c) for c in a["channels"]],
                parameters=dict(a["parameters"]),
                zparams=a.get("zparams"),
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
        level="drive",
        hardware=True,
        params={"beam_type": "ELECTRON or ION"},
        optional=("beam_type",),
        line=_beam_auto_brightness_contrast,
        call=lambda a: ({}, kwargs_body(**beam_kwargs(a))),
    ),
)
