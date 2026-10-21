"""The tools on an experiment's items (lamellae): their records, poses and task configs."""

import os
import re
from typing import Any, Dict, Optional, Tuple

from fibsem.ai.agent.catalog.spec import RETRACT_FIRST_SAFE_MOVE, ToolSpec
from fibsem.server.parts import ITEMS, LAMELLAE, MILLING_STAGES, POSES
from fibsem.util import wording


def _fit_quality(result: Any) -> str:
    """The fit a correlation's point comes from, on a line of its own: its RMS
    and verdict, a bad one said plainly (FIB-1492)."""
    from fibsem.correlation.verdict import FitEvidence, verdict

    data = result.input_data
    pixel_size = data.fib_image_pixel_size if data is not None else None
    parts = []
    if result.rms_error is not None and pixel_size:
        parts.append(f"RMS {wording.distance(result.rms_error * pixel_size)}")
    found = None
    if result.diagnostics:
        try:
            found = verdict(FitEvidence.from_dict(result.diagnostics))
        except Exception:  # noqa: BLE001 - the line reads without it
            found = None
    if found is not None:
        parts.append(found.headline.rstrip("."))
    if not parts:
        return ""
    line = "\nFit: " + ", ".join(parts)
    if found is not None and found.tier == "bad":
        line += "\nWarning: a bad fit; the point may be off"
    return line


def _item_route(item: Any) -> str:
    from urllib.parse import quote

    return f"/app/experiment/items/{quote(str(item), safe='')}"


def _pose_of(detail: Any, pose: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(detail, dict):
        return None
    position = (detail.get("poses") or {}).get(pose)
    return position if isinstance(position, dict) else None


def _overview_name(overview: Dict[str, Any]) -> str:
    """ "the Grid-01 overview (04:58)": whose, and when it was taken, which the
    file's stamp says; the file name only when it has no stamp."""

    grid = overview.get("grid_name")
    filename = str(overview.get("filename") or "")
    stamp = re.search(r"(\d{2})-(\d{2})-\d{2}", os.path.basename(filename))
    if stamp:
        name = f"the {grid} overview" if grid else "the overview"
        return f"{name} ({stamp.group(1)}:{stamp.group(2)})"
    name = filename or "the overview"
    return f"{name} of {grid}" if grid else name


def _add_item(a: Dict[str, Any], stage_now: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    named = f" '{a['name']}'" if a.get("name") else ""
    overview = a.get("overview")
    position = a.get("stage_position")
    if isinstance(overview, dict):
        where = _overview_name(overview)
        if overview.get("fraction") is not None:
            fx, fy = (list(overview["fraction"]) + [None, None])[:2]
            at = f"{wording.number(float(fx) * 100)}% across, {wording.number(float(fy) * 100)}% down"
        else:
            at = f"pixel {wording.short(overview.get('px'))}"
        change = f"Add{named} on {where}, {at}"
    elif isinstance(position, dict):
        change = f"Add{named} at {wording.position_text(position)}"
    else:
        change = f"Add{named} at the stage position now"
        if stage_now:
            change += f": {wording.position_text(stage_now)}"
    return {"what": "Lamella", "change": change}


def _item_update_task_config(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    task = a.get("task_name")
    return {
        "what": f"{item} · {task}",
        "change": wording.patch_lines(a.get("patch"), ctx.document("config")),
    }


def _item_update(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    item = a.get("item_name")
    return {
        "what": str(item),
        "change": wording.item_patch_lines(a.get("patch"), ctx.document(None)),
    }


def _item_add(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return _add_item(a, None if a.get("stage_position") else ctx.stage_now())


def _item_remove(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    item = a.get("item_name")
    return {
        "what": str(item),
        "change": "Remove it from the experiment. Its files stay on disk",
    }


def _item_move_to_pose(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    item = a.get("item_name")
    pose = a.get("pose")
    change = f"Move the stage to its {pose} pose"
    target = _pose_of(ctx.get(_item_route(item)), pose)
    if target:
        change += f": {wording.position_text(target)}"
    now = ctx.stage_now()
    if target and now:
        change += f"\nfrom {wording.position_text(now)}"
    return {"what": str(item), "change": change}


def _item_apply_correlation(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    run = str(a.get("run_path") or "")
    name = os.path.basename(os.path.normpath(run)) if run else run
    change = f"Move its point of interest to the correlation saved in {name}"
    try:
        from fibsem.correlation.history import load_run

        result = load_run(run).result
        poi = result.poi[0].px_m
        change += (
            f": ({wording.minus(f'{poi.x * 1e6:.2f}')}, {wording.minus(f'{poi.y * 1e6:.2f}')}) "
            "µm from the image centre"
        )
        change += _fit_quality(result)
    except Exception:  # the request still reads without the run's numbers
        pass
    change += "\nThe patterns that follow the point move with it"
    return {"what": str(item), "change": change}


def _item_set_pose(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    item = a.get("item_name")
    pose = a.get("pose")
    change = f"Set its {pose} pose to where the stage is now"
    now = ctx.stage_now()
    if now:
        change += f": {wording.position_text(now)}"
    old = _pose_of(ctx.get(_item_route(item)), pose)
    if old:
        change += f"\nwas {wording.position_text(old)}"
    if str(pose).upper() == "MILLING":
        change += "\nIts milling angle and fluorescence pose follow"
    return {"what": str(item), "change": change}


def _item_set_objective_position(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    return {
        "what": str(item),
        "change": "Store the objective's position now on its fluorescence pose",
    }


def _item_set_position(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    item = a.get("item_name")
    overview = a.get("overview")
    if isinstance(overview, dict):
        to = f"a point on {_overview_name(overview)}"
    else:
        to = wording.position_text(a.get("stage_position") or {})
    change = f"Move it to {to}"
    old = _pose_of(ctx.get(_item_route(item)), "MILLING")
    if old:
        change += f"\nwas {wording.position_text(old)}"
    change += "\nIts milling angle and fluorescence pose follow; the stage stays"
    return {"what": str(item), "change": change}


def _item_set_objective_position_on_all(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    return {
        "what": "Every lamella",
        "change": f"Store {item}'s objective position on every fluorescence pose",
    }


def _item_move_to_objective_position(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    return {
        "what": str(item),
        "change": "Move the FM objective to its stored position",
    }


def _item_apply_protocol(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    tasks = wording.names_text(a.get("task_names"), "every task")
    return {
        "what": str(item),
        "change": f"Copy the protocol's settings for {tasks}",
    }


def _item_reorder_milling_stages(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    task = a.get("task_name")
    return {
        "what": f"{item} · {task}",
        "change": f"Reorder {a.get('milling_key')} stages: {wording.names_text(a.get('order'), 'as they are')}",
    }


TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="item_list",
        returns="ExperimentSummary",
        description="Per-item summary of the open experiment.",
        method="GET",
        path="/app/experiment",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="item_get",
        returns="ItemDetail",
        description="Everything durable about one item: status, failure flag, POI, alignment area, milling angle, and where its poses put the stage.",
        method="GET",
        path="/app/experiment/items/{item_name}",
        level="watch",
        router="app",
        part=ITEMS,
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="item_list_outputs",
        returns="TaskOutputs",
        description="The files an item's task runs produced (reference images by role). tasks has every run with its status; completed_tasks names only the runs that completed.",
        method="GET",
        path="/app/experiment/items/{item_name}/outputs",
        level="watch",
        router="app",
        part=ITEMS,
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="item_get_output_image",
        description="One saved image of an item, as an image you can look at: a 768 px JPEG preview of a file item_list_outputs lists (stacks and channels max-projected). A preview, not the data. A beam image comes with its metadata block (pixel size, preview_pixelsize, stage position, beam, detector). A name it neither lists nor finds in the item's folder is refused with the valid ones.",
        method="GET",
        path="/app/experiment/items/{item_name}/outputs/{filename}",
        level="watch",
        router="app",
        part=ITEMS,
        params={
            "item_name": "the item (lamella) name",
            "filename": "a basename from item_list_outputs",
        },
    ),
    ToolSpec(
        name="item_get_stack",
        description="An item's images under one role as a stack, in iteration order: a volume position's slices (role slice, SEM; slice_fib, the FIB image its milling kept). Each has iteration, filename (for item_get_output_image), acquired_at, pixel_size and hfw (m), and z (m into the position: iteration times the slice thickness; null for an app that doesn't slice).",
        method="GET",
        path="/app/experiment/items/{item_name}/stack",
        level="watch",
        router="app",
        part=ITEMS,
        params={
            "item_name": "the item name",
            "role": "the output role, e.g. slice or slice_fib",
        },
    ),
    ToolSpec(
        name="item_get_history",
        returns="ItemHistory",
        description="The item's History, as the operator's History tab shows it: one run per task run (plus each session's operations outside a task), each with status, times, who decided its proposals, its shown images (paths relative to the experiment folder) and its operations (align, autofocus, mill with each stage's strategy and monitors), filtered as the tab filters. operations_open says a run's operations need a look (one did not complete).",
        method="GET",
        path="/app/experiment/items/{item_name}/history",
        level="watch",
        router="app",
        part=ITEMS,
        params={
            "item_name": "the item name",
            "show": "all (default), images (runs that left images) or operations (runs that recorded operations)",
            "status": "all (default), failed or cancelled",
            "task": "one task name, or omit for every task",
            "decided_by": "anyone (default), agent or operator: the runs whose proposals they decided",
            "order": "oldest (default) or newest first",
        },
        optional=("show", "status", "task", "decided_by", "order"),
    ),
    ToolSpec(
        name="item_measure_thickness",
        returns="Thickness",
        description="The lamella's thickness, read off any saved FIB image of the item inside the experiment (full resolution): a task's reference (item_list_outputs names them; they sit in the item's folder, <item_name>/<file>) or one you took at the item's milling position. The bright band between the trenches, its width at half its peak brightness column by column, in a window around the item's POI (a point in the image's own frame, so the image must be centred where the item's tasks image it), sized from the item's own trench patterns (window_from says; width_um / height_um override). Answers thickness in nm (mean, median, min, max, by thirds), wedge (nm per um along the lamella), angle_deg (the band's rotation, counter-clockwise), straightness, centre_offset_nm (band above the POI is positive), continuity (columns dropped and where: holes or breaks), contrast (band vs trench brightness), other_bands (other bright lines in the window, e.g. trench residue, with their offsets from the POI), resolution (band_px, flagged under 8 px with a suggested field width), uncertainty_nm (one pixel) and warnings (low_contrast under a 2x ratio, several_bands, under_resolved, no_band: check the image before using the numbers), plus a plot: the window with the band's edges, and thickness along the lamella. A top-down estimate. Nothing moves; the result (data.json, window.tif, plot.png) is saved in a thickness folder beside the image, at saved_to, and replaced when the same image is measured again.",
        method="POST",
        path="/app/experiment/items/{item_name}/thickness",
        level="watch",
        router="app",
        part=LAMELLAE,
        params={
            "item_name": "the item (lamella) name",
            "path": "a FIB image file path inside the experiment folder",
            "width_um": "window along the lamella, um (default: from the trench patterns)",
            "height_um": "window across the lamella, um (default: from the trench patterns)",
        },
        optional=("width_um", "height_um"),
    ),
    ToolSpec(
        name="item_get_task_config",
        returns="TaskConfigRead",
        description="One item's own copy of a task configuration (what its run actually executes — tasks may have rewritten it mid-run), with a version token naming exactly this state of it.",
        method="GET",
        path="/app/experiment/items/{item_name}/task_config/{task_name}",
        level="watch",
        router="app",
        part=ITEMS,
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name, as listed by protocol_get",
        },
    ),
    ToolSpec(
        name="item_apply_protocol",
        returns="ConfigEdit",
        description="Re-copy protocol task configs onto an existing item (protocol-level edits only reach items created after them; this brings an existing item up to date). Wholesale replace of that item's copies for the named tasks — omit task_names for all. Running tasks are refused.",
        method="POST",
        path="/app/experiment/items/{item_name}/apply_protocol",
        level="drive",
        router="app",
        part=ITEMS,
        params={
            "item_name": "the item (lamella) name",
            "task_names": "optional list of task names; omitted = every task in the protocol",
        },
        optional=("task_names",),
        line=_item_apply_protocol,
    ),
    ToolSpec(
        name="item_add",
        returns="ItemEdit",
        description="Add a lamella, as the Lamella tab's Add does: at the stage position now (no where; refused while a workflow runs, since the run moves the stage), at stage_position, or under a point of a grid overview (overview = {grid_name, filename: one grid_get lists or any saved image's path in the experiment, px: [x, y] source pixels or fraction: [x, y] 0-1}), which also puts it on that grid. Its poses, milling angle and grid are worked out as for a click, and it is saved. Leave name out: the app's item names (petnames) are the house style; put where the site came from in its description with item_update. Answers with the new item.",
        method="POST",
        path="/app/experiment/items",
        level="drive",
        router="app",
        part=ITEMS,
        params={
            "name": "optional name; the app picks one when omitted",
            "stage_position": "optional {x, y, z, r, t} (metres, radians) to add at",
            "overview": "optional {grid_name, filename, px | fraction}: a point on a grid overview",
        },
        optional=("name", "stage_position", "overview"),
        line=_item_add,
    ),
    ToolSpec(
        name="item_remove",
        returns="ItemEdit",
        description="Remove a lamella from the experiment, as its row's Remove does. Its files stay on disk. Refused for the lamella a run is working on.",
        method="DELETE",
        path="/app/experiment/items/{item_name}",
        level="drive",
        router="app",
        part=ITEMS,
        params={"item_name": "the item (lamella) name"},
        line=_item_remove,
    ),
    ToolSpec(
        name="item_move_to_pose",
        returns="ItemEdit",
        description="Move the stage to one of a lamella's poses (MILLING, FLUORESCENCE, ... as item_get lists), through the safe move that goes flat before a large rotation. Answers with the stage position after."
        + RETRACT_FIRST_SAFE_MOVE,
        method="POST",
        path="/app/experiment/items/{item_name}/move_to",
        level="drive",
        router="app",
        part=POSES,
        hardware=True,
        params={"item_name": "the item (lamella) name", "pose": "the pose name"},
        line=_item_move_to_pose,
    ),
    ToolSpec(
        name="item_set_pose",
        returns="ItemEdit",
        description="Set the stage position now as one of a lamella's poses, as the tab's Set pose does. Setting MILLING moves the lamella: its milling angle and fluorescence pose follow. Saved; answers with the item.",
        method="POST",
        path="/app/experiment/items/{item_name}/poses/{pose}",
        level="drive",
        router="app",
        part=POSES,
        params={"item_name": "the item (lamella) name", "pose": "the pose name"},
        line=_item_set_pose,
    ),
    ToolSpec(
        name="item_apply_correlation",
        returns="ConfigEdit",
        description="Move a lamella's point of interest to a saved correlation's target, as the correlation tab's Continue does: run_path is a run folder in the lamella's Correlation folder, as correlation_correlate saves it with lamella_path. The patterns that follow the point move with it; the correlation is recorded with its run. Saved; answers with the changes.",
        method="POST",
        path="/app/experiment/items/{item_name}/correlation",
        level="drive",
        router="app",
        part=LAMELLAE,
        params={
            "item_name": "the item (lamella) name",
            "run_path": "the run folder correlation_correlate answered",
        },
        line=_item_apply_correlation,
    ),
    ToolSpec(
        name="item_set_objective_position",
        returns="ItemEdit",
        description="Store the FM objective's position now on a lamella's fluorescence pose, as the tab's Use current does: where it is when inserted, else its focus position. Saved. Refused while a workflow runs, since the run moves the objective.",
        method="POST",
        path="/app/experiment/items/{item_name}/objective_position",
        level="drive",
        router="app",
        part=POSES,
        params={"item_name": "the item (lamella) name"},
        line=_item_set_objective_position,
    ),
    ToolSpec(
        name="item_set_objective_position_on_all",
        returns="ItemEdit",
        description="Store a lamella's stored objective position on every lamella with a fluorescence pose, as its Apply to All Lamella does, except the lamella a run is working on (answered as skipped). Refused when it has none stored. Saved; answers with how many it was stored on.",
        method="POST",
        path="/app/experiment/items/{item_name}/objective_position/apply_to_all",
        level="drive",
        router="app",
        part=POSES,
        params={"item_name": "the item (lamella) whose position to copy"},
        line=_item_set_objective_position_on_all,
    ),
    ToolSpec(
        name="item_set_position",
        returns="ItemEdit",
        description="Move a lamella to a point, as moving it on an overview does: to stage_position, or under a point of a grid overview (overview = {grid_name, filename: one grid_get lists or any saved image's path in the experiment, px: [x, y] source pixels or fraction: [x, y] 0-1}). A point on the fluorescence side moves its fluorescence pose and the milling pose follows; otherwise its milling pose moves, with its milling angle, and the fluorescence pose follows. The stage does not move. Refused for the lamella a run is working on. Saved; answers with the item.",
        method="POST",
        path="/app/experiment/items/{item_name}/position",
        level="drive",
        router="app",
        part=POSES,
        params={
            "item_name": "the item (lamella) name",
            "stage_position": "optional {x, y, z, r, t} (metres, radians) to move it to",
            "overview": "optional {grid_name, filename, px | fraction}: a point on a grid overview",
        },
        optional=("stage_position", "overview"),
        line=_item_set_position,
    ),
    ToolSpec(
        name="item_move_to_objective_position",
        returns="ItemEdit",
        description="Move the FM objective to the position stored on a lamella's fluorescence pose, as the tab's objective Move to does: the objective only, not the stage. Refused when nothing is stored or the objective is not inserted.",
        method="POST",
        path="/app/experiment/items/{item_name}/objective/move_to",
        level="drive",
        router="app",
        part=POSES,
        hardware=True,
        params={"item_name": "the item (lamella) name"},
        line=_item_move_to_objective_position,
    ),
    ToolSpec(
        name="item_reorder_milling_stages",
        returns="ConfigEdit",
        description="Reorder the milling stages inside one of an item's task configs ('run the stress relief first'). order must be exactly the current stage names in their new sequence — this can never add, drop, or duplicate a stage. Echo the version from item_get_task_config; stale is refused.",
        method="POST",
        path="/app/experiment/items/{item_name}/task_config/{task_name}/stages/reorder",
        level="drive",
        router="app",
        part=MILLING_STAGES,
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name",
            "milling_key": "which milling config within the task (e.g. mill_rough)",
            "order": "the current stage names in their new sequence",
            "version": "the version token from item_get_task_config",
        },
        line=_item_reorder_milling_stages,
    ),
    ToolSpec(
        name="item_update",
        returns="ItemEdit",
        description="Patch an item's own document with dotted-path edits: poi.x/poi.y (metres, milling frame), alignment_area.left/top/width/height (frame fractions), description, defect.verdict (UNASSESSED/GOOD/REWORK/FAILED) and defect.reason. milling_angle is read-only: it is an outcome of the Setup task, not an input. Echo the version from item_get — a mismatch is refused (409 stale_config); the version also changes when a pose moves, since a POI is in the milling frame. Tasks re-record geometry at their own moments, so an edit here is what the NEXT run starts from, not a permanent override.",
        method="POST",
        path="/app/experiment/items/{item_name}",
        level="drive",
        router="app",
        part=ITEMS,
        params={
            "item_name": "the item (lamella) name",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from item_get",
        },
        line=_item_update,
    ),
    ToolSpec(
        name="item_update_task_config",
        returns="ConfigEdit",
        description="Patch an item's task config with dotted-path edits (e.g. {'milling.mill_rough.stages.0.pattern.depth': 2e-6}). Echo the version from item_get_task_config — a mismatch is refused (409 stale_config): re-read and re-patch. A task currently running for that item is refused (it already copied its config); pending tasks pick the change up at start. Values are validated against the live field's type and declared bounds; the response echoes every change old -> new. Some tasks edit other configs as they run: Setup moves the fiducial to the centre of the alignment area (sync_fiducial_position), Rough Milling moves the Polishing patterns to its own position (sync_polishing_position); turn the sync off in that task's config to keep a hand edit.",
        method="POST",
        path="/app/experiment/items/{item_name}/task_config/{task_name}",
        level="drive",
        router="app",
        part=ITEMS,
        params={
            "item_name": "the item (lamella) name",
            "task_name": "the task name, as listed by protocol_get",
            "patch": (
                "object of dotted-path: value entries to set; a list of settings "
                "(channel_settings) is set whole, as a list of their dicts"
            ),
            "version": "the version token from item_get_task_config",
        },
        line=_item_update_task_config,
    ),
)
