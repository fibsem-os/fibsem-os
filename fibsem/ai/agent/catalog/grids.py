"""The grid tools: the sample handling service's, and the experiment's grid records."""

from typing import Any, Dict, Tuple

from fibsem.ai.agent.catalog.spec import ToolSpec, kwargs_body
from fibsem.server.parts import GRIDS
from fibsem.util import wording


def _grid_add(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    """Which grids the inventory read will record: those present and not yet
    in the experiment."""
    inventory = wording.result_of(
        ctx.get("/microscope/services/sample_handling/queries/get_inventory")
    )
    records = ctx.get("/app/experiment/grids")
    if not isinstance(inventory, list) or not isinstance(records, dict):
        change = "Record the inventory's grids the experiment has no record of"
        return {"what": "Grids", "change": change}
    known = {g.get("name") for g in records.get("items") or []}
    new = [
        e.get("name")
        for e in inventory
        if e.get("name")
        and e.get("state") in ("loaded", "occupied")
        and e.get("name") not in known
    ]
    if not new:
        change = "Record the inventory's grids: every one is recorded already"
    else:
        change = f"Record {wording.names_cut(new)} from the inventory"
    return {"what": "Grids", "change": change}


def _grid_rename(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {
        "what": f"Grid '{a.get('grid_name')}'",
        "change": f"Rename it '{a.get('name')}', on its slot and in the experiment",
    }


def _grid_set_quality(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    verdict = str(a.get("verdict", "")).upper()
    reason = a.get("reason")
    return {
        "what": f"Grid '{a.get('grid_name')}'",
        "change": "Clear its verdict"
        if verdict == "UNASSESSED"
        else f"Mark it {verdict.lower()}" + (f": {reason}" if reason else ""),
    }


def _grid_set_note(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    note = str(a.get("note") or "")
    return {
        "what": f"Grid '{a.get('grid_name')}'",
        "change": f"Note: {wording.clip(note)}" if note else "Clear its note",
    }


def _grid_remove(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {
        "what": f"Grid '{a.get('grid_name')}'",
        "change": "Stop tracking it; its files stay, its lamellae keep no grid",
    }


TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="grid_list",
        description="Every grid the experiment records: quality verdict, task status, completed tasks, linked item count, and the basename of its latest SEM/FIB/FM overview by role. Records only — not the live slot inventory (grid_get_inventory).",
        method="GET",
        path="/app/experiment/grids",
        level="watch",
        router="app",
        part=GRIDS,
    ),
    ToolSpec(
        name="grid_get",
        description="One grid: its summary plus every task run with the files it recorded by role, and the names of the items linked to it.",
        method="GET",
        path="/app/experiment/grids/{grid_name}",
        level="watch",
        router="app",
        part=GRIDS,
        params={"grid_name": "the grid name"},
    ),
    ToolSpec(
        name="grid_get_output_image",
        description="One saved overview (mosaic) of a grid, as an image you can look at: a 768 px JPEG preview of a file grid_list / grid_get lists. A preview, not the data; grid_get_markers places the items on it. A beam image comes with its metadata block (pixel size, preview_pixelsize, stage position, beam, detector). A name it neither lists nor finds in the experiment is refused with the valid ones.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}",
        level="watch",
        router="app",
        part=GRIDS,
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from grid_list / grid_get, or any saved image's path in the experiment",
        },
    ),
    ToolSpec(
        name="grid_get_stage_position",
        description="The stage position under a point of a grid overview (one grid_get lists, or any saved image in the experiment): the inverse of grid_get_markers, from the overview's own metadata, so it holds wherever the stage is now. Give x, y in source pixels (as the markers use), or fx, fy from 0 to 1 for a point read off a preview. Nothing moves.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}/stage_position",
        level="watch",
        router="app",
        part=GRIDS,
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from grid_list / grid_get, or any saved image's path in the experiment",
            "x": "source-pixel x (with y)",
            "y": "source-pixel y, +y down (with x)",
            "fx": "fraction across the image, 0-1 (with fy)",
            "fy": "fraction down the image, 0-1 (with fx)",
        },
        optional=("x", "y", "fx", "fy"),
    ),
    ToolSpec(
        name="grid_get_markers",
        description="Where the experiment's items fall on one overview (one grid_get lists, or any saved image in the experiment), in source-image pixels with each item's status, plus the image's scale block (size, pixel size, field width, stage position). Computed server-side; never re-derive stage geometry. Items that cannot be placed are listed with the reason.",
        method="GET",
        path="/app/experiment/grids/{grid_name}/outputs/{filename}/markers",
        level="watch",
        router="app",
        part=GRIDS,
        params={
            "grid_name": "the grid name",
            "filename": "an overview basename from grid_list / grid_get, or any saved image's path in the experiment",
        },
    ),
    ToolSpec(
        name="grid_add",
        description=(
            "Record every grid in the inventory that the experiment has no record "
            "of, as the Grids tab's inventory read does: the loader's last scan "
            "(grid_run_inventory scans again); nothing moves. Answers the names "
            "added and every grid the experiment now records."
        ),
        method="POST",
        path="/app/experiment/grids",
        level="drive",
        router="app",
        part=GRIDS,
        line=_grid_add,
    ),
    ToolSpec(
        name="grid_rename",
        description=(
            "Rename a grid as the Grids tab's Rename does: the slot that holds it "
            "first (the loader's slot description, or the holder's record), then "
            "the experiment's record, with its history and lamellae. Refused for "
            "a grid that has run (its folder and images carry the name), a name "
            "another grid has, and while a run is on it or one of its lamellae "
            "or has it queued."
        ),
        method="POST",
        path="/app/experiment/grids/{grid_name}/rename",
        level="drive",
        router="app",
        part=GRIDS,
        params={
            "grid_name": "the grid's name now, from grid_list or grid_get_inventory",
            "name": "its new name",
        },
        line=_grid_rename,
    ),
    ToolSpec(
        name="grid_set_quality",
        description="Judge a grid, as its card's quality menu does: GOOD, REWORK or FAILED with a reason the operator reads, or UNASSESSED to clear it. Signed as the agent's; saved.",
        method="POST",
        path="/app/experiment/grids/{grid_name}/quality",
        level="drive",
        router="app",
        part=GRIDS,
        params={
            "grid_name": "the grid name, from grid_list",
            "verdict": "GOOD, REWORK, FAILED, or UNASSESSED to clear",
            "reason": "why, for the operator",
        },
        optional=("reason",),
        line=_grid_set_quality,
    ),
    ToolSpec(
        name="grid_set_note",
        description="Keep a note on a grid's record, as the Grids tab's Note does; an empty note clears it. Apart from the verdict, so judging the grid again keeps it. Saved, and a note (not a clearing) goes on the record as your note on the grid.",
        method="POST",
        path="/app/experiment/grids/{grid_name}/note",
        level="drive",
        router="app",
        part=GRIDS,
        params={
            "grid_name": "the grid name, from grid_list",
            "note": "the note, at most 4000 characters ('' clears it)",
        },
        line=_grid_set_note,
    ),
    ToolSpec(
        name="grid_remove",
        description="Stop tracking a grid, as the Grids tab's Remove does: its files stay on disk, and its lamellae stay with no grid (answered as items_without_grid). Refused while a run is on it or one of its lamellae.",
        method="DELETE",
        path="/app/experiment/grids/{grid_name}",
        level="drive",
        router="app",
        part=GRIDS,
        params={"grid_name": "the grid name, from grid_list"},
        line=_grid_remove,
    ),
    # The grids on the hardware: the sample handling service, which loads them
    # onto the stage and names them where the loader keeps names.
    ToolSpec(
        name="grid_get_inventory",
        description=(
            "The grids on the hardware now: every slot of the autoloader's "
            "magazine (or of the holder, without one), the grid in it by name and "
            "its state: loaded (on the stage, ready to use), occupied (in the "
            "magazine), empty, or unknown (not scanned since the magazine was "
            "docked). The loader's last scan, read at once; grid_run_inventory "
            "scans again. Not the experiment's records (grid_list)."
        ),
        method="POST",
        path="/microscope/services/sample_handling/queries/get_inventory",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="grid_get_current",
        description=(
            "Where the stage is among the holder's slots: the slot it is at and "
            "the grid loaded there, each null when it is at none. Moves nothing."
        ),
        method="POST",
        path="/microscope/services/sample_handling/queries/current",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="grid_get_holder_slots",
        description=(
            "The holder's slots in order: each one's stage position (null until "
            "it is calibrated), its calibration and the grid loaded in it. "
            "grid_move_to_slot goes to one."
        ),
        method="POST",
        path="/microscope/services/sample_handling/queries/holder_slots",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="grid_run_inventory",
        description=(
            "Scan the magazine slot by slot and answer the inventory, as "
            "grid_get_inventory gives it. With an autoloader it takes minutes "
            "and needs the magazine shut; on a fixed holder it is the same "
            "instant read. A long command: it answers with its run at once; "
            "service_wait_for_run sample_handling gives the inventory."
        ),
        method="POST",
        path="/microscope/services/sample_handling/commands/run_inventory",
        level="drive",
        hardware=True,
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="grid_load",
        description=(
            "Bring a grid onto the stage, by its name in grid_get_inventory. With "
            "an autoloader it is an exchange: the grid on the stage goes back to "
            "the magazine first, then this one is loaded, minutes in all. Already "
            "loaded, or on a fixed holder (every grid already sits in a slot), "
            "nothing moves. It does not move the stage: grid_move_to follows it. "
            "A grid not in the inventory is refused (409). A long command: it "
            "answers with its run at once; service_wait_for_run sample_handling "
            "gives the slot it is in."
        ),
        method="POST",
        path="/microscope/services/sample_handling/commands/ensure_loaded",
        level="drive",
        hardware=True,
        params={"grid_name": "the grid's name, from grid_get_inventory"},
        call=lambda a: ({}, kwargs_body(grid_name=str(a["grid_name"]))),
    ),
    ToolSpec(
        name="grid_unload",
        description=(
            "Return the grid on the stage to the magazine. Nothing to do on a "
            "fixed holder. A long command: it answers with its run at once; "
            "service_wait_for_run sample_handling says how it ended."
        ),
        method="POST",
        path="/microscope/services/sample_handling/commands/unload",
        level="drive",
        hardware=True,
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="grid_move_to",
        description=(
            "Move the stage to the holder slot that holds a grid, by name: the "
            "slot's calibrated position, rotation and tilt included. The grid "
            "must be on the stage (grid_load first, with an autoloader); a slot "
            "not calibrated is refused."
        ),
        method="POST",
        path="/microscope/services/sample_handling/commands/move_to_grid",
        level="drive",
        hardware=True,
        params={"grid_name": "the grid's name"},
        call=lambda a: ({}, kwargs_body(grid_name=str(a["grid_name"]))),
    ),
    ToolSpec(
        name="grid_move_to_slot",
        description=(
            "Move the stage to a holder slot, by name from grid_get_holder_slots: "
            "its calibrated position, rotation and tilt included. A slot not "
            "calibrated is refused."
        ),
        method="POST",
        path="/microscope/services/sample_handling/commands/move_to_slot",
        level="drive",
        hardware=True,
        params={"slot_name": "the holder slot's name"},
        call=lambda a: ({}, kwargs_body(slot_name=str(a["slot_name"]))),
    ),
)
