"""The application's tools: the experiment, protocol, workflow, proposals and requests."""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from fibsem.ai.agent.catalog.spec import ToolSpec
from fibsem.server.parts import GRIDS, LAMELLAE, MILLING_STAGES
from fibsem.util import wording

# A fluorescence stack's plane and channels, as the files route reads them.
_STACK_PARAMS = {
    "z": "a fluorescence stack's plane to show, 0 the first (default: the max projection over z)",
    "channels": "the fluorescence channels to show, by name, as its fm block lists them (default: all)",
}


def _options_line(a: Dict[str, Any]) -> str:
    """The workflow options a call sets, in one line."""
    parts = []
    if "show_run_summary" in a and a["show_run_summary"] is not None:
        parts.append(
            "Open the run summary when a run ends"
            if a["show_run_summary"]
            else "Don't open the run summary when a run ends; a toast says it ended"
        )
    if "turn_beams_off" in a and a["turn_beams_off"] is not None:
        parts.append(
            "Turn the beams off when a run ends"
            if a["turn_beams_off"]
            else "Leave the beams on when a run ends"
        )
    return "; ".join(parts) or "No change"


def _open_experiment(get) -> Optional[str]:
    """The name of the experiment open now, or None."""
    status = get("/app/workflow/status")
    current = (status or {}).get("experiment") if isinstance(status, dict) else None
    return (current or {}).get("name") if isinstance(current, dict) else None


def _experiment_name(path: str) -> str:
    """An experiment's name from its path: the folder, not experiment.yaml."""
    path = path.rstrip("/\\")
    name = os.path.basename(path.replace("\\", "/"))
    if os.path.splitext(name)[1].lower() in (".yaml", ".yml"):
        name = os.path.basename(os.path.dirname(path.replace("\\", "/")))
    return name or "an experiment"


def _recent_experiment_about(get, path: str) -> Optional[str]:
    """'5 lamellae, created 15 Oct' for a path on the recents list, else None."""
    recent = get("/app/experiment/recent")
    items = recent.get("items") if isinstance(recent, dict) else None
    wanted = path.replace("\\", "/").rstrip("/")
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        entry_path = str(entry.get("path") or "").replace("\\", "/").rstrip("/")
        if wanted not in (entry_path, os.path.dirname(entry_path)):
            continue
        count = entry.get("num_items")
        parts = []
        if isinstance(count, int):
            parts.append(f"{count} lamell{'a' if count == 1 else 'ae'}")
        try:
            created = datetime.fromisoformat(str(entry.get("created_at")))
            parts.append(f"created {created.day} {created:%b}")
        except ValueError:
            pass
        return ", ".join(parts) or None
    return None


def _proposal_item(proposals: Any, item_id: Any) -> Optional[str]:
    if not isinstance(proposals, dict):
        return None
    for proposal in proposals.get("proposals") or []:
        if isinstance(proposal, dict) and proposal.get("item_id") == item_id:
            return proposal.get("item_name")
    return None


def _request_create(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    actions = [x for x in a.get("actions") or [] if isinstance(x, dict)]
    change = "Ask you to approve " + wording.plural(len(actions), "action")
    if actions:
        first = actions[0]
        from fibsem.ai.agent.catalog.describe import describe_action

        line = describe_action(str(first.get("tool")), first.get("arguments") or {}, "")
        said = str(line.get("change") or "").split("\n", 1)[0]
        change += f": {line.get('what')} · {said}" if line.get("what") else f": {said}"
    return {"what": "Request", "change": change}


def _protocol_update_task_config(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    return {
        "what": f"Protocol · {task}",
        "change": wording.patch_lines(a.get("patch"), ctx.document("config")),
    }


def _protocol_reorder_milling_stages(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    return {
        "what": f"Protocol · {task}",
        "change": f"Reorder {a.get('milling_key')} stages: {wording.names_text(a.get('order'), 'as they are')}",
    }


def _protocol_set_schedule(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    when = a.get("scheduled_at")
    return {
        "what": str(task),
        "change": f"Schedule for {when}" if when else "Clear its schedule",
    }


def _workflow_set_options(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {"what": "Workflow", "change": _options_line(a)}


def _protocol_add_task(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    task = a.get("task_name")
    return {
        "what": "Protocol",
        "change": f"Add a {a.get('task_type')} task named '{task}'",
    }


def _protocol_remove_task(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    return {
        "what": "Protocol",
        "change": f"Remove {task} from the protocol and every lamella",
    }


def _protocol_reorder_tasks(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Protocol",
        "change": f"Put the tasks in this order: {wording.names_text(a.get('order'), 'none')}",
    }


def _protocol_set_task_requirements(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    changes = []
    if a.get("requires") is not None:
        changes.append(
            f"needs {wording.names_text(a.get('requires'), 'no task')} first"
        )
    if a.get("required") is not None:
        changes.append("required" if a.get("required") else "optional")
    return {
        "what": str(task),
        "change": ("; ".join(changes) or "No change").capitalize(),
    }


def _protocol_set_lamella_defaults(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    given = {
        k: a[k]
        for k in ("use_petname", "name_prefix", "poi", "alignment_area")
        if k in a
    }
    return {
        "what": "New lamellae",
        "change": f"Start with {wording.values_text(given)}" if given else "No change",
    }


def _protocol_save(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {
        "what": "Protocol",
        "change": f"Save it to the protocol folder as {a.get('name')}"
        + (", replacing that file" if a.get("overwrite") else ""),
    }


def _protocol_update_grid_task_config(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    return {
        "what": f"Grid protocol · {task}",
        "change": wording.patch_lines(a.get("patch"), ctx.document("config")),
    }


def _protocol_reorder_grid_tasks(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Grid protocol",
        "change": f"Put the grid tasks in this order: {wording.names_text(a.get('order'), 'none')}",
    }


def _protocol_set_grid_task_requirements(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    return {
        "what": f"Grid protocol · {task}",
        "change": f"Needs {wording.names_text(a.get('requires'), 'no task')} first",
    }


def _workflow_start(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {
        "what": "Workflow",
        "change": f"Run {wording.names_text(a.get('task_names'), 'no tasks')} on "
        f"{wording.names_text(a.get('item_names'), 'every lamella')}",
    }


def _workflow_start_grids(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    ends = [
        words
        for key, words in (
            ("unload_at_end", "unload the last grid"),
            ("report_at_end", "write the report"),
        )
        if a.get(key)
    ]
    change = (
        f"Run {wording.names_text(a.get('task_names'), 'no tasks')} on "
        f"{wording.names_text(a.get('grid_names'), 'every grid')}"
        + (f", then {' and '.join(ends)}" if ends else "")
    )
    plan = ctx.get(
        "/app/workflow/grids/plan",
        {
            key: a[key]
            for key in ("task_names", "grid_names", "screen_all")
            if a.get(key) is not None
        },
    )
    estimate = plan.get("estimate") if isinstance(plan, dict) else None
    if isinstance(estimate, dict):
        change += wording.about(estimate.get("seconds"))
        for task in estimate.get("tasks") or []:
            overview = task.get("overview")
            if isinstance(overview, dict):
                change += f"\n{task.get('task_name')}: {_grid_overview_text(overview)}"
    return {"what": "Grids", "change": change}


def _grid_overview_text(overview: Dict[str, Any]) -> str:
    """A grid overview task's tiles and size: "3 × 3 tiles of 500 µm, about
    1.4 × 0.93 mm"."""
    text = f"{overview.get('nrows')} × {overview.get('ncols')} tiles"
    if overview.get("tile_hfw"):
        text += f" of {wording.distance(overview['tile_hfw'])}"
    if overview.get("width") and overview.get("height"):
        text += (
            f", about {wording.distance(overview['width'])} × "
            f"{wording.distance(overview['height'])}"
        )
    return text


def _workflow_queue_grids(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    return {
        "what": "Grids",
        "change": f"Add {wording.names_text(a.get('task_names'), 'no tasks')} for "
        f"{wording.names_text(a.get('grid_names'), 'no grids')} to the end of the run",
    }


def _workflow_stop(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {
        "what": "Workflow",
        "change": "Stop the run now running. A mill in progress stops too",
    }


def _workflow_stop_task(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    task = a.get("task_name")
    return {
        "what": f"{item} · {task}",
        "change": "Stop this task; the run carries on. A mill in progress stops too",
    }


def _protocol_set_supervision(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    task = a.get("task_name")
    return {
        "what": str(task),
        "change": "Supervised: ask at each decision"
        if a.get("supervise")
        else "Automated: run without asking",
    }


def _workflow_requeue(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    item = a.get("item_name")
    task = a.get("task_name")
    return {
        "what": str(item),
        "change": f"Queue {task} to run again" + (", first" if a.get("front") else ""),
    }


def _workflow_edit_queue(
    a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    item = a.get("item_name")
    task = a.get("task_name")
    return {
        "what": str(item),
        "change": {
            "remove": f"Take {task} out of the queue",
            "move_up": f"Move {task} up the queue",
            "move_down": f"Move {task} down the queue",
            "run_next": f"Run {task} next",
        }.get(str(a.get("action")), f"{a.get('action')} {task} in the queue"),
    }


def _proposal_decide(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    task = a.get("task_name")
    outcome = str(a.get("outcome", "")).lower()
    if outcome.startswith("reject"):
        change = f"Reject the proposal: {a.get('reason', '')}".rstrip(": ")
    else:
        change = (
            f"Decide the pass: {a.get('action')}"
            if a.get("action")
            else "Confirm the proposal"
        )
        if a.get("values"):
            change += f" with {wording.values_text(a.get('values'))}"
        if a.get("patch"):
            change += f", setting {wording.values_text(a.get('patch'))}"
    name = _proposal_item(ctx.get("/app/workflow/proposals"), a.get("item_id"))
    return {"what": f"{name or a.get('item_id')} · {task}", "change": change}


def _proposal_suggest(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    task = a.get("task_name")
    name = _proposal_item(ctx.get("/app/workflow/proposals"), a.get("item_id"))
    return {
        "what": f"{name or a.get('item_id')} · {task}",
        "change": f"Suggest a decision: {a.get('reason', '')}".rstrip(": "),
    }


def _experiment_create(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    protocol = a.get("protocol")
    replaces = _open_experiment(ctx.get)
    return {
        "what": "Experiment",
        "change": f"Create and open '{a.get('name')}', with "
        + (f"protocol {protocol}" if protocol else "the default protocol"),
        "details": wording.table(
            ("Setting", "Value"),
            [
                ("Folder", wording.short(a.get("name"))),
                ("Protocol", protocol or "the default"),
                ("Description", a.get("description") or ""),
                ("Replaces", replaces or "none open"),
            ],
        ),
    }


def _experiment_load(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    path = str(a.get("path") or "")
    replaces = _open_experiment(ctx.get)
    about = _recent_experiment_about(ctx.get, path)
    return {
        "what": "Experiment",
        "change": f"Open '{_experiment_name(path)}'"
        + (f" ({about})" if about else "")
        + (f" in place of '{replaces}'" if replaces else ""),
        "details": wording.table(
            ("Setting", "Value"),
            [("Path", path), ("Replaces", replaces or "none open")],
        ),
    }


def _protocol_load(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    return {
        "what": "Protocol",
        "change": f"Load '{a.get('protocol')}' into the open experiment, "
        "replacing its protocol",
    }


TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="app_get_status",
        returns="AppStatus",
        description="What the application is doing: experiment, running workflow, current task and item.",
        method="GET",
        path="/app/workflow/status",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="workflow_get_queue",
        returns="Queue",
        description="The live work queue: id-anchored items with status, plus the mutation version.",
        method="GET",
        path="/app/workflow/queue",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="experiment_get_task_history",
        returns="TaskHistory",
        description="Every task run so far: outcomes, durations, errors. Oldest first, by start time.",
        method="GET",
        path="/app/experiment/task_history",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="workflow_get_summary",
        returns="RunSummary",
        description="The most recent workflow run's outcome table.",
        method="GET",
        path="/app/workflow/summary",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="app_get_log",
        returns="LogRead",
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
        optional=("task_id", "since", "until", "level", "lines"),
    ),
    ToolSpec(
        name="protocol_get",
        returns="ProtocolView",
        description="The workflow definition with live supervision flags and schedules. Each task carries `attendance`: whether a person must be at the microscope while it runs and what waits for their decision afterwards, under its current attention.",
        method="GET",
        path="/app/experiment/protocol",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="experiment_get_image_metadata",
        returns="SavedImageMetadata",
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
        name="experiment_render_markers",
        returns="MarkersDrawn",
        description="Any saved SEM or FIB image in the experiment (an overview, yours or a grid task's, or a single image) with the experiment's items drawn on it, as the overview plot dialog draws them: each item's number, green, red for a failed one, and a scale bar. grid_name draws only that grid's items; without it, every item with a pose. Saved beside the image as markers/<image>.png (replaced when drawn again): cite that path as evidence ({image: path}) when you report on the items, so the operator sees it in your note. Answers the path, the items drawn and a preview. Fluorescence images are refused.",
        method="POST",
        path="/app/experiment/images/markers",
        level="watch",
        router="app",
        part=LAMELLAE,
        params={
            "path": "a beam image file path inside the experiment folder",
            "grid_name": "draw only this grid's items",
        },
        optional=("grid_name",),
    ),
    ToolSpec(
        name="experiment_get_image_stage_position",
        returns="SavedImagePosition",
        description="The stage position under a point of any saved image in the experiment, beam or FM, from the image's own metadata, so it holds wherever the stage is now. Give x, y in source pixels, or fx, fy from 0 to 1 for a point read off a preview. Nothing moves.",
        method="GET",
        path="/app/experiment/images/stage_position",
        level="watch",
        router="app",
        params={
            "path": "an image file path inside the experiment folder",
            "x": "source-pixel x (with y)",
            "y": "source-pixel y, +y down (with x)",
            "fx": "fraction across the image, 0-1 (with fy)",
            "fy": "fraction down the image, 0-1 (with fx)",
        },
        optional=("x", "y", "fx", "fy"),
    ),
    ToolSpec(
        name="experiment_get_image_pixels",
        returns="SavedImagePixels",
        description="Where stage positions fall on any saved image in the experiment, beam or FM: source pixels for each (and whether inside the image), with the image's scale block. A position the image can't place answers why. Computed from the image's own metadata; never re-derive stage geometry. Nothing moves.",
        method="POST",
        path="/app/experiment/images/pixels",
        level="watch",
        router="app",
        params={
            "path": "an image file path inside the experiment folder",
            "stage_positions": "stage position dicts (x, y, z, r, t, coordinate_system)",
        },
    ),
    ToolSpec(
        name="experiment_list_files",
        returns="FolderListing",
        description="One folder of the open experiment: its folders, then its files, each with name, path (relative to the experiment), kind (folder, text, image, other), size and modified time; a folder also says how many images it holds at any depth. Start at the top (no dir) and go down one folder at a time.",
        method="GET",
        path="/app/experiment/files",
        level="watch",
        router="app",
        params={
            "dir": "a folder path from an earlier listing, relative to the experiment (default: the top)",
            "experiment": "optional: a past experiment to read instead of the open one, by its relative_path from experiment_list_all (or its path from experiment_list_recent); only one the operator named. Every read of it goes on the record",
        },
        optional=("dir", "experiment"),
    ),
    ToolSpec(
        name="experiment_read_file",
        returns="FileRead",
        description="One file of the open experiment: text (yaml, json, log, md, csv) as text, up to 512 KB; an image (tif, png, jpg) as a 768 px preview you can look at, with its metadata block; a folder as its listing. A fluorescence stack (.ome.tiff) shows the max projection of every channel; give z for one plane and channels for some of them. Its metadata's fm block lists the channels, z_positions, and each channel's sharpness per plane (1 the sharpest) and sharpest_plane. Previews and metadata: for detail in part of an image, experiment_crop_image; for the file itself, to work on with your own code, experiment_fetch_file. Paths outside the experiment are refused.",
        method="GET",
        path="/app/experiment/files/{path}",
        level="watch",
        router="app",
        params={
            "path": "a path from experiment_list_files, or an image path an image answer gave",
            **_STACK_PARAMS,
            "experiment": "optional: a past experiment to read instead of the open one, by its relative_path from experiment_list_all (or its path from experiment_list_recent); only one the operator named. Every read of it goes on the record",
        },
        optional=("z", "channels", "experiment"),
    ),
    ToolSpec(
        name="experiment_fetch_file",
        returns="FileRead",
        description="Fetch one file of the open experiment as it is (a tif with its full data and metadata, a log, a yaml) and save it on the machine you run on, for your own code to read. Answers where it saved it: a folder named for the experiment, under fetched/ in your working folder, the file at its path in the experiment. The fetch goes on the record. A folder is refused: fetch its files one at a time.",
        method="GET",
        path="/app/experiment/files/{path}",
        level="watch",
        router="app",
        params={
            "path": "a path from experiment_list_files, or an image path an image answer gave",
            "experiment": "optional: a past experiment to read instead of the open one, by its relative_path from experiment_list_all (or its path from experiment_list_recent); only one the operator named. Every read of it goes on the record",
        },
        optional=("experiment",),
    ),
    ToolSpec(
        name="experiment_crop_image",
        returns="FileRead",
        description="One region of an image in the open experiment, at native resolution (downscaled only when wider than 768 px): full detail where it matters, without the whole file. Give box in source pixels or fraction (0 to 1, as read off a preview), each [x, y, width, height]. The answer's preview carries the region and its own scale.",
        method="GET",
        path="/app/experiment/files/{path}",
        level="watch",
        router="app",
        params={
            "path": "an image path from experiment_list_files, or one an image answer gave",
            "box": "[x, y, width, height] in source pixels (instead of fraction)",
            "fraction": "[left, top, width, height] as fractions of the image, 0 to 1 (instead of box)",
            **_STACK_PARAMS,
            "experiment": "optional: a past experiment to read instead of the open one, by its relative_path from experiment_list_all (or its path from experiment_list_recent); only one the operator named. Every read of it goes on the record",
        },
        optional=("box", "fraction", "z", "channels", "experiment"),
    ),
    ToolSpec(
        name="protocol_get_task_config",
        returns="TaskConfigRead",
        description="One task's protocol-level configuration document (the defaults new items copy), with a version token naming exactly this state of it. Full document, not a summary.",
        method="GET",
        path="/app/experiment/protocol/task_config/{task_name}",
        level="watch",
        router="app",
        params={"task_name": "the task name, as listed by protocol_get"},
    ),
    ToolSpec(
        name="experiment_add_note",
        returns="Recorded",
        description="Put an observation on the record — the event stream and the experiment log. Use it for judgments worth keeping ('curtaining on 02's face, accepted anyway'); notes change nothing. Optionally name the item it concerns. To answer the operator's message (a MESSAGE line from fibsem-mcp watch), pass its id as reply_to: the card shows the note as your reply. A message is never an approval: your level still decides what you may do.",
        method="POST",
        path="/app/experiment/notes",
        level="watch",  # Watch adds notes: they change nothing
        router="app",
        params={
            "text": "the note, at most 4000 characters",
            "item_name": "optional: the item this note concerns",
            "reply_to": "optional: the id of the operator's message this answers",
        },
        optional=("item_name", "reply_to"),
    ),
    ToolSpec(
        name="request_create",
        returns="AgentRequestView",
        description="At Assist, ask the operator to approve one or more actions: each is {tool, arguments}, the tool you would call and the arguments you would give it, run in order once approved. Give the reason the operator reads before deciding. One request waits at a time (409 request_pending names it): to change it, withdraw it (request_withdraw) and file again. Answers the request, status pending; follow it with request_get or the events (proposal_decided of kind actions, an action record per action, request_finished). Each request is saved on the experiment's record as an actions proposal; with nothing to record it on it is refused (409 no_experiment). Approved actions run through their own routes, so each is checked and can be refused as a direct call would be; the first refusal stops the batch. A rejection carries the operator's reason. An action the agent may never use (agent: false in device_list) is refused at filing (403 not_allowed). Changing the level or Stop agent cancels a waiting request. Reads need no request, nor does stopping your own mill (milling_stop): call those directly. Stopping anything else (workflow_stop, someone else's mill) goes in a request.",
        method="POST",
        path="/app/agent/requests",
        level="watch",
        router="app",
        params={
            "actions": "list of {tool, arguments}, at most 20",
            "reason": "why, for the operator, in two or three short sentences: what you saw, then why these actions follow. The card already says what each action does: don't restate it. Attach the image you judged from as evidence: it is the card's picture, with the actions' points, areas and patterns drawn on it when it is a beam image; without it, the card draws them on the image the values were measured on",
        },
        note="Works at Assist only: at Drive make the calls yourself; refused (403 level) at Drive and Watch.",
        line=_request_create,
    ),
    ToolSpec(
        name="request_withdraw",
        returns="AgentRequestView",
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
        returns="AgentRequestView",
        description="One request you filed with request_create: its status (pending, applying, applied, failed, rejected, cancelled), who decided it, the operator's reason on a reject, why it was cancelled, and each action's result or the route's error.",
        method="GET",
        path="/app/agent/requests/{request_id}",
        level="watch",
        router="app",
        params={"request_id": "the request_id request_create answered"},
    ),
    ToolSpec(
        name="request_list",
        returns="RequestList",
        description="Every request filed since the app started, newest first, each as request_get gives it: the way back to a request that is still waiting after you reconnect, and to the ids request_get and request_withdraw take.",
        method="GET",
        path="/app/agent/requests",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="protocol_reorder_milling_stages",
        returns="ConfigEdit",
        description="Reorder the milling stages inside a protocol-level task config. Same rules as the item form; reaches items created (or re-applied) after.",
        method="POST",
        path="/app/experiment/protocol/task_config/{task_name}/stages/reorder",
        level="drive",
        router="app",
        part=MILLING_STAGES,
        params={
            "task_name": "the task name",
            "milling_key": "which milling config within the task",
            "order": "the current stage names in their new sequence",
            "version": "the version token from protocol_get_task_config",
        },
        line=_protocol_reorder_milling_stages,
    ),
    ToolSpec(
        name="protocol_set_schedule",
        returns="TaskEdit",
        description="Set (or clear) when a task may start: ISO-8601 timestamp (naive = instrument-local time), or null to clear. The workflow reads schedules at each task start, so a change takes effect at the next start; the schedule persists with the protocol.",
        method="POST",
        path="/app/workflow/schedule",
        level="drive",
        router="app",
        params={
            "task_name": "the task name, as listed by protocol_get",
            "scheduled_at": "ISO-8601 timestamp, or null to clear",
        },
        optional=("scheduled_at",),
        line=_protocol_set_schedule,
    ),
    ToolSpec(
        name="workflow_set_options",
        returns="OptionsSet",
        description="Set the workflow's run options, any of: whether the run summary dialog opens when a run ends (off still toasts each run's end with a Show summary link; turn it off before starting runs yourself, so no dialog stands over the operator's window, and tell the operator); whether the beams are turned off when a run ends. Persists with the protocol; protocol_get's options show the current values.",
        method="POST",
        path="/app/workflow/options",
        level="drive",
        router="app",
        params={
            "show_run_summary": "true to open the summary dialog at a run's end, false for the toast only",
            "turn_beams_off": "true to turn the beams off when a run ends",
        },
        optional=("show_run_summary", "turn_beams_off"),
        line=_workflow_set_options,
    ),
    ToolSpec(
        name="protocol_update_task_config",
        returns="ConfigEdit",
        description="Patch a task's protocol-level defaults (what new items copy) with dotted-path edits. Echo the version from protocol_get_task_config — a mismatch is refused (409 stale_config). Running tasks are never affected: they hold their item's copy; the edit reaches items created after.",
        method="POST",
        path="/app/experiment/protocol/task_config/{task_name}",
        level="drive",
        router="app",
        params={
            "task_name": "the task name, as listed by protocol_get",
            "patch": (
                "object of dotted-path: value entries to set; a list of settings "
                "(channel_settings) is set whole, as a list of their dicts"
            ),
            "version": "the version token from protocol_get_task_config",
        },
        line=_protocol_update_task_config,
    ),
    ToolSpec(
        name="protocol_add_task",
        returns="TaskEdit",
        description="Add a task of a registered type to the protocol, as the protocol editor's Add Task does: its default settings, at the end of the workflow, and a copy on every item. An unknown type is refused with the type names. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/protocol/tasks",
        level="drive",
        router="app",
        params={
            "task_type": "a registered task type",
            "task_name": "the new task's name, unique in the protocol",
        },
        line=_protocol_add_task,
    ),
    ToolSpec(
        name="protocol_remove_task",
        returns="TaskEdit",
        description="Remove a task from the protocol, the workflow and every item, as the protocol editor's Remove does. What already ran stays in the task history. Refused while a workflow runs.",
        method="DELETE",
        path="/app/experiment/protocol/tasks/{task_name}",
        level="drive",
        router="app",
        params={"task_name": "the task name, as listed by protocol_get"},
        line=_protocol_remove_task,
    ),
    ToolSpec(
        name="protocol_reorder_tasks",
        returns="TaskEdit",
        description="Put the workflow's tasks in a new order, as dragging them on the Workflow tab does. order is every task, each once; a task before one it requires is refused. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/protocol/tasks/reorder",
        level="drive",
        router="app",
        params={"order": "every workflow task name, in the new order"},
        line=_protocol_reorder_tasks,
    ),
    ToolSpec(
        name="protocol_set_task_requirements",
        returns="TaskEdit",
        description="Set what a workflow task requires (the earlier tasks an item must have completed first) and whether it is required or optional, as the Workflow tab's Edit Task does. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/protocol/tasks/{task_name}/requirements",
        level="drive",
        router="app",
        params={
            "task_name": "the task name, as listed by protocol_get",
            "requires": "the earlier tasks it needs ([] for none)",
            "required": "true for required, false for optional",
        },
        optional=("requires", "required"),
        line=_protocol_set_task_requirements,
    ),
    ToolSpec(
        name="protocol_set_lamella_defaults",
        returns="LamellaDefaultsSet",
        description="Set what every new item starts with, as the protocol editor's Lamella Template does: use_petname, name_prefix, poi ({x, y}) and alignment_area ({left, top, width, height}, frame fractions). Only what you pass changes; {} clears poi or alignment_area. Existing items keep theirs. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/protocol/lamella_defaults",
        level="drive",
        router="app",
        part=LAMELLAE,
        params={
            "use_petname": "true to name new items with petnames",
            "name_prefix": "the prefix for new item names",
            "poi": "the default point of interest {x, y}, or {} to clear it",
            "alignment_area": "the default alignment area {left, top, width, height}, or {} to clear it",
        },
        optional=("use_petname", "name_prefix", "poi", "alignment_area"),
        line=_protocol_set_lamella_defaults,
    ),
    ToolSpec(
        name="protocol_save",
        returns="ProtocolSaved",
        description="Save the experiment's protocol to the protocol folder under a file name, as Export Protocol does, so a later experiment can be created with it (experiment_create) or load it (protocol_load). An existing file is refused unless overwrite.",
        method="POST",
        path="/app/experiment/protocol/save",
        level="drive",
        router="app",
        params={
            "name": "a file name (.yaml is added if missing), no folders",
            "overwrite": "true to replace a file of that name",
        },
        optional=("overwrite",),
        line=_protocol_save,
    ),
    ToolSpec(
        name="protocol_get_grid_task_config",
        returns="TaskConfigRead",
        description="A grid task's settings (shared by every grid), with the version a patch echoes. The task names are protocol_get's grid_tasks.",
        method="GET",
        path="/app/experiment/grid_protocol/task_config/{task_name}",
        level="watch",
        router="app",
        part=GRIDS,
        params={"task_name": "a grid task name, from protocol_get's grid_tasks"},
    ),
    ToolSpec(
        name="protocol_update_grid_task_config",
        returns="ConfigEdit",
        description="Patch a grid task's settings with dotted-path edits, as the Protocol tab's grid task form does; attention (automated, supervised) is the Grids view's Automated/Supervised chip. Echo the version from protocol_get_grid_task_config — a mismatch is refused (409 stale_config). Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/grid_protocol/task_config/{task_name}",
        level="drive",
        router="app",
        part=GRIDS,
        params={
            "task_name": "a grid task name, from protocol_get's grid_tasks",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from protocol_get_grid_task_config",
        },
        line=_protocol_update_grid_task_config,
    ),
    ToolSpec(
        name="protocol_reorder_grid_tasks",
        returns="TaskEdit",
        description="Put the grid tasks in a new order, as dragging them on the Grids view does: every grid task, each once. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/grid_protocol/tasks/reorder",
        level="drive",
        router="app",
        part=GRIDS,
        params={"order": "every grid task name, in the new order"},
        line=_protocol_reorder_grid_tasks,
    ),
    ToolSpec(
        name="protocol_set_grid_task_requirements",
        returns="TaskEdit",
        description="Set what a grid task requires on the same grid first, as the Grids view's Edit Task does: any other grid tasks. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/grid_protocol/tasks/{task_name}/requirements",
        level="drive",
        router="app",
        part=GRIDS,
        params={
            "task_name": "a grid task name, from protocol_get's grid_tasks",
            "requires": "the grid tasks it needs first ([] for none)",
        },
        line=_protocol_set_grid_task_requirements,
    ),
    ToolSpec(
        name="app_get_events",
        returns="Events",
        description="Events after a sequence number: milling/acquisition progress, stage moves, task lifecycle. Long-polls up to timeout seconds.",
        method="GET",
        path="/app/events",
        level="watch",
        router="app",
        params={
            "since": "return events with seq greater than this (0 for all held)",
            "timeout": "seconds to wait for new events before returning (max 30)",
        },
        optional=("since", "timeout"),
    ),
    ToolSpec(
        name="workflow_stop",
        returns="Stopped",
        description="Stop the running workflow — the same path as the GUI's Stop button; the mill halts too. Give the reason the operator reads.",
        method="POST",
        path="/app/workflow/stop",
        level="drive",
        router="app",
        params={"reason": "why you are stopping it, for the operator (required)"},
        note="Needs the agent at Drive: the workflow is never the agent's own. At Assist, put it in request_create and the operator approves it (a direct call is refused, 403 ask_first); the operator can also press Stop. Refused (403) at Watch. Never refused for someone else being in control of the microscope.",
        line=_workflow_stop,
    ),
    ToolSpec(
        name="workflow_stop_task",
        returns="Stopped",
        description="Stop the task now running and carry on with the rest of the queue, as the timeline's Stop Task does; the mill halts too, and the task ends Cancelled. Name the task you mean (workflow_get_queue): if it is no longer the one running, nothing is stopped. Give the reason the operator reads.",
        method="POST",
        path="/app/workflow/stop_task",
        level="drive",
        router="app",
        params={
            "item_name": "the item (or grid) of the task running",
            "task_name": "the task running",
            "reason": "why you are stopping it, for the operator (required)",
        },
        note="Needs the agent at Drive: the workflow is never the agent's own. At Assist, put it in request_create and the operator approves it (a direct call is refused, 403 ask_first). Refused (403) at Watch. Never refused for someone else being in control of the microscope.",
        line=_workflow_stop_task,
    ),
    ToolSpec(
        name="workflow_start",
        returns="RunStarted",
        description="Start a workflow — the Run button, remotely. Names the tasks to run (from the protocol) and optionally the items; omitted items means all of them. Refused while a workflow is already running.",
        method="POST",
        path="/app/workflow/start",
        level="drive",
        router="app",
        params={
            "task_names": "the tasks to run, from the protocol's task names",
            "item_names": "the items (lamellae) to run them for; omit for all",
        },
        optional=("item_names",),
        line=_workflow_start,
    ),
    ToolSpec(
        name="workflow_plan_grids",
        returns="GridPlan",
        description="What a grid run would execute — the preflight dialog's content: the (grid, step) sequence with each grid's load first, validated against the grid protocol and the recorded grids. screen_all plans over the known grids and says the run will use whatever the inventory finds. Computes only; reads no hardware.",
        method="POST",
        path="/app/workflow/grids/plan",
        level="watch",
        router="app",
        part=GRIDS,
        params={
            "task_names": "grid tasks to run, from the protocol's grid_tasks names",
            "grid_names": "grids to run them on; omit for every recorded grid",
            "screen_all": "true = inventory first, then every present grid (no grid_names)",
        },
        optional=("grid_names", "screen_all"),
    ),
    ToolSpec(
        name="workflow_start_grids",
        returns="RunStarted",
        description="Start a grid run — the Grids view's Run button, remotely; with screen_all, its Screen all grids button (inventory, then every present grid). Refused while any workflow, lamella or grid, is running. Call workflow_plan_grids first and show the plan.",
        method="POST",
        path="/app/workflow/grids/start",
        level="drive",
        router="app",
        part=GRIDS,
        params={
            "task_names": "grid tasks to run, from the protocol's grid_tasks names",
            "grid_names": "grids to run them on; omit for every recorded grid",
            "screen_all": "true = inventory first, then every present grid (no grid_names)",
            "unload_at_end": "true to unload the last grid when the run ends",
            "report_at_end": "true to write the grid report when the run ends",
        },
        optional=("grid_names", "screen_all", "unload_at_end", "report_at_end"),
        line=_workflow_start_grids,
    ),
    ToolSpec(
        name="protocol_set_supervision",
        returns="TaskEdit",
        description="Set whether a task asks for supervision, in the live protocol. Takes effect at the workflow's next decision point, mid-run — the same behaviour as the GUI's supervised/automated toggle.",
        method="POST",
        path="/app/experiment/protocol/supervision",
        level="drive",
        router="app",
        params={
            "task_name": "the task to change, e.g. 'Rough Milling'",
            "supervise": "true to ask before acting, false to run automatically",
        },
        line=_protocol_set_supervision,
    ),
    ToolSpec(
        name="workflow_requeue",
        returns="Requeued",
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
        optional=("front",),
        line=_workflow_requeue,
    ),
    ToolSpec(
        name="workflow_edit_queue",
        returns="QueueEdited",
        description="Remove or move a task still waiting in the RUNNING workflow's queue, as the queue's own menu does — 'just do it for 03' without stopping the run. action: remove (it will not run; kept in the run record as Removed), move_up, move_down, or run_next. Only a waiting pair can be edited: the running one is refused (409 not_pending; stop it with workflow_stop), as is one that already ran. Read the queue (workflow_get_queue) first. changed=false means it was already first or last.",
        method="POST",
        path="/app/workflow/queue/edit",
        level="drive",
        router="app",
        params={
            "item_name": "the item (lamella) the queued task is for",
            "task_name": "the queued task",
            "action": "remove, move_up, move_down or run_next",
        },
        line=_workflow_edit_queue,
    ),
    ToolSpec(
        name="workflow_queue_grids",
        returns="GridTasksQueued",
        description="Add grid tasks for grids to the RUNNING grid run, at the end of the queue, as the Grids view's Add to queue does: each grid's load, then its tasks, less what is already waiting. Refused when no grid run is going.",
        method="POST",
        path="/app/workflow/queue/grids",
        level="drive",
        router="app",
        part=GRIDS,
        params={
            "task_names": "grid tasks to add, from the protocol's grid_tasks names",
            "grid_names": "the grids to add them for",
        },
        line=_workflow_queue_grids,
    ),
    ToolSpec(
        name="app_get_display_images",
        returns="DisplayImages",
        description="The SEM and FIB images the app's GUI is displaying right now, as previews — the display cache, not a new acquisition. Post-mill images appear here as soon as the GUI shows them, before they reach disk.",
        method="GET",
        path="/app/experiment/images",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="proposal_list",
        returns="Proposals",
        description="Every pending proposal: what a task is stopped on, waiting for a decision. Which item and task, its own id (proposal_id, which proposal_decide passes back), the run it is from (task_id), its kind, the proposed values (a point of interest in metres, milling frame), provenance, and the reference image the values sit on as a JPEG preview. An empty list when nothing is pending.",
        method="GET",
        path="/app/workflow/proposals",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="proposal_decide",
        returns="Decided",
        description="Decide a proposal a task is stopped on (confirm or reject it), as the operator's Continue would. Pass the proposal_id proposal_list gave for it: the decision is on that proposal, and is refused (409 stale_proposal) if it is no longer the one on the record -- the task re-ran, or asked again in the same run -- so re-read and decide the current one. (task_id, the run, is accepted in its place but cannot tell two proposals from one run apart.) Confirming needs all its values (e.g. {'poi': {'x','y'}} metres, milling frame), as proposed or adjusted; the task carries on with them, so pass the values you actually judge right. A proposal already decided (by anyone, or recorded Unreviewed when nobody was asked) is refused (422 invalid_value); to change a value, re-run the task. A wrong, missing or foreign value is refused (422 invalid_value) with nothing written. An alignment_area confirmed at Setup also moves the fiducial to the area's centre when Setup's sync_fiducial_position is on. Rejected needs a reason, and what it does depends on the kind: detection, alignment_area, and a milling or spot_burn pass fail the task, so nothing that requires it runs (the lamella itself is not marked defective); state at a tilt means don't tilt, and the task carries on; any other state and point_of_interest are recorded and the task carries on as if confirmed. Refused (409) when the item has a task running -- stop it instead -- unless that task is the one stopped on this very proposal, which a decision releases. Also refused when there is no proposal for that item and task. A milling or spot_burn proposal is one pass of a loop the task runs: Confirm it with action, one of the pass's provenance.actions -- run (mill or burn the value as decided, then the task asks again), edit (take the value as decided and ask again; nothing runs), skip (go on without running it; before any pass), accept (go on with what was run; after a pass) -- and with no values to run it as proposed. values.milling_config is a whole config document (for adding, removing or reordering stages); patch then sets fields by dotted path, SI units, a stage named by its name or index (e.g. {'stages.Polish 1.pattern.depth': 1e-6}); values.spot_burn is {coordinates: [{x, y}] fractions of the frame, origin top-left, milling_current A, exposure_time s}. The decision is recorded with the agent as its actor and the experiment saved; save_error is set when it landed but could not be saved (it is in the app, not yet on disk).",
        method="POST",
        path="/app/workflow/decide",
        level="drive",
        router="app",
        params={
            "item_id": "the item id from proposal_list (by id, never name)",
            "task_name": "the task name from proposal_list",
            "proposal_id": "the proposal_id from proposal_list: the proposal you looked at",
            "task_id": "optional, and only in place of proposal_id: the run the proposal is from",
            "outcome": "Confirmed or Rejected",
            "values": "on Confirmed of a pending proposal: every value its kind carries, as proposed or adjusted",
            "reason": "required on Rejected",
            "actor": "optional agent/model name, recorded as agent:<name>",
            "action": "on Confirmed of a milling or spot_burn pass: one of its provenance.actions",
            "patch": "optional, on a milling pass: {dotted path: value} set on the config (given in values, else as proposed)",
        },
        optional=(
            "proposal_id",
            "values",
            "reason",
            "actor",
            "task_id",
            "action",
            "patch",
        ),
        line=_proposal_decide,
    ),
    ToolSpec(
        name="proposal_suggest",
        returns="Suggested",
        description="Suggest a decision on a pending proposal, for the operator to approve or decline in the window. Nothing is decided: the operator sees 'Agent suggests' with your reason and the suggested value drawn on the proposal's image, and Approve decides it as you suggested, as the operator's decision, recorded with your suggestion_id (proposal_suggested, then proposal_decided with suggested_by). Decline drops it, with the operator's optional reason (suggestion_declined); deciding any other way leaves it unapproved. One suggestion per proposal: a new one replaces yours. Checked as proposal_decide checks the decision it would make (outcome, values or patch, action, all as there), so one that could not be approved is refused now (422 invalid_value); a proposal no longer pending is 409 stale_proposal. proposal_list shows the suggestion on each proposal, declined and why when it was.",
        method="POST",
        path="/app/workflow/proposals/suggest",
        level="assist",
        router="app",
        params={
            "item_id": "the item id from proposal_list (by id, never name)",
            "task_name": "the task name from proposal_list",
            "proposal_id": "the proposal_id from proposal_list",
            "reason": "why: the operator reads it when deciding",
            "outcome": "Confirmed (default) or Rejected, as for proposal_decide",
            "values": "optional: the values you suggest, as for proposal_decide",
            "patch": "optional, on a milling pass: {dotted path: value} set on the proposed config",
            "action": "on a milling or spot_burn pass: one of its provenance.actions",
            "actor": "optional agent/model name, recorded as agent:<name>",
        },
        optional=("outcome", "values", "patch", "action", "actor"),
        note="Works at Assist and Drive; refused (403) at Watch. At Drive you may proposal_decide directly instead.",
        line=_proposal_suggest,
    ),
    ToolSpec(
        name="experiment_list_recent",
        returns="RecentExperiments",
        description="Recently opened experiments on this machine, without loading them. experiment_load takes a path from here.",
        method="GET",
        path="/app/experiment/recent",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="experiment_list_all",
        returns="SavedExperiments",
        description="Every experiment under the experiments folder (up to 3 folders deep), newest first, without loading them: where experiment_create puts a new one, and what experiment_load can open (its relative_path). Narrow it by instrument, operator or date.",
        method="GET",
        path="/app/experiment/all",
        level="watch",
        router="app",
        params={
            "instrument": "only experiments run on this instrument: its serial or model",
            "operator": "only experiments run by this operator",
            "since": "ISO date or time: created at or after it",
            "until": "ISO date or time: created at or before it",
        },
        optional=("instrument", "operator", "since", "until"),
    ),
    ToolSpec(
        name="protocol_list",
        returns="ProtocolFiles",
        description="The protocol files in the protocol folder, by name, with the default marked: what experiment_create and protocol_load take.",
        method="GET",
        path="/app/experiment/protocols",
        level="watch",
        router="app",
    ),
    ToolSpec(
        name="experiment_create",
        returns="ExperimentOpened",
        description="Create an experiment under the experiments folder and open it in place of the current one, with a protocol from protocol_list (the default when omitted). name is a folder, or a path of up to 3 folders (e.g. 2026-10-20/grid-a), made as needed; an existing folder is refused, never overwritten. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/create",
        level="drive",
        router="app",
        params={
            "name": "the new experiment's folder under the experiments folder: a name, or a relative path of folders",
            "protocol": "optional protocol name from protocol_list",
            "description": "optional one-line description",
        },
        optional=("protocol", "description"),
        line=_experiment_create,
    ),
    ToolSpec(
        name="experiment_load",
        returns="ExperimentOpened",
        description="Open an experiment in place of the current one: a path from experiment_list_recent or experiment_list_all. Any other experiment is the operator's to open. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/load",
        level="drive",
        router="app",
        params={"path": "a path from experiment_list_recent or experiment_list_all"},
        line=_experiment_load,
    ),
    ToolSpec(
        name="protocol_load",
        returns="ProtocolLoaded",
        description="Load a protocol from protocol_list into the open experiment, replacing its protocol, and save it beside the experiment. Existing items keep their task configs; item_apply_protocol updates one. Refused while a workflow runs.",
        method="POST",
        path="/app/experiment/protocol/load",
        level="drive",
        router="app",
        params={"protocol": "a protocol name from protocol_list"},
        line=_protocol_load,
    ),
)
