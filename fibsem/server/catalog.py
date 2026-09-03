"""The tool catalog: the single source of truth for what agents can do.

The sidecar (``fibsem-mcp``) builds its MCP tool list from this module, and a
contract test asserts every entry maps onto a real route on the server app —
so the two surfaces cannot drift apart. Pure data, no dependencies beyond the
standard library: importable everywhere, including the py3.8 floor.

Scopes mirror the server's enforcement (fibsem/server/auth.py): the sidecar
registers ``read`` tools for any valid token and ``hardware`` tools only when
``/capabilities`` reports that scope armed — but the server enforces either
way; the catalog filter is a courtesy, not the security boundary.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    method: str  # "GET" | "POST"
    path: str
    scope: str  # "read" | "hardware"
    params: Dict[str, str] = field(default_factory=dict)  # name -> description
    # Which router serves it: "microscope" is always mounted; "app" only when
    # the server was built with an app_context (/capabilities reports which).
    router: str = "microscope"


CATALOG: Tuple[ToolSpec, ...] = (
    # --- read ---
    ToolSpec(
        name="get_capabilities",
        description="Server capabilities: API version, manufacturer, mounted routers, armed scopes.",
        method="GET",
        path="/capabilities",
        scope="read",
    ),
    ToolSpec(
        name="get_system_info",
        description="Microscope system settings and stage geometry flags.",
        method="GET",
        path="/system",
        scope="read",
    ),
    ToolSpec(
        name="get_stage_position",
        description="Current stage position (x/y/z in metres, r/t in radians).",
        method="GET",
        path="/stage_position",
        scope="read",
    ),
    ToolSpec(
        name="get_stage_orientation",
        description="Named stage orientation (e.g. SEM, FIB, MILLING, UNKNOWN).",
        method="GET",
        path="/stage_orientation",
        scope="read",
    ),
    ToolSpec(
        name="get_microscope_state",
        description="Full microscope state snapshot (beams, detectors, stage).",
        method="GET",
        path="/microscope_state",
        scope="read",
    ),
    ToolSpec(
        name="get_milling_angle",
        description="Current milling angle in degrees.",
        method="GET",
        path="/milling_angle",
        scope="read",
    ),
    ToolSpec(
        name="get_milling_state",
        description="Milling state (IDLE, RUNNING, PAUSED, ...).",
        method="GET",
        path="/milling_state",
        scope="read",
    ),
    ToolSpec(
        name="estimate_milling_time",
        description="Estimated time in seconds for the currently drawn patterns.",
        method="GET",
        path="/estimate_milling_time",
        scope="read",
    ),
    ToolSpec(
        name="stop_milling",
        description="Emergency stop for an in-progress mill. Always allowed with a valid token.",
        method="POST",
        path="/stop_milling",
        scope="read",  # deliberate: stopping is never gated behind arming
    ),
    # --- hardware ---
    ToolSpec(
        name="acquire_image_preview",
        description="Acquire a fresh image and return an agent-sized JPEG preview with metadata.",
        method="POST",
        path="/acquire_image_preview",
        scope="hardware",
        params={"beam_type": "ELECTRON or ION"},
    ),
    ToolSpec(
        name="last_image_preview",
        description="JPEG preview of the last acquired image for a beam (switches the active channel).",
        method="POST",
        path="/last_image_preview",
        scope="hardware",
        params={"beam_type": "ELECTRON or ION"},
    ),
    ToolSpec(
        name="move_stage_relative",
        description="Move the stage by a relative offset (metres and radians).",
        method="POST",
        path="/move_stage_relative",
        scope="hardware",
        params={
            "dx": "relative x in metres",
            "dy": "relative y in metres",
            "dz": "relative z in metres",
        },
    ),
    ToolSpec(
        name="move_stage_absolute",
        description="Move the stage to an absolute position (metres and radians).",
        method="POST",
        path="/move_stage_absolute",
        scope="hardware",
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
        path="/milling_angle/move",
        scope="hardware",
        params={"milling_angle_deg": "target milling angle in degrees"},
    ),
    ToolSpec(
        name="autocontrast",
        description="Run autocontrast for a beam (changes detector settings).",
        method="POST",
        path="/autocontrast",
        scope="hardware",
        params={"beam_type": "ELECTRON or ION"},
    ),
)


APP_TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_app_status",
        description="What the application is doing: experiment, running workflow, current task and item.",
        method="GET",
        path="/app/status",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="get_app_queue",
        description="The live work queue: id-anchored items with status, plus the mutation version.",
        method="GET",
        path="/app/queue",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="get_experiment_summary",
        description="Per-item summary of the open experiment.",
        method="GET",
        path="/app/experiment_summary",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="get_task_history",
        description="Every task run so far: outcomes, durations, errors.",
        method="GET",
        path="/app/task_history",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="get_run_summary",
        description="The most recent workflow run's outcome table.",
        method="GET",
        path="/app/run_summary",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="get_protocol",
        description="The workflow definition with live supervision flags and schedules.",
        method="GET",
        path="/app/protocol",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="get_item_detail",
        description="Everything durable about one item: status, failure flag, POI, alignment area, milling angle, and where its poses put the stage.",
        method="GET",
        path="/app/items/{item_name}",
        scope="read",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="get_task_outputs",
        description="The files an item's completed tasks produced (reference images by role).",
        method="GET",
        path="/app/task_outputs/{item_name}",
        scope="read",
        router="app",
        params={"item_name": "the item (lamella) name"},
    ),
    ToolSpec(
        name="get_protocol_task_config",
        description="One task's protocol-level configuration document (the defaults new items copy), with a version token naming exactly this state of it. Full document, not a summary.",
        method="GET",
        path="/app/protocol/task_config/{task_name}",
        scope="read",
        router="app",
        params={"task_name": "the task name, as listed by get_protocol"},
    ),
    ToolSpec(
        name="get_item_task_config",
        description="One item's own copy of a task configuration (what its run actually executes — tasks may have rewritten it mid-run), with a version token naming exactly this state of it.",
        method="GET",
        path="/app/items/{item_name}/task_config/{task_name}",
        scope="read",
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
        scope="control",
        router="app",
        params={
            "text": "the note, at most 4000 characters",
            "item_name": "optional: the item this note concerns",
        },
    ),
    ToolSpec(
        name="apply_protocol_to_item",
        description="Re-copy protocol task configs onto an existing item (protocol-level edits only reach items created after them; this brings an existing item up to date). Wholesale replace of that item's copies for the named tasks — omit task_names for all. Running tasks are refused. Needs the configure permission.",
        method="POST",
        path="/app/items/{item_name}/apply_protocol",
        scope="configure",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "task_names": "optional list of task names; omitted = every task in the protocol",
        },
    ),
    ToolSpec(
        name="reorder_item_milling_stages",
        description="Reorder the milling stages inside one of an item's task configs ('run the stress relief first'). order must be exactly the current stage names in their new sequence — this can never add, drop, or duplicate a stage. Echo the version from get_item_task_config; stale is refused. Needs the configure permission.",
        method="POST",
        path="/app/items/{item_name}/task_config/{task_name}/stages/reorder",
        scope="configure",
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
        description="Reorder the milling stages inside a protocol-level task config. Same rules as the item form; reaches items created (or re-applied) after. Needs the configure permission.",
        method="POST",
        path="/app/protocol/task_config/{task_name}/stages/reorder",
        scope="configure",
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
        description="Set (or clear) when a task may start: ISO-8601 timestamp (naive = instrument-local time), or null to clear. The workflow reads schedules at each task start, so a change takes effect at the next start; the schedule persists with the protocol. Needs the configure permission.",
        method="POST",
        path="/app/workflow/schedule",
        scope="configure",
        router="app",
        params={
            "task_name": "the task name, as listed by get_protocol",
            "scheduled_at": "ISO-8601 timestamp, or null to clear",
        },
    ),
    ToolSpec(
        name="update_item_detail",
        description="Patch an item's own document with dotted-path edits: poi.x/poi.y (metres, milling frame), alignment_area.left/top/width/height (frame fractions), description, defect.state (NONE/FAILURE/REWORK) and defect.description. milling_angle is read-only: it is an outcome of the Setup task, not an input. Echo the version from get_item_detail — a mismatch is refused (409 stale_config). Tasks re-record geometry at their own moments, so an edit here is what the NEXT run starts from, not a permanent override. Needs the configure permission.",
        method="POST",
        path="/app/items/{item_name}",
        scope="configure",
        router="app",
        params={
            "item_name": "the item (lamella) name",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from get_item_detail",
        },
    ),
    ToolSpec(
        name="update_protocol_task_config",
        description="Patch a task's protocol-level defaults (what new items copy) with dotted-path edits. Echo the version from get_protocol_task_config — a mismatch is refused (409 stale_config). Running tasks are never affected: they hold their item's copy; the edit reaches items created after. Needs the configure permission.",
        method="POST",
        path="/app/protocol/task_config/{task_name}",
        scope="configure",
        router="app",
        params={
            "task_name": "the task name, as listed by get_protocol",
            "patch": "object of dotted-path: value entries to set",
            "version": "the version token from get_protocol_task_config",
        },
    ),
    ToolSpec(
        name="update_item_task_config",
        description="Patch an item's task config with dotted-path edits (e.g. {'milling.mill_rough.stages.0.pattern.depth': 2e-6}). Echo the version from get_item_task_config — a mismatch is refused (409 stale_config): re-read and re-patch. A task currently running for that item is refused (it already copied its config); pending tasks pick the change up at start. Values are validated against the live field's type and declared bounds; the response echoes every change old -> new. Needs the configure permission.",
        method="POST",
        path="/app/items/{item_name}/task_config/{task_name}",
        scope="configure",
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
        scope="read",
        router="app",
        params={
            "since": "return events with seq greater than this (0 for all held)",
            "timeout": "seconds to wait for new events before returning (max 30)",
        },
    ),
    ToolSpec(
        name="stop_workflow",
        description="Stop the running workflow — the same path as the GUI's Stop button; the mill halts too. The safety action: available on the read scope, never waits on the command lock.",
        method="POST",
        path="/app/workflow/stop",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="start_workflow",
        description="Start a workflow — the Run button, remotely. Names the tasks to run (from the protocol) and optionally the items; omitted items means all of them. Refused while a workflow is already running.",
        method="POST",
        path="/app/workflow/start",
        scope="control",
        router="app",
        params={
            "task_names": "the tasks to run, from the protocol's task names",
            "item_names": "the items (lamellae) to run them for; omit for all",
        },
    ),
    ToolSpec(
        name="set_task_supervision",
        description="Set whether a task asks for supervision, in the live protocol. Takes effect at the workflow's next prompt-or-proceed decision, mid-run — the same behaviour as the GUI's supervised/automated toggle.",
        method="POST",
        path="/app/supervision",
        scope="control",
        router="app",
        params={
            "task_name": "the task to change, e.g. 'Rough Milling'",
            "supervise": "true to ask before acting, false to run automatically",
            "supervisor": "optional: who supervised questions are addressed to — 'human' (default) or 'agent'; the operator can always answer first",
        },
    ),
    ToolSpec(
        name="requeue_task",
        description="Queue a task for an item (again) in the RUNNING workflow — 'run 03's fiducial again'. Re-running a completed pair is the queue's own re-run mechanism; the queue only exists during a run.",
        method="POST",
        path="/app/queue/requeue",
        scope="control",
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
        path="/app/images",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="get_pending_prompt",
        description="The supervision question awaiting an answer, if any — message, options, and context images.",
        method="GET",
        path="/app/prompt",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="answer_prompt",
        description="Answer the pending supervision question, exactly as clicking the matching button would. True = the positive option. Echo the nonce from get_pending_prompt; if that question is no longer pending the answer is refused (409 stale_prompt) — re-read and answer the current one. For EditAlignmentArea and PickPOI the answer may carry a value: adjusted geometry that is placed into the widget (the operator sees it land) and then accepted through the same click path. An out-of-bounds or wrong-shape value is refused (422 invalid_value) without clicking anything.",
        method="POST",
        path="/app/prompt/answer",
        scope="control",
        router="app",
        params={
            "response": "true for the positive option, false for the negative",
            "nonce": "the nonce from get_pending_prompt, naming the question being answered",
            "value": "optional adjusted geometry: {left, top, width, height} (fractions of the frame, origin top-left) for EditAlignmentArea, or {x, y} (metres, microscope image coordinates, origin centre, +y up) for PickPOI",
        },
    ),
    ToolSpec(
        name="list_recent_experiments",
        description="Recently opened experiments on this machine, without loading them.",
        method="GET",
        path="/app/recent_experiments",
        scope="read",
        router="app",
    ),
)

CATALOG = CATALOG + APP_TOOLS


def tools_for_scopes(armed: Dict[str, bool]) -> Tuple[ToolSpec, ...]:
    """Catalog entries usable given armed scopes alone (router-blind)."""
    return tuple(t for t in CATALOG if armed.get(t.scope, False))


def tools_for_capabilities(capabilities: Dict) -> Tuple[ToolSpec, ...]:
    """Catalog entries usable given a full /capabilities payload.

    Filters on both dimensions: the tool's scope must be armed AND the router
    serving it must be mounted. The server enforces either way -- this filter
    only keeps the agent's tool list honest about what can succeed.
    """
    scopes = capabilities.get("scopes", {})
    routers = capabilities.get("routers", {})
    return tuple(
        t
        for t in CATALOG
        if scopes.get(t.scope, False) and routers.get(t.router, False)
    )
