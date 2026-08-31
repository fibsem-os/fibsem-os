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
        name="get_task_outputs",
        description="The files an item's completed tasks produced (reference images by role).",
        method="GET",
        path="/app/task_outputs/{item_name}",
        scope="read",
        router="app",
        params={"item_name": "the item (lamella) name"},
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
        name="get_pending_prompt",
        description="The supervision question awaiting an answer, if any — message, options, and context images.",
        method="GET",
        path="/app/prompt",
        scope="read",
        router="app",
    ),
    ToolSpec(
        name="answer_prompt",
        description="Answer the pending supervision question, exactly as clicking the matching button would. True = the positive option.",
        method="POST",
        path="/app/prompt/answer",
        scope="control",
        router="app",
        params={"response": "true for the positive option, false for the negative"},
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
