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


def tools_for_scopes(armed: Dict[str, bool]) -> Tuple[ToolSpec, ...]:
    """The catalog entries usable given the /capabilities scopes payload."""
    return tuple(t for t in CATALOG if armed.get(t.scope, False))
