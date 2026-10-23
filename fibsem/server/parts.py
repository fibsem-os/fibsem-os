"""The optional parts of the ``/app`` interface.

Every app context serves the core (status, queue, runs, proposals, events,
log, notes, experiment and protocol). A part is served only when the context
implements its Protocol: its routes are mounted, ``/capabilities`` lists it,
and the catalog offers its tools (``ToolSpec.part``).
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "GRIDS",
    "Grids",
    "ITEMS",
    "Items",
    "LAMELLAE",
    "Lamellae",
    "MILLING_STAGES",
    "MillingStages",
    "PARTS",
    "POSES",
    "Poses",
    "served_parts",
]

ITEMS = "items"
POSES = "poses"
GRIDS = "grids"
MILLING_STAGES = "milling_stages"
LAMELLAE = "lamellae"


@runtime_checkable
class Items(Protocol):
    """The experiment's items: list, detail, outputs, history and per-item
    edits."""

    def item_detail(self, item_name: str) -> Dict[str, Any]: ...

    def task_outputs(self, item_name: str) -> Dict[str, Any]: ...

    def output_image(
        self, item_name: str, filename: str, max_width: int = 768
    ) -> Dict[str, Any]: ...

    def item_task_config(self, item_name: str, task_name: str) -> Dict[str, Any]: ...

    def item_stack(self, item_name: str, role: str) -> Any: ...

    def item_history(
        self, item_name: str, flt: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...

    def add_item(
        self,
        name: Optional[str] = None,
        stage_position: Optional[Dict[str, Any]] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    def remove_item(self, item_name: str) -> Dict[str, Any]: ...

    def apply_item_patch(
        self, item_name: str, patch: Dict[str, Any], version: str, timeout: float = 10.0
    ) -> Dict[str, Any]: ...

    def apply_item_task_config_patch(
        self,
        item_name: str,
        task_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]: ...

    def apply_protocol_to_item(
        self,
        item_name: str,
        task_names: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class Poses(Protocol):
    """Saved stage poses and objective positions on items, and moving to them."""

    def move_to_item_pose(self, item_name: str, pose_name: str) -> Dict[str, Any]: ...

    def set_item_pose(self, item_name: str, pose_name: str) -> Dict[str, Any]: ...

    def set_item_objective_position(self, item_name: str) -> Dict[str, Any]: ...

    def move_objective_to_item(self, item_name: str) -> Dict[str, Any]: ...

    def move_item(
        self,
        item_name: str,
        stage_position: Optional[Dict[str, Any]] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    def set_objective_position_on_all(self, item_name: str) -> Dict[str, Any]: ...


@runtime_checkable
class Grids(Protocol):
    """Grid records, their overviews, the grid protocol and grid runs."""

    def grids(self) -> Dict[str, Any]: ...

    def grid_detail(self, grid_name: str) -> Dict[str, Any]: ...

    def grid_output_image(
        self, grid_name: str, filename: str, max_width: int = 768
    ) -> Dict[str, Any]: ...

    def grid_markers(self, grid_name: str, filename: str) -> Dict[str, Any]: ...

    def overview_stage_position(
        self, grid_name: str, filename: str, px=None, fraction=None
    ) -> Dict[str, Any]: ...

    def add_grids(self) -> Dict[str, Any]: ...

    def rename_grid(self, grid_name: str, name: str) -> Dict[str, Any]: ...

    def set_grid_quality(
        self, grid_name: str, verdict: str, reason: str = ""
    ) -> Dict[str, Any]: ...

    def set_grid_note(
        self, grid_name: str, note: str, evidence: Optional[str] = None
    ) -> Dict[str, Any]: ...

    def remove_grid(self, grid_name: str) -> Dict[str, Any]: ...

    def grid_task_config(self, task_name: str) -> Dict[str, Any]: ...

    def apply_grid_task_config_patch(
        self, task_name: str, patch: Dict[str, Any], version: str
    ) -> Dict[str, Any]: ...

    def reorder_grid_tasks(self, order: List[str]) -> Dict[str, Any]: ...

    def set_grid_task_requirements(
        self, task_name: str, requires: List[str]
    ) -> Dict[str, Any]: ...

    def grid_workflow_plan(
        self,
        task_names: List[str],
        grid_names: Optional[List[str]] = None,
        screen_all: bool = False,
    ) -> Dict[str, Any]: ...

    def start_grid_workflow(
        self,
        task_names: List[str],
        grid_names: Optional[List[str]] = None,
        screen_all: bool = False,
        unload_at_end: bool = False,
        report_at_end: bool = False,
        timeout: float = 15.0,
    ) -> Dict[str, Any]: ...

    def queue_grid_tasks(
        self, task_names: List[str], grid_names: List[str]
    ) -> Dict[str, Any]: ...


@runtime_checkable
class MillingStages(Protocol):
    """Reordering the milling stages of a task config."""

    def reorder_milling_stages(
        self,
        level: str,
        item_name: str,
        task_name: str,
        milling_key: str,
        order: List[str],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class Lamellae(Protocol):
    """Lamella-only routes: thickness, correlation results, lamella
    defaults, the lamella run's plan, a lamella's animation."""

    def measure_thickness(
        self,
        item_name: str,
        path: str,
        width_um: Optional[float] = None,
        height_um: Optional[float] = None,
    ) -> Dict[str, Any]: ...

    def apply_item_correlation(
        self, item_name: str, run_path: str, timeout: float = 10.0
    ) -> Dict[str, Any]: ...

    def set_lamella_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]: ...

    def render_markers(
        self, path: str, grid_name: Optional[str] = None
    ) -> Dict[str, Any]: ...

    def lamella_run_plan(
        self,
        task_names: List[str],
        item_names: Optional[List[str]] = None,
        run_next: bool = False,
        add: bool = False,
    ) -> Dict[str, Any]: ...

    def animation_steps(self, item_name: str) -> Dict[str, Any]: ...


PARTS = {
    ITEMS: Items,
    POSES: Poses,
    GRIDS: Grids,
    MILLING_STAGES: MillingStages,
    LAMELLAE: Lamellae,
}


def served_parts(context: Any) -> List[str]:
    """The parts *context* implements, in PARTS order; none without a context."""
    if context is None:
        return []
    return [name for name, part in PARTS.items() if isinstance(context, part)]
