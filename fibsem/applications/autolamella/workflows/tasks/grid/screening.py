"""One action: inventory the grids, then run the protocol on every one of them.

The Arctis user's real ask is "inventory all my grids and acquire the overviews".
Nothing here is new; it is the order in which the pieces that exist get called,
written once so the button, a script and the agent server run the same thing.
On a fixed holder every present grid is already loaded, so the same call
is a run with zero exchanges.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    GridTaskManager,
    plan_grid_run,
    run_grid_tasks,
)
from fibsem.hooks import HookManager

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    from fibsem.microscope import FibsemMicroscope


def present_grids(
    microscope: "FibsemMicroscope", experiment: "Experiment"
) -> List[str]:
    """Refresh the inventory, record every present grid, and return their names
    in slot order. The selection "Screen all grids" runs over."""
    stage = microscope._stage
    # A read, not a scan: the operator will have inventoried the magazine (in
    # xT or from the Sample view), and a physical scan in front of every
    # screening run is the wait this call exists to avoid.
    inventory = stage.get_inventory()
    added = experiment.sync_grids_from_inventory(stage)
    if added:
        logging.info(f"Inventory added {len(added)} grid(s): {[g.name for g in added]}")
    names = [e.name for e in inventory if e.present and e.name]
    if not names:
        logging.warning("Inventory found no grids to screen.")
    return names


def screening_plan(
    microscope: "FibsemMicroscope",
    experiment: "Experiment",
    task_names: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """What "Screen all grids" would run, for a confirmation: the load and task
    steps per present grid. Refreshes the inventory to answer."""
    if task_names is None:
        task_names = experiment.grid_protocol.ordered_task_names
    return plan_grid_run(task_names, present_grids(microscope, experiment))


def screen_grids(
    microscope: "FibsemMicroscope",
    experiment: "Experiment",
    task_names: Optional[List[str]] = None,
    parent_ui: Optional["AutoLamellaUI"] = None,
    hook_manager: Optional[HookManager] = None,
) -> GridTaskManager:
    """Inventory, then the protocol's tasks (in its order, unless given) on every
    present grid. Returns the manager, for its queue and run summary."""
    return run_grid_tasks(
        microscope,
        experiment,
        task_names=task_names,
        grid_names=present_grids(microscope, experiment),
        parent_ui=parent_ui,
        hook_manager=hook_manager,
    )
