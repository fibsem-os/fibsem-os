"""Automated lamella targeting grid task (stub).

Screens a loaded grid to pick lamella milling targets: acquire an overview,
segment / detect features, then select + emit lamella targets. Stub for now —
``_run`` only logs the intended pipeline steps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, Type

from fibsem.applications.autolamella.workflows.tasks.grid.base import (
    GridTask,
    GridTaskConfig,
)


@dataclass
class AutoLamellaTargetingGridTaskConfig(GridTaskConfig):
    """Configuration for the automated lamella targeting grid task."""
    task_type: ClassVar[str] = "AUTOLAMELLA_TARGETING_GRID"
    display_name: ClassVar[str] = "AutoLamella Targeting"


class AutoLamellaTargetingGridTask(GridTask):
    """Screen the grid and select lamella milling targets (stub: logs only)."""
    config_cls: ClassVar[Type[GridTaskConfig]] = AutoLamellaTargetingGridTaskConfig
    config: AutoLamellaTargetingGridTaskConfig

    def _run(self):
        # TODO: implement the targeting pipeline (stub: log steps with a delay)
        logging.info(f"AutoLamella targeting for grid {self.grid.name} — not yet implemented:")
        logging.info("1. Acquire overview tiles")
        time.sleep(5)
        logging.info("2. Run segmentation / detection on the overview")
        time.sleep(5)
        logging.info("3. Select + emit lamella targets")
        self.update_status_ui("AutoLamella targeting (not implemented)")
