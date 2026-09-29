"""Fixture milling strategy."""

import threading
from dataclasses import dataclass, field
from typing import Optional

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import (
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
)
from fibsem_test_plugin import STRATEGY_NAME


@dataclass
class FixtureMillingStrategyConfig(MillingStrategyConfig):
    passes: int = field(default=1, metadata={"label": "Passes"})


class FixtureMillingStrategy(MillingStrategy[FixtureMillingStrategyConfig]):
    name: str = STRATEGY_NAME
    fullname: str = "Fixture Milling Strategy"
    config_class = FixtureMillingStrategyConfig

    # Keeps the fixture out of the strategy dropdown if a developer happens to
    # have it installed in the environment they run the app from.
    selectable: bool = False

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui=None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        raise NotImplementedError("fixture strategy is never executed")
