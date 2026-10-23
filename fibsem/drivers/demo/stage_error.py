"""A simulated stage that does not go exactly where it is told.

Off unless the configuration asks for it::

    sim:
        stage_error:
            enabled: true
            scale: [1.06, 1.09]   # x, y: how far the stage really goes per metre it reports
            jitter: 0.01          # each move lands off by up to about this fraction of it
            seed: null            # an int repeats the same jitter run to run

The stage still reads back the position it was sent to, as a real stage's encoders
do. Only the sample scene sees where it really is, so a tiled overview's pictures
step further (or less far) than planned while every tile's recorded position looks
right, which is what the Arctis overviews showed and what the tile layout check and
``register_tiles`` are there to catch.

The scale is about where the stage was when the simulator started, so a position
reached twice is the same place, give or take the jitter. The jitter is drawn fresh
on every move, in proportion to that move's length on each axis, and does not add up
from one move to the next.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from fibsem.structures import FibsemStagePosition


class StageMoveError:
    """Where the simulated stage really is, given where it reports it is."""

    def __init__(
        self,
        scale: Tuple[float, float] = (1.0, 1.0),
        jitter: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.scale = (float(scale[0]), float(scale[1]))
        self.jitter = float(jitter)
        self._rng = np.random.default_rng(seed)
        self.origin: Tuple[float, float] = (0.0, 0.0)
        # This move's jitter (m, x and y), drawn by `moved`.
        self.residual: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_config(
        cls, config: Optional[Mapping[str, Any]]
    ) -> Optional["StageMoveError"]:
        """The `sim.stage_error` block, or None when it is absent or not enabled."""
        if not config or not config.get("enabled", False):
            return None
        scale = config.get("scale", 1.0)
        if isinstance(scale, (int, float)):
            scale = (scale, scale)
        elif isinstance(scale, Mapping):
            scale = (scale.get("x", 1.0), scale.get("y", 1.0))
        return cls(
            scale=tuple(scale),
            jitter=config.get("jitter", 0.0),
            seed=config.get("seed"),
        )

    def start_at(self, position: FibsemStagePosition) -> None:
        """Fix the point the scale is about: the stage's position at start."""
        self.origin = (position.x or 0.0, position.y or 0.0)
        self.residual = (0.0, 0.0)

    def moved(self, before: FibsemStagePosition, after: FibsemStagePosition) -> None:
        """Draw the jitter for a move from `before` to `after` (both as reported)."""
        if self.jitter <= 0:
            return
        dx = abs((after.x or 0.0) - (before.x or 0.0))
        dy = abs((after.y or 0.0) - (before.y or 0.0))
        # An axis that did not move stays where its last move left it.
        rx, ry = self.residual
        self.residual = (
            float(self._rng.normal(0.0, self.jitter * dx)) if dx else rx,
            float(self._rng.normal(0.0, self.jitter * dy)) if dy else ry,
        )

    def calibrated(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Where a move to `position` lands, leaving out the jitter: where a
        position saved on this stage (a holder slot) really is."""
        out = deepcopy(position)
        if out.x is not None:
            out.x = self.origin[0] + self.scale[0] * (out.x - self.origin[0])
        if out.y is not None:
            out.y = self.origin[1] + self.scale[1] * (out.y - self.origin[1])
        return out

    def actual(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Where the stage really is when it reports `position`."""
        out = self.calibrated(position)
        if out.x is not None:
            out.x += self.residual[0]
        if out.y is not None:
            out.y += self.residual[1]
        return out
