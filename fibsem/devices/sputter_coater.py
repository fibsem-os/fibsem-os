"""The SputterCoater device: the coater that sputters a conductive layer on the grids.

``current`` is the sputter current, in amperes, within the limits the driver
reports. ``inserted`` reads whether the coater is in its working position. Both
come from the instrument; ``current`` is the only one a caller sets.

``run(seconds, grids)`` coats. Where the instrument positions the stage for each
grid itself, ``grids`` names which ones, by the coater's own 1-based number. A
coater that coats whatever is under it (``coats_grids`` False) refuses ``grids``
rather than ignoring them. Some instruments move the stage to the coater and back during a run, so ``run``
claims the ``stage`` resource as well as ``sputter_coater``.

What each instrument needs around a run (the beams on, the voltage put back after)
is its driver's, behind ``_run``. A backend implements ``read_current``,
``write_current`` and ``metadata_current``, ``read_inserted``, and ``_run``; one
whose instrument can say whether the coater is fitted answers ``is_installed``.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Optional, Sequence, Tuple

from fibsem.devices.core import Device, Parameter, command
from fibsem.devices.stage import STAGE_RESOURCE

SPUTTER_COATER_RESOURCE = "sputter_coater"


class SputterCoater(Device):
    current = Parameter(float, unit="A", doc="The sputter current.")
    inserted = Parameter(bool, doc="The coater is in its working position.")

    # Whether ``run`` takes the grids to coat: the instrument moves the stage to
    # each one itself.
    coats_grids = False

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="sputter_coater", parent=parent, **kwargs)

    @command
    def run(self, seconds: float, grids: Optional[Sequence[int]] = None) -> bool:
        """Sputter for *seconds*, on the numbered *grids* where the coater moves to
        each itself. Returns whether the coater is inserted afterwards."""
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError(f"{self.name}: a sputter run cannot last {seconds} s.")
        numbers = self._grid_numbers(grids)
        with ExitStack() as stack:
            for name in (SPUTTER_COATER_RESOURCE, STAGE_RESOURCE):
                stack.enter_context(self.resources.claim(name))
            self._run(seconds, numbers)
            return self._read_back()

    def _grid_numbers(
        self, grids: Optional[Sequence[int]]
    ) -> Optional[Tuple[int, ...]]:
        if grids is None:
            return None
        numbers = tuple(int(n) for n in grids)
        if not self.coats_grids:
            raise ValueError(
                f"{self.name}: this coater coats what is under it and takes no grids."
            )
        if not numbers or any(n < 1 for n in numbers):
            raise ValueError(
                f"{self.name}: grids are numbered from 1, not {list(numbers)}."
            )
        return numbers

    def _read_back(self) -> bool:
        if "current" in self._bound:
            self.current.get_value()
        return self.inserted.get_value()

    def facts(self) -> dict:
        return {
            "coats_grids": self.coats_grids,
            "is_installed": self._installed_or_none(),
        }

    def _installed_or_none(self) -> Optional[bool]:
        try:
            return self.is_installed()
        except Exception:  # a missing coater and a sick connection look the same
            return None

    # -- what a backend implements -------------------------------------------------

    def is_installed(self) -> Optional[bool]:
        """Whether the instrument reports the coater fitted, or None if it cannot say."""
        return None

    def _run(self, seconds: float, grids: Optional[Tuple[int, ...]]) -> None:
        raise NotImplementedError
