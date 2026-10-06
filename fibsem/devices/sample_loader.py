"""The SampleLoader device: the autoloader that exchanges grids with the stage.

It reports hardware facts only. The magazine is a row per slot, numbered from 1 as on
the magazine and in the vendor API, each with a state and the description the
hardware keeps for it. ``on_stage`` is what the loader says sits on the stage. Grid
names, which grid is which across scans and the holder's working slot are the grid
model's (``fibsem.microscopes._stage``), not the device's.

All parameters are read-only; what moves is a command. ``load`` and ``unload`` move
the stage, so they claim the ``stage`` resource as well as ``sample_loader``. ``scan``
has the loader check every slot physically, where a read of ``magazine`` is what it
already knows. ``set_description`` writes a slot's description and reads it back.

A backend implements ``read_magazine``, ``read_capacity``, ``read_exchange_time`` and,
when its hardware reports one, ``read_on_stage``; and the hooks ``_load``,
``_unload``, ``_scan`` and ``_set_description``. The base class claims the resources, keeps
``busy``, and after a scan or a description write reads the magazine back, which
updates its cache and emits its change signal (a scan that answers with the magazine
reports that instead of reading again).

A load or unload does not read anything back. AutoScript before 4.14 reports the
home slot of the grid on the stage as empty, so a read straight after an exchange
would say that grid is gone; the grid model keeps track of it instead, and reads
the magazine only when asked (``get_inventory``, ``run_inventory``).
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Optional, Tuple

from fibsem.devices.core import Device, Parameter, command
from fibsem.devices.stage import STAGE_RESOURCE

SAMPLE_LOADER_RESOURCE = "sample_loader"


class GridExchangeError(RuntimeError):
    """A grid could not be moved into, or out of, the holder's working slot."""


class MagazineSlotState(str, Enum):
    """What the loader reports for one magazine slot.

    ``UNKNOWN``: not scanned since the magazine was docked. ``EMPTY``: no grid.
    ``OCCUPIED``: a grid, in the magazine. ``LOADED``: the slot's grid is on the
    stage (only hardware that reports it; older AutoScript reads ``EMPTY``).
    """

    UNKNOWN = "unknown"
    EMPTY = "empty"
    OCCUPIED = "occupied"
    LOADED = "loaded"

    @classmethod
    def from_name(cls, name: Any) -> "MagazineSlotState":
        """A vendor's state (``"Occupied"``, ``State.OCCUPIED``), case-blind;
        anything not recognised is ``UNKNOWN``."""
        text = str(name).rsplit(".", 1)[-1].strip().lower()
        try:
            return cls(text)
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class MagazineSlot:
    """One magazine slot: its 1-based number, state and description."""

    number: int
    state: MagazineSlotState = MagazineSlotState.UNKNOWN
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "state": self.state.value,
            "description": self.description,
        }

    @staticmethod
    def from_dict(data: dict) -> "MagazineSlot":
        return MagazineSlot(
            number=int(data["number"]),
            state=MagazineSlotState(data.get("state", "unknown")),
            description=str(data.get("description") or ""),
        )


@dataclass(frozen=True)
class Magazine:
    """Every magazine slot, in slot order."""

    slots: Tuple[MagazineSlot, ...] = ()

    def slot(self, number: int) -> Optional[MagazineSlot]:
        """The slot of this number, or None."""
        for slot in self.slots:
            if slot.number == number:
                return slot
        return None

    def to_dict(self) -> dict:
        return {"slots": [slot.to_dict() for slot in self.slots]}

    @staticmethod
    def from_dict(data: dict) -> "Magazine":
        return Magazine(
            tuple(MagazineSlot.from_dict(s) for s in (data or {}).get("slots", ()))
        )


@dataclass(frozen=True)
class StageSample:
    """What the loader reports on the stage: whether a grid is there (None when it
    does not say), and its description."""

    present: Optional[bool] = None
    description: str = ""

    def to_dict(self) -> dict:
        return {"present": self.present, "description": self.description}

    @staticmethod
    def from_dict(data: dict) -> "StageSample":
        data = data or {}
        return StageSample(
            present=data.get("present"),
            description=str(data.get("description") or ""),
        )


class SampleLoader(Device):
    magazine = Parameter(Magazine, doc="Every magazine slot, as the loader last knew.")
    on_stage = Parameter(StageSample, doc="What the loader reports on the stage.")
    capacity = Parameter(int, doc="How many slots the magazine has.")
    exchange_time = Parameter(
        float, unit="s", doc="How long one exchange takes, an unload and a load."
    )
    busy = Parameter(bool, doc="Whether a command is running on the loader.")

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="sample_loader", parent=parent, **kwargs)
        self._busy = False

    @command
    def load(self, slot: int) -> None:
        """Bring the grid in a magazine slot (1-based) onto the stage."""
        with self._running(STAGE_RESOURCE):
            self._load(int(slot))

    @command
    def unload(self) -> None:
        """Put the grid on the stage back in the magazine."""
        with self._running(STAGE_RESOURCE):
            self._unload()

    @command
    def scan(self) -> Magazine:
        """Check every slot physically: slow, and the magazine must stay shut.
        Returns what it found."""
        with self._running():
            found = self._scan()
        if found is None:
            return self._read_back()
        # what the scan itself answered, without asking again
        self.magazine.report(found)
        if "on_stage" in self._bound:
            self.on_stage.get_value()
        return found

    @command
    def set_description(self, slot: int, text: str) -> MagazineSlot:
        """Write a slot's description and read it back. Raises
        ``GridExchangeError`` when the slot does not exist or the text did not stick."""
        slot = int(slot)
        with self._running():
            self._set_description(slot, text)
        written = self._read_back().slot(slot)
        if written is None:
            raise GridExchangeError(f"The sample loader has no slot {slot}.")
        if written.description.strip() != text.strip():
            raise GridExchangeError(
                f"Sample loader slot {slot} reads back '{written.description}', not "
                f"'{text}': the description did not stick."
            )
        return written

    def connect(self) -> "SampleLoader":
        super().connect()
        # A first value, so the first command's busy is a change, and is signalled.
        self.busy.get_value()
        return self

    def read_busy(self) -> bool:
        return self._busy

    @contextmanager
    def _running(self, *more: str) -> Iterator[None]:
        """Claim ``sample_loader`` and *more*, and hold ``busy``, while a command runs."""
        with ExitStack() as stack:
            for name in (SAMPLE_LOADER_RESOURCE, *more):
                stack.enter_context(self.resources.claim(name))
            self._set_busy(True)
            try:
                yield
            finally:
                self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if "busy" in self._bound:
            self.busy.report(busy)

    def _read_back(self) -> Magazine:
        if "on_stage" in self._bound:
            self.on_stage.get_value()
        return self.magazine.get_value()

    # -- what a backend implements -------------------------------------------------

    def _load(self, slot: int) -> None:
        raise NotImplementedError

    def _unload(self) -> None:
        raise NotImplementedError

    def _scan(self) -> Optional[Magazine]:
        """Scan; return the magazine when the scan answers with it, else None and
        it is read afterwards."""
        raise NotImplementedError

    def _set_description(self, slot: int, text: str) -> None:
        raise NotImplementedError
