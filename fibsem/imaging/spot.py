from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from fibsem.structures import BeamType, Point

if TYPE_CHECKING:
    # Type-checking only. `fibsem/imaging/__init__.py` star-imports this module, so a
    # runtime import here makes every `fibsem.imaging.*` import pull in the microscope
    # -- and `fibsem.microscope` imports `fibsem.fm.microscope`, which reaches back
    # into `fibsem.imaging`. `from __future__ import annotations` is already on, so the
    # annotation below needs no runtime object.
    from fibsem.microscope import FibsemMicroscope


@dataclass
class SpotBurnSettings:
    """The payload for a spot-burn run: where to burn, and with what.

    Shared, fibsem-level currency for the coordinate editor + the live spot-burn widget
    + :func:`run_spot_burn`. Workflow concerns such as the stage orientation live on the
    task config, not here.
    """

    coordinates: List[Point] = field(default_factory=list)
    milling_current: float = 60e-12  # amperes
    exposure_time: float = 10.0  # seconds
    # beam preset for preset-driven backends (TESCAN); None means the backend's
    # default. milling_current is ignored wherever a preset drives the beam.
    preset: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "coordinates": [pt.to_dict() for pt in self.coordinates],
            "milling_current": self.milling_current,
            "exposure_time": self.exposure_time,
            "preset": self.preset,
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> "SpotBurnSettings":
        return cls(
            coordinates=[Point.from_dict(pt) for pt in ddict.get("coordinates", [])],
            milling_current=ddict.get("milling_current", 60e-12),
            exposure_time=ddict.get("exposure_time", 10.0),
            preset=ddict.get("preset", None),
        )


def run_spot_burn(
    microscope: FibsemMicroscope,
    settings: SpotBurnSettings,
    beam_type: BeamType = BeamType.ION,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run a spot burn job on the microscope. Exposes the coordinates in *settings* for
    the exposure time at the milling current it specifies.

    Delegates to ``microscope.run_spot_burn`` so each backend supplies its own
    mechanism: the default implementation blanks and parks the beam per point
    (ThermoFisher, simulator), while TESCAN overrides it with a DrawBeam layer of
    timed dots — its FIB scan API has no blanker or beam parking.

    Progress is reported via ``microscope.spot_burn_progress_signal`` (a dict), which the
    status bar and the spot burn widget subscribe to.
    Args:
        microscope: The microscope object.
        settings: What to burn — coordinates (0-1 image coordinates), exposure time per
            point in seconds, and the milling current to burn at.
        beam_type: The type of beam to use. (Default: BeamType.ION)
        stop_event: Threading event to signal cancellation. (Default: None)
    Returns:
        None
    """
    return microscope.run_spot_burn(
        settings=settings, beam_type=beam_type, stop_event=stop_event
    )
