from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from fibsem.codec import decode, encode
from fibsem.structures import (
    BeamType,
    Configurable,
    Display,
    Point,
    RangeLimit,
    param,
)

if TYPE_CHECKING:
    # Type-checking only. `fibsem/imaging/__init__.py` star-imports this module, so a
    # runtime import here makes every `fibsem.imaging.*` import pull in the microscope
    # -- and `fibsem.microscope` imports `fibsem.fm.microscope`, which reaches back
    # into `fibsem.imaging`. `from __future__ import annotations` is already on, so the
    # annotation below needs no runtime object.
    from fibsem.microscope import FibsemMicroscope


@dataclass
class SpotBurnSettings(Configurable):
    """The payload for a spot-burn run: where to burn, and with what.

    Shared, fibsem-level currency for the coordinate editor + the live spot-burn widget
    + :func:`run_spot_burn`. Workflow concerns such as the stage orientation live on the
    task config, not here.
    """

    coordinates: List[Point] = param(
        default_factory=list,
        doc="Where to burn, as points in the ion beam's image, 0 to 1 on both axes. Points outside the image are dropped.",
    )
    milling_current: float = param(
        100e-12,
        device_parameter="beam.current",
        doc="The ion beam current each spot is burned with.",
        display=Display("Beam Current"),
    )
    # The widget's range, and what an answer at the spot burn question is checked against.
    exposure_time: float = param(
        10.0,
        unit="s",
        limits=RangeLimit(0.1, 60.0),
        doc="How long each spot is exposed.",
        display=Display("Exposure Time", decimals=3),
    )

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "SpotBurnSettings":
        return decode(cls, ddict)


class SpotBurnStatus(str, Enum):
    """What a spot burn is doing, or how it ended.

    A `str` mixin so a report reads as itself in a log line, matching `TiledStatus`
    and `FluorescenceAcquisitionStatus`.
    """

    BURNING = "burning"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether the run is over, however it ended.

        Asked by both consumers. Answered once here rather than restated as a
        membership tuple at each of them: the next status added is the one somebody
        forgets to add to one of the two, and the symptom is a progress bar that
        never clears.
        """
        return self is not SpotBurnStatus.BURNING


@dataclass(frozen=True)
class SpotBurnProgress:
    """One report on the spot burn service's ``progress``.

    Replaces the bare dict the signal carried, whose three shapes were told apart by
    the presence of a `finished` key and then a nested `error` flag -- two booleans
    encoding four outcomes, one of which could not be expressed at all. A cancelled
    burn reported `{"finished": True}`, identical to a completed one, so cancelling
    rendered "Done".

    Every field but `status` is absent on some report, so every field but `status`
    has a default.

    Equality is left generated. Nothing here carries a numpy array, so unlike
    `TiledProgress` there is no reason to turn it off.
    """

    status: SpotBurnStatus
    # 1-based, because `run_spot_burn` counts with `enumerate(coordinates, 1)`. The
    # initial report carries 0, meaning "not started". Deliberately unlike
    # milling progress's 0-based `current_stage`: there is nothing to correct
    # here, so do not port a `display_*` property over -- it would double-count.
    current_point: Optional[int] = None
    total_points: Optional[int] = None
    remaining_time: Optional[float] = None
    total_remaining_time: Optional[float] = None
    total_estimated_time: Optional[float] = None
    # Only on FAILED. The text of the exception that ended the run.
    error: Optional[str] = None


def show_after_burn(microscope: FibsemMicroscope) -> None:
    """Take a FIB image of what a burn left and push it to the view, so whoever
    looks next sees the marks."""
    image = microscope.acquire_image(beam_type=BeamType.ION)
    if microscope.imaging is not None:
        microscope.imaging.show(image)


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

    Progress is reported on ``microscope.spot_burn.progress``, which the
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
