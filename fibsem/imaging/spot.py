from __future__ import annotations

import logging
import threading
import time
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

SLEEP_TIME = 1


@dataclass
class SpotBurnSettings:
    """The payload for a spot-burn run: where to burn, and with what.

    Shared, fibsem-level currency for the coordinate editor + the live spot-burn widget
    + :func:`run_spot_burn`. Workflow concerns such as the stage orientation live on the
    task config, not here.
    """

    coordinates: List[Point] = field(default_factory=list)
    milling_current: float = 60e-12  # amperes
    exposure_time: float = 10.0      # seconds

    def to_dict(self) -> dict:
        return {
            "coordinates": [pt.to_dict() for pt in self.coordinates],
            "milling_current": self.milling_current,
            "exposure_time": self.exposure_time,
        }

    @classmethod
    def from_dict(cls, ddict: dict) -> "SpotBurnSettings":
        return cls(
            coordinates=[Point.from_dict(pt) for pt in ddict.get("coordinates", [])],
            milling_current=ddict.get("milling_current", 60e-12),
            exposure_time=ddict.get("exposure_time", 10.0),
        )


def run_spot_burn(microscope: FibsemMicroscope,
                  settings: SpotBurnSettings,
                  beam_type: BeamType = BeamType.ION,
                  stop_event: Optional[threading.Event] = None) -> None:
    """Run a spot burner job on the microscope. Exposes the coordinates in *settings* for
    the exposure time at the milling current it specifies.

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
    # - QUERY: do we need to set the full frame scanning mode each time, or only at the end?

    # coerce numeric parameters: protocol-editor fields can arrive as strings
    # (e.g. "3e-11"), which would break beam-current/timing arithmetic on hardware.
    # Read into locals rather than writing back — settings belongs to the caller.
    exposure_time = float(settings.exposure_time)
    milling_current = float(settings.milling_current)

    # drop points outside the image bounds (0-1 normalised); set_spot rejects out-of-range
    # coordinates on hardware. The supervised widget filters these, so filter here too for
    # the unsupervised/automatic path (coordinates come straight from the stored config).
    in_bounds, dropped = [], []
    for pt in settings.coordinates:
        (in_bounds if 0 <= pt.x <= 1 and 0 <= pt.y <= 1 else dropped).append(pt)
    if dropped:
        logging.warning(
            f"Skipping {len(dropped)} spot burn coordinate(s) outside image bounds (0-1): {dropped}"
        )
    coordinates = in_bounds

    total_estimated_time = len(coordinates) * exposure_time
    total_remaining_time = total_estimated_time

    # emit initial progress signal
    microscope.spot_burn_progress_signal.emit(
        {
            "current_point": 0,
            "total_points": len(coordinates),
            "remaining_time": exposure_time,
            "total_remaining_time": total_remaining_time,
            "total_estimated_time": total_estimated_time,
        }
    )

    # set the beam current to the milling current
    imaging_current = microscope.get_beam_current(beam_type=beam_type)
    microscope.set_beam_current(current=milling_current, beam_type=beam_type)

    for i, pt in enumerate(coordinates, 1):

        if stop_event is not None and stop_event.is_set():
            logging.info(f"Spot burn cancelled before point {i}/{len(coordinates)}.")
            break

        logging.info(f'burning spot {i}: {pt}, exposure time: {exposure_time}, milling current: {milling_current}')

        microscope.blank(beam_type=beam_type)
        microscope.set_spot_scanning_mode(point=pt, beam_type=beam_type)
        microscope.unblank(beam_type=beam_type)

        # countdown for the exposure time, emit progress signal
        remaining_time = exposure_time
        while remaining_time > 0:
            if stop_event is not None and stop_event.is_set():
                microscope.blank(beam_type=beam_type)
                logging.info(f"Spot burn cancelled during point {i}/{len(coordinates)}.")
                break
            time.sleep(SLEEP_TIME)
            remaining_time -= SLEEP_TIME
            total_remaining_time -= SLEEP_TIME
            microscope.spot_burn_progress_signal.emit(
                {
                    "current_point": i,
                    "total_points": len(coordinates),
                    "remaining_time": remaining_time,
                    "total_remaining_time": total_remaining_time,
                    "total_estimated_time": total_estimated_time,
                }
            )

    # always restore full frame scanning mode and imaging current
    microscope.set_full_frame_scanning_mode(beam_type=beam_type)

    # emit finished signal
    microscope.spot_burn_progress_signal.emit({"finished": True})

    microscope.set_beam_current(current=imaging_current, beam_type=beam_type)
