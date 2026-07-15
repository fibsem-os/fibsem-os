from  __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass, field

from typing import List, Optional, TYPE_CHECKING

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point

if TYPE_CHECKING:
    from fibsem.ui.FibsemSpotBurnWidget import FibsemSpotBurnWidget

SLEEP_TIME = 1

def run_spot_burn(microscope: FibsemMicroscope,
                  coordinates: List[Point],
                  exposure_time: float,
                  milling_current: float,
                  beam_type: BeamType = BeamType.ION,
                  parent_ui: Optional['FibsemSpotBurnWidget'] = None,
                  stop_event: Optional[threading.Event] = None) -> None:
    """Run a spot burner job on the microscope. Exposes the specified coordinates for a the specified
    time at the specified current.
    Args:
        microscope: The microscope object.
        coordinates: List of points to burn. (0 - 1 in image coordinates)
        exposure_time: Time to expose each point in seconds.
        milling_current: Current to use for the spot.
        beam_type: The type of beam to use. (Default: BeamType.ION)
        parent_ui: The parent UI object to emit progress signals. (Default: None)
        stop_event: Threading event to signal cancellation. (Default: None)
    Returns:
        None
    """
    # - QUERY: do we need to set the full frame scanning mode each time, or only at the end?

    # coerce numeric parameters: protocol-editor fields can arrive as strings
    # (e.g. "3e-11"), which would break beam-current/timing arithmetic on hardware.
    exposure_time = float(exposure_time)
    milling_current = float(milling_current)

    # drop points outside the image bounds (0-1 normalised); set_spot rejects out-of-range
    # coordinates on hardware. The supervised widget filters these, so filter here too for
    # the unsupervised/automatic path (coordinates come straight from the stored config).
    in_bounds, dropped = [], []
    for pt in coordinates:
        (in_bounds if 0 <= pt.x <= 1 and 0 <= pt.y <= 1 else dropped).append(pt)
    if dropped:
        logging.warning(
            f"Skipping {len(dropped)} spot burn coordinate(s) outside image bounds (0-1): {dropped}"
        )
    coordinates = in_bounds

    total_estimated_time = len(coordinates) * exposure_time
    total_remaining_time = total_estimated_time

    # emit initial progress signal
    if parent_ui is not None:
        parent_ui.spot_burn_progress_signal.emit(
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
            if parent_ui is not None:
                parent_ui.spot_burn_progress_signal.emit(
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
    if parent_ui is not None:
        parent_ui.spot_burn_progress_signal.emit({"finished": True})

    microscope.set_beam_current(current=imaging_current, beam_type=beam_type)


@dataclass
class SpotBurnSettings:
    """The payload for a spot-burn run: where to burn, and with what.

    Shared, fibsem-level currency for the coordinate editor + the live spot-burn widget
    + :func:`run_spot_burn` (to which it maps 1:1). Workflow concerns such as the stage
    orientation live on the task config, not here.
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

    def run(self,
            microscope: FibsemMicroscope,
            beam_type: BeamType = BeamType.ION,
            parent_ui: Optional['FibsemSpotBurnWidget'] = None,
            stop_event: Optional[threading.Event] = None) -> None:
        """Run this spot burn on *microscope* (thin wrapper over :func:`run_spot_burn`)."""
        run_spot_burn(microscope=microscope,
                      coordinates=self.coordinates,
                      exposure_time=self.exposure_time,
                      milling_current=self.milling_current,
                      beam_type=beam_type,
                      parent_ui=parent_ui,
                      stop_event=stop_event)
