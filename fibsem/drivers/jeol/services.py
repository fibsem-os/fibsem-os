"""JEOL's services: milling and spot burn on JEOL's process queue.

`JeolMilling` mills by adding items to the FIB's process group: an ``area`` item for a
rectangle and a ``line`` item for a horizontal or vertical line, placed in pixels of a
fixed patterning frame. Each item carries its own beam mode and dose, so milling never
sets the beam's current: the recipe's milling current picks the beam mode of every
item. JEOL estimates the time of each item, and reports the time remaining while the
process runs.

`JeolSpotBurn` burns spots the same way: the FIB has no blanker and can't park the
beam, so each point is a ``spot`` item, all run as one process by the milling service.

`JeolStageMovement` is the shared stage movement (the stage is in fibsem's frame, so
its geometry holds) with no safe-rotation sequence before an absolute move.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

from fibsem import constants
from fibsem.cancellation import AnyStopEvent, OperationCancelledError
from fibsem.devices.beam import Beam
from fibsem.devices.core import ParameterMetadata, command
from fibsem.drivers.jeol.devices import nearest_beam_current
from fibsem.drivers.jeol.settings import JeolMillingSettings
from fibsem.imaging.spot import SpotBurnProgress, SpotBurnSettings, SpotBurnStatus
from fibsem.milling.progress import MillingProgress
from fibsem.services.milling import Milling, bind_milling, progress_update
from fibsem.services.spot_burn import SpotBurn, bind_spot_burn
from fibsem.services.stage_movement import StageMovement
from fibsem.structures.settings import (
    BeamType,
    CrossSectionPattern,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    MillingState,
    Point,
)

if TYPE_CHECKING:
    import threading

    from fibsem.drivers.jeol.connection import JeolConnection
    from fibsem.drivers.jeol.microscope import JeolMicroscope

# JEOL's process status -> fibsem's milling state. An unlisted status is an error.
PROCESS_STATES: Dict[str, MillingState] = {
    "Exit": MillingState.IDLE,
    "Stop": MillingState.IDLE,
    "End": MillingState.RUNNING,
    "Start": MillingState.RUNNING,
    "AfterCare": MillingState.RUNNING,
    "Pause": MillingState.PAUSED,
}

# The pixel frames items are placed in: areas in one, lines in a finer one.
AREA_FRAME = (1280, 960)
LINE_FRAME = (5120, 3840)
SPOT_FRAME = (1280, 960)

# The scan directions a rectangle can take, as the side the pass finishes on.
SCAN_DIRECTIONS = ("BottomToTop", "TopToBottom", "LeftToRight", "RightToLeft")
_FINISH_SIDES = {
    "BottomToTop": "Top",
    "TopToBottom": "Bottom",
    "LeftToRight": "Right",
    "RightToLeft": "Left",
}
# The other directions fibsem knows, as the nearest JEOL has; anything else is
# milled in the default direction.
_NEAREST_SCAN_DIRECTIONS = {
    "DynamicTopToBottom": "TopToBottom",
    "DynamicLeftToRight": "LeftToRight",
}
DEFAULT_SCAN_DIRECTION = "TopToBottom"

# How long to wait before starting a process: the beam mode is still being set.
START_DELAY = 2.0

# The area JEOL assumes for a spot when it turns a dose (nC/um2) into an exposure
# time (time = dose * area / current). Measured 2026-10-01: dose 333 ran 5 s at beam
# mode 9 (100 pA), at 50 and 100 um HFW. Taken to be fixed, not the probe size.
SPOT_AREA_UM2 = 1 / 666


def _scan_direction(direction: str) -> str:
    """The JEOL scan direction for fibsem's *direction*: itself, the nearest, or the
    default, with a warning when it changed."""
    if direction in _FINISH_SIDES:
        return direction
    nearest = _NEAREST_SCAN_DIRECTIONS.get(direction, DEFAULT_SCAN_DIRECTION)
    logging.warning(f"JEOL can't scan {direction!r}; using {nearest!r}.")
    return nearest


def dose_for_depth(depth: float, milling_rate: float) -> float:
    """The dose (nC/um2) that mills *depth* (m) at *milling_rate*, the charge that
    removes a unit of volume (C/m3, ``JeolMillingSettings.milling_rate``)."""
    if milling_rate <= 0:
        raise ValueError(f"The milling rate must be positive, got {milling_rate}.")
    return depth * milling_rate * 1e-3  # C/m2 -> nC/um2


def spot_burn_dose(exposure_time: float, current: float) -> float:
    """The dose (nC/um2) for a spot that exposes for *exposure_time* s at *current* A."""
    return current * 1e9 * exposure_time / SPOT_AREA_UM2


def to_frame_pixels(
    point: Point, hfw: float, aspect: float, frame: Tuple[int, int]
) -> Tuple[int, int]:
    """A point in fibsem's pattern coordinates (metres from the image centre, +y up)
    in pixels of *frame* (from the top left, +y down). *aspect* is the image's height
    over its width."""
    width, height = frame
    vfw = hfw * aspect
    x = (point.x + hfw / 2) / hfw * width
    y = (vfw / 2 - point.y) / vfw * height
    return int(round(x)), int(round(y))


class JeolMilling(Milling):
    """JEOL milling, on the FIB's process queue, over the microscope's connection."""

    parent: JeolMicroscope

    # The field the patterns are placed in; the current picks each item's beam mode;
    # JEOL's own milling rate makes its dose.
    setting_names = ("milling_channel", "hfw", "milling_current")
    settings_class = JeolMillingSettings
    scan_directions = SCAN_DIRECTIONS

    def __init__(self, connection: JeolConnection, **kwargs: Any):
        super().__init__(**kwargs)
        self.connection = connection
        # The beam mode and milling rate the last setup chose, which each item is
        # drawn at.
        self._beam_mode: Optional[int] = None
        self._current: Optional[float] = None
        self._milling_rate: Optional[float] = None

    def _fib(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.connection.fib_call(method, *args, **kwargs)

    @property
    def _models(self) -> Any:
        return self.connection.sdk.models

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "milling_channel":
            return ParameterMetadata(choices=(BeamType.ION,))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        if settings.milling_channel is not BeamType.ION:
            raise ValueError("JEOL mills with the ion beam only.")
        self.connection.select_view(BeamType.ION)
        self._clear()
        self.ion.hfw.write_through(settings.hfw)
        self.parent.milling_channel = settings.milling_channel
        self._beam_mode, self._current = nearest_beam_current(
            float(settings.milling_current), BeamType.ION
        )
        self._milling_rate = self.own_settings(settings).milling_rate
        logging.info(
            f"JEOL milling at beam mode {self._beam_mode} ({self._current:.3e} A)"
        )

    def _draw(self, pattern: FibsemPatternSettings, hook: str) -> None:
        # rectangles and lines only (the hooks below); neither can be rotated
        if self._beam_mode is None:
            raise RuntimeError("Milling is not set up: call setup before drawing.")
        if getattr(pattern, "rotation", 0):
            raise ValueError(
                f"JEOL can't rotate a pattern; {type(pattern).__name__} has rotation "
                f"{pattern.rotation}."
            )
        super()._draw(pattern, hook)

    def _field(self) -> Tuple[float, float]:
        """The ion field of view and the image's height over its width."""
        ion = self.ion
        width, height = ion.resolution.get_value()
        return ion.hfw.get_value(), height / width

    def _draw_rectangle(self, pattern: FibsemRectangleSettings) -> None:
        models = self._models
        direction = _scan_direction(pattern.scan_direction)
        finish_face = getattr(models.EProcessEndside, _FINISH_SIDES[direction])
        if pattern.cross_section is CrossSectionPattern.Rectangle:
            mode = models.EProcessMode.MILL
        elif pattern.cross_section is CrossSectionPattern.RegularCrossSection:
            mode = models.EProcessMode.RAPID
        elif pattern.cross_section is CrossSectionPattern.CleaningCrossSection:
            # JEOL has no cleaning cross section: milled as a regular one, as always
            logging.warning("JEOL has no cleaning cross section; milling it as RAPID.")
            mode = models.EProcessMode.RAPID
        else:
            logging.warning(
                f"JEOL has no {pattern.cross_section} cross section; milling it as MILL."
            )
            mode = models.EProcessMode.MILL

        hfw, aspect = self._field()
        # an area is placed by its top-left corner
        corner = Point(
            pattern.centre_x - pattern.width / 2, pattern.centre_y + pattern.height / 2
        )
        x, y = to_frame_pixels(corner, hfw, aspect, AREA_FRAME)
        item = models.ProcessItemData.area(
            mode=mode,
            beam=self._beam_mode,
            dose=dose_for_depth(pattern.depth, self._milling_rate),
            x=x,
            y=y,
            width=pattern.width * constants.METRE_TO_MICRON,
            height=pattern.height * constants.METRE_TO_MICRON,
            width_resolution=AREA_FRAME[0],
            height_resolution=AREA_FRAME[1],
            finish_face=finish_face,
        )
        self._add(item)

    def _draw_line(self, pattern: FibsemLineSettings) -> None:
        models = self._models
        if pattern.start_y == pattern.end_y:
            line_type = models.EProcessLineType.HORIZONTAL
        elif pattern.start_x == pattern.end_x:
            line_type = models.EProcessLineType.VERTICAL
        else:
            # as the old driver did: nothing to convert it to, so it is left out
            logging.warning(
                "JEOL mills horizontal or vertical lines only; a diagonal line is "
                "not drawn."
            )
            return
        hfw, aspect = self._field()
        x, y = to_frame_pixels(
            Point(pattern.start_x, pattern.start_y), hfw, aspect, LINE_FRAME
        )
        item = models.ProcessItemData.line(
            mode=models.EProcessMode.MILL,
            line_type=line_type,
            beam=self._beam_mode,
            dose=dose_for_depth(pattern.depth, self._milling_rate),
            x=x,
            y=y,
            width=(pattern.end_x - pattern.start_x) * constants.METRE_TO_MICRON,
            height=(pattern.end_y - pattern.start_y) * constants.METRE_TO_MICRON,
            width_resolution=LINE_FRAME[0],
            height_resolution=LINE_FRAME[1],
        )
        self._add(item)

    def _draw_spot(self, point: Point, exposure_time: float) -> None:
        """A spot at *point* (0 to 1, from the top left) that exposes for
        *exposure_time* at the set-up current. Only `JeolSpotBurn` draws one."""
        models = self._models
        width, height = SPOT_FRAME
        item = models.ProcessItemData.spot(
            mode=models.EProcessMode.MILL,
            beam=self._beam_mode,
            dose=spot_burn_dose(exposure_time, self._current),
            x=int(round(point.x * width)),
            y=int(round(point.y * height)),
            width_resolution=width,
            height_resolution=height,
        )
        self._add(item)

    def _add(self, item: Any) -> None:
        """Add *item* to the process group. One JEOL can't run is warned about and
        the group cleared, as the old driver did."""
        self._fib("addProcessItem", item)
        if not self._fib("getProcessExecutable"):
            logging.warning(f"JEOL can't run the process item {item}; cleared.")
            self._clear()

    # -- running ---------------------------------------------------------------------

    def _status(self) -> str:
        _, name = self._fib("getProcessStatus")
        return name

    def read_state(self) -> MillingState:
        name = self._status()
        state = PROCESS_STATES.get(name, MillingState.ERROR)
        logging.debug(f"JEOL process status {name!r} -> {state.name}")
        return state

    def _start(self) -> None:
        self._fib("startProcess")

    def _stop(self) -> None:
        if PROCESS_STATES.get(self._status()) in (
            MillingState.RUNNING,
            MillingState.PAUSED,
        ):
            self._fib("stopProcess")

    def _pause(self) -> None:
        if self._status() == "Start":
            self._fib("pauseProcess")

    def _resume(self) -> None:
        if self._status() == "Pause":
            self._fib("startProcess")

    def _estimate(self) -> float:
        """The sum of JEOL's own time for each item."""
        count = self._fib("getProcessItemCount")
        return float(
            sum(
                vars(self._fib("getProcessItem", i))["process_time"]
                for i in range(count)
            )
        )

    def _poll(self) -> MillingProgress:
        state = self.read_state()
        remaining = self._fib("getRemainProcessTime")
        return progress_update(state=state, remaining=float(remaining))

    def _before_run(self) -> None:
        self.connection.wait(START_DELAY)

    def _clear(self) -> None:
        self._fib("deleteProcessGroup")


def bind_jeol_milling(microscope: JeolMicroscope) -> Optional[JeolMilling]:
    """Build ``milling`` for a connected JEOL microscope whose beams are built."""
    return bind_milling(JeolMilling, microscope, connection=microscope._connection)


class JeolStageMovement(StageMovement):
    """The shared moves, with no safe-rotation sequence before an absolute move: the
    old driver moved straight to the position, and the JEOL stage reaches the FIB by
    tilting, not turning."""

    def _safe_rotation(self, stage_position: FibsemStagePosition) -> None:
        """Nothing before the move itself."""


class JeolSpotBurn(SpotBurn):
    """JEOL spot burns, through the milling service.

    The FIB has no blanker and can't park the beam, so the point-by-point burn is
    impossible. The points go into one process instead, a spot item each, at the beam
    mode nearest the requested current, and the milling service runs it.
    """

    parent: JeolMicroscope

    # Milling never sets the beam: each item carries its own beam mode.
    saved_conditions = ()

    @classmethod
    def _can_burn(cls, ion: Optional[Beam]) -> bool:
        return ion is not None

    @command
    def run(
        self,
        settings: SpotBurnSettings,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
    ) -> SpotBurnStatus:
        """As `SpotBurn.run`, but an exposure time that isn't positive is refused
        before anything is touched."""
        if float(settings.exposure_time) <= 0:
            raise ValueError(
                f"exposure_time must be positive, got {settings.exposure_time}."
            )
        return super().run(settings, stop_event=stop_event)

    def supported_settings(self) -> Dict[str, ParameterMetadata]:
        # the burn current picks a beam mode, so it has the ion beam's currents
        supported = {name: ParameterMetadata() for name in self.setting_names}
        supported["milling_current"] = self.ion.current.metadata
        return supported

    def _setup(self, ion: Beam, current: float) -> None:
        self.parent.milling.setup(
            FibsemMillingSettings(
                milling_channel=BeamType.ION,
                milling_current=current,
                hfw=ion.hfw.get_value(),
            ),
            name="SpotBurn",
        )

    def _burn(
        self,
        ion: Beam,
        points: Sequence[Point],
        exposure_time: float,
        current: float,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    ) -> SpotBurnStatus:
        """Draw a spot per point, then run the process, reporting the point JEOL is on
        from the time remaining (it reports no per-item progress, and runs the spots in
        order at the exposure time each)."""
        microscope = self.parent
        milling = microscope.milling
        for point in points:
            milling._draw_spot(point, exposure_time)

        n = len(points)
        total = n * exposure_time

        def report(progress: MillingProgress) -> None:
            if progress.remaining_time is None or progress.estimated_time is None:
                return
            elapsed = max(0.0, progress.estimated_time - progress.remaining_time)
            point = min(n, int(elapsed // exposure_time) + 1)
            self._report(
                SpotBurnProgress(
                    status=SpotBurnStatus.BURNING,
                    current_point=point,
                    total_points=n,
                    remaining_time=max(0.0, point * exposure_time - elapsed),
                    total_remaining_time=max(0.0, total - elapsed),
                    total_estimated_time=total,
                )
            )

        # The run reports a spot burn, not a mill: the milling progress signal is
        # blocked while it runs.
        signal = getattr(microscope, "milling_progress_signal", None)
        quiet = signal.blocked() if signal is not None else contextlib.nullcontext()
        milling.progress.changed.connect(report)
        try:
            with quiet:
                milling.run(
                    stop_event=AnyStopEvent(self._stop_requested, *_events(stop_event))
                )
        except OperationCancelledError:
            logging.info("Spot burn cancelled.")
            return SpotBurnStatus.CANCELLED
        finally:
            milling.progress.changed.disconnect(report)
        return SpotBurnStatus.FINISHED

    def _finish(self, ion: Beam) -> None:
        try:
            self.parent.milling.restore()
        except Exception:
            logging.exception("Failed to restore the ion beam after the spot burn")


def _events(stop_event: Any) -> tuple:
    return () if stop_event is None else (stop_event,)


def bind_jeol_spot_burn(microscope: JeolMicroscope) -> Optional[JeolSpotBurn]:
    """Build ``spot_burn`` for a JEOL microscope whose milling is built."""
    if microscope.milling is None:
        return None
    return bind_spot_burn(JeolSpotBurn, microscope)
