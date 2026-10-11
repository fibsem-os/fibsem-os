"""Tescan's services.

`TescanMilling` mills on DrawBeam: a layer made from the milling preset on the ion
column, and a second connection to stop it from another thread.

`TescanSpotBurn` burns spots through the milling service: the FIB has no blanker and
can't park the beam, so the points are timed dots on a DrawBeam layer at the spot
burn preset, run and put back by ``microscope.milling``.

A ``run`` loads the layer (DrawBeam estimates only a loaded one), shows a progress bar
in Essence for its length, and reads the state, the elapsed time and the total from
``DrawBeam.GetStatus`` in one call each look.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import fibsem.constants as constants
from fibsem.cancellation import AnyStopEvent, OperationCancelledError
from fibsem.devices.beam import Beam
from fibsem.devices.core import ParameterMetadata, command
from fibsem.drivers.tescan import microscope as tescan
from fibsem.drivers.tescan.microscope import (
    DEFAULT_IMAGING_PRESET,
    SPOT_BURN_PRESET,
    TESCAN_SCAN_DIRECTIONS,
)
from fibsem.imaging.spot import SpotBurnProgress, SpotBurnSettings, SpotBurnStatus
from fibsem.milling.progress import MillingProgress
from fibsem.services.milling import Milling, bind_milling, progress_update
from fibsem.services.spot_burn import SpotBurn, bind_spot_burn
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    MillingState,
    Point,
)

if TYPE_CHECKING:
    import threading

    from fibsem.drivers.tescan.microscope import TescanMicroscope


class TescanMilling(Milling):
    """Tescan milling, on a DrawBeam layer."""

    parent: TescanMicroscope

    # The preset sets the current and voltage; the rest go into the DrawBeam layer.
    setting_names = (
        "milling_channel",
        "hfw",
        "preset",
        "spot_size",
        "rate",
        "dwell_time",
        "spacing",
        "patterning_mode",
    )
    scan_directions = TESCAN_SCAN_DIRECTIONS

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "milling_channel":
            # the DrawBeam layer is on the ion column only
            return ParameterMetadata(choices=(BeamType.ION,))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # ion only: the milling preset, then a DrawBeam layer from the recipe
        if settings.milling_channel is not BeamType.ION:
            raise ValueError("Only FIB milling is currently supported.")
        microscope = self.parent
        microscope._prepare_beam(settings.milling_channel)
        self._clear()
        microscope.milling_channel = settings.milling_channel
        # QUERY: do we need to set this here as it is also set in IEtching?
        microscope._write_beam("preset", settings.preset, BeamType.ION)
        layer_settings = tescan.IEtching(
            syncWriteField=False,
            writeFieldSize=settings.hfw,
            beamCurrent=microscope.get_beam_current(settings.milling_channel),
            spotSize=settings.spot_size,
            rate=settings.rate,
            dwellTime=settings.dwell_time,
            parallel=bool(settings.patterning_mode == "Parallel"),
            preset=settings.preset,
            spacing=settings.spacing,
        )
        # TODO: change the layer name to milling stage name
        with microscope._connection_lock:
            microscope.layer = microscope.connection.DrawBeam.Layer(
                "Layer1", layer_settings
            )

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        if isinstance(pattern, FibsemRectangleSettings):
            self._draw_rectangle(pattern)
        elif isinstance(pattern, FibsemLineSettings):
            self._draw_line(pattern)
        elif isinstance(pattern, FibsemCircleSettings):
            self._draw_circle(pattern)
        elif isinstance(pattern, FibsemBitmapSettings):
            pass  # DrawBeam has no bitmap patterns; drawn as nothing, as always
        elif isinstance(pattern, FibsemPolygonSettings):
            raise NotImplementedError("draw_polygon not implemented for Tescan API")

    def _draw_rectangle(self, pattern: FibsemRectangleSettings) -> None:
        microscope = self.parent
        if pattern.scan_direction in TESCAN_SCAN_DIRECTIONS:
            scanning_path = pattern.scan_direction
        else:
            scanning_path = "Flyback"
            logging.warning(
                f"Scan direction {pattern.scan_direction} not supported. Using Flyback instead."
            )
        microscope.connection.DrawBeam.ScanningPath = scanning_path

        if pattern.cross_section is CrossSectionPattern.CleaningCrossSection:
            add_pattern = microscope.layer.addRectanglePolish
        else:
            add_pattern = microscope.layer.addRectangleFilled
        add_pattern(
            CenterX=pattern.centre_x,
            CenterY=pattern.centre_y,
            Depth=pattern.depth,
            DepthUnit="m",
            Width=pattern.width,
            Height=pattern.height,
            Angle=pattern.rotation * constants.RADIANS_TO_DEGREES,
            ScanningPath=scanning_path,
        )

    def _draw_line(self, pattern: FibsemLineSettings) -> None:
        self.parent.layer.addLine(
            BeginX=pattern.start_x,
            BeginY=pattern.start_y,
            EndX=pattern.end_x,
            EndY=pattern.end_y,
            Depth=pattern.depth,
            DepthUnit="m",
        )

    def _draw_circle(self, pattern: FibsemCircleSettings) -> None:
        self.parent.layer.addAnnulusFilled(
            CenterX=pattern.centre_x,
            CenterY=pattern.centre_y,
            RadiusA=pattern.radius,
            RadiusB=0,
            Depth=pattern.depth,
            DepthUnit="m",
        )

    def _draw_dot(self, centre: Point, exposure_time: float) -> None:
        """A timed dot: DepthUnit.Second makes the depth an exposure time, which is
        a spot burn. Only `TescanSpotBurn` draws one."""
        self.parent.layer.addDot(
            CenterX=centre.x,
            CenterY=centre.y,
            Depth=exposure_time,
            DepthUnit=tescan.DepthUnit.Second,
        )

    def read_state(self) -> MillingState:
        microscope = self.parent
        with microscope._connection_lock:
            status = microscope.connection.DrawBeam.GetStatus()[0]
        return tescan.DrawBeamStatusToPatterningState[status]

    def _start(self) -> None:
        with self.parent._connection_lock:
            self.parent.connection.DrawBeam.Start()

    def _stop(self) -> None:
        # From another thread too: on a second connection, not the locked one.
        # TODO: improve thread safety to stop from another thread
        thread_connection = None
        try:
            thread_connection = tescan.Automation(
                self.parent.system.info.ip_address, port=self.parent._port
            )
            if (
                thread_connection.DrawBeam.GetStatus()[0]
                == tescan.DBStatus.ProjectLoadedExpositionInProgress
            ):
                logging.info("Milling is in progress, stopping now...")
                thread_connection.DrawBeam.Stop()
        except Exception as e:
            logging.error(f"Error in stop_milling: {e}")
        finally:
            del thread_connection

    def _pause(self) -> None:
        with self.parent._connection_lock:
            self.parent.connection.DrawBeam.Pause()

    def _resume(self) -> None:
        with self.parent._connection_lock:
            self.parent.connection.DrawBeam.Resume()

    def _estimate(self) -> float:
        # DrawBeam estimates only a loaded layer, and a layer cannot be loaded twice
        try:
            with self.parent._connection_lock:
                return self.parent.connection.DrawBeam.EstimateTime()
        except Exception as e:
            logging.error(f"Error in estimating milling time: {e}")
            return 0

    def _before_run(self) -> None:
        microscope = self.parent
        microscope._prepare_beam(microscope.milling_channel)
        with microscope._connection_lock:
            microscope.connection.DrawBeam.LoadLayer(microscope.layer)
            microscope.connection.Progress.Show(
                Title="DrawBeam Milling (OpenFIBSEM)",
                Text="Layer 1 in progress",
                HideButton=True,
                Marquee=False,
                ProgressMin=0,
                ProgressMax=100,
            )

    def _poll(self) -> MillingProgress:
        microscope = self.parent
        with microscope._connection_lock:
            status, total, elapsed = microscope.connection.DrawBeam.GetStatus()
        state = tescan.DrawBeamStatusToPatterningState[status]
        if total <= 0:  # DrawBeam reports no total until the exposition is under way
            return progress_update(state=state)
        if state is MillingState.RUNNING:
            with microscope._connection_lock:
                microscope.connection.Progress.SetPercents(
                    min(100, elapsed / total * 100)
                )
        return progress_update(
            state=state, total=total, remaining=max(0.0, total - elapsed)
        )

    def _after_run(self) -> None:
        microscope = self.parent
        with microscope._connection_lock:
            microscope.connection.Progress.Hide()

    def _clear(self) -> None:
        # DrawBeam.UnloadLayer raises when no layer is loaded (and while an exposition
        # is still active); swallowed, so clearing needs no record of a layer.
        try:
            with self.parent._connection_lock:
                self.parent.connection.DrawBeam.UnloadLayer()
        except Exception as e:
            logging.debug(f"Error unloading layer: {e}")

    def _save(self, beam: Beam) -> None:
        super()._save(beam)
        # The column reports no preset until one is activated this session; put it
        # back on the imaging preset then, as milling always has.
        if self._saved is not None and "preset" in beam.parameters:
            if self._saved.get("preset") is None and beam.preset.settable:
                self._saved["preset"] = DEFAULT_IMAGING_PRESET

    def _write_back(self, beam: Beam, name: str, value: Any) -> None:
        # Activating a preset is the fragile step; a failure leaves the column where
        # it is, with a warning, rather than failing the end of milling.
        try:
            super()._write_back(beam, name, value)
        except Exception as e:
            logging.warning(f"Error restoring {name} {value!r} after milling: {e}")


def bind_tescan_milling(microscope: TescanMicroscope) -> Optional[TescanMilling]:
    """Build ``milling`` for a connected Tescan microscope whose beams are built."""
    return bind_milling(TescanMilling, microscope)


class TescanSpotBurn(SpotBurn):
    """Tescan spot burns, through the milling service.

    The FIB has no blanker and can't park the beam (FIB.Scan lacks SetBlanker,
    GetBlanker and SetBeamPosition), so the point-by-point burn is impossible. The
    points go onto one DrawBeam layer instead, a timed dot each, set up at the spot
    burn preset and run by ``microscope.milling``, which also puts the preset and the
    field of view back afterwards. The requested burn current is not used: the preset
    sets it.
    """

    parent: TescanMicroscope

    # Milling saves and restores the beam (the preset and the field of view).
    saved_conditions = ()
    setting_names = ("coordinates", "exposure_time")

    @classmethod
    def _can_burn(cls, ion: Optional[Beam]) -> bool:
        return ion is not None

    @command
    def run(
        self,
        settings: SpotBurnSettings,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
    ) -> SpotBurnStatus:
        """Burn the points of ``settings`` and return how the run ended.

        As `SpotBurn.run`, but an exposure time that isn't positive is refused before
        anything is touched (DrawBeam would draw dots that burn nothing), and the burn
        current may be None: the preset sets it.
        """
        exposure_time = float(settings.exposure_time)
        if exposure_time <= 0:
            raise ValueError(f"exposure_time must be positive, got {exposure_time}.")
        if settings.milling_current is None:
            settings = dataclasses.replace(settings, milling_current=0.0)
        else:
            logging.info(
                f"Spot burn milling_current is ignored on TESCAN; using preset "
                f"{SPOT_BURN_PRESET!r}. (requested: {settings.milling_current})"
            )
        return super().run(settings, stop_event=stop_event)

    def _setup(self, ion: Beam, current: float) -> None:
        """A layer at the spot burn preset, as wide as the field of view; the
        remaining DrawBeam fields come from the milling defaults."""
        recipe = FibsemMillingSettings(
            milling_channel=BeamType.ION,
            hfw=ion.hfw.get_value(),
            preset=SPOT_BURN_PRESET,
            patterning_mode="Serial",
        )
        self.parent.milling.setup(recipe, name="SpotBurn")

    def _burn(
        self,
        ion: Beam,
        points: Sequence[Point],
        exposure_time: float,
        current: float,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    ) -> SpotBurnStatus:
        """Draw a dot per point, then run the layer, reporting the point DrawBeam is
        on from its elapsed time (it reports no per-dot progress, and burns the dots
        in order at the exposure time each)."""
        microscope = self.parent
        milling = microscope.milling
        hfw = ion.hfw.get_value()
        resolution = ion.resolution.get_value()
        for point in points:
            centre = microscope._spot_burn_point_to_metres(
                point, hfw=hfw, resolution=resolution
            )
            logging.info(
                f"spot burn point: {point} -> ({centre.x:.3e}, {centre.y:.3e}) m, "
                f"exposure time: {exposure_time}s"
            )
            milling._draw_dot(centre, exposure_time)

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
        # blocked while it runs. (Blocked, not disconnected: reconnecting a psygnal
        # signal's ``emit`` calls it once, with nonsense, to read its signature.)
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
        """Put the preset and the field of view back, as milling found them."""
        try:
            self.parent.milling.restore()
        except Exception:
            logging.exception("Failed to restore the ion beam after the spot burn")


def _events(stop_event: Any) -> tuple:
    return () if stop_event is None else (stop_event,)


def bind_tescan_spot_burn(microscope: TescanMicroscope) -> Optional[TescanSpotBurn]:
    """Build ``spot_burn`` for a Tescan microscope whose milling is built; without
    milling (no ion beam) there is none."""
    if microscope.milling is None:
        return None
    return bind_spot_burn(TescanSpotBurn, microscope)
