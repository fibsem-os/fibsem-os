"""Tescan's services.

`TescanStageMovement` is the shared stage movement with Tescan's own coincidence move
from the SEM view, and no safe-rotation sequence before an absolute move.

`TescanMilling` mills on DrawBeam: a layer made from the milling preset on the ion
column, and a second connection to stop it from another thread.

A ``run`` loads the layer (DrawBeam estimates only a loaded one), shows a progress bar
in Essence for its length, and reads the state, the elapsed time and the total from
``DrawBeam.GetStatus`` in one call each look.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

import fibsem.constants as constants
from fibsem.devices.beam import Beam
from fibsem.devices.core import ParameterMetadata
from fibsem.drivers.tescan import microscope as tescan
from fibsem.drivers.tescan.microscope import (
    DEFAULT_IMAGING_PRESET,
    TESCAN_SCAN_DIRECTIONS,
)
from fibsem.milling.progress import MillingProgress
from fibsem.services.milling import Milling, bind_milling, progress_update
from fibsem.services.stage_movement import StageMovement
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
    FibsemStagePosition,
    MillingState,
)

if TYPE_CHECKING:
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


class TescanStageMovement(StageMovement):
    """The shared moves, with two of Tescan's own.

    An absolute move goes straight to the position: Tescan has no tilt-flat and
    compucentric-rotation sequence before it (it never had). The coincidence move from
    the SEM view is Tescan's, verified on hardware, in place of the shared stable move
    then FIB-view vertical move.
    """

    def _safe_rotation(self, stage_position: FibsemStagePosition) -> None:
        """Nothing before the move itself."""

    def _vertical_move_from_sem(
        self, dx: float, dy: float, relaxation: float = 1.0
    ) -> FibsemStagePosition:
        """Correct the coincidence point from the SEM view.

        Tescan's own move, kept in place of the shared stable move then FIB-vertical
        move because it is the one verified on hardware. ``relaxation`` is not applied,
        as it never was here.

        The mirror of the FIB-view move: the stage slides along the FIB
        line of sight, which is invisible in the FIB image, until the clicked
        feature is centred in the SEM. A feature already positioned in the FIB
        view (e.g. just milled, or just corrected with vertical_move) therefore
        lands on both beam axes at once -- at the coincidence point. The math
        is :func:`coincident_from_sem_stage_movement_tescan_from_geometry`; the sample
        plane, and with it the shuttle pre-tilt, cancels out of this move, so only the
        stage tilt and the column tilts appear. See
        https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414
        for the derivation and figures.

        Verified on hardware 2026-08-26 (acceptance test: Alt-double-click a
        feature in the SEM view lands it centred in the SEM image with no
        movement in the FIB image). Small focus shifts in both views are
        inherent to the move.

        Args:
            dx (float): distance along the image x-axis (SEM view), in metres.
            dy (float): distance along the image y-axis (SEM view), in metres.
        """
        # adjust for scan rotation (radians, codebase convention)
        scan_rotation = self._beam_value(BeamType.ELECTRON, "scan_rotation")
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        y_move, z_move = tescan.coincident_from_sem_stage_movement_tescan_from_geometry(
            geometry=self.parent.hardware_geometry(),
            stage_position=self._position(),
            dy=dy,
        )

        # The move in Tescan's frame: x and y run opposite the image, z as computed
        # (+z is down). The stage device takes fibsem's frame, at the current tilt.
        stage_position = FibsemStagePosition(x=-dx, y=-y_move, z=z_move, r=0, t=0)
        stage_position = self.stage.native_delta(stage_position, self._position().t)
        logging.info(f"coincident move from SEM: {stage_position}")
        self._move_stage(stage_position, relative=True)

        logging.debug(
            {
                "msg": "move_coincident_from_sem",
                "dx": dx,
                "dy": dy,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )
        return self._position()
