"""Tescan's services.

`TescanStageMovement` is the shared stage movement with Tescan's own coincidence move
from the SEM view, and no safe-rotation sequence before an absolute move.

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

import dataclasses
import logging
import re
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import numpy as np

import fibsem.constants as constants
from fibsem import conversions
from fibsem.cancellation import AnyStopEvent, OperationCancelledError
from fibsem.devices.beam import Beam
from fibsem.devices.core import ParameterMetadata, command
from fibsem.drivers.tescan import sdk
from fibsem.drivers.tescan.sdk import (
    DEFAULT_IMAGING_PRESET,
    SPOT_BURN_PRESET,
    TESCAN_SCAN_DIRECTIONS,
)
from fibsem.drivers.tescan.settings import TescanMillingSettings
from fibsem.imaging.spot import SpotBurnProgress, SpotBurnSettings, SpotBurnStatus
from fibsem.milling.progress import MillingProgress
from fibsem.services.milling import Milling, bind_milling, progress_update
from fibsem.services.spot_burn import SpotBurn, bind_spot_burn
from fibsem.services.stage_movement import StageMovement
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemCircleSettings,
    FibsemHardwareGeometry,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    MillingState,
    Point,
)

if TYPE_CHECKING:
    import threading

    from fibsem.drivers.tescan.connection import TescanConnection
    from fibsem.drivers.tescan.microscope import TescanMicroscope
    from fibsem.milling.base import FibsemMillingStage


# A current token inside a free-form TESCAN preset name, e.g. "30 keV; 100 pA" or
# "30 keV; 2nA; my cool preset". Only prefixed units (pA/nA/uA/µA): a bare "A" in an
# arbitrary name (e.g. "slot 2A") is far more likely noise than a beam current.
_PRESET_CURRENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([pnuµ])A(?![a-zA-Z])")
_SI_CURRENT_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6}


def parse_current_from_preset(preset: Optional[str]) -> Optional[float]:
    """Parse the beam current (in A) out of a TESCAN preset name, or None.

    Preset names are free-form on the instrument, but conventionally embed the
    beam conditions ("30 keV; 100 pA"). The first current-looking token wins.
    """
    if not preset:
        return None
    match = _PRESET_CURRENT_RE.search(preset)
    if match is None:
        return None
    return float(match.group(1)) * _SI_CURRENT_PREFIX[match.group(2)]


def estimate_preset_milling_time(
    stage: FibsemMillingStage, own: TescanMillingSettings
) -> Optional[float]:
    """Dose-model estimate t = volume / (rate × current) for a preset-driven stage,
    milled with *own*, the stage's Tescan settings.

    The same inputs DrawBeam computes the real exposure from: the stage's own
    (per-material) etch rate and the current embedded in the preset name. The
    shared sputter-rate table is a silicon calibration keyed on a current field
    TESCAN milling ignores. Returns None (the caller falls back to the table)
    when the preset carries no parseable current or the rate is unusable.
    """
    pattern_time = getattr(stage.pattern, "time", 0)
    if pattern_time:
        return pattern_time

    current = parse_current_from_preset(own.preset)
    rate = own.rate  # m³/A/s
    if current is None or current <= 0 or not rate or rate <= 0:
        return None

    volume = stage.pattern.volume  # m³
    if (
        hasattr(stage.pattern, "cross_section")
        and stage.pattern.cross_section is CrossSectionPattern.CleaningCrossSection
    ):
        volume *= 0.66  # ccs is approx 2/3 of the volume of a rectangle
    return volume / (rate * current)


def coincident_from_sem_stage_movement_tescan_from_geometry(
    geometry: FibsemHardwareGeometry,
    stage_position: FibsemStagePosition,
    dy: float,
) -> Tuple[float, float]:
    """Tescan coincidence correction from the SEM view — the single source of the math.

    The SEM-view mirror of ``vertical_move``: slide the sample along the FIB line
    of sight, which is invisible in the FIB image, until the offset seen in the
    SEM closes. A feature already positioned in the FIB view therefore lands on
    both beam axes at once — at the coincidence point. Both gestures move the
    sample by dy/sin(angle between the two axes), symmetric in the two
    directions.

    The FIB axis is chamber-fixed, so the sample plane — and with it the shuttle
    pre-tilt — cancels out of this move entirely; only the stage tilt (the
    y-axis rides the tilt module) and the column tilts appear:

        y_move = dy * sin(fib) / (cos(tilt) * sin(fib - sem))
        z_move = dy * cos(fib - tilt) / (cos(tilt) * sin(fib - sem))

    which for a vertical SEM column (sem = 0) reduces to y = dy/cos(tilt),
    z = dy*(tan(tilt) + cot(fib)). See
    https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414
    for the derivation, the figures, and how the signs are pinned by the two
    hardware-verified moves.

    Args:
        geometry: fixed instrument geometry (only the column tilts are used —
            the pre-tilt and rotation references drop out of this move).
        stage_position: stage pose (tilt t, radians).
        dy: offset along the image y-axis in the SEM view, in metres.

    Returns:
        (y_move, z_move) in metres — before the stage-axis inversion the caller
        applies (``y_stage = -y_move``, z unchanged), in Tescan's own frame.
    """
    stage_tilt = stage_position.t if stage_position.t is not None else 0.0
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    fib_column_tilt = np.deg2rad(geometry.fib_column_tilt)

    # distance along the FIB axis that closes a SEM-view offset of dy: one
    # beam axis, seen from the other, is foreshortened by sin(angle between them)
    axis_move = dy / np.sin(fib_column_tilt - sem_column_tilt)

    y_move = axis_move * np.sin(fib_column_tilt) / np.cos(stage_tilt)
    z_move = axis_move * np.cos(fib_column_tilt - stage_tilt) / np.cos(stage_tilt)
    return float(y_move), float(z_move)


class TescanMilling(Milling):
    """Tescan milling, on a DrawBeam layer, over the microscope's connection."""

    parent: TescanMicroscope

    # The preset sets the current and voltage; it and the rest of Tescan's own
    # settings go into the DrawBeam layer.
    setting_names = ("milling_channel", "hfw", "patterning_mode")
    settings_class = TescanMillingSettings
    scan_directions = TESCAN_SCAN_DIRECTIONS

    def __init__(self, connection: TescanConnection, **kwargs: Any):
        super().__init__(**kwargs)
        self.connection = connection
        # The DrawBeam layer the last setup made, which the patterns are drawn on.
        self.layer: Any = None

    @classmethod
    def estimate_stage(cls, stage: FibsemMillingStage) -> float:
        # Tescan milling is preset-driven: the dose model (stage rate x preset
        # current), not the shared table keyed on the unused milling_current. A
        # preset the model can't read falls back to the table.
        estimate = estimate_preset_milling_time(stage, cls.own_settings(stage.milling))
        if estimate is None:
            return super().estimate_stage(stage)
        return estimate

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "milling_channel":
            # the DrawBeam layer is on the ion column only
            return ParameterMetadata(choices=(BeamType.ION,))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # ion only: the milling preset, then a DrawBeam layer from the recipe
        if settings.milling_channel is not BeamType.ION:
            raise ValueError("Only FIB milling is currently supported.")
        own = self.own_settings(settings)
        microscope = self.parent
        connection = self.connection
        connection.prepare_beam(settings.milling_channel)
        self._clear()
        microscope.milling_channel = settings.milling_channel
        self.ion.preset.write_through(own.preset)
        layer_settings = sdk.IEtching(
            syncWriteField=False,
            writeFieldSize=settings.hfw,
            beamCurrent=microscope.get_beam_current(settings.milling_channel),
            spotSize=own.spot_size,
            rate=own.rate,
            dwellTime=own.dwell_time,
            parallel=bool(settings.patterning_mode == "Parallel"),
            preset=own.preset,
            spacing=own.spacing,
        )
        with connection.lock:
            self.layer = connection.sdk.DrawBeam.Layer("Layer1", layer_settings)

    def _draw_rectangle(self, pattern: FibsemRectangleSettings) -> None:
        if pattern.scan_direction in TESCAN_SCAN_DIRECTIONS:
            scanning_path = pattern.scan_direction
        else:
            scanning_path = "Flyback"
            logging.warning(
                f"Scan direction {pattern.scan_direction} not supported. Using Flyback instead."
            )
        self.connection.sdk.DrawBeam.ScanningPath = scanning_path

        if pattern.cross_section is CrossSectionPattern.CleaningCrossSection:
            add_pattern = self.layer.addRectanglePolish
        else:
            add_pattern = self.layer.addRectangleFilled
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
        self.layer.addLine(
            BeginX=pattern.start_x,
            BeginY=pattern.start_y,
            EndX=pattern.end_x,
            EndY=pattern.end_y,
            Depth=pattern.depth,
            DepthUnit="m",
        )

    def _draw_circle(self, pattern: FibsemCircleSettings) -> None:
        self.layer.addAnnulusFilled(
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
        self.layer.addDot(
            CenterX=centre.x,
            CenterY=centre.y,
            Depth=exposure_time,
            DepthUnit=sdk.DepthUnit.Second,
        )

    def read_state(self) -> MillingState:
        with self.connection.lock:
            status = self.connection.sdk.DrawBeam.GetStatus()[0]
        return sdk.DrawBeamStatusToPatterningState[status]

    def _start(self) -> None:
        with self.connection.lock:
            self.connection.sdk.DrawBeam.Start()

    def _stop(self) -> None:
        # From another thread too: on a second connection, not the locked one.
        thread_connection = None
        try:
            thread_connection = sdk.Automation(
                self.connection.ip_address, port=self.connection.port
            )
            if (
                thread_connection.DrawBeam.GetStatus()[0]
                == sdk.DBStatus.ProjectLoadedExpositionInProgress
            ):
                logging.info("Milling is in progress, stopping now...")
                thread_connection.DrawBeam.Stop()
        except Exception as e:
            logging.error(f"Error in stop_milling: {e}")
        finally:
            del thread_connection

    def _pause(self) -> None:
        with self.connection.lock:
            self.connection.sdk.DrawBeam.Pause()

    def _resume(self) -> None:
        with self.connection.lock:
            self.connection.sdk.DrawBeam.Resume()

    def _estimate(self) -> float:
        # DrawBeam estimates only a loaded layer, and a layer cannot be loaded twice
        try:
            with self.connection.lock:
                return self.connection.sdk.DrawBeam.EstimateTime()
        except Exception as e:
            logging.error(f"Error in estimating milling time: {e}")
            return 0

    def _before_run(self) -> None:
        connection = self.connection
        connection.prepare_beam(self.parent.milling_channel)
        with connection.lock:
            connection.sdk.DrawBeam.LoadLayer(self.layer)
            connection.sdk.Progress.Show(
                Title="DrawBeam Milling (OpenFIBSEM)",
                Text="Layer 1 in progress",
                HideButton=True,
                Marquee=False,
                ProgressMin=0,
                ProgressMax=100,
            )

    def _poll(self) -> MillingProgress:
        connection = self.connection
        with connection.lock:
            status, total, elapsed = connection.sdk.DrawBeam.GetStatus()
        state = sdk.DrawBeamStatusToPatterningState[status]
        if total <= 0:  # DrawBeam reports no total until the exposition is under way
            return progress_update(state=state)
        if state is MillingState.RUNNING:
            with connection.lock:
                connection.sdk.Progress.SetPercents(min(100, elapsed / total * 100))
        return progress_update(
            state=state, total=total, remaining=max(0.0, total - elapsed)
        )

    def _after_run(self) -> None:
        with self.connection.lock:
            self.connection.sdk.Progress.Hide()

    def _clear(self) -> None:
        # DrawBeam.UnloadLayer raises when no layer is loaded (and while an exposition
        # is still active); swallowed, so clearing needs no record of a layer.
        try:
            with self.connection.lock:
                self.connection.sdk.DrawBeam.UnloadLayer()
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
    return bind_milling(TescanMilling, microscope, connection=microscope._connection)


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

        Verified on hardware (acceptance test: Alt-double-click a
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

        y_move, z_move = coincident_from_sem_stage_movement_tescan_from_geometry(
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

    @staticmethod
    def _spot_burn_point_to_metres(
        point: Point, hfw: float, resolution: Tuple[int, int]
    ) -> Point:
        """Convert a normalised (0-1, top-left origin) image coordinate to DrawBeam coordinates.

        DrawBeam objects are positioned in metres from the image centre with +y up, the same
        convention the draw_* pattern methods already use.

        Args:
            point: normalised image coordinate, (0, 0) top-left to (1, 1) bottom-right.
            hfw: horizontal field width in metres.
            resolution: image resolution as (width, height) in pixels.

        Returns:
            Point: position in metres relative to the image centre.
        """
        width, height = resolution
        pixelsize = hfw / width
        pixel_coordinate = Point(x=point.x * width, y=point.y * height)
        return conversions.image_to_microscope_image_coordinates2(
            coord=pixel_coordinate,
            image_shape=(height, width),
            pixelsize=pixelsize,
            subpixel_precision=True,  # spot coordinates are fractional, don't round to a pixel
        )

    def _setup(self, ion: Beam, current: float) -> None:
        """A layer at the spot burn preset, as wide as the field of view; the
        remaining DrawBeam fields come from the milling defaults."""
        milling = self.parent.milling
        recipe = FibsemMillingSettings(
            milling_channel=BeamType.ION,
            hfw=ion.hfw.get_value(),
            patterning_mode="Serial",
        )
        recipe.driver[milling.own_name()] = TescanMillingSettings(
            preset=SPOT_BURN_PRESET
        )
        milling.setup(recipe, name="SpotBurn")

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
            centre = self._spot_burn_point_to_metres(
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

        try:
            # A spot burn records itself, not a mill, and its mill's progress is
            # the spot burn's: milling progress does not see it.
            with milling.part_of(report=report):
                milling.run(
                    stop_event=AnyStopEvent(self._stop_requested, *_events(stop_event))
                )
        except OperationCancelledError:
            logging.info("Spot burn cancelled.")
            return SpotBurnStatus.CANCELLED
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
