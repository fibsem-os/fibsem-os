from __future__ import annotations

import dataclasses
import datetime
import logging
import threading
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from psygnal import Signal

import fibsem.constants as constants
from fibsem import manufacturers
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.imaging.spot import SpotBurnProgress, SpotBurnStatus
from fibsem.imaging.tiling.progress import TiledProgress
from fibsem.milling.progress import MillingProgress
from fibsem.structures import (
    DEFAULT_STAGE_DEVICES,
    DEVICE_AXES,
    BeamSettings,
    BeamSystemSettings,
    BeamType,
    CameraImageTransform,
    DeviceImagingState,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemGasInjectionSettings,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemImageMetadata,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    MillingState,
    Point,
    RangeLimit,
    StageDeviceSettings,
    SystemSettings,
)
from fibsem.transformations import (
    get_stage_tilt_from_milling_angle,
    inverse_view_corrected_dy,
)

if TYPE_CHECKING:
    from fibsem.imaging.spot import SpotBurnSettings


# The device the orientation transform is defined at. `_get_compucentric_rotation_position`
# is a half turn about a chamber-fixed centre, and the beams are the only place it has
# ever been applied on any instrument -- so `get_target_position` carries a position
# into this device's frame before re-posing it, and back out afterwards.
ROTATION_FRAME_DEVICE = "FIBSEM"


class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""

    # THREADING CONTRACT: these are psygnal Signals — subscribers run synchronously
    # on whatever thread emits (workflow/movement/acquisition workers, not the GUI
    # thread). Any handler that touches Qt or a canvas MUST marshal, e.g. with
    # @superqt.ensure_main_thread; a bare .connect() of a GUI handler is a
    # crash-on-hardware bug that won't reproduce on a dev machine.
    milling_progress_signal = Signal(MillingProgress)
    tiled_acquisition_signal = Signal(TiledProgress)
    spot_burn_progress_signal = Signal(SpotBurnProgress)
    _last_imaging_settings: ImageSettings
    system: SystemSettings
    _patterns: List
    stage_is_compustage: bool = False
    milling_channel: BeamType = BeamType.ION

    # The views a coincidence correction can be measured in -- the beam_type
    # values vertical_move accepts. The FIB view is universal; the SEM view
    # needs a backend that knows how to slide along the FIB line of sight.
    vertical_move_views: Tuple[BeamType, ...] = (BeamType.ION,)

    # live acquisition
    sem_acquisition_signal = Signal(FibsemImage)
    fib_acquisition_signal = Signal(FibsemImage)
    _stop_acquisition_event = threading.Event()
    _acquisition_thread: threading.Thread = None
    _threading_lock: threading.RLock = threading.RLock()

    # fluorescence
    fm: Optional[FluorescenceMicroscope]

    stage_position_changed = Signal(FibsemStagePosition)
    _stage_position: FibsemStagePosition = None

    @abstractmethod
    def connect_to_microscope(
        self, ip_address: str, port: int, reset_beam_shift: bool = True
    ) -> None:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def acquire_image(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:
        pass

    @abstractmethod
    def last_image(self, beam_type: BeamType) -> FibsemImage:
        pass

    @property
    def is_acquiring(self) -> bool:
        """Check if the microscope is currently acquiring an image."""
        return self._acquisition_thread and self._acquisition_thread.is_alive()

    def start_acquisition(self, beam_type: BeamType) -> None:
        """Start the image acquisition process.
        Args:
            beam_type: The beam type to start acquisition for.
        """
        if self.is_acquiring:
            logging.warning("Acquisition thread is already running.")
            return

        # reset stop event if needed
        self._stop_acquisition_event.clear()

        # start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_worker, args=(beam_type,), daemon=True
        )
        self._acquisition_thread.start()

    def stop_acquisition(self) -> None:
        """Stop the image acquisition process."""
        if self._stop_acquisition_event and not self._stop_acquisition_event.is_set():
            self._stop_acquisition_event.set()
            if self._acquisition_thread:
                self._acquisition_thread.join(timeout=2)
            # Disconnect signal handler
            # self.sem_acquisition_signal.disconnect()
            # self.fib_acquisition_signal.disconnect()

    def _acquisition_worker(self, beam_type: BeamType) -> None:
        """The worker function for the acquisition thread.
        Acquires images from the microscope, and emits them as signals."""
        pass

    @abstractmethod
    def acquire_chamber_image(self) -> FibsemImage:
        pass

    @abstractmethod
    def autocontrast(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        pass

    @abstractmethod
    def auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        pass

    def reset_beam_shifts(self) -> None:
        """Set the beam shift to zero for the electron and ion beams."""
        self.set_beam_shift(Point(0, 0), BeamType.ELECTRON)
        self.set_beam_shift(Point(0, 0), BeamType.ION)

    @abstractmethod
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> Point:
        pass

    def get_stage_position(self) -> FibsemStagePosition:
        """
        Get the current stage position.

        This method retrieves the current stage position from the microscope and returns it as
        a FibsemStagePosition object. FibsemStage Position is in the RAW coordinate frame

        Returns:
            FibsemStagePosition: The current stage position.
        """

        stage_position = self.get("stage_position")

        if not isinstance(stage_position, FibsemStagePosition):
            raise TypeError(f"Expected FibsemStagePosition, got {type(stage_position)}")

        logging.debug({"msg": "get_stage_position", "pos": stage_position.to_dict()})

        if self._stage_position is None:
            self._stage_position = deepcopy(stage_position)

        if not self._stage_position.is_close2(stage_position, tol=1e-6):
            self._stage_position = deepcopy(stage_position)
            self.stage_position_changed.emit(self._stage_position)

        return deepcopy(stage_position)

    def _read_stage_capabilities(self) -> None:
        """Ask the instrument what its stage can do, and record it.

        Only `rotation` today, and it is not a preference: it decides where the FIB
        orientation is, because `rotation_180` is derived from it (FIB-834). A
        configuration file used to state it, which meant a compustage could be
        described as a rotating stage by a typo and nothing would disagree --
        `sim-arctis-configuration.yaml` was, for as long as nothing read the flag.

        The axes are the honest form of the question: a compustage has no `r` limit
        because it has no rotation axis. It agrees with `stage_is_compustage` on every
        backend today, and deliberately does not ask that instead -- "has a rotation
        axis" is the property the derivation needs, and a future stage that lacks one
        without being a compustage would answer correctly here for free.
        """
        self.system.stage.rotation = "r" in self._get_axis_limits()

    def _create_sample_stage(self) -> None:
        """Create the sample stage and holder based on the system settings."""
        from fibsem.microscopes._stage import _create_sample_stage

        # Before the stage object, not after. Every backend wraps this call in a
        # try/except that logs and carries on, so a failure below leaves the
        # capability at its field default -- `True`, which on a compustage is the
        # wrong answer and a silent one. Reading it first means the only way to miss
        # it is `_get_axis_limits` itself raising, and on a compustage that is a
        # lookup of a module constant, which cannot.
        self._read_stage_capabilities()

        self._stage = _create_sample_stage(self)

    def _create_grid_loader(self) -> Optional["SampleGridLoader"]:
        """The grid loader for a compustage system, or None when it has no autoloader.

        Backends wrap their own hardware; this default is the in-memory model.
        """
        from fibsem.microscopes._stage import SampleGridLoader

        return SampleGridLoader(parent=self)

    def _get_axis_limits(self) -> Dict[str, RangeLimit]:
        """Get the stage axis limits from the microscope."""

        axes_limits: Dict[str, RangeLimit] = {}
        axes_limits["x"] = RangeLimit(min=-100.0e-3, max=100.0e-3)
        axes_limits["y"] = RangeLimit(min=-100.0e-3, max=100.0e-3)
        axes_limits["z"] = RangeLimit(min=0.0e-3, max=50.0e-3)
        axes_limits["r"] = RangeLimit(min=-360.0, max=360.0)
        axes_limits["t"] = RangeLimit(min=-10.0, max=90.0)

        return axes_limits

    @abstractmethod
    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        pass

    @abstractmethod
    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        pass

    @abstractmethod
    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemStagePosition:
        pass

    @abstractmethod
    def vertical_move(
        self, dy: float, dx: float = 0, beam_type: BeamType = BeamType.ION
    ) -> FibsemStagePosition:
        """Restore coincidence from an offset measured in one of the beam views.

        Args:
            dy: offset along the image y-axis, in the view named by beam_type.
            dx: offset along the image x-axis, in the same view.
            beam_type: the view the offset was measured in. ION (the default, and
                the historical behaviour) corrects a feature already centred in the
                SEM; ELECTRON corrects one already centred in the FIB.

        Raises:
            NotImplementedError: if this backend cannot correct from that view.
                Ask supports_vertical_move first rather than catching this.
        """
        pass

    def supports_vertical_move(self, beam_type: BeamType = BeamType.ION) -> bool:
        """Whether coincidence can be restored from the given view on this system."""
        return beam_type in self.vertical_move_views

    def _check_vertical_move_supported(self, beam_type: BeamType) -> None:
        """Guard for a vertical_move implementation -- one message, one source of truth."""
        if not self.supports_vertical_move(beam_type):
            raise NotImplementedError(
                f"{type(self).__name__} cannot restore coincidence from the "
                f"{beam_type.name} view."
            )

    @abstractmethod
    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        pass

    def move_flat_to_beam(self, beam_type: BeamType, _safe: bool = True) -> None:
        """Move the sample surface flat to the electron or ion beam.

        .. deprecated::
            Use :meth:`move_to_orientation` instead. This method will be removed in the next version.
            ``move_flat_to_beam(BeamType.ELECTRON)`` → ``move_to_orientation("SEM")``
            ``move_flat_to_beam(BeamType.ION)`` → ``move_to_orientation("FIB")``
        """
        warnings.warn(
            "move_flat_to_beam is deprecated and will be removed in the next version. "
            "Use move_to_orientation('SEM') or move_to_orientation('FIB') instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        stage_settings = self.system.stage
        shuttle_pre_tilt = stage_settings.shuttle_pre_tilt

        if beam_type is BeamType.ELECTRON:
            rotation = np.deg2rad(stage_settings.rotation_reference)
            tilt = np.deg2rad(shuttle_pre_tilt)

        if beam_type is BeamType.ION:
            rotation = np.deg2rad(stage_settings.rotation_180)
            tilt = np.deg2rad(self.system.ion.column_tilt - shuttle_pre_tilt)

        # new style
        # omap = {BeamType.ELECTRON: "SEM", BeamType.ION: "FIB"}
        # pos = self.get_orientation(omap[beam_type])
        # rotation, tilt = pos.r, pos.t

        # compustage is tilted by 180 degrees for flat to beam, because we image the backside fo the grid,
        # therefore, we need to offset the tilt by 180 degrees
        if self.stage_is_compustage and beam_type is BeamType.ION:
            rotation = 0
            tilt = -np.pi + tilt

        # updated safe rotation move
        logging.info(f"moving flat to {beam_type.name}")
        stage_position = FibsemStagePosition(
            r=rotation, t=tilt, coordinate_system="Raw"
        )

        logging.debug(
            {
                "msg": "move_flat_to_beam",
                "stage_position": stage_position.to_dict(),
                "beam_type": beam_type.name,
            }
        )

        if _safe:
            self.safe_absolute_stage_movement(stage_position)
        else:
            self.move_stage_absolute(stage_position)

    def _axis_restrictions_apply(self) -> bool:
        """Whether the microscope refuses z and rotation, so an absolute move drops them.

        Two halves, and they are not the same rule.

        The **orientation** half is a compustage flipped to face its objective. It
        stays as it was: `get_stage_orientation` can never return "FM" on an offset
        mount -- the FM is a device there and `orientations["FM"]` is a copy of the FIB
        entry -- so that half is naturally confined to the mounting it was written for.

        The **objective** half is gated on `stage_is_compustage` **temporarily**, and
        that gate belongs to FIB-640 to remove. It has only ever run on a compustage,
        because `self.fm` is None everywhere else, and the axes it drops are not
        equivalent across stage types: `stage_position_to_autoscript` returns a
        `CompustagePosition(x, y, z, a)` with no `r` field at all, so dropping `r`
        there has never done anything, while on an offset mount it would drop a real
        rotation axis. Opening the connection gate is what makes that reachable, so
        the gate goes on first.

        Removing it silently would be the worse failure of the two. Without the gate
        an offset move half-succeeds -- lands at x and y, no z, no rotation -- where
        with it the full move is sent and the *microscope* refuses if it objects,
        which is an error an operator can see and report. FIB-640 argues for exactly
        that preference, and is also where the axis pair gets settled: it measured
        z and t, not z and r.
        """
        if self.get_stage_orientation() == "FM":
            return True

        return (
            self.stage_is_compustage
            and self.fm is not None
            and self.fm.objective.state == "Inserted"
        )

    def _fluorescence_is_configured(self) -> bool:
        """Whether this site has said its instrument has a fluorescence microscope.

        The **flag decides**; the driver's own probe only confirms afterwards, and
        that order matters. There is no `is_installed` for the FM in AutoScript --
        every other subsystem has one -- so the only capability test available is to
        select it and see whether the microscope throws. Running that on every system
        would be autodetection, and an Aquilos or Helios with an iFLM fitted would
        find half-built offset support appearing in its UI on upgrade: a fluorescence
        tab that builds, a button that traverses 48 mm, pose derivation that is only
        partly right. Probing also touches the shared imaging channel on machines that
        have never had an FM.

        **A compustage keeps its answer.** `stage_is_compustage` is read from the
        hardware (`compustage.is_installed`), not from configuration, and no shipped
        Arctis configuration carries the flag -- `tfs-arctis-configuration.yaml` has
        no `fm:` block at all. Replacing the old check rather than widening it would
        take the FM away from every Arctis site on upgrade. So the compustage stays
        exactly as it was, and an offset mount must opt in.
        """
        return self.system.fm.enabled or self.stage_is_compustage

    def _refuse_rotation_at_the_fluorescence_microscope(
        self, stage_position: FibsemStagePosition
    ) -> None:
        """Refuse a stage rotation while the stage is parked at the FM.

        The objective is inserted over the sample there, and the rotation is
        compucentric about a centre back at the beams -- some 48.8 mm away -- so a half
        turn swings the sample most of the width of the chamber, under the objective.

        The route to another pose at the FM is the one `get_target_position` computes:
        traverse back to the beams, re-pose there, traverse out again. This refuses the
        shortcut. It is a refusal rather than a silent correction because a caller
        asking for the shortcut has a wrong idea of where the stage is going, and
        quietly sending it somewhere else would leave that idea intact.

        **Rotation only.** A tilt pivots about an axis through the sample instead of
        swinging it, and where the objective does restrict tilt the microscope refuses
        it itself -- FIB-640 measured z and t. This does not duplicate that.

        Not a restriction on FM-MILLING. That pose is a half turn from the FM's own
        orientation (measured: FM sits at r=180, MILLING at r=0), so it was never
        reachable by rotating in place -- it is reached the way everything else at the
        FM is, via the beams.

        Dormant until the connection gate opens: `microscope.fm` is `None` on every
        non-compustage system today, so nothing can park at the FM to begin with.
        """
        # A compustage reaches the FM by flipping, and its devices are the same place,
        # so "parked at the FM" is not a state it can be in -- and it has no rotation
        # axis to be compucentric about either.
        if self.stage_is_compustage or self.fm is None:
            return

        if stage_position.r is None:
            return

        current_position = self.get_stage_position()
        if self.get_current_device(current_position) != "FM":
            return

        from fibsem import movement

        # The same 5 degrees `get_stage_orientation` classifies within, so a caller
        # asking for the pose the stage is already in does not trip this on the slop a
        # real stage always carries.
        if movement.rotation_angle_is_smaller(
            stage_position.r, current_position.r, atol=5
        ):
            return

        raise ValueError(
            "Cannot rotate the stage while it is at the fluorescence microscope: the "
            "rotation is compucentric about a centre at the beams, so it would swing "
            "the sample across the chamber under the objective. Move to the beams "
            "first (move_to_device('FIBSEM')), re-pose there, and travel back -- "
            "or ask move_to_device for the pose and let it order the legs."
        )

    def move_to_orientation(self, orientation: str) -> FibsemStagePosition:
        """Move the stage to the given named orientation (e.g. 'SEM', 'FIB', 'MILLING').
        Args:
            orientation: The name of the orientation to move to.
        Returns:
            FibsemStagePosition: The new stage position after moving to the orientation.
        """
        stage_position = self.get_orientation(orientation)
        self.safe_absolute_stage_movement(stage_position)
        return self._stage.position

    @abstractmethod
    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        pass

    def get_manipulator_state(self) -> bool:
        """Get the manipulator state (Inserted = True, Retracted = False)"""
        # TODO: convert to enum
        return self.get("manipulator_state")

    def get_manipulator_position(self) -> FibsemManipulatorPosition:
        """Get the manipulator position."""
        return self.get("manipulator_position")

    @abstractmethod
    def insert_manipulator(self, name: str) -> None:
        pass

    @abstractmethod
    def retract_manipulator(self):
        pass

    @abstractmethod
    def move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
        pass

    @abstractmethod
    def move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
        pass

    @abstractmethod
    def move_manipulator_corrected(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> None:
        pass

    @abstractmethod
    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> None:
        pass

    @abstractmethod
    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        pass

    @abstractmethod
    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        pass

    @abstractmethod
    def run_milling(
        self, milling_current: float, milling_voltage: float, asynch: bool = False
    ) -> None:
        pass

    @abstractmethod
    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        pass

    def finish_milling2(self):
        pass

    @abstractmethod
    def clear_patterns(self) -> None:
        pass

    @abstractmethod
    def stop_milling(self) -> None:
        return

    @abstractmethod
    def start_milling(self) -> None:
        pass

    @abstractmethod
    def pause_milling(self) -> None:
        return

    @abstractmethod
    def resume_milling(self) -> None:
        return

    @abstractmethod
    def get_milling_state(self) -> MillingState:
        pass

    @abstractmethod
    def estimate_milling_time(self) -> float:
        pass

    def draw_patterns(self, patterns: List[FibsemPatternSettings]) -> None:
        """Draw milling patterns on the microscope from the list of settings
        Args:
            patterns (List[FibsemPatternSettings]): List of milling patterns
        """
        for pattern in patterns:
            self.draw_pattern(pattern)

    def draw_pattern(self, pattern: FibsemPatternSettings) -> None:
        """Draw a milling pattern from settings

        Args:
            pattern_settings (FibsemPatternSettings): pattern settings
        """
        if not isinstance(pattern, FibsemPatternSettings):
            raise TypeError(f"Expected FibsemPatternSettings, got {type(pattern)}")

        if isinstance(pattern, FibsemRectangleSettings):
            self.draw_rectangle(pattern)

        elif isinstance(pattern, FibsemLineSettings):
            self.draw_line(pattern)

        elif isinstance(pattern, FibsemCircleSettings):
            self.draw_circle(pattern)

        elif isinstance(pattern, FibsemBitmapSettings):
            self.draw_bitmap_pattern(pattern)

        elif isinstance(pattern, FibsemPolygonSettings):
            self.draw_polygon(pattern)

    @abstractmethod
    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        pass

    @abstractmethod
    def draw_line(self, pattern_settings: FibsemLineSettings):
        pass

    @abstractmethod
    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        pass

    @abstractmethod
    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings) -> None:
        pass

    @abstractmethod
    def draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        pass

    @abstractmethod
    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        pass

    @abstractmethod
    def setup_sputter(self, *args, **kwargs):
        pass

    @abstractmethod
    def draw_sputter_pattern(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def run_sputter(self, *args, **kwargs):
        pass

    @abstractmethod
    def finish_sputter(self):
        pass

    def run_sputter_coater(self, time_seconds: int) -> None:
        raise NotImplementedError("Sputter coater not implemented for this microscope.")

    @abstractmethod
    def get_available_values(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> List[Union[str, float, int]]:
        pass

    def get_available_values_cached(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> List[Union[str, float, int]]:
        """Get available values with caching to avoid repeated microscope queries.

        Args:
            key: The parameter key to get available values for.
            beam_type: The beam type (optional).

        Returns:
            List of available values for the given key.
        """
        if not hasattr(self, "_available_values_cache"):
            logging.info("Initializing available values cache.")
            self._available_values_cache: Dict[str, List[Union[str, float, int]]] = {}

        cache_key = f"{key}_{beam_type.name if beam_type else 'None'}"
        if cache_key not in self._available_values_cache:
            logging.info(
                f"Caching available values for key: {key}, beam_type: {beam_type}"
            )
            self._available_values_cache[cache_key] = self.get_available_values(
                key, beam_type
            )
        return self._available_values_cache[cache_key]

    def clear_available_values_cache(
        self, key: Optional[str] = None, beam_type: Optional[BeamType] = None
    ) -> None:
        """Clear the available values cache.

        Args:
            key: If provided, only clear cache for this key. Otherwise clear all.
            beam_type: The beam type (used with key to clear specific entry).
        """
        if not hasattr(self, "_available_values_cache"):
            return

        if key is None:
            self._available_values_cache.clear()
        else:
            cache_key = f"{key}_{beam_type.name if beam_type else 'None'}"
            self._available_values_cache.pop(cache_key, None)

    # TODO: use a decorator instead?
    def get(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> Union[float, int, bool, str, list, tuple, Point]:
        """Get wrapper for logging."""
        value = self._get(key, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug(
            {"msg": "get", "key": key, "beam_type": beam_name, "value": value}
        )
        return value

    def set(
        self,
        key: str,
        value: Union[str, float, int, tuple, list, Point],
        beam_type: Optional[BeamType] = None,
    ) -> None:
        """Set wrapper for logging"""
        self._set(key, value, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug(
            {"msg": "set", "key": key, "beam_type": beam_name, "value": value}
        )

    @abstractmethod
    def _get(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> Union[float, int, bool, str, list]:
        pass

    @abstractmethod
    def _set(
        self,
        key: str,
        value: Union[str, float, int, list, tuple, Point],
        beam_type: Optional[BeamType] = None,
    ) -> None:
        pass

    # TODO: i dont think this is needed, you set the beam settings and detector settings separately
    # you can't set image settings, only when acquiring an image
    def get_imaging_settings(self, beam_type: BeamType) -> ImageSettings:
        """Get the current imaging settings for the specified beam type."""
        # TODO: finish this with the other imaging settings... @patrick
        logging.debug(f"Getting {beam_type.name} imaging settings...")
        image_settings = ImageSettings(
            beam_type=beam_type,
            resolution=self.get_resolution(beam_type),
            dwell_time=self.get_dwell_time(beam_type),
            hfw=self.get_field_of_view(beam_type),
            path=self._last_imaging_settings.path,
            filename=self._last_imaging_settings.filename,
        )
        logging.debug(
            {
                "msg": "get_imaging_settings",
                "image_settings": image_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )
        return image_settings

    def set_imaging_settings(self, image_settings: ImageSettings) -> None:
        """Set the imaging settings for the specified beam type."""
        logging.debug(f"Setting {image_settings.beam_type.name} imaging settings...")
        self.set_resolution(image_settings.resolution, image_settings.beam_type)
        self.set_dwell_time(image_settings.dwell_time, image_settings.beam_type)
        self.set_field_of_view(image_settings.hfw, image_settings.beam_type)
        # self.set("frame_integration", image_settings.frame_integration, image_settings.beam_type)
        # self.set("line_integration", image_settings.line_integration, image_settings.beam_type)
        # self.set("scan_interlacing", image_settings.scan_interlacing, image_settings.beam_type)
        # self.set("drift_correction", image_settings.drift_correction, image_settings.beam_type)

        # TODO: implement the rest of these settings... @patrick
        logging.debug(
            {
                "msg": "set_imaging_settings",
                "image_settings": image_settings.to_dict(),
                "beam_type": image_settings.beam_type.name,
            }
        )

        return

    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:
        """Get the current beam settings for the specified beam type."""

        logging.debug(f"Getting {beam_type.name} beam settings...")
        beam_settings = BeamSettings(
            beam_type=beam_type,
            working_distance=self.get_working_distance(beam_type),
            beam_current=self.get_beam_current(beam_type),
            voltage=self.get_beam_voltage(beam_type),
            hfw=self.get_field_of_view(beam_type),
            resolution=self.get_resolution(beam_type),
            dwell_time=self.get_dwell_time(beam_type),
            stigmation=self.get_stigmation(beam_type),
            shift=self.get_beam_shift(beam_type),
            scan_rotation=self.get_scan_rotation(beam_type),
            preset=self.get("preset", beam_type),
        )
        logging.debug(
            {
                "msg": "get_beam_settings",
                "beam_settings": beam_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )

        return beam_settings

    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        """Set the beam settings for the specified beam type"""
        logging.debug(f"Setting {beam_settings.beam_type.name} beam settings...")
        self.set_working_distance(
            beam_settings.working_distance, beam_settings.beam_type
        )
        self.set_beam_current(beam_settings.beam_current, beam_settings.beam_type)
        self.set_beam_voltage(beam_settings.voltage, beam_settings.beam_type)
        self.set_field_of_view(beam_settings.hfw, beam_settings.beam_type)
        self.set_resolution(beam_settings.resolution, beam_settings.beam_type)
        self.set_dwell_time(beam_settings.dwell_time, beam_settings.beam_type)
        self.set_stigmation(beam_settings.stigmation, beam_settings.beam_type)
        self.set_beam_shift(beam_settings.shift, beam_settings.beam_type)
        self.set_scan_rotation(beam_settings.scan_rotation, beam_settings.beam_type)
        self.set("preset", beam_settings.preset, beam_settings.beam_type)

        logging.debug(
            {
                "msg": "set_beam_settings",
                "beam_settings": beam_settings.to_dict(),
                "beam_type": beam_settings.beam_type.name,
            }
        )
        return

    def get_beam_system_settings(self, beam_type: BeamType) -> BeamSystemSettings:
        """Get the current beam system settings for the specified beam type."""
        logging.debug(f"Getting {beam_type.name} beam system settings...")
        beam_system_settings = BeamSystemSettings(
            beam_type=beam_type,
            enabled=self.get("beam_enabled", beam_type),
            beam=self.get_beam_settings(beam_type),
            detector=self.get_detector_settings(beam_type),
            eucentric_height=self.get("eucentric_height", beam_type),
            column_tilt=self.get("column_tilt", beam_type),
            plasma=self.get("plasma", beam_type),
            plasma_gas=self.get("plasma_gas", beam_type),
        )

        logging.debug(
            {
                "msg": "get_beam_system_settings",
                "settings": beam_system_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )
        return beam_system_settings

    def set_beam_system_settings(self, settings: BeamSystemSettings) -> None:
        """Set the beam system settings for the specified beam type."""
        beam_type = settings.beam_type
        logging.debug(f"Setting {settings.beam_type.name} beam system settings...")
        self.set("beam_enabled", settings.enabled, beam_type)
        self.set_beam_settings(settings.beam)
        self.set_detector_settings(settings.detector, beam_type)
        self.set("eucentric_height", settings.eucentric_height, beam_type)
        self.set("column_tilt", settings.column_tilt, beam_type)

        if beam_type is BeamType.ION:
            self.set("plasma_gas", settings.plasma_gas, beam_type)
            self.set("plasma", settings.plasma, beam_type)

        logging.debug(
            {
                "msg": "set_beam_system_settings",
                "settings": settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )

        return

    def get_detector_settings(
        self, beam_type: BeamType = BeamType.ELECTRON
    ) -> FibsemDetectorSettings:
        """Get the current detector settings for the specified beam type."""
        logging.debug(f"Getting {beam_type.name} detector settings...")
        detector_settings = FibsemDetectorSettings(
            type=self.get_detector_type(beam_type),
            mode=self.get_detector_mode(beam_type),
            brightness=self.get_detector_brightness(beam_type),
            contrast=self.get_detector_contrast(beam_type),
        )
        logging.debug(
            {
                "msg": "get_detector_settings",
                "detector_settings": detector_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )
        return detector_settings

    def set_detector_settings(
        self,
        detector_settings: FibsemDetectorSettings,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> None:
        """Set the detector settings for the specified beam type"""
        logging.debug(f"Setting {beam_type.name} detector settings...")
        self.set_detector_type(detector_settings.type, beam_type)
        self.set_detector_mode(detector_settings.mode, beam_type)
        self.set_detector_brightness(detector_settings.brightness, beam_type)
        self.set_detector_contrast(detector_settings.contrast, beam_type)
        logging.debug(
            {
                "msg": "set_detector_settings",
                "detector_settings": detector_settings.to_dict(),
                "beam_type": beam_type.name,
            }
        )

        return

    def get_microscope_state(
        self, beam_type: Optional[BeamType] = None
    ) -> MicroscopeState:
        """Get the current microscope state."""

        # default values
        electron_beam, electron_detector = None, None
        ion_beam, ion_detector = None, None
        stage_position = None
        get_electron_state = beam_type in [BeamType.ELECTRON, None]
        get_ion_state = beam_type in [BeamType.ION, None]

        # get the state of the electron beam
        if self.is_available("electron_beam") and get_electron_state:
            electron_beam = self.get_beam_settings(beam_type=BeamType.ELECTRON)
            electron_detector = self.get_detector_settings(beam_type=BeamType.ELECTRON)

        # get the state of the ion beam
        if self.is_available("ion_beam") and get_ion_state:
            ion_beam = self.get_beam_settings(beam_type=BeamType.ION)
            ion_detector = self.get_detector_settings(beam_type=BeamType.ION)

        # get the state of the stage
        if self.is_available("stage"):
            stage_position = self.get_stage_position()

        current_microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            stage_position=stage_position,  # get absolute stage coordinates (RAW)
            electron_beam=electron_beam,  # electron beam state
            ion_beam=ion_beam,  # ion beam state
            electron_detector=electron_detector,  # electron beam detector state
            ion_detector=ion_detector,  # ion beam detector state
        )

        logging.debug(
            {"msg": "get_microscope_state", "state": current_microscope_state.to_dict()}
        )

        return deepcopy(current_microscope_state)

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        """Reset the microscope state to the provided state."""

        if self.is_available("electron_beam"):
            if microscope_state.electron_beam is not None:
                self.set_beam_settings(microscope_state.electron_beam)
            if microscope_state.electron_detector is not None:
                self.set_detector_settings(
                    microscope_state.electron_detector, BeamType.ELECTRON
                )
        if self.is_available("ion_beam"):
            if microscope_state.ion_beam is not None:
                self.set_beam_settings(microscope_state.ion_beam)
            if microscope_state.ion_detector is not None:
                self.set_detector_settings(microscope_state.ion_detector, BeamType.ION)
        if self.is_available("stage") and microscope_state.stage_position is not None:
            self.safe_absolute_stage_movement(microscope_state.stage_position)
        if self.fm is not None and microscope_state.objective_position is not None:
            self.fm.objective.move_absolute(microscope_state.objective_position)

        logging.debug(
            {"msg": "set_microscope_state", "state": microscope_state.to_dict()}
        )

        return

    def set_milling_settings(self, mill_settings: FibsemMillingSettings) -> None:
        self.set(
            "active_view", mill_settings.milling_channel, mill_settings.milling_channel
        )
        self.set(
            "active_device",
            mill_settings.milling_channel,
            mill_settings.milling_channel,
        )
        self.set(
            "default_patterning_beam_type",
            mill_settings.milling_channel,
            mill_settings.milling_channel,
        )
        self.set(
            "application_file",
            mill_settings.application_file,
            mill_settings.milling_channel,
        )
        self.set(
            "patterning_mode",
            mill_settings.patterning_mode,
            mill_settings.milling_channel,
        )
        self.set("hfw", mill_settings.hfw, mill_settings.milling_channel)
        self.set(
            "current", mill_settings.milling_current, mill_settings.milling_channel
        )
        self.set(
            "voltage", mill_settings.milling_voltage, mill_settings.milling_channel
        )

    def is_available(self, system: str) -> bool:

        if system == "electron_beam":
            return self.system.electron.enabled
        elif system == "ion_beam":
            return self.system.ion.enabled
        elif system == "ion_plasma":
            return self.system.ion.plasma
        elif system == "stage":
            return self.system.stage.enabled
        elif system == "stage_rotation":
            return self.system.stage.rotation
        elif system == "manipulator":
            return self.system.manipulator.enabled
        elif system == "manipulator_rotation":
            return self.system.manipulator.rotation
        elif system == "manipulator_tilt":
            return self.system.manipulator.tilt
        elif system == "gis":
            return self.system.gis.enabled
        elif system == "gis_multichem":
            return self.system.gis.multichem
        elif system == "gis_sputter_coater":
            return self.system.gis.sputter_coater
        else:
            return False

    def set_available(self, system: str, value: bool) -> None:

        if system == "electron_beam":
            self.system.electron.enabled = value
        elif system == "ion_beam":
            self.system.ion.enabled = value
        elif system == "ion_plasma":
            self.system.ion.plasma = value
        elif system == "stage":
            self.system.stage.enabled = value
        elif system == "manipulator":
            self.system.manipulator.enabled = value
        elif system == "manipulator_rotation":
            self.system.manipulator.rotation = value
        elif system == "manipulator_tilt":
            self.system.manipulator.tilt = value
        elif system == "gis":
            self.system.gis.enabled = value
        elif system == "gis_multichem":
            self.system.gis.multichem = value
        elif system == "gis_sputter_coater":
            self.system.gis.sputter_coater = value

    def apply_configuration(
        self, system_settings: Optional[SystemSettings] = None
    ) -> None:
        """Apply the system settings to the microscope."""

        logging.info("Applying Microscope Configuration...")

        if system_settings is None:
            system_settings = self.system
            logging.info("Using current system settings.")

        # apply the system settings
        if self.is_available("electron_beam"):
            self.set_beam_system_settings(system_settings.electron)
        if self.is_available("ion_beam"):
            self.set_beam_system_settings(system_settings.ion)

        if self.is_available("stage"):
            self.system.stage = system_settings.stage
            # The line above replaces the whole record, including the capability the
            # instrument told us about at connect. `system_settings` came from a file,
            # and files no longer state `rotation` -- so on a compustage this would
            # silently restore the field default of `True`, move the FIB orientation
            # half a turn away, and hand the compucentric correction a rotation the
            # stage cannot make. Re-read rather than preserve: the instrument is the
            # authority, and it has not changed because someone pressed Apply.
            self._read_stage_capabilities()

        if self.is_available("manipulator"):
            self.system.manipulator = system_settings.manipulator

        if self.is_available("gis"):
            self.system.gis = system_settings.gis

        # dont update info -> read only
        logging.info("Microscope configuration applied.")
        logging.debug(
            {"msg": "apply_configuration", "system_settings": system_settings.to_dict()}
        )

    @abstractmethod
    def check_available_values(
        self, key: str, values, beam_type: Optional[BeamType] = None
    ) -> bool:
        pass

    def home(self) -> bool:
        """Home the stage."""
        self.set("stage_home", True)
        return self.get("stage_homed")

    def link_stage(self) -> bool:
        """Link the stage to the working distance"""
        self.set("stage_link", True)
        return self.get("stage_linked")

    def pump(self) -> str:
        """ "Pump the chamber."""
        self.set("pump_chamber", True)
        return self.get("chamber_state")

    def vent(self) -> str:
        """Vent the chamber."""
        self.set("vent_chamber", True)
        return self.get("chamber_state")

    def turn_on(self, beam_type: BeamType) -> bool:
        """Turn on the specified beam type."""
        self.set("on", True, beam_type)
        return self.get("on", beam_type)

    def turn_off(self, beam_type: BeamType) -> bool:
        "Turn off the specified beam type."
        self.set("on", False, beam_type)
        return self.get("on", beam_type)

    def is_on(self, beam_type: BeamType) -> bool:
        """Check if the specified beam type is on."""
        return self.get("on", beam_type)

    def blank(self, beam_type: BeamType) -> bool:
        """Blank the specified beam type."""
        self.set("blanked", True, beam_type)
        return self.get("blanked", beam_type)

    def unblank(self, beam_type: BeamType) -> bool:
        """Unblank the specified beam type."""
        self.set("blanked", False, beam_type)
        return self.get("blanked", beam_type)

    def is_blanked(self, beam_type: BeamType) -> bool:
        """Check if the specified beam type is blanked."""
        return self.get("blanked", beam_type)

    def get_available_beams(self) -> List[BeamType]:
        """Get the available beams for the microscope."""
        available_beams = []
        if self.is_available("electron_beam"):
            available_beams.append(BeamType.ELECTRON)
        if self.is_available("ion_beam"):
            available_beams.append(BeamType.ION)
        return available_beams

    def set_spot_scanning_mode(self, point: Point, beam_type: BeamType) -> None:
        """Set the spot scanning mode for the specified beam type."""
        self.set("spot_mode", point, beam_type)
        return

    def set_reduced_area_scanning_mode(
        self, reduced_area: FibsemRectangle, beam_type: BeamType
    ) -> None:
        """Set the reduced area scanning mode for the specified beam type."""
        self.set("reduced_area", reduced_area, beam_type)
        return

    def set_full_frame_scanning_mode(self, beam_type: BeamType) -> None:
        """Set the full frame scanning mode for the specified beam type."""
        self.set("full_frame", None, beam_type)
        return

    def run_spot_burn(
        self,
        settings: SpotBurnSettings,
        beam_type: BeamType = BeamType.ION,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Burn each coordinate in *settings* with the beam for the configured exposure time.

        Default implementation: blank -> park the beam on the point (spot scanning mode)
        -> unblank, at ``settings.milling_current``, restoring full-frame scanning and
        the imaging current afterwards. Backends whose scan API cannot park the beam
        (e.g. TESCAN, whose FIB has no blanker) override this with a native
        implementation.

        Progress is reported via ``spot_burn_progress_signal`` (a dict), which the
        status bar and the spot burn widget subscribe to.

        Args:
            settings: What to burn — coordinates (0-1 image coordinates), exposure time
                per point in seconds, and the milling current to burn at.
            beam_type: The type of beam to use. (Default: BeamType.ION)
            stop_event: Threading event to signal cancellation. (Default: None)
        """
        # - QUERY: do we need to set the full frame scanning mode each time, or only at the end?
        SLEEP_TIME = 1

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
        self.spot_burn_progress_signal.emit(
            SpotBurnProgress(
                status=SpotBurnStatus.BURNING,
                current_point=0,
                total_points=len(coordinates),
                remaining_time=exposure_time,
                total_remaining_time=total_remaining_time,
                total_estimated_time=total_estimated_time,
            )
        )

        cancelled = False

        # Read before the `try`, so the `finally` below can always restore it. A
        # failure here means there is nothing to restore anyway.
        imaging_current = self.get_beam_current(beam_type=beam_type)

        try:
            self.set_beam_current(current=milling_current, beam_type=beam_type)

            for i, pt in enumerate(coordinates, 1):
                if stop_event is not None and stop_event.is_set():
                    logging.info(
                        f"Spot burn cancelled before point {i}/{len(coordinates)}."
                    )
                    cancelled = True
                    break

                logging.info(
                    f"burning spot {i}: {pt}, exposure time: {exposure_time}, milling current: {milling_current}"
                )

                self.blank(beam_type=beam_type)
                self.set_spot_scanning_mode(point=pt, beam_type=beam_type)
                self.unblank(beam_type=beam_type)

                # countdown for the exposure time, emit progress signal
                remaining_time = exposure_time
                while remaining_time > 0:
                    if stop_event is not None and stop_event.is_set():
                        self.blank(beam_type=beam_type)
                        logging.info(
                            f"Spot burn cancelled during point {i}/{len(coordinates)}."
                        )
                        cancelled = True
                        break
                    time.sleep(SLEEP_TIME)
                    remaining_time -= SLEEP_TIME
                    total_remaining_time -= SLEEP_TIME
                    self.spot_burn_progress_signal.emit(
                        SpotBurnProgress(
                            status=SpotBurnStatus.BURNING,
                            current_point=i,
                            total_points=len(coordinates),
                            remaining_time=remaining_time,
                            total_remaining_time=total_remaining_time,
                            total_estimated_time=total_estimated_time,
                        )
                    )

                if cancelled:
                    # The inner `break` only leaves this point's countdown. The outer
                    # loop's own stop_event check would catch it on the next iteration
                    # anyway, so this is not a fix -- it just stops the run here rather
                    # than one log line later, now that the outcome is recorded.
                    break

            # A cancelled burn is not a completed one. Both used to emit `{"finished": True}`,
            # so cancelling rendered "Done" -- the defect the status enum exists to remove.
            self.spot_burn_progress_signal.emit(
                SpotBurnProgress(
                    status=SpotBurnStatus.CANCELLED
                    if cancelled
                    else SpotBurnStatus.FINISHED,
                    current_point=len(coordinates),
                    total_points=len(coordinates),
                )
            )
        except Exception as e:
            logging.error(f"Error in run_spot_burn: {e}")
            # The failure terminal belongs to the producer. It used to be emitted by
            # `FibsemSpotBurnWidget`, which only ever sees a burn it started itself --
            # so an unsupervised workflow burn that raised (`tasks/spot_burn.py` calls
            # this directly) reported nothing at all, and left the bar mid-run for the
            # rest of the session.
            self.spot_burn_progress_signal.emit(
                SpotBurnProgress(status=SpotBurnStatus.FAILED, error=str(e))
            )
            raise
        finally:
            # Restores the beam on the failing path too. The comment above this block
            # used to say "always restore" while sitting in the success path only, so a
            # burn that raised left the beam parked in spot scanning mode at the
            # milling current -- a hazard, not just untidy state.
            #
            # Each restore is guarded separately so that a failing restore cannot
            # replace the exception that actually ended the run: an error raised in a
            # `finally` discards the one in flight, and the original is the one worth
            # having. They are also independent -- neither should be skipped because
            # the other failed.
            try:
                self.set_full_frame_scanning_mode(beam_type=beam_type)
            except Exception:
                logging.exception(
                    "Failed to restore full-frame scanning after the spot burn"
                )
            try:
                self.set_beam_current(current=imaging_current, beam_type=beam_type)
            except Exception:
                logging.exception(
                    "Failed to restore the imaging current after the spot burn"
                )

    def get_beam_current(self, beam_type: BeamType) -> float:
        """Get the beam current for the specified beam type."""
        return self.get("current", beam_type)

    def set_beam_current(self, current: float, beam_type: BeamType) -> float:
        """Set the beam current for the specified beam type."""
        self.set("current", current, beam_type)
        return self.get("current", beam_type)

    def get_beam_voltage(self, beam_type: BeamType) -> float:
        """Get the beam voltage for the specified beam type."""
        return self.get("voltage", beam_type)

    def set_beam_voltage(self, voltage: float, beam_type: BeamType) -> float:
        """Set the beam voltage for the specified beam type."""
        self.set("voltage", voltage, beam_type)
        return self.get("voltage", beam_type)

    def set_resolution(
        self, resolution: Tuple[int, int], beam_type: BeamType
    ) -> List[int]:
        """Set the resolution for the specified beam type."""
        self.set("resolution", resolution, beam_type)
        return self.get("resolution", beam_type)

    def get_resolution(self, beam_type: BeamType) -> Tuple[int, int]:
        """Get the resolution for the specified beam type."""
        return self.get("resolution", beam_type)

    def get_field_of_view(self, beam_type: BeamType) -> float:
        """Get the field of view for the specified beam type."""
        return self.get("hfw", beam_type)

    def set_field_of_view(self, hfw: float, beam_type: BeamType) -> float:
        """Set the field of view for the specified beam type."""
        self.set("hfw", hfw, beam_type)
        return self.get("hfw", beam_type)

    def get_working_distance(self, beam_type: BeamType) -> float:
        """Get the working distance for the specified beam type."""
        return self.get("working_distance", beam_type)

    def set_working_distance(self, wd: float, beam_type: BeamType) -> float:
        """Set the working distance for the specified beam type."""
        self.set("working_distance", wd, beam_type)
        return self.get("working_distance", beam_type)

    def is_working_distance_settable(self, beam_type: BeamType) -> bool:
        """Whether set_working_distance actually reaches the hardware for this beam.

        The image-based autofocus sweep gates on this: on backends where the write is
        a best-effort no-op (TESCAN ION -- focus there is preset-driven, the SDK has
        no FIB working-distance control), the sweep would score images against a focus
        that never moved and report a working distance that was never applied (FIB-508).
        """
        return True

    def get_dwell_time(self, beam_type: BeamType) -> float:
        """Get the dwell time for the specified beam type."""
        return self.get("dwell_time", beam_type)

    def set_dwell_time(self, dwell_time: float, beam_type: BeamType) -> float:
        """Set the dwell time for the specified beam type."""
        self.set("dwell_time", dwell_time, beam_type)
        return self.get("dwell_time", beam_type)

    def get_stigmation(self, beam_type: BeamType) -> Point:
        """Get the stigmation for the specified beam type."""
        return self.get("stigmation", beam_type)

    def set_stigmation(self, stigmation: Point, beam_type: BeamType) -> Point:
        """Set the stigmation for the specified beam type."""
        self.set("stigmation", stigmation, beam_type)
        return self.get("stigmation", beam_type)

    def get_beam_shift(self, beam_type: BeamType) -> Point:
        """Get the beam shift for the specified beam type."""
        return self.get("shift", beam_type)

    def set_beam_shift(self, shift: Point, beam_type: BeamType) -> Point:
        """Set the beam shift for the specified beam type."""
        self.set("shift", shift, beam_type)
        return self.get("shift", beam_type)

    def get_scan_rotation(self, beam_type: BeamType) -> float:
        """Get the scan rotation for the specified beam type."""
        return self.get("scan_rotation", beam_type)

    def set_scan_rotation(self, rotation: float, beam_type: BeamType) -> float:
        """Set the scan rotation for the specified beam type."""
        self.set("scan_rotation", rotation, beam_type)
        return self.get("scan_rotation", beam_type)

    def get_detector_type(self, beam_type: BeamType) -> str:
        """Get the detector type for the specified beam type."""
        return self.get("detector_type", beam_type)

    def set_detector_type(self, detector_type: str, beam_type: BeamType) -> str:
        """Set the detector type for the specified beam type."""
        self.set("detector_type", detector_type, beam_type)
        return self.get("detector_type", beam_type)

    def get_detector_mode(self, beam_type: BeamType) -> str:
        """Get the detector mode for the specified beam type."""
        return self.get("detector_mode", beam_type)

    def set_detector_mode(self, mode: str, beam_type: BeamType) -> str:
        """Set the detector mode for the specified beam type."""
        self.set("detector_mode", mode, beam_type)
        return self.get("detector_mode", beam_type)

    def get_detector_contrast(self, beam_type: BeamType) -> float:
        """Get the detector contrast for the specified beam type."""
        return self.get("detector_contrast", beam_type)

    def set_detector_contrast(self, contrast: float, beam_type: BeamType) -> float:
        """Set the detector contrast for the specified beam type."""
        self.set("detector_contrast", contrast, beam_type)
        return self.get("detector_contrast", beam_type)

    def get_detector_brightness(self, beam_type: BeamType) -> float:
        """Get the detector brightness for the specified beam type."""
        return self.get("detector_brightness", beam_type)

    def set_detector_brightness(self, brightness: float, beam_type: BeamType) -> float:
        """Set the detector brightness for the specified beam type."""
        self.set("detector_brightness", brightness, beam_type)
        return self.get("detector_brightness", beam_type)

    def set_preset(self, preset: str, beam_type: BeamType) -> str:
        """Set the preset for the specified beam type."""
        self.set("preset", preset, beam_type)
        return self.get("preset", beam_type)

    def _get_compucentric_rotation_offset(self) -> FibsemStagePosition:
        return FibsemStagePosition(x=0, y=0)  # assume no offset to rotation centre

    def _get_compucentric_rotation_position(
        self, position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """Get the compucentric rotation position for the given stage position.
        Assumes 180deg rotation. TFS only"""

        # compustage does not support compucentric rotation
        if self.stage_is_compustage:
            return position

        # get the compucentric rotation offset
        offset = self._get_compucentric_rotation_offset()

        # convert the raw stage position to specimen coordinates
        specimen_position = deepcopy(position)
        specimen_position.x += offset.x
        specimen_position.y += offset.y

        # apply "compucentric" rotation offset (invert x,y)
        target_position = deepcopy(specimen_position)
        target_position.r += np.radians(180)
        target_position.x = -specimen_position.x
        target_position.y = -specimen_position.y

        # convert the target position to raw coordinates
        target_position.x -= offset.x
        target_position.y -= offset.y

        return target_position

    def _apply_device_translation(
        self, stage_position: FibsemStagePosition, source: str, target: str
    ) -> FibsemStagePosition:
        """`stage_position`, moved from one device to another. Mutates and returns it."""
        translation = self._device_translation(source, target)
        for axis in DEVICE_AXES:
            delta = getattr(translation, axis)
            if delta is None:
                continue
            value = getattr(stage_position, axis)
            if value is None:
                raise ValueError(
                    f"Cannot convert between devices {source} and {target}: the "
                    f"position has no {axis}, and the two differ along it."
                )
            setattr(stage_position, axis, value + delta)
        return stage_position

    def get_target_position(
        self,
        stage_position: FibsemStagePosition,
        target_orientation: Optional[str] = None,
        target_device: Optional[str] = None,
    ) -> FibsemStagePosition:
        """Convert a stage position across the orientation axis, the device axis, or both.

        Where the objective is offset the two are independent -- the device is a place
        the stage travels to, the orientation is the pose it is held in once there --
        so either may be asked for alone:

            (p, "FIB")                        re-pose, stay put
            (p, target_device="FM")           relocate, keep the pose
            (p, "MILLING", target_device="FM") both -- this is FM-MILLING, which is
                                              a pair rather than a fifth orientation

        **Asking for an orientation snaps the pose.** An orientation *is* a canonical
        r and t, so a position a degree or two off nominal is written to nominal
        rather than carried across. That is right for a conversion and wrong for a
        relocation, which is why the second form exists: the traverse must not discard
        a milling angle somebody dialled in on the way to the FM.

        Note `target_orientation="FM"` means something on a compustage, where the FM
        really is an orientation -- a flip to `t = -180`. On an offset mount it is
        refused: the FM is a device there, and `orientations["FM"]` is a copy of the
        FIB entry carrying no positional term.
        """

        currrent_orientation = self.get_stage_orientation(stage_position)
        logging.info(
            f"Getting target position for {target_orientation} from {currrent_orientation}"
        )

        if target_orientation is None and target_device is None:
            return stage_position

        if currrent_orientation == target_orientation and target_device is None:
            return stage_position

        if currrent_orientation == "NONE":
            raise ValueError("Unknown orientation. Cannot convert stage position.")

        # The FM is an orientation only where the objective is under the grid. On an
        # offset mount it is a *place*, reached by translating rather than re-posing,
        # and `orientations["FM"]` there is a copy of the FIB entry carrying no
        # positional term -- so converting into it would return a position under the
        # beam wearing the FM's rotation and tilt. Ask for it as a device instead:
        # `target_device="FM"`, with whichever orientation the sample should be in.
        if "FM" in (currrent_orientation, target_orientation) and (
            not self.stage_is_compustage
        ):
            raise ValueError("Cannot move to FM position on non-compustage systems.")

        # Which device the stage is at, read *before* either leg runs. The orientation
        # leg below can move x and y most of the way across the grid, so asking
        # afterwards would sometimes name a different device, or none.
        source_device = self.get_current_device(stage_position)
        if target_device is not None and source_device is None:
            raise ValueError(
                f"The stage is not at any configured device "
                f"({sorted(self.system.stage.devices)}), so there is nothing to "
                f"convert from. Position: {stage_position}."
            )

        stage_position = deepcopy(stage_position)
        orientation = (
            self.get_orientation(target_orientation)
            if target_orientation is not None
            else None
        )

        # The device legs **bracket** the orientation leg rather than following it,
        # so that the re-pose always happens at the beams. Written out, that is the
        # order an operator would use:
        #
        #   leaving the beams:  re-pose first, then traverse to the device
        #   returning:          traverse back to the beams first, then re-pose
        #
        # Both come out of one expression, `f(p) = rotate(p - source) + target`.
        # Leaving, `source` is zero and the rotation happens before the traverse;
        # returning, `target` is zero and the traverse happens before the rotation.
        #
        # **The stage must not be re-posed while it is parked at the FM.** The
        # objective is inserted over the sample there, and a half turn swings the
        # sample about a centre ~48.8 mm away. Bracketing is what keeps the rotation
        # out of that pose; the arrangement that re-poses wherever the stage happens
        # to be would compute a target that commands exactly that move.
        #
        # It is also the only arrangement that round-trips, which is how the wrong one
        # was caught rather than reasoned about: re-posing at the FM and translating
        # afterwards sends `(SEM, beams) -> (FIB, FM) -> (SEM, beams)` back to
        # -97.6 mm instead of 0. `rotate` is an involution, so `f` inverts by swapping
        # source and target -- which is exactly what the reverse call does.
        #
        # This computes a target; it does not order the moves. Nothing yet stops a
        # caller re-posing while at the FM -- see FIB-841, a prerequisite for opening
        # the connection gate.
        #
        # A compustage brackets with a zero translation, so it takes this path and
        # gets its old answer untouched.
        if source_device is not None:
            stage_position = self._apply_device_translation(
                stage_position, source_device, ROTATION_FRAME_DEVICE
            )

        # One rule, in place of a branch per ordered pair.
        #
        # Re-posing between orientations is always a rewrite of r and t. What decides
        # whether x and y move with it is whether the *rotation* changes: turning the
        # sample half way round swings it about the compucentric centre, which is
        # somewhere else entirely, so the coordinates have to be carried around with
        # it. A change of tilt alone pivots about an axis through the sample and
        # leaves x/y where they were.
        #
        # That is why SEM <-> MILLING needs no positional term -- both sit at
        # `rotation_reference` -- while anything crossing to or from FIB, which sits at
        # `rotation_180`, does.
        #
        # A compustage takes the same path and gets the same answer for free:
        # `_get_compucentric_rotation_position` returns its argument untouched there
        # (it has no rotation axis to be compucentric about), so no stage-type branch
        # is needed here to say so.
        #
        # Read from the *orientations*, not from `stage_position.r`, and the difference
        # matters. `get_stage_orientation` classifies within a 5 degree tolerance, so a
        # position that reads as SEM is usually a fraction off the canonical rotation --
        # a real stage never sits at exactly 0.000. Comparing the position's own r would
        # then call a 4 degree discrepancy a "rotation change" and apply the correction
        # below, which is not a small correction: `_get_compucentric_rotation_position`
        # computes `p -> -p - 2 * offset` and says so in its own docstring ("Assumes
        # 180deg rotation"). It is a half turn or nothing. Firing it for a few degrees
        # of slop would throw the sample to the far side of the grid.
        #
        # So the test below asks whether the rotation **is a half turn**, not whether it
        # changed at all. That is what the correction can express, and it makes the
        # assumption safe rather than merely documented: every rotation between named
        # orientations is a half turn today -- SEM and MILLING share
        # `rotation_reference`, FIB sits at `rotation_180` -- and an orientation ever
        # added at, say, 90 degrees gets **no** correction instead of the wrong one.
        # Wrong by the offset beats wrong by the whole grid.
        #
        # Measured with `angle_difference`, which is wrap-aware -- a stage rotates
        # continuously, so the same rotation is written many ways (270 and -90, 180 and
        # -180, 360 and 0), and plain modulo breaks either side of zero. It is also what
        # `get_stage_orientation` uses to decide which orientation a position is *at*,
        # and this rule is keyed on that classifier's answer, so the two share one
        # definition rather than agreeing by coincidence. Same 5 degree tolerance, for
        # the same reason.
        # Only when a pose was asked for. Relocating alone changes no rotation, so
        # there is nothing for the compucentric correction to correct and nothing to
        # write to r and t -- the bracket below collapses to a plain translation from
        # one device to the other, which is exactly what a traverse is.
        if orientation is not None:
            from fibsem import movement

            rotation = movement.angle_difference(
                self.get_orientation(currrent_orientation).r, orientation.r
            )
            rotation_is_half_turn = movement.rotation_angle_is_smaller(
                rotation, np.pi, atol=5
            )
            if rotation_is_half_turn:
                stage_position = self._get_compucentric_rotation_position(
                    stage_position
                )

            stage_position.r = orientation.r
            stage_position.t = orientation.t

        # ... and back out, to whichever device was asked for. `target_device` of None
        # means the one it started at, so an orientation-only conversion at the FM is
        # still re-posed about the right centre rather than about wherever it is
        # parked.
        if source_device is not None:
            stage_position = self._apply_device_translation(
                stage_position, ROTATION_FRAME_DEVICE, target_device or source_device
            )

        return stage_position

    def get_stage_orientation(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> str:
        """Get the current stage orientation based on the stage position (r,t).
        Args:
            stage_position (FibsemStagePosition, optional): stage position to use. If None, uses current stage position.
        Returns:
            str: current stage orientation ("SEM", "FIB", "MILLING", "NONE")
        """
        # TODO: update this to an enum

        # current stage position
        if stage_position is None:
            stage_position = self.get_stage_position()
        if stage_position.r is None or stage_position.t is None:
            raise ValueError(
                "Stage position must have both rotation (r) and tilt (t) defined."
            )
        stage_rotation = stage_position.r % (2 * np.pi)
        stage_tilt = stage_position.t

        from fibsem import movement
        # TODO: also check xyz ranges?

        sem = self.get_orientation("SEM")
        fib = self.get_orientation("FIB")
        milling = self.get_orientation("MILLING")
        # FM is an orientation only on a compustage -- see `_update_orientations`. On
        # an offset mount there is no FM pose to classify against, and there never
        # effectively was: the deleted copy was byte-identical to FIB, which matches
        # first, so no position ever classified as FM off a compustage.
        fm = self.orientations.get("FM")
        if sem is None or fib is None or milling is None:
            raise ValueError(
                "SEM, FIB or MILLING orientation not defined in the system."
            )
        if (
            sem.r is None
            or sem.t is None
            or fib.r is None
            or fib.t is None
            or milling.r is None
            or milling.t is None
        ):
            raise ValueError(
                "SEM, FIB or MILLING orientation must have both rotation (r) and tilt (t) defined."
            )

        is_sem_rotation = movement.rotation_angle_is_smaller(
            stage_rotation, sem.r, atol=5
        )  # query: do we need rotation_angle_is_smaller, since we % 2pi the rotation?
        is_fib_rotation = movement.rotation_angle_is_smaller(
            stage_rotation, fib.r, atol=5
        )
        is_fm_rotation = fm is not None and movement.rotation_angle_is_smaller(
            stage_rotation, fm.r, atol=5
        )

        is_sem_tilt = np.isclose(stage_tilt, sem.t, atol=0.1)
        is_fib_tilt = np.isclose(stage_tilt, fib.t, atol=0.1)

        is_milling_tilt = np.radians(-45) < stage_tilt and not is_sem_tilt
        is_fm_tilt = fm is not None and np.isclose(stage_tilt, fm.t, atol=0.1)

        if is_sem_rotation and is_sem_tilt:
            return "SEM"
        if is_sem_rotation and is_milling_tilt:
            return "MILLING"
        if is_fib_rotation and is_fib_tilt:
            return "FIB"
        if is_fm_rotation and is_fm_tilt:
            return "FM"

        return "NONE"

    def get_orientation(self, orientation: str) -> FibsemStagePosition:
        """Get the orientation (r,t) for the given orientation string."""

        # if orientations not initialised, update
        if not hasattr(self, "orientations"):
            self._update_orientations()

        if orientation not in self.orientations:
            raise ValueError(f"Orientation {orientation} not supported.")

        return self.orientations[orientation]

    def _update_orientations(self) -> None:
        """Update the stage orientations based on the current system settings."""

        stage_settings = self.system.stage
        shuttle_pre_tilt = stage_settings.shuttle_pre_tilt  # deg
        milling_angle = stage_settings.milling_angle  # deg

        # needs to be dynmaically updated as it can change.
        milling_stage_tilt = get_stage_tilt_from_milling_angle(
            self, np.radians(milling_angle)
        )

        self.orientations = {
            "SEM": FibsemStagePosition(
                r=np.radians(stage_settings.rotation_reference),
                t=np.radians(shuttle_pre_tilt),
            ),
            "FIB": FibsemStagePosition(
                r=np.radians(stage_settings.rotation_180),
                t=np.radians(self.system.ion.column_tilt - shuttle_pre_tilt),
            ),
            "MILLING": FibsemStagePosition(
                r=np.radians(stage_settings.rotation_reference), t=milling_stage_tilt
            ),
        }

        # FM is an orientation only where reaching the FM *is* a re-pose: on a
        # compustage the objective is under the grid and the stage turns over to face
        # it. On an offset mount the FM is a place, not a pose -- the stage travels
        # there holding whatever orientation it was in -- so there is no FM entry to
        # derive. (There used to be: a `deepcopy` of the FIB pose, a second name for
        # a pose that already had one. The classifier matched FM last, so the copy
        # was never returned, and deleting it changes no classification -- it only
        # stops `get_orientation("FM")` naming a pose that does not exist.)
        if self.stage_is_compustage:
            self.orientations["FIB"].r = np.radians(
                0
            )  # Compustage is always at 0 rotation
            self.orientations["FIB"].t -= np.radians(180)

            self.orientations["FM"] = FibsemStagePosition(
                r=np.radians(0),
                t=np.radians(-180),
            )

    def set_milling_angle(self, milling_angle: float) -> None:
        """Set the 'stored' milling angle in the system settings."""
        self.system.stage.milling_angle = milling_angle
        self._update_orientations()

    def get_current_milling_angle(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> float:
        """Get the current milling angle in degrees based on the current stage tilt."""

        from fibsem.transformations import convert_stage_tilt_to_milling_angle

        if stage_position is None:
            stage_position = self.get_stage_position()

        # NOTE: this is only valid for sem orientation
        if self.get_stage_orientation(stage_position=stage_position) == "FIB":
            return 90  # stage-tilt + pre-tilt + 90 - column-tilt

        stage_tilt = stage_position.t

        if stage_tilt is None:
            raise ValueError(
                "Stage tilt is not available. Cannot calculate milling angle."
            )

        if self.stage_is_compustage and stage_tilt < np.radians(-90):
            # Compustage stage tilt is inverted, so we need to adjust the angle
            stage_tilt += np.radians(180)

        # Calculate the milling angle from the stage tilt
        milling_angle = convert_stage_tilt_to_milling_angle(
            stage_tilt=stage_tilt,
            pretilt=np.radians(self.system.stage.shuttle_pre_tilt),
            column_tilt=np.radians(self.system.ion.column_tilt),
        )
        return float(np.degrees(milling_angle))

    def is_close_to_milling_angle(
        self, milling_angle: float, atol: float = 2.0
    ) -> bool:
        """Check if the current milling angle is close to the specified milling angle.
        Args:
            milling_angle (float): The target milling angle in degrees.
            atol (float): The absolute tolerance for the comparison.
        Returns:
            bool: True if the current milling angle is close to the specified milling angle, False otherwise
        """
        current_milling_angle = self.get_current_milling_angle()  # degrees

        return bool(np.isclose(current_milling_angle, milling_angle, atol=atol))

    def move_to_milling_angle(
        self, milling_angle: float, rotation: Optional[float] = None
    ) -> bool:
        """Move the stage to the milling angle, based on the current pretilt and column tilt.
        Args:
            milling_angle (float): The target milling angle in radians.
            rotation (Optional[float]): The target rotation angle in radians. If None, uses the current rotation reference.
        Returns:
            bool: True if the stage is close to the target milling angle after the move, False otherwise.
        """

        if rotation is None:
            rotation = np.radians(self.system.stage.rotation_reference)

        # calculate the stage tilt from the milling angle
        stage_tilt = get_stage_tilt_from_milling_angle(self, milling_angle)
        stage_position = FibsemStagePosition(t=stage_tilt, r=rotation)
        self.safe_absolute_stage_movement(stage_position)

        # milling_angle is radians here; is_close_to_milling_angle compares degrees (FIB-853)
        return self.is_close_to_milling_angle(np.degrees(milling_angle))

    def _beam_view_tilt(self, beam_type: BeamType) -> float:
        """Tilt of a beam column's viewing axis from the electron column, in radians."""
        if beam_type is BeamType.ELECTRON:
            return 0.0
        if beam_type is BeamType.ION:
            return np.deg2rad(self.system.ion.column_tilt)
        # the previous inline form left the adjustment unbound here; keep failing loudly
        raise ValueError(f"Unsupported beam type: {beam_type}")

    def _view_corrected_stage_movement(
        self,
        expected_y: float,
        view_tilt: float = 0.0,
    ) -> FibsemStagePosition:
        """Project a displacement seen in an image onto the tilted sample plane.

        Every view of the sample shares this projection and differs only in how far
        its viewing axis is tilted from the electron column: the electron column is
        0, the ion column is its ``column_tilt``, and the fluorescence camera is its
        ``camera_tilt``. The sample is additionally tilted by the shuttle pre-tilt,
        so an in-image y-displacement becomes a movement along the holder, split
        across the stage y- and z-axes.

        Args:
            expected_y: distance along the image y-axis, in metres.
            view_tilt: tilt of the viewing axis from the electron column, in radians.

        Returns:
            FibsemStagePosition: y-corrected stage movement (relative position)
        """

        # TODO: replace with camera matrix * inverse kinematics

        # all angles in radians
        sem_column_tilt = np.deg2rad(self.system.electron.column_tilt)

        stage_pretilt = np.deg2rad(self.system.stage.shuttle_pre_tilt)

        stage_rotation_flat_to_eb = np.deg2rad(self.system.stage.rotation_reference) % (
            2 * np.pi
        )
        stage_rotation_flat_to_ion = np.deg2rad(self.system.stage.rotation_180) % (
            2 * np.pi
        )

        # current stage position
        current_stage_position = self.get_stage_position()
        stage_rotation = current_stage_position.r % (2 * np.pi)
        stage_tilt = current_stage_position.t

        # the compustage does not have pre-tilt, cannot rotate, but tilts 180 deg.
        if self.stage_is_compustage:
            # if stage_tilt < 0:
            expected_y *= -1.0

            stage_tilt += np.pi

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation # TODO: migrate to orientation
        from fibsem import movement

        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_eb, atol=5
        ):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        if self.stage_is_compustage and self.get_stage_orientation() == "FIB":
            # Stays after FIB-834 made `rotation_180` derived. A compustage does not
            # rotate, so it derives to `rotation_reference` -- the same value the
            # configuration used to state, which is why both comparisons above still
            # match and this flip is still what separates the two sides. Deriving it to
            # a half turn instead would have moved the sign here, silently.
            expected_y *= -1.0
            PRETILT_SIGN = -1.0

        corrected_pretilt_angle = PRETILT_SIGN * (
            stage_pretilt + sem_column_tilt
        )  # electron angle = 0, ion = 52

        # perspective tilt adjustment (difference between perspective view and sample coordinate system)
        perspective_tilt_adjustment = -corrected_pretilt_angle - view_tilt

        # the amount the sample has to move in the y-axis
        y_sample_move = expected_y / np.cos(stage_tilt + perspective_tilt_adjustment)

        # the amount the stage has to move in each axis
        y_move = y_sample_move * np.cos(corrected_pretilt_angle)
        z_move = -y_sample_move * np.sin(
            corrected_pretilt_angle
        )  # TODO: investigate this

        return FibsemStagePosition(x=0, y=y_move, z=z_move)

    def _inverse_view_corrected_stage_movement(
        self,
        dy: float,
        dz: float,
        view_tilt: float = 0.0,
    ) -> float:
        """Recover the in-image y-displacement from a y/z stage movement.

        Inverse of :meth:`_view_corrected_stage_movement`.

        Deferred to :func:`fibsem.transformations.inverse_view_corrected_dy` rather than
        derived here. This method used to carry its own copy of the trigonometry, as did
        `imaging/tiling/reprojection.py`, so one decision about the geometry lived in
        three places and only stayed consistent by everyone editing all three. They now
        share the one implementation, which is what `transformations` was extracted for.

        Two consequences, both wanted:

        * **No hardware read for the orientation.** The old copy asked
          `get_stage_orientation()` to decide whether a compustage was at the FIB pose;
          the shared version derives it from the pose it was handed. The stage position
          is still read here, because "the current pose" is this method's contract.
        * **The compustage FIB test gains the rotation term** the live path always had
          and the tiled copy lacked (FIB-500). Only reachable poses matter and none
          change: a compustage has no rotation axis, so the combinations that differ --
          tilt -128 with a non-zero rotation -- cannot be produced by any acquisition.

        Args:
            dy: actual y stage movement
            dz: actual z stage movement
            view_tilt: tilt of the viewing axis from the electron column, in radians.

        Returns:
            float: expected_y input that would produce the given dy, dz movements
        """
        position = self.get_stage_position()
        return inverse_view_corrected_dy(
            dy=dy,
            dz=dz,
            view_tilt=view_tilt,
            geometry=self.hardware_geometry(),
            stage_rotation=position.r if position.r is not None else 0.0,
            stage_tilt=position.t if position.t is not None else 0.0,
        )

    def _fm_image_to_stage_delta(self, dx: float, dy: float) -> Tuple[float, float]:
        """Map a displacement in the displayed FM image onto stage axes.

        The driver hands out stage-aligned images -- any fixed rotation or flip of the
        mount is corrected by `FluorescenceMicroscope.mount_transform` before the
        user's display preference is applied -- so undoing that preference is all
        that is needed here. It is read live rather than from configuration, since
        the user can change it mid-session, and every remaining transform is its own
        inverse, so applying it maps in both directions.
        """
        if self.fm is None:
            return dx, dy
        return self.fm._transform.apply_to_delta(dx, dy)

    def _fm_stage_delta(self, dx: float, dy: float) -> FibsemStagePosition:
        """Relative stage movement for a displacement seen in the displayed FM image.

        The projection shared by :meth:`fm_stable_move` and
        :meth:`project_fm_stable_move`, so the two cannot disagree about where a
        given displacement lands.

        Input is in the frame the user is looking at, so the display transform is
        undone here -- the direct counterpart of `project_stable_move` undoing the
        beam's scan rotation before projecting. Doing it in the shared helper rather
        than at one entry point means neither path can skip it.

        That applies to synthesised displacements as much as to clicks. A tile step is
        expressed in the same frame as the tile it positions, and the mosaic canvas is
        in display space, because `stitch_tileset` pastes image data that already
        carries the transform. If the arrangement did not carry it while the content
        did, the two would disagree and every seam would break.

        Args:
            dx: distance along the x-axis, in displayed image coordinates.
            dy: distance along the y-axis, in displayed image coordinates.

        Returns:
            FibsemStagePosition: relative movement, with the y-displacement split
            across the stage y- and z-axes by the sample tilt.

        Raises:
            ValueError: if no fluorescence microscope is available.
        """
        if self.fm is None:
            raise ValueError("Fluorescence microscope is not available.")

        dx, dy = self._fm_image_to_stage_delta(dx, dy)

        yz_move = self._view_corrected_stage_movement(
            expected_y=dy,
            view_tilt=np.deg2rad(self.fm.camera_tilt),
        )
        return FibsemStagePosition(
            x=dx, y=yz_move.y, z=yz_move.z, r=0, t=0, coordinate_system="RAW"
        )

    def project_fm_stable_move(
        self, dx: float, dy: float, base_position: FibsemStagePosition
    ) -> FibsemStagePosition:
        """Where the stage would end up after an FM displacement, without moving.

        The fluorescence counterpart of :meth:`project_stable_move`. That one is
        abstract and reimplemented by every driver, because it carries beam-specific
        work -- scan rotation, `beam_type` dispatch. A camera has neither, and both
        `camera_tilt` and the projection itself are already concrete here, so this
        needs no per-driver override.

        Like its beam counterpart, it maps a displacement in the image the user is
        looking at, so the display transform is undone before projecting -- the same
        role scan rotation plays for a beam. Both take the displayed frame as their
        input convention, including for synthesised displacements: `tiled.py` hands
        `project_stable_move` raw grid offsets and relies on the scan-rotation undo
        for exactly this reason.

        Args:
            dx: distance along the x-axis, in displayed image coordinates.
            dy: distance along the y-axis, in displayed image coordinates.
            base_position: the position the displacement is measured from.

        Returns:
            FibsemStagePosition: the absolute position the displacement lands on.

        Raises:
            ValueError: if no fluorescence microscope is available.
        """
        delta = self._fm_stage_delta(dx, dy)

        new_position = deepcopy(base_position)
        new_position.x += delta.x
        new_position.y += delta.y
        new_position.z += delta.z

        return new_position

    def hardware_geometry(self) -> FibsemHardwareGeometry:
        """The fixed geometry this instrument is arranged in.

        The beam counterpart of :meth:`fm_image_geometry`, and its shared core. Both
        stamp the same terms onto an acquired image so it can be reprojected later
        without a live microscope to consult.

        ``stage_is_compustage`` is the authority here -- ThermoFisher reads it from
        ``connection.specimen.compustage.is_installed``. Recording it is what lets
        the reprojection stop inferring it from the model name (FIB-481).
        """
        return FibsemHardwareGeometry.from_system_settings(
            self.system, is_compustage=self.stage_is_compustage
        )

    def _set_additional_metadata(self, image: FibsemImage) -> None:
        """Stamp who, which run, which instrument and how it is arranged onto an image.

        Lifted here from ``ThermoMicroscope`` in FIB-481: the same three lines were
        written out at eight acquisition sites across three microscope classes, so a
        change to what an image records meant finding all eight.
        """
        image.metadata.user = self.user
        # Copied, not referenced. This now carries the item and task as well as the
        # experiment (FIB-466), and those change as the run progresses -- sharing one
        # object would rewrite the item on every image already acquired. It was shared
        # before, which was harmless only because nothing mutated it.
        image.metadata.experiment = deepcopy(self.experiment)
        # Copied, not referenced. `hardware_geometry()` returns a fresh record, so
        # aliasing the live SystemInfo here would leave one field on the image a
        # snapshot and the other a window onto the microscope. TescanMicroscope
        # rewrites system.info.model/serial/software_version on every acquisition,
        # so the alias reached back into images already taken. Pre-dates FIB-481 --
        # `metadata.system = self.system` aliased the whole thing -- but the two
        # fields set together should not disagree about what they are.
        image.metadata.system_info = deepcopy(self.system.info)
        image.metadata.hardware_geometry = self.hardware_geometry()

    def fm_image_geometry(self) -> FibsemHardwareGeometry:
        """The geometry the FM is currently imaging under.

        The same record :meth:`hardware_geometry` returns, with the camera's own two
        terms filled in. Stamped onto acquired images so they can be reprojected later
        without a live microscope, and used for the live view here so that both paths
        run the same projection rather than two that merely agree today.

        Raises:
            ValueError: if no fluorescence microscope is available.
        """
        if self.fm is None:
            raise ValueError("Fluorescence microscope is not available.")

        # replace() on the shared record rather than rebuilding it: the instrument
        # terms are gathered in one place, and only the camera's are added here.
        return dataclasses.replace(
            self.hardware_geometry(),
            transform=self.fm._transform or CameraImageTransform.NONE,
            camera_tilt=self.fm.camera_tilt,
        )

    def fm_stable_move(self, dx: float, dy: float) -> FibsemStagePosition:
        """Move the stage by a displacement seen in the fluorescence image.

        The fluorescence counterpart of :meth:`stable_move`: click a point in the FM
        view and the stage goes there, holding the focal plane. The projection is the
        same one the beams use, with the camera's own axis tilt
        (`FluorescenceMicroscope.camera_tilt`) in place of a column tilt, so the
        foreshortening of a tilted sample is accounted for and the movement stays in
        the sample plane -- which is what keeps the objective in focus.

        Args:
            dx: distance along the x-axis (image coordinates), as for stable_move.
            dy: distance along the y-axis (image coordinates), as for stable_move.

        Returns:
            FibsemStagePosition: the stage position after the move.

        Raises:
            ValueError: if no fluorescence microscope is available.
        """
        if self.fm is None:
            raise ValueError("Fluorescence microscope is not available. Cannot move.")

        if self.fm.objective.state != "Inserted":
            logging.warning(
                "Moving via the fluorescence image while the objective is not inserted "
                f"(state: {self.fm.objective.state}); the view may not match the sample."
            )

        # The display transform is undone inside _fm_stage_delta, so this and
        # project_fm_stable_move share one input convention.
        stage_position = self._fm_stage_delta(dx, dy)

        # NOTE: no working-distance restore. That is beam bookkeeping; the objective
        # keeps focus because the move stays in the sample plane.
        self.move_stage_relative(stage_position)

        logging.debug(
            {
                "msg": "fm_stable_move",
                "dx": dx,
                "dy": dy,
                "camera_tilt": self.fm.camera_tilt,
                "position": stage_position.to_dict(),
            }
        )

        return self.get_stage_position()

    def move_to_device(self, device: str) -> None:
        """Move the stage to the predefined device position."""
        logging.warning(
            f"move_to_device is not implemented for {self.__class__.__name__}."
        )
        pass

    def _get_device(self, device: str) -> StageDeviceSettings:
        """The configuration for `device`, or a refusal naming the ones there are."""
        try:
            return self.system.stage.devices[device]
        except KeyError:
            raise ValueError(
                f"Microscope {device} not supported. "
                f"Configured devices: {sorted(self.system.stage.devices)}."
            ) from None

    def get_device_origin(self, device: str) -> FibsemStagePosition:
        """Where the stage travels for `device` to see the sample.

        The device's *origin*, not a position expressed at it -- for that, ask
        `get_target_position(position, target_device=...)`.

        Partial: an offset fluorescence microscope is an x location and leaves y, z, r
        and t free, so the axes it does not constrain come back `None`.
        """
        return deepcopy(self._get_device(device).origin)

    def is_at_device(
        self, device: str, stage_position: Optional[FibsemStagePosition] = None
    ) -> bool:
        """Is the stage at `device`?

        The question nothing could ask before. `get_stage_orientation` cannot answer
        it on an offset mount -- the FM orientation there is byte-identical to the FIB
        one -- because it is not a question about orientation at all.

        Only meaningful for a device the stage *travels to*. A device that comes to
        the sample instead has no origin, and answers `False` rather than pretending
        position decides it.
        """
        if stage_position is None:
            stage_position = self.get_stage_position()
        return self._get_device(device).contains(
            stage_position, self.system.stage.device_range
        )

    def get_current_device(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> Optional[str]:
        """Which device the stage is at, or `None` if it is at no configured device.

        `None` is a real answer, not a failure to find one: the device ranges
        deliberately leave a gap between them, so a stage part-way through a traverse
        -- or left there by one that was aborted -- is at neither.

        Positional, so it is the wrong question on a compustage, where the beams and
        the FM are the same place reached by flipping and the devices fully overlap.
        Nothing asks it there: `move_to_microscope` branches to the compustage path
        first, and the device is decided by orientation instead.
        """
        if stage_position is None:
            stage_position = self.get_stage_position()

        for device in self.system.stage.devices:
            if self.is_at_device(device, stage_position):
                return device
        return None

    def get_device_imaging_state(
        self, device: str, stage_position: Optional[FibsemStagePosition] = None
    ) -> DeviceImagingState:
        """Can `device` see the sample from where the stage is -- and if not, why not.

        One question, answered the same way on both mountings, with the mounting
        expressed entirely in configuration (FIB-839). Two terms:

        * **place** -- `is_at_device`, against the device's declared origin
        * **pose** -- `get_stage_orientation`, against the device's declared
          `acquisition_orientations`; an empty list constrains nothing and the term
          is vacuously true

        Each mounting makes a *different* term trivially true. A compustage FM shares
        the beams' origin, so the place carries nothing and the pose carries it all;
        an offset FM images from the pose the sample was carried out in, so the pose
        carries nothing and the place carries it all. Which is why the same
        conjunction discriminates on both -- and why the failing term names the
        remedy: a wrong place means travel, a wrong pose means re-pose.

        Callers act on the value by policy, not uniformly -- see
        `DeviceImagingState`. Pass `stage_position` to ask about a stored pose rather
        than the current one; both workflow tasks do.
        """
        # The fluorescence microscope is the one device whose instrument can be
        # absent -- the beams always exist, and there is no third device yet. `fm` is
        # an object or None (a present-but-faulted state is not modelled; noted on
        # FIB-839), so this is the whole of the "no device" test.
        if device == "FM" and self.fm is None:
            return DeviceImagingState.NO_DEVICE

        at_device = self.is_at_device(device, stage_position)
        orientations = self._get_device(device).acquisition_orientations
        in_orientation = (
            not orientations
            or self.get_stage_orientation(stage_position) in orientations
        )

        if at_device and in_orientation:
            return DeviceImagingState.READY
        if in_orientation:
            return DeviceImagingState.NEEDS_TRAVEL
        if at_device:
            return DeviceImagingState.NEEDS_REPOSE
        return DeviceImagingState.NEEDS_REPOSE_THEN_TRAVEL

    def describe_device_imaging_state(
        self,
        device: str,
        state: Optional[DeviceImagingState] = None,
        stage_position: Optional[FibsemStagePosition] = None,
    ) -> str:
        """One sentence a person can act on, for each imaging state.

        The vocabulary is the two axes' (FIB-858): the stage travels between
        *devices* and is re-posed between *orientations*, and the failing term names
        the verb -- so a refusal built from this says which of the two to do, in
        which order, rather than announcing a false generality like "not in a valid
        orientation" for a stage that is 48.8 mm from the instrument.

        Pass `state` when it has already been asked, so a message and the decision it
        explains cannot be about two different moments.
        """
        if state is None:
            state = self.get_device_imaging_state(device, stage_position)

        instrument = (
            "the fluorescence microscope" if device == "FM" else f"the {device} device"
        )

        if state is DeviceImagingState.NO_DEVICE:
            return "This system has no fluorescence microscope."
        if state is DeviceImagingState.READY:
            return f"{instrument.capitalize()} can image the sample from here."

        orientation = self.get_stage_orientation(stage_position)
        held = (
            f"held in the {orientation} orientation"
            if orientation != "NONE"
            else "held in an unrecognised orientation"
        )
        allowed = self._get_device(device).acquisition_orientations
        images_from = (
            f"images from the {' or '.join(allowed)} orientation"
            if allowed
            else "images from any orientation"
        )

        if state is DeviceImagingState.NEEDS_TRAVEL:
            return (
                f"The stage is {held}, which {instrument} images from, but it is "
                f"away from the {device} device: travel there "
                f"(move_to_device('{device}'))."
            )
        if state is DeviceImagingState.NEEDS_REPOSE:
            return (
                f"The stage is at the {device} device but {held}; {instrument} "
                f"{images_from}. Re-pose via the beams "
                f"(move_to_device('{device}', orientation='{allowed[0]}'))."
            )
        return (
            f"The stage is {held}, away from the {device} device; {instrument} "
            f"{images_from}. Re-pose at the beams and travel out "
            f"(move_to_device('{device}'))."
        )

    def _warn_on_fluorescence_geometry(self) -> None:
        """Warn, at connect, about FM geometry that will misbehave quietly later.

        Both cases produce no error at all in operation -- the imaging-state
        conjunction is simply never (or always) true somewhere it should not be --
        so the one loud moment available is connection, while the configuration is
        in front of the person who wrote it.
        """
        if self.fm is None:
            return

        devices = self.system.stage.devices

        # An offset mount that enabled the FM but declared no geometry inherits the
        # default -- the objective under the grid, sharing the beams' origin -- so
        # every place-term answer is about somewhere its FM is not.
        if not self.stage_is_compustage and devices == DEFAULT_STAGE_DEVICES:
            logging.warning(
                "A fluorescence microscope is enabled but no `stage.devices` block "
                "is declared, so the FM defaults to the beams' origin. An offset "
                "mount (METEOR, iFLM) must declare its traverse -- see "
                "sim-iflm-configuration.yaml."
            )

        # A compustage FM declared away from the beams is a phantom: the stage
        # reaches its FM by flipping, not travelling, so a distinct origin is
        # somewhere it never goes and `is_at_device(\"FM\")` is False at the
        # objective itself.
        if self.stage_is_compustage and "FM" in devices and "FIBSEM" in devices:
            if devices["FM"].origin != devices["FIBSEM"].origin:
                logging.warning(
                    "This compustage declares an FM device origin away from the "
                    "beams. Its objective is under the grid: the FM shares the "
                    "beams' origin, and a distinct origin is a place the stage "
                    "never travels to."
                )

    def _device_translation(self, source: str, target: str) -> FibsemStagePosition:
        """The relative stage move from one device to another.

        A difference of two configured places rather than a constant, so the traverse
        and the "am I already there" windows can no longer drift apart. Relative
        rather than absolute on purpose: the devices constrain x only, and a relative
        move carries y, z, r and t across unchanged.

        **Nothing on a compustage.** There the objective is under the grid, so the
        beams and the FM are the same place and the stage reaches one from the other
        by flipping, not travelling -- the configured origins describe an offset
        chamber and do not apply. Answering here rather than at each call site is the
        same arrangement `_get_compucentric_rotation_position` already uses: the
        primitive is the no-op, so no caller needs a stage-type branch.
        """
        if self.stage_is_compustage:
            return FibsemStagePosition()

        source_origin = self._get_device(source).origin
        target_origin = self._get_device(target).origin

        translation = FibsemStagePosition()
        for axis in DEVICE_AXES:
            start, end = getattr(source_origin, axis), getattr(target_origin, axis)
            if start is not None and end is not None:
                setattr(translation, axis, end - start)
        return translation

    def move_to_device(self, device: str, orientation: Optional[str] = None) -> None:
        """Travel to `device`, re-posing on the way when the pose has to change.

        One call that owns the safe order -- retract the objective, re-pose at the
        beams, travel out -- so a rotation never happens with the stage parked under
        an objective. The rotation guard (FIB-841) stays underneath as the last-line
        assert; the route this composes never trips it.

        `orientation` names the pose to arrive in. Omitted, the pose is carried
        across untouched whenever the target device can image from it -- that is the
        point of a traverse, and the reason this must not pass the current
        orientation's *name* through `move_to_orientation`: doing so would snap r and
        t to nominal and quietly discard a milling angle somebody dialled in. When
        the pose does have to change (an offset FM images in the FIB pose; asking
        for it from SEM used to be a refusal), the device's first declared
        acquisition orientation is used.
        """
        target_device = self._get_device(device)  # refuses by name

        if self.stage_is_compustage:
            self._move_to_device_compustage(device, orientation)
            return

        if device == "FM" and not self.fm:
            raise ValueError("FM module is not available. Cannot move to FM position.")

        stage_position = self.get_stage_position()
        source = self.get_current_device(stage_position)
        if source is None:
            raise ValueError(
                f"The stage is not at any configured device "
                f"({sorted(self.system.stage.devices)}), so there is nothing to "
                f"travel from. Position: {stage_position}."
            )

        # The pose to arrive in. An explicit ask is honoured as asked; otherwise the
        # pose is carried across, unless the target device cannot image from it --
        # then its first declared acquisition orientation stands in.
        desired = orientation
        allowed = target_device.acquisition_orientations
        if desired is None and allowed:
            if self.get_stage_orientation(stage_position) not in allowed:
                desired = allowed[0]
                logging.info(
                    f"The {device} device images from {allowed}; re-posing to "
                    f"{desired} at the beams before travelling."
                )

        if desired is None and source == device:
            logging.info(f"Already at {device} position, no need to move.")
        else:
            # Retracted immediately before the stage moves, and only then. The
            # objective must not be out over the sample while the stage moves, but
            # every reason to retract it is the motion itself -- so a call that
            # refuses, or finds it has nowhere to go, leaves the objective exactly
            # as it found it rather than pulling it out of the sample for nothing.
            logging.info(f"Moving to {device} position...")
            if self.fm is not None:
                self.fm.objective.retract()

            if desired is not None:
                # The bracketing order: every re-pose happens at the beams, where
                # the rotation is about the sample rather than a 48.8 mm arm.
                if source != "FIBSEM":
                    self.move_stage_relative(self._device_translation(source, "FIBSEM"))
                self.move_to_orientation(desired)
                if device != "FIBSEM":
                    self.move_stage_relative(self._device_translation("FIBSEM", device))
            else:
                self.move_stage_relative(self._device_translation(source, device))

        # Unconditional, so that the postcondition is the device *and* the objective
        # state together: asking again for a device the stage is already at cannot
        # leave the FM blind.
        if device == "FM":
            self.fm.objective.insert()

    def _move_to_device_compustage(
        self, device: str, orientation: Optional[str] = None
    ) -> None:
        """The compustage's devices are one place: reaching either is a re-pose.

        With no `orientation` asked for, FIBSEM lands at SEM -- the pose every
        caller of the old `move_to_microscope` relied on -- and the FM lands at its
        own orientation, under the objective.
        """
        if not self.fm:
            raise ValueError("FM module is not available. Cannot move to FM position.")

        self.fm.objective.retract()  # retract objective (safety precaution)

        if device == "FIBSEM":
            self.move_to_orientation(orientation or "SEM")

        if device == "FM":
            if orientation is not None:
                self.move_to_orientation(orientation)
            else:
                self.move_stage_absolute(self.get_orientation("FM"))
            self.fm.objective.insert()  # insert objective

    def move_to_microscope(self, target: str) -> None:
        """Deprecated name for `move_to_device(target)` -- the last place a device
        was called a microscope. Kept as a shim for its many callers."""
        self.move_to_device(target)

    def move_to_microscope_compustage(self, target: str) -> None:
        """Deprecated name for the compustage half of `move_to_device`."""

        if not self.stage_is_compustage:
            raise ValueError(
                "This method is only available for Compustage microscopes."
            )
        self._get_device(target)  # refuses by name if it is not a configured device
        self._move_to_device_compustage(target)

    @property
    def current_grid(self) -> str:
        try:
            grid = self._stage.current_grid
            if grid is None:
                return "NONE"
            return grid.name
        except Exception:
            return "NONE"

    @property
    def manufacturer(self) -> str:
        # NOTE: this base default means every backend that does not override the
        # property (Demo, Zeiss, Odemis) reports ThermoFisher -- FIB-300 tracks
        # whether it should serve self.system.info.manufacturer instead.
        return manufacturers.THERMOFISHER


# `ThermoMicroscope` moved to `fibsem.microscopes.autoscript`. These names are served
# lazily so `from fibsem.microscope import ThermoMicroscope` keeps working for external
# scripts and plugins, without this module importing the AutoScript backend at load.
_MOVED_TO_AUTOSCRIPT = frozenset(
    {"ThermoMicroscope", "THERMO_API_AVAILABLE", "AutoScriptException"}
)


def __getattr__(name: str) -> Any:
    if name in _MOVED_TO_AUTOSCRIPT:
        warnings.warn(
            f"fibsem.microscope.{name} has moved to fibsem.microscopes.autoscript; "
            "import it from there.",
            DeprecationWarning,
            stacklevel=2,
        )
        from fibsem.microscopes import autoscript

        return getattr(autoscript, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
