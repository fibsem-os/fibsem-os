from __future__ import annotations
import copy
import datetime
import logging
import os
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING

import numpy as np
from skimage import transform
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from psygnal import Signal


import fibsem.constants as constants
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    BeamSettings,
    BeamSystemSettings,
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemExperiment,
    FibsemGasInjectionSettings,
    FibsemImage,
    FibsemImageMetadata,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemPolygonSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    MillingState,
    Point,
    RangeLimit,
    SystemSettings,
)
from fibsem.transformations import get_stage_tilt_from_milling_angle

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray
    from fibsem.structures import TFibsemPatternSettings

class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""
    milling_progress_signal = Signal(dict)
    tiled_acquisition_signal = Signal(dict)
    _last_imaging_settings: ImageSettings
    system: SystemSettings
    _patterns: List
    stage_is_compustage: bool = False
    milling_channel: BeamType = BeamType.ION

    # live acquisition
    sem_acquisition_signal = Signal(FibsemImage)
    fib_acquisition_signal = Signal(FibsemImage)
    _stop_acquisition_event = threading.Event()
    _acquisition_thread: threading.Thread = None
    _threading_lock: threading.RLock = threading.RLock()

    fm: 'FluorescenceMicroscope' = None

    stage_position_changed = Signal(FibsemStagePosition)
    _stage_position: FibsemStagePosition = None

    @abstractmethod
    def connect_to_microscope(self, ip_address: str, port: int, reset_beam_shift: bool = True) -> None:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def acquire_image(self, image_settings: Optional[ImageSettings] = None, beam_type: Optional[BeamType] = None) -> FibsemImage:
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
            target=self._acquisition_worker,
            args=(beam_type,),
            daemon=True
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
    def autocontrast(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        pass

    @abstractmethod
    def auto_focus(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
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

    def _create_sample_stage(self) -> None:
        """Create the sample stage and holder based on the system settings."""

        from fibsem.microscopes._stage import SampleGrid, SampleHolder, Stage, SampleGridLoader
        if self.stage_is_compustage:
            grid01 = SampleGrid(name="Grid-01", index=1, 
                                position=FibsemStagePosition(name="Grid-01", x=-0e-3, y=0.0, z=0.0, r=0.0, t=np.radians(0)))
            holder = SampleHolder(name="CompuStage Holder", pre_tilt=0.0, reference_rotation=0.0, grids={"Grid-01": grid01})
            loader = SampleGridLoader(parent=self)
        else:
            from fibsem.config import SAMPLE_HOLDER_CONFIGURATION_PATH
            orientation = self.get_orientation("SEM")
            holder = SampleHolder.load(SAMPLE_HOLDER_CONFIGURATION_PATH)
            for grid_name, grid in holder.grids.items():
                grid.position.r = orientation.r
                grid.position.t = orientation.t

            loader = None

        self._stage = Stage(parent=self, holder=holder, loader=loader)

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
    def stable_move(self,dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        pass

    @abstractmethod
    def vertical_move(self, dy: float, dx: float = 0, static_wd: bool = True) -> FibsemStagePosition:
        pass

    @abstractmethod
    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        pass

    def move_flat_to_beam(self, beam_type: BeamType, _safe:bool = True) -> None:
        """Move the sample surface flat to the electron or ion beam."""

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
        stage_position = FibsemStagePosition(r=rotation, t=tilt, coordinate_system="Raw")

        logging.debug({"msg": "move_flat_to_beam", "stage_position": stage_position.to_dict(), "beam_type": beam_type.name})

        if _safe:
            self.safe_absolute_stage_movement(stage_position)
        else:
            self.move_stage_absolute(stage_position)

    def move_to_orientation(self, orientation: str) -> None:
        """Move the stage to the given named orientation (e.g. 'SEM', 'FIB', 'MILLING')."""
        pos = self.get_orientation(orientation)
        stage_position = FibsemStagePosition(r=pos.r, t=pos.t, coordinate_system="Raw")
        logging.info(f"moving to orientation: {orientation}")
        self.safe_absolute_stage_movement(stage_position)

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
    def move_manipulator_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
        pass

    @abstractmethod
    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str) -> None:
        pass

    @abstractmethod
    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        pass

    @abstractmethod
    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        pass

    @abstractmethod
    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False) -> None:
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
    def get_available_values(self, key: str, beam_type: Optional[BeamType] = None) -> List[Union[str, float, int]]:
        pass

    def get_available_values_cached(self, key: str, beam_type: Optional[BeamType] = None) -> List[Union[str, float, int]]:
        """Get available values with caching to avoid repeated microscope queries.

        Args:
            key: The parameter key to get available values for.
            beam_type: The beam type (optional).

        Returns:
            List of available values for the given key.
        """
        if not hasattr(self, '_available_values_cache'):
            logging.info("Initializing available values cache.")
            self._available_values_cache: Dict[str, List[Union[str, float, int]]] = {}

        cache_key = f"{key}_{beam_type.name if beam_type else 'None'}"
        if cache_key not in self._available_values_cache:
            logging.info(f"Caching available values for key: {key}, beam_type: {beam_type}")
            self._available_values_cache[cache_key] = self.get_available_values(key, beam_type)
        return self._available_values_cache[cache_key]

    def clear_available_values_cache(self, key: Optional[str] = None, beam_type: Optional[BeamType] = None) -> None:
        """Clear the available values cache.

        Args:
            key: If provided, only clear cache for this key. Otherwise clear all.
            beam_type: The beam type (used with key to clear specific entry).
        """
        if not hasattr(self, '_available_values_cache'):
            return

        if key is None:
            self._available_values_cache.clear()
        else:
            cache_key = f"{key}_{beam_type.name if beam_type else 'None'}"
            self._available_values_cache.pop(cache_key, None)

    # TODO: use a decorator instead?
    def get(self, key: str, beam_type: Optional[BeamType] = None) -> Union[float, int, bool, str, list, tuple, Point]:
        """Get wrapper for logging."""
        value = self._get(key, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug({"msg": "get", "key": key, "beam_type": beam_name, "value": value})
        return value

    def set(self, key: str, value: Union[str, float, int, tuple, list, Point], beam_type: Optional[BeamType] = None) -> None:
        """Set wrapper for logging"""
        self._set(key, value, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug({"msg": "set", "key": key, "beam_type": beam_name, "value": value})

    @abstractmethod
    def _get(self, key: str, beam_type: Optional[BeamType] = None) -> Union[float, int, bool, str, list]:
        pass

    @abstractmethod
    def _set(self, key: str, value: Union[str, float, int, list, tuple, Point], beam_type: Optional[BeamType] = None) -> None:
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
        logging.debug({"msg": "get_imaging_settings", "image_settings": image_settings.to_dict(), "beam_type": beam_type.name})
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
        logging.debug({"msg": "set_imaging_settings", "image_settings": image_settings.to_dict(), "beam_type": image_settings.beam_type.name})

        return 

    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:
        """Get the current beam settings for the specified beam type.
        """

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
        logging.debug({"msg": "get_beam_settings", "beam_settings": beam_settings.to_dict(), "beam_type": beam_type.name})

        return beam_settings

    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        """Set the beam settings for the specified beam type"""
        logging.debug(f"Setting {beam_settings.beam_type.name} beam settings...")
        self.set_working_distance(beam_settings.working_distance, beam_settings.beam_type)
        self.set_beam_current(beam_settings.beam_current, beam_settings.beam_type)
        self.set_beam_voltage(beam_settings.voltage, beam_settings.beam_type)
        self.set_field_of_view(beam_settings.hfw, beam_settings.beam_type)
        self.set_resolution(beam_settings.resolution, beam_settings.beam_type)
        self.set_dwell_time(beam_settings.dwell_time, beam_settings.beam_type)
        self.set_stigmation(beam_settings.stigmation, beam_settings.beam_type)
        self.set_beam_shift(beam_settings.shift, beam_settings.beam_type)
        self.set_scan_rotation(beam_settings.scan_rotation, beam_settings.beam_type)
        self.set("preset", beam_settings.preset, beam_settings.beam_type)

        logging.debug({"msg": "set_beam_settings", "beam_settings": beam_settings.to_dict(), "beam_type": beam_settings.beam_type.name})
        return

    def get_beam_system_settings(self, beam_type: BeamType) -> BeamSystemSettings:
        """Get the current beam system settings for the specified beam type.
        """
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

        logging.debug({"msg": "get_beam_system_settings", "settings": beam_system_settings.to_dict(), "beam_type": beam_type.name})
        return beam_system_settings

    def set_beam_system_settings(self, settings: BeamSystemSettings) -> None:
        """Set the beam system settings for the specified beam type.
        """
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

        logging.debug( {"msg": "set_beam_system_settings", "settings": settings.to_dict(), "beam_type": beam_type.name})
    
        return

    def get_detector_settings(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemDetectorSettings:
        """Get the current detector settings for the specified beam type.
        """
        logging.debug(f"Getting {beam_type.name} detector settings...")
        detector_settings = FibsemDetectorSettings(
            type=self.get_detector_type(beam_type),
            mode=self.get_detector_mode(beam_type),
            brightness=self.get_detector_brightness(beam_type),
            contrast=self.get_detector_contrast(beam_type),
        )
        logging.debug({"msg": "get_detector_settings", "detector_settings": detector_settings.to_dict(), "beam_type": beam_type.name})
        return detector_settings
    
    def set_detector_settings(self, detector_settings: FibsemDetectorSettings, beam_type: BeamType = BeamType.ELECTRON) -> None:
        """Set the detector settings for the specified beam type"""
        logging.debug(f"Setting {beam_type.name} detector settings...")
        self.set_detector_type(detector_settings.type, beam_type)
        self.set_detector_mode(detector_settings.mode, beam_type)
        self.set_detector_brightness(detector_settings.brightness, beam_type)
        self.set_detector_contrast(detector_settings.contrast, beam_type)
        logging.debug({"msg": "set_detector_settings", "detector_settings": detector_settings.to_dict(), "beam_type": beam_type.name})

        return

    def get_microscope_state(self, beam_type: Optional[BeamType] = None) -> MicroscopeState:
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
            stage_position=stage_position,                                  # get absolute stage coordinates (RAW)
            electron_beam=electron_beam,                                    # electron beam state
            ion_beam=ion_beam,                                              # ion beam state
            electron_detector=electron_detector,                            # electron beam detector state
            ion_detector=ion_detector,                                      # ion beam detector state
        )

        logging.debug({"msg": "get_microscope_state", "state": current_microscope_state.to_dict()})

        return deepcopy(current_microscope_state)

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        """Reset the microscope state to the provided state."""

        if self.is_available("electron_beam"):
            if microscope_state.electron_beam is not None:
                self.set_beam_settings(microscope_state.electron_beam)
            if microscope_state.electron_detector is not None:
                self.set_detector_settings(microscope_state.electron_detector, BeamType.ELECTRON)
        if self.is_available("ion_beam"):
            if microscope_state.ion_beam is not None:
                self.set_beam_settings(microscope_state.ion_beam)
            if microscope_state.ion_detector is not None:
                self.set_detector_settings(microscope_state.ion_detector, BeamType.ION)
        if self.is_available("stage") and microscope_state.stage_position is not None:
            self.safe_absolute_stage_movement(microscope_state.stage_position)
        if self.fm is not None and microscope_state.objective_position is not None:
            self.fm.objective.move_absolute(microscope_state.objective_position)

        logging.debug({"msg": "set_microscope_state", "state": microscope_state.to_dict()})

        return

    def set_milling_settings(self, mill_settings: FibsemMillingSettings) -> None:
        self.set("active_view", mill_settings.milling_channel, mill_settings.milling_channel)
        self.set("active_device", mill_settings.milling_channel, mill_settings.milling_channel)
        self.set("default_patterning_beam_type", mill_settings.milling_channel, mill_settings.milling_channel)
        self.set("application_file", mill_settings.application_file, mill_settings.milling_channel)
        self.set("patterning_mode", mill_settings.patterning_mode, mill_settings.milling_channel)
        self.set("hfw", mill_settings.hfw, mill_settings.milling_channel)
        self.set("current", mill_settings.milling_current, mill_settings.milling_channel)
        self.set("voltage", mill_settings.milling_voltage, mill_settings.milling_channel)

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
        elif system == "stage_tilt":
            return self.system.stage.tilt
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
        elif system == "stage_rotation":
            self.system.stage.rotation = value
        elif system == "stage_tilt":
            self.system.stage.tilt = value
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

    def apply_configuration(self, system_settings: Optional[SystemSettings] = None) -> None:
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

        if self.is_available("manipulator"):
            self.system.manipulator = system_settings.manipulator

        if self.is_available("gis"):
            self.system.gis = system_settings.gis

        # dont update info -> read only
        logging.info("Microscope configuration applied.")
        logging.debug({"msg": "apply_configuration", "system_settings": system_settings.to_dict()})

    @abstractmethod
    def check_available_values(self, key:str, values, beam_type: Optional[BeamType] = None) -> bool:
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
        """"Pump the chamber."""
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

    def set_reduced_area_scanning_mode(self, reduced_area: FibsemRectangle, beam_type: BeamType) -> None:
        """Set the reduced area scanning mode for the specified beam type."""
        self.set("reduced_area", reduced_area, beam_type)
        return

    def set_full_frame_scanning_mode(self, beam_type: BeamType) -> None:
        """Set the full frame scanning mode for the specified beam type."""
        self.set("full_frame", None, beam_type)
        return

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

    def set_resolution(self, resolution: Tuple[int, int], beam_type: BeamType) -> List[int]:
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
        return FibsemStagePosition(x=0, y=0) # assume no offset to rotation centre

    def _get_compucentric_rotation_position(self, position: FibsemStagePosition) -> FibsemStagePosition:
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

    def get_target_position(self, stage_position: FibsemStagePosition, target_orientation: str) -> FibsemStagePosition:
        """Convert the stage position to the target position for the given orientation."""

        currrent_orientation = self.get_stage_orientation(stage_position)
        logging.info(f"Getting target position for {target_orientation} from {currrent_orientation}")

        if currrent_orientation == target_orientation:
            return stage_position

        if currrent_orientation == "NONE":
            raise ValueError("Unknown orientation. Cannot convert stage position.")

        stage_position = deepcopy(stage_position)
        orientation = self.get_orientation(target_orientation)

        if currrent_orientation in ["SEM", "MILLING"] and target_orientation == "FIB":
            # Convert from SEM/MILLING to FIB
            target_position = self._get_compucentric_rotation_position(stage_position)
            target_position.r = orientation.r
            target_position.t = orientation.t

        elif currrent_orientation == "FIB" and target_orientation in ["SEM", "MILLING"]:
            # Convert from FIB to SEM/MILLING
            target_position = self._get_compucentric_rotation_position(stage_position)
            target_position.r = orientation.r
            target_position.t = orientation.t
        elif currrent_orientation == "SEM" and target_orientation == "MILLING":
            # Convert from SEM to MILLING
            target_position = stage_position
            target_position.r = orientation.r
            target_position.t = orientation.t
        elif currrent_orientation == "MILLING" and target_orientation == "SEM":
            # Convert from MILLING to SEM
            target_position = stage_position
            target_position.r = orientation.r
            target_position.t = orientation.t
        elif ((currrent_orientation in ["SEM", "FIB"] and target_orientation == "FM") or
              (currrent_orientation == "FM" and target_orientation in ["SEM", "FIB"])):
            if not self.stage_is_compustage:
                raise ValueError("Cannot move to FM position on non-compustage systems.")
            # Convert from FIB to FM
            target_position = stage_position
            target_position.r = orientation.r
            target_position.t = orientation.t
        else:
            raise ValueError(f"Cannot convert from {currrent_orientation} to {target_orientation}")

        return target_position

    def get_stage_orientation(self, stage_position: Optional[FibsemStagePosition] = None) -> str:
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
            raise ValueError("Stage position must have both rotation (r) and tilt (t) defined.")
        stage_rotation = stage_position.r % (2 * np.pi)
        stage_tilt = stage_position.t

        from fibsem import movement
        # TODO: also check xyz ranges?

        sem = self.get_orientation("SEM")
        fib = self.get_orientation("FIB")
        milling = self.get_orientation("MILLING")
        if sem is None or fib is None or milling is None:
            raise ValueError("SEM, FIB or MILLING orientation not defined in the system.")
        if sem.r is None or sem.t is None or fib.r is None or fib.t is None or milling.r is None or milling.t is None:
            raise ValueError("SEM, FIB or MILLING orientation must have both rotation (r) and tilt (t) defined.")

        is_sem_rotation = movement.rotation_angle_is_smaller(stage_rotation, sem.r, atol=5) # query: do we need rotation_angle_is_smaller, since we % 2pi the rotation?
        is_fib_rotation = movement.rotation_angle_is_smaller(stage_rotation, fib.r, atol=5)

        is_sem_tilt = np.isclose(stage_tilt, sem.t, atol=0.1)
        is_fib_tilt = np.isclose(stage_tilt, fib.t, atol=0.1)
        is_milling_tilt = np.radians(-45) < stage_tilt and not is_sem_tilt

        if is_sem_rotation and is_sem_tilt:
            return "SEM"
        if is_sem_rotation and is_milling_tilt:
            return "MILLING"
        if is_fib_rotation and is_fib_tilt:
            return "FIB"

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
        milling_angle = stage_settings.milling_angle        # deg

        # needs to be dynmaically updated as it can change.
        milling_stage_tilt = get_stage_tilt_from_milling_angle(self, np.radians(milling_angle))

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
                r=np.radians(stage_settings.rotation_reference),
                t=milling_stage_tilt
            ),
        }

        if self.stage_is_compustage:
            self.orientations["FIB"].r = np.radians(0)  # Compustage is always at 0 rotation
            self.orientations["FIB"].t -= np.radians(180)

            self.orientations["FM"] = FibsemStagePosition(
                r=np.radians(0),
                t=np.radians(-180),
            )
        else:
            # only x/y translation, no rotation
            self.orientations["FM"] = deepcopy(self.orientations["FIB"])

    def set_milling_angle(self, milling_angle: float) -> None:
        """Set the 'stored' milling angle in the system settings."""
        self.system.stage.milling_angle = milling_angle
        self._update_orientations()

    def get_current_milling_angle(self, stage_position: Optional[FibsemStagePosition] = None) -> float:
        """Get the current milling angle in degrees based on the current stage tilt."""

        from fibsem.transformations import convert_stage_tilt_to_milling_angle

        if stage_position is None:
            stage_position = self.get_stage_position()

        # NOTE: this is only valid for sem orientation
        if self.get_stage_orientation(stage_position=stage_position) == "FIB":
            return 90  # stage-tilt + pre-tilt + 90 - column-tilt

        stage_tilt = stage_position.t

        if stage_tilt is None:
            raise ValueError("Stage tilt is not available. Cannot calculate milling angle.")
        
        if self.stage_is_compustage and stage_tilt < np.radians(-90):
            # Compustage stage tilt is inverted, so we need to adjust the angle
            stage_tilt += np.radians(180)

        # Calculate the milling angle from the stage tilt
        milling_angle = convert_stage_tilt_to_milling_angle(
            stage_tilt=stage_tilt, 
            pretilt=np.radians(self.system.stage.shuttle_pre_tilt), 
            column_tilt=np.radians(self.system.ion.column_tilt)
        )
        return float(np.degrees(milling_angle))

    def is_close_to_milling_angle(self, milling_angle: float, atol: float = 2.0) -> bool:
        """Check if the current milling angle is close to the specified milling angle.
        Args:
            milling_angle (float): The target milling angle in degrees.
            atol (float): The absolute tolerance for the comparison.
        Returns:
            bool: True if the current milling angle is close to the specified milling angle, False otherwise
        """
        current_milling_angle = self.get_current_milling_angle() # degrees

        return bool(np.isclose(current_milling_angle, milling_angle, atol=atol))

    def move_to_milling_angle(self,milling_angle: float, rotation: Optional[float] = None) -> bool:
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

        return self.is_close_to_milling_angle(milling_angle)

    def move_to_device(self, device: str) -> None:
        """Move the stage to the predefined device position."""
        logging.warning(f"move_to_device is not implemented for {self.__class__.__name__}.")
        pass

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
        return "ThermoFisher"



def __getattr__(name: str):
    if name == "ThermoMicroscope":
        from fibsem.microscopes.autoscript import ThermoMicroscope
        globals()[name] = ThermoMicroscope
        return ThermoMicroscope
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
