"""AutoScript (ThermoFisher) specific conversion utilities.

This module contains all ThermoFisher AutoScript-specific conversion functions,
isolated from the general fibsem data structures.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional, Union
from fibsem.microscopes._stage import SampleGridLoader, SampleHolder, Stage

if TYPE_CHECKING:
    from fibsem.microscope import ThermoMicroscope
    from fibsem.structures import (
        BeamType,
        FibsemImage,
        FibsemManipulatorPosition,
        FibsemStagePosition,
        ImageSettings,
        MicroscopeState,
    )

THERMO_API_AVAILABLE = False

try:
    sys.path.append(r"C:\Program Files\Thermo Scientific AutoScript")
    sys.path.append(r"C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages")
    sys.path.append(r"C:\Program Files\Python36\envs\AutoScript")
    sys.path.append(r"C:\Program Files\Python36\envs\AutoScript\Lib\site-packages")
    from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        CompustagePosition,
        ManipulatorPosition,
        StagePosition,
    )
    THERMO_API_AVAILABLE = True
except ImportError:
    pass


def stage_position_to_autoscript(
    position: "FibsemStagePosition", compustage: bool = False
) -> Union["StagePosition", "CompustagePosition"]:
    """Convert a FibsemStagePosition to an AutoScript StagePosition or CompustagePosition.

    Args:
        position: The FibsemStagePosition to convert.
        compustage: Whether the stage is a compustage.

    Returns:
        StagePosition or CompustagePosition compatible with AutoScript.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError("AutoScript libraries not available. Cannot convert to AutoScript position.")

    if compustage:
        return CompustagePosition(
            x=position.x,
            y=position.y,
            z=position.z,
            a=position.t,
            coordinate_system=CoordinateSystem.SPECIMEN,
        )
    else:
        return StagePosition(
            x=position.x,
            y=position.y,
            z=position.z,
            r=position.r,
            t=position.t,
            coordinate_system=CoordinateSystem.RAW,
        )


def stage_position_from_autoscript(
    position: Union["StagePosition", "CompustagePosition"],
) -> "FibsemStagePosition":
    """Create a FibsemStagePosition from an AutoScript position object.

    Args:
        position: AutoScript StagePosition or CompustagePosition.

    Returns:
        FibsemStagePosition: Converted position.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError("AutoScript libraries not available. Cannot convert from AutoScript position.")

    from fibsem.structures import FibsemStagePosition

    if isinstance(position, CompustagePosition):
        return FibsemStagePosition(
            x=position.x,
            y=position.y,
            z=position.z,
            r=0.0,
            t=position.a,
            coordinate_system=CoordinateSystem.SPECIMEN.upper(),
        )

    return FibsemStagePosition(
        x=position.x,
        y=position.y,
        z=position.z,
        r=position.r,
        t=position.t,
        coordinate_system=position.coordinate_system.upper(),
    )


def manipulator_position_to_autoscript(
    position: "FibsemManipulatorPosition",
) -> "ManipulatorPosition":
    """Convert a FibsemManipulatorPosition to an AutoScript ManipulatorPosition.

    Args:
        position: The FibsemManipulatorPosition to convert.

    Returns:
        ManipulatorPosition compatible with AutoScript.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError("AutoScript libraries not available. Cannot convert to AutoScript position.")

    if position.coordinate_system == "RAW":
        coordinate_system = "Raw"
    elif position.coordinate_system == "STAGE":
        coordinate_system = "Stage"
    else:
        coordinate_system = position.coordinate_system

    return ManipulatorPosition(
        x=position.x,
        y=position.y,
        z=position.z,
        r=None,
        coordinate_system=coordinate_system,
    )


def manipulator_position_from_autoscript(
    position: "ManipulatorPosition",
) -> "FibsemManipulatorPosition":
    """Create a FibsemManipulatorPosition from an AutoScript ManipulatorPosition.

    Args:
        position: AutoScript ManipulatorPosition.

    Returns:
        FibsemManipulatorPosition: Converted position.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError("AutoScript libraries not available. Cannot convert from AutoScript position.")

    from fibsem.structures import FibsemManipulatorPosition

    return FibsemManipulatorPosition(
        x=position.x,
        y=position.y,
        z=position.z,
        coordinate_system=position.coordinate_system.upper(),
    )


def image_settings_from_adorned_image(
    image: "AdornedImage",
    beam_type: Optional["BeamType"] = None,
) -> "ImageSettings":
    """Create ImageSettings from an AutoScript AdornedImage.

    Args:
        image: AutoScript AdornedImage.
        beam_type: Beam type for the image settings.

    Returns:
        ImageSettings: Converted image settings.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError("AutoScript libraries not available. Cannot convert from AdornedImage.")

    from fibsem.structures import BeamType, ImageSettings
    from fibsem.utils import current_timestamp

    if beam_type is None:
        beam_type = BeamType.ELECTRON

    return ImageSettings(
        resolution=(image.width, image.height),
        dwell_time=image.metadata.scan_settings.dwell_time,
        hfw=image.width * image.metadata.binary_result.pixel_size.x,
        autocontrast=True,
        beam_type=beam_type,
        autogamma=True,
        save=False,
        path="path",
        filename=current_timestamp(),
        reduced_area=None,
    )


def fibsem_image_from_adorned_image(
    adorned: "AdornedImage",
    image_settings: Optional["ImageSettings"] = None,
    state: Optional["MicroscopeState"] = None,
    beam_type: Optional["BeamType"] = None,
) -> "FibsemImage":
    """Create a FibsemImage from an AutoScript AdornedImage.

    Args:
        adorned: AutoScript AdornedImage.
        image_settings: Image settings. Defaults to None (derived from adorned).
        state: Microscope state. Defaults to None (derived from adorned).
        beam_type: Beam type for the image. Defaults to BeamType.ELECTRON.

    Returns:
        FibsemImage: Converted image.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError("AutoScript libraries not available. Cannot convert from AdornedImage.")

    from fibsem.structures import (
        BeamSettings,
        BeamType,
        FibsemImage,
        FibsemImageMetadata,
        FibsemStagePosition,
        MicroscopeState,
        Point,
    )

    if beam_type is None:
        beam_type = BeamType.ELECTRON

    if state is None:
        state = MicroscopeState(
            timestamp=adorned.metadata.acquisition.acquisition_datetime,
            stage_position=FibsemStagePosition(
                adorned.metadata.stage_settings.stage_position.x,
                adorned.metadata.stage_settings.stage_position.y,
                adorned.metadata.stage_settings.stage_position.z,
                adorned.metadata.stage_settings.stage_position.r,
                adorned.metadata.stage_settings.stage_position.t,
            ),
            electron_beam=BeamSettings(beam_type=BeamType.ELECTRON),
            ion_beam=BeamSettings(beam_type=BeamType.ION),
        )
    else:
        state.timestamp = adorned.metadata.acquisition.acquisition_datetime

    if image_settings is None:
        image_settings = image_settings_from_adorned_image(adorned, beam_type)

    pixel_size = Point(
        adorned.metadata.binary_result.pixel_size.x,
        adorned.metadata.binary_result.pixel_size.y,
    )

    metadata = FibsemImageMetadata(
        image_settings=image_settings,
        pixel_size=pixel_size,
        microscope_state=state,
    )
    return FibsemImage(data=adorned.data, metadata=metadata)


class AutoscriptManipulator:
    """Manipulator interface for AutoScript-based microscopes."""

    def __init__(
        self,
        parent: "ThermoMicroscope",
    ) -> None:
        self.parent = parent

    def __repr__(self) -> str:
        return f"<Manipulator: position={self.position}>"

    @property
    def position(self) -> FibsemManipulatorPosition:
        return self.parent.get_manipulator_position()

    def insert(self) -> None:
        """Insert the manipulator."""
        self.parent.insert_manipulator()

    def retract(self) -> None:
        """Retract the manipulator."""
        self.parent.retract_manipulator()

    def move_absolute(self, position: FibsemManipulatorPosition) -> FibsemManipulatorPosition:
        pass

    def move_relative(self, position: FibsemManipulatorPosition) -> FibsemManipulatorPosition:
        pass

    def move_corrected(self, dx: float, dy: float, beam_type: BeamType) -> FibsemManipulatorPosition:
        pass


class AutoscriptStage(Stage):
    """Stage interface for AutoScript-based microscopes."""

    def __init__(
        self,
        parent: "ThermoMicroscope",
        holder: SampleHolder,
        loader: Optional["SampleGridLoader"] = None,
    ) -> None:
        super().__init__(parent, holder, loader)


class AutoscriptCompustage(Stage):
    """Compustage interface for AutoScript-based microscopes."""

    def __init__(
        self,
        parent: "ThermoMicroscope",
        holder: SampleHolder,
        loader: Optional["SampleGridLoader"] = None,
    ) -> None:
        super().__init__(parent, holder, loader)


class AutoscriptSputterCoater:
    pass



import logging
import time
from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from fibsem.microscope import ThermoMicroscope
from fibsem.structures import FibsemStagePosition

class AutoscriptGISPort:
    port_name: str = "Pt dep"
    zlimit: float = 4.0e-3 # RAW_COORDINATES

    def __init__(self, parent: 'ThermoMicroscope'):
        self.parent = parent

        available_ports = self.parent.connection.gas.list_all_gis_ports()

        print(f"available gis ports: {available_ports}")
        self._port = self.parent.connection.gas.get_gis_port(self.port_name)

    def insert(self):

        self._run_safety_check()

        self._port.insert()

    def retract(self):
        self._port.retract()

    def _move_to_safe_gis_position(self):

        self.parent.move_stage_absolute(FibsemStagePosition(z=self.zlimit-500e-6))

    def _run_safety_check(self):

        stage_position = self.parent.get_stage_position()
        if stage_position.z > self.zlimit:
            raise ValueError(f"Unable to insert gis at current z-position{stage_position.pretty}, {self.zlimit*1e3}mm")

    def open(self):
        self._port.open()

    def close(self):
        self._port.close()

    @property
    def temperature(self) -> float:
        return self._port.get_temperature()

    def turn_heater_on(self, target_temp: float = 300, timeout: float = 15):
        self._port.turn_heater_on(target_temp, timeout)

    def turn_heater_off(self):
        self._port.turn_heater_off()

    def run_deposition(self, duration: int,
                       stop_event: Optional['threading.Event'] = None,
                       on_progress: Optional[Callable[[float], None]] = None) -> None:
        """Insert the GIS, open the port for ``duration`` seconds, then close + retract.

        Self-contained: the insert/open/wait/close/retract sequence lives here so
        callers (tasks, notebooks) get the full deposition cycle in one call.
        ``stop_event`` (when set) ends the wait early; ``on_progress`` is invoked
        each second with the remaining time. Close + retract always run, even on
        early stop or error.
        """
        self.insert()

        # QUERY: acquire diagnostic sem image?

        # once inserted, always close + retract — even if open/wait raises or stops
        try:
            self.open()
            start_time = time.time()
            while (time.time() - start_time) < duration:
                if stop_event is not None and stop_event.is_set():
                    logging.info(f"Deposition stopped: {self.port_name}")
                    break
                time.sleep(1)
                remaining_time = duration - (time.time() - start_time)
                if on_progress is not None:
                    on_progress(remaining_time)
                else:
                    print(f"Depositing: {self.port_name} - {remaining_time:.0f}s")
        finally:
            self.close()
            self.retract()