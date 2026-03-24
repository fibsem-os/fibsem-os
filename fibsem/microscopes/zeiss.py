"""
ZeissMicroscope — fibsem-os wrapper for the Zeiss SmartSEM / CrossBeam system.

Migrated from SerialFIB's CrossbeamDriver (Klumpe, Goetz, Fung et al.,
Max-Planck-Institute for Biochemistry / EMBL Heidelberg).

The Zeiss SDK (crossbeam_client / SEM_API) is vendored under
fibsem/microscopes/zeiss_api/ and requires:
  - Windows OS
  - SmartSEM installed with COM server CZ.EMApiCtrl.1 registered
  - (optional) SmartFIB for patterning
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
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
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MillingState,
    MicroscopeState,
    Point,
    SystemSettings,
)

# ---------------------------------------------------------------------------
# Optional import guard — allows fibsem-os to load on non-Zeiss machines
# ---------------------------------------------------------------------------

ZEISS_API_AVAILABLE = False
try:
    from fibsem.microscopes.zeiss_api import (  # noqa: F401
        MicroscopeClient,
        GrabFrameSettings,
        ZeissPoint,
    )
    from fibsem.microscopes.zeiss_api.tiff_handle import read_tiff

    ZEISS_API_AVAILABLE = True
except Exception as exc:
    logging.debug(f"zeiss_api not available — Zeiss microscope support disabled. ({exc})")

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _zeiss_stage_to_fibsem(pos_tuple) -> FibsemStagePosition:
    """Convert Zeiss stage position tuple (x, y, z, t, r) to FibsemStagePosition.

    Zeiss reports positions in metres and degrees; fibsem uses metres and
    degrees consistently (the base class converts to radians where needed).
    """
    x, y, z, t, r = float(pos_tuple[0]), float(pos_tuple[1]), float(pos_tuple[2]), float(pos_tuple[3]), float(pos_tuple[4])
    return FibsemStagePosition(x=x, y=y, z=z, t=t, r=r)


def _fibsem_to_zeiss_stage(pos: FibsemStagePosition) -> tuple:
    """Convert FibsemStagePosition to Zeiss (x, y, z, t, r) tuple."""
    return (pos.x, pos.y, pos.z, pos.t, pos.r)


def _zeiss_image_to_fibsem(
    img_array: np.ndarray,
    pixel_size_m: float,
    image_settings: Optional[ImageSettings],
    beam_type: BeamType,
    state: Optional[MicroscopeState] = None,
) -> FibsemImage:
    """Wrap a numpy array read from a Zeiss TIFF into a FibsemImage."""

    h, w = img_array.shape[:2]
    hfw = pixel_size_m * w if pixel_size_m else 100e-6

    if image_settings is None:
        image_settings = ImageSettings(
            resolution=(w, h),
            dwell_time=100e-9,
            hfw=hfw,
            beam_type=beam_type,
        )

    pixel_size = Point(x=pixel_size_m or 1e-9, y=pixel_size_m or 1e-9)

    metadata = FibsemImageMetadata(
        image_settings=image_settings,
        pixel_size=pixel_size,
        microscope_state=state or MicroscopeState(),
    )

    # Zeiss images can be RGB; take first channel / convert to 2-D
    if img_array.ndim == 3:
        img_array = img_array[:, :, 0]

    return FibsemImage(data=img_array, metadata=metadata)


# ---------------------------------------------------------------------------
# Resolutions available on Zeiss CrossBeam
# ---------------------------------------------------------------------------

ZEISS_RESOLUTIONS = [
    "512x384",
    "768x512",
    "1024x768",
    "2048x1536",
    "3072x2304",
    "4096x3072",
]

ZEISS_DEFAULT_RESOLUTION = "1024x768"


# ---------------------------------------------------------------------------
# ZeissMicroscope
# ---------------------------------------------------------------------------


class ZeissMicroscope(FibsemMicroscope):
    """fibsem-os adapter for the Zeiss SmartSEM / CrossBeam microscope.

    This is an initial thin-wrapper release that exposes the operations
    implemented in SerialFIB's CrossbeamDriver.  Methods without a Zeiss
    equivalent raise ``NotImplementedError``.
    """

    def __init__(self, system_settings: SystemSettings):
        if not ZEISS_API_AVAILABLE:
            raise ImportError(
                "zeiss_api is not available. "
                "Ensure the Zeiss SDK (crossbeam_client/SEM_API) is installed and "
                "that fibsem/microscopes/zeiss_api/ is present."
            )

        self.connection: "MicroscopeClient" = MicroscopeClient()
        self.system: SystemSettings = system_settings

        # Path where grab_frame() saves the TIFF on disk (SmartSEM default)
        self._api_grab_path: str = "C:/api/Grab.tif"

        # In-memory cache of the last acquired image per beam
        self._last_image: Dict[BeamType, FibsemImage] = {}

        # Minimal beam-state caches (populated on connect)
        self._beam_current: Dict[BeamType, float] = {
            BeamType.ELECTRON: 1e-9,
            BeamType.ION: 1e-10,
        }

        logging.info("ZeissMicroscope created. Call connect_to_microscope() to connect.")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_to_microscope(self, ip_address: str, port: int = 8080) -> None:
        """Connect to the Zeiss SmartSEM COM server.

        ``ip_address`` is accepted for API compatibility but the Zeiss
        SmartSEM COM interface (CZ.EMApiCtrl.1) connects to the locally
        running server, so the address is not used directly.
        """
        logging.info(f"Connecting to Zeiss microscope (ip_address={ip_address!r} ignored for COM connection).")
        self.connection.connect()
        logging.info("Zeiss microscope connected.")

    def disconnect(self) -> None:
        self.connection.disconnect()
        logging.info("Zeiss microscope disconnected.")

    # ------------------------------------------------------------------
    # Imaging
    # ------------------------------------------------------------------

    def acquire_image(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:

        if beam_type is None:
            beam_type = image_settings.beam_type if image_settings is not None else BeamType.ELECTRON

        beam_str = "ION" if beam_type == BeamType.ION else "ELECTRON"
        self.connection.beams.change_beam(beam_str)

        # Build GrabFrameSettings from ImageSettings (or use defaults)
        if image_settings is not None:
            res_w, res_h = image_settings.resolution
            res_str = f"{res_w}x{res_h}"
            dwell = image_settings.dwell_time
        else:
            res_str = ZEISS_DEFAULT_RESOLUTION
            dwell = 100e-9

        grab_settings = GrabFrameSettings(
            dwell_time=dwell,
            resolution=res_str,
            line_integration=1,
        )

        self.connection.imaging.grab_frame(grab_settings)

        # Read the TIFF that SmartSEM wrote to disk
        img_array, pixel_size = read_tiff(self._api_grab_path)
        if pixel_size is None:
            logging.warning("Could not read pixel size from TIFF; using 1 nm fallback.")
            pixel_size = 1e-9

        state = self._get_microscope_state()
        image = _zeiss_image_to_fibsem(img_array, pixel_size, image_settings, beam_type, state)
        image.metadata.system = self.system

        self._last_image[beam_type] = image
        logging.info(f"Acquired image: beam_type={beam_type.name}, shape={img_array.shape}, pixel_size={pixel_size:.3e} m")
        return image

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        if beam_type not in self._last_image:
            raise RuntimeError(f"No image cached for {beam_type.name}. Call acquire_image() first.")
        return self._last_image[beam_type]

    def acquire_chamber_image(self) -> FibsemImage:
        raise NotImplementedError("acquire_chamber_image is not implemented for ZeissMicroscope.")

    # ------------------------------------------------------------------
    # Beam auto-functions
    # ------------------------------------------------------------------

    def autocontrast(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        logging.info(f"autocontrast: beam_type={beam_type.name}")
        self.connection.auto_functions.run_auto_cb()

    def auto_focus(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        logging.info(f"auto_focus: beam_type={beam_type.name}")
        self.connection.auto_functions.run_auto_focus()

    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> Point:
        logging.info(f"beam_shift: dx={dx}, dy={dy}, beam_type={beam_type.name}")
        z_point = ZeissPoint(dx, dy)
        if beam_type == BeamType.ION:
            self.connection.beams.ion_beam.beam_shift.value = z_point
        else:
            self.connection.beams.electron_beam.beam_shift.value = z_point
        return Point(x=dx, y=dy)

    # ------------------------------------------------------------------
    # Stage movement
    # ------------------------------------------------------------------

    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        logging.info(f"move_stage_absolute: {position}")
        stage_tuple = _fibsem_to_zeiss_stage(position)
        self.connection.specimen.stage.absolute_move(stage_tuple)
        return self.get_stage_position()

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        logging.info(f"move_stage_relative: {position}")
        stage_tuple = _fibsem_to_zeiss_stage(position)
        self.connection.specimen.stage.relative_move(stage_tuple)
        return self.get_stage_position()

    def stable_move(self, dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        raise NotImplementedError("stable_move is not yet implemented for ZeissMicroscope.")

    def vertical_move(self, dy: float, dx: float = 0, static_wd: bool = True) -> FibsemStagePosition:
        raise NotImplementedError("vertical_move is not yet implemented for ZeissMicroscope.")

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        raise NotImplementedError("project_stable_move is not yet implemented for ZeissMicroscope.")

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        """Move to an absolute position.

        Safety sequencing (tilt-to-zero before large XY moves etc.) is not
        yet implemented — this delegates directly to move_stage_absolute.
        """
        logging.warning("safe_absolute_stage_movement: safety sequencing not implemented, delegating to move_stage_absolute.")
        self.move_stage_absolute(position)

    # ------------------------------------------------------------------
    # Manipulator — not available on Zeiss CrossBeam
    # ------------------------------------------------------------------

    def insert_manipulator(self, name: str) -> None:
        raise NotImplementedError("Manipulator not available on ZeissMicroscope.")

    def retract_manipulator(self) -> None:
        raise NotImplementedError("Manipulator not available on ZeissMicroscope.")

    def move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
        raise NotImplementedError("Manipulator not available on ZeissMicroscope.")

    def move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
        raise NotImplementedError("Manipulator not available on ZeissMicroscope.")

    def move_manipulator_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
        raise NotImplementedError("Manipulator not available on ZeissMicroscope.")

    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str) -> None:
        raise NotImplementedError("Manipulator not available on ZeissMicroscope.")

    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        raise NotImplementedError("Manipulator not available on ZeissMicroscope.")

    # ------------------------------------------------------------------
    # Milling
    # ------------------------------------------------------------------

    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        """Prepare for milling by setting the ion beam current."""
        logging.info(f"setup_milling: current={mill_settings.milling_current:.3e} A")
        self.connection.beams.ion_beam.beam_current.value = mill_settings.milling_current
        self._beam_current[BeamType.ION] = mill_settings.milling_current

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False) -> None:
        """Start milling.

        Pattern drawing (draw_rectangle, draw_line etc.) is not yet
        implemented — patterns must be pre-loaded via the SmartFIB / SEM UI,
        or via ``patterning.load_pattern()`` directly.
        """
        logging.info(f"run_milling: current={milling_current:.3e} A, asynch={asynch}")
        self.connection.beams.ion_beam.beam_current.value = milling_current
        self.connection.beams.change_beam("ION")
        self.start_milling()
        if not asynch:
            self._wait_for_milling_complete()

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        """Restore ion beam to imaging current after milling."""
        logging.info(f"finish_milling: restoring current to {imaging_current:.3e} A")
        self.connection.beams.ion_beam.beam_current.value = imaging_current
        self._beam_current[BeamType.ION] = imaging_current

    def clear_patterns(self) -> None:
        self.connection._patterning.clear_patterns()

    def start_milling(self) -> None:
        self.connection._patterning.start()

    def stop_milling(self) -> None:
        self.connection._patterning.stop()

    def pause_milling(self) -> None:
        raise NotImplementedError("pause_milling is not yet implemented for ZeissMicroscope.")

    def resume_milling(self) -> None:
        raise NotImplementedError("resume_milling is not yet implemented for ZeissMicroscope.")

    def get_milling_state(self) -> MillingState:
        is_idle = self.connection._patterning.is_idle
        return MillingState.IDLE if is_idle else MillingState.RUNNING

    def estimate_milling_time(self) -> float:
        raise NotImplementedError("estimate_milling_time is not yet implemented for ZeissMicroscope.")

    def _wait_for_milling_complete(self, poll_interval: float = 0.5) -> None:
        """Block until patterning reports idle."""
        logging.info("Waiting for milling to complete...")
        while not self.connection._patterning.is_idle:
            time.sleep(poll_interval)
        logging.info("Milling complete.")

    # ------------------------------------------------------------------
    # Pattern drawing — not yet implemented
    # The Zeiss patterning API uses ELY/PTF files (SmartFIB format).
    # Conversion from FibsemPatternSettings to ELY format is deferred.
    # ------------------------------------------------------------------

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        raise NotImplementedError(
            "draw_rectangle is not yet implemented for ZeissMicroscope. "
            "Zeiss uses ELY/PTF pattern files — conversion from FibsemRectangleSettings is deferred."
        )

    def draw_line(self, pattern_settings: FibsemLineSettings):
        raise NotImplementedError("draw_line is not yet implemented for ZeissMicroscope.")

    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        raise NotImplementedError("draw_circle is not yet implemented for ZeissMicroscope.")

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings) -> None:
        raise NotImplementedError("draw_bitmap_pattern is not yet implemented for ZeissMicroscope.")

    def draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        raise NotImplementedError("draw_polygon is not yet implemented for ZeissMicroscope.")

    # ------------------------------------------------------------------
    # GIS / Sputter — not available
    # ------------------------------------------------------------------

    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        raise NotImplementedError("cryo_deposition_v2 is not available on ZeissMicroscope.")

    def setup_sputter(self, *args, **kwargs):
        raise NotImplementedError("setup_sputter is not available on ZeissMicroscope.")

    def draw_sputter_pattern(self, *args, **kwargs) -> None:
        raise NotImplementedError("draw_sputter_pattern is not available on ZeissMicroscope.")

    def run_sputter(self, *args, **kwargs):
        raise NotImplementedError("run_sputter is not available on ZeissMicroscope.")

    def finish_sputter(self):
        raise NotImplementedError("finish_sputter is not available on ZeissMicroscope.")

    # ------------------------------------------------------------------
    # Property get/set dispatch
    # ------------------------------------------------------------------

    def get_available_values(self, key: str, beam_type: Optional[BeamType] = None) -> List[Union[str, float, int]]:
        if key == "resolution":
            return ZEISS_RESOLUTIONS
        if key == "detector_type":
            return []  # dynamic, requires querying SEM_API.get_detector_list()
        if key == "detector_mode":
            return []
        return []

    def check_available_values(self, key: str, values, beam_type: Optional[BeamType] = None) -> bool:
        available = self.get_available_values(key, beam_type)
        if not available:
            return True  # unknown key — allow
        return all(v in available for v in (values if isinstance(values, list) else [values]))

    def _get_beam_obj(self, beam_type: BeamType):
        if beam_type == BeamType.ION:
            return self.connection.beams.ion_beam
        return self.connection.beams.electron_beam

    def _get(self, key: str, beam_type: Optional[BeamType] = None) -> Union[float, int, bool, str, list, FibsemStagePosition]:
        """Read a property from the microscope."""

        # Stage
        if key == "stage_position":
            pos_tuple = self.connection.specimen.stage.current_position
            return _zeiss_stage_to_fibsem(pos_tuple)

        if beam_type is None:
            raise ValueError(f"beam_type is required for key '{key}'")

        beam = self._get_beam_obj(beam_type)

        if key == "current":
            return beam.beam_current.value
        if key == "hfw":
            return beam.horizontal_field_width.value
        if key == "resolution":
            return beam.scanning.resolution.value
        if key == "dwell_time":
            return beam.scanning.dwell_time.value
        if key == "scan_rotation":
            return beam.scanning.rotation.value
        if key == "shift":
            bs = beam.beam_shift.value
            return Point(x=bs.x, y=bs.y)

        # Properties not exposed by crossbeam_client — return None for read
        if key in ("working_distance", "voltage", "stigmation"):
            logging.debug(f"_get: key '{key}' is not exposed by the Zeiss crossbeam_client API; returning None.")
            return None

        # Detector — not wired up in crossbeam_client
        if key in ("detector_type", "detector_mode", "detector_brightness", "detector_contrast"):
            logging.debug(f"_get: detector key '{key}' is not exposed by the Zeiss crossbeam_client API; returning None.")
            return None

        # Beam on/blanked state
        if key == "on":
            return True  # crossbeam_client has no read-back for beam on/off
        if key == "blanked":
            return beam.is_blanked

        raise ValueError(f"Unknown key for ZeissMicroscope._get: '{key}'")

    def _set(self, key: str, value, beam_type: Optional[BeamType] = None) -> None:
        """Write a property to the microscope."""

        if beam_type is None and key != "stage_position":
            raise ValueError(f"beam_type is required for key '{key}'")

        if key == "stage_position":
            self.move_stage_absolute(value)
            return

        beam = self._get_beam_obj(beam_type)

        if key == "current":
            beam.beam_current.value = value
            self._beam_current[beam_type] = value
            return
        if key == "hfw":
            self.connection.imaging.set_field_width(value)
            return
        if key == "resolution":
            beam.scanning.resolution.value = value
            return
        if key == "dwell_time":
            beam.scanning.dwell_time.value = value
            return
        if key == "scan_rotation":
            beam.scanning.rotation.value = value
            return
        if key == "shift":
            z_point = ZeissPoint(value.x, value.y)
            beam.beam_shift.value = z_point
            return
        if key == "blanked":
            beam.is_blanked = value
            return

        # Not wired in crossbeam_client
        if key in ("working_distance", "voltage", "stigmation",
                   "detector_type", "detector_mode", "detector_brightness", "detector_contrast",
                   "on"):
            logging.warning(f"_set: key '{key}' is not exposed by the Zeiss crossbeam_client API; ignored.")
            return

        raise ValueError(f"Unknown key for ZeissMicroscope._set: '{key}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_microscope_state(self) -> MicroscopeState:
        """Snapshot the current microscope state for image metadata."""
        try:
            pos_tuple = self.connection.specimen.stage.current_position
            stage_pos = _zeiss_stage_to_fibsem(pos_tuple)
        except Exception:
            stage_pos = FibsemStagePosition()

        electron_beam = BeamSettings(
            beam_type=BeamType.ELECTRON,
            beam_current=self._beam_current[BeamType.ELECTRON],
        )
        ion_beam = BeamSettings(
            beam_type=BeamType.ION,
            beam_current=self._beam_current[BeamType.ION],
        )

        return MicroscopeState(
            stage_position=stage_pos,
            electron_beam=electron_beam,
            ion_beam=ion_beam,
        )
