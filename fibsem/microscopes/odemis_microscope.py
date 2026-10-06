from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional

import numpy as np
from psygnal import Signal

from fibsem import manufacturers
from fibsem.devices.beam import BEAM_ROUTES, STAGE_ROUTES
from fibsem.microscope import (
    FibsemMicroscope,
    _records_beam_shift,
    _records_stage_move,
)
from fibsem.microscopes.autoscript import THERMO_VOLTAGE_CHOICES
from fibsem.microscopes.registry import DriverEntry
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.milling.progress import MillingProgress
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    BeamSettings,
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemExperimentRef,
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
    MicroscopeState,
    MillingState,
    Point,
    SystemSettings,
)


def add_odemis_path(config_path: str = "/etc/odemis.conf"):
    """Add the odemis path to the python path.

    Safe to call on machines without an odemis installation (no config file,
    or a config without DEVPATH): it simply leaves sys.path unchanged so the
    subsequent `import odemis` fails with a regular ImportError.
    """

    def parse_config(path) -> dict:
        """Parse the odemis config file and return a dict with the config values"""

        with open(path) as f:
            config = f.read()

        config = config.split("\n")
        config = [line.split("=") for line in config]
        config = {
            line[0]: line[1].replace('"', "") for line in config if len(line) == 2
        }
        return config

    try:
        config = parse_config(config_path)
    except Exception as e:
        logging.debug(f"Odemis config not available at {config_path}: {e}")
        return

    devpath = config.get("DEVPATH")
    if devpath:
        sys.path.append(f"{devpath}/odemis/src")  # dev version
    sys.path.append("/usr/lib/python3/dist-packages")  # release version + pyro4


add_odemis_path()

# Guarded like the other drivers' SDKs, so the registry can read DRIVER below on a
# computer without odemis.
try:
    from odemis import model
    from odemis.util.dataio import open_acquisition

    ODEMIS_API_AVAILABLE = True
except ImportError as e:
    ODEMIS_API_AVAILABLE = False
    logging.debug(f"Odemis not installed. {e}")

if TYPE_CHECKING:
    from odemis.driver.autoscript_client import SEM as OdemisAutoscriptClient

    from fibsem.fm.microscope import FluorescenceMicroscope


def stage_position_to_odemis_dict(position: FibsemStagePosition) -> dict:
    """Convert a FibsemStagePosition to a dict with the odemis keys"""
    pdict = position.to_dict()
    pdict.pop("name")
    pdict.pop("coordinate_system")  # no longer used in odemis
    pdict["rz"] = pdict.pop("r")
    pdict["rx"] = pdict.pop("t")

    # if any values are None, remove them
    pdict = {k: v for k, v in pdict.items() if v is not None}

    return pdict


def odemis_dict_to_stage_position(pdict: dict) -> FibsemStagePosition:
    """Convert a dict with the odemis keys to a FibsemStagePosition"""

    pdict = deepcopy(pdict)

    pdict["r"] = pdict.pop("rz")
    pdict["t"] = pdict.pop("rx")
    pdict["coordinate_system"] = pdict.get("coordinate_system", "RAW")
    return FibsemStagePosition.from_dict(pdict)


def beam_settings_from_odemis_dict(channel: str, md: dict, wd: float) -> BeamSettings:
    c2b = {"electron": BeamType.ELECTRON, "ion": BeamType.ION}

    shift = md["shift"][0]
    stigmator = md["stigmator"][0]

    return BeamSettings(
        beam_type=c2b[channel],
        working_distance=wd,
        beam_current=md["beamCurrent"][0],
        dwell_time=md["dwellTime"][0],
        voltage=md["accelVoltage"][0],
        hfw=md["horizontalFoV"][0],
        resolution=md["resolution"][0],
        scan_rotation=md["rotation"][0],
        shift=Point(shift[0], shift[1]),
        stigmation=Point(stigmator[0], stigmator[1]),
    )


def detector_settings_from_odemis_dict(md: dict) -> FibsemDetectorSettings:
    return FibsemDetectorSettings(
        type=md["type"][0],
        mode=md["mode"][0],
        brightness=md["brightness"][0],
        contrast=md["contrast"][0],
    )


def odemis_md_to_microscope_state(md) -> MicroscopeState:
    # stage position
    stage_md = md["Stage"]["position"][0]
    stage_position = FibsemStagePosition.from_odemis_dict(stage_md)

    # electron beam
    ebs = BeamSettings.from_odemis_dict(
        channel="electron",
        md=md["Electron-Beam"],
        wd=md["Electron-Focus"]["position"][0]["z"],
    )

    # ion beam
    ibs = BeamSettings.from_odemis_dict(
        channel="ion", md=md["Ion-Beam"], wd=md["Ion-Focus"]["position"][0]["z"]
    )

    # electron detector
    eds = FibsemDetectorSettings.from_odemis_dict(md["Electron-Detector"])

    # ion detector
    ids = FibsemDetectorSettings.from_odemis_dict(md["Ion-Detector"])

    ms = MicroscopeState(
        stage_position=stage_position,
        electron_beam=ebs,
        electron_detector=eds,
        ion_beam=ibs,
        ion_detector=ids,
    )

    return ms


def from_odemis_image(image: model.DataArray, path: str = None) -> FibsemImage:
    md = image.metadata
    ms = MicroscopeState.from_odemis_dict(md[model.MD_EXTRA_SETTINGS])

    # image settings
    pixel_size = Point.from_list(md[model.MD_PIXEL_SIZE])

    # TODO: this should be acq_type, but it's not saved in the metadata yet...
    d2b = {"SEM": BeamType.ELECTRON, "FIB": BeamType.ION}

    if d2b[md[model.MD_DESCRIPTION]] is BeamType.ELECTRON:
        sys_state = ms.electron_beam
    if d2b[md[model.MD_DESCRIPTION]] is BeamType.ION:
        sys_state = ms.ion_beam

    filename = None
    if path is not None:
        path = os.path.dirname(path)
        filename = os.path.basename(path)

    image_settings = ImageSettings(
        resolution=[image.shape[1], image.shape[0]],
        dwell_time=sys_state.dwell_time,
        hfw=sys_state.hfw,
        beam_type=sys_state.beam_type,
        path=path,
        filename=filename,
        save=True,
        autocontrast=False,
    )

    image_md = FibsemImageMetadata(
        image_settings=image_settings,
        pixel_size=pixel_size,
        microscope_state=ms,
        # TODO: the rest of the metadata is not saved in the odemis metadata
    )

    # get the data
    da = image.getData() if isinstance(image, model.DataArrayShadow) else image

    return FibsemImage(data=da, metadata=image_md)


def load_odemis_image(path: str) -> FibsemImage:
    """Load an odemis image from a file and convert it to a FibsemImage"""
    acq = open_acquisition(path)
    image: FibsemImage = FibsemImage.from_odemis(acq[0], path=path)
    return image


# add as class methods
FibsemStagePosition.to_odemis_dict = stage_position_to_odemis_dict
FibsemStagePosition.from_odemis_dict = odemis_dict_to_stage_position
BeamSettings.from_odemis_dict = beam_settings_from_odemis_dict
FibsemDetectorSettings.from_odemis_dict = detector_settings_from_odemis_dict
MicroscopeState.from_odemis_dict = odemis_md_to_microscope_state
FibsemImage.from_odemis = from_odemis_image
FibsemImage.load_odemis_image = load_odemis_image

beam_type_to_odemis = {
    BeamType.ELECTRON: "electron",
    BeamType.ION: "ion",
}

# Pattern settings the Delmic AutoScript adapter (xtadapter 1.16.0) does not pass on.
ODEMIS_DROPPED_PATTERN_SETTINGS = ("is_exclusion", "passes", "time")

# xT's vacuum states, as the odemis client documents them, by the names
# ThermoMicroscope reports. A state not listed here is passed through.
ODEMIS_CHAMBER_STATES = {
    "vacuum": "Pumped",
    "pumped": "Pumped",
    "vented": "Vented",
    "pumping": "Pumping",
    "venting": "Venting",
    "vacuum_error": "Error",
}

# TODO: load default system settings?


# This driver, as the registry knows it (fibsem.microscopes.registry). No port:
# Odemis reaches the instrument through its own back end.
DRIVER = DriverEntry(
    manufacturer=manufacturers.ODEMIS,
    microscope_class="fibsem.microscopes.odemis_microscope:OdemisThermoMicroscope",
)


class OdemisThermoMicroscope(FibsemMicroscope):
    """TFS integration through Odemis.
    Requires Odemis installation, unlike ThermoMicroscope which provides direct TFS integration."""

    #: An Odemis system has no manipulator, and no GIS reachable from here: the Delmic
    #: AutoScript adapter exposes no gas injection, so the GIS and sputter methods are
    #: the base class's, which raise. Nothing here can ask the instrument, so this is
    #: the backend's own answer.
    DEFAULT_FITTED = {
        "manipulator": False,
        "gis": False,
        "gis_multichem": False,
        "gis_sputter_coater": False,
    }

    milling_progress_signal = Signal(MillingProgress)
    _last_imaging_settings: ImageSettings
    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    def __init__(self, system_settings: SystemSettings):
        if not ODEMIS_API_AVAILABLE:
            raise ImportError("Odemis is not installed, so its driver cannot connect.")
        self.system: SystemSettings = system_settings

        self.connection: OdemisAutoscriptClient = model.getComponent(role="fibsem")

        # stage
        self._vendor_stage: model.Actuator = model.getComponent(role="stage-bare")

        logging.info("OdemisThermoMicroscope initialized")

        # system information # TODO: split this version info properly
        software_version = self.connection.get_software_version()
        hardware_version = self.connection.get_hardware_version()
        self.system.info.model = hardware_version
        self.system.info.serial_number = hardware_version
        self.system.info.hardware_version = software_version
        self.system.info.software_version = software_version
        info = self.system.info
        logging.info(
            f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}."
        )

        # internal parameters
        self.stage_is_compustage = False
        self.milling_channel: BeamType = BeamType.ION
        self._default_application_file: str = "Si"
        self._last_imaging_settings: ImageSettings = ImageSettings()

        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        self._build_devices()

        self.fm = None
        try:
            if (
                self._fluorescence_is_configured()
                and self._fluorescence_uses_own_driver()
            ):
                self.fm = self._connect_fluorescence_devices()
        except (ImportError, AttributeError) as e:
            logging.info(f"Fluorescence support is not available: {e}")
        except Exception as e:
            logging.warning(f"Failed to initialize fluorescence microscope: {e}")
        if self.fm is None:
            self.fm = self._connect_remote_fluorescence()
        self._apply_fluorescence_calibration()

        try:
            self._create_sample_stage()
        except Exception as e:
            logging.warning(f"Could not create sample stage: {e}")

    def _build_devices(self) -> None:
        """Build the beam and stage devices and route the beam and stage keys to them.

        The moves and ``home`` then go through the stage device. A ``stage_link`` set
        stays with ``_set``: a false value unlinks there, and the device's ``link``
        command only links. A failure leaves every key on the old code, as it was
        before the devices, and says so.
        """
        from fibsem.devices.drivers.odemis import bind_odemis_beams, bind_odemis_stage

        try:
            beams = bind_odemis_beams(self)
            stage = bind_odemis_stage(self)
        except Exception as e:
            logging.warning(
                f"Could not build the beam and stage devices, using the old code: {e}"
            )
            return
        self.beams = MappingProxyType(beams)
        self._beam_routes = MappingProxyType(dict(BEAM_ROUTES))
        self.stage = stage
        self._device_routes = MappingProxyType(
            {key: ("stage", name) for key, name in STAGE_ROUTES.items()}
        )
        self._command_routes = MappingProxyType({"stage_home": ("stage", "home")})

    def _connect_fluorescence_devices(self) -> "FluorescenceMicroscope":
        """The FM as the FM API over the Odemis FM devices, which make the odemis
        calls ``OdemisFluorescenceMicroscope`` made, on the same components and
        stream. ``fm_devices`` are those devices."""
        from fibsem.devices.drivers.odemis_fm import bind_odemis_fm
        from fibsem.fm.odemis import DeviceOdemisFluorescenceMicroscope

        devices = bind_odemis_fm(self, config=self.system.fm.to_dict())
        # The FM API's own live view pulls every frame, so nothing needs to stop it
        # when no frame is asked for.
        devices["fm"].live_timeout = None
        fm = DeviceOdemisFluorescenceMicroscope(devices, parent=self)
        self.fm_devices = MappingProxyType(dict(devices))
        return fm

    def connect_to_microscope(
        self, ip_address: str, port: int, reset_beam_shift: bool = True
    ) -> None:
        pass

    def disconnect(self):
        pass

    def set_channel(self, channel: BeamType):
        """Set the active channels for the microscope."""
        self.connection.set_active_view(channel.value)
        self.connection.set_active_device(channel.value)

    def acquire_chamber_image(self) -> FibsemImage:
        pass

    def acquire_image(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:
        """Acquire an image with `image_settings`, or with the current settings of
        `beam_type` when that is given instead."""
        if beam_type is not None:
            return self._acquire_current_image(beam_type)
        if image_settings is None:
            raise ValueError(
                "Must provide image_settings to acquire a new image if beam_type is not specified."
            )

        # TODO: migrate to updated api that allows acquiring without setting the imaging settings first
        beam_type = image_settings.beam_type
        channel = beam_type_to_odemis[beam_type]

        # reduced area imaging
        if image_settings.reduced_area is not None:
            reduced_area = image_settings.reduced_area
            self.connection.set_reduced_area_scan_mode(
                channel=channel,
                left=reduced_area.left,
                top=reduced_area.top,
                width=reduced_area.width,
                height=reduced_area.height,
            )
        else:
            self.connection.set_full_frame_scan_mode(channel=channel)

        # set imaging settings
        # TODO: this is a change in behaviour..., restore the previous conditions or use GrabFrameSettings?
        # This is the source of the error with square resolutions.
        # can't set square resolution, but can acquire an image with square
        frame_settings = None
        tmp_resolution = None
        resolution = image_settings.resolution
        if resolution[0] == resolution[1]:
            # can't set square resolution directly
            frame_settings = {"resolution": f"{resolution[0]}x{resolution[1]}"}
            tmp_resolution = resolution
            image_settings.resolution = self.get_resolution(beam_type=beam_type)
        self.set_imaging_settings(image_settings)

        # acquire image
        image, _md = self.connection.acquire_image(
            channel=channel, frame_settings=frame_settings
        )

        # restore to full frame imaging
        if image_settings.reduced_area is not None:
            self.connection.set_full_frame_scan_mode(channel=channel)

        # restore the previous resolution
        if tmp_resolution is not None:
            image_settings.resolution = tmp_resolution

        # store last imaging settings
        self._last_imaging_settings = image_settings

        return self._construct_image(image, image_settings)

    def _acquire_current_image(self, beam_type: BeamType) -> FibsemImage:
        """A frame with the beam's current settings, as ThermoMicroscope.acquire_image3."""
        image, _md = self.connection.acquire_image(
            channel=beam_type_to_odemis[beam_type], frame_settings=None
        )
        return self._construct_image(
            image, self._current_image_settings(beam_type, image)
        )

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        image = self.connection.get_last_image(channel=beam_type_to_odemis[beam_type])
        # The client is annotated as returning (image, metadata), but the AutoScript
        # adapter (1.16.0) returns the bare array.
        if isinstance(image, tuple):
            image = image[0]
        return self._construct_image(
            image, self._current_image_settings(beam_type, image)
        )

    def _current_image_settings(
        self, beam_type: BeamType, image: np.ndarray
    ) -> ImageSettings:
        """The beam's current imaging settings, at the resolution of the frame it gave."""
        image_settings = self.get_imaging_settings(beam_type)
        image_settings.resolution = [image.shape[1], image.shape[0]]
        return image_settings

    def _construct_image(
        self, data: np.ndarray, image_settings: ImageSettings
    ) -> FibsemImage:
        """A FibsemImage with the microscope's metadata, as every backend stamps it."""
        # TODO: retrieve the full image metadata from image md, rather than reconstruct
        pixel_size = image_settings.hfw / image_settings.resolution[0]
        md = FibsemImageMetadata(
            image_settings=image_settings,
            pixel_size=Point(pixel_size, pixel_size),
            microscope_state=self.get_microscope_state(
                beam_type=image_settings.beam_type
            ),
        )
        image = FibsemImage(data, md)
        self._set_additional_metadata(image)
        return image

    def autocontrast(
        self, beam_type: BeamType, reduced_area: FibsemRectangle = None
    ) -> None:
        channel = beam_type_to_odemis[beam_type]
        if reduced_area is not None:
            self.connection.set_reduced_area_scan_mode(
                channel, **reduced_area.to_dict()
            )
        self.connection.run_auto_contrast_brightness(channel=channel)
        if reduced_area is not None:
            self.connection.set_full_frame_scan_mode(channel)

    def auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        """An image-based working-distance sweep, at the beam's current field of view.

        The odemis client exposes no autofocus call (the AutoScript adapter has one,
        but odemis does not forward it), so the microscope's own focus routine is out
        of reach from here.
        """
        from fibsem.autofunctions.autofocus import AutoFocusSettings, run_auto_focus

        # TODO: restore the beam's imaging settings afterwards. acquire_image writes
        # resolution, dwell time and field of view to the beam, so the sweep leaves it
        # at its probe settings.
        run_auto_focus(
            self,
            beam_type=beam_type,
            hfw=self.get_field_of_view(beam_type),
            settings=AutoFocusSettings(reduced_area=reduced_area),
        )

    @_records_beam_shift
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:
        """Move the beam shift by dx and dy in meters (relative movement).
        Args:
            dx (float): The x shift in meters.
            dy (float): The y shift in meters.
            beam_type (BeamType): The type of beam to shift (ELECTRON or ION).
        """
        current_shift = self.get_beam_shift(beam_type=beam_type)
        new_shift = Point(x=current_shift.x + dx, y=current_shift.y + dy)
        self.set_beam_shift(new_shift, beam_type=beam_type)

    def _fluorescence_default(self) -> bool:
        """The Odemis stack drives its own FM, and has always built one unasked."""
        return True

    def _get(self, key: str, beam_type: BeamType = None) -> str:
        if beam_type is not None:
            channel = beam_type_to_odemis[beam_type]

        # beam properties
        if key == "on":
            return self.connection.get_beam_is_on(channel)
        if key == "blanked":
            return self.connection.beam_is_blanked(channel)
        if key == "working_distance":
            return self.connection.get_working_distance(channel)

        if key == "current":
            return self.connection.get_beam_current(channel)
        if key == "voltage":
            return self.connection.get_high_voltage(channel)
        if key == "hfw":
            return self.connection.get_field_of_view(channel)
        if key == "dwell_time":
            return self.connection.get_dwell_time(channel)
        if key == "scan_rotation":
            return self.connection.get_scan_rotation(channel)
        if key == "voltage_limits":
            voltage_info = self.connection.high_voltage_info(channel)
            return [voltage_info["range"][0], voltage_info["range"][1]]
        if key == "voltage_controllable":
            return True
        if key == "shift":  # beam shift
            beam_shift = self.connection.get_beam_shift(channel)
            return Point(beam_shift[0], beam_shift[1])
        if key == "stigmation":
            stigmation = self.connection.get_stigmator(channel)
            return Point(stigmation[0], stigmation[1])
        if key == "resolution":
            width, height = self.connection.get_resolution(channel)
            return [width, height]

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                return self.system.ion.plasma
            else:
                return False

        if key == "plasma_gas":
            if beam_type is BeamType.ION and self.system.ion.plasma:
                raise NotImplementedError()
            else:
                return None

        # stage properties
        if key == "stage_position":
            pdict = self._vendor_stage.position.value
            return FibsemStagePosition.from_odemis_dict(pdict)

        if key == "stage_homed":
            return self.connection.is_homed()
        if key == "stage_linked":
            return self.connection.is_linked()

        # chamber properties
        if key == "chamber_state":
            state = self.connection.get_chamber_state()
            # The AutoScript adapter passes AutoScript's names through ("Pumped",
            # "Vented"); the odemis client documents the xT names ("vacuum",
            # "vented"). Both read the way ThermoMicroscope reports them.
            return ODEMIS_CHAMBER_STATES.get(str(state).lower(), state)

        if key == "chamber_pressure":
            return self.connection.get_pressure()

        # detector mode and type
        if key == "detector_type":
            return self.connection.get_detector_type(channel)
        if key == "detector_mode":
            return self.connection.get_detector_mode(channel)
        if key == "detector_brightness":
            return self.connection.get_brightness(channel)
        if key == "detector_contrast":
            return self.connection.get_contrast(channel)

        # manipulator properties
        if key == "manipulator_position":
            raise NotImplementedError()
        if key == "manipulator_state":
            raise NotImplementedError()

        if key in ["preset"]:
            return None

        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def _set(self, key: str, value: str, beam_type: BeamType = None) -> None:
        # get beam
        if beam_type is not None:
            channel = beam_type_to_odemis[beam_type]

        # beam properties
        if key == "working_distance":
            self.connection.set_working_distance(value, channel)
            logging.info(f"{beam_type.name} working distance set to {value} m.")
            return
        if key == "current":
            self.connection.set_beam_current(value, channel)
            logging.info(f"{beam_type.name} current set to {value} A.")
            return
        if key == "voltage":
            self.connection.set_high_voltage(value, channel)
            logging.info(f"{beam_type.name} voltage set to {value} V.")
            return
        if key == "hfw":
            self.connection.set_field_of_view(value, channel)
            logging.info(f"{beam_type.name} HFW set to {value} m.")
            return
        if key == "dwell_time":
            self.connection.set_dwell_time(value, channel)
            logging.info(f"{beam_type.name} dwell time set to {value} s.")
            return
        if key == "scan_rotation":
            self.connection.set_scan_rotation(value, channel)
            logging.info(f"{beam_type.name} scan rotation set to {value} radians.")
            return
        if key == "shift":
            self.connection.set_beam_shift(value.x, value.y, channel)
            logging.info(f"{beam_type.name} shift set to {value}.")
            return
        if key == "stigmation":
            self.connection.set_stigmator(value.x, value.y, channel)
            logging.info(f"{beam_type.name} stigmation set to {value}.")
            return

        if key == "resolution":
            self.connection.set_resolution(value, channel)
            return

        # patterning
        if key == "patterning_mode":
            if value in ["Serial", "Parallel"]:
                self.connection.set_patterning_mode(value)
                logging.info(f"Patterning mode set to {value}.")
                return
        if key == "application_file":
            self.connection.set_default_application_file(value)
            logging.info(f"Default application file set to {value}.")
            return
        if key == "default_patterning_beam_type":
            channel = beam_type_to_odemis[value]
            self.connection.set_default_patterning_beam_type(channel)
            logging.info(f"Patterning beam type set to {value} - {channel} .")
            return

        # beam control
        if key == "on":
            self.connection.set_beam_power(value, channel)
            logging.info(f"{beam_type.name} beam turned {'on' if value else 'off'}.")
            return
        if key == "blanked":
            self.connection.blank_beam(
                channel
            ) if value else self.connection.unblank_beam(channel)
            logging.info(
                f"{beam_type.name} beam {'blanked' if value else 'unblanked'}."
            )
            return

        # detector properties
        if key == "detector_mode":
            if value in self.get_available_values("detector_mode", beam_type):
                self.connection.set_detector_mode(value, channel)
                logging.info(f"Detector mode set to {value}.")
            else:
                logging.warning(f"Detector mode {value} not available.")
            return
        if key == "detector_type":
            if value in self.get_available_values("detector_type", beam_type):
                self.connection.set_detector_type(value, channel)
                logging.info(f"Detector type set to {value}.")
            else:
                logging.warning(f"Detector type {value} not available.")
            return
        if key == "detector_brightness":
            if 0 < value <= 1:
                self.connection.set_brightness(value, channel)
                logging.info(f"Detector brightness set to {value}.")
            else:
                logging.warning(
                    f"Detector brightness {value} not available, must be between 0 and 1."
                )
            return
        if key == "detector_contrast":
            if 0 < value <= 1:
                self.connection.set_contrast(value, channel)
                logging.info(f"Detector contrast set to {value}.")
            else:
                logging.warning(
                    f"Detector contrast {value} not available, mut be between 0 and 1."
                )
            return

        if key == "spot_mode":
            self.connection.set_spot_scan_mode(channel=channel, x=value.x, y=value.y)
            return

        if key == "full_frame":
            self.connection.set_full_frame_scan_mode(channel)
            return

        # ion beam properties
        if beam_type is BeamType.ION:
            if key == "plasma_gas":
                if not self.system.ion.plasma:
                    logging.debug("Plasma gas cannot be set on this microscope.")
                    return
                if not self.check_available_values("plasma_gas", [value], beam_type):
                    logging.warning(
                        f"Plasma gas {value} not available. Available values: {self.get_available_values('plasma_gas', beam_type)}"
                    )

                logging.info(
                    f"Setting plasma gas to {value}... this may take some time..."
                )
                raise NotImplementedError()
                logging.info(f"Plasma gas set to {value}.")

                return

        # stage properties
        if key == "stage_home":
            logging.info("Homing stage...")
            self.connection.home_stage()
            logging.info("Stage homed.")
            return

        if key == "stage_link":
            if self.stage_is_compustage:
                logging.debug("Compustage does not support linking.")
                return

            logging.info("Linking stage...")
            self.connection.link(value)
            logging.info(f"Stage {'linked' if value else 'unlinked'}.")
            return

        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info("Pumping chamber...")
                self.connection.pump()
                logging.info("Chamber pumped.")
                return
            else:
                logging.warning(f"Invalid value for pump_chamber: {value}.")
                return

        if key == "vent_chamber":
            if value:
                logging.info("Venting chamber...")
                self.connection.vent()
                logging.info("Chamber vented.")
                return
            else:
                logging.warning(f"Invalid value for vent_chamber: {value}.")
                return

        if key == "active_view":
            self.connection.set_active_view(value.value)  # value == BeamType
            return
        if key == "active_device":
            self.connection.set_active_device(value.value)  # value == BeamType
            return

        # known keys that are not implemented
        if key in ["preset"]:
            return

        logging.warning(f"Unknown key: {key} ({beam_type})")

        return

    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        pass

    def get_available_values(self, key: str, beam_type: BeamType = None) -> list:
        values = []
        if key == "application_file":
            values = self.connection.get_available_application_files()
        if key == "scan_direction":
            values = ["TopToBottom", "BottomToTop", "LeftToRight", "RightToLeft"]
        if key == "detector_type":
            values = self.connection.detector_type_info(beam_type_to_odemis[beam_type])[
                "choices"
            ]
        if key == "detector_mode":
            values = self.connection.detector_mode_info(beam_type_to_odemis[beam_type])[
                "choices"
            ]
        if key == "current":
            # The adapter gives the ion beam's currents as choices and the electron
            # beam's as a range; a range is stepped by doubling, as ThermoMicroscope
            # does, to match the choices the microscope offers.
            info = self.connection.beam_current_info(beam_type_to_odemis[beam_type])
            if "choices" in info:
                values = list(info["choices"])
            else:
                low, high = info["range"]
                current = low
                while current <= high:
                    values.append(current)
                    current *= 2.0
        if key == "voltage":
            low, high = self.connection.high_voltage_info(
                beam_type_to_odemis[beam_type]
            )["range"]
            values = [v for v in THERMO_VOLTAGE_CHOICES[beam_type] if low <= v <= high]
        if key == "plasma_gas":
            values = ["Argon", "Oxygen", "Xenon"]

        logging.debug({"msg": "get_available_values", "key": key, "values": values})

        return values

    def check_available_values(self, key: str) -> list:
        pass

    def insert_manipulator(self) -> None:
        pass

    def move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_corrected(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> None:
        pass

    def retract_manipulator(self) -> None:
        pass

    # Through the stage device once it is built; the code below stays until a session
    # on an instrument confirms the device's moves.

    @_records_stage_move
    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        if self.stage is not None:
            return super().move_stage_absolute(position)
        pdict = stage_position_to_odemis_dict(position)
        f = self._vendor_stage.moveAbs(pdict)
        f.result()
        # TODO: implement compucentric rotation
        return self.get_stage_position()

    @_records_stage_move
    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        if self.stage is not None:
            return super().move_stage_relative(position)
        pdict = stage_position_to_odemis_dict(position)
        f = self._vendor_stage.moveRel(pdict)
        f.result()
        return self.get_stage_position()

    def move_coincident_from_sem(self, dx: float, dy: float) -> FibsemStagePosition:
        """Correct coincident point from SEM to FIB stage position.

        Deprecated: call ``vertical_move(dy, dx, beam_type=BeamType.ELECTRON)``.
        """
        return self.vertical_move(dy=dy, dx=dx, beam_type=BeamType.ELECTRON)

    # Raised rather than skipped. Drawing nothing left a stage with no patterns, which
    # never leaves IDLE, so the milling run waited on it indefinitely; raising inside
    # the milling task fails the task with this message and still restores the beams.
    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} cannot draw bitmap patterns: the Delmic "
            "AutoScript adapter has no bitmap patterning."
        )

    def draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} cannot draw polygon patterns: the Delmic "
            "AutoScript adapter has no polygon patterning."
        )

    def _warn_dropped_pattern_settings(self, pattern_settings) -> None:
        """Warn about the settings the Delmic adapter ignores (xtadapter 1.16.0).

        It creates each pattern from its geometry and depth only: a pattern meant as
        an exclusion zone is milled like any other, and a pass count or milling time
        is replaced by the microscope's own.
        """
        dropped = [
            name
            for name in ODEMIS_DROPPED_PATTERN_SETTINGS
            if getattr(pattern_settings, name, None)
        ]
        if dropped:
            logging.warning(
                f"{type(self).__name__} cannot apply {', '.join(dropped)} to "
                f"{type(pattern_settings).__name__}: the Delmic AutoScript adapter "
                "ignores them, and the pattern is drawn without."
            )

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        self._warn_dropped_pattern_settings(pattern_settings)
        pdict = pattern_settings.to_dict()

        pdict["center_x"] = pdict.pop("centre_x")
        pdict["center_y"] = pdict.pop("centre_y")

        # select the correct pattern function
        create_pattern_function = self.connection.create_rectangle
        self.connection.set_default_application_file("Si")
        if pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            create_pattern_function = self.connection.create_cleaning_cross_section
            self.connection.set_default_application_file("Si-ccs")
        if pattern_settings.cross_section is CrossSectionPattern.RegularCrossSection:
            create_pattern_function = self.connection.create_regular_cross_section
            self.connection.set_default_application_file("Si-multipass")

        # create the pattern (draw)
        pinfo = create_pattern_function(pdict)

        # restore the default application file
        self.connection.set_default_application_file(self._default_application_file)

        logging.debug(
            {
                "msg": "draw_rectangle",
                "pattern_settings": pattern_settings.to_dict(),
                "pinfo": pinfo,
            }
        )

    def draw_line(self, pattern_settings: FibsemLineSettings):
        pdict = pattern_settings.to_dict()

        self.connection.set_default_application_file("Si")

        pinfo = self.connection.create_line(pdict)

        self.connection.set_default_application_file(self._default_application_file)

        logging.debug(
            {
                "msg": "draw_line",
                "pattern_settings": pattern_settings.to_dict(),
                "pinfo": pinfo,
            }
        )

    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        self._warn_dropped_pattern_settings(pattern_settings)
        pdict = pattern_settings.to_dict()
        pdict["outer_diameter"] = 2 * pattern_settings.radius
        # an annulus, as ThermoMicroscope draws one: the adapter takes the inner
        # diameter, but this sent 0 and milled the whole disc
        pdict["inner_diameter"] = 0
        if pattern_settings.thickness != 0:
            pdict["inner_diameter"] = (
                pdict["outer_diameter"] - 2 * pattern_settings.thickness
            )
        pdict["center_x"] = pattern_settings.centre_x
        pdict["center_y"] = pattern_settings.centre_y

        self.connection.set_default_application_file("Si")

        pinfo = self.connection.create_circle(pdict)

        self.connection.set_default_application_file(self._default_application_file)

        logging.debug(
            {
                "msg": "draw_circle",
                "pattern_settings": pattern_settings.to_dict(),
                "pinfo": pinfo,
            }
        )

    def setup_milling(self, mill_settings: FibsemMillingSettings):
        self._default_application_file = mill_settings.application_file
        self.milling_channel = mill_settings.milling_channel
        self.set_milling_settings(mill_settings)
        self.clear_patterns()

        logging.debug(
            {"msg": "setup_milling", "mill_settings": mill_settings.to_dict()}
        )

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        """Restore the imaging beam, then reset the patterning mode, as ThermoMicroscope
        does: the mode persists in xT."""
        super().finish_milling(imaging_current, imaging_voltage)
        self.set_patterning_mode("Serial")

    def set_patterning_mode(self, mode: str) -> str:
        """Set the patterning mode, "Serial" or "Parallel", as ThermoMicroscope does.

        Called by `finish_milling`; without it every milling task raised in its
        cleanup.
        """
        if mode not in ("Serial", "Parallel"):
            raise ValueError(
                f"Patterning mode {mode} not supported. Supported modes: Serial, Parallel"
            )
        self.connection.set_patterning_mode(mode)
        logging.debug({"msg": "set_patterning_mode", "mode": mode})
        return mode

    def clear_patterns(self) -> None:
        self.connection.clear_patterns()

    def get_milling_state(self):
        # The patterning state is that of the active view, so the milling channel is
        # selected first, under the same lock as ThermoMicroscope.get_milling_state.
        with self._threading_lock:
            self.set_channel(self.milling_channel)
            return MillingState[self.connection.get_patterning_state().upper()]

    def start_milling(self) -> None:
        """Start the milling process."""
        if self.get_milling_state() is MillingState.IDLE:
            self.connection.start_milling()
            logging.info("Starting milling...")

    def stop_milling(self) -> None:
        """Stop the milling process."""
        if self.get_milling_state() in ACTIVE_MILLING_STATES:
            logging.info("Stopping milling...")
            self.connection.stop_milling()
            logging.info("Milling stopped.")

    def pause_milling(self) -> None:
        """Pause the milling process."""
        if self.get_milling_state() == MillingState.RUNNING:
            logging.info("Pausing milling...")
            self.connection.pause_milling()
            logging.info("Milling paused.")

    def resume_milling(self) -> None:
        """Resume the milling process."""
        if self.get_milling_state() == MillingState.PAUSED:
            logging.info("Resuming milling...")
            self.connection.resume_milling()
            logging.info("Milling resumed.")

    def estimate_milling_time(self) -> float:
        return self.connection.estimate_milling_time()


class OdemisTescanMicroscope(TescanMicroscope):
    """Tescan integration through Odemis.
    Currently wraps TescanMicroscope; will be extended
    to support Odemis-specific features (e.g., MicroscopePostureManager)."""

    def __init__(self, system_settings: SystemSettings):
        super().__init__(system_settings=system_settings)
        logging.info("OdemisTescanMicroscope initialized")
