from __future__ import annotations

import logging
import os
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from psygnal import Signal

from fibsem import manufacturers
from fibsem.devices.beam import BEAM_ROUTES, STAGE_COMMAND_ROUTES, STAGE_ROUTES
from fibsem.devices.chamber import CHAMBER_COMMAND_ROUTES, CHAMBER_ROUTES
from fibsem.devices.entries import build_device_entries, resolve_system_devices
from fibsem.drivers.odemis import add_odemis_path
from fibsem.drivers.tescan.microscope import TescanMicroscope
from fibsem.microscope import (
    FibsemMicroscope,
    _records_beam_shift,
)
from fibsem.milling.progress import MillingProgress
from fibsem.structures import (
    BeamSettings,
    BeamType,
    DeviceEntry,
    FibsemDetectorSettings,
    FibsemExperimentRef,
    FibsemImage,
    FibsemImageMetadata,
    FibsemRectangle,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    Point,
    SystemSettings,
)

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
# The scan directions a pattern can take through the adapter.
ODEMIS_SCAN_DIRECTIONS = ("TopToBottom", "BottomToTop", "LeftToRight", "RightToLeft")

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


# The devices Odemis builds by itself. There is no other code for their keys, so
# each is required: one that cannot be built fails the connection. One the
# configuration switches off (`enabled: false`) is not built.
ODEMIS_DEVICES = (
    DeviceEntry(name="electron", type="beam", required=True),
    DeviceEntry(name="ion", type="beam", required=True),
    DeviceEntry(name="stage", type="stage", required=True),
    DeviceEntry(name="chamber", type="chamber", required=True),
)
# The FM is built on its own path; any other type a configuration adds is built
# after these.
_FM_TYPES = ("fm",)


class OdemisThermoMicroscope(FibsemMicroscope):
    """TFS integration through Odemis.
    Requires Odemis installation, unlike ThermoMicroscope which provides direct TFS integration."""

    #: An Odemis system has no manipulator: the base class's manipulator methods
    #: raise. Nothing here can ask the instrument, so this is the backend's own answer.
    DEFAULT_FITTED = {
        "manipulator": False,
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
        self.milling_channel: BeamType = BeamType.ION
        self._last_imaging_settings: ImageSettings = ImageSettings()

        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        self._build_devices()
        self._build_milling()

        self.fm = self._build_fluorescence(manufacturers.ODEMIS)
        self._apply_fluorescence_calibration()

        try:
            self._create_sample_stage()
        except Exception as e:
            logging.warning(f"Could not create sample stage: {e}")

    def _build_devices(self) -> None:
        """Build the beam, stage and chamber devices from the configuration's
        ``hardware.devices`` entries over ``ODEMIS_DEVICES``, then any other device it
        adds, and route their keys to them.

        The moves, ``home``, ``link``, ``pump`` and ``vent`` then go through the
        devices; a false ``stage_link``, ``pump_chamber`` or ``vent_chamber`` runs
        nothing. There is no other code for these keys, so a device that cannot be
        built fails the connection, and one switched off has no keys to answer.
        """
        resolved = resolve_system_devices(
            self.system,
            ODEMIS_DEVICES,
            exclude_types=_FM_TYPES,
            driver=manufacturers.ODEMIS,
        )
        built: Dict[str, Any] = build_device_entries(resolved, self)
        for name, device in built.items():
            self._set_device(name, device)
        self._bind_device_roles()

        self._beam_routes = MappingProxyType(dict(BEAM_ROUTES))
        self._device_routes = MappingProxyType(
            {
                **{key: ("stage", name) for key, name in STAGE_ROUTES.items()},
                **{key: ("chamber_device", n) for key, n in CHAMBER_ROUTES.items()},
            }
        )
        self._command_routes = MappingProxyType(
            {
                **{key: ("stage", n) for key, n in STAGE_COMMAND_ROUTES.items()},
                **{
                    key: ("chamber_device", n)
                    for key, n in CHAMBER_COMMAND_ROUTES.items()
                },
            }
        )

    def _build_milling(self) -> None:
        """Build the milling service over the beams; the milling methods then go to it
        (`FibsemMicroscope`). Without an ion beam there is none, and they raise."""
        from fibsem.drivers.odemis.services import bind_odemis_milling

        self.milling = bind_odemis_milling(self)

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

    def acquire_image(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:
        """Acquire an image with `image_settings`, or with the current settings of
        `beam_type` when that is given instead."""
        # The beam's acquire command; a beam_type takes precedence and means the
        # current settings.
        if beam_type is not None:
            return self._beam_device(beam_type).acquire(None)
        if image_settings is None:
            raise ValueError(
                "Must provide image_settings to acquire a new image if beam_type is not specified."
            )
        return self._beam_device(image_settings.beam_type).acquire(image_settings)

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        return self._beam_device(beam_type).last_image()

    def _beam_device(self, beam_type: BeamType):
        """The beam device for ``beam_type``; a column disabled in the config has none."""
        device = self.beams.get(beam_type)
        if device is None:
            raise ValueError(f"The {beam_type.name} beam is not enabled.")
        return device

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
        self._beam_device(beam_type).autocontrast(reduced_area)

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

    def move_coincident_from_sem(self, dx: float, dy: float) -> FibsemStagePosition:
        """Correct coincident point from SEM to FIB stage position.

        Deprecated: call ``vertical_move(dy, dx, beam_type=BeamType.ELECTRON)``.
        """
        return self.vertical_move(dy=dy, dx=dx, beam_type=BeamType.ELECTRON)


class OdemisTescanMicroscope(TescanMicroscope):
    """Tescan integration through Odemis.

    Currently wraps TescanMicroscope; will be extended to support Odemis-specific
    features (e.g., MicroscopePostureManager).

    Kept for an external consumer that imports it by this name, although nothing in
    fibsem uses it and the driver registry does not list it. It is the one place the
    Odemis driver imports another driver (Tescan), so keep it when moving or removing
    code here.
    """

    def __init__(self, system_settings: SystemSettings):
        super().__init__(system_settings=system_settings)
        logging.info("OdemisTescanMicroscope initialized")
