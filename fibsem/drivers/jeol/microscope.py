import logging
from typing import Any, Dict, List, Optional, Tuple

from fibsem.devices.entries import build_device_entries, resolve_system_devices
from fibsem.drivers.jeol.connection import JeolConnection, load_sdk
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    DeviceEntry,
    FibsemExperimentRef,
    FibsemUser,
    SystemSettings,
)

# The device types each connect step builds (``JeolMicroscope._build_devices``); any
# other type a configuration adds is built last.
_BEAM_TYPES = ("beam",)
_STAGE_TYPES = ("stage",)
_OWN_TYPES = _BEAM_TYPES + _STAGE_TYPES + ("fm",)

# The detectors each column images with after connect: the SEM's in-lens detector,
# else the first it lists; the FIB's only one.
DEFAULT_SEM_DETECTOR = "SEM_LED"


class JeolMicroscope(FibsemMicroscope):
    """A JEOL FIB-SEM (the CryoLameller), through JEOL's external control clients.

    The beams, stage, milling and spot burn are JEOL's devices and services
    (``fibsem.drivers.jeol``); stage movement and imaging are the shared services.
    JEOL drives no manipulator, gas injection or sputter coater here.
    """

    driver_name = "JEOL"
    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    DEFAULT_FITTED = {**FibsemMicroscope.DEFAULT_FITTED, "manipulator": False}

    # Both clients, from connect until disconnect. The devices and services are
    # handed it; nothing else calls JEOL.
    _connection: Optional[JeolConnection] = None

    def __init__(self, system_settings: SystemSettings):
        self.system: SystemSettings = system_settings
        self.milling_channel: BeamType = BeamType.ION

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        logging.debug(
            {
                "msg": "create_microscope_client",
                "system_settings": system_settings.to_dict(),
            }
        )

    def disconnect(self) -> None:
        if self._connection is not None:
            self._connection.disconnect()
            self._connection = None

    def connect_to_microscope(
        self,
        ip_address: str = "localhost",
        port: Optional[int] = None,
        reset_beam_shift: bool = True,
        sem_port: Optional[int] = None,
        api_path: Optional[str] = None,
    ) -> None:
        """Connect JEOL's FIB client on *port* and its SEM client on *sem_port*, each
        defaulting to the driver's; the client library is imported from *api_path*,
        else the driver's, else ``FIBSEM_JEOL_API_PATH``."""
        from fibsem.drivers.jeol import DRIVER

        config = DRIVER.config
        sdk = load_sdk(api_path or config.get("api_path"))
        self._connection = JeolConnection.open(
            sdk,
            ip_address,
            port if port is not None else config["port"],
            sem_port if sem_port is not None else config["sem_port"],
        )

        self._build_beams()
        self._build_stage()
        self._build_milling()
        self._build_stage_movement()
        self._build_spot_burn()
        self._build_imaging()
        # whatever else the configuration adds, such as a device on its own PC
        self._build_devices([], exclude_types=_OWN_TYPES)
        self.fm = self._build_fluorescence(self.driver_name)
        self._apply_fluorescence_calibration()
        self._bind_device_roles()

        electron = self.beams.get(BeamType.ELECTRON)
        if electron is not None:
            names = electron.detector_type.choices or []
            if names:
                detector = (
                    DEFAULT_SEM_DETECTOR if DEFAULT_SEM_DETECTOR in names else names[0]
                )
                self.set_detector_type(detector, BeamType.ELECTRON)
        ion = self.beams.get(BeamType.ION)
        if ion is not None:
            self.set_detector_type(ion.detector_type.choices[0], BeamType.ION)

        self.system.info.manufacturer = self.driver_name
        logging.info(f"Microscope client connected to JEOL at {ip_address}.")

        if reset_beam_shift:
            self.reset_beam_shifts()

        try:
            self._build_sample_handling()
        except Exception as e:
            logging.warning(f"Could not build the sample handling: {e}")

    def _build_devices(
        self,
        defaults: List[DeviceEntry],
        types: Optional[Tuple[str, ...]] = None,
        exclude_types: Tuple[str, ...] = (),
    ) -> Dict[str, Any]:
        """Build one connect step's devices: *defaults*, with the configuration's
        ``hardware.devices`` entries of *types* over them, and put them in ``devices``."""
        resolved = resolve_system_devices(
            self.system, defaults, types, exclude_types, self.driver_name
        )
        built = build_device_entries(resolved, self)
        for name, device in built.items():
            self._set_device(name, device)
        return built

    def _build_beams(self) -> None:
        """A device per enabled column; a disabled one gets none and is never touched."""
        self._build_devices(
            [
                DeviceEntry(name="electron", type="beam"),
                DeviceEntry(name="ion", type="beam"),
            ],
            _BEAM_TYPES,
        )

    def _build_stage(self) -> None:
        self._build_devices([DeviceEntry(name="stage", type="stage")], _STAGE_TYPES)

    def _build_milling(self) -> None:
        from fibsem.drivers.jeol.services import bind_jeol_milling

        self.milling = bind_jeol_milling(self)

    def _build_stage_movement(self, service: Optional[type] = None) -> None:
        from fibsem.drivers.jeol.services import JeolStageMovement

        super()._build_stage_movement(service or JeolStageMovement)

    def _build_spot_burn(self) -> None:
        from fibsem.drivers.jeol.services import bind_jeol_spot_burn

        self.spot_burn = bind_jeol_spot_burn(self)
