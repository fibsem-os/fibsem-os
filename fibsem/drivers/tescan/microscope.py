import logging
from typing import Optional

from fibsem.drivers.tescan import sdk
from fibsem.drivers.tescan.connection import TescanConnection
from fibsem.drivers.tescan.devices import STAGE_FRAME_TESCAN
from fibsem.drivers.tescan.services import (
    TescanStageMovement,
    bind_tescan_milling,
    bind_tescan_spot_burn,
)
from fibsem.microscope import (
    BEAM_TYPES,
    STAGE_TYPES,
    FibsemMicroscope,
)
from fibsem.structures import (  # noqa
    ACTIVE_MILLING_STATES,
    BeamSystemSettings,
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemExperimentRef,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MillingState,
    Point,
    SystemSettings,
)

FibsemStagePosition.from_tescan_stage_position = sdk.from_tescan_stage_position

# The device types each connect step builds (``FibsemMicroscope._build_devices``).
# The FM is built on its own path; any other type a configuration adds is built last.
_OWN_TYPES = BEAM_TYPES + STAGE_TYPES + ("fm",)


class TescanMicroscope(FibsemMicroscope):
    """
    A class representing a TESCAN FIB-SEM microscope.

    This class inherits from the abstract base class `FibsemMicroscope`, which defines the core functionality of a
    microscope. In addition to the methods defined in the base class, this class provides additional methods specific
    to the TESCAN FIB-SEM microscope.
    """

    driver_name = "Tescan"
    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    #: fibsem does not drive the Tescan Nanomanipulator (its support was removed), so
    #: the backend reports none and the manipulator methods are the base class's,
    #: which raise.
    DEFAULT_FITTED = {**FibsemMicroscope.DEFAULT_FITTED, "manipulator": False}

    # The SharkSEM connection, from connect until disconnect. The devices and services
    # are handed it; nothing else reaches the SDK.
    _connection: Optional[TescanConnection] = None

    def __init__(self, system_settings: SystemSettings):
        if not sdk.TESCAN_API_AVAILABLE:
            raise ImportError(
                "The TESCAN Automation API is not available. Please see the user guide for installation instructions."
            )

        # initialise system settings
        self.system: SystemSettings = system_settings
        self.milling_channel: BeamType = BeamType.ION

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        # logging
        logging.debug(
            {
                "msg": "create_microscope_client",
                "system_settings": system_settings.to_dict(),
            }
        )

    def _disconnect(self) -> None:
        self._connection.disconnect()
        self._connection = None

    def connect_to_microscope(
        self,
        ip_address: str = "localhost",
        port: int = 8300,
        reset_beam_shift: bool = True,
    ) -> None:
        """
        Connects to a microscope with the specified IP address and port.

        Args:
            ip_address: ip address of the microscope server (default: localhost).
            port: port of the microscope server (default 8300).
            reset_beam_shift: Whether to reset beam shifts on connect (default: True).
        """
        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        self._connection = TescanConnection(
            sdk.Automation(ip_address, port), ip_address, port
        )
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")

        # the beams and stage as devices, before anything below goes through them
        self._build_beams()
        self._build_stage()
        self._build_milling()
        self._build_stage_movement()
        self._build_spot_burn()
        self._build_imaging()
        # whatever else the configuration adds, such as a device on its own PC
        self._build_devices([], exclude_types=_OWN_TYPES)
        # Tescan's own driver has no FM; one on its own PC is built from its entry.
        self.fm = self._build_fluorescence(self.driver_name)
        self._apply_fluorescence_calibration()
        self._bind_device_roles()

        # the default detectors: SE, or E-T on an electron column without one
        electron = self.beams.get(BeamType.ELECTRON)
        if electron is not None:
            names = [d.name for d in electron.list_detectors()]
            self.set_detector_type("SE" if "SE" in names else "E-T", BeamType.ELECTRON)
        if BeamType.ION in self.beams:
            self.set_detector_type("SE", BeamType.ION)

        # system info: the hardware reports "TESCAN" in image headers; store the
        # canonical spelling so downstream comparisons never meet the raw form
        self.system.info.manufacturer = self.driver_name
        info = self.system.info
        logging.info(
            f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}."
        )

        # reset beam shifts
        if reset_beam_shift:
            self.reset_beam_shifts()

        # create sample stage holder (needed by UI widgets)
        try:
            self._build_sample_handling()
        except Exception as e:
            logging.warning(f"Could not build the sample handling: {e}")

        logging.debug(
            {
                "msg": "connect_to_microscope",
                "ip_address": ip_address,
                "port": port,
                "system_info": self.system.info.to_dict(),
            }
        )

    def _build_milling(self) -> None:
        """Build the milling service over the beams; the milling methods then go to it
        (`FibsemMicroscope`). Without an ion beam there is none, and they raise."""
        self.milling = bind_tescan_milling(self)

    def _build_stage_movement(self, service: Optional[type] = None) -> None:
        """Build the stage movement service, with Tescan's own coincidence move from
        the SEM view (`TescanStageMovement`)."""
        super()._build_stage_movement(service or TescanStageMovement)

    def _build_spot_burn(self) -> None:
        """Build the spot burn service over the milling service; `run_spot_burn` then
        goes to it. Without milling there is none, and the method below burns."""
        self.spot_burn = bind_tescan_spot_burn(self)

    @property
    def stage_frame(self) -> str:
        """fibsem's frame through the stage device; Tescan's own without one."""
        return self.stage.frame if self.stage is not None else STAGE_FRAME_TESCAN
