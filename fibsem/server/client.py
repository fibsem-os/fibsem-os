"""
FibsemClient: network client that delegates FibsemMicroscope calls to the App
Server's ``/microscope`` routes.

Usage:
    from fibsem.server.client import FibsemClient
    from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

    m = FibsemClient(host="192.168.1.100", port=8001, token="<agent token>")
    image = m.acquire_image(BeamType.ELECTRON)
    state = m.get_microscope_state()
"""

import io
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import requests

from fibsem.devices.wire import NPY_MEDIA_TYPE, to_wire
from fibsem.server.images import TIFF_MEDIA_TYPE
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemImage,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    MillingState,
    Point,
    SystemSettings,
)

if TYPE_CHECKING:
    from fibsem.devices.core import Device
    from fibsem.drivers.remote.devices import DeviceClient


#: The services the stage moves, the milling and the focus go through.
STAGE_MOVEMENT = "stage_movement"
MILLING = "milling"
IMAGING = "imaging"

#: The App Server's instrument routes.
MICROSCOPE = "microscope"


def _beam(beam_type: BeamType) -> str:
    """A beam's device name: "electron" or "ion"."""
    return beam_type.name.lower()


class FibsemClient:
    """Network client for the App Server's microscope routes.

    Fetches system settings from the server at init
    so they are available as direct attributes, matching the FibsemMicroscope interface.
    Every call goes to a device or service route, so it needs what that route needs:
    reads the agent at Watch, writes and commands at Drive.
    """

    def __init__(
        self, host: str = "localhost", port: int = 8001, token: Optional[str] = None
    ):
        self.base_url = f"http://{host}:{port}"
        self._host, self._port, self._token = host, port, token
        self._session = requests.Session()
        if token is not None:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._device_client: Optional["DeviceClient"] = None
        self._devices: Optional[Dict[str, "Device"]] = None
        self._devices_lock = threading.Lock()
        self._fetch_system()

    def close(self) -> None:
        """Close the connection, and the device proxies' if they were built."""
        if self._device_client is not None:
            self._device_client.close()
        self._session.close()

    def __enter__(self) -> "FibsemClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _fetch_system(self) -> None:
        """Fetch and cache system settings from the server."""
        data = self._get(MICROSCOPE)
        self.system: SystemSettings = SystemSettings.from_dict(data["system"])
        # What is fitted travels beside the configuration; the server sends only
        # the manipulator.
        fitted = data.get("fitted") or {}
        if "manipulator" in fitted:
            self.system.manipulator.enabled = bool(fitted["manipulator"])

    def _get(self, endpoint: str, timeout: int = 10) -> dict:
        resp = self._session.get(f"{self.base_url}/{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, body: dict = None, timeout: int = 30) -> dict:
        resp = self._session.post(
            f"{self.base_url}/{endpoint}", json=body or {}, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _put(self, endpoint: str, body: dict, timeout: int = 30) -> dict:
        resp = self._session.put(
            f"{self.base_url}/{endpoint}", json=body, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _post_image(
        self, endpoint: str, body: dict = None, timeout: int = 60
    ) -> FibsemImage:
        resp = self._session.post(
            f"{self.base_url}/{endpoint}", json=body or {}, timeout=timeout
        )
        resp.raise_for_status()
        return FibsemImage.load(io.BytesIO(resp.content))

    # --- Health ---

    def health(self) -> dict:
        return self._get("health", timeout=5)

    # --- Image acquisition ---

    def acquire_image(
        self,
        beam_type: BeamType = BeamType.ELECTRON,
        image_settings: Optional[ImageSettings] = None,
    ) -> FibsemImage:
        """Acquire a fresh image. Pass image_settings to override current microscope settings."""
        kwargs = {} if image_settings is None else {"image_settings": image_settings}
        return self.call_command(_beam(beam_type), "acquire", **kwargs)

    def acquire_chamber_image(self) -> FibsemImage:
        return FibsemImage(data=self.call_command("chamber_camera", "acquire"))

    def autocontrast(self, beam_type: BeamType) -> None:
        self.call_command(_beam(beam_type), "autocontrast")

    def auto_focus(self, beam_type: BeamType) -> None:
        self.call_service_command(IMAGING, "auto_focus", beam_type=beam_type)

    # --- Stage movement ---

    def get_stage_position(self) -> FibsemStagePosition:
        return FibsemStagePosition.from_dict(self.get_parameter("stage", "position"))

    def get_stage_orientation(self) -> str:
        return self.call_service_query(STAGE_MOVEMENT, "stage_orientation")

    def _move(self, command: str, **kwargs: Any) -> FibsemStagePosition:
        return FibsemStagePosition.from_dict(
            self.call_service_command(STAGE_MOVEMENT, command, **kwargs)
        )

    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        return self._move("move_absolute", position=position)

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        return self._move("move_relative", delta=position)

    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemStagePosition:
        return self._move("stable_move", dx=dx, dy=dy, beam_type=beam_type)

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        result = self.call_service_query(
            STAGE_MOVEMENT,
            "project_stable_move",
            dx=dx,
            dy=dy,
            beam_type=beam_type,
            base_position=base_position,
        )
        return FibsemStagePosition.from_dict(result)

    def vertical_move(self, dy: float, dx: float = 0.0) -> FibsemStagePosition:
        return self._move("vertical_move", dy=dy, dx=dx)

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        self.call_service_command(
            STAGE_MOVEMENT, "safe_absolute_stage_movement", stage_position=position
        )

    def move_to_orientation(self, orientation: str) -> FibsemStagePosition:
        return self._move("move_to_orientation", orientation=orientation)

    # --- Devices ---
    # Every device the microscope built (``microscope.devices``), by name: the beams
    # are "electron" and "ion", then "stage", "chamber", "manipulator", FM parts...

    @property
    def devices(self) -> Mapping[str, "Device"]:
        """The server's beams and FM parts as live devices, by name, like
        ``FibsemMicroscope.devices``: ``client.devices["ion"].current.get_value()``.

        They are the remote driver's devices (``fibsem.drivers.remote.devices``)
        over this server's ``/microscope/devices`` routes, with this client's token, built on
        first access. A read is live, a write is checked here and again on the
        server, and a command runs on the server; writes and commands need the
        agent at Drive.

        The App Server has no event stream, so no change made elsewhere (the
        microscope's own UI, another client) is signalled: ``changed`` fires only
        for this client's own reads and writes, ``cached`` is as current as the
        last of them, and limits and choices as current as the last
        ``refresh_metadata()``. The client's ``disconnected`` and ``reconnected``
        never fire; a read of an unreachable server raises
        ``RemoteDeviceUnreachable``.

        Only the devices the remote driver has a type for: not the stage, chamber
        or manipulator, which ``list_devices``, ``get_parameter``,
        ``set_parameter`` and ``call_command`` reach. A beam's ``acquire`` is
        unavailable here; use ``acquire_image``.
        """
        with self._devices_lock:
            if self._devices is None:
                self._devices = self._connect_devices()
        return MappingProxyType(self._devices)

    def _connect_devices(self) -> Dict[str, "Device"]:
        from fibsem.drivers.remote.devices import (
            DeviceClient,
            connect_remote_beams,
            connect_remote_fm,
        )

        client = DeviceClient(
            self._host,
            self._port,
            token=self._token,
            events=False,
            root="/" + MICROSCOPE,
        )
        try:
            beams = connect_remote_beams(self._host, self._port, client=client)
            devices: Dict[str, "Device"] = {b.name: b for b in beams.values()}
            devices.update(connect_remote_fm(self._host, self._port, client=client))
        except Exception:
            client.close()
            raise
        self._device_client = client
        return devices

    def list_devices(self) -> Dict[str, Any]:
        """Every device: its parameters (type, unit, limits, choices, settable) and
        commands."""
        return self._get(f"{MICROSCOPE}/devices")

    def get_parameter(self, device: str, parameter: str) -> Any:
        """A live read, as JSON: an enum as its value, a ``Point`` as its dict."""
        return self._get(f"{MICROSCOPE}/devices/{device}/{parameter}")["value"]

    def set_parameter(self, device: str, parameter: str, value: Any) -> Any:
        """Write a parameter; the server checks it and answers the value written."""
        return self._put(
            f"{MICROSCOPE}/devices/{device}/{parameter}", {"value": to_wire(value)}
        )["value"]

    def parameter_metadata(self, device: str, parameter: str) -> Dict[str, Any]:
        """A parameter's limits, choices and whether it is settable."""
        return self._get(f"{MICROSCOPE}/devices/{device}/{parameter}/metadata")

    def get_values(self, device: str) -> Dict[str, Any]:
        """Every parameter's value on a device in one read, as JSON."""
        return self._get(f"{MICROSCOPE}/devices/{device}/values")["values"]

    def call_command(self, device: str, command: str, **kwargs: Any) -> Any:
        """Run a device command: its JSON result, the image a beam acquired, or
        the frame a camera took."""
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        resp = self._session.post(
            f"{self.base_url}/{MICROSCOPE}/devices/{device}/commands/{command}",
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if content_type.startswith(TIFF_MEDIA_TYPE):
            return FibsemImage.load(io.BytesIO(resp.content))  # a beam's acquire
        if content_type.startswith(NPY_MEDIA_TYPE):
            return np.load(io.BytesIO(resp.content), allow_pickle=False)
        return resp.json()["result"]

    def call_query(self, device: str, query: str, **kwargs: Any) -> Any:
        """Ask a device a query: a read that takes arguments and moves nothing."""
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        return self._post(f"{MICROSCOPE}/devices/{device}/queries/{query}", body)[
            "result"
        ]

    def call_service_command(
        self, service: str, command: str, timeout: int = 60, **kwargs: Any
    ) -> Any:
        """Run a service command, e.g. ``stage_movement``'s ``move_to_orientation``."""
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        return self._post(
            f"{MICROSCOPE}/services/{service}/commands/{command}", body, timeout
        )["result"]

    def call_service_query(self, service: str, query: str, **kwargs: Any) -> Any:
        """Ask a service a query, e.g. ``stage_movement``'s ``project_stable_move``."""
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        return self._post(f"{MICROSCOPE}/services/{service}/queries/{query}", body)[
            "result"
        ]

    def get_service_parameter(self, service: str, parameter: str) -> Any:
        """A service parameter's value, as JSON."""
        return self._get(f"{MICROSCOPE}/services/{service}/{parameter}")["value"]

    # --- Microscope state ---

    def get_microscope_state(self) -> MicroscopeState:
        return MicroscopeState.from_dict(
            self._get(f"{MICROSCOPE}/state")["microscope_state"]
        )

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        self._put(
            f"{MICROSCOPE}/state", {"microscope_state": microscope_state.to_dict()}
        )

    # --- A beam's settings, as one group ---
    # Read in one call (the device's values) and written a parameter at a time,
    # in the order FibsemMicroscope writes them.

    def get_imaging_settings(self, beam_type: BeamType) -> ImageSettings:
        values = self.get_values(_beam(beam_type))
        return ImageSettings(
            beam_type=beam_type,
            resolution=tuple(values["resolution"]),
            dwell_time=values["dwell_time"],
            hfw=values["hfw"],
        )

    def set_imaging_settings(self, image_settings: ImageSettings) -> None:
        beam_type = image_settings.beam_type
        self.set_resolution(image_settings.resolution, beam_type)
        self.set_dwell_time(image_settings.dwell_time, beam_type)
        self.set_field_of_view(image_settings.hfw, beam_type)

    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:
        values = self.get_values(_beam(beam_type))
        return BeamSettings(
            beam_type=beam_type,
            working_distance=values.get("working_distance"),
            beam_current=values.get("current"),
            voltage=values.get("voltage"),
            hfw=values.get("hfw"),
            resolution=tuple(values["resolution"]) if "resolution" in values else None,
            dwell_time=values.get("dwell_time"),
            stigmation=_point(values.get("stigmation")),
            shift=_point(values.get("shift")),
            scan_rotation=values.get("scan_rotation"),
            preset=values.get("preset"),
        )

    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        # voltage before current: on some instruments the available currents are
        # calibrated per voltage. A None is "not stated", not a value to push.
        beam = _beam(beam_settings.beam_type)
        for name, value in (
            ("working_distance", beam_settings.working_distance),
            ("voltage", beam_settings.voltage),
            ("current", beam_settings.beam_current),
            ("hfw", beam_settings.hfw),
            ("resolution", beam_settings.resolution),
            ("dwell_time", beam_settings.dwell_time),
            ("stigmation", beam_settings.stigmation),
            ("shift", beam_settings.shift),
            ("scan_rotation", beam_settings.scan_rotation),
            ("preset", beam_settings.preset),
        ):
            if value is not None:
                self.set_parameter(beam, name, value)

    def get_detector_settings(self, beam_type: BeamType) -> FibsemDetectorSettings:
        values = self.get_values(_beam(beam_type))
        return FibsemDetectorSettings(
            type=values.get("detector_type"),
            mode=values.get("detector_mode"),
            brightness=values.get("detector_brightness"),
            contrast=values.get("detector_contrast"),
        )

    def set_detector_settings(
        self, detector_settings: FibsemDetectorSettings, beam_type: BeamType
    ) -> None:
        self.set_detector_type(detector_settings.type, beam_type)
        self.set_detector_mode(detector_settings.mode, beam_type)
        self.set_detector_brightness(detector_settings.brightness, beam_type)
        self.set_detector_contrast(detector_settings.contrast, beam_type)

    # --- A beam's parameters, one at a time (FibsemMicroscope's wrappers) ---

    def get_beam_current(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "current")

    def set_beam_current(self, current: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "current", current)

    def get_beam_voltage(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "voltage")

    def set_beam_voltage(self, voltage: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "voltage", voltage)

    def get_field_of_view(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "hfw")

    def set_field_of_view(self, hfw: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "hfw", hfw)

    def get_working_distance(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "working_distance")

    def set_working_distance(self, wd: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "working_distance", wd)

    def get_dwell_time(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "dwell_time")

    def set_dwell_time(self, dwell_time: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "dwell_time", dwell_time)

    def get_scan_rotation(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "scan_rotation")

    def set_scan_rotation(self, rotation: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "scan_rotation", rotation)

    def get_detector_type(self, beam_type: BeamType) -> str:
        return self.get_parameter(_beam(beam_type), "detector_type")

    def set_detector_type(self, detector_type: str, beam_type: BeamType) -> str:
        return self.set_parameter(_beam(beam_type), "detector_type", detector_type)

    def get_detector_mode(self, beam_type: BeamType) -> str:
        return self.get_parameter(_beam(beam_type), "detector_mode")

    def set_detector_mode(self, mode: str, beam_type: BeamType) -> str:
        return self.set_parameter(_beam(beam_type), "detector_mode", mode)

    def get_detector_contrast(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "detector_contrast")

    def set_detector_contrast(self, contrast: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "detector_contrast", contrast)

    def get_detector_brightness(self, beam_type: BeamType) -> float:
        return self.get_parameter(_beam(beam_type), "detector_brightness")

    def set_detector_brightness(self, brightness: float, beam_type: BeamType) -> float:
        return self.set_parameter(_beam(beam_type), "detector_brightness", brightness)

    def get_resolution(self, beam_type: BeamType) -> Tuple[int, int]:
        return tuple(self.get_parameter(_beam(beam_type), "resolution"))

    def set_resolution(
        self, resolution: Tuple[int, int], beam_type: BeamType
    ) -> Tuple[int, int]:
        return tuple(
            self.set_parameter(_beam(beam_type), "resolution", tuple(resolution))
        )

    def get_stigmation(self, beam_type: BeamType) -> Point:
        return Point.from_dict(self.get_parameter(_beam(beam_type), "stigmation"))

    def set_stigmation(self, stigmation: Point, beam_type: BeamType) -> Point:
        return Point.from_dict(
            self.set_parameter(_beam(beam_type), "stigmation", stigmation)
        )

    def get_beam_shift(self, beam_type: BeamType) -> Point:
        return Point.from_dict(self.get_parameter(_beam(beam_type), "shift"))

    def set_beam_shift(self, shift: Point, beam_type: BeamType) -> Point:
        return Point.from_dict(self.set_parameter(_beam(beam_type), "shift", shift))

    # --- Milling angle ---
    # The stage movement service's: the stored milling angle is its parameter, in
    # degrees; the reads are its queries. move_to_milling_angle takes degrees
    # too, as the FibsemMicroscope method does.

    def get_current_milling_angle(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> float:
        """The milling angle, in degrees, a stage position gives (the current one
        when None)."""
        return self.call_service_query(
            STAGE_MOVEMENT, "current_milling_angle", position=stage_position
        )

    def set_milling_angle(self, milling_angle: float) -> None:
        """Set the stored milling angle, in degrees."""
        self._put(
            f"{MICROSCOPE}/services/{STAGE_MOVEMENT}/milling_angle",
            {"value": milling_angle},
        )

    def move_to_milling_angle(
        self, milling_angle: float, rotation: Optional[float] = None
    ) -> bool:
        """Move the stage to a milling angle, in degrees, and a rotation in
        degrees; the rotation reference when ``rotation`` is None."""
        return self.call_service_command(
            STAGE_MOVEMENT,
            "move_to_milling_angle",
            milling_angle=milling_angle,
            rotation=rotation,
        )

    def is_close_to_milling_angle(
        self, milling_angle: float, atol: float = 2.0
    ) -> bool:
        """Whether the current milling angle is within ``atol`` of ``milling_angle``,
        both in degrees."""
        return self.call_service_query(
            STAGE_MOVEMENT,
            "is_close_to_milling_angle",
            milling_angle=milling_angle,
            atol=atol,
        )

    # --- Milling ---
    # The milling service's commands; the ion beam is the milling beam.

    def _milling(self, command: str, timeout: int = 60, **kwargs: Any) -> Any:
        return self.call_service_command(MILLING, command, timeout=timeout, **kwargs)

    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        self._milling("setup", settings=mill_settings)

    def draw_patterns(self, patterns: List[FibsemPatternSettings]) -> None:
        self._milling("draw", patterns=list(patterns))

    def run_milling(self) -> None:
        self._milling("run", timeout=3600)

    def start_milling(self) -> None:
        self._milling("start")

    def stop_milling(self, reason: str = "stopped from a script") -> None:
        """Stop the mill. The App Server records the reason with the stop."""
        self._post(
            f"{MICROSCOPE}/services/{MILLING}/commands/stop",
            {"kwargs": {}, "reason": reason},
        )

    def pause_milling(self) -> None:
        self._milling("pause")

    def resume_milling(self) -> None:
        self._milling("resume")

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        """Clear the patterns, put the milling beam back as setup found it, then
        the imaging voltage and current given."""
        self._milling("clear")
        self._milling("restore")
        if imaging_voltage is not None:
            self.set_beam_voltage(imaging_voltage, BeamType.ION)
        if imaging_current is not None:
            self.set_beam_current(imaging_current, BeamType.ION)

    def clear_patterns(self) -> None:
        self._milling("clear")

    def get_milling_state(self) -> MillingState:
        return MillingState(self.get_service_parameter(MILLING, "state"))

    def estimate_milling_time(self) -> float:
        return self.call_service_query(MILLING, "estimate")


def _point(value: Optional[Dict[str, Any]]) -> Optional[Point]:
    return None if value is None else Point.from_dict(value)
