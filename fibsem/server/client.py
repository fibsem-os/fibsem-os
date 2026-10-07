"""
FibsemClient: network client that delegates FibsemMicroscope calls to a remote FibsemServer.

Usage:
    from fibsem.server.client import FibsemClient
    from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

    m = FibsemClient(host="192.168.1.100", port=8001)
    image = m.acquire_image(BeamType.ELECTRON)
    state = m.get_microscope_state()
"""

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import requests

from fibsem.devices.wire import to_wire
from fibsem.server.images import TIFF_MEDIA_TYPE
from fibsem.structures import (
    BeamSettings,
    BeamSystemSettings,
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


def _beam(beam_type: BeamType) -> str:
    """A beam's device name: "electron" or "ion"."""
    return beam_type.name.lower()


class FibsemClient:
    """Network client for FibsemServer.

    Fetches system settings from the server at init
    so they are available as direct attributes, matching the FibsemMicroscope interface.
    """

    def __init__(
        self, host: str = "localhost", port: int = 8001, token: Optional[str] = None
    ):
        self.base_url = f"http://{host}:{port}"
        self._session = requests.Session()
        if token is not None:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._fetch_system()

    def _fetch_system(self) -> None:
        """Fetch and cache system settings from the server."""
        data = self._get("system")
        self.system: SystemSettings = SystemSettings.from_dict(data["system"])
        for key, present in (data.get("fitted") or {}).items():
            self.set_available(key, bool(present))

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
        body = {"beam_type": beam_type.name}
        if image_settings is not None:
            body["image_settings"] = image_settings.to_dict()
        return self._post_image("acquire_image", body)

    def last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage:
        return self._post_image("last_image", {"beam_type": beam_type.name})

    def acquire_chamber_image(self) -> FibsemImage:
        return self._post_image("acquire_chamber_image")

    def autocontrast(self, beam_type: BeamType) -> None:
        self._post("autocontrast", {"beam_type": beam_type.name})

    def auto_focus(self, beam_type: BeamType) -> None:
        self._post("auto_focus", {"beam_type": beam_type.name})

    # --- Stage movement ---

    def get_stage_position(self) -> FibsemStagePosition:
        return FibsemStagePosition.from_dict(self._get("stage_position")["position"])

    def get_stage_orientation(self) -> str:
        return self._get("stage_orientation")["orientation"]

    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        result = self._post("move_stage_absolute", {"position": position.to_dict()})
        return FibsemStagePosition.from_dict(result["position"])

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        result = self._post("move_stage_relative", {"position": position.to_dict()})
        return FibsemStagePosition.from_dict(result["position"])

    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemStagePosition:
        result = self._post(
            "stable_move", {"dx": dx, "dy": dy, "beam_type": beam_type.name}
        )
        return FibsemStagePosition.from_dict(result["position"])

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        result = self._post(
            "project_stable_move",
            {
                "dx": dx,
                "dy": dy,
                "beam_type": beam_type.name,
                "base_position": base_position.to_dict(),
            },
        )
        return FibsemStagePosition.from_dict(result["position"])

    def vertical_move(self, dy: float, dx: float = 0.0) -> FibsemStagePosition:
        result = self._post("vertical_move", {"dy": dy, "dx": dx})
        return FibsemStagePosition.from_dict(result["position"])

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        self._post("safe_absolute_stage_movement", {"position": position.to_dict()})

    def move_to_orientation(self, orientation: str) -> FibsemStagePosition:
        result = self._post("move_to_orientation", {"orientation": orientation})
        return FibsemStagePosition.from_dict(result["position"])

    # --- Devices ---
    # Every device the microscope built (``microscope.devices``), by name: the beams
    # are "electron" and "ion", then "stage", "chamber", "manipulator", FM parts...

    def list_devices(self) -> Dict[str, Any]:
        """Every device: its parameters (type, unit, limits, choices, settable) and
        commands."""
        return self._get("devices")

    def get_parameter(self, device: str, parameter: str) -> Any:
        """A live read, as JSON: an enum as its value, a ``Point`` as its dict."""
        return self._get(f"devices/{device}/{parameter}")["value"]

    def set_parameter(self, device: str, parameter: str, value: Any) -> Any:
        """Write a parameter; the server checks it and answers the value written."""
        return self._put(f"devices/{device}/{parameter}", {"value": to_wire(value)})[
            "value"
        ]

    def parameter_metadata(self, device: str, parameter: str) -> Dict[str, Any]:
        """A parameter's limits, choices and whether it is settable."""
        return self._get(f"devices/{device}/{parameter}/metadata")

    def call_command(self, device: str, command: str, **kwargs: Any) -> Any:
        """Run a device command: its JSON result, or the image a beam acquired."""
        body = {"kwargs": {name: to_wire(value) for name, value in kwargs.items()}}
        resp = self._session.post(
            f"{self.base_url}/devices/{device}/commands/{command}",
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith(TIFF_MEDIA_TYPE):
            return FibsemImage.load(io.BytesIO(resp.content))  # a beam's acquire
        return resp.json()["result"]

    # --- Microscope state ---

    def get_microscope_state(self) -> MicroscopeState:
        return MicroscopeState.from_dict(
            self._get("microscope_state")["microscope_state"]
        )

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        self._put("microscope_state", {"microscope_state": microscope_state.to_dict()})

    # --- A beam's settings, as one group ---

    def get_imaging_settings(self, beam_type: BeamType) -> ImageSettings:
        result = self._get(f"beams/{_beam(beam_type)}/imaging_settings")
        return ImageSettings.from_dict(result["image_settings"])

    def set_imaging_settings(self, image_settings: ImageSettings) -> None:
        self._put(
            f"beams/{_beam(image_settings.beam_type)}/imaging_settings",
            {"image_settings": image_settings.to_dict()},
        )

    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:
        result = self._get(f"beams/{_beam(beam_type)}/beam_settings")
        return BeamSettings.from_dict(result["beam_settings"])

    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        self._put(
            f"beams/{_beam(beam_settings.beam_type)}/beam_settings",
            {"beam_settings": beam_settings.to_dict()},
        )

    def get_beam_system_settings(self, beam_type: BeamType) -> BeamSystemSettings:
        result = self._get(f"beams/{_beam(beam_type)}/beam_system_settings")
        return BeamSystemSettings.from_dict(result["beam_system_settings"])

    def set_beam_system_settings(self, settings: BeamSystemSettings) -> None:
        self._put(
            f"beams/{_beam(settings.beam_type)}/beam_system_settings",
            {"beam_system_settings": settings.to_dict()},
        )

    def get_detector_settings(self, beam_type: BeamType) -> FibsemDetectorSettings:
        result = self._get(f"beams/{_beam(beam_type)}/detector_settings")
        return FibsemDetectorSettings.from_dict(result["detector_settings"])

    def set_detector_settings(
        self, detector_settings: FibsemDetectorSettings, beam_type: BeamType
    ) -> None:
        self._put(
            f"beams/{_beam(beam_type)}/detector_settings",
            {"detector_settings": detector_settings.to_dict()},
        )

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

    def get_current_milling_angle(
        self, stage_position: Optional[FibsemStagePosition] = None
    ) -> float:
        """Get the current milling angle in degrees. Pass a stage_position to calculate from a specific position."""
        if stage_position is None:
            return self._get("milling_angle")["milling_angle_deg"]
        body = {"stage_position": stage_position.to_dict()}
        return self._post("milling_angle/from_position", body)["milling_angle_deg"]

    def set_milling_angle(self, milling_angle: float) -> None:
        """Set the stored milling angle in degrees."""
        self._post("milling_angle/set", {"milling_angle_deg": milling_angle})

    def move_to_milling_angle(
        self, milling_angle: float, rotation: Optional[float] = None
    ) -> bool:
        """Move the stage to the specified milling angle (RADIANS, matching the
        FibsemMicroscope ABC — FIB-853). The wire speaks degrees; converted here."""
        body = {"milling_angle_deg": math.degrees(milling_angle)}
        if rotation is not None:
            body["rotation_deg"] = math.degrees(rotation)
        return self._post("milling_angle/move", body)["success"]

    def is_close_to_milling_angle(
        self, milling_angle: float, atol: float = 2.0
    ) -> bool:
        """Check if the current milling angle (degrees) is within atol degrees of the target."""
        return self._post(
            "milling_angle/is_close",
            {"milling_angle_deg": milling_angle, "atol_deg": atol},
        )["is_close"]

    # --- Milling ---

    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        self._post("setup_milling", {"mill_settings": mill_settings.to_dict()})

    def draw_patterns(self, patterns: List[FibsemPatternSettings]) -> None:
        payload = []
        for p in patterns:
            d = p.to_dict()
            d["type"] = type(p).__name__.replace("Fibsem", "").replace("Settings", "")
            payload.append(d)
        self._post("draw_patterns", {"patterns": payload})

    def run_milling(
        self, milling_current: float, milling_voltage: float, asynch: bool = False
    ) -> None:
        self._post(
            "run_milling",
            {
                "milling_current": milling_current,
                "milling_voltage": milling_voltage,
                "asynch": asynch,
            },
            timeout=3600,
        )

    def start_milling(self) -> None:
        self._post("start_milling")

    def stop_milling(self) -> None:
        self._post("stop_milling")

    def pause_milling(self) -> None:
        self._post("pause_milling")

    def resume_milling(self) -> None:
        self._post("resume_milling")

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        self._post(
            "finish_milling",
            {
                "imaging_current": imaging_current,
                "imaging_voltage": imaging_voltage,
            },
        )

    def clear_patterns(self) -> None:
        self._post("clear_patterns")

    def get_milling_state(self) -> MillingState:
        return MillingState[self._get("milling_state")["state"]]

    def estimate_milling_time(self) -> float:
        return self._get("estimate_milling_time")["seconds"]
