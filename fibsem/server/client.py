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
from typing import List, Optional, Tuple

import requests

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
    MillingState,
    MicroscopeState,
    Point,
    SystemSettings,
)


class FibsemClient:
    """Network client for FibsemServer.

    Fetches system settings and stage_is_compustage from the server at init
    so they are available as direct attributes, matching the FibsemMicroscope interface.
    """

    def __init__(self, host: str = "localhost", port: int = 8001):
        self.base_url = f"http://{host}:{port}"
        self._session = requests.Session()
        self._fetch_system()

    def _fetch_system(self) -> None:
        """Fetch and cache system settings from the server."""
        data = self._get("system")
        self.system: SystemSettings = SystemSettings.from_dict(data["system"])
        self.stage_is_compustage: bool = data["stage_is_compustage"]

    def _get(self, endpoint: str, timeout: int = 10) -> dict:
        resp = self._session.get(f"{self.base_url}/{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, body: dict = None, timeout: int = 30) -> dict:
        resp = self._session.post(f"{self.base_url}/{endpoint}", json=body or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post_image(self, endpoint: str, body: dict = None, timeout: int = 60) -> FibsemImage:
        resp = self._session.post(f"{self.base_url}/{endpoint}", json=body or {}, timeout=timeout)
        resp.raise_for_status()
        return FibsemImage.load(io.BytesIO(resp.content))

    # --- Health ---

    def health(self) -> dict:
        return self._get("health", timeout=5)

    # --- Image acquisition ---

    def acquire_image(self, beam_type: BeamType = BeamType.ELECTRON, image_settings: Optional[ImageSettings] = None) -> FibsemImage:
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

    def stable_move(self, dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        result = self._post("stable_move", {"dx": dx, "dy": dy, "beam_type": beam_type.name})
        return FibsemStagePosition.from_dict(result["position"])

    def project_stable_move(self, dx: float, dy: float, beam_type: BeamType, base_position: FibsemStagePosition) -> FibsemStagePosition:
        result = self._post("project_stable_move", {
            "dx": dx, "dy": dy,
            "beam_type": beam_type.name,
            "base_position": base_position.to_dict(),
        })
        return FibsemStagePosition.from_dict(result["position"])

    def vertical_move(self, dy: float, dx: float = 0.0, static_wd: bool = True) -> FibsemStagePosition:
        result = self._post("vertical_move", {"dy": dy, "dx": dx, "static_wd": static_wd})
        return FibsemStagePosition.from_dict(result["position"])

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        self._post("safe_absolute_stage_movement", {"position": position.to_dict()})

    def move_flat_to_beam(self, beam_type: BeamType) -> None:
        self._post("move_flat_to_beam", {"beam_type": beam_type.name})

    # --- Microscope state ---

    def get_microscope_state(self) -> MicroscopeState:
        return MicroscopeState.from_dict(self._get("microscope_state")["microscope_state"])

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        self._post("microscope_state", {"microscope_state": microscope_state.to_dict()})

    # --- Imaging settings ---

    def get_imaging_settings(self, beam_type: BeamType) -> ImageSettings:
        result = self._post("imaging_settings/get", {"beam_type": beam_type.name})
        return ImageSettings.from_dict(result["image_settings"])

    def set_imaging_settings(self, image_settings: ImageSettings) -> None:
        self._post("imaging_settings/set", {"image_settings": image_settings.to_dict()})

    # --- Beam settings ---

    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:
        result = self._post("beam_settings/get", {"beam_type": beam_type.name})
        return BeamSettings.from_dict(result["beam_settings"])

    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        self._post("beam_settings/set", {"beam_settings": beam_settings.to_dict()})

    def get_beam_system_settings(self, beam_type: BeamType) -> BeamSystemSettings:
        result = self._post("beam_system_settings/get", {"beam_type": beam_type.name})
        return BeamSystemSettings.from_dict(result["beam_system_settings"])

    def set_beam_system_settings(self, settings: BeamSystemSettings) -> None:
        self._post("beam_system_settings/set", {"beam_system_settings": settings.to_dict()})

    # --- Detector settings ---

    def get_detector_settings(self, beam_type: BeamType) -> FibsemDetectorSettings:
        result = self._post("detector_settings/get", {"beam_type": beam_type.name})
        return FibsemDetectorSettings.from_dict(result["detector_settings"])

    def set_detector_settings(self, detector_settings: FibsemDetectorSettings, beam_type: BeamType) -> None:
        self._post("detector_settings/set", {
            "detector_settings": detector_settings.to_dict(),
            "beam_type": beam_type.name,
        })

    # --- Individual beam getters / setters ---

    def get_beam_current(self, beam_type: BeamType) -> float:
        return self._post("beam_current/get", {"beam_type": beam_type.name})["value"]

    def set_beam_current(self, current: float, beam_type: BeamType) -> float:
        return self._post("beam_current/set", {"value": current, "beam_type": beam_type.name})["value"]

    def get_beam_voltage(self, beam_type: BeamType) -> float:
        return self._post("beam_voltage/get", {"beam_type": beam_type.name})["value"]

    def set_beam_voltage(self, voltage: float, beam_type: BeamType) -> float:
        return self._post("beam_voltage/set", {"value": voltage, "beam_type": beam_type.name})["value"]

    def get_field_of_view(self, beam_type: BeamType) -> float:
        return self._post("field_of_view/get", {"beam_type": beam_type.name})["value"]

    def set_field_of_view(self, hfw: float, beam_type: BeamType) -> float:
        return self._post("field_of_view/set", {"value": hfw, "beam_type": beam_type.name})["value"]

    def get_working_distance(self, beam_type: BeamType) -> float:
        return self._post("working_distance/get", {"beam_type": beam_type.name})["value"]

    def set_working_distance(self, wd: float, beam_type: BeamType) -> float:
        return self._post("working_distance/set", {"value": wd, "beam_type": beam_type.name})["value"]

    def get_dwell_time(self, beam_type: BeamType) -> float:
        return self._post("dwell_time/get", {"beam_type": beam_type.name})["value"]

    def set_dwell_time(self, dwell_time: float, beam_type: BeamType) -> float:
        return self._post("dwell_time/set", {"value": dwell_time, "beam_type": beam_type.name})["value"]

    def get_resolution(self, beam_type: BeamType) -> Tuple[int, int]:
        return tuple(self._post("resolution/get", {"beam_type": beam_type.name})["value"])

    def set_resolution(self, resolution: Tuple[int, int], beam_type: BeamType) -> Tuple[int, int]:
        return tuple(self._post("resolution/set", {"value": list(resolution), "beam_type": beam_type.name})["value"])

    def get_scan_rotation(self, beam_type: BeamType) -> float:
        return self._post("scan_rotation/get", {"beam_type": beam_type.name})["value"]

    def set_scan_rotation(self, rotation: float, beam_type: BeamType) -> float:
        return self._post("scan_rotation/set", {"value": rotation, "beam_type": beam_type.name})["value"]

    def get_stigmation(self, beam_type: BeamType) -> Point:
        return Point.from_dict(self._post("stigmation/get", {"beam_type": beam_type.name})["value"])

    def set_stigmation(self, stigmation: Point, beam_type: BeamType) -> Point:
        return Point.from_dict(self._post("stigmation/set", {"value": stigmation.to_dict(), "beam_type": beam_type.name})["value"])

    def get_beam_shift(self, beam_type: BeamType) -> Point:
        return Point.from_dict(self._post("beam_shift/get", {"beam_type": beam_type.name})["value"])

    def set_beam_shift(self, shift: Point, beam_type: BeamType) -> Point:
        return Point.from_dict(self._post("beam_shift/set", {"value": shift.to_dict(), "beam_type": beam_type.name})["value"])

    # --- Detector individual getters / setters ---

    def get_detector_type(self, beam_type: BeamType) -> str:
        return self._post("detector_type/get", {"beam_type": beam_type.name})["value"]

    def set_detector_type(self, detector_type: str, beam_type: BeamType) -> str:
        return self._post("detector_type/set", {"value": detector_type, "beam_type": beam_type.name})["value"]

    def get_detector_mode(self, beam_type: BeamType) -> str:
        return self._post("detector_mode/get", {"beam_type": beam_type.name})["value"]

    def set_detector_mode(self, mode: str, beam_type: BeamType) -> str:
        return self._post("detector_mode/set", {"value": mode, "beam_type": beam_type.name})["value"]

    def get_detector_contrast(self, beam_type: BeamType) -> float:
        return self._post("detector_contrast/get", {"beam_type": beam_type.name})["value"]

    def set_detector_contrast(self, contrast: float, beam_type: BeamType) -> float:
        return self._post("detector_contrast/set", {"value": contrast, "beam_type": beam_type.name})["value"]

    def get_detector_brightness(self, beam_type: BeamType) -> float:
        return self._post("detector_brightness/get", {"beam_type": beam_type.name})["value"]

    def set_detector_brightness(self, brightness: float, beam_type: BeamType) -> float:
        return self._post("detector_brightness/set", {"value": brightness, "beam_type": beam_type.name})["value"]

    # --- Available values ---

    def get_available_values(self, key: str, beam_type: Optional[BeamType] = None) -> list:
        """Get the list of available values for a given key (e.g. 'detector_type', 'application_file')."""
        body = {"key": key, "beam_type": beam_type.name if beam_type else None}
        return self._post("available_values", body)["values"]

    # --- Milling angle ---

    def get_current_milling_angle(self, stage_position: Optional[FibsemStagePosition] = None) -> float:
        """Get the current milling angle in degrees. Pass a stage_position to calculate from a specific position."""
        if stage_position is None:
            return self._get("milling_angle")["milling_angle"]
        body = {"stage_position": stage_position.to_dict()}
        return self._post("milling_angle/from_position", body)["milling_angle"]

    def set_milling_angle(self, milling_angle: float) -> None:
        """Set the stored milling angle in degrees."""
        self._post("milling_angle/set", {"milling_angle": milling_angle})

    def move_to_milling_angle(self, milling_angle: float, rotation: Optional[float] = None) -> bool:
        """Move the stage to the specified milling angle (radians). Returns True if the move was successful."""
        body = {"milling_angle": milling_angle}
        if rotation is not None:
            body["rotation"] = rotation
        return self._post("milling_angle/move", body)["success"]

    def is_close_to_milling_angle(self, milling_angle: float, atol: float = 2.0) -> bool:
        """Check if the current milling angle (degrees) is within atol degrees of the target."""
        return self._post("milling_angle/is_close", {"milling_angle": milling_angle, "atol": atol})["is_close"]

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

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False) -> None:
        self._post("run_milling", {
            "milling_current": milling_current,
            "milling_voltage": milling_voltage,
            "asynch": asynch,
        }, timeout=3600)

    def start_milling(self) -> None:
        self._post("start_milling")

    def stop_milling(self) -> None:
        self._post("stop_milling")

    def pause_milling(self) -> None:
        self._post("pause_milling")

    def resume_milling(self) -> None:
        self._post("resume_milling")

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        self._post("finish_milling", {
            "imaging_current": imaging_current,
            "imaging_voltage": imaging_voltage,
        })

    def clear_patterns(self) -> None:
        self._post("clear_patterns")

    def get_milling_state(self) -> MillingState:
        return MillingState[self._get("milling_state")["state"]]

    def estimate_milling_time(self) -> float:
        return self._get("estimate_milling_time")["seconds"]
