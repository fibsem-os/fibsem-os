from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


# --- Shared ---

class BeamTypeRequest(BaseModel):
    beam_type: str  # "ELECTRON" or "ION"


# --- Image ---

class AcquireImageRequest(BaseModel):
    beam_type: str
    image_settings: Optional[Dict[str, Any]] = None  # ImageSettings.to_dict(); if None uses current settings


# --- Stage ---

class StagePositionRequest(BaseModel):
    position: Dict[str, Any]


class StagePositionResponse(BaseModel):
    position: Dict[str, Any]


class StableMoveRequest(BaseModel):
    dx: float
    dy: float
    beam_type: str


class ProjectStableMoveRequest(BaseModel):
    dx: float
    dy: float
    beam_type: str
    base_position: Dict[str, Any]  # FibsemStagePosition.to_dict()


class VerticalMoveRequest(BaseModel):
    dy: float
    dx: float = 0.0
    static_wd: bool = True


class FlatToBeamRequest(BaseModel):
    beam_type: str


# --- Beam / Detector / State ---

class BeamSettingsRequest(BaseModel):
    beam_settings: Dict[str, Any]


class BeamSystemSettingsRequest(BaseModel):
    beam_system_settings: Dict[str, Any]


class DetectorSettingsRequest(BaseModel):
    detector_settings: Dict[str, Any]
    beam_type: str


class MicroscopeStateRequest(BaseModel):
    microscope_state: Dict[str, Any]


class ImageSettingsRequest(BaseModel):
    image_settings: Dict[str, Any]


class FloatBeamRequest(BaseModel):
    value: float
    beam_type: str


class StringBeamRequest(BaseModel):
    value: str
    beam_type: str


class PointBeamRequest(BaseModel):
    value: Dict[str, Any]  # Point.to_dict()
    beam_type: str


class ResolutionBeamRequest(BaseModel):
    value: List[int]  # [width, height]
    beam_type: str


# --- Milling ---

class MillingSettingsRequest(BaseModel):
    mill_settings: Dict[str, Any]


class RunMillingRequest(BaseModel):
    milling_current: float
    milling_voltage: float
    asynch: bool = False


class FinishMillingRequest(BaseModel):
    imaging_current: float
    imaging_voltage: float


class AvailableValuesRequest(BaseModel):
    key: str
    beam_type: Optional[str] = None  # "ELECTRON", "ION", or None


class DrawPatternsRequest(BaseModel):
    # Each pattern dict must include a "type" key: "Rectangle", "Line", "Circle", "Bitmap", "Polygon"
    patterns: List[Dict[str, Any]]


class MillingAngleRequest(BaseModel):
    milling_angle: float  # degrees

class MillingAngleFromPositionRequest(BaseModel):
    stage_position: Optional[Dict[str, Any]] = None  # FibsemStagePosition.to_dict(); if None uses current position

class MoveToMillingAngleRequest(BaseModel):
    milling_angle: float                # radians (underlying move_to_milling_angle takes radians)
    rotation: Optional[float] = None   # radians; if None uses rotation_reference from system settings

class IsCloseToMillingAngleRequest(BaseModel):
    milling_angle: float        # degrees
    atol: float = 2.0           # degrees
