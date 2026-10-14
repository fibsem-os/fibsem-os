from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# --- Shared ---


class BeamTypeRequest(BaseModel):
    beam_type: str  # "ELECTRON" or "ION"


# --- Image ---


class AcquireImageRequest(BaseModel):
    beam_type: str
    image_settings: Optional[Dict[str, Any]] = (
        None  # ImageSettings.to_dict(); if None uses current settings
    )


# --- Stage ---


class StagePositionRequest(BaseModel):
    position: Dict[str, Any]


class StagePositionResponse(BaseModel):
    position: Dict[str, Any]


class StableMoveRequest(BaseModel):
    dx: float
    dy: float
    beam_type: str


class VerticalMoveRequest(BaseModel):
    dy: float
    dx: float = 0.0


class OrientationRequest(BaseModel):
    orientation: str


# --- Beam / Detector / State ---


class BeamSettingsRequest(BaseModel):
    beam_settings: Dict[str, Any]


class BeamSystemSettingsRequest(BaseModel):
    beam_system_settings: Dict[str, Any]


class DetectorSettingsRequest(BaseModel):
    detector_settings: Dict[str, Any]


class MicroscopeStateRequest(BaseModel):
    microscope_state: Dict[str, Any]


class ImageSettingsRequest(BaseModel):
    image_settings: Dict[str, Any]


# --- Milling ---


class MillingSettingsRequest(BaseModel):
    mill_settings: Dict[str, Any]


class FinishMillingRequest(BaseModel):
    imaging_current: float
    imaging_voltage: float


class DrawPatternsRequest(BaseModel):
    # Each pattern dict must include a "type" key: "Rectangle", "Line", "Circle", "Bitmap", "Polygon"
    patterns: List[Dict[str, Any]]
