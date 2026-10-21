"""Request bodies for the App Server's own routes."""

from typing import Any, Dict

from pydantic import BaseModel


class MicroscopeStateRequest(BaseModel):
    microscope_state: Dict[str, Any]


class ControlRequest(BaseModel):
    """Take control of the microscope: what for, as the operator reads it."""

    action: str = "remote run"
