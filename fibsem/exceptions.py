"""Custom exceptions for fibsemOS.

Exception hierarchy::

    FibsemError
    ├── HardwareError
    │   ├── MicroscopeConnectionError
    │   ├── BeamError
    │   ├── StageError
    │   ├── ManipulatorError
    │   └── GasInjectionError
    ├── ConfigurationError
    ├── AcquisitionError
    ├── MillingError
    ├── AlignmentError
    ├── APIError
    │   └── AutoScriptError
    ├── ValidationError
    └── DataError
"""


class FibsemError(Exception):
    """Base exception for all fibsemOS errors."""


# ---------------------------------------------------------------------------
# Hardware errors
# ---------------------------------------------------------------------------

class HardwareError(FibsemError):
    """Raised for physical hardware or equipment failures."""


class MicroscopeConnectionError(HardwareError, ConnectionError):
    """Raised when the microscope is not connected or cannot be reached."""


class BeamError(HardwareError):
    """Raised for beam-related hardware failures (on/off, blanking)."""


class StageError(HardwareError):
    """Raised for stage movement or configuration failures."""


class ManipulatorError(HardwareError):
    """Raised for manipulator availability or movement failures."""


class GasInjectionError(HardwareError, TimeoutError):
    """Raised for gas injection system (GIS) errors, including heating timeouts."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(FibsemError, ValueError):
    """Raised for invalid or missing configuration / settings."""


# ---------------------------------------------------------------------------
# Acquisition errors
# ---------------------------------------------------------------------------

class AcquisitionError(FibsemError):
    """Raised when image acquisition fails."""


# ---------------------------------------------------------------------------
# Milling errors
# ---------------------------------------------------------------------------

class MillingError(FibsemError):
    """Raised for milling pattern or patterning-state errors."""


# ---------------------------------------------------------------------------
# Alignment errors
# ---------------------------------------------------------------------------

class AlignmentError(FibsemError, ValueError):
    """Raised for beam-shift or cross-correlation alignment failures."""


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------

class APIError(FibsemError):
    """Raised for hardware-API errors (AutoScript, TESCAN, etc.)."""


class AutoScriptError(APIError):
    """Raised for AutoScript (ThermoFisher) API errors."""


# Backward-compatible alias
AutoScriptException = AutoScriptError


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ValidationError(FibsemError, ValueError):
    """Raised when parameter or type validation fails."""


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------

class DataError(FibsemError):
    """Raised for image or metadata format issues."""


__all__ = [
    "FibsemError",
    "HardwareError",
    "MicroscopeConnectionError",
    "BeamError",
    "StageError",
    "ManipulatorError",
    "GasInjectionError",
    "ConfigurationError",
    "AcquisitionError",
    "MillingError",
    "AlignmentError",
    "APIError",
    "AutoScriptError",
    "AutoScriptException",
    "ValidationError",
    "DataError",
]
