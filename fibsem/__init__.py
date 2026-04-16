
try:
    import importlib.metadata
    __version__ = importlib.metadata.version('fibsem')
except ModuleNotFoundError:
    __version__ = "unknown"

from fibsem.exceptions import (
    FibsemError,
    HardwareError,
    MicroscopeConnectionError,
    BeamError,
    StageError,
    ManipulatorError,
    GasInjectionError,
    ConfigurationError,
    AcquisitionError,
    MillingError,
    AlignmentError,
    APIError,
    AutoScriptError,
    AutoScriptException,
    ValidationError,
    DataError,
)
