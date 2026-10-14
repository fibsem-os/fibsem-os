"""The microscope as a container of devices: the vendor-neutral device API.

Every device a microscope builds is in ``microscope.devices``, by name (``electron``,
``ion``, ``stage``, ``chamber``, ``manipulator``, and the FM's ``fm``, ``camera``,
...). A device's parameters describe themselves (type, unit, limits, choices) and
emit ``changed``; its commands are plain methods it lists in ``commands``:

    from fibsem import utils
    from fibsem.structures import BeamType, FibsemStagePosition

    microscope, _ = utils.setup_session(manufacturer="Demo")
    sem = microscope.beams[BeamType.ELECTRON]     # microscope.devices["electron"]

    sem.current.choices                     # cached metadata, no instrument call
    sem.current.changed.connect(print)      # every change, with the value
    sem.hfw.set_value(100e-6)               # checked, then written
    "preset" in sem.parameters              # False: the Demo has no presets
    image = sem.acquire()

    microscope.stage.move_absolute(FibsemStagePosition(x=1e-3))  # refused outside limits

This package is vendor-neutral and never imports a driver; each driver's device
classes and builders are in ``fibsem.drivers.<driver>.devices``. ``docs/developers/devices.md``
is the guide, including where each removed ``get``/``set`` key went.
"""

from fibsem.devices.beam import Beam
from fibsem.devices.chamber import CHAMBER_RESOURCE, Chamber
from fibsem.devices.chamber_camera import ChamberCamera
from fibsem.devices.core import (
    IMAGING_CHANNEL,
    Axes,
    Axis,
    BoundParameter,
    CommandInfo,
    Device,
    DeviceFault,
    Parameter,
    ParameterMetadata,
    ParameterReadOnly,
    ParameterUnavailable,
    Resources,
    command,
)
from fibsem.devices.manipulator import MANIPULATOR_RESOURCE, Manipulator
from fibsem.devices.stage import (
    AXIS_UNITS,
    STAGE_RESOURCE,
    UNLIMITED,
    Stage,
    StageLimitError,
)

__all__ = [
    "AXIS_UNITS",
    "Axes",
    "Axis",
    "CHAMBER_RESOURCE",
    "Chamber",
    "ChamberCamera",
    "STAGE_RESOURCE",
    "IMAGING_CHANNEL",
    "MANIPULATOR_RESOURCE",
    "Manipulator",
    "CommandInfo",
    "Beam",
    "BoundParameter",
    "Device",
    "DeviceFault",
    "ParameterMetadata",
    "Parameter",
    "ParameterReadOnly",
    "ParameterUnavailable",
    "Resources",
    "Stage",
    "StageLimitError",
    "UNLIMITED",
    "command",
]
