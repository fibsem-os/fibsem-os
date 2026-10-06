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
classes and builders are in ``fibsem.devices.drivers``. ``docs/developers/devices.md``
is the guide, including where each deprecated ``get``/``set`` key went.
"""

from fibsem.devices.beam import (
    BEAM_ROUTES,
    STAGE_COMMAND_ROUTES,
    STAGE_ROUTES,
    Beam,
    KeyRouter,
)
from fibsem.devices.chamber import (
    CHAMBER_COMMAND_ROUTES,
    CHAMBER_RESOURCE,
    CHAMBER_ROUTES,
    Chamber,
)
from fibsem.devices.core import (
    IMAGING_CHANNEL,
    BoundParameter,
    CommandInfo,
    Device,
    Parameter,
    ParameterMetadata,
    ParameterReadOnly,
    ParameterUnavailable,
    Resources,
    command,
)
from fibsem.devices.gis import GIS_RESOURCE, GasInjector
from fibsem.devices.manipulator import (
    MANIPULATOR_RESOURCE,
    MANIPULATOR_ROUTES,
    Manipulator,
)
from fibsem.devices.stage import (
    AXIS_UNITS,
    STAGE_RESOURCE,
    UNLIMITED,
    Axes,
    Axis,
    Stage,
    StageLimitError,
)

__all__ = [
    "AXIS_UNITS",
    "Axes",
    "Axis",
    "BEAM_ROUTES",
    "CHAMBER_COMMAND_ROUTES",
    "CHAMBER_RESOURCE",
    "CHAMBER_ROUTES",
    "Chamber",
    "STAGE_COMMAND_ROUTES",
    "STAGE_RESOURCE",
    "STAGE_ROUTES",
    "GIS_RESOURCE",
    "GasInjector",
    "IMAGING_CHANNEL",
    "MANIPULATOR_RESOURCE",
    "MANIPULATOR_ROUTES",
    "Manipulator",
    "CommandInfo",
    "Beam",
    "BoundParameter",
    "KeyRouter",
    "Device",
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
