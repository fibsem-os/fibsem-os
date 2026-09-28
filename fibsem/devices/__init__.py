"""A prototype of the microscope as a container of devices.

Nothing in fibsem uses this package yet, and ``FibsemMicroscope`` is unchanged. This
package is the vendor-neutral API, and backend implementations live in
``fibsem.devices.drivers``. It shows the device model on the Demo backend:
parameters that describe themselves, commands, named shared resources, and a key
router that sends today's ``get``/``set`` keys to parameters without changing what
the old calls do.

    from fibsem import utils
    from fibsem.devices import KeyRouter
    from fibsem.devices.drivers.demo import bind_demo_beams
    from fibsem.structures import BeamType

    microscope, _ = utils.setup_session(manufacturer="Demo")
    beams = bind_demo_beams(microscope)
    sem = beams[BeamType.ELECTRON]

    sem.current.choices                     # cached metadata, no instrument call
    sem.current.changed.connect(print)      # every change, with the value
    sem.current.set_value(1e-9)             # the new API: checked, then written
    sem.scan_rotation.set_value(7.0)        # clipped to 2*pi, with a warning
    sem.hfw.value = 100e-6                  # shorthand for set_value / get_value
    "preset" in sem.parameters              # False: Demo has no presets
    sem.commands["acquire"].signature        # "(image_settings=None)"

    router = KeyRouter(microscope, beams)
    router.get("current", BeamType.ELECTRON) == microscope.get("current", BeamType.ELECTRON)

The stage is a device the same way, with moves as commands:

    from fibsem.devices.drivers.demo import bind_demo_stage
    from fibsem.structures import FibsemStagePosition

    stage = bind_demo_stage(microscope)
    list(stage.axes)                        # ["x", "y", "z", "r", "t"]; no r on a compustage
    stage.axes.t.limits                     # radians, cached at connect
    stage.axes.t.cached                     # stage.position.cached.t, no instrument call
    stage.position.changed.connect(print)   # every move, and every read that differs
    stage.axes.t.changed.connect(print)     # only when t moved
    stage.move_absolute(FibsemStagePosition(x=1e-3))   # refused outside the limits
    stage.move_through(FibsemStagePosition(x=1e-3))    # the old API's move: no new check
    stage.home()

    router = KeyRouter(microscope, beams, stage=stage)
    router.get("stage_position") == microscope.get("stage_position")
"""

from fibsem.devices.beam import (
    BEAM_ROUTES,
    STAGE_COMMAND_ROUTES,
    STAGE_ROUTES,
    Beam,
    KeyRouter,
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
    "STAGE_COMMAND_ROUTES",
    "STAGE_RESOURCE",
    "STAGE_ROUTES",
    "IMAGING_CHANNEL",
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
