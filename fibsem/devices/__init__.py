"""A prototype of the microscope as a container of devices.

Nothing in fibsem uses this package yet, and ``FibsemMicroscope`` is unchanged. It
shows the device model on the Demo backend: parameters that describe themselves,
actions, named shared resources, and a key router that sends today's
``get``/``set`` keys to parameters without changing what the old calls do.

    from fibsem import utils
    from fibsem.devices import KeyRouter, bind_demo_beams
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
    sem.actions["acquire"].signature        # "(image_settings=None)"

    router = KeyRouter(microscope, beams)
    router.get("current", BeamType.ELECTRON) == microscope.get("current", BeamType.ELECTRON)
"""

from fibsem.devices.beam import BEAM_ROUTES, Beam, KeyRouter
from fibsem.devices.core import (
    IMAGING_CHANNEL,
    ActionInfo,
    BoundParameter,
    Device,
    Parameter,
    ParameterReadOnly,
    ParameterUnavailable,
    ParamMeta,
    Resources,
    action,
)
from fibsem.devices.demo import DemoBeam, bind_demo_beams

__all__ = [
    "BEAM_ROUTES",
    "IMAGING_CHANNEL",
    "ActionInfo",
    "Beam",
    "BoundParameter",
    "KeyRouter",
    "DemoBeam",
    "Device",
    "ParamMeta",
    "Parameter",
    "ParameterReadOnly",
    "ParameterUnavailable",
    "Resources",
    "action",
    "bind_demo_beams",
]
