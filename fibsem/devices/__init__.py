"""A prototype of the microscope as a container of devices.

Nothing in fibsem uses this package yet, and ``FibsemMicroscope`` is unchanged. It
shows the device model on the Demo backend: parameters that describe themselves,
actions, named shared resources, and a compatibility front that routes today's
``get``/``set`` keys to parameters without changing what the old calls do.

    from fibsem import utils
    from fibsem.devices import CompatibilityFront, bind_demo_beams
    from fibsem.structures import BeamType

    microscope, _ = utils.setup_session(manufacturer="Demo")
    beams = bind_demo_beams(microscope)
    sem = beams[BeamType.ELECTRON]

    sem.current.choices                     # cached metadata, no instrument call
    sem.current.changed.connect(print)      # every change, with the value
    sem.current.set(1e-9)                   # the new API: checked, then written
    sem.scan_rotation.set(7.0)              # clipped to 2*pi, with a warning
    "preset" in sem.parameters              # False: Demo has no presets
    sem.actions["acquire"].signature        # "(image_settings=None)"

    front = CompatibilityFront(microscope, beams)
    front.get("current", BeamType.ELECTRON) == microscope.get("current", BeamType.ELECTRON)
"""

from fibsem.devices.beam import BEAM_ROUTES, Beam, CompatibilityFront
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
from fibsem.devices.demo import bind_demo_beams

__all__ = [
    "BEAM_ROUTES",
    "IMAGING_CHANNEL",
    "ActionInfo",
    "Beam",
    "BoundParameter",
    "CompatibilityFront",
    "Device",
    "ParamMeta",
    "Parameter",
    "ParameterReadOnly",
    "ParameterUnavailable",
    "Resources",
    "action",
    "bind_demo_beams",
]
