"""Services: capabilities of the instrument that use devices over time.

A service has the device shape (parameters, commands, roles, change events, the
microscope's resources) but is not hardware, so it is not a `Device` and is not one of
the microscope's devices. The microscope holds each service as an attribute
(``microscope.milling``), built by its driver the way devices are.

The rule of thumb: one beam's own operation is a command on the beam (imaging, the scan
modes); anything that coordinates devices or runs steps over time is a service
(milling; later, spot burn and stage movement). A service reaches the devices it uses
through roles (``beam = Role(Beam)``), so it doesn't care which driver's device fills
them.
"""

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Sequence

from fibsem.devices.core import _Controllable


class Service(_Controllable):
    """A capability of the instrument that uses devices: named, with parameters,
    commands and roles, and a parent."""


def save_beam_conditions(beam: Any, names: Sequence[str]) -> Dict[str, Any]:
    """The conditions in *names* the beam has, can set, and reports, to write back
    after a service has changed them.

    A condition the beam can't set (a Tescan ion column's current and voltage come
    with its preset) is left out, and so is one it reads as None.
    """
    saved = {}
    for name in names:
        if name in beam.parameters and getattr(beam, name).settable:
            value = getattr(beam, name).get_value()
            if value is not None:
                saved[name] = value
    logging.debug({"msg": "saved the beam", "beam": beam.name, "saved": saved})
    return saved


# Each service the microscope hosts, by the name it serves it under: where its
# class is, for what is declared on it without a microscope (a command's line).
_CLASSES = {
    "milling": "fibsem.services.milling:Milling",
    "stage_movement": "fibsem.services.stage_movement:StageMovement",
    "spot_burn": "fibsem.services.spot_burn:SpotBurn",
    "sample_handling": "fibsem.services.sample_handling:SampleHandling",
    "imaging": "fibsem.services.imaging:Imaging",
    "alignment": "fibsem.services.alignment:Alignment",
}


def command_line(service: str, command: str) -> Optional[Callable[..., Dict[str, Any]]]:
    """The line a service's command declares (``command(line=...)``), to say a
    call to it in an agent's request; None for a command with none."""
    where = _CLASSES.get(str(service))
    if where is None:
        return None
    module, _, name = where.partition(":")
    cls = getattr(importlib.import_module(module), name)
    return getattr(getattr(cls, str(command), None), "__command_line__", None)
