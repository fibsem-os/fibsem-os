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

import logging
from typing import Any, Callable, Dict, Sequence

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


def forward_to(signal: Any) -> Callable[[Any], None]:
    """A slot that emits *signal* with each value it gets, to connect a service's
    ``progress.changed`` to a microscope signal.

    Not ``signal.emit`` itself: psygnal checks an ``emit`` it is given by calling it
    once with junk arguments, which the microscope signal would send on.
    """

    def forward(value: Any) -> None:
        signal.emit(value)

    return forward
