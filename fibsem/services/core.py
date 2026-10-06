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

from fibsem.devices.core import _Controllable


class Service(_Controllable):
    """A capability of the instrument that uses devices: named, with parameters,
    commands and roles, and a parent."""
