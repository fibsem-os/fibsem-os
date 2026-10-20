"""A client for the App Server.

::

    from fibsem.client import FibsemClient

    with FibsemClient() as client:  # the App Server on this machine
        image = client.devices["electron"].acquire()
        client.devices["ion"].set("current", 1e-9)
        run = client.services["milling"].run()
        run.wait()

Devices and services are reached by name from the server's descriptions. With no
url and token it finds the App Server on this machine (`transport.resolve`). What a
call may do is what its token may: the agent token's level gates it.
"""

from typing import Any, Dict, Optional

from fibsem.client.microscope import Remote, Remotes, Run
from fibsem.client.transport import ServerError, ServerUnreachable, Transport
from fibsem.structures import MicroscopeState, SystemSettings

__all__ = [
    "FibsemClient",
    "Remote",
    "Run",
    "ServerError",
    "ServerUnreachable",
    "Transport",
]


class FibsemClient:
    """The App Server's microscope: ``devices``, the microscope's ``services``
    (milling, stage movement, imaging, ...) and the ``analysis`` services, which
    read saved data and are served with or without a microscope."""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        user_agent: Optional[str] = None,
        transport: Optional[Transport] = None,
    ):
        """*transport*, when given, is the connection to use (one already open),
        and *url*, *token* and *user_agent* are not read."""
        self.transport = transport or Transport(url, token, user_agent=user_agent)
        self.devices = Remotes(self.transport, "/microscope/devices")
        self.services = Remotes(self.transport, "/microscope/services")
        self.analysis = Remotes(self.transport, "/services")
        self._system: Optional[SystemSettings] = None

    def capabilities(self) -> Dict[str, Any]:
        """What the server serves, and the agent level."""
        return self.transport.request("GET", "/capabilities")

    def health(self) -> Dict[str, Any]:
        return self.transport.request("GET", "/health")

    def info(self) -> Dict[str, Any]:
        """Who the microscope is: manufacturer, model, serial number, software,
        what is fitted, and its configuration."""
        return self.transport.request("GET", "/microscope")

    @property
    def system(self) -> SystemSettings:
        """The microscope's configuration, read once."""
        if self._system is None:
            info = self.info()
            system = SystemSettings.from_dict(info["system"])
            # What is fitted travels beside the configuration.
            fitted = info.get("fitted") or {}
            if "manipulator" in fitted:
                system.manipulator.enabled = bool(fitted["manipulator"])
            self._system = system
        return self._system

    def state(self) -> MicroscopeState:
        answer = self.transport.request("GET", "/microscope/state")
        return MicroscopeState.from_dict(answer["microscope_state"])

    def set_state(self, state: MicroscopeState) -> None:
        """Restore beams and stage, in their set order."""
        self.transport.request(
            "PUT", "/microscope/state", {"microscope_state": state.to_dict()}
        )

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "FibsemClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
