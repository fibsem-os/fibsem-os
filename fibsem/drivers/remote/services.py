"""The microscope's services on an App Server, used as if they were local.

A `RemoteService` has the service's parameters, commands, queries and signals as
the server describes them, and runs every command on the server: ``milling.run``
mills with the server's run, not a local copy of its logic over remote devices.

    services = connect_remote_services(client)
    services["milling"].patterns_changed.connect(...)

Its class is made from the service's own (``fibsem.services.core._CLASSES``), so
its methods take the same arguments, positional or by name, and its signals are
the same signals. Only what the server serves crosses: a method the service has
but doesn't serve as a command or query is not there.

A long command (a mill, an overview) waits on its run, as a local call does, and
raises as the run did. A stop event passed to it stays here: setting it stops the
run on the server, and the call raises `OperationCancelledError`.

Not yet: a result that is an image list, or an image from a long command, comes
back as the server's summary of it (its saved path), not the image.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import math
import threading
from typing import Any, Callable, Dict, Optional, Type

from psygnal import Signal

from fibsem.cancellation import OperationCancelledError
from fibsem.client.microscope import Remote, Run, typed
from fibsem.devices.core import DeviceParameter
from fibsem.drivers.remote.devices import (
    READ_TIMEOUT,
    UNTIL_DONE,
    DeviceClient,
    RemoteParts,
)
from fibsem.services.core import _CLASSES, Service

STOP_POLL_S = 0.1  # how often a long command's stop event is looked at


class RemoteService(RemoteParts, Service):
    """A service served by the App Server under ``/microscope/services``."""

    noun = "service"

    def connect(self, description: Optional[Dict[str, Any]] = None) -> "RemoteService":
        if description is None:
            description = self.client.request(
                "GET", f"{self.collection}/{self.name}", READ_TIMEOUT
            )
        self._description = description
        return super().connect(description)

    def call_command(
        self, command: str, timeout: Any = UNTIL_DONE, **kwargs: Any
    ) -> Any:
        """Run *command* on the server: its result read as the type it declares.
        A long one waits on its run."""
        # A stop event stays here: setting it stops the run on the server.
        stops = [v for v in kwargs.values() if callable(getattr(v, "is_set", None))]
        kwargs = {k: v for k, v in kwargs.items() if not any(v is e for e in stops)}
        if any(event.is_set() for event in stops):
            raise OperationCancelledError(f"{self.name}.{command} was stopped")
        answer = self._post_command(command, timeout, kwargs)
        if not isinstance(answer, dict):  # an image, an array, a list of frames
            return answer
        info = self._description["commands"].get(command, {})
        if not info.get("long"):
            return typed(info.get("returns"), answer["result"])
        run = Run(self._remote(), answer["result"])
        if not stops:
            return run.wait(timeout=math.inf)
        ended = threading.Event()

        def watch() -> None:
            while not ended.wait(STOP_POLL_S):
                if any(event.is_set() for event in stops):
                    self.call_command("stop")
                    return

        threading.Thread(target=watch, name=f"{self.name}-stop", daemon=True).start()
        try:
            result = run.wait(timeout=math.inf)
        finally:
            ended.set()
        if any(event.is_set() for event in stops):  # as a local run raises
            raise OperationCancelledError(f"{self.name}.{command} was stopped")
        return result

    def call_query(self, query: str, timeout: Any = READ_TIMEOUT, **kwargs: Any) -> Any:
        answer = super().call_query(query, timeout=timeout, **kwargs)
        return typed(self._description["queries"].get(query, {}).get("returns"), answer)

    def _remote(self) -> Remote:
        """The service as `fibsem.client` reaches it, for a long command's run."""
        return Remote(
            self.client._transport,
            f"{self.client.root}/{self.collection}",
            self.name,
            self._description,
        )

    def signalled(self, name: str, value: Any) -> None:
        signal = getattr(self, name, None)
        if signal is not None and name in type(self).remote_signals:
            signal.emit(value)  # as the server sent it: JSON


def remote_service_class(name: str, description: Dict[str, Any]) -> Type[RemoteService]:
    """The remote class for service *name*: the service's own parameters, signals
    and the signatures of the commands and queries the server serves. A service
    this client doesn't know gets the commands and queries without signatures."""
    where = _CLASSES.get(name)
    local: Optional[type] = None
    if where is not None:
        module, _, attr = where.partition(":")
        local = getattr(importlib.import_module(module), attr)
    namespace: Dict[str, Any] = {}
    if local is not None:
        for klass in reversed(local.__mro__):
            for attr, value in vars(klass).items():
                if isinstance(value, (Signal, DeviceParameter)):
                    namespace[attr] = value
        namespace["remote_signals"] = local.remote_signals
    for kind in ("commands", "queries"):
        for method, info in description.get(kind, {}).items():
            real = getattr(local, method, None) if local is not None else None
            namespace[method] = _forward(method, kind, info, real)
    title = local.__name__ if local is not None else name.title().replace("_", "")
    return type(f"Remote{title}", (RemoteService,), namespace)


def _forward(
    name: str, kind: str, info: Dict[str, Any], real: Optional[Callable]
) -> Callable:
    """A method that runs *name* on the server, taking *real*'s arguments."""
    signature = inspect.signature(real) if real is not None else None

    def method(self: RemoteService, *args: Any, **kwargs: Any) -> Any:
        if signature is not None:
            bound = signature.bind(self, *args, **kwargs).arguments
            bound.pop(next(iter(signature.parameters)))  # self
            kwargs = {}
            for key, value in bound.items():
                if signature.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                    kwargs.update(value)
                else:
                    kwargs[key] = value
        elif args:
            raise TypeError(f"{name} takes its arguments by name")
        if kind == "commands":
            return self.call_command(name, **kwargs)
        return self.call_query(name, **kwargs)

    if real is not None:
        functools.update_wrapper(method, real)
        method.__dict__.pop("__wrapped__", None)
        method.__dict__.pop("__device_long__", None)
    method.__name__ = name
    # Listed as the server lists it, never tracked as a local run.
    for mark in ("__device_command__", "__device_query__"):
        method.__dict__.pop(mark, None)
    setattr(
        method, "__device_command__" if kind == "commands" else "__device_query__", None
    )
    if info.get("long"):
        method.__device_long__ = True  # type: ignore[attr-defined]
    if signature is not None:
        method.__signature__ = signature  # type: ignore[attr-defined]
    return method


def connect_remote_services(client: DeviceClient) -> Dict[str, RemoteService]:
    """Every service the App Server behind *client* serves, by name, connected."""
    descriptions = client.request("GET", "services", READ_TIMEOUT)
    return {
        name: remote_service_class(name, description)(name, client=client).connect(
            description
        )
        for name, description in descriptions.items()
    }
