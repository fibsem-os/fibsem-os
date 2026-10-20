"""The App Server's devices and services, reached by name from the server's own
descriptions, their values and results read as the types the server declares.

A new device, parameter, command or query on the server reaches the client with no
change here.
"""

import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from psygnal import Signal
from pydantic import PydanticSchemaGenerationError

from fibsem.cancellation import OperationCancelledError
from fibsem.client.transport import ServerError, Transport, error_for
from fibsem.codec import decode
from fibsem.devices.wire import to_wire, type_from_path

RUN_WAIT_S = 60.0  # the most the server waits on a run in one answer
RUN_TIMEOUT_S = 3600.0


def typed(path: Optional[str], data: Any) -> Any:
    """JSON *data* read as the type *path* names (`type_path`). A type that never
    travels as JSON (an image) comes back as the JSON the server sent for it:
    its summary."""
    if path is None:
        return data
    try:
        return decode(type_from_path(path), data)
    except PydanticSchemaGenerationError:
        return data


class Run:
    """A long command's run. It runs on the server whatever the client does: `wait`
    for its result, or read its `state`."""

    def __init__(self, remote: "Remote", answer: Dict[str, Any]):
        self._remote = remote
        self.run_id: str = answer["run_id"]
        self.command: str = answer["command"]
        self.answer = answer

    @property
    def state(self) -> str:
        """``running``, ``done``, ``cancelled`` or ``failed``, as the server saw it
        last."""
        return self.answer["state"]

    def refresh(self, wait: float = 0.0) -> str:
        """Read the run again, waiting up to *wait* seconds (at most a minute) for
        it to end. Its state."""
        answer = self._remote._transport.fetch(
            "GET",
            f"{self._remote.path}/run",
            params={"run_id": self.run_id, "wait": min(wait, RUN_WAIT_S)},
            timeout=min(wait, RUN_WAIT_S) + 30.0,
        )["run"]
        if answer["run_id"] != self.run_id:
            raise RuntimeError(f"{self} was replaced by run {answer['run_id']}")
        self.answer = answer
        return self.state

    def wait(self, timeout: float = RUN_TIMEOUT_S) -> Any:
        """Its result, once it ends. A run that raised raises again, as the
        exception its error names (a stop as `OperationCancelledError`); a run
        stopped without raising (a mill answering ``cancelled``) returns that
        answer. `TimeoutError` past *timeout*, the run still running."""
        deadline = time.monotonic() + timeout
        while self.state == "running":
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{self} is still running")
            self.refresh(wait=left)
        error = self.answer.get("error")
        if error is None:
            return typed(self._remote.returns(self.command), self.answer["result"])
        raised = error_for(str(self), None, error)
        if isinstance(raised, ServerError) and self.state == "cancelled":
            raised = OperationCancelledError(error.get("message"))
        raise raised

    def __repr__(self) -> str:
        return f"{self._remote.name}.{self.command} run {self.run_id}"


class Remote:
    """One device or service on the server, as its description says.

    Its commands and queries are methods: ``remote.acquire(image_settings=...)``.
    A long command answers at once with its `Run`.

    Its signals fire once the client follows the server's events
    (`FibsemClient.follow`), for a change whoever made it, on the stream's thread.
    """

    changed = Signal(str, object)
    """(parameter, value) read as the parameter's type."""

    metadata_changed = Signal(str, object)
    """(parameter, metadata) as `metadata` reads it."""

    signalled = Signal(str, object)
    """(signal, value) for one of the server's ``remote_signals`` (a milling
    reading, the holder), its value as the server sent it."""

    def __init__(
        self, transport: Transport, root: str, name: str, description: Dict[str, Any]
    ):
        self._transport = transport
        self.name = name
        self.path = f"{root}/{name}"
        self.description = description

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return self.description.get("parameters", {})

    @property
    def commands(self) -> Dict[str, Dict[str, Any]]:
        return self.description.get("commands", {})

    @property
    def queries(self) -> Dict[str, Dict[str, Any]]:
        return self.description.get("queries", {})

    def _value_type(self, parameter: str) -> Optional[str]:
        return self.parameters.get(parameter, {}).get("type_path")

    def returns(self, name: str) -> Optional[str]:
        """What a command or query returns, as its type's path."""
        return {**self.queries, **self.commands}.get(name, {}).get("returns")

    def get(self, parameter: str) -> Any:
        """A live read."""
        value = self._transport.fetch("GET", f"{self.path}/{parameter}")["value"]
        return typed(self._value_type(parameter), value)

    def set(self, parameter: str, value: Any) -> Any:
        """Write a parameter; the server checks it and answers the value written."""
        answer = self._transport.fetch(
            "PUT", f"{self.path}/{parameter}", {"value": to_wire(value)}
        )
        return typed(self._value_type(parameter), answer["value"])

    def values(self) -> Dict[str, Any]:
        """Every parameter's value in one read; those this backend lacks are left
        out."""
        answer = self._transport.fetch("GET", f"{self.path}/values")["values"]
        return {name: typed(self._value_type(name), v) for name, v in answer.items()}

    def report(self, event: Mapping[str, Any]) -> None:
        """Emit the signal a server event names."""
        name, value = event["parameter"], event["value"]
        if event["kind"] == "changed":
            self.changed.emit(name, typed(self._value_type(name), value))
        elif event["kind"] == "metadata":
            self.metadata_changed.emit(name, value)
        elif event["kind"] == "signal":
            self.signalled.emit(name, value)

    def metadata(self, parameter: str) -> Dict[str, Any]:
        """A parameter's limits, choices and whether it is settable, now."""
        return self._transport.fetch("GET", f"{self.path}/{parameter}/metadata")

    def call(self, command: str, /, reason: Optional[str] = None, **kwargs: Any) -> Any:
        """Run a command: its result, a `FibsemImage` for a beam's image, an array
        for a camera's frame, or a `Run` for a long command. A stop takes a
        *reason*, which the operator reads."""
        body: Dict[str, Any] = {"kwargs": _wire(kwargs)}
        if reason is not None:
            body["reason"] = reason
            if "reason" in self.commands.get(command, {}).get("arguments", {}).get(
                "properties", {}
            ):
                body["kwargs"]["reason"] = reason  # a command that records it too
        answer = self._transport.fetch("POST", f"{self.path}/commands/{command}", body)
        if not isinstance(answer, dict):  # an image, an array: data, not JSON
            return answer
        if self.commands.get(command, {}).get("long"):
            return Run(self, answer["result"])
        return typed(self.returns(command), answer["result"])

    def ask(self, query: str, /, **kwargs: Any) -> Any:
        """Ask a query: a read that takes arguments and moves nothing."""
        answer = self._transport.fetch(
            "POST", f"{self.path}/queries/{query}", {"kwargs": _wire(kwargs)}
        )
        return typed(self.returns(query), answer["result"])

    def latest_run(self) -> Optional[Run]:
        """The latest run of a long command, whoever started it; None when none
        has run."""
        try:
            answer = self._transport.fetch("GET", f"{self.path}/run")["run"]
        except ServerError as error:
            if error.status == 404 and error.error_type == "not_found":
                return None
            raise
        return Run(self, answer)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.commands:
            return partial(self.call, name)
        if name in self.queries:
            return partial(self.ask, name)
        raise AttributeError(
            f"{self.name} has no command or query '{name}'; it has "
            f"{', '.join(sorted({**self.commands, **self.queries})) or 'none'}"
        )

    def __dir__(self) -> List[str]:
        return sorted({*super().__dir__(), *self.commands, *self.queries})

    def __repr__(self) -> str:
        return (
            f"<{self.description.get('class', 'Remote')} {self.name!r} at {self.path}>"
        )


class Remotes(Mapping[str, Remote]):
    """Every device or service under *root*, by name, described once on first use;
    `refresh` reads the descriptions again."""

    def __init__(self, transport: Transport, root: str):
        self._transport = transport
        self.root = root
        self._remotes: Optional[Dict[str, Remote]] = None

    def refresh(self) -> None:
        listing = self._transport.fetch("GET", self.root)
        self._remotes = {
            name: Remote(self._transport, self.root, name, description)
            for name, description in listing.items()
        }

    def _all(self) -> Dict[str, Remote]:
        if self._remotes is None:
            self.refresh()
        return self._remotes

    def __getitem__(self, name: str) -> Remote:
        try:
            return self._all()[name]
        except KeyError:
            raise KeyError(
                f"no '{name}' at {self.root}; there are {', '.join(self._all())}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._all())


def _wire(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return {name: to_wire(value) for name, value in kwargs.items()}
