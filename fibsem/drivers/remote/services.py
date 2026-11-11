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

A method the service lists as ``client_side`` runs here, on this class (an
estimate, an image sent to this app's views). A property it lists in
``remote_properties`` is read with its query (``stage_movement.orientations``).

Not yet: an image from a long command comes back as the server's summary of it
(its saved path), not the image.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import io
import logging
import math
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Type

from psygnal import Signal

from fibsem import config as fcfg
from fibsem.cancellation import OperationCancelledError
from fibsem.client.microscope import Remote, Run, typed
from fibsem.devices.core import DeviceParameter
from fibsem.drivers.remote.devices import (
    READ_TIMEOUT,
    UNTIL_DONE,
    DeviceClient,
    RemoteParts,
)
from fibsem.milling.progress import MillingProgressStatus
from fibsem.services.core import _CLASSES, Service
from fibsem.services.imaging import Imaging, acquisition_record
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import BeamType
from fibsem.util import timestamps

STOP_POLL_S = 0.1  # how often a long command's stop event is looked at
STOP_RETRY_S = 0.5  # how long after a refused stop it is asked again


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
            for image in _images(answer):
                self.session.stamp(image)  # this computer's user and experiment
            return answer
        info = self._description["commands"].get(command, {})
        if not info.get("long"):
            return typed(info.get("returns"), answer["result"])
        run = Run(self._remote(), answer["result"])
        self.last_run_id = run.run_id
        if not stops:
            return run.wait(timeout=math.inf)
        ended = threading.Event()

        def watch() -> None:
            while not ended.wait(STOP_POLL_S):
                if any(event.is_set() for event in stops):
                    try:
                        self.call_command("stop")
                        return
                    except Exception as error:  # noqa: BLE001
                        # Refused before it mills (nothing to stop yet): again.
                        logging.debug(f"{self.name}.stop: {error}")
                        ended.wait(STOP_RETRY_S)

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
    if local is not None:
        for method in local.client_side:
            namespace[method] = inspect.getattr_static(local, method)
    namespace.update(_SAVED_HERE.get(name, {}))
    if local is not None:
        queries = description.get("queries", {})
        for attr, query in local.remote_properties.items():
            if query in queries:
                namespace[attr] = property(lambda self, q=query: self.call_query(q))
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


def _images(result: Any) -> List[FibsemImage]:
    """The FibsemImages in a command's *result*: it, or those in a list of them."""
    if isinstance(result, FibsemImage):
        return [result]
    if isinstance(result, (list, tuple)):
        return [image for item in result for image in _images(item)]
    return []


def _acquire_saved_here(
    self: RemoteService,
    beam_type: BeamType = BeamType.ELECTRON,
    image_settings: Any = None,
    path: Optional[str] = None,
    filename: Optional[str] = None,
) -> FibsemImage:
    """`Imaging.acquire` on a remote microscope: acquired on the server, saved in
    *path* on this computer and recorded here."""
    name = filename or f"image-{timestamps.now():%Y%m%d-%H%M%S}"
    if os.path.basename(name) != name or name in (".", ".."):
        raise ValueError(f"filename is a name, not a path: {name!r}")
    image = self.call_command(
        "acquire", beam_type=beam_type, image_settings=image_settings
    )
    if path is None:
        return image
    settings = image.metadata.image_settings if image.metadata else None
    beam = settings.beam_type if settings and settings.beam_type else beam_type
    suffix = "eb" if beam is BeamType.ELECTRON else "ib"
    saved = image.save(path=os.path.join(path, f"{name}_{suffix}"))
    self.session.record("image_acquired", acquisition_record(image, saved))
    return image


functools.update_wrapper(_acquire_saved_here, Imaging.acquire)
_acquire_saved_here.__dict__.pop("__wrapped__", None)
_acquire_saved_here.__name__ = "acquire"


class ServerMillingTask:
    """A milling task run on the server for a remote microscope, as
    `run_milling_task` answers one: how it ended, its folder brought back here."""

    def __init__(self, config: Any, output_dir: Optional[str]):
        self.config = config
        self.name = config.name
        root = output_dir or config.acquisition.imaging.path or fcfg.DATA_CC_PATH
        self.output_dir = os.path.join(
            str(root), "Milling", self.name.replace(" ", "-")
        )
        self.task_id: Optional[str] = None
        self.error: Optional[BaseException] = None
        self.outcome = MillingProgressStatus.TASK_FAILED
        self.images: List[FibsemImage] = []
        self._ended: Dict[str, Any] = {}

    def ended(self) -> Dict[str, Any]:
        """How it ended, as the server's run answered, its ``path`` this computer's
        copy of the folder."""
        return {**self._ended, "path": self.output_dir}


_OUTCOMES = {
    "completed": MillingProgressStatus.TASK_FINISHED,
    "skipped": MillingProgressStatus.TASK_FINISHED,
    "cancelled": MillingProgressStatus.TASK_CANCELLED,
}


def _run_task_on_server(
    self: RemoteService,
    config: Any,
    stop_event: Optional[threading.Event] = None,
    raise_on_failure: bool = False,
    output_dir: Optional[str] = None,
) -> ServerMillingTask:
    """`run_milling_task` on a remote microscope: the whole task runs on the
    server (``run_task``), saving into its data folder, and that folder comes
    back here, under *output_dir* as a local task saves, and is removed there."""
    task = ServerMillingTask(config, output_dir)
    self.last_run_id = None
    stops = {"stop_event": stop_event} if stop_event is not None else {}
    try:
        task._ended = self.call_command("run_task", config=config, path=None, **stops)
        task.outcome = _OUTCOMES.get(task._ended.get("status"), task.outcome)
    except OperationCancelledError:
        task.outcome = MillingProgressStatus.TASK_CANCELLED
        task._ended = {"status": "cancelled"}
    except Exception as error:  # noqa: BLE001 - answered as a local task's error
        task.error = error
        task._ended = {"status": "failed", "error": str(error)}
    task.task_id = self.last_run_id
    if task.task_id is not None:
        try:
            _unpack(self._folder_of(task.task_id), task.output_dir)
            task.images = _saved_images(task.output_dir)
        except Exception as error:  # noqa: BLE001 - the mill itself is done
            logging.warning(f"{task.name}: its folder did not come back: {error}")
    if raise_on_failure and task.error is not None:
        raise task.error
    return task


def _folder_of(self: RemoteService, run_id: str) -> bytes:
    path = f"{self.collection}/{self.name}/run/files?run_id={run_id}&remove=true"
    return self.client.request("GET", path, READ_TIMEOUT)


def _unpack(content: bytes, folder: str) -> None:
    import zipfile

    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.namelist():
            target = os.path.realpath(os.path.join(folder, member))
            if os.path.commonpath([os.path.realpath(folder), target]) != (
                os.path.realpath(folder)
            ):
                continue  # never outside the task's folder
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as out:
                out.write(archive.read(member))


def _saved_images(folder: str) -> List[FibsemImage]:
    """The beam images a task saved in *folder*, in the order they were saved."""
    paths = []
    for root, _, files in os.walk(folder):
        paths += [os.path.join(root, f) for f in files if f.endswith(".tif")]
    images = []
    for path in sorted(paths, key=os.path.getmtime):
        try:
            images.append(FibsemImage.load(path))
        except Exception:  # noqa: BLE001 - not a beam image (a monitor's frame)
            continue
    return images


# A served command a remote microscope runs its own way: one that saves, saving
# on this computer, since the paths given are this computer's.
_SAVED_HERE: Dict[str, Dict[str, Callable]] = {
    "imaging": {"acquire": _acquire_saved_here},
    "milling": {"run_task_on_server": _run_task_on_server, "_folder_of": _folder_of},
}


def connect_remote_services(client: DeviceClient) -> Dict[str, RemoteService]:
    """Every service the App Server behind *client* serves, by name, connected."""
    descriptions = client.request("GET", "services", READ_TIMEOUT)
    return {
        name: remote_service_class(name, description)(name, client=client).connect(
            description
        )
        for name, description in descriptions.items()
    }
