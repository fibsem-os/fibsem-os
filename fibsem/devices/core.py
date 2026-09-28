"""Devices, self-describing parameters, actions and shared resources.

A device declares which parameters it can have on its class (type and unit). A backend
binds the ones its hardware has to two small functions, read and write, and supplies
their metadata (limits, choices, settable) once, at bind time. Everything a backend
repeats today (checking, logging, caching, change events) happens here, once.

Two paths reach the hardware:

- the new API, ``param.set_value(v)``: type and choice checks raise, numeric values clip to
  their limits with a warning, and then the write path runs;
- the old API, through a compatibility front, ``param.write_through(v)``: no new
  validation, so each backend keeps its current behaviour, and then the same write path.

The write path claims the parameter's resource, selects the imaging channel when the
backend says the parameter needs it, makes the write, caches the value, emits
``changed`` and refreshes the metadata of parameters that depend on it.
"""

from __future__ import annotations

import inspect
import logging
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from psygnal import Signal

IMAGING_CHANNEL = "imaging_channel"
_SHARED = "shared"
_UNSET = object()


class ParameterUnavailable(AttributeError):
    """The device declares the parameter, but this backend did not bind it."""


class ParameterReadOnly(Exception):
    """The parameter is bound but cannot be set."""


@dataclass(frozen=True)
class ParamMeta:
    """What a parameter allows. Filled from the vendor API once, then cached."""

    limits: Optional[Tuple[float, float]] = None
    choices: Optional[Sequence[Any]] = None
    settable: bool = True


class Parameter:
    """A parameter a device class may have. Declared on the class, bound per instance.

    ``limits`` and ``choices`` given here are static defaults; a backend's metadata
    replaces them. ``depends_on`` names parameters whose change alters this one's
    metadata (the ion current choices depend on the plasma gas).
    """

    def __init__(
        self,
        type_: type,
        unit: Optional[str] = None,
        limits: Optional[Tuple[float, float]] = None,
        choices: Optional[Sequence[Any]] = None,
        depends_on: Sequence[str] = (),
        doc: str = "",
    ):
        self.type = type_
        self.unit = unit
        self.limits = limits
        self.choices = choices
        self.depends_on = tuple(depends_on)
        self.doc = doc
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, device: Optional[Device], owner: type) -> Any:
        if device is None:
            return self
        try:
            return device._bound[self.name]
        except KeyError:
            raise ParameterUnavailable(
                f"{type(device).__name__} '{device.name}' has no '{self.name}' on this backend"
            ) from None

    def __repr__(self) -> str:
        unit = f", unit={self.unit!r}" if self.unit else ""
        return f"Parameter({self.type.__name__}{unit})"


MetaSource = Union[ParamMeta, Callable[[], ParamMeta], None]


class BoundParameter:
    """One parameter on one device, bound to a backend's read and write."""

    changed = Signal(object)
    """The new value, after every change: a set, a write through the old API, a live
    read that found a different value, or a change the backend reports."""

    meta_changed = Signal(object)
    """The new ParamMeta, after a parameter this one depends on changed."""

    def __init__(
        self,
        device: Device,
        spec: Parameter,
        read: Callable[[], Any],
        write: Optional[Callable[[Any], None]],
        meta: MetaSource,
        needs_channel: bool,
    ):
        self.device = device
        self.spec = spec
        self._read = read
        self._write = write
        self._meta_source = meta
        self.needs_channel = needs_channel
        self._cached: Any = _UNSET
        self.meta = ParamMeta()
        self.refresh_meta(emit=False)

    # -- description --------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def type(self) -> type:
        return self.spec.type

    @property
    def unit(self) -> Optional[str]:
        return self.spec.unit

    @property
    def limits(self) -> Optional[Tuple[float, float]]:
        return self.meta.limits

    @property
    def choices(self) -> Optional[Sequence[Any]]:
        return self.meta.choices

    @property
    def settable(self) -> bool:
        return self.meta.settable

    def refresh_meta(self, emit: bool = True) -> ParamMeta:
        """Read the metadata again from the backend. Called at bind and on dependencies."""
        source = self._meta_source
        meta = source() if callable(source) else (source or ParamMeta())
        limits = meta.limits if meta.limits is not None else self.spec.limits
        choices = meta.choices if meta.choices is not None else self.spec.choices
        settable = meta.settable and self._write is not None
        self.meta = replace(meta, limits=limits, choices=choices, settable=settable)
        if emit:
            self.meta_changed.emit(self.meta)
            self.device.meta_changed.emit(self.name, self.meta)
        return self.meta

    # -- reads --------------------------------------------------------------------

    @property
    def cached(self) -> Any:
        """The last known value, with no instrument call. For displays and UIs.

        Only the very first access, before anything was read or written, reads live.
        """
        if self._cached is _UNSET:
            return self.get_value()
        return self._cached

    def get_value(self) -> Any:
        """A live read from the instrument, for guards and anything that needs it now.

        It makes the same call the old branch made, and updates the cache.
        """
        with self.device._claim(self):
            value = self._read()
        self._remember(value)
        return value

    # -- writes -------------------------------------------------------------------

    def set_value(self, value: Any) -> Any:
        """The new API: validate, then write. Returns the value actually written.

        Wrong type or a value outside the choices raises. A number outside the limits
        is clipped to them, with a warning.
        """
        if not self.settable:
            raise ParameterReadOnly(f"{self.device.name}.{self.name} is read-only")
        value = self.validate(value)
        self._write_path(value)
        return value

    @property
    def value(self) -> Any:
        """Shorthand for scripts: reading is ``get_value()``, assigning is ``set_value()``.

        Both are live instrument calls. Assignment checks, clips and signals exactly as
        ``set_value`` does, but cannot return the clipped value; call ``set_value`` when
        the value actually written matters.
        """
        return self.get_value()

    @value.setter
    def value(self, value: Any) -> None:
        self.set_value(value)

    def write_through(self, value: Any) -> None:
        """The old API: the same write path with no new checks, so no behaviour changes."""
        if self._write is None:
            raise ParameterReadOnly(f"{self.device.name}.{self.name} is read-only")
        self._write_path(value)

    def report(self, value: Any) -> None:
        """A change the backend observed itself (vendor UI, another client, a push)."""
        self._remember(value)

    def validate(self, value: Any) -> Any:
        value = _coerce(self.type, value, f"{self.device.name}.{self.name}")
        if self.choices is not None:
            value = _match_choice(
                value, self.choices, f"{self.device.name}.{self.name}"
            )
        if self.limits is not None and isinstance(value, (int, float)):
            low, high = self.limits
            clipped = min(max(value, low), high)
            if clipped != value:
                logging.warning(
                    f"{self.device.name}.{self.name}: {value} is outside {self.limits}, "
                    f"set to {clipped}"
                )
                value = self.type(clipped)
        return value

    def _write_path(self, value: Any) -> None:
        with self.device._claim(self):
            self._write(value)
        logging.debug(
            {
                "msg": "set",
                "device": self.device.name,
                "param": self.name,
                "value": value,
            }
        )
        self._cached = value
        self._emit(value)
        self.device._dependency_changed(self.name)

    def _remember(self, value: Any) -> None:
        previous, self._cached = self._cached, value
        if previous is _UNSET or not _same(previous, value):
            if previous is not _UNSET:
                self._emit(value)

    def _emit(self, value: Any) -> None:
        self.changed.emit(value)
        self.device.changed.emit(self.name, value)

    def __repr__(self) -> str:
        cached = "?" if self._cached is _UNSET else repr(self._cached)
        return f"<{self.device.name}.{self.name} = {cached} {self.meta}>"


@dataclass(frozen=True)
class ActionInfo:
    name: str
    signature: str
    available: bool
    doc: str


def action(
    fn: Optional[Callable] = None, *, available: Optional[Callable[[Any], bool]] = None
) -> Any:
    """Mark a device method as an action. It stays a plain method.

    The mark records only what a UI, the agent API or a remote proxy needs to list it:
    its name, its arguments and whether it is available on this device right now.
    """

    def mark(method: Callable) -> Callable:
        method.__device_action__ = available  # type: ignore[attr-defined]
        return method

    return mark(fn) if fn is not None else mark


class Resources:
    """Named shared resources, each a re-entrant lock.

    Names not listed in ``groups`` share one lock, which is today's behaviour and
    always safe. A backend separates only what its hardware really allows in parallel.
    A device on another computer gets its own ``Resources`` and shares nothing.
    """

    def __init__(self, groups: Optional[Mapping[str, str]] = None):
        self._groups = dict(groups or {})
        self._locks: Dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def lock(self, name: str) -> threading.RLock:
        group = self._groups.get(name, _SHARED)
        with self._guard:
            if group not in self._locks:
                self._locks[group] = threading.RLock()
            return self._locks[group]

    @contextmanager
    def claim(self, name: str) -> Iterator[None]:
        with self.lock(name):
            yield


class Device:
    """One piece of hardware: named, with parameters and actions, and a parent."""

    changed = Signal(str, object)
    """(parameter name, value) for any parameter on the device."""

    meta_changed = Signal(str, object)
    """(parameter name, ParamMeta) when a parameter's metadata was refreshed."""

    def __init__(
        self,
        name: str,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        self.name = name
        self.parent = parent
        self.resources = resources if resources is not None else Resources()
        self._bound: Dict[str, BoundParameter] = {}
        self._select_channel: Optional[Callable[[], None]] = None

    # -- binding, done by a backend ------------------------------------------------

    @classmethod
    def declared_parameters(cls) -> Dict[str, Parameter]:
        found: Dict[str, Parameter] = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if isinstance(attr, Parameter):
                    found[name] = attr
        return found

    def bind(
        self,
        name: str,
        read: Callable[[], Any],
        write: Optional[Callable[[Any], None]] = None,
        meta: MetaSource = None,
        needs_channel: bool = False,
    ) -> BoundParameter:
        """Give a declared parameter to this device. Without ``write`` it is read-only."""
        spec = self.declared_parameters().get(name)
        if spec is None:
            raise AttributeError(
                f"{type(self).__name__} declares no parameter '{name}'"
            )
        param = BoundParameter(self, spec, read, write, meta, needs_channel)
        self._bound[name] = param
        return param

    def bind_channel(self, select: Callable[[], None]) -> None:
        """How to make this device the active imaging channel, for needs_channel params."""
        self._select_channel = select

    # -- description ----------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, BoundParameter]:
        """The parameters this backend bound. A declared one missing here is absent."""
        return dict(self._bound)

    @property
    def actions(self) -> Dict[str, ActionInfo]:
        found = {}
        for name, attr in inspect.getmembers(type(self), inspect.isfunction):
            if not hasattr(attr, "__device_action__"):
                continue
            available = attr.__device_action__
            signature = inspect.signature(attr)
            arguments = list(signature.parameters.values())[1:]  # without self
            found[name] = ActionInfo(
                name=name,
                signature=str(signature.replace(parameters=arguments)),
                available=True if available is None else bool(available(self)),
                doc=inspect.getdoc(attr) or "",
            )
        return found

    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Plain data for every bound parameter: enough for a UI or a remote proxy."""
        return {
            name: {
                "type": p.type.__name__,
                "unit": p.unit,
                "limits": p.limits,
                "choices": list(p.choices) if p.choices is not None else None,
                "settable": p.settable,
            }
            for name, p in self._bound.items()
        }

    # -- internals -----------------------------------------------------------------

    @contextmanager
    def _claim(self, param: BoundParameter) -> Iterator[None]:
        if not param.needs_channel:
            yield
            return
        with self.resources.claim(IMAGING_CHANNEL):
            if self._select_channel is not None:
                self._select_channel()
            yield

    def _dependency_changed(self, name: str) -> None:
        for param in self._bound.values():
            if name in param.spec.depends_on:
                param.refresh_meta()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}': {sorted(self._bound)}>"


def _coerce(type_: type, value: Any, label: str) -> Any:
    if isinstance(value, bool) and type_ is not bool:
        raise TypeError(f"{label} takes {type_.__name__}, not bool")
    if type_ is float and isinstance(value, (int, float)):
        return float(value)
    if type_ is int and isinstance(value, int):
        return value
    if isinstance(value, type_):
        return value
    raise TypeError(f"{label} takes {type_.__name__}, not {type(value).__name__}")


def _match_choice(value: Any, choices: Sequence[Any], label: str) -> Any:
    for choice in choices:
        if isinstance(value, float) and isinstance(choice, (int, float)):
            if math.isclose(value, choice, rel_tol=1e-6):
                return type(value)(choice)
        elif value == choice:
            return choice
    raise ValueError(f"{label}: {value!r} is not one of {list(choices)}")


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return a is b
