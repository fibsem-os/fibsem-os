"""Devices, self-describing parameters, commands and shared resources.

A device declares which parameters it can have on its class (type and unit). A backend
binds the ones its hardware has to two small functions, read and write, and supplies
their metadata (limits, choices, settable) once, at bind time. Everything a backend
repeats today (checking, logging, caching, change events) happens here, once.

Two paths reach the hardware:

- the new API, ``param.set_value(v)``: type and choice checks raise, numeric values clip to
  their limits with a warning, and then the write path runs;
- the old API, through the key router, ``param.write_through(v)``: no new
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
    FrozenSet,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from psygnal import Signal

from fibsem.devices.display import DisplayHint
from fibsem.structures import RangeLimit

IMAGING_CHANNEL = "imaging_channel"
_SHARED = "shared"
_UNSET = object()
_C = TypeVar("_C", bound="_Controllable")


class ParameterUnavailable(AttributeError):
    """The device declares the parameter, but this backend did not bind it."""


class ParameterReadOnly(Exception):
    """The parameter is bound but cannot be set."""


class RoleUnfilled(AttributeError):
    """The device declares the role, but no device fills it."""


Limits = Union[RangeLimit, Mapping[str, RangeLimit]]


@dataclass(frozen=True)
class ParameterMetadata:
    """What a parameter allows. Filled from the vendor API once, then cached.

    ``limits`` is a ``RangeLimit`` for a number. For a composite value, such as a
    stage position, it is one ``RangeLimit`` per field, by field name.

    ``native_max`` and ``native_unit`` are for a parameter kept as a fraction of the
    hardware's maximum (power, gain): what 1.0 is in the hardware's own units, when
    the driver knows. A display can then show both.
    """

    limits: Optional[Limits] = None
    choices: Optional[Sequence[Any]] = None
    settable: bool = True
    native_max: Optional[float] = None
    native_unit: Optional[str] = None


class Parameter:
    """A parameter a device class may have. Declared on the class, bound per instance.

    ``limits`` and ``choices`` given here are static defaults; a backend's metadata
    replaces them. ``depends_on`` names parameters whose change alters this one's
    metadata (the ion current choices depend on the plasma gas).

    A backend implements a parameter in its device subclass with plain methods named
    after it, and the device binds them at ``connect()``::

        class AutoscriptBeam(Beam):
            def read_current(self):
                return self._beam.beam_current.value

            def write_current(self, value):
                self._beam.beam_current.value = value

            def metadata_current(self):  # optional: limits, choices, settable
                return ParameterMetadata(limits=...)

    ``available_<name>`` (optional) says whether this instance has it at all. Without
    ``read_<name>`` the parameter stays absent; without ``write_<name>`` it is
    read-only. A method named for a parameter the device doesn't declare is an error
    when the class is defined, so a typo can't silently hide a parameter. So is a
    subclass redeclaring a parameter with another type or unit: what differs between
    backends goes in metadata, and the parameter keeps one meaning everywhere.

    ``nearest`` declares that a number between the choices means the choice nearest
    it, as a filter wheel's bands do: 488 nm asks for the band at 485. Such a value is
    snapped, with a warning when it moved, rather than refused; and after every write
    the parameter is read back, so its cache and its ``changed`` signal carry what the
    hardware applied rather than what was asked for.

    ``display`` says how to show it (a ``fibsem.devices.display.Display``, or one per field for
    a composite value). Like the type and unit it means the same on every backend: a
    subclass that redeclares the parameter keeps it, and can't give another.
    """

    def __init__(
        self,
        type_: type,
        unit: Optional[str] = None,
        limits: Optional[Limits] = None,
        choices: Optional[Sequence[Any]] = None,
        depends_on: Sequence[str] = (),
        doc: str = "",
        nearest: bool = False,
        display: Optional[DisplayHint] = None,
    ):
        self.type = type_
        self.unit = unit
        self.limits = limits
        self.choices = choices
        self.depends_on = tuple(depends_on)
        self.doc = doc
        self.nearest = nearest
        self.display = display
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, device: Optional[_Controllable], owner: type) -> Any:
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


class Role:
    """A place on a device that another device fills, typed by what it must be.

    Declared on the class next to the parameters; a builder fills it per instance with
    `Device.fill_roles`, and the device reaches the filler through the attribute::

        class FM(Device):
            camera = Role(Camera)

        group.fill_roles(camera=camera)
        group.camera.acquire()

    The device talks to a role only through its interface, so it doesn't care which
    driver's device fills it. A required role left unfilled is an error at
    ``connect()``; reading any unfilled role raises `RoleUnfilled`.
    """

    def __init__(self, interface: type, required: bool = True, doc: str = ""):
        self.interface = interface
        self.required = required
        self.doc = doc
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, device: Optional[_Controllable], owner: type) -> Any:
        if device is None:
            return self
        try:
            return device._roles[self.name]
        except KeyError:
            raise RoleUnfilled(
                f"{type(device).__name__} '{device.name}' has no device in its "
                f"'{self.name}' role"
            ) from None

    def __repr__(self) -> str:
        optional = "" if self.required else ", required=False"
        return f"Role({self.interface.__name__}{optional})"


MetadataSource = Union[ParameterMetadata, Callable[[], ParameterMetadata], None]


class BoundParameter:
    """One parameter on one device, bound to a backend's read and write."""

    changed = Signal(object)
    """The new value, after every change: a set, a write through the old API, a live
    read that found a different value, or a change the backend reports."""

    metadata_changed = Signal(object)
    """The new ParameterMetadata, after a parameter this one depends on changed."""

    def __init__(
        self,
        device: _Controllable,
        spec: Parameter,
        read: Callable[[], Any],
        write: Optional[Callable[[Any], None]],
        metadata: MetadataSource,
        needs_channel: bool,
    ):
        self.device = device
        self.spec = spec
        self._read = read
        self._write = write
        self._metadata_source = metadata
        self.needs_channel = needs_channel
        self._cached: Any = _UNSET
        self.previous: Any = None
        """The value before the last change, for a ``changed`` handler to compare with."""
        self.metadata = ParameterMetadata()
        self.refresh_metadata(emit=False)

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
    def display(self) -> Optional[DisplayHint]:
        return self.spec.display

    @property
    def limits(self) -> Optional[Limits]:
        return self.metadata.limits

    @property
    def choices(self) -> Optional[Sequence[Any]]:
        return self.metadata.choices

    @property
    def settable(self) -> bool:
        return self.metadata.settable

    @property
    def writable(self) -> bool:
        """Whether the backend has a write at all: what the old API's path needs."""
        return self._write is not None

    def refresh_metadata(self, emit: bool = True) -> ParameterMetadata:
        """Read the metadata again from the backend. Called at bind and on dependencies."""
        source = self._metadata_source
        metadata = source() if callable(source) else (source or ParameterMetadata())
        limits = metadata.limits if metadata.limits is not None else self.spec.limits
        choices = (
            metadata.choices if metadata.choices is not None else self.spec.choices
        )
        settable = metadata.settable and self._write is not None
        self.metadata = replace(
            metadata, limits=limits, choices=choices, settable=settable
        )
        if emit:
            self.metadata_changed.emit(self.metadata)
            self.device.metadata_changed.emit(self.name, self.metadata)
        return self.metadata

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
        return self._write_path(value)

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
        """The old API: the same write path with no new checks, so no behaviour changes.

        A ``nearest`` parameter is still snapped to its nearest choice first. That is
        what the value means rather than a check, and it never refuses: a driver that
        snaps by itself gets the same choice, and a remote write sends the server a
        value its own checks accept.
        """
        if self._write is None:
            raise ParameterReadOnly(f"{self.device.name}.{self.name} is read-only")
        if self._snaps(value):
            value = self._snap(value)
        self._write_path(value)

    def report(self, value: Any) -> None:
        """A change the backend observed itself (vendor UI, another client, a push)."""
        self._remember(value)

    def validate(self, value: Any) -> Any:
        value = _coerce(self.type, value, f"{self.device.name}.{self.name}")
        if self._snaps(value):
            value = self._snap(value)
        elif self.choices is not None:
            value = _match_choice(
                value, self.choices, f"{self.device.name}.{self.name}"
            )
        if isinstance(self.limits, RangeLimit) and isinstance(value, (int, float)):
            clipped = self.limits.clamp(value)
            if clipped != value:
                logging.warning(
                    f"{self.device.name}.{self.name}: {value} is outside {self.limits}, "
                    f"set to {clipped}"
                )
                value = self.type(clipped)
        return value

    def _snaps(self, value: Any) -> bool:
        """Whether `value` is snapped to a choice: a number, for a ``nearest`` parameter."""
        return (
            self.spec.nearest
            and bool(_numeric_choices(self.choices))
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    def _snap(self, value: Any) -> Any:
        """The choice nearest `value`, warning when that is not `value` itself."""
        choices = _numeric_choices(self.choices)
        choice = min(choices, key=lambda c: abs(c - value))
        if not math.isclose(choice, value, rel_tol=1e-6):
            logging.warning(
                f"{self.device.name}.{self.name}: {value} is not one of {choices}, "
                f"set to the nearest, {choice}"
            )
        return self.type(choice)

    def _write_path(self, value: Any) -> Any:
        """Write, cache and signal; returns the value now held."""
        with self.device._claim(self):
            self._write(value)
            if self.spec.nearest:
                # what the hardware applied, which a driver may have adjusted again
                value = self._read()
        logging.debug(
            {
                "msg": "set",
                "device": self.device.name,
                "param": self.name,
                "value": value,
            }
        )
        previous, self._cached = self._cached, value
        self.previous = None if previous is _UNSET else previous
        self._emit(value)
        self.device._dependency_changed(self.name)
        return value

    def _remember(self, value: Any) -> None:
        previous, self._cached = self._cached, value
        if previous is _UNSET or not _same(previous, value):
            if previous is not _UNSET:
                self.previous = previous
                self._emit(value)
                # changed outside this device (the vendor UI): what depends on it
                # is read again, as after a write
                self.device._dependency_changed(self.name)

    def _emit(self, value: Any) -> None:
        self.changed.emit(value)
        self.device.changed.emit(self.name, value)

    def __repr__(self) -> str:
        cached = "?" if self._cached is _UNSET else repr(self._cached)
        return f"<{self.device.name}.{self.name} = {cached} {self.metadata}>"


@dataclass(frozen=True)
class CommandInfo:
    name: str
    signature: str
    available: bool
    doc: str


def command(
    fn: Optional[Callable] = None, *, available: Optional[Callable[[Any], bool]] = None
) -> Any:
    """Mark a device method as a command. It stays a plain method.

    The mark records only what a UI, the agent API or a remote proxy needs to list it:
    its name, its arguments and whether it is available on this device right now.
    """

    def mark(method: Callable) -> Callable:
        method.__device_command__ = available  # type: ignore[attr-defined]
        return method

    return mark(fn) if fn is not None else mark


class Resources:
    """Named shared resources, each a re-entrant lock.

    Names not listed in ``groups`` share one lock, which is today's behaviour and
    always safe. A backend separates only what its hardware really allows in parallel.
    A device on another computer gets its own ``Resources`` and shares nothing.

    ``locks`` gives a group a lock that already exists, so devices and older code that
    takes that lock directly exclude each other. The microscope's own registry
    (`FibsemMicroscope.resources`) gives ``imaging_channel`` its ``_threading_lock``.
    """

    def __init__(
        self,
        groups: Optional[Mapping[str, str]] = None,
        locks: Optional[Mapping[str, threading.RLock]] = None,
    ):
        self._groups = dict(groups or {})
        self._locks: Dict[str, threading.RLock] = dict(locks or {})
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


def resources_of(owner: Any) -> Resources:
    """The registry ``owner`` (usually the microscope) keeps, or a new one if it has none."""
    resources = getattr(owner, "resources", None)
    return resources if isinstance(resources, Resources) else Resources()


class _Controllable:
    """What a device and a service share: a name, a parent, parameters, commands,
    roles, change events and the parent's resources.

    Private: code names `Device` (a piece of hardware) or
    `fibsem.services.Service` (a capability that uses devices over time, such as
    milling), never this.
    """

    changed = Signal(str, object)
    """(parameter name, value) for any parameter on the device."""

    metadata_changed = Signal(str, object)
    """(parameter name, ParameterMetadata) when a parameter's metadata was refreshed."""

    def __init__(
        self,
        name: str,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        self.name = name
        self.parent = parent
        # Without one given, a device claims its parent microscope's resources.
        self.resources = resources if resources is not None else resources_of(parent)
        self._bound: Dict[str, BoundParameter] = {}
        self._roles: Dict[str, Device] = {}
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

    @classmethod
    def declared_roles(cls) -> Dict[str, Role]:
        found: Dict[str, Role] = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if isinstance(attr, Role):
                    found[name] = attr
        return found

    def fill_roles(self: _C, **devices: Device) -> _C:
        """Put a device in each named role. A builder calls this before ``connect()``.

        A name the class doesn't declare, or a device that isn't the role's interface,
        is an error. Returns the device.
        """
        declared = self.declared_roles()
        for name, device in devices.items():
            role = declared.get(name)
            if role is None:
                raise TypeError(f"{type(self).__name__} has no '{name}' role")
            if not isinstance(device, role.interface):
                raise TypeError(
                    f"{type(self).__name__}.{name} takes a {role.interface.__name__}, "
                    f"not {type(device).__name__} '{device.name}'"
                )
        self._roles.update(devices)
        return self

    def check_roles(self) -> None:
        """Raise `RoleUnfilled` if a required role has no device. ``connect()`` calls it."""
        missing = [
            name
            for name, role in self.declared_roles().items()
            if role.required and name not in self._roles
        ]
        if missing:
            raise RoleUnfilled(
                f"{type(self).__name__} '{self.name}' has no device in its required "
                f"roles {missing}"
            )

    def bind(
        self,
        name: str,
        read: Callable[[], Any],
        write: Optional[Callable[[Any], None]] = None,
        metadata: MetadataSource = None,
        needs_channel: bool = False,
    ) -> BoundParameter:
        """Give a declared parameter to this device. Without ``write`` it is read-only."""
        spec = self.declared_parameters().get(name)
        if spec is None:
            raise AttributeError(
                f"{type(self).__name__} declares no parameter '{name}'"
            )
        param = BoundParameter(self, spec, read, write, metadata, needs_channel)
        self._bound[name] = param
        return param

    IMPLEMENTATION_PREFIXES = ("read_", "write_", "metadata_", "available_")

    needs_channel: FrozenSet[str] = frozenset()
    """Parameters whose read and write claim the imaging channel on this backend."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A parameter means the same thing on every backend: a subclass may redeclare
        # one (new static limits, choices, doc) but not change its type or unit.
        inherited = {
            name: param
            for base in cls.__bases__
            if issubclass(base, _Controllable)
            for name, param in base.declared_parameters().items()
        }
        for name, attr in vars(cls).items():
            base_param = inherited.get(name)
            if not isinstance(attr, Parameter) or base_param is None:
                continue
            if (attr.type, attr.unit) != (base_param.type, base_param.unit):
                raise TypeError(
                    f"{cls.__name__}.{name} is {attr.type.__name__} in {attr.unit!r}, "
                    f"but is declared {base_param.type.__name__} in "
                    f"{base_param.unit!r}; a backend can't change a parameter's type "
                    f"or unit"
                )
            if attr.display is None:
                attr.display = base_param.display
            elif attr.display != base_param.display:
                raise TypeError(
                    f"{cls.__name__}.{name} declares another display than "
                    f"{base_param.display!r}; a backend can't change how a parameter "
                    f"is shown"
                )
        declared = cls.declared_parameters()
        for attr, value in vars(cls).items():
            if not callable(value):
                continue
            for prefix in cls.IMPLEMENTATION_PREFIXES:
                if attr.startswith(prefix) and attr[len(prefix) :] not in declared:
                    raise TypeError(
                        f"{cls.__name__}.{attr} implements '{attr[len(prefix) :]}', "
                        f"which {cls.__name__} does not declare"
                    )
        unknown = set(cls.needs_channel) - set(declared)
        if unknown:
            raise TypeError(f"{cls.__name__}.needs_channel names {sorted(unknown)}")

    def connect(self: _C) -> _C:
        """Bind every parameter this class implements, reading its metadata once.

        A backend calls this after constructing its device. A parameter with no
        ``read_<name>``, or whose ``available_<name>()`` is False, stays absent. A
        required role with no device is an error. Returns the device.
        """
        self.check_roles()
        for name in self.declared_parameters():
            read = getattr(self, f"read_{name}", None)
            if read is None:
                continue
            available = getattr(self, f"available_{name}", None)
            if available is not None and not available():
                continue
            self.bind(
                name,
                read=read,
                write=getattr(self, f"write_{name}", None),
                metadata=getattr(self, f"metadata_{name}", None),
                needs_channel=name in self.needs_channel,
            )
        return self

    def bind_channel(self, select: Callable[[], Optional[Callable[[], None]]]) -> None:
        """How to make this device the active imaging channel, for needs_channel params.

        ``select`` may return a callable that puts the previous channel back. It runs
        when the read, write or command is done, still under ``imaging_channel``.
        """
        self._select_channel = select

    # -- description ----------------------------------------------------------------

    @property
    def parameters(self) -> Dict[str, BoundParameter]:
        """The parameters this backend bound. A declared one missing here is absent."""
        return dict(self._bound)

    @property
    def roles(self) -> Dict[str, Device]:
        """The devices in this device's roles, by role name. A declared one missing
        here is unfilled."""
        return dict(self._roles)

    @property
    def commands(self) -> Dict[str, CommandInfo]:
        found = {}
        for name, attr in inspect.getmembers(type(self), inspect.isfunction):
            if not hasattr(attr, "__device_command__"):
                continue
            available = attr.__device_command__
            signature = inspect.signature(attr)
            arguments = list(signature.parameters.values())[1:]  # without self
            found[name] = CommandInfo(
                name=name,
                signature=str(signature.replace(parameters=arguments)),
                available=True if available is None else bool(available(self)),
                doc=inspect.getdoc(attr) or "",
            )
        return found

    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Plain data for every bound parameter: enough for a UI or a remote proxy."""
        return {name: _describe(p) for name, p in self._bound.items()}

    # -- internals -----------------------------------------------------------------

    @contextmanager
    def claim_channel(self) -> Iterator[None]:
        """Hold the imaging channel with this device selected on it.

        What a ``needs_channel`` parameter does around its read or write, for a command
        whose vendor call needs the channel: claim, select, act, restore, release.
        """
        with self.resources.claim(IMAGING_CHANNEL):
            restore = self.select_channel()
            try:
                yield
            finally:
                if restore is not None:
                    restore()

    @contextmanager
    def _claim(self, param: BoundParameter) -> Iterator[None]:
        if not param.needs_channel:
            yield
            return
        with self.claim_channel():
            yield

    def select_channel(self) -> Optional[Callable[[], None]]:
        """Make this device the active imaging channel. Backends that share one override it.

        Returns how to put the previous channel back, or None when there is nothing to
        restore.
        """
        if self._select_channel is not None:
            return self._select_channel()
        return None

    def _dependency_changed(self, name: str) -> None:
        for param in self._bound.values():
            if name in param.spec.depends_on:
                param.refresh_metadata()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} '{self.name}': {sorted(self._bound)}>"


class Device(_Controllable):
    """One piece of hardware: named, with parameters and commands, and a parent."""

    def facts(self) -> Dict[str, Any]:
        """What the device says about itself that is neither a parameter nor a
        command, as JSON-ready data: what a remote copy of it needs to answer the
        same questions (the stage's frame, the needle's named positions). The device
        server sends it with the device's description. Nothing by default.
        """
        return {}


def _describe(p: BoundParameter) -> Dict[str, Any]:
    described = {
        "type": p.type.__name__,
        "unit": p.unit,
        "limits": _limits_to_dict(p.limits),
        "choices": list(p.choices) if p.choices is not None else None,
        "settable": p.settable,
    }
    # Only a fraction of a hardware maximum has these, so only it carries them.
    if p.metadata.native_max is not None:
        described["native_max"] = p.metadata.native_max
        described["native_unit"] = p.metadata.native_unit
    return described


def _limits_to_dict(limits: Optional[Limits]) -> Any:
    if isinstance(limits, RangeLimit):
        return limits.to_dict()
    if isinstance(limits, Mapping):
        return {name: limit.to_dict() for name, limit in limits.items()}
    return None


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


def _numeric_choices(choices: Optional[Sequence[Any]]) -> list:
    return [
        c
        for c in (choices or ())
        if isinstance(c, (int, float)) and not isinstance(c, bool)
    ]


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return a is b
