"""Field declarations: parameters, units, limits and display hints."""

from __future__ import annotations

import logging
import math
import sys
import weakref
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from fibsem.codec import LowerBound, UpperBound, decode, encode, saved


@saved(required=("min", "max"))
@dataclass
class RangeLimit:
    """A value's range; an unbounded end is infinite, and null as data."""

    min: LowerBound
    max: UpperBound

    def clamp(self, value: float) -> float:
        return max(self.min, min(self.max, value))

    def to_dict(self) -> dict:
        return encode(self)

    @staticmethod
    def from_dict(d: dict) -> "RangeLimit":
        return decode(RangeLimit, d)


Limits = Union[RangeLimit, Mapping[str, RangeLimit]]


@dataclass(frozen=True)
class Display:
    """How to show a value. Every field is optional; a form falls back to its own
    default for the ones left out. What a value may be (its unit, limits and
    choices) is not here: the ``Parameter`` says that, in SI.

    ``scale`` turns the SI value into the shown one (1e6: metres to µm). ``unit`` is the
    shown unit; left out, it is the parameter's unit with the scale's SI prefix (1e6 and
    "m" give "µm"). Give it when the scale isn't an SI prefix: 180/π for degrees, 100
    for percent. ``step`` is in shown units. ``format_fn`` writes each choice in a
    combo box; it stays in the process (``to_dict`` leaves it out).
    """

    label: Optional[str] = None
    scale: Optional[float] = None
    unit: Optional[str] = None
    step: Optional[float] = None
    decimals: Optional[int] = None
    advanced: bool = False
    # never shown in a form: set by code, not by hand (a pattern's drawn shapes)
    hidden: bool = False
    format_fn: Optional[Callable[[Any], str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """What it sets, as JSON-ready data: what a description carries, so a client
        with no copy of the class (an agent) shows the value the same way."""
        shown = {
            key: value
            for key, value in (
                ("label", self.label),
                ("scale", self.scale),
                ("unit", self.unit),
                ("step", self.step),
                ("decimals", self.decimals),
            )
            if value is not None
        }
        if self.advanced:
            shown["advanced"] = True
        if self.hidden:
            shown["hidden"] = True
        return shown


# A composite value (a Point, a stage position) takes one Display per field.
DisplayHint = Union[Display, Mapping[str, Display]]


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


@dataclass(frozen=True)
class Parameter:
    """A value's meaning and look: its type, SI ``unit``, ``limits`` in SI, ``choices``,
    a ``doc``, and a ``Display`` for how a form shows it.

    One shape for both places a value is declared: a device's parameter
    (``fibsem.devices.core.DeviceParameter`` is one, bound to the hardware) and a
    dataclass field (declared with ``param(...)``, read by ``.params``). A form,
    a description or a JSON schema reads either the same way.

    For a field, ``type`` is its annotation (Optional and List left on).
    ``device_parameter`` binds it to a device parameter, as ``"beam.current"``: the
    device's live limits and choices apply, and the device's display fills what the
    field's leaves out. ``filepath`` marks a string that names a file.
    """

    type: Any = None
    unit: Optional[str] = None
    limits: Optional[Limits] = None
    choices: Optional[Sequence[Any]] = None
    doc: str = ""
    display: Optional[DisplayHint] = None
    name: str = ""
    device_parameter: Optional[str] = None
    filepath: bool = False
    # Why setting it does nothing, when it doesn't: nothing reads it yet. A patch to
    # it is refused with this, and a description says it beside the field.
    not_used: Optional[str] = None

    def _checked(self) -> "Parameter":
        """This Parameter, checked as a field's: the field gives the name and the
        type (its annotation), so neither is set here. Checked when the module is
        imported, so a wrong declaration fails there rather than in a form that
        renders without it."""
        if self.name or self.type is not None:
            raise ValueError(
                "a field's Parameter takes its name and type from the field; "
                f"got name={self.name!r}, type={self.type!r}"
            )
        if self.limits is not None and not isinstance(self.limits, RangeLimit):
            raise TypeError(f"limits must be a RangeLimit, got {self.limits!r}")
        if self.display is not None and not isinstance(self.display, Display):
            raise TypeError(f"display must be a Display, got {self.display!r}")
        if self.device_parameter is not None and "." not in self.device_parameter:
            raise ValueError(
                f"device_parameter names the device and the parameter, as "
                f'"beam.current"; got {self.device_parameter!r}'
            )
        if self.choices is not None:
            return replace(self, choices=tuple(self.choices))
        return self

    # -- how it is shown: one Display (a composite's are per field) ----------------

    @property
    def _shown(self) -> Display:
        return self.display if isinstance(self.display, Display) else Display()

    @property
    def label(self) -> Optional[str]:
        return self._shown.label

    @property
    def scale(self) -> Optional[float]:
        """SI to shown: 1e6 shows metres as µm. None shows the value as it is."""
        return self._shown.scale

    @property
    def advanced(self) -> bool:
        return self._shown.advanced

    @property
    def hidden(self) -> bool:
        return self._shown.hidden

    @property
    def format_fn(self) -> Optional[Callable[[Any], str]]:
        return self._shown.format_fn

    @property
    def suffix(self) -> str:
        """The shown unit: the display's, else the SI unit with the scale's prefix
        (1e6 and "m" give "µm"). Empty for a value with no unit."""
        if self._shown.unit:
            return self._shown.unit
        if not self.unit:
            return ""
        if not self.scale:
            return self.unit
        from fibsem.utils import _get_display_unit

        return _get_display_unit(self.scale, self.unit)

    @property
    def shown_range(self) -> Optional[Tuple[float, float]]:
        """The limits in shown units, lowest first; None for no finite limits.
        Unscaled, they stay as declared, so an int field keeps int bounds."""
        limits = self.limits
        if not isinstance(limits, RangeLimit) or not (
            math.isfinite(limits.min) and math.isfinite(limits.max)
        ):
            return None
        if not self.scale:
            return (limits.min, limits.max)
        low, high = sorted((limits.min * self.scale, limits.max * self.scale))
        return (low, high)

    @property
    def bound_name(self) -> Optional[str]:
        """The name of the device parameter it is bound to: ``"current"`` for
        ``"beam.current"``. None when it is bound to none."""
        if not self.device_parameter:
            return None
        return split_device_parameter(self.device_parameter)[1]

    @property
    def instrument_choices(self) -> bool:
        """Bound to a device parameter that is picked from what the instrument offers
        (scan direction, resolution) rather than set within a range: its choices come
        from the instrument when a form has one."""
        if self.device_parameter is None or self.choices is not None:
            return False
        bound = declared_device_parameter(self.device_parameter)
        return bound is None or bound.limits is None

    # -- variants ----------------------------------------------------------------

    def part(self, name: Optional[str]) -> "Parameter":
        """One field of a composite value (``"x"`` of a shift): its own Display and
        limits. A plain value, or ``name`` None, comes back as it is."""
        if name is None:
            return self
        display = self.display
        limits = self.limits
        return replace(
            self,
            display=display.get(name) if isinstance(display, Mapping) else display,
            limits=limits.get(name) if isinstance(limits, Mapping) else limits,
        )

    def with_reported(self, reported: Any, part: Optional[str] = None) -> "Parameter":
        """This value with what an instrument reports (a ``ParameterMetadata``) in
        place of its own limits and choices. Unbounded limits (a driver that doesn't
        know its range) keep the declared ones. ``part`` picks one field's limits
        when they are given per field (``"x"`` of a shift)."""
        limits = reported.limits
        if isinstance(limits, Mapping):
            limits = limits.get(part)
        changes: Dict[str, Any] = {}
        if (
            isinstance(limits, RangeLimit)
            and math.isfinite(limits.min)
            and math.isfinite(limits.max)
        ):
            changes["limits"] = limits
        if reported.choices is not None:
            changes["choices"] = tuple(reported.choices)
        return replace(self, **changes) if changes else self


def display_to_dict(hint: Optional[DisplayHint]) -> Optional[Dict[str, Any]]:
    """A Display, or one per field of a composite value, as JSON-ready data; None
    for none."""
    if hint is None:
        return None
    if isinstance(hint, Mapping):
        return {name: shown.to_dict() for name, shown in hint.items()}
    return hint.to_dict()


# The key a dataclass field's metadata keeps its Parameter under (param()).
PARAMETER_KEY = "parameter"


# The range a pattern or strategy length takes unless it says otherwise.
DISTANCE_LIMITS = RangeLimit(0.01e-6, 1000e-6)


def distance_param(
    label: str,
    doc: str,
    *,
    limits: RangeLimit = DISTANCE_LIMITS,
    step: float = 0.1,
    decimals: int = 2,
    advanced: bool = False,
) -> Parameter:
    """A length in metres, shown in µm."""
    return Parameter(
        unit="m",
        limits=limits,
        doc=doc,
        display=Display(
            label, scale=1e6, step=step, decimals=decimals, advanced=advanced
        ),
    )


def param(
    default: Any = MISSING,
    parameter: Optional[Parameter] = None,
    *,
    default_factory: Any = MISSING,
    init: bool = True,
    repr: bool = True,
    compare: bool = True,
    **declared: Any,
) -> Any:
    """A dataclass field that declares its ``Parameter``::

        width: float = param(10e-6, distance_param("Width", "Width of the pattern."))
        rotation: float = param(
            0.0,
            unit="°",
            limits=RangeLimit(0.0, 360.0),
            doc="Rotation of the pattern.",
            display=Display("Rotation", step=1.0, decimals=2),
        )

    The Parameter is ``parameter`` (a shared one, from a helper) with the keywords
    on top, or just the keywords. The field gives the name and the type. The value
    stays plain (``pattern.width`` is the float); the Parameter is read with
    ``.params`` (``pattern.params.width.limits``) or ``field_specs``.
    """
    base = parameter if parameter is not None else Parameter()
    checked = (replace(base, **declared) if declared else base)._checked()
    return field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        compare=compare,
        metadata={PARAMETER_KEY: checked},
    )


def field_specs(struct_cls: Type[Any]) -> Dict[str, Parameter]:
    """Each field of a dataclass as a ``Parameter``, in declaration order: the one it
    declares (``param(...)``), with its name and annotation filled in, or
    an empty one for a field that declares none.

    A field bound to a device parameter (``device_parameter``) takes the device's
    unit and display for what it leaves out, keeping its own label and placement,
    so a milling current reads like the beam panel's current.
    """
    types = _field_types(struct_cls)
    specs: Dict[str, Parameter] = {}
    for f in fields(struct_cls):
        _warn_unknown_metadata_keys(struct_cls, f.name, f.metadata)
        declared = f.metadata.get(PARAMETER_KEY) or Parameter()
        unit = declared.unit
        display = declared.display or Display()
        bound = declared_device_parameter(declared.device_parameter)
        if bound is not None:
            if unit is None:
                unit = bound.unit
            display = _fill_display(display, bound.display)
        specs[f.name] = replace(
            declared,
            name=f.name,
            type=types.get(f.name),
            unit=unit,
            doc=declared.doc or "",
            display=display,
        )
    return specs


def flagged_fields(struct_cls: Type[Any], flag: str) -> Tuple[str, ...]:
    """The names of the fields whose ``Display`` sets ``flag`` ("advanced" or
    "hidden"), in declaration order."""
    return tuple(
        name for name, spec in field_specs(struct_cls).items() if getattr(spec, flag)
    )


class Params(Mapping[str, Parameter]):
    """A dataclass's Parameters: by attribute (``params.width``), by name
    (``params["width"]``), and in declaration order (``params.items()``)."""

    def __init__(self, specs: Mapping[str, Parameter]):
        self._specs = dict(specs)

    def __getitem__(self, name: str) -> Parameter:
        return self._specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __getattr__(self, name: str) -> Parameter:
        try:
            return self.__dict__["_specs"][name]
        except KeyError:
            raise AttributeError(f"no field {name!r}") from None

    def __repr__(self) -> str:
        return f"Params({', '.join(self._specs)})"


_params_cache: "weakref.WeakKeyDictionary[type, Params]" = weakref.WeakKeyDictionary()


class _ParamsAccessor:
    """``cls.params`` and ``instance.params``: the class's ``Params``, built once."""

    def __get__(self, obj: Any, cls: type) -> Params:
        found = _params_cache.get(cls)
        if found is None:
            found = _params_cache[cls] = Params(field_specs(cls))
        return found


class Configurable:
    """A dataclass that describes itself: ``params`` reads each field's Parameter
    (``RectanglePattern.params.width.limits``) and ``describe()`` gives the whole
    class as a JSON Schema. The values stay plain dataclass fields."""

    params = _ParamsAccessor()

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        """This class as a JSON Schema object; the dataclasses its fields hold are
        under ``$defs``."""
        from fibsem.schema import value_schema

        defs: Dict[str, Any] = {}
        schema = value_schema(cls, defs)
        if defs:
            schema["$defs"] = defs
        return schema


def _fill_display(own: Display, device: Optional[DisplayHint]) -> Display:
    """``own`` with the device's scale, unit, step and decimals where it gives none.
    The label and the placement (advanced, hidden) stay the field's: it names the
    value in its own context."""
    if not isinstance(device, Display):
        return own
    return replace(
        own,
        scale=own.scale if own.scale is not None else device.scale,
        unit=own.unit if own.unit is not None else device.unit,
        step=own.step if own.step is not None else device.step,
        decimals=own.decimals if own.decimals is not None else device.decimals,
    )


_warned_metadata_keys: set = set()


def _warn_unknown_metadata_keys(
    struct_cls: Type[Any], field_name: str, metadata: Mapping[str, Any]
) -> None:
    """Log once for a metadata key nothing will ever read.

    A mis-keyed field is otherwise silent: the form renders, the value is right,
    and the label or suffix is simply missing. Warned once per (class, field, key)
    because form metadata is re-read on every rebuild.
    """
    for key in metadata:
        if key == PARAMETER_KEY:
            continue
        marker = (struct_cls.__qualname__, field_name, key)
        if marker in _warned_metadata_keys:
            continue
        _warned_metadata_keys.add(marker)
        logging.warning(
            f"{struct_cls.__name__}.{field_name} declares metadata key {key!r}, which no "
            f"form reads. A field declares its unit, limits, doc and display as a "
            f"Parameter: name: float = param(default, unit=..., doc=...)."
        )


def _field_types(struct_cls: Type[Any]) -> Dict[str, Any]:
    """Each field's annotation, resolved: a string annotation is evaluated in the
    module of the class that declared it, since an inherited field's names are the
    base class's. One that doesn't resolve (a name imported only for type checking)
    is left out."""
    found: Dict[str, Any] = {}
    for f in fields(struct_cls):
        hint = f.type
        if isinstance(hint, str):
            owner = next(
                (
                    c
                    for c in struct_cls.__mro__
                    if f.name in vars(c).get("__annotations__", {})
                ),
                struct_cls,
            )
            module = sys.modules.get(owner.__module__)
            try:
                hint = eval(hint, dict(vars(module)) if module else {})  # noqa: S307
            except Exception:
                continue
        found[f.name] = hint
    return found


# The device (or service) a ``device_parameter`` names, by its first part: module,
# class. Looked up when first read, since fibsem.devices imports this module.
_DEVICE_CLASSES: Dict[str, Tuple[str, str]] = {
    "milling": ("fibsem.services.milling", "Milling"),
    "beam": ("fibsem.devices.beam", "Beam"),
    "scanner": ("fibsem.devices.scanner", "Scanner"),
    "stage": ("fibsem.devices.stage", "Stage"),
    "manipulator": ("fibsem.devices.manipulator", "Manipulator"),
    "chamber": ("fibsem.devices.chamber", "Chamber"),
    "sample_loader": ("fibsem.devices.sample_loader", "SampleLoader"),
    "camera": ("fibsem.devices.fm", "Camera"),
    "light_source": ("fibsem.devices.fm", "LightSource"),
    "filter_set": ("fibsem.devices.fm", "FilterSet"),
    "objective": ("fibsem.devices.fm", "Objective"),
}


def split_device_parameter(ref: str) -> Tuple[str, str]:
    """``"beam.current"`` as ``("beam", "current")``."""
    device, _, name = ref.partition(".")
    return device, name


def declared_device_parameter(ref: Optional[str]) -> Optional[Parameter]:
    """The device parameter ``ref`` names (``"beam.current"``), as its device class
    declares it; None for no ref, or one naming nothing a device declares."""
    if not ref:
        return None
    device, name = split_device_parameter(ref)
    where = _DEVICE_CLASSES.get(device)
    if where is None:
        return None
    import importlib

    cls = getattr(importlib.import_module(where[0]), where[1])
    parameter = getattr(cls, name, None)
    return parameter if isinstance(parameter, Parameter) else None
