"""One control builder for every metadata-driven config form.

The pattern, strategy and milling-settings widgets each carried their own copy of
the same loop: map field metadata to a control, then a matching `isinstance`
ladder in `get_*` to read the value back and a third in `set_*` to push one in.
Three ladders per form, four forms, and they had already drifted -- the strategy
form honoured `format_fn` on comboboxes and the pattern form did not; the pattern
form could resolve a field's choices against the microscope and the strategy form
could not.

`build_control` returns the control *together with* the adapters that move values
through it, so a form never inspects a control's type. Adding a control type is
one branch here rather than four branches in three places each.

Each field arrives as a ``Parameter`` (``fibsem.structures``): SI unit, SI limits,
choices, doc, and a ``Display`` (label, scale, step, decimals). A device
parameter becomes one through ``parameter_spec``, so a panel of beam controls and a
recipe form read the same thing.

Deliberately Qt-only: no microscope, no pattern or strategy imports. A field whose
choices are the instrument's needs the microscope, so the caller passes a resolver
rather than the builder reaching for one -- which is also what makes this testable
without hardware.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from enum import Enum
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
)

from PyQt5.QtWidgets import QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget

from fibsem.devices.core import BoundParameter
from fibsem.structures import (
    Display,
    Parameter,
    ParameterMetadata,
    Point,
    declared_device_parameter,
    split_device_parameter,
)
from fibsem.ui.widgets.custom_widgets import (
    IntegerValueSpinBox,
    QFilePathLineEdit,
    QLineEdit,
    ValueComboBox,
    ValueSpinBox,
    unit_suffix,
)


@dataclass(frozen=True)
class FormDefaults:
    """What a form falls back to for keys a field does not declare.

    A form's existing look is not something a shared builder should quietly
    change, and the spinbox classes' own defaults are not neutral -- a range of
    (0.0, 1e6) puts a floor of zero on anything that never declared a minimum,
    and clamping is silent. Each form passes what it already had.
    """

    float_range: Optional[Tuple[float, float]] = None
    int_range: Optional[Tuple[int, int]] = None
    step: Optional[float] = None
    decimals: Optional[int] = None


# A point is an offset from the image centre, so it is signed by definition and
# must never inherit a floor of zero. The pattern form carried this same pair as
# a literal default before the row was generalised.
POINT_FALLBACK_RANGE = (-1000.0, 1000.0)


@dataclass
class Control:
    """A built form control plus the adapters that move values through it.

    `widget` is what goes into the form layout and what gets shown or hidden --
    for a compound control it is the container, so visibility stays one call.
    `inputs` is the actual input widgets, which is what `blockSignals` has to
    reach; they differ only for compound controls.
    """

    widget: QWidget
    read: Callable[[], Any]
    write: Callable[[Any], None]
    signals: Tuple[Any, ...] = ()
    inputs: Tuple[QWidget, ...] = dataclass_field(default_factory=tuple)
    editable: bool = True

    def __post_init__(self) -> None:
        if not self.inputs:
            self.inputs = (self.widget,)

    def connect(self, slot: Callable[[], None]) -> None:
        """Wire every signal that means "the user edited this".

        The slot is always called with **no arguments**. The signals behind a
        control carry different payloads -- `valueChanged` a number, `toggled` a
        bool, `currentIndexChanged` an index, `editingFinished` nothing -- and a
        form wants none of them; it calls `read()` instead.

        Normalising here rather than at each call site is deliberate. PyQt sizes
        the call to whatever the slot will accept, so a slot that takes one
        argument silently receives the payload. That is not hypothetical: the
        task form captured its field name with `lambda name=field:`, PyQt passed
        `valueChanged`'s float as `name`, and every spinbox, checkbox and combo
        edit was dropped without a word. A slot that genuinely cannot
        take zero arguments now raises instead.
        """
        for signal in self.signals:
            signal.connect(lambda *_: slot())

    def set_blocked(self, blocked: bool) -> None:
        """Block or unblock the inputs, so a programmatic write is not echoed back."""
        for widget in self.inputs:
            widget.blockSignals(blocked)


@dataclass
class Row:
    """One built form row: the label, the control, and whether it is advanced."""

    label: QLabel
    control: Control
    field: str
    advanced: bool


class AdvancedRows:
    """The show-advanced toggle every parameter form shares: the form keeps its
    built rows in ``_rows`` and hides the advanced ones until asked."""

    _rows: List[Row]
    _advanced_visible: bool = False

    def _row_shown(self, row: Row) -> bool:
        """Whether *row* is shown at all, advanced or not."""
        return True

    def _update_visibility(self) -> None:
        for row in self._rows:
            visible = self._row_shown(row) and (
                not row.advanced or self._advanced_visible
            )
            row.label.setVisible(visible)
            row.control.widget.setVisible(visible)

    def set_advanced_visible(self, show: bool) -> None:
        self._advanced_visible = show
        self._update_visibility()


def _default_label(name: str) -> str:
    """``dwell_time`` -> ``Dwell time``."""
    return name.replace("_", " ").capitalize()


def parameter_spec(
    parameter: Union[Parameter, BoundParameter], part: Optional[str] = None
) -> Parameter:
    """A device parameter as a form field: its unit, doc and ``Display``, with what
    the instrument reports (or, for a declared parameter, what every instrument
    has) as its limits and choices.

    A bound parameter carries what its instrument reports; a declared one (``Beam.hfw``,
    for a form with no instrument) its static limits and choices. ``part`` picks one
    field of a composite value (``"x"`` of a shift), for its hint and its limits.
    """
    spec = getattr(parameter, "spec", parameter)
    hint = spec.display
    if isinstance(hint, Mapping):
        hint = hint.get(part)
    hint = hint or Display()
    field = Parameter(
        name=spec.name,
        unit=spec.unit,
        doc=spec.doc,
        display=replace(hint, label=hint.label or _default_label(spec.name)),
    )
    return field.with_reported(reported_metadata(parameter), part)


def reported_metadata(parameter: Union[Parameter, BoundParameter]) -> ParameterMetadata:
    """What a parameter allows: what its instrument reports, and what every
    instrument has (its declared limits and choices) where it reports none."""
    spec = getattr(parameter, "spec", parameter)
    reported = getattr(parameter, "metadata", None) or ParameterMetadata()
    return replace(
        reported,
        limits=spec.limits if reported.limits is None else reported.limits,
        choices=spec.choices if reported.choices is None else reported.choices,
    )


def bound_spec(
    spec: Parameter, devices: Optional[Mapping[str, Any]] = None
) -> Parameter:
    """A field bound to a device parameter (``device_parameter="camera.exposure_time"``)
    with what that parameter allows in place of its own limits and choices.

    The live parameter comes from ``devices`` by the binding's device name
    (``devices["camera"]``) and carries what its instrument reports. Without that
    device, or when it can't answer, it is what the device class declares, so a form
    with no instrument still gets a range. A field bound to nothing comes back as it is.
    """
    declared = declared_device_parameter(spec.device_parameter)
    if declared is None:
        return spec
    device_name, name = split_device_parameter(spec.device_parameter)
    device = (devices or {}).get(device_name)
    parameter = None
    if device is not None:
        try:
            parameter = device.parameters.get(name)
        except Exception as e:  # a form must open even when the device can't answer
            logging.warning(
                f"Could not read {spec.device_parameter}, using its declared limits: {e}"
            )
    return spec.with_reported(reported_metadata(parameter or declared))


def configure_spinbox(spinbox: QDoubleSpinBox, spec: Parameter) -> None:
    """Set a spin box's suffix, step, decimals, range and tooltip from a field, for
    a panel that keeps its own boxes. What the field leaves out keeps the box's own."""
    spinbox.setSuffix(unit_suffix(spec.suffix))
    if spec.display.decimals is not None:
        spinbox.setDecimals(spec.display.decimals)
    if spec.display.step is not None:
        spinbox.setSingleStep(spec.display.step)
    if spec.shown_range is not None:
        spinbox.setRange(*spec.shown_range)
    if spec.doc:
        spinbox.setToolTip(spec.doc)


def _enum_label(member: Any) -> str:
    """``CleaningCrossSection`` -> ``Cleaning Cross Section``, ``EACH_ROW`` -> ``Each Row``."""
    if not isinstance(member, Enum):
        return str(member)
    name = member.name
    if "_" in name or name.isupper():
        return " ".join(part.capitalize() for part in name.split("_"))
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def _combo(items: Sequence[Any], value: Any, spec: Parameter) -> Control:
    # "CleaningCrossSection" and "EACH_ROW" are names for code; the form shows
    # words. Here rather than on the Enum branch alone: a field can also reach a
    # combo through declared `choices`, as cross_section does. A declared
    # format_fn still wins.
    format_fn = spec.format_fn
    if format_fn is None and any(isinstance(item, Enum) for item in items):
        format_fn = _enum_label
    control = ValueComboBox(list(items), value, spec.unit, format_fn=format_fn)
    # ValueComboBox's constructor skips `set_value` for None, which leaves the
    # first item selected. `None` is a real choice for an Optional field -- the
    # trench task's orientation defaults to it -- so select it explicitly.
    if value is None:
        control.set_value(None)

    def read() -> Any:
        # A combo with nothing selected returns None; writing that onto the
        # config would replace a real value with nothing.
        return control.value()

    return Control(
        widget=control,
        read=read,
        write=control.set_value,
        signals=(control.currentIndexChanged,),
    )


def _point(
    value: Point, spec: Parameter, fallback: Optional[Tuple[float, float]]
) -> Control:
    """Two spinboxes for a Point-typed field.

    Dispatched on the field's type, not on the field being called "point", so
    a plugin pattern with its own Point field gets the same row.
    """
    scale = spec.scale or 1.0
    suffix = spec.suffix
    minimum, maximum = _bounds(spec, fallback or POINT_FALLBACK_RANGE)

    def spinbox(initial: float) -> ValueSpinBox:
        box = ValueSpinBox(
            suffix, minimum, maximum, spec.display.step, spec.display.decimals
        )
        box.setValue(initial * scale)
        return box

    point = value if isinstance(value, Point) else Point()
    x_control = spinbox(point.x)
    y_control = spinbox(point.y)

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(x_control)
    layout.addWidget(y_control)

    def read() -> Point:
        return Point(x=x_control.value() / scale, y=y_control.value() / scale)

    def write(new: Any) -> None:
        new = new if isinstance(new, Point) else Point()
        x_control.setValue(new.x * scale)
        y_control.setValue(new.y * scale)

    return Control(
        widget=container,
        read=read,
        write=write,
        signals=(x_control.valueChanged, y_control.valueChanged),
        inputs=(x_control, y_control),
    )


def _bounds(
    spec: Parameter, fallback: Optional[Tuple[float, float]]
) -> Tuple[Optional[float], Optional[float]]:
    """The field's limits in shown units, else the form's own defaults."""
    if spec.shown_range is not None:
        return spec.shown_range
    if fallback is not None:
        return fallback
    return None, None


def _is_optional(annotation: Any) -> bool:
    """Whether the field may legitimately hold None."""
    return get_origin(annotation) is not None and type(None) in get_args(annotation)


def _resolve(annotation: Any) -> Any:
    """Unwrap Optional[T] to T.

    A task config declaring `Optional[str]` is a string field that may be unset,
    not an unrenderable one -- without this it falls through to the read-only
    display.
    """
    if get_origin(annotation) is not None and type(None) in get_args(annotation):
        real = [a for a in get_args(annotation) if a is not type(None)]
        if len(real) == 1:
            return real[0]
    return annotation


def _read_only(value: Any, annotation: Any) -> Control:
    """Show a value this form has no editor for, without letting it be written back.

    A plain QLineEdit holding str(value) looks equivalent and is not: for a
    nested dataclass the box shows a repr that cannot be parsed back, and
    reading it returns a str which then replaces the object on the config --
    destroying it on a focus-out alone. `read` hands back the untouched value
    and `editable` stops anything calling it in the first place.
    """
    name = getattr(annotation, "__name__", None) or str(annotation)
    control = QLineEdit()
    control.setText(str(value))
    control.setReadOnly(True)
    control.setEnabled(False)
    control.setToolTip(
        f"{name} is not editable in this form.\n"
        "Edit it in this task's own settings panel, or in the protocol file."
    )
    return Control(
        widget=control,
        read=lambda: value,
        write=lambda new: control.setText(str(new)),
        signals=(),
        editable=False,
    )


def _scalar_list(value: Any, annotation: Any) -> Control:
    """A list of scalars as comma-separated text.

    Only scalars survive the round trip: a list of dataclasses comes back as a
    list of strings, which is why the caller checks the item type first.
    """
    item_types = get_args(annotation)
    item_type = item_types[0] if item_types else str
    control = QLineEdit()
    control.setPlaceholderText("Enter comma-separated values")
    control.setText(
        ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    )

    def read() -> List[Any]:
        text = control.text().strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(",")]
        try:
            if item_type is int:
                return [int(p) for p in parts]
            if item_type is float:
                return [float(p) for p in parts]
            if item_type is bool:
                return [p.lower() in ("true", "1", "yes") for p in parts]
        except ValueError:
            # Half-typed input is not a reason to throw away what was typed.
            return parts
        return parts

    return Control(
        widget=control,
        read=read,
        write=lambda new: control.setText(
            ", ".join(str(v) for v in new) if isinstance(new, list) else str(new)
        ),
        signals=(control.editingFinished,),
    )


def build_control(
    spec: Parameter,
    value: Any,
    *,
    resolve_choices: Optional[Callable[[str], Sequence[Any]]] = None,
    defaults: FormDefaults = FormDefaults(),
    read_only_fallback: bool = True,
) -> Optional[Control]:
    """Build the control for one field, or None if nothing here can render it.

    Dispatches on the field's type (its annotation), and on the value where the
    field has none. ``resolve_choices`` gives a field whose choices are the
    instrument's (``spec.instrument_choices``) the microscope's, by its
    ``bound_name``; without it such a field offers its current value
    alone.

    A type this form has no editor for gets a read-only Control --
    ``editable=False`` -- so the value stays visible, or None with
    ``read_only_fallback=False``.
    """
    optional = _is_optional(spec.type)
    kind = _resolve(spec.type)

    if spec.instrument_choices and spec.choices is None:
        resolved = resolve_choices(spec.bound_name) if resolve_choices else None
        return _combo(resolved if resolved else [value], value, spec)

    if spec.choices:
        return _combo(spec.choices, value, spec)

    # A Literal is a fixed set of allowable values written in the type itself,
    # which is what `choices` expresses -- render it the same way rather than as a
    # free-text box the operator can mistype.
    if get_origin(kind) is Literal:
        return _combo(list(get_args(kind)), value, spec)

    if kind is Point or isinstance(value, Point):
        return _point(value, spec, defaults.float_range)

    if kind is str:
        control = QFilePathLineEdit() if spec.filepath else QLineEdit()
        control.setText(str(value) if value else "")

        def read_text():
            # An empty box on an Optional[str] field means "unset", not "".
            # str(value) would show the literal text "None" and read it straight
            # back onto the config on a focus-out.
            text = control.text()
            return None if (optional and not text) else text

        return Control(
            widget=control,
            read=read_text,
            write=lambda new: control.setText(str(new) if new else ""),
            signals=(control.editingFinished,),
        )

    # Before int: bool is a subclass of int, so an isinstance check for int
    # would swallow every checkbox field.
    if kind is bool or isinstance(value, bool):
        control = QCheckBox()
        control.setChecked(bool(value))
        return Control(
            widget=control,
            read=control.isChecked,
            write=lambda new: control.setChecked(bool(new)),
            signals=(control.toggled,),
        )

    if kind is int:
        scale = int(round(spec.scale or 1))
        minimum, maximum = _bounds(spec, defaults.int_range)
        # An integer spinbox stepping by less than its own scale cannot reach
        # every representable value, so the step is floored at the scale.
        # int(): QSpinBox.setSingleStep rejects a float outright, and a form's
        # fallback step is naturally written as one.
        step = int(max(spec.display.step or defaults.step or 1, scale))
        control = IntegerValueSpinBox(spec.suffix, minimum, maximum, step)
        control.setValue(int(round(value * scale)))
        return Control(
            widget=control,
            read=lambda: (
                int(round(control.value() / scale)) if scale else control.value()
            ),
            write=lambda new: control.setValue(int(round(new * scale))),
            signals=(control.valueChanged,),
        )

    if kind is float or (kind is None and isinstance(value, (float, int))):
        scale = spec.scale or 1
        minimum, maximum = _bounds(spec, defaults.float_range)
        control = ValueSpinBox(
            spec.suffix,
            minimum,
            maximum,
            spec.display.step if spec.display.step is not None else defaults.step,
            spec.display.decimals
            if spec.display.decimals is not None
            else defaults.decimals,
        )
        control.setValue(value * scale)
        return Control(
            widget=control,
            read=lambda: control.value() / scale if scale else control.value(),
            write=lambda new: control.setValue(new * scale),
            signals=(control.valueChanged,),
        )

    if inspect.isclass(kind) and issubclass(kind, Enum):
        return _combo(list(kind), value, spec)

    if get_origin(kind) is list or (inspect.isclass(kind) and issubclass(kind, list)):
        item_types = get_args(kind)
        if not item_types or item_types[0] in (bool, int, float, str):
            return _scalar_list(value, kind)

    if not read_only_fallback:
        # A milling form: there is no value in showing a field it can't edit.
        logging.warning(
            "No control for field %s of type %r (value %r)",
            spec.name,
            spec.type,
            type(value),
        )
        return None

    # Debug, not warning: the form says this itself, visibly and in the right
    # place -- greyed out with a tooltip naming where to edit it instead.
    logging.debug("No editor for %r; showing it read-only.", kind)
    return _read_only(value, kind)
