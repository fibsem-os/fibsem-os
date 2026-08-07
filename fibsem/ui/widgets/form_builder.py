"""One control builder for every metadata-driven config form.

The pattern, strategy and milling-settings widgets each carried their own copy of
the same loop: map field metadata to a control, then a matching `isinstance`
ladder in `get_*` to read the value back and a third in `set_*` to push one in.
Three ladders per form, four forms, and they had already drifted -- the strategy
form honoured `format_fn` on comboboxes and the pattern form did not; the pattern
form could resolve `items="dynamic"` against the microscope and the strategy form
could not.

`build_control` returns the control *together with* the adapters that move values
through it, so a form never inspects a control's type. Adding a control type is
one branch here rather than four branches in three places each.

Deliberately Qt-only: no microscope, no pattern or strategy imports. `items:
"dynamic"` needs the microscope, so the caller passes a resolver rather than the
builder reaching for one -- which is also what makes this testable without
hardware.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    get_args,
    get_origin,
)

from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from fibsem import utils
from fibsem.structures import Point
from fibsem.ui.widgets.custom_widgets import (
    IntegerValueSpinBox,
    QFilePathLineEdit,
    QLineEdit,
    ValueComboBox,
    ValueSpinBox,
)

# Fallback spinbox bounds, used when a field declares no minimum/maximum.
#
# `ValueSpinBox` itself falls back to (0.0, 1e6), which is wrong for any field
# that can go negative -- and a floor of zero does not error, it silently clamps.
# The forms that had their own hardcoded ranges keep them by passing this
# explicitly; the milling forms keep the spinbox defaults by passing None.
DEFAULT_FLOAT_RANGE = (-1e10, 1e10)
DEFAULT_INT_RANGE = (-2147483648, 2147483647)


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
        edit was dropped without a word (FIB-526). A slot that genuinely cannot
        take zero arguments now raises instead.
        """
        for signal in self.signals:
            signal.connect(lambda *_: slot())

    def set_blocked(self, blocked: bool) -> None:
        """Block or unblock the inputs, so a programmatic write is not echoed back."""
        for widget in self.inputs:
            widget.blockSignals(blocked)


def effective_scale(metadata: dict) -> Optional[float]:
    """The scale actually applied to a value, raised to `dimensions` if declared.

    An area declares `scale=1e6, dimensions=2` because m² -> µm² is 1e12, not
    1e6. The display *suffix* is still built from the base scale, so that a
    cubic rate reads "mm³/A/s" rather than being prefixed by 1e9.
    """
    base = metadata.get("scale")
    dims = metadata.get("dimensions")
    return (base ** dims) if (base and dims) else base


def display_suffix(metadata: dict) -> str:
    """The spinbox suffix, prefixed for the scale (1e6 + "m" -> "µm").

    Empty when the field declares no unit, even if it declares a scale. The
    milling forms used to render a bare prefix -- a lone "µ" with nothing after
    it -- for a scaled field with no unit.
    """
    unit = metadata.get("unit")
    if not unit:
        return ""
    base = metadata.get("scale")
    return utils._get_display_unit(base, unit) if base else unit


def _combo(items: Sequence[Any], value: Any, metadata: dict) -> Control:
    control = ValueComboBox(
        list(items), value, metadata.get("unit"), format_fn=metadata.get("format_fn")
    )
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


def _point(value: Point, metadata: dict, fallback: Optional[Tuple[float, float]]) -> Control:
    """Two spinboxes for a Point-typed field.

    Dispatched on the declared `type`, not on the field being called "point", so
    a plugin pattern with its own Point field gets the same row.
    """
    scale = effective_scale(metadata) or 1.0
    suffix = display_suffix(metadata)
    minimum, maximum = _bounds(metadata, fallback or POINT_FALLBACK_RANGE)

    def spinbox(initial: float) -> ValueSpinBox:
        box = ValueSpinBox(
            suffix, minimum, maximum, metadata.get("step"), metadata.get("decimals")
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
    metadata: dict, fallback: Optional[Tuple[float, float]]
) -> Tuple[Optional[float], Optional[float]]:
    """Declared bounds, falling back to the form's own defaults when unset."""
    minimum = metadata.get("minimum")
    maximum = metadata.get("maximum")
    if fallback is not None:
        minimum = fallback[0] if minimum is None else minimum
        maximum = fallback[1] if maximum is None else maximum
    return minimum, maximum


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
    control.setText(", ".join(str(v) for v in value) if isinstance(value, list) else str(value))

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
    metadata: dict,
    value: Any,
    *,
    annotation: Any = None,
    dynamic_items: Optional[Callable[[str], Sequence[Any]]] = None,
    defaults: FormDefaults = FormDefaults(),
) -> Optional[Control]:
    """Build the control for one field, or None if nothing here can render it.

    `metadata` is the merged view from `get_fields_with_metadata`, so every key
    is present. `annotation` is the field's resolved type, consulted only where
    the metadata declares no `type`: the milling forms declare one, the task
    configs mostly do not, and dispatching on the value alone cannot tell a
    Literal or an Enum from a plain string.

    `dynamic_items` resolves `items: "dynamic"` against the microscope; without
    it such a field falls back to its current value alone.

    Returns a read-only Control -- `editable=False` -- for a type this form has
    no editor for, rather than None, so the value stays visible.
    """
    items = metadata.get("items")
    type_ = metadata.get("type")
    optional = _is_optional(annotation)
    annotation = _resolve(annotation)
    kind = type_ if type_ is not None else annotation

    if items == "dynamic":
        resolved = dynamic_items(metadata["microscope_parameter"]) if dynamic_items else None
        return _combo(resolved if resolved else [value], value, metadata)

    if items:
        return _combo(items, value, metadata)

    # A Literal is a fixed set of allowable values written in the type itself,
    # which is what `items` expresses -- render it the same way rather than as a
    # free-text box the operator can mistype.
    if get_origin(annotation) is Literal:
        return _combo(list(get_args(annotation)), value, metadata)

    if kind is Point or isinstance(value, Point):
        return _point(value, metadata, defaults.float_range)

    if kind is str:
        control = QFilePathLineEdit() if metadata.get("filepath") else QLineEdit()
        control.setText(str(value) if value else "")

        def read_text():
            # An empty box on an Optional[str] field means "unset", not "".
            # Rendering used to be str(value), so a None field showed the literal
            # text "None" and read it straight back onto the config -- an
            # AcquireReferenceImage filename of "None" instead of the
            # auto-generated one, from a focus-out alone.
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
        scale = int(round(effective_scale(metadata) or 1))
        minimum, maximum = _bounds(metadata, defaults.int_range)
        # An integer spinbox stepping by less than its own scale cannot reach
        # every representable value, so the step is floored at the scale.
        # int(): QSpinBox.setSingleStep rejects a float outright, and a form's
        # fallback step is naturally written as one.
        step = int(max(metadata.get("step") or defaults.step or 1, scale))
        control = IntegerValueSpinBox(display_suffix(metadata), minimum, maximum, step)
        control.setValue(int(round(value * scale)))
        return Control(
            widget=control,
            read=lambda: int(round(control.value() / scale)) if scale else control.value(),
            write=lambda new: control.setValue(int(round(new * scale))),
            signals=(control.valueChanged,),
        )

    if kind is float or (annotation is None and isinstance(value, (float, int))):
        scale = effective_scale(metadata) or 1
        minimum, maximum = _bounds(metadata, defaults.float_range)
        control = ValueSpinBox(
            display_suffix(metadata),
            minimum,
            maximum,
            metadata.get("step") if metadata.get("step") is not None else defaults.step,
            metadata.get("decimals") if metadata.get("decimals") is not None else defaults.decimals,
        )
        control.setValue(value * scale)
        return Control(
            widget=control,
            read=lambda: control.value() / scale if scale else control.value(),
            write=lambda new: control.setValue(new * scale),
            signals=(control.valueChanged,),
        )

    if inspect.isclass(kind) and issubclass(kind, Enum):
        return _combo(list(kind), value, metadata)

    if get_origin(annotation) is list or (
        inspect.isclass(kind) and issubclass(kind, list)
    ):
        item_types = get_args(annotation)
        if not item_types or item_types[0] in (bool, int, float, str):
            return _scalar_list(value, annotation)

    if annotation is None:
        # A milling form, which has no annotation to fall back on: nothing here
        # can render this, and there is no value in showing it disabled.
        logging.warning("No control for field of type %r (value %r)", type_, type(value))
        return None

    # Debug, not warning: the form says this itself, visibly and in the right
    # place -- greyed out with a tooltip naming where to edit it instead.
    logging.debug("No editor for %r; showing it read-only.", kind)
    return _read_only(value, kind)
