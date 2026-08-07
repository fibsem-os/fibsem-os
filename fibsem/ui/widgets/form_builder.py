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

import logging
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Optional, Sequence, Tuple

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
        """Wire every signal that means "the user edited this"."""
        for signal in self.signals:
            signal.connect(slot)

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


def build_control(
    metadata: dict,
    value: Any,
    *,
    dynamic_items: Optional[Callable[[str], Sequence[Any]]] = None,
    float_range: Optional[Tuple[float, float]] = None,
    int_range: Optional[Tuple[int, int]] = None,
) -> Optional[Control]:
    """Build the control for one field, or None if nothing here can render it.

    `metadata` is the merged view from `get_fields_with_metadata`, so every key
    is present. `dynamic_items` resolves `items: "dynamic"` against the
    microscope; without it such a field falls back to its current value alone.

    `float_range` / `int_range` supply bounds for fields that declare none.
    Leave them None to keep the spinbox classes' own defaults.
    """
    items = metadata.get("items")
    type_ = metadata.get("type")

    if items == "dynamic":
        resolved = dynamic_items(metadata["microscope_parameter"]) if dynamic_items else None
        return _combo(resolved if resolved else [value], value, metadata)

    if items:
        return _combo(items, value, metadata)

    if type_ is Point or isinstance(value, Point):
        return _point(value, metadata, float_range)

    if type_ is str:
        control = QFilePathLineEdit() if metadata.get("filepath") else QLineEdit()
        control.setText(str(value) if value else "")
        return Control(
            widget=control,
            read=control.text,
            write=lambda new: control.setText(str(new) if new else ""),
            signals=(control.editingFinished,),
        )

    # Before int: bool is a subclass of int, so an isinstance check for int
    # would swallow every checkbox field.
    if type_ is bool or isinstance(value, bool):
        control = QCheckBox()
        control.setChecked(bool(value))
        return Control(
            widget=control,
            read=control.isChecked,
            write=lambda new: control.setChecked(bool(new)),
            signals=(control.toggled,),
        )

    if type_ is int:
        scale = int(round(effective_scale(metadata) or 1))
        minimum, maximum = _bounds(metadata, int_range)
        # An integer spinbox stepping by less than its own scale cannot reach
        # every representable value, so the step is floored at the scale.
        step = max(metadata.get("step") or 1, scale)
        control = IntegerValueSpinBox(display_suffix(metadata), minimum, maximum, step)
        control.setValue(int(round(value * scale)))
        return Control(
            widget=control,
            read=lambda: int(round(control.value() / scale)) if scale else control.value(),
            write=lambda new: control.setValue(int(round(new * scale))),
            signals=(control.valueChanged,),
        )

    if type_ is float or isinstance(value, (float, int)):
        scale = effective_scale(metadata) or 1
        minimum, maximum = _bounds(metadata, float_range)
        control = ValueSpinBox(
            display_suffix(metadata),
            minimum,
            maximum,
            metadata.get("step"),
            metadata.get("decimals"),
        )
        control.setValue(value * scale)
        return Control(
            widget=control,
            read=lambda: control.value() / scale if scale else control.value(),
            write=lambda new: control.setValue(new * scale),
            signals=(control.valueChanged,),
        )

    logging.warning("No control for field of type %r (value %r)", type_, type(value))
    return None
