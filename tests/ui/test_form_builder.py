"""The shared control builder: one branch per control type, for every form.

The pattern, strategy and milling-settings widgets each carried their own copy
of this dispatch, plus a matching `isinstance` ladder to read values back and a
third to push them in. They had already drifted apart. These tests pin the
behaviour the single builder now owns, and in particular the two places where it
had to stop keying on field *names*.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from dataclasses import dataclass, field

from PyQt5.QtWidgets import QCheckBox

from fibsem.structures import Point, field_meta, get_fields_with_metadata
from fibsem.ui.widgets.custom_widgets import (
    IntegerValueSpinBox,
    QFilePathLineEdit,
    ValueComboBox,
    ValueSpinBox,
)
from fibsem.ui.widgets.form_builder import build_control, display_suffix, effective_scale


def meta(**kwargs) -> dict:
    """Metadata as a form sees it -- merged, so every key is present."""

    @dataclass
    class Probe:
        value: float = field(default=0.0, metadata=field_meta(**kwargs))

    return get_fields_with_metadata(Probe)["value"]


# --- the two de-hardcoded behaviours ------------------------------------------


def test_a_point_field_is_dispatched_on_its_type_not_its_name(qapp):
    """The compound row keys on `type is Point`, so any field can have one.

    The pattern form used to test `field_name == "point"`, which meant a plugin
    pattern with an equivalent field could not get the two-spinbox row no matter
    what it declared. Nothing here is named "point".
    """
    control = build_control(meta(type=Point, scale=1e6, unit="m"), Point(x=1e-6, y=-2e-6))

    spinboxes = control.widget.findChildren(ValueSpinBox)
    assert len(spinboxes) == 2
    assert (spinboxes[0].value(), spinboxes[1].value()) == (1.0, -2.0)

    assert control.read() == Point(x=1e-6, y=-2e-6)


def test_a_point_control_round_trips_through_its_own_scale(qapp):
    """Scale comes from the metadata, not a module constant.

    The old row hardcoded 1e6 and a "µm" suffix, so a Point declared in any
    other unit would have rendered wrong.
    """
    control = build_control(meta(type=Point, scale=1e3, unit="m"), Point())
    control.write(Point(x=0.25, y=-0.5))

    spinboxes = control.widget.findChildren(ValueSpinBox)
    assert (spinboxes[0].value(), spinboxes[1].value()) == (250.0, -500.0)
    assert control.read() == Point(x=0.25, y=-0.5)


def test_a_scaled_field_with_no_unit_gets_no_suffix():
    """The milling forms used to render a bare SI prefix.

    `_get_display_unit(1e6, "")` is "µ" -- a lone prefix with nothing after it,
    which the spinbox then showed as " µ". The task form's own helper returned
    "" for the same input, and it was the one that was right.
    """
    assert display_suffix(meta(scale=1e6)) == ""
    assert display_suffix(meta(scale=1e6, unit="m")) == "μm"
    assert display_suffix(meta(unit="m")) == "m"


# --- scaling ------------------------------------------------------------------


def test_dimensions_raise_the_scale_but_not_the_suffix():
    """An area is m² -> µm², which is 1e12, but it still reads "µm²"."""
    area = meta(scale=1e6, dimensions=2, unit="m²")

    assert effective_scale(area) == 1e12
    assert display_suffix(area) == "μm²"


# --- one branch per control type, each round-tripping -------------------------


@pytest.mark.parametrize(
    "metadata, value, control_type, new_value",
    [
        (dict(type=bool), True, QCheckBox, False),
        (dict(type=str), "abc", None, "xyz"),
        (dict(type=str, filepath=True), "/tmp/a", QFilePathLineEdit, "/tmp/b"),
        (dict(type=int), 3, IntegerValueSpinBox, 8),
        (dict(type=float, minimum=-10.0), 1.5, ValueSpinBox, -2.5),
        (dict(items=["a", "b"]), "a", ValueComboBox, "b"),
    ],
)
def test_each_control_round_trips(qapp, metadata, value, control_type, new_value):
    control = build_control(meta(**metadata), value)

    if control_type is not None:
        assert isinstance(control.widget, control_type)
    assert control.read() == value

    control.write(new_value)
    assert control.read() == new_value


def test_a_bool_is_a_checkbox_even_though_bool_subclasses_int(qapp):
    """Ordering guard: an isinstance check for int would swallow every checkbox."""
    assert isinstance(build_control(meta(), True).widget, QCheckBox)


def test_an_int_field_steps_by_at_least_its_own_scale(qapp):
    """A step below the scale cannot reach every representable value."""
    control = build_control(meta(type=int, scale=1000, step=1), 2)

    assert control.widget.singleStep() == 1000
    assert control.read() == 2


def test_declared_bounds_win_over_the_fallback(qapp):
    control = build_control(meta(type=float, minimum=-5.0, maximum=5.0), 0.0, float_range=(-1e9, 1e9))

    assert (control.widget.minimum(), control.widget.maximum()) == (-5.0, 5.0)


def test_the_fallback_range_fills_in_undeclared_bounds(qapp):
    """Why the fallback is a parameter rather than baked in.

    `ValueSpinBox` defaults to (0.0, 1e6), which puts a floor of zero on any
    field that never declared a minimum -- and clamping is silent. Each form
    passes the bounds it already had, so a form's ranges do not change just
    because the builder is now shared.
    """
    control = build_control(meta(type=float), 0.0, float_range=(-1e10, 1e10))

    assert (control.widget.minimum(), control.widget.maximum()) == (-1e10, 1e10)


def test_dynamic_items_are_resolved_through_the_caller(qapp):
    """The builder never reaches for a microscope itself.

    That is what keeps it importable and testable without hardware, and keeps
    the microscope out of the strategy and task forms that have no use for one.
    """
    asked = []

    def resolver(parameter):
        asked.append(parameter)
        return ["TopToBottom", "BottomToTop"]

    control = build_control(
        meta(items="dynamic", microscope_parameter="scan_direction", type=str),
        "TopToBottom",
        dynamic_items=resolver,
    )

    assert asked == ["scan_direction"]
    assert isinstance(control.widget, ValueComboBox)
    assert control.read() == "TopToBottom"


def test_dynamic_items_fall_back_to_the_current_value(qapp):
    """No resolver, or nothing cached: the field still shows what it holds."""
    control = build_control(
        meta(items="dynamic", microscope_parameter="scan_direction", type=str), "TopToBottom"
    )

    assert control.read() == "TopToBottom"


def test_an_unrenderable_field_returns_none(qapp):
    """The caller logs and skips; it must not get a control it cannot drive."""
    assert build_control(meta(), object()) is None


# --- the adapters -------------------------------------------------------------


def test_connect_wires_every_signal_of_a_compound_control(qapp):
    """Both spinboxes of a Point row have to report edits, not just the first."""
    control = build_control(meta(type=Point, scale=1e6, unit="m"), Point())
    fired = []
    control.connect(lambda: fired.append(1))

    x_control, y_control = control.widget.findChildren(ValueSpinBox)
    x_control.setValue(5.0)
    y_control.setValue(6.0)

    assert len(fired) == 2


def test_blocking_reaches_the_inputs_of_a_compound_control(qapp):
    """Blocking the container would not stop its children emitting."""
    control = build_control(meta(type=Point, scale=1e6, unit="m"), Point())
    fired = []
    control.connect(lambda: fired.append(1))

    control.set_blocked(True)
    control.write(Point(x=1e-6, y=1e-6))
    control.set_blocked(False)

    assert fired == []
    assert control.read() == Point(x=1e-6, y=1e-6)
