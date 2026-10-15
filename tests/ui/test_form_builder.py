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

from PyQt5.QtWidgets import QCheckBox

from fibsem.structures import Display, Parameter, Point, RangeLimit
from fibsem.ui.widgets.custom_widgets import (
    IntegerValueSpinBox,
    QFilePathLineEdit,
    ValueComboBox,
    ValueSpinBox,
)
from fibsem.ui.widgets.form_builder import FormDefaults, build_control

SPEC_KEYS = (
    "type",
    "unit",
    "limits",
    "choices",
    "device_parameter",
    "filepath",
)


def spec(**kwargs) -> Parameter:
    """A field as a form sees it: what the value means, and the rest (scale, step,
    the shown unit as ``shown_unit``) its Display."""
    meaning = {key: kwargs.pop(key) for key in SPEC_KEYS if key in kwargs}
    if "shown_unit" in kwargs:
        kwargs["unit"] = kwargs.pop("shown_unit")
    return Parameter(name="value", display=Display(**kwargs), **meaning)


# --- the two de-hardcoded behaviours ------------------------------------------


def test_a_point_field_is_dispatched_on_its_type_not_its_name(qapp):
    """The compound row keys on `type is Point`, so any field can have one.

    The pattern form used to test `field_name == "point"`, which meant a plugin
    pattern with an equivalent field could not get the two-spinbox row no matter
    what it declared. Nothing here is named "point".
    """
    control = build_control(
        spec(type=Point, scale=1e6, unit="m"), Point(x=1e-6, y=-2e-6)
    )

    spinboxes = control.widget.findChildren(ValueSpinBox)
    assert len(spinboxes) == 2
    assert (spinboxes[0].value(), spinboxes[1].value()) == (1.0, -2.0)

    assert control.read() == Point(x=1e-6, y=-2e-6)


def test_a_point_control_round_trips_through_its_own_scale(qapp):
    """Scale comes from the field's display, not a module constant.

    The old row hardcoded 1e6 and a "µm" suffix, so a Point declared in any
    other unit would have rendered wrong.
    """
    control = build_control(spec(type=Point, scale=1e3, unit="m"), Point())
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
    assert spec(scale=1e6).suffix == ""
    assert spec(scale=1e6, unit="m").suffix == "µm"
    assert spec(unit="m").suffix == "m"


# --- scaling ------------------------------------------------------------------


def test_a_scale_that_is_not_an_si_prefix_shows_its_own_unit():
    """An area is m² -> µm², which is 1e12: the prefix rule can't spell that, so
    the display says the unit itself."""
    area = spec(scale=1e12, unit="m²", shown_unit="µm²")

    assert area.scale == 1e12
    assert area.suffix == "µm²"


# --- one branch per control type, each round-tripping -------------------------


@pytest.mark.parametrize(
    "metadata, value, control_type, new_value",
    [
        (dict(type=bool), True, QCheckBox, False),
        (dict(type=str), "abc", None, "xyz"),
        (dict(type=str, filepath=True), "/tmp/a", QFilePathLineEdit, "/tmp/b"),
        (dict(type=int), 3, IntegerValueSpinBox, 8),
        (dict(type=float, limits=RangeLimit(-10.0, 10.0)), 1.5, ValueSpinBox, -2.5),
        (dict(choices=("a", "b")), "a", ValueComboBox, "b"),
    ],
)
def test_each_control_round_trips(qapp, metadata, value, control_type, new_value):
    control = build_control(spec(**metadata), value)

    if control_type is not None:
        assert isinstance(control.widget, control_type)
    assert control.read() == value

    control.write(new_value)
    assert control.read() == new_value


def test_a_bool_is_a_checkbox_even_though_bool_subclasses_int(qapp):
    """Ordering guard: an isinstance check for int would swallow every checkbox."""
    assert isinstance(build_control(spec(), True).widget, QCheckBox)


def test_an_int_field_steps_by_at_least_its_own_scale(qapp):
    """A step below the scale cannot reach every representable value."""
    control = build_control(spec(type=int, scale=1000, step=1), 2)

    assert control.widget.singleStep() == 1000
    assert control.read() == 2


def test_declared_bounds_win_over_the_fallback(qapp):
    control = build_control(
        spec(type=float, limits=RangeLimit(-5.0, 5.0)),
        0.0,
        defaults=FormDefaults(float_range=(-1e9, 1e9)),
    )

    assert (control.widget.minimum(), control.widget.maximum()) == (-5.0, 5.0)


def test_the_fallback_range_fills_in_undeclared_bounds(qapp):
    """Why the fallback is a parameter rather than baked in.

    `ValueSpinBox` defaults to (0.0, 1e6), which puts a floor of zero on any
    field that never declared a minimum -- and clamping is silent. Each form
    passes the bounds it already had, so a form's ranges do not change just
    because the builder is now shared.
    """
    control = build_control(
        spec(type=float), 0.0, defaults=FormDefaults(float_range=(-1e10, 1e10))
    )

    assert (control.widget.minimum(), control.widget.maximum()) == (-1e10, 1e10)


def test_instrument_choices_are_resolved_through_the_caller(qapp):
    """The builder never reaches for a microscope itself.

    That is what keeps it importable and testable without hardware, and keeps
    the microscope out of the strategy and task forms that have no use for one.
    """
    asked = []

    def resolver(parameter):
        asked.append(parameter)
        return ["TopToBottom", "BottomToTop"]

    control = build_control(
        spec(device_parameter="milling.scan_direction", type=str),
        "TopToBottom",
        resolve_choices=resolver,
    )

    assert asked == ["scan_direction"]
    assert isinstance(control.widget, ValueComboBox)
    assert control.read() == "TopToBottom"


def test_instrument_choices_fall_back_to_the_current_value(qapp):
    """No resolver, or nothing cached: the field still shows what it holds."""
    control = build_control(
        spec(device_parameter="milling.scan_direction", type=str),
        "TopToBottom",
    )

    assert control.read() == "TopToBottom"


def test_an_unrenderable_field_returns_none_for_a_milling_form(qapp):
    """The caller logs and skips; it must not get a control it cannot drive."""
    assert build_control(spec(), object(), read_only_fallback=False) is None


def test_an_unrenderable_field_is_shown_read_only_by_default(qapp):
    control = build_control(spec(type=dict), {"a": 1})

    assert control.editable is False
    assert control.read() == {"a": 1}


# --- the adapters -------------------------------------------------------------


def test_connect_wires_every_signal_of_a_compound_control(qapp):
    """Both spinboxes of a Point row have to report edits, not just the first."""
    control = build_control(spec(type=Point, scale=1e6, unit="m"), Point())
    fired = []
    control.connect(lambda: fired.append(1))

    x_control, y_control = control.widget.findChildren(ValueSpinBox)
    x_control.setValue(5.0)
    y_control.setValue(6.0)

    assert len(fired) == 2


def test_blocking_reaches_the_inputs_of_a_compound_control(qapp):
    """Blocking the container would not stop its children emitting."""
    control = build_control(spec(type=Point, scale=1e6, unit="m"), Point())
    fired = []
    control.connect(lambda: fired.append(1))

    control.set_blocked(True)
    control.write(Point(x=1e-6, y=1e-6))
    control.set_blocked(False)

    assert fired == []
    assert control.read() == Point(x=1e-6, y=1e-6)


def test_connect_calls_the_slot_with_no_arguments(qapp):
    """The signals carry different payloads; a form wants none of them.

    PyQt sizes the call to whatever the slot accepts, so a one-argument slot
    silently receives the payload instead of whatever it closed over. The task
    form captured its field name that way and every spinbox, checkbox and combo
    edit was dropped in silence, so the normalisation lives here rather than at
    each call site.
    """
    received = []

    for metadata, value, drive in [
        (
            spec(type=float, limits=RangeLimit(-10.0, 10.0)),
            1.0,
            lambda w: w.setValue(2.0),
        ),
        (spec(type=int), 1, lambda w: w.setValue(2)),
        (spec(), True, lambda w: w.setChecked(False)),
        (spec(choices=("a", "b")), "a", lambda w: w.setCurrentIndex(1)),
    ]:
        control = build_control(metadata, value)
        control.connect(lambda captured="field": received.append(captured))
        drive(control.widget)

    assert received == ["field"] * 4
