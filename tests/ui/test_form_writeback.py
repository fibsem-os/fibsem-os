"""Every metadata-driven form writes an edit back, driven through real signals.

The task form's write-back broke completely and no test noticed, because the
tests called the change handler directly. That exercises the write but not the
wiring, and the wiring was what failed: a slot that accepted one argument
received the signal's payload instead of the field name it had closed over, so
the lookup found nothing and the edit vanished without an error.

The four forms share `build_control` but each connects its own slot, so each one
can break this way independently. These tests take the only route a user has:
change the control, then assert the config changed.

Deliberately not asserting on control classes or row internals -- those are the
things that shifted under the last refactor while the real behaviour rotted.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem import utils  # noqa: E402
from fibsem.milling.base import get_strategy  # noqa: E402
from fibsem.milling.patterning import get_pattern  # noqa: E402
from fibsem.structures import FibsemMillingSettings  # noqa: E402
from fibsem.ui.widgets.custom_widgets import (  # noqa: E402
    IntegerValueSpinBox,
    ValueComboBox,
    ValueSpinBox,
)
from fibsem.ui.widgets.milling_settings_widget import FibsemMillingSettingsWidget  # noqa: E402
from fibsem.ui.widgets.pattern_settings_widget import FibsemPatternSettingsWidget  # noqa: E402
from fibsem.ui.widgets.strategy_settings_widget import (  # noqa: E402
    FibsemStrategySettingsWidget,
)

from PyQt5.QtWidgets import QCheckBox  # noqa: E402


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return scope


def _row(widget, field):
    return next(r for r in widget._rows if r.field == field)


def _first_of(widget, control_type, minimum_items: int = 0):
    """The first row whose control is *control_type*, or None."""
    for row in widget._rows:
        if isinstance(row.control.widget, control_type):
            if minimum_items and row.control.widget.count() < minimum_items:
                continue
            return row
    return None


# --- pattern ------------------------------------------------------------------


def test_a_pattern_spinbox_edit_reaches_the_pattern(qapp, microscope):
    widget = FibsemPatternSettingsWidget(microscope, get_pattern("Rectangle"))
    emitted = []
    widget.pattern_changed.connect(emitted.append)

    _row(widget, "width").control.widget.setValue(33.0)

    assert emitted, "editing the control emitted nothing"
    assert emitted[-1].width == pytest.approx(33e-6)


def test_a_pattern_combo_edit_reaches_the_pattern(qapp, microscope):
    """A combo emits currentIndexChanged(int) -- the payload that broke the task form."""
    widget = FibsemPatternSettingsWidget(microscope, get_pattern("Rectangle"))
    emitted = []
    widget.pattern_changed.connect(emitted.append)

    row = _first_of(widget, ValueComboBox, minimum_items=2)
    assert row is not None, "expected a pattern with a combo field"
    row.control.widget.setCurrentIndex(1)

    assert emitted
    assert getattr(emitted[-1], row.field) == row.control.widget.currentData()


@pytest.mark.parametrize("axis", ["x", "y"])
def test_a_pattern_point_edit_reaches_the_pattern(qapp, microscope, axis):
    """The compound row: two spinboxes behind one container.

    Both axes, because a compound control has to wire every one of its inputs --
    dropping the second signal leaves half the row silently dead, and a test that
    only drives x cannot tell.
    """
    widget = FibsemPatternSettingsWidget(microscope, get_pattern("Rectangle"))
    emitted = []
    widget.pattern_changed.connect(emitted.append)

    x_control, y_control = _row(widget, "point").control.widget.findChildren(ValueSpinBox)
    {"x": x_control, "y": y_control}[axis].setValue(5.0)

    assert emitted, f"editing the {axis} spinbox emitted nothing"
    assert getattr(emitted[-1].point, axis) == pytest.approx(5e-6)


# --- strategy -----------------------------------------------------------------


def test_a_strategy_spinbox_edit_reaches_the_config(qapp):
    widget = FibsemStrategySettingsWidget(get_strategy("Overtilt"))
    emitted = []
    widget.strategy_changed.connect(emitted.append)

    _row(widget, "overtilt").control.widget.setValue(4.0)

    assert emitted
    assert emitted[-1].config.overtilt == 4.0


def test_a_strategy_checkbox_edit_reaches_the_config(qapp):
    """A checkbox emits toggled(bool) -- another payload-carrying signal."""
    widget = FibsemStrategySettingsWidget(get_strategy("Overtilt"))
    emitted = []
    widget.strategy_changed.connect(emitted.append)

    row = _first_of(widget, QCheckBox)
    assert row is not None, "expected a strategy with a boolean field"
    row.control.widget.setChecked(not getattr(widget._strategy.config, row.field))

    assert emitted
    assert getattr(emitted[-1].config, row.field) == row.control.widget.isChecked()


# --- milling settings ---------------------------------------------------------


def test_a_milling_settings_edit_reaches_the_settings(qapp, microscope):
    widget = FibsemMillingSettingsWidget(
        microscope=microscope, settings=FibsemMillingSettings()
    )
    emitted = []
    widget.settings_changed.connect(emitted.append)

    row = _first_of(widget, ValueSpinBox)
    assert row is not None
    control = row.control.widget
    control.setValue(control.value() + 1)

    assert emitted
    assert getattr(emitted[-1], row.field) == row.control.read()


def test_a_milling_settings_combo_edit_reaches_the_settings(qapp, microscope):
    widget = FibsemMillingSettingsWidget(
        microscope=microscope, settings=FibsemMillingSettings()
    )
    emitted = []
    widget.settings_changed.connect(emitted.append)

    row = _first_of(widget, ValueComboBox, minimum_items=2)
    assert row is not None
    row.control.widget.setCurrentIndex(1)

    assert emitted
    assert getattr(emitted[-1], row.field) == row.control.widget.currentData()


def test_an_integer_field_survives_the_round_trip_as_an_int(qapp, microscope):
    """Guards the int adapter specifically -- a float here reaches the microscope."""
    widget = FibsemPatternSettingsWidget(microscope, get_pattern("ArrayPattern"))
    emitted = []
    widget.pattern_changed.connect(emitted.append)

    row = _first_of(widget, IntegerValueSpinBox)
    assert row is not None
    row.control.widget.setValue(6)

    assert emitted
    value = getattr(emitted[-1], row.field)
    assert type(value) is int and value == 6
