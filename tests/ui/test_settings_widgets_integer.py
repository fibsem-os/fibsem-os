"""Headless tests for integer handling in the pattern/strategy settings widgets.

Guards the contract that a metadata-declared ``int`` field round-trips through
the GUI as a Python ``int`` (not a ``float``). Regression coverage for the class
of bug where ``get_pattern`` / ``get_strategy`` leaked floats into strategies
that expect integers.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem import utils
from fibsem.milling.patterning import get_pattern
from fibsem.ui.widgets.custom_widgets import IntegerValueSpinBox
from fibsem.ui.widgets.pattern_settings_widget import FibsemPatternSettingsWidget

# ArrayPattern exposes several integer fields (n_columns, n_rows, passes) with
# no scale, which is exactly the path that regressed.
_INT_FIELDS = {"n_columns": 7, "n_rows": 3, "passes": 4}


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


def _array_pattern(**overrides):
    pattern = get_pattern("ArrayPattern")
    for field, value in overrides.items():
        setattr(pattern, field, value)
    return pattern


def _row(widget, field):
    return next(row for row in widget._rows if getattr(row, "field", None) == field)


def test_integer_fields_use_integer_spinbox(qapp, microscope):
    widget = FibsemPatternSettingsWidget(microscope=microscope, pattern=_array_pattern())
    for field in _INT_FIELDS:
        assert isinstance(_row(widget, field).control, IntegerValueSpinBox), field


def test_get_pattern_returns_int_not_float(qapp, microscope):
    """The core regression: integer fields must come back as ``int``."""
    widget = FibsemPatternSettingsWidget(
        microscope=microscope, pattern=_array_pattern(**_INT_FIELDS)
    )
    result = widget.get_pattern()
    for field, expected in _INT_FIELDS.items():
        value = getattr(result, field)
        assert type(value) is int, f"{field} came back as {type(value).__name__}"
        assert value == expected


def test_set_pattern_roundtrip_preserves_int(qapp, microscope):
    widget = FibsemPatternSettingsWidget(microscope=microscope, pattern=_array_pattern())
    widget.set_pattern(_array_pattern(n_columns=9, n_rows=2, passes=6))
    result = widget.get_pattern()
    assert (type(result.n_columns), result.n_columns) == (int, 9)
    assert (type(result.n_rows), result.n_rows) == (int, 2)
    assert (type(result.passes), result.passes) == (int, 6)


def test_float_fields_remain_float(qapp, microscope):
    """Guard against over-correcting float controls into ints."""
    widget = FibsemPatternSettingsWidget(microscope=microscope, pattern=_array_pattern())
    result = widget.get_pattern()
    assert isinstance(result.width, float)
    assert isinstance(result.height, float)
