"""One home for rendering a value as text.

`utils.format_value` already existed and was used in ~38 places, but four widgets and
the CLI had each written their own anyway, because it could not express three things
they needed: a value the instrument did not report, a zero that stays in the units of
its neighbours, and a precision that varies with the SI band. The named helpers here
are those three conventions, and the tests below are mostly *equivalence* tests -- the
originals are transcribed and the replacements asserted to agree, because a formatting
consolidation is only worth doing if nothing on screen moves.
"""

import math

import pytest

from fibsem import constants
from fibsem.utils import (
    NOT_AVAILABLE,
    SI_PREFIXES,
    format_angle,
    format_current,
    format_distance,
    format_value,
    format_voltage,
)

# ---------------------------------------------------------------------------
# The micro sign
# ---------------------------------------------------------------------------


def test_the_si_table_uses_the_house_micro_sign():
    """U+00B5 MICRO SIGN, not U+03BC GREEK SMALL LETTER MU.

    They are indistinguishable on screen and unequal to every string comparison, and
    this table used to hold the Greek letter while `constants.MICRON_SYMBOL` -- which
    every spin-box suffix and scalebar uses -- held the micro sign. `imaging/drawing.py`
    records why the micro sign is the house character: the default Windows font has no
    glyph for the Greek one, so it renders as a box on the platform most instruments
    are driven from.

    Asserted by codepoint rather than by `==` on a literal, because a literal in this
    file could be the wrong character and the test would still pass.
    """
    assert ord(SI_PREFIXES[-6]) == 0x00B5
    assert SI_PREFIXES[-6] == constants.MU_SYMBOL


def test_format_value_agrees_with_the_micron_symbol_used_everywhere_else():
    assert format_value(150e-6, "m").endswith(constants.MICRON_SYMBOL)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [format_value, format_angle, format_distance, format_current, format_voltage]
)
def test_every_formatter_renders_a_missing_value(fn):
    """The reason four widgets wrote their own: `format_value(None)` raised.

    `MicroscopeState` is full of `Optional` fields -- a beam that was never configured
    reports `None` for its current -- so a formatter that raises on None cannot be used
    to render one, which is how each of these came to exist separately.
    """
    assert fn(None) == NOT_AVAILABLE


def test_a_missing_value_is_distinguishable_from_zero():
    """The distinction an operator has to be able to make at a glance: the instrument
    did not report this, versus it reported zero."""
    assert format_current(None) != format_current(0.0)
    assert format_distance(None) != format_distance(0.0)


# ---------------------------------------------------------------------------
# Equivalence with what each site used to do
# ---------------------------------------------------------------------------


def _original_distance(metres: float) -> str:
    """Transcribed from `drag_distance._fmt_distance`, which
    `canvas/overlays/ruler_overlay._format_distance` duplicated line for line."""
    magnitude = abs(metres)
    if magnitude == 0:
        return "0 nm"
    if magnitude < 1e-6:
        return f"{metres * 1e9:.1f} nm"
    if magnitude < 1e-3:
        return f"{metres * 1e6:.2f} µm"
    if magnitude < 1.0:
        return f"{metres * 1e3:.3f} mm"
    return f"{metres:.4f} m"


def _original_current(amps: float) -> str:
    """Transcribed from `coincidence_milling_confirmation_dialog._format_current`."""
    if amps >= 1e-9:
        return f"{amps * constants.SI_TO_NANO:.1f} nA"
    return f"{amps * constants.SI_TO_PICO:.0f} pA"


def _original_voltage(volts: float) -> str:
    """Transcribed from `hud_ticker._kv`."""
    return f"{volts / 1e3:.2f} kV"


def _original_angle(radians: float) -> str:
    """Transcribed from `hud_ticker._deg`."""
    return f"{math.degrees(radians):.1f}°"


DISTANCES = [0.0, 1e-12, 1e-9, 5e-9, 999e-9, 1e-6, 150e-6, 1e-3, 7e-3, 0.5, 1.0, 12.3]
CURRENTS = [1e-13, 50e-12, 2e-11, 1e-9, 2e-9, 30e-9, 1e-6]
VOLTAGES = [0.0, 500.0, 2000.0, 8000.0, 30000.0]
ANGLES = [0.0, math.pi, -math.pi / 4, 0.6108652381980153]


@pytest.mark.parametrize("metres", DISTANCES)
def test_format_distance_matches_the_two_copies_it_replaced(metres):
    assert format_distance(metres) == _original_distance(metres)


@pytest.mark.parametrize("amps", CURRENTS)
def test_format_current_matches_the_dialog_it_replaced(amps):
    assert format_current(amps) == _original_current(amps)


@pytest.mark.parametrize("volts", VOLTAGES)
def test_format_voltage_matches_the_hud_it_replaced(volts):
    assert format_voltage(volts) == _original_voltage(volts)


@pytest.mark.parametrize("radians", ANGLES)
def test_format_angle_matches_the_hud_it_replaced(radians):
    assert format_angle(radians) == _original_angle(radians)


# ---------------------------------------------------------------------------
# The band conventions, stated rather than inferred
# ---------------------------------------------------------------------------


def test_distance_precision_follows_the_prefix():
    """A stage move is interesting to the nanometre, a stage position to the micron."""
    assert format_distance(5e-9) == "5.0 nm"
    assert format_distance(2.5e-6) == f"2.50 {constants.MICRON_SYMBOL}"
    assert format_distance(3e-3) == "3.000 mm"
    assert format_distance(1.5) == "1.5000 m"


def test_zero_distance_stays_in_nanometres():
    """`format_value` lands zero on bare metres -- `_get_scale_from_value` returns 1.0
    for it -- and "0.00 m" in a column of micron readings reads as another quantity."""
    assert format_distance(0.0) == "0 nm"
    assert format_value(0.0, "m") == "0.00 m"


def test_current_does_not_run_past_nanoamps():
    """Milling currents span three orders of magnitude and the operator thinks in
    pA/nA; switching to µA at the top of the range breaks the comparison."""
    assert format_current(60e-12) == "60 pA"
    assert format_current(1e-9) == "1.0 nA"
    assert format_current(1e-6) == "1000.0 nA"


def test_voltage_is_pinned_to_kilovolts():
    """So a 500 V landing energy and a 30 kV beam stay comparable down a column.
    `format_value` would auto-scale the first to volts."""
    assert format_voltage(500) == "0.50 kV"
    assert format_voltage(30000) == "30.00 kV"
    assert format_value(500, "V") == "500.00 V"


def test_angle_is_degrees_not_an_si_prefix():
    """Nobody reads a stage tilt in milliradians, which is what `format_value` would
    give a small angle."""
    assert format_angle(math.radians(35)) == "35.0°"
    assert format_angle(0.0) == "0.0°"
