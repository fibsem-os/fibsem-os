import pytest

from fibsem.utils import (
    _get_scale_from_value,
    _get_prefix_from_scale,
    _get_display_unit,
    format_value,
)


def test_get_scale_from_value_basic():
    assert _get_scale_from_value(0) == 1.0
    assert _get_scale_from_value(1) == 1.0
    assert _get_scale_from_value(1e-8) == pytest.approx(1e9)
    assert _get_scale_from_value(1e-11) == pytest.approx(1e12)


def test_get_prefix_from_scale_rounding_and_multiplier():
    prefix, multiplier = _get_prefix_from_scale(1e-8)
    assert prefix == "G"
    assert multiplier == pytest.approx(0.1)

    prefix, multiplier = _get_prefix_from_scale(1e9)
    assert prefix == "n"
    assert multiplier == pytest.approx(1.0)


def test_get_prefix_from_scale_small_values():
    prefix, multiplier = _get_prefix_from_scale(1e12)
    assert prefix == "p"
    assert multiplier == pytest.approx(1.0)

    prefix, multiplier = _get_prefix_from_scale(1e6)
    assert prefix == "μ"
    assert multiplier == pytest.approx(1.0)

    prefix, multiplier = _get_prefix_from_scale(1e9)
    assert prefix == "n"
    assert multiplier == pytest.approx(1.0)


def test_get_display_unit():
    scale = _get_scale_from_value(2e-3)  # -> 1e3, prefix 'm'
    assert _get_display_unit(scale, "m") == "mm"
    assert _get_display_unit(scale) == "m"


def test_format_value_auto_scale_and_override():
    assert format_value(1e-8, unit="m", precision=2) == "10.00 nm"

    # override scale (non-canonical) still produces correct scaling via multiplier
    assert format_value(5, unit="m", precision=3, scale=1e-8) == "0.000 Gm"

    # override with canonical scale
    assert format_value(5000, unit="m", precision=1, scale=0.001) == "5.0 km"

    # tiny values between micro and pico
    assert format_value(2.5e-6, unit="m", precision=1) == "2.5 μm"
    assert format_value(7e-10, unit="m", precision=1) == "700.0 pm"
    assert format_value(3e-12, unit="m", precision=2) == "3.00 pm"
    assert format_value(0.2e-9, unit="A", precision=0) == "200 pA"
    assert format_value(0.2e-9, unit="A", precision=1, scale=1e9) == "0.2 nA"
    assert format_value(0.2e-9, unit="A", precision=1, scale=1e8) == "0.2 nA"
