"""The electron beam's source tilt and lens alignment on the AutoScript driver.

The fake follows the AutoScript reference: ``electron_beam.source_tilt`` (radians)
and ``electron_beam.lens_alignment`` (metres) each have a ``value`` (a Point; set from
a Point, list or tuple) and ``limits`` (a Limits2d). The ion column has neither.
Nothing here has run on an instrument.
"""

from types import SimpleNamespace

import pytest

from fibsem import utils
from fibsem.drivers.autoscript.connection import AutoScriptConnection
from fibsem.drivers.autoscript.devices import AutoscriptBeam
from fibsem.structures.params import RangeLimit
from fibsem.structures.settings import BeamType, Point

NAMES = ("source_tilt", "lens_alignment")


class FakeControl:
    """A Point value with Limits2d; a write stores what it is given."""

    def __init__(self, x: float, y: float, limit: float) -> None:
        self.value = SimpleNamespace(x=x, y=y)
        span = SimpleNamespace(min=-limit, max=limit)
        self.limits = SimpleNamespace(limits_x=span, limits_y=span)


class BrokenColumn:
    """A column whose AutoScript has no such member: reading it raises."""

    def __getattr__(self, name):
        raise AttributeError(name)


def _beam(beam_type: BeamType, column) -> AutoscriptBeam:
    microscope, _ = utils.setup_session(manufacturer="Demo")
    beams = SimpleNamespace(electron_beam=column, ion_beam=column)
    connection = AutoScriptConnection(
        SimpleNamespace(beams=beams), microscope._threading_lock
    )
    return AutoscriptBeam(beam_type, microscope, connection)


def _column():
    return SimpleNamespace(
        source_tilt=FakeControl(1e-3, -2e-3, 0.01),
        lens_alignment=FakeControl(5e-6, 6e-6, 1e-4),
    )


@pytest.mark.parametrize("name", NAMES)
def test_reads_writes_and_limits_on_the_electron_column(name):
    column = _column()
    beam = _beam(BeamType.ELECTRON, column)
    param = beam.bind_implemented(name)
    assert param is not None
    control = getattr(column, name)
    assert param.get_value() == Point(control.value.x, control.value.y)
    limit = control.limits.limits_x.max
    assert param.limits == {
        "x": RangeLimit(min=-limit, max=limit),
        "y": RangeLimit(min=-limit, max=limit),
    }
    param.set_value(Point(limit / 2, -limit / 2))
    assert control.value == [limit / 2, -limit / 2]


@pytest.mark.parametrize("name", NAMES)
def test_the_ion_column_has_none(name):
    assert _beam(BeamType.ION, _column()).bind_implemented(name) is None


@pytest.mark.parametrize("name", NAMES)
def test_a_column_without_it_has_none(name):
    assert _beam(BeamType.ELECTRON, BrokenColumn()).bind_implemented(name) is None
