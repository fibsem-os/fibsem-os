"""``vertical_move`` is parameterised by the view the offset was measured in (FIB-785).

One method, not two: ``beam_type`` names the view, ION (the FIB) being the default
and the historical behaviour. ``supports_vertical_move`` is the capability query
callers gate on, so nothing has to dispatch on ``hasattr`` to find out whether a
backend can correct from the SEM view.

These run against a real ``DemoMicroscope`` rather than a stand-in: the simulator
reaches the ThermoFisher bodies by delegation, which is exactly the wiring that
would break silently.
"""

import inspect
import os

import pytest

import fibsem.config as fibsem_config
from fibsem import utils
from fibsem.microscope import FibsemMicroscope, ThermoMicroscope
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType

CONFIG_PATH = os.path.join(
    os.path.dirname(fibsem_config.__file__), "config", "microscope-configuration.yaml"
)


@pytest.fixture
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG_PATH)
    return scope


# ---------------------------------------------------------------------------
# the capability query
# ---------------------------------------------------------------------------


def test_the_simulator_corrects_from_both_views(microscope):
    """The simulator delegates to the ThermoFisher bodies, so it has both views.

    Before FIB-785 it had only the FIB one, and the movement widget refused the
    SEM-view Alt-click in demo mode -- the feature could not be exercised at all
    away from hardware.
    """
    assert microscope.supports_vertical_move(BeamType.ION)
    assert microscope.supports_vertical_move(BeamType.ELECTRON)


@pytest.mark.parametrize(
    "cls",
    [ThermoMicroscope, TescanMicroscope, DemoMicroscope],
    ids=lambda c: c.__name__,
)
def test_every_backend_declares_the_views_it_supports(cls):
    assert set(cls.vertical_move_views) == {BeamType.ION, BeamType.ELECTRON}


def test_the_fib_view_is_the_universal_default():
    """A backend that declares nothing still corrects from the FIB view."""
    assert FibsemMicroscope.vertical_move_views == (BeamType.ION,)
    assert (
        inspect.signature(FibsemMicroscope.vertical_move)
        .parameters["beam_type"]
        .default
        is BeamType.ION
    )


@pytest.mark.parametrize(
    "cls",
    [ThermoMicroscope, TescanMicroscope, DemoMicroscope],
    ids=lambda c: c.__name__,
)
def test_every_backend_takes_the_view_as_a_parameter(cls):
    """The signature drift FIB-269 warns about: an override that quietly lost beam_type
    would send every SEM-view correction down the FIB geometry."""
    assert "beam_type" in inspect.signature(cls.vertical_move).parameters


def test_an_unsupported_view_raises_instead_of_moving(microscope):
    """The failure mode being designed out: correcting a SEM-measured offset with the
    FIB's geometry, which is what the old fallbacks did."""
    microscope.vertical_move_views = (BeamType.ION,)
    before = microscope.get_stage_position()

    with pytest.raises(NotImplementedError) as excinfo:
        microscope.vertical_move(dy=5e-6, beam_type=BeamType.ELECTRON)

    assert "DemoMicroscope" in str(excinfo.value)
    assert "ELECTRON" in str(excinfo.value)
    assert microscope.get_stage_position().is_close2(before, tol=1e-9)


# ---------------------------------------------------------------------------
# the two views are two different moves
# ---------------------------------------------------------------------------


def test_the_two_views_move_the_stage_differently(microscope):
    """The SEM view is not the FIB view's mirror image but a superset -- a stable move
    to centre in the SEM, then a height correction to bring the FIB back."""
    start = microscope.get_stage_position()

    microscope.vertical_move(dy=5e-6, beam_type=BeamType.ION)
    from_fib = microscope.get_stage_position()

    microscope.move_stage_absolute(start)
    microscope.vertical_move(dy=5e-6, beam_type=BeamType.ELECTRON)
    from_sem = microscope.get_stage_position()

    assert not from_fib.is_close2(from_sem, tol=1e-9)
    assert not from_fib.is_close2(start, tol=1e-9)
    assert not from_sem.is_close2(start, tol=1e-9)
