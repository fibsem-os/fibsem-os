"""Which implementation each backend's milling and application-file methods resolve to.

Odemis and Demo used to call ThermoMicroscope's `run_milling`, `finish_milling`,
`get_orientation` and `get_application_file` unbound, which fails only at runtime when
the borrowed body reaches something the borrower lacks (FIB-1154). Those bodies use
only base-class members, so they are base-class defaults now, and this pins which
class each backend gets them from.
"""

import pytest

from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.autoscript import ThermoMicroscope
from fibsem.microscopes.simulator import DemoMicroscope
from tests.fm import _odemis_stubs as stubs


@pytest.fixture(scope="module")
def odemis_cls():
    import sys

    saved = {
        name: sys.modules.pop(name)
        for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES
        if name in sys.modules
    }
    stubs.install_odemis_stubs()
    from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope

    yield OdemisThermoMicroscope
    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


def _owner(cls, name):
    """The class in `cls`'s MRO that defines `name`."""
    return next(klass for klass in cls.__mro__ if name in vars(klass))


def test_thermo_runs_the_base_milling_loop_and_resets_the_patterning_mode():
    assert _owner(ThermoMicroscope, "run_milling") is FibsemMicroscope
    assert _owner(ThermoMicroscope, "finish_milling") is ThermoMicroscope
    assert _owner(ThermoMicroscope, "get_application_file") is FibsemMicroscope


def test_odemis_inherits_rather_than_borrows(odemis_cls):
    assert _owner(odemis_cls, "run_milling") is FibsemMicroscope
    assert _owner(odemis_cls, "finish_milling") is odemis_cls
    assert _owner(odemis_cls, "get_orientation") is FibsemMicroscope


def test_demo_keeps_its_own_milling_and_inherits_the_application_file():
    assert _owner(DemoMicroscope, "run_milling") is DemoMicroscope
    assert _owner(DemoMicroscope, "finish_milling") is DemoMicroscope
    assert _owner(DemoMicroscope, "get_application_file") is FibsemMicroscope


def test_milling_loop_is_no_longer_abstract():
    assert "run_milling" not in FibsemMicroscope.__abstractmethods__
    assert "finish_milling" not in FibsemMicroscope.__abstractmethods__


def test_demo_application_file_falls_back_to_the_closest_match():
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    available = microscope.get_available_values("application_file")
    assert microscope.set_default_application_file(available[0]) == available[0]
    with pytest.raises(ValueError):
        microscope.set_default_application_file("not-an-application-file")
    assert (
        microscope.set_default_application_file(available[0] + "x", strict=False)
        == available[0]
    )
