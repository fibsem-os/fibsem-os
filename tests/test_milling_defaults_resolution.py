"""Which implementation each backend's milling and application-file methods resolve to.

Odemis and Demo used to call ThermoMicroscope's `run_milling`, `finish_milling`,
`get_orientation` and `get_application_file` unbound, which fails only at runtime when
the borrowed body reaches something the borrower lacks (FIB-1154). The milling bodies
use only base-class members, so they are base-class defaults now, and this pins which
class each backend gets them from. Application files are a ThermoFisher setting, so
their matching stays off the base class: it is
`fibsem.util.application_file.match_application_file`, which ThermoFisher and Demo (a
simulated ThermoFisher system) both call.
"""

import pytest

from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.autoscript import ThermoMicroscope
from fibsem.microscopes.simulator import DemoMicroscope, DemoMilling
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


def test_thermo_runs_and_finishes_milling_through_its_service():
    from fibsem.microscopes.autoscript import ThermoMilling
    from fibsem.services.milling import ServiceMilling

    assert _owner(ThermoMicroscope, "run_milling") is ServiceMilling
    assert _owner(ThermoMicroscope, "finish_milling") is ServiceMilling
    # without a service, ThermoFisher's own finish, which resets the patterning mode
    assert "finish_milling" in vars(ThermoMilling)
    assert _owner(ThermoMicroscope, "get_application_file") is ThermoMilling


def test_odemis_inherits_rather_than_borrows(odemis_cls):
    from fibsem.services.milling import ServiceMilling

    assert _owner(odemis_cls, "run_milling") is ServiceMilling
    assert _owner(odemis_cls, "finish_milling") is ServiceMilling
    assert _owner(odemis_cls, "get_orientation") is FibsemMicroscope


def test_demo_runs_and_finishes_milling_through_its_service():
    from fibsem.services.milling import ServiceMilling

    assert _owner(DemoMicroscope, "run_milling") is ServiceMilling
    # its own timed loop stays, for `asynch` and a Demo without an ion beam
    assert "run_milling" in vars(DemoMilling)
    assert _owner(DemoMicroscope, "finish_milling") is ServiceMilling


def test_application_files_stay_off_the_base_class():
    # a ThermoFisher patterning setting, not something every backend has
    assert not hasattr(FibsemMicroscope, "get_application_file")


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
