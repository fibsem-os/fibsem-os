"""Which implementation each backend's milling and application-file methods resolve to.

Odemis and Demo used to call ThermoMicroscope's `run_milling`, `finish_milling`,
`get_orientation` and `get_application_file` unbound, which fails only at runtime when
the borrowed body reaches something the borrower lacks (FIB-1154). The milling methods
are the base class's now, over each backend's milling service, and this pins that no
backend has its own. Application files are a ThermoFisher setting, so
their matching stays off the base class: it is
`fibsem.util.application_file.match_application_file`, which ThermoFisher and Demo (a
simulated ThermoFisher system) both call.
"""

import pytest

from fibsem.drivers.autoscript.microscope import ThermoMicroscope
from fibsem.drivers.demo.simulator import DemoMicroscope
from fibsem.microscope import FibsemMicroscope
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
    from fibsem.drivers.odemis.microscope import OdemisThermoMicroscope

    yield OdemisThermoMicroscope
    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


def _owner(cls, name):
    """The class in `cls`'s MRO that defines `name`."""
    return next(klass for klass in cls.__mro__ if name in vars(klass))


MILLING = ("setup_milling", "run_milling", "finish_milling", "draw_rectangle")


def test_thermo_mills_with_the_base_class_methods():
    for name in MILLING:
        assert _owner(ThermoMicroscope, name) is FibsemMicroscope
    # application files are its milling service's, not the microscope's
    assert not hasattr(ThermoMicroscope, "get_application_file")


def test_odemis_inherits_rather_than_borrows(odemis_cls):
    for name in MILLING:
        assert _owner(odemis_cls, name) is FibsemMicroscope
    assert _owner(odemis_cls, "get_orientation") is FibsemMicroscope


def test_demo_mills_with_the_base_class_methods():
    for name in MILLING:
        assert _owner(DemoMicroscope, name) is FibsemMicroscope
    # the Demo's milling code is its service's: the microscope has none of its own
    assert not hasattr(DemoMicroscope, "set_patterning_mode")


def test_application_files_stay_off_the_base_class():
    # a ThermoFisher patterning setting, not something every backend has
    assert not hasattr(FibsemMicroscope, "get_application_file")


def test_the_milling_methods_go_to_each_backend_s_service():
    # concrete on the base class, over ``microscope.milling``: none is abstract
    assert not set(MILLING) & FibsemMicroscope.__abstractmethods__


def test_an_application_file_falls_back_to_the_closest_match():
    from fibsem import utils
    from fibsem.util.application_file import match_application_file

    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    available = microscope.milling_system.application_files
    assert match_application_file(available[0], available) == available[0]
    with pytest.raises(ValueError):
        match_application_file("not-an-application-file", available)
    assert (
        match_application_file(available[0] + "x", available, strict=False)
        == available[0]
    )
