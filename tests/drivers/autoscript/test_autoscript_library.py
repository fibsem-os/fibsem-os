"""AutoScript comes from its own environment's folder unless this one has a copy."""

from fibsem import vendor_libraries
from fibsem.drivers.autoscript.setup import LIBRARY, SETUP


def test_the_driver_declares_autoscript_as_its_library():
    assert SETUP.library is LIBRARY
    assert vendor_libraries.libraries()["ThermoFisher"] is LIBRARY
    # the package the driver imports is the one searched for and versioned
    assert LIBRARY.packages[0] == SETUP.api_module


def test_the_old_sys_path_folders_are_candidates():
    for folder in (
        r"C:\Program Files\Thermo Scientific AutoScript",
        r"C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages",
        r"C:\Program Files\Python*\envs\AutoScript",
        r"C:\Program Files\Python*\envs\AutoScript\Lib\site-packages",
    ):
        assert folder in LIBRARY.candidates
