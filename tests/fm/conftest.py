import os

import pytest

import fibsem.config as fconfig

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml"
)


@pytest.fixture(autouse=True, scope="package")
def use_sim_arctis_config():
    """Use sim-arctis-configuration.yaml for all fm tests (required for FM support).

    Package scope, not session: a session-scoped swap is only torn down when
    the whole run ends, so every test collected after the first fm test - in
    any other directory, on any xdist worker - silently ran on the compustage
    configuration too (pre-tilt 0, flat boot pose).
    """
    original = fconfig.DEFAULT_CONFIGURATION_PATH
    fconfig.DEFAULT_CONFIGURATION_PATH = SIM_ARCTIS_CONFIG_PATH
    yield
    fconfig.DEFAULT_CONFIGURATION_PATH = original
