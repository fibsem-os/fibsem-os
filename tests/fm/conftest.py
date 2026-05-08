import os
import pytest
import fibsem.config as fconfig


SIM_ARCTIS_CONFIG_PATH = os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")


@pytest.fixture(autouse=True, scope="session")
def use_sim_arctis_config():
    """Use sim-arctis-configuration.yaml for all fm tests (required for FM support)."""
    original = fconfig.DEFAULT_CONFIGURATION_PATH
    fconfig.DEFAULT_CONFIGURATION_PATH = SIM_ARCTIS_CONFIG_PATH
    yield
    fconfig.DEFAULT_CONFIGURATION_PATH = original
