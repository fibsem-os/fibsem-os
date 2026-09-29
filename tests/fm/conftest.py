import os

import pytest
import yaml

import fibsem.config as fconfig

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml"
)


@pytest.fixture(autouse=True, scope="package")
def use_sim_arctis_config(tmp_path_factory):
    """Use sim-arctis-configuration.yaml for all fm tests (required for FM support).

    Package scope, not session: a session-scoped swap is only torn down when
    the whole run ends, so every test collected after the first fm test - in
    any other directory, on any xdist worker - silently ran on the compustage
    configuration too (pre-tilt 0, flat boot pose).

    The shipped configuration has the sample scene on, for development; the
    tests here run on a copy with it off, so an FM frame is noise rather than
    a render (a 220-frame tileset went from 8 s to 150 s with the scene on).
    A test that wants the scene enables it itself.
    """
    with open(SIM_ARCTIS_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path_factory.mktemp("fm-config") / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    original = fconfig.DEFAULT_CONFIGURATION_PATH
    fconfig.DEFAULT_CONFIGURATION_PATH = str(path)
    yield
    fconfig.DEFAULT_CONFIGURATION_PATH = original
