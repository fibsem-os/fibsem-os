"""The Demo microscope a test connects to.

``tests/conftest.py`` gives every test a module-scoped ``microscope`` on the
default configuration. A file that needs another configuration, or a fresh
microscope per test, declares its own with ``demo_microscope``:

    microscope = demo_microscope("sim-arctis-configuration.yaml", scope="function")
"""

import os
from typing import Optional

import pytest


def demo_session(config: Optional[str] = None):
    """A connected Demo microscope. ``config`` is a file name in
    ``fibsem/config`` (the shipped ones) or a path; None is the default configuration."""
    from fibsem import config as cfg
    from fibsem import utils

    if config is not None and not os.path.isabs(config):
        config = os.path.join(cfg.SHIPPED_CONFIG_PATH, config)
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        ip_address="localhost",
        config_path=config,
        setup_logging=False,
    )
    return microscope


def demo_microscope(config: Optional[str] = None, scope: str = "module"):
    """A ``microscope`` fixture: ``demo_session(config)``, disconnected after."""

    @pytest.fixture(scope=scope, name="microscope")
    def _microscope():
        microscope = demo_session(config)
        yield microscope
        microscope.disconnect()

    return _microscope
