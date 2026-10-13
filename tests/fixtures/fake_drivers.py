"""The built-in drivers a whole microscope can be connected to without an instrument.

Each entry names the configuration a session on that driver starts from and how to put
its fake SDK in place of the vendor's. :func:`connect` then builds the microscope the
way a session does, through ``utils.setup_session``, so whatever connect builds
(devices, services, routes) is what a test sees.

A new driver joins the contract suite (``tests/test_microscope_contract.py``) by
writing ``tests/fixtures/<driver>_sdk.py`` with an ``install(monkeypatch)`` and adding
one entry here. ``PENDING`` names the built-ins with no whole-microscope fake yet, and
why; a built-in in neither fails ``test_every_built_in_driver_has_a_fake``.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from fibsem import config as cfg
from fibsem import utils
from tests.fixtures import tescan_sdk


@dataclass(frozen=True)
class FakeDriver:
    # The configuration a session on this driver starts from.
    configuration: str
    # Puts the fake SDK in place of the vendor's (monkeypatch -> anything).
    install: Callable[[Any], Any]


def _no_sdk(monkeypatch) -> None:
    """The Demo simulates its instrument itself."""


FAKE_DRIVERS: Dict[str, FakeDriver] = {
    "Demo": FakeDriver(cfg.DEFAULT_CONFIGURATION_PATH, _no_sdk),
    "Tescan": FakeDriver(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml"),
        tescan_sdk.install,
    ),
}

# Built-in drivers whose fakes cover parts of the instrument, not a whole microscope.
PENDING: Dict[str, str] = {
    "ThermoFisher": "the Demo stands in for it (its fakes are per-part recordings)",
    "Odemis": "its stubs cover the FM and parts, not a whole microscope",
}


def connect(driver: str, monkeypatch, configuration: Optional[str] = None):
    """A microscope on *driver*'s fake SDK, connected as a session connects it, from
    *configuration* (the driver's own when not given)."""
    fake = FAKE_DRIVERS[driver]
    fake.install(monkeypatch)
    microscope, _ = utils.setup_session(
        config_path=configuration or fake.configuration,
        manufacturer=driver,
        setup_logging=False,
    )
    return microscope


def built_in_packages() -> Dict[str, str]:
    """Each built-in driver's package under ``fibsem.drivers``, by its registry name,
    so a test over every driver picks up a new built-in."""
    from fibsem.drivers.registry import _BUILT_IN

    return {
        name: target.partition(":")[0].rsplit(".", 1)[-1]
        for name, target in _BUILT_IN.items()
    }
