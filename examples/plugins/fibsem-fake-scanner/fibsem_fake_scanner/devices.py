"""The fake scan generator as a fibsem device, and its builder.

``FakeScanner`` is a :class:`fibsem.devices.scanner.Scanner`, so a beam's ``scanner``
role can be bound to it. It adds what this unit has beyond the interface: the input
it digitises, that input's gain and offset, and the test pattern it scans (the
stand-in for a sample), plus a ``self_test`` command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from fibsem.devices import (
    DeviceFault,
    Parameter,
    ParameterMetadata,
    Resources,
    command,
)
from fibsem.devices.scanner import Scanner
from fibsem.display import Display
from fibsem.structures import RangeLimit
from fibsem_fake_scanner import MANUFACTURER
from fibsem_fake_scanner.sdk import (
    INPUTS,
    PATTERNS,
    ControllerError,
    FakeScanController,
)

if TYPE_CHECKING:
    from fibsem.drivers.registry import BuildContext
    from fibsem.structures import DeviceEntry


class FakeScanner(Scanner):
    """A simulated external scan generator on its own controller."""

    input = Parameter(
        str, choices=INPUTS, doc="The input (ADC) the frame is read from."
    )
    gain = Parameter(
        float,
        limits=RangeLimit(0.0, 1.0),
        doc="The selected input's analogue gain, as a fraction of its range.",
        display=Display("Gain", scale=100, unit="%", step=1, decimals=0),
    )
    offset = Parameter(
        float,
        limits=RangeLimit(-1.0, 1.0),
        doc="The selected input's offset, as a fraction of full scale.",
    )
    pattern = Parameter(str, choices=PATTERNS, doc="The test pattern the unit scans.")

    def __init__(
        self, name: str, controller: FakeScanController, parent: Any = None, **kw: Any
    ):
        super().__init__(name=name, parent=parent, **kw)
        self._controller = controller

    def facts(self) -> Dict[str, Any]:
        c = self._controller
        return {
            "serial": c.serial,
            "firmware": c.firmware,
            "address": c.address,
            "inputs": list(INPUTS),
            "max_resolution": list(c.max_resolution),
            "min_dwell_time": c.min_dwell_time,
            "max_dwell_time": c.max_dwell_time,
        }

    # -- the Scanner interface

    def _acquire(self, resolution: Tuple[int, int], dwell_time: float) -> np.ndarray:
        try:
            return self._controller.scan(resolution[0], resolution[1], dwell_time)
        except ControllerError as e:
            # the unit refused: say so as a device fault, not the SDK's own error
            raise DeviceFault(str(e), device=self.name) from e

    def _stop(self) -> None:
        self._controller.abort()

    def _dwell_time_limits(self) -> RangeLimit:
        c = self._controller
        return RangeLimit(c.min_dwell_time, c.max_dwell_time)

    def _resolution_limits(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (1, 1), self._controller.max_resolution

    @command
    def self_test(self) -> Dict[str, Any]:
        """Run the unit's self test and return its report, which goes on the
        experiment's record too."""
        report = self._controller.self_test()
        self.session.record("scanner_self_test", {"device": self.name, **report})
        return report

    # -- parameters

    def read_input(self) -> str:
        return self._controller.input

    def write_input(self, value: str) -> None:
        self._controller.input = value
        # gain and offset are per input: what they read changes with it
        for name in ("gain", "offset"):
            param = self.parameters.get(name)
            if param is not None:
                param.report(param.get_value())

    def read_gain(self) -> float:
        return self._controller.gain[self._controller.input]

    def write_gain(self, value: float) -> None:
        self._controller.gain[self._controller.input] = value

    def read_offset(self) -> float:
        return self._controller.offset[self._controller.input]

    def write_offset(self, value: float) -> None:
        self._controller.offset[self._controller.input] = value

    def read_pattern(self) -> str:
        return self._controller.pattern

    def write_pattern(self, value: str) -> None:
        self._controller.pattern = value

    def metadata_pattern(self) -> ParameterMetadata:
        return ParameterMetadata(choices=PATTERNS)


# The keys a FakeScan entry takes, beside name/type/driver.
ENTRY_KEYS = {"address": "127.0.0.1", "pattern": "gradient", "simulate_delay": False}


def build_fake_scanner(entry: "DeviceEntry", context: "BuildContext") -> FakeScanner:
    """The scanner a ``type: scan_generator, driver: FakeScan`` entry names.

    Entries at one address share one controller connection, through the connect's
    shared scratch space, closed when the microscope disconnects. The unit has its
    own resources: scanning with it doesn't hold the microscope's locks.
    """
    options = {
        key: entry.options.get(key, default) for key, default in ENTRY_KEYS.items()
    }
    unknown = set(entry.options) - set(ENTRY_KEYS)
    if unknown:
        raise ValueError(
            f"unknown keys for a {MANUFACTURER} scanner: {sorted(unknown)}"
        )
    controllers = context.shared.setdefault(MANUFACTURER, {})
    address = str(options["address"])
    if address not in controllers:
        controllers[address] = FakeScanController(
            address, simulate_delay=bool(options["simulate_delay"])
        )
        # shared by every entry at this address, so the builder closes it
        context.on_disconnect(controllers[address].close)
    controller = controllers[address]
    controller.pattern = options["pattern"]
    return FakeScanner(
        entry.name, controller, parent=context.microscope, resources=Resources()
    ).connect()
