"""The fake scan generator as a fibsem device, and its builder.

``FakeScanner`` is a :class:`fibsem.devices.scanner.Scanner`, so a beam's ``scanner``
role can be bound to it. ``FakeScanUnit`` declares what this unit has beyond the
interface: the input it digitises, that input's gain and offset, and the test pattern
it scans (the stand-in for a sample), plus a ``self_test`` command. The remote copy
(``remote.py``) declares the same through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from fibsem.devices import (
    DeviceFault,
    DeviceParameter,
    ParameterMetadata,
    Resources,
    command,
)
from fibsem.devices.scanner import Scanner
from fibsem.structures import Display, RangeLimit
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


class FakeScanUnit(Scanner):
    """What a FakeScan unit has beyond a Scanner, wherever it is driven from: declared
    once, so the local device (`FakeScanner`) and the remote copy of it
    (``fibsem_fake_scanner.remote.RemoteFakeScanner``) have the same parameters and
    commands."""

    input = DeviceParameter(
        str, choices=INPUTS, doc="The input (ADC) the frame is read from."
    )
    gain = DeviceParameter(
        float,
        limits=RangeLimit(0.0, 1.0),
        doc="The selected input's analogue gain, as a fraction of its range.",
        display=Display("Gain", scale=100, unit="%", step=1, decimals=0),
    )
    offset = DeviceParameter(
        float,
        limits=RangeLimit(-1.0, 1.0),
        doc="The selected input's offset, as a fraction of full scale.",
    )
    pattern = DeviceParameter(
        str, choices=PATTERNS, doc="The test pattern the unit scans."
    )

    @command
    def self_test(self) -> Dict[str, Any]:
        """Run the unit's self test and return its report, which goes on the
        experiment's record too."""
        return self._self_test()

    def _self_test(self) -> Dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} can't self test")


class FakeScanner(FakeScanUnit):
    """A simulated external scan generator on its own controller."""

    def __init__(
        self, name: str, controller: FakeScanController, parent: Any = None, **kw: Any
    ):
        super().__init__(name=name, parent=parent, **kw)
        self._controller = controller

    def facts(self) -> Dict[str, Any]:
        c = self._controller
        return {
            **super().facts(),  # the limits, which a remote copy checks against
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

    def hardware_check(self, report: Any, level: Any) -> None:
        """The READ step every device has, then: at SETTINGS, the gain changed, read
        back and put back; at ACQUIRE, one small frame scanned."""
        from fibsem.tools.hardware_check import Level, check_round_trip

        super().hardware_check(report, level)
        if level.includes(Level.SETTINGS):
            gain = self.gain.get_value()
            check_round_trip(self, "gain", 0.25 if gain != 0.25 else 0.75, report)
        if level.includes(Level.ACQUIRE):
            with report.step(self.name, "scan a frame", (64, 48)) as finding:
                finding.read = self.acquire((64, 48), self.dwell_time_limits.min).shape

    def _self_test(self) -> Dict[str, Any]:
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


def build_fake_scanner(entry: "DeviceEntry", context: "BuildContext") -> FakeScanner:
    """The scanner a ``type: scan_generator, driver: FakeScan`` entry names.

    Entries at one address share one controller connection, through the connect's
    shared scratch space, closed when the microscope disconnects. The unit has its
    own resources: scanning with it doesn't hold the microscope's locks. The entry's
    keys arrive checked against ``setup.SCANNER``, with its defaults filled in.
    """
    options = entry.options
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
