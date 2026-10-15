"""The fake scanner on another computer, served there by ``fibsem.server.devices``.

A ``driver: remote`` entry reaching a FakeScan scanner is made with this class when
this plugin is installed on the coordinator too (the driver's ``remote_devices``).
``RemoteScanner`` scans over the wire; ``FakeScanUnit`` adds the unit's own
parameters, which then bind to the server's. Without the plugin here, the entry is a
plain ``RemoteScanner``, without them.
"""

from __future__ import annotations

from typing import Any, Dict

from fibsem.drivers.remote.devices import RemoteScanner
from fibsem_fake_scanner.devices import FakeScanUnit


class RemoteFakeScanner(RemoteScanner, FakeScanUnit):
    """A FakeScan scanner whose controller is on the device server's computer."""

    def _self_test(self) -> Dict[str, Any]:
        return self.call_command("self_test")
