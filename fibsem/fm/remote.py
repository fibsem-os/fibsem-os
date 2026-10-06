"""A fluorescence microscope on another computer, behind today's FM API.

``RemoteFluorescenceMicroscope`` is the FM API over devices
(``fibsem.fm.microscope.FluorescenceMicroscope``), here the remote devices of
``fibsem.devices.drivers.remote``, served by ``fibsem.server.devices`` on the FM's
computer (the METEOR PC, say). It is the same class a local FM uses; all this adds
is connecting to the server:

    fm = RemoteFluorescenceMicroscope.connect("192.168.0.20", 8765, parent=microscope)
    fm.objective.insert()
    image = fm.acquire_image(channel_settings)

A guard reading ``fm.objective.state`` asks the FM's computer, and fails closed with
``RemoteDeviceUnreachable`` if it can't. Acquiring a channel is one call, run on the
FM's computer; only the metadata reads come back separately. The objective's saved
focus position and the channel name and colour stay on this computer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from fibsem.fm.microscope import FM_DEVICE_NAMES, FluorescenceMicroscope

if TYPE_CHECKING:
    from fibsem.devices.drivers.remote import DeviceClient
    from fibsem.microscope import FibsemMicroscope


class RemoteFluorescenceMicroscope(FluorescenceMicroscope):
    """Today's FM API over an FM served from another computer: the FM API over
    devices, here remote ones."""

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        parent: Optional[FibsemMicroscope] = None,
        client: Optional[DeviceClient] = None,
        offline: bool = False,
    ) -> RemoteFluorescenceMicroscope:
        """Connect to the FM's device server.

        Raises ``RemoteDeviceUnreachable`` if it isn't running, unless ``offline``:
        then the FM is built offline, every read fails closed, and it comes online by
        itself when the server starts (``client.reconnected`` fires).
        """
        from fibsem.devices.drivers.remote import connect_remote_fm

        devices = connect_remote_fm(host, port, client=client, offline=offline)
        missing = set(FM_DEVICE_NAMES) - set(devices)
        if missing:
            if devices:
                next(iter(devices.values())).client.close()
            raise RuntimeError(f"{host}:{port} serves no FM {sorted(missing)}")
        fm = cls(devices, parent=parent)
        if fm.online:
            logging.info(f"Connected to the fluorescence microscope at {host}:{port}")
        return fm

    @property
    def online(self) -> bool:
        """Whether the FM's server has answered and its parts are bound. A server
        that answered once and then went away reads True until its next read fails."""
        return all(device.online for device in self.devices.values())

    @property
    def client(self) -> Any:
        return self.devices["fm"].client
