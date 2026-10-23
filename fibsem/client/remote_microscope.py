"""A `FibsemMicroscope` whose instrument is behind an App Server on another
computer: its devices and services are the server's, used as if they were local.

::

    from fibsem.client.remote_microscope import RemoteMicroscope

    microscope = RemoteMicroscope.connect("http://microscope-pc:8000", token)
    microscope.acquire_image(image_settings)
    microscope.milling.run(stages)  # mills with the server's run

Each device is the remote driver's (`fibsem.drivers.remote.devices`), made with
the class its driver names when that driver is installed here, and each service a
`fibsem.drivers.remote.services.RemoteService`. The experiment stays on the
microscope PC: what a service saves is saved there.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from fibsem.client import FibsemClient
from fibsem.client.control import RemoteControl
from fibsem.client.pairing import DEFAULT_URL, pair, paired_token
from fibsem.client.transport import NO_SERVER, ServerUnreachable, resolve, server_env
from fibsem.drivers.remote.devices import REMOTE_FM_PARTS, DeviceClient
from fibsem.drivers.remote.services import connect_remote_services
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemExperimentRef, FibsemUser

ROOT = "/microscope"


class RemoteMicroscope(FibsemMicroscope):
    """The microscope an App Server serves, reached over its ``/microscope``
    routes with its token."""

    driver_name = "Remote"

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        """The App Server at *url* with *token*; with neither, the one this
        machine's environment or discovery file names (`transport.resolve`).
        `connect_to_microscope` reaches it. With no token, the one this
        computer was paired with (`pair`) at *url*, or at the SSH tunnel's end.

        Only the operator's app reads a paired token: it is the operator's, so
        an agent's client never finds it (`transport.resolve` doesn't look)."""
        env_url, env_token = server_env()
        if token is None and not env_token:
            where = url or env_url or DEFAULT_URL
            if (paired := paired_token(where)) is not None:
                url, token = where, paired
        resolved = resolve(url, token)
        if resolved is None:
            raise ServerUnreachable(NO_SERVER)
        self.url, self._token = resolved
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()
        self._client: Optional[DeviceClient] = None

    @classmethod
    def connect(
        cls, url: Optional[str] = None, token: Optional[str] = None
    ) -> "RemoteMicroscope":
        microscope = cls(url, token)
        split = urlsplit(microscope.url)
        microscope.connect_to_microscope(split.hostname, split.port or 80)
        return microscope

    @staticmethod
    def pair(
        code: str, url: str = DEFAULT_URL, name: Optional[str] = None
    ) -> "RemoteMicroscope":
        """Pair this computer with the App Server at *url* (the SSH tunnel's end
        by default) using the code its Pair a computer shows, and connect. The
        token is kept, so later ``connect(url)`` needs no code."""
        token, _ = pair(code, url, name)
        return RemoteMicroscope.connect(url, token)

    @property
    def client(self) -> FibsemClient:
        """The App Server's client, on this microscope's connection: its ``app``,
        its ``info``."""
        return FibsemClient(transport=self._device_client._transport)

    @property
    def _device_client(self) -> DeviceClient:
        if self._client is None:
            raise ServerUnreachable("the remote microscope is not connected")
        return self._client

    def connect_to_microscope(
        self, ip_address: str, port: int = 8000, reset_beam_shift: bool = False
    ) -> None:
        """Reach the App Server at *ip_address*:*port*: take its configuration,
        then build its devices and services. *reset_beam_shift* is off by
        default: connecting from another computer changes nothing there."""
        from fibsem.devices.entries import ResolvedEntry, build_device_entries
        from fibsem.drivers.remote.devices import REMOTE_DRIVER
        from fibsem.structures import DeviceEntry

        client = DeviceClient(ip_address, port, token=self._token, root=ROOT)
        self._client = client
        # Control is the server's: a run here holds it there, by a lease.
        self.control = RemoteControl(client._transport)
        try:
            self.system = self.client.system
            self.system.info.ip_address = ip_address
            descriptions = client.describe()
            fm_parts = {n: d for n, d in descriptions.items() if n in REMOTE_FM_PARTS}
            others = {n: d for n, d in descriptions.items() if n not in fm_parts}
            resolved = [
                ResolvedEntry(
                    DeviceEntry.from_dict(
                        {
                            "name": name,
                            "type": description.get("type"),
                            "driver": REMOTE_DRIVER,
                            "address": ip_address,
                            "port": port,
                        }
                    ),
                    REMOTE_DRIVER,
                )
                for name, description in others.items()
            ]
            # The builders find this connection as the one to that address.
            shared = {REMOTE_DRIVER: {(ip_address, port): (client, others)}}
            for name, device in build_device_entries(
                resolved, self, shared=shared
            ).items():
                self._set_device(name, device)
            self._bind_device_roles()
            for name, service in connect_remote_services(client).items():
                service.parent = self
                setattr(self, name, service)
            self.fm = self._remote_fm(ip_address, port, fm_parts)
        except Exception:
            self._client = None
            client.close()
            raise
        if reset_beam_shift:
            self.reset_beam_shifts()
        logging.info(
            f"Connected to the {self.system.info.model} at {self.url} "
            f"({len(self.devices)} devices, {len(self.services)} services)"
        )

    def _remote_fm(self, host: str, port: int, parts: Dict[str, Any]) -> Any:
        """The FM over its served parts, on this connection; None when the
        server serves no FM."""
        if set(parts) != set(REMOTE_FM_PARTS):
            return None
        from fibsem.fm.remote import RemoteFluorescenceMicroscope

        fm = RemoteFluorescenceMicroscope.connect(
            host, port, parent=self, client=self._device_client
        )
        self.fm_devices = MappingProxyType(dict(fm.devices or {}))
        return fm

    def _disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        logging.info(f"Disconnected from the microscope at {self.url}")
