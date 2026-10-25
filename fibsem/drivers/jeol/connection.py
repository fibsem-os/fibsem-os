"""The connection the JEOL devices and services share: JEOL's two clients.

JEOL's external control is two clients: the FIB client (``command``), which also
drives the stage, the view layout, the FIB scan and the process (milling) queue, and
the SEM client (``ecommand``). `JeolMicroscope` opens both at connect and wraps them
in a `JeolConnection`, which it hands to every device and service it builds. They make
every call to JEOL through it, so this is the one place to change when JEOL's calls
change.

Every JEOL call returns a status, or a status and its values, and a status of 0 is
success. `JeolConnection.call` checks it and logs a warning otherwise, as the old
driver did, and hands the devices only the values.

JEOL's client library is not an installed package: it is a folder of top-level
modules (``command``, ``ecommand``, ``models``, ``common``, ``sem_live``, ...).
`load_sdk` serves those modules from that folder (``fibsem.vendor_libraries``) and
imports them at connect, never at import, so listing the drivers never needs it.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any, NamedTuple, Optional, Tuple

from fibsem import vendor_libraries
from fibsem.drivers.jeol.setup import LIBRARY
from fibsem.structures.settings import BeamType

# The SEM client's port; the FIB client's is the driver's (``DRIVER.port``).
SEM_PORT = 22300

# The modules of JEOL's client library the driver uses.
_SDK_MODULES = ("common", "command", "ecommand", "models", "sem_live")

# The FIB scan memory the driver images with at connect: SCAN1, scan type 1 (640x480).
DEFAULT_FIB_SCAN_MODE = 0
DEFAULT_FIB_SCAN_TYPE = 1


class JeolSdk(NamedTuple):
    """The modules of JEOL's client library."""

    common: Any
    command: Any
    ecommand: Any
    models: Any
    sem_live: Any


def load_sdk(api_path: Optional[str] = None) -> JeolSdk:
    """Import JEOL's client library from *api_path*, else from where this PC has it
    (``fibsem.vendor_libraries``: ``FIBSEM_JEOL_API_PATH``, then the folder chosen in
    the guided setup or Preferences)."""
    folder = vendor_libraries.serve("JEOL", LIBRARY, api_path)
    try:
        modules = {name: importlib.import_module(name) for name in _SDK_MODULES}
    except Exception as e:
        raise ImportError(
            f"The JEOL client library could not be imported from {folder}: {e}"
        ) from e
    return JeolSdk(**modules)


def _succeeded(status: Any, allowed: Tuple[int, ...]) -> bool:
    # 0 is success; JEOL's own status enum compares to it too
    return not status or status in allowed


class JeolConnection:
    """JEOL's FIB and SEM clients, one lock over both, and the state they share.

    ``fib_scan_mode`` is the FIB scan memory (SCAN1 to SCAN4, 0 to 3) the FIB images
    with, which the ion beam's resolution and dwell time set.
    """

    def __init__(
        self,
        sdk: JeolSdk,
        fib: Any,
        sem: Any,
        ip_address: str,
        port: int,
        sem_port: int,
    ):
        self.sdk = sdk
        self.fib = fib
        self.sem = sem
        self.ip_address = ip_address
        self.port = port
        self.sem_port = sem_port
        self.lock = threading.RLock()
        self.fib_scan_mode = DEFAULT_FIB_SCAN_MODE

    # -- calls -----------------------------------------------------------------------

    def call(
        self,
        client: Any,
        method: str,
        *args: Any,
        ok: Tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Call *method* on *client* and return its values: None for a bare status,
        the value for one, a tuple for several. A status that is neither success nor
        in *ok* is logged as a warning, not raised: the instrument carries on."""
        with self.lock:
            result = getattr(client, method)(*args, **kwargs)
        if isinstance(result, tuple):
            status, values = result[0], result[1:]
        else:
            status, values = result, ()
        if not _succeeded(status, ok):
            logging.warning(f"JEOL {method}{args} returned status {status}")
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def fib_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.call(self.fib, method, *args, **kwargs)

    def sem_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.call(self.sem, method, *args, **kwargs)

    def wait(self, seconds: float) -> None:
        """Wait for the instrument (a scan, a beam switching on); tests skip it."""
        time.sleep(seconds)

    # -- the view layout -------------------------------------------------------------

    def view(self, beam_type: BeamType) -> Any:
        """The view a beam images in: the SEM on the left, the FIB on the right."""
        views = self.sdk.command.EViewType
        return views.LEFT if beam_type is BeamType.ELECTRON else views.RIGHT

    def select_view(self, beam_type: BeamType) -> None:
        self.fib_call("setSelectedView", self.view(beam_type))

    def set_view_layout(self) -> None:
        """Dual view, the SEM on the left and the FIB on the right, as the driver
        expects."""
        command = self.sdk.command
        self.fib_call("setViewMode", command.EViewMode["DUAL_VIEW"])
        self.fib_call(
            "setViewSignal", command.EViewType["LEFT"], command.EViewSignal["SEM_LED"]
        )
        self.fib_call(
            "setViewSignal", command.EViewType["RIGHT"], command.EViewSignal["FIB"]
        )

    # -- connect and disconnect ------------------------------------------------------

    @classmethod
    def open(
        cls,
        sdk: JeolSdk,
        ip_address: str,
        port: int,
        sem_port: int,
        loglevel: str = "DEBUG",
    ) -> "JeolConnection":
        """Connect both clients and image with the default FIB scan."""
        logging.info(f"JEOL FIB client connecting to [{ip_address}:{port}]")
        # What connect returns is not known well enough to check (the old driver kept
        # it and never read it), so it is logged; the first call fails if it failed.
        fib = sdk.command.JeolFibExternalControlClient(ip_address, port, loglevel)
        logging.info(f"JEOL FIB client connect returned {fib.connect()!r}")
        logging.info(f"JEOL SEM client connecting to [{ip_address}:{sem_port}]")
        sem = sdk.ecommand.PcSemExternalControlClient(ip_address, sem_port, loglevel)
        logging.info(f"JEOL SEM client connect returned {sem.connect_sem()!r}")
        connection = cls(sdk, fib, sem, ip_address, port, sem_port)
        connection.fib_call(
            "setFibScanType", DEFAULT_FIB_SCAN_MODE, DEFAULT_FIB_SCAN_TYPE
        )
        connection.set_view_layout()
        return connection

    def disconnect(self) -> None:
        """Close both clients. Safe to call twice."""
        with self.lock:
            for name in ("fib", "sem"):
                client = getattr(self, name)
                if client is not None:
                    client.close()
                    setattr(self, name, None)
