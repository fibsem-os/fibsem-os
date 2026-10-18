"""The SharkSEM connection the Tescan devices and services share.

``TescanMicroscope`` opens one ``Automation`` connection at connect and wraps it in a
`TescanConnection`, which it hands to every device and service it builds. They make
their SDK calls through it rather than through the microscope.

SharkSEM is one socket and is NOT thread-safe. UI signal callbacks (e.g.
stage.position.changed -> update_ui) run on the main thread while worker threads drive
movements and acquisition, so two threads can hit the socket at once and corrupt the
byte stream (empty strings -> ValueError, or None -> TypeError). Every SDK call holds
``lock`` (FIB-786).

This module imports nothing from the SDK; the Tescan backend's guarded import is the
only one.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from fibsem.structures import BeamType

TESCAN_BEAM_READY_TIMEOUT = 60  # Max time in seconds to wait for the beam to become ready (busy-wait when using Tescanautomation API)
# Seconds to wait before an ion image that directly follows an electron image. The
# reference pair has always paused here (3 s, then 1 s since #341); why was never
# recorded, so the pause is kept as it was rather than replaced by a status wait.
TESCAN_ELECTRON_TO_ION_SETTLE_TIME = 1


class TescanConnection:
    """One SharkSEM connection: the SDK's ``Automation`` object and its lock.

    A plain driver object, not a device: it has no parameters. ``sdk`` is the
    ``Automation``; ``ip_address`` and ``port`` are where it connected, for the milling
    service's second connection to stop from another thread.
    """

    def __init__(self, sdk: Any, ip_address: str, port: int):
        self.sdk = sdk
        self.ip_address = ip_address
        self.port = port
        self.lock = threading.RLock()
        # The beam of the last requested (non-live) acquisition, for the settle before
        # an ion image that follows an electron one.
        self._last_requested_beam_type: Optional[BeamType] = None

    def disconnect(self) -> None:
        with self.lock:
            self.sdk.Disconnect()

    def beam(self, beam_type: BeamType) -> Any:
        """The vendor beam for ``beam_type``: ``SEM`` or ``FIB``."""
        if not isinstance(beam_type, BeamType):
            raise ValueError(f"Invalid beam type: {beam_type}")

        if beam_type is BeamType.ELECTRON:
            return self.sdk.SEM
        if beam_type is BeamType.ION:
            return self.sdk.FIB

    def wait_for_beam_ready(
        self, beam: Any, beam_type: BeamType, operation: str = "operation"
    ) -> None:
        """Wait for the beam to become ready (not busy)."""
        start_time = time.monotonic()
        while True:
            with self.lock:
                busy = beam.IsBusy()
            if not busy:
                break
            logging.debug(
                f"Waiting for the {beam_type.name} beam to become ready after {operation}."
            )
            if time.monotonic() - start_time > TESCAN_BEAM_READY_TIMEOUT:
                raise TimeoutError(
                    f"{beam_type.name} beam is not ready after {operation}. "
                    f"Timeout of {TESCAN_BEAM_READY_TIMEOUT} seconds expired."
                )
            time.sleep(1)

    def prepare_beam(self, beam_type: BeamType) -> Any:
        """Prepare the beam for imaging, milling, or other operations."""
        beam = self.beam(beam_type)

        with self.lock:
            # check the beam is on
            status = beam.Beam.GetStatus()
            if status != beam.Beam.Status.BeamOn:
                beam.Beam.On()

            # stop the scanning before we start scanning or before automatic procedures,
            beam.Scan.Stop()

        self.wait_for_beam_ready(beam, beam_type, operation="preparation")

        return beam

    def settle_after_electron_image(self, beam_type: BeamType) -> None:
        """Pause before an ion image that directly follows an electron image.

        Requested acquisitions only: the live re-acquire loop passes a beam type, not
        image settings, and does not reach this. The pause used to sit in the shared
        `acquire.take_reference_images` behind a Tescan check; it is the driver's
        business, and here it also covers any other electron-then-ion pair.
        """
        previous = self._last_requested_beam_type
        self._last_requested_beam_type = beam_type
        if previous is BeamType.ELECTRON and beam_type is BeamType.ION:
            time.sleep(TESCAN_ELECTRON_TO_ION_SETTLE_TIME)
