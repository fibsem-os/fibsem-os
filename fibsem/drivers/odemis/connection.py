"""The odemis connection the Odemis devices and services share.

``OdemisThermoMicroscope`` looks up the two odemis components it drives when it is
created and wraps them in an `OdemisConnection`, which it hands to every device and
service it builds. They make their odemis calls through it rather than through the
microscope.

``sdk`` is the ``fibsem`` component (the AutoScript adapter's client), which takes the
channel on every call. ``vendor_stage`` is the ``stage-bare`` component. ``lock`` is the
microscope's ``imaging_channel`` lock (its ``_threading_lock``), held where a call
depends on the active view; the connection does not make one of its own.

This module imports nothing from odemis; the backend's guarded import is the only one.
"""

from __future__ import annotations

from typing import Any

from fibsem.structures import BeamType


class OdemisConnection:
    """The odemis ``fibsem`` client and ``stage-bare`` stage, and the channel lock.

    A plain driver object, not a device: it has no parameters.
    """

    def __init__(self, sdk: Any, vendor_stage: Any, lock: Any):
        self.sdk = sdk
        self.vendor_stage = vendor_stage
        self.lock = lock

    def set_channel(self, channel: BeamType) -> None:
        """Point the active view and device at ``channel``'s beam."""
        self.sdk.set_active_view(channel.value)
        self.sdk.set_active_device(channel.value)
