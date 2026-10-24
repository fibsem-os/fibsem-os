"""The odemis connection the Odemis devices and services share.

``OdemisThermoMicroscope`` looks up the two odemis components it drives when it is
created and wraps them in an `OdemisConnection`, which it hands to every device and
service it builds. They make their odemis calls through it rather than through the
microscope.

``sdk`` is the ``fibsem`` component (the AutoScript adapter's client), which takes the
channel on every call. ``vendor_stage`` is the ``stage-bare`` component. ``lock`` is the
microscope's ``imaging_channel`` lock (its ``_threading_lock``), held where a call
depends on the active view; the connection does not make one of its own.

Both are wrapped in `PlainCalls`, as the FM's components are, so nothing fibsem sends
odemis carries a numpy value (see there).

This module imports nothing from odemis; the backend's guarded import is the only one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fibsem.structures.settings import BeamType


def plain(value: Any) -> Any:
    """*value* with every numpy scalar and array in it as plain Python: what can cross
    to odemis. Anything else passes through unchanged."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return type(value)(plain(v) for v in value)
    return value


class PlainCalls:
    """An odemis component, VA or stream as fibsem uses it, sending only plain Python.

    The odemis back end runs on the system Python with that system's numpy (1.x on
    Ubuntu 22.04) and unpickles every request; fibsem runs on numpy 2, whose pickles
    name ``numpy._core``, which numpy 1 does not have. A numpy value in a request then
    fails: a VA write raises ``ModuleNotFoundError``, and a move's future never
    completes. Through this every argument of a call and every VA write
    (``.value = ...``) goes out as `plain`. Reads come back untouched, and so does any
    other attribute assigned (a test configuring a stub).

    `raw` gives back the wrapped object, for odemis's own type checks (``hasVA``) and
    for odemis code that needs the component itself (a ``FluoStream``).
    """

    def __init__(self, target: Any):
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if callable(attr):
            # no functools.wraps: a Pyro remote method answers every attribute,
            # __name__ included, with another remote method
            def call(*args: Any, **kwargs: Any) -> Any:
                return attr(*plain(args), **plain(kwargs))

            return call
        if hasattr(attr, "subscribe"):  # a VA (or a dataflow); not .value, a live read
            return PlainCalls(attr)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, plain(value) if name == "value" else value)

    def __repr__(self) -> str:
        return f"PlainCalls({self._target!r})"


def raw(obj: Any) -> Any:
    """What *obj* wraps, if it is a `PlainCalls`; otherwise *obj*."""
    return obj._target if isinstance(obj, PlainCalls) else obj


class OdemisConnection:
    """The odemis ``fibsem`` client and ``stage-bare`` stage, and the channel lock.

    A plain driver object, not a device: it has no parameters.
    """

    def __init__(self, sdk: Any, vendor_stage: Any, lock: Any):
        self.sdk = PlainCalls(sdk)
        self.vendor_stage = PlainCalls(vendor_stage)
        self.lock = lock

    def set_channel(self, channel: BeamType) -> None:
        """Point the active view and device at ``channel``'s beam."""
        self.sdk.set_active_view(channel.value)
        self.sdk.set_active_device(channel.value)
