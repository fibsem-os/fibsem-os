"""Connect a widget to a model's psygnal signal for the widget's lifetime."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, pyqtSignal

__all__ = ["connect_for_lifetime", "connect_on_gui_thread"]


def connect_for_lifetime(owner: QObject, signal: Any, slot: Callable) -> None:
    """Connect *slot* to the psygnal *signal* until *owner* is destroyed.

    psygnal holds a bound method weakly, but a widget's Python wrapper can
    outlive its C++ object (a lambda on one of its own signals is enough), so a
    model that outlives the widget would keep calling into deleted children.
    """
    signal.connect(slot)
    owner.destroyed.connect(lambda *_: signal.disconnect(slot))


class _Relay(QObject):
    """Carries a psygnal emission's arguments to its parent's thread."""

    fired = pyqtSignal(object)


def _max_args(slot: Callable) -> Optional[int]:
    """How many positional arguments *slot* takes; None for any number."""
    try:
        parameters = inspect.signature(slot).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = 0
    for parameter in parameters:
        if parameter.kind is parameter.VAR_POSITIONAL:
            return None
        if parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional


def connect_on_gui_thread(
    owner: QObject, signal: Any, slot: Callable
) -> Callable[[], None]:
    """Connect *slot* to the psygnal *signal*, called on *owner*'s thread, until
    *owner* is destroyed. Returns a call that disconnects it sooner.

    A model emits from whichever thread changed it; a widget must only be
    touched on the GUI thread. The relay is *owner*'s child, so it lives on that
    thread and goes with it: an emission from another thread is queued there,
    one from the GUI thread is delivered at once. *slot* gets as many of the
    emission's arguments as it takes, as psygnal would give it.
    """
    relay = _Relay(owner)
    relay.fired.connect(lambda args: slot(*args))

    def forward(*args: Any) -> None:
        relay.fired.emit(args)

    signal.connect(forward, max_args=_max_args(slot))

    def disconnect() -> None:
        signal.disconnect(forward, missing_ok=True)

    owner.destroyed.connect(lambda *_: disconnect())
    return disconnect
