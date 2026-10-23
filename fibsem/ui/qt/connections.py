"""Connect a widget to a model's psygnal signal for the widget's lifetime."""

from __future__ import annotations

import inspect
import weakref
from typing import Any, Callable, Optional

from PyQt5 import sip
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
    """Carries a psygnal emission's arguments to its parent's thread.

    psygnal holds `forward` weakly, as it holds any bound method, and the parent
    keeps the relay: dropping the parent drops the connection with it, whether
    or not it was ever closed.
    """

    fired = pyqtSignal(object)

    def __init__(self, owner: QObject, slot: Callable) -> None:
        super().__init__(owner)
        # Weakly for a bound method, as psygnal would have held it.
        self._slot = (
            weakref.WeakMethod(slot) if inspect.ismethod(slot) else lambda: slot
        )
        self.fired.connect(self._deliver)

    def forward(self, *args: Any) -> None:
        if not sip.isdeleted(self):
            self.fired.emit(args)

    def _deliver(self, args: tuple) -> None:
        slot = self._slot()
        if slot is not None:
            slot(*args)


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

    Nothing here keeps *owner* alive: a widget dropped without being closed
    goes, and takes the connection with it.
    """
    relay = _Relay(owner, slot)
    signal.connect(relay.forward, max_args=_max_args(slot))

    def disconnect() -> None:
        try:
            signal.disconnect(relay.forward, missing_ok=True)
        except AttributeError:
            # Collected in the same pass as *owner*, the signal is already half
            # gone; an exception in a Qt slot aborts the process.
            pass

    owner.destroyed.connect(lambda *_: disconnect())
    return disconnect
