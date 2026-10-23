"""Connect a widget to a model's psygnal signal for the widget's lifetime."""

from __future__ import annotations

from typing import Any, Callable

from PyQt5.QtCore import QObject

__all__ = ["connect_for_lifetime"]


def connect_for_lifetime(owner: QObject, signal: Any, slot: Callable) -> None:
    """Connect *slot* to the psygnal *signal* until *owner* is destroyed.

    psygnal holds a bound method weakly, but a widget's Python wrapper can
    outlive its C++ object (a lambda on one of its own signals is enough), so a
    model that outlives the widget would keep calling into deleted children.
    """
    signal.connect(slot)
    owner.destroyed.connect(lambda *_: signal.disconnect(slot))
