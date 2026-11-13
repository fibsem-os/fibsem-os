"""`isdeleted`, `delete` and `receivers` for whichever Qt binding qtpy loaded.

PyQt has the first two in `sip`; PySide has them in `shiboken` under other names.
"""

from __future__ import annotations

import qtpy

if qtpy.PYSIDE6:
    import shiboken6

    def isdeleted(obj) -> bool:
        return not shiboken6.isValid(obj)

    def delete(obj) -> None:
        shiboken6.delete(obj)

else:
    from qtpy.sip import delete, isdeleted  # noqa: F401

__all__ = ["delete", "isdeleted", "receivers"]


def receivers(obj, signal) -> int:
    """How many slots are connected to *signal* of *obj*.

    PySide takes the signal's signature as a string, not the bound signal.
    """
    if not qtpy.PYSIDE6:
        return obj.receivers(signal)
    from qtpy.QtCore import SIGNAL, QMetaMethod

    signature = bytes(QMetaMethod.fromSignal(signal).methodSignature()).decode()
    return obj.receivers(SIGNAL(signature))
