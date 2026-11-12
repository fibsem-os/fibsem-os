"""`isdeleted` and `delete` for whichever Qt binding qtpy loaded.

PyQt has these in `sip`; PySide has them in `shiboken` under other names.
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

__all__ = ["delete", "isdeleted"]
