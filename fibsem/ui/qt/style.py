"""The application style: Fusion, with what Qt 6 changed put back."""

from __future__ import annotations

import qtpy
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QProxyStyle,
    QStyleFactory,
)

__all__ = ["set_app_style"]

# Qt 6 drops a style sheet's left padding from a spin box's text, which then sits
# against the frame. This is about what the sheets' 4-8 px padding gave on Qt 5.
SPINBOX_TEXT_INDENT = 7


class _Qt6Fusion(QProxyStyle):
    def polish(self, widget):  # QStyle.polish is overloaded; only widgets matter here
        if isinstance(widget, QAbstractSpinBox):
            widget.lineEdit().setTextMargins(SPINBOX_TEXT_INDENT, 0, 0, 0)
        return super().polish(widget)


def set_app_style(app: QApplication) -> None:
    """Fusion for *app*. On Qt 6 through a proxy that fixes spin box text."""
    if not qtpy.QT6:
        app.setStyle("Fusion")
        return
    style = _Qt6Fusion(QStyleFactory.create("Fusion"))
    app.setStyle(style)
    app._fibsem_style = style  # PySide: keep the Python side alive with the app
