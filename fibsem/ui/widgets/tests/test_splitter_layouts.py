"""Standalone test script for exploring QSplitter-based panel layouts.

Run directly:
    python fibsem/ui/widgets/tests/test_splitter_layouts.py
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas
from fibsem.ui.stylesheets import NAPARI_STYLE

_BORDER_NORMAL = QColor("#444444")
_BORDER_SELECTED = QColor("#5ba3f5")
_BORDER_WIDTH = 2


class PanelWidget(QFrame):
    _all: list["PanelWidget"] = []

    def __init__(self, label: str):
        super().__init__()
        PanelWidget._all.append(self)
        self._selected = False
        self.setStyleSheet("background: #2b2d31;")
        lbl = QLabel(label, alignment=Qt.AlignCenter)
        lbl.setStyleSheet("color: #d1d2d4; font-size: 14px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(lbl)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        color = _BORDER_SELECTED if self._selected else _BORDER_NORMAL
        pen = QPen(color, _BORDER_WIDTH)
        painter.setPen(pen)
        half = _BORDER_WIDTH // 2
        painter.drawRect(self.rect().adjusted(half, half, -half, -half))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        for panel in PanelWidget._all:
            panel._selected = False
            panel.update()
        self._selected = True
        self.update()
        super().mousePressEvent(event)


def _panel(label: str) -> PanelWidget:
    return PanelWidget(label)


def _canvas_panel(title: str, image: FibsemImage) -> QWidget:
    w = QWidget()
    w.setStyleSheet("background: #1e2124;")
    lbl = QLabel(title, alignment=Qt.AlignLeft)
    lbl.setStyleSheet("color: #888; font-size: 11px; padding: 2px 6px; background: #1e2124;")
    canvas = FibsemImageCanvas()
    canvas.set_image(image)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(lbl)
    lay.addWidget(canvas)
    return w


def _hsplitter(*widgets) -> QSplitter:
    s = QSplitter(Qt.Horizontal)
    s.setChildrenCollapsible(False)
    for w in widgets:
        s.addWidget(w)
    s.setSizes([1000] * len(widgets))
    return s


def _vsplitter(*widgets) -> QSplitter:
    s = QSplitter(Qt.Vertical)
    s.setChildrenCollapsible(False)
    for w in widgets:
        s.addWidget(w)
    s.setSizes([1000] * len(widgets))
    return s


def splitter_2x2() -> QSplitter:
    sem_img = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
    fib_img = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
    col0 = _vsplitter(_canvas_panel("SEM", sem_img), _panel("R1 C0"))
    col1 = _vsplitter(_canvas_panel("FIB", fib_img), _panel("R1 C1"))
    return _hsplitter(col0, col1)


def splitter_3x1() -> QSplitter:
    return _hsplitter(_panel("C0"), _panel("C1"), _panel("C2"))


def splitter_3x3() -> QSplitter:
    cols = [
        _vsplitter(_panel(f"R0 C{c}"), _panel(f"R1 C{c}"), _panel(f"R2 C{c}"))
        for c in range(3)
    ]
    return _hsplitter(*cols)


def splitter_2x2_br_split() -> QSplitter:
    left = _vsplitter(_panel("R0 C0"), _panel("R1 C0"))
    br = _hsplitter(_panel("R1 C1a"), _panel("R1 C1b"))
    right = _vsplitter(_panel("R0 C1"), br)
    return _hsplitter(left, right)


def splitter_2x2_br_2x2() -> QSplitter:
    left = _vsplitter(_panel("R0 C0"), _panel("R1 C0"))
    br = _hsplitter(
        _vsplitter(_panel("R1 C1 R0 C0"), _panel("R1 C1 R1 C0")),
        _vsplitter(_panel("R1 C1 R0 C1"), _panel("R1 C1 R1 C1")),
    )
    right = _vsplitter(_panel("R0 C1"), br)
    return _hsplitter(left, right)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QWidget()
    win.setWindowTitle("Splitter layouts — test")
    win.setStyleSheet(NAPARI_STYLE + "QWidget { background: #262930; color: #d1d2d4; }")
    win.resize(1200, 800)

    tabs = QTabWidget()
    tabs.setStyleSheet(
        "QTabBar::tab { background: #1e2124; color: #d1d2d4; padding: 6px 14px; }"
        "QTabBar::tab:selected { background: #2b2d31; }"
        "QTabWidget::pane { border: none; }"
    )
    tabs.addTab(splitter_2x2(), "2×2")
    tabs.addTab(splitter_3x1(), "3×1")
    tabs.addTab(splitter_3x3(), "3×3")
    tabs.addTab(splitter_2x2_br_split(), "2×2 BR-split")
    tabs.addTab(splitter_2x2_br_2x2(), "2×2 BR-2×2")

    root = QVBoxLayout(win)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(0)
    root.addWidget(tabs)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
