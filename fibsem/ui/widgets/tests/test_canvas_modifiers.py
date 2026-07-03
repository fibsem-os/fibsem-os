"""Standalone test for FibsemImageCanvas modifier-aware mouse signals.

Run directly:
    python fibsem/ui/widgets/tests/test_canvas_modifiers.py

Click / double-click / right-click / scroll on the image while holding
Alt / Shift / Ctrl and watch each event — with its modifier tuple — appear in
the log panel.  This exercises the widened signals:

    canvas_clicked        (x, y, modifiers)
    canvas_double_clicked (x, y, modifiers)
    canvas_right_clicked  (x, y, modifiers)
    canvas_scrolled       (x, y, direction, modifiers)

The "legacy 2-arg slot" counter is connected to ``canvas_clicked`` with a
2-argument slot; it proves the widening stays backward-compatible (PyQt5
truncates the extra ``modifiers`` argument for slots that don't want it).
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import FibsemImage
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas


class ModifierTestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Canvas modifiers — test")
        self.resize(1100, 700)

        image = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
        self.canvas = FibsemImageCanvas()
        self.canvas.set_image(image)

        instructions = QLabel(
            "Click · double-click · right-click · scroll on the image.\n"
            "Hold Alt / Shift / Ctrl and watch the modifier tuple in the log.\n"
            "Top-right buttons: reset · scalebar · crosshair · contrast/gamma "
            "(open the last one and drag Min / Max / Gamma)."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #d1d2d4; padding: 6px;")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setStyleSheet(
            "QPlainTextEdit { background: #1e2124; color: #d1d2d4;"
            " font-family: monospace; font-size: 12px; border: 1px solid #444; }"
        )

        self.legacy_label = QLabel("legacy 2-arg slot (backward-compat): fired 0×")
        self.legacy_label.setStyleSheet("color: #888; padding: 4px 6px;")
        self._legacy_count = 0

        clear_btn = QPushButton("Clear log")
        clear_btn.clicked.connect(self.log.clear)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(instructions)
        right.addWidget(self.log)
        right.addWidget(self.legacy_label)
        right.addWidget(clear_btn)
        right_w = QWidget()
        right_w.setLayout(right)

        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas, stretch=3)
        layout.addWidget(right_w, stretch=2)

        # New 3-arg / 4-arg slots
        self.canvas.canvas_clicked.connect(self._on_click)
        self.canvas.canvas_double_clicked.connect(self._on_double)
        self.canvas.canvas_right_clicked.connect(self._on_right)
        self.canvas.canvas_scrolled.connect(self._on_scroll)
        # Legacy 2-arg slot on the same (widened) signal — must still fire
        self.canvas.canvas_clicked.connect(self._legacy_click)

    @staticmethod
    def _fmt(mods) -> str:
        return f"({', '.join(mods)})" if mods else "(none)"

    def _on_click(self, x, y, mods):
        self.log.appendPlainText(
            f"click          x={x:7.1f} y={y:7.1f}  mods={self._fmt(mods)}"
        )

    def _on_double(self, x, y, mods):
        self.log.appendPlainText(
            f"double-click   x={x:7.1f} y={y:7.1f}  mods={self._fmt(mods)}"
        )

    def _on_right(self, x, y, mods):
        self.log.appendPlainText(
            f"right-click    x={x:7.1f} y={y:7.1f}  mods={self._fmt(mods)}"
        )

    def _on_scroll(self, x, y, direction, mods):
        arrow = "up" if direction == 1 else "down"
        self.log.appendPlainText(
            f"scroll {arrow:<4}    x={x:7.1f} y={y:7.1f}  mods={self._fmt(mods)}"
        )

    def _legacy_click(self, x, y):
        """2-arg slot connected to the 3-arg canvas_clicked (truncation check)."""
        self._legacy_count += 1
        self.legacy_label.setText(
            f"legacy 2-arg slot (backward-compat): fired {self._legacy_count}×"
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ModifierTestWidget()
    win.setStyleSheet(NAPARI_STYLE + "QWidget { background: #262930; color: #d1d2d4; }")
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
