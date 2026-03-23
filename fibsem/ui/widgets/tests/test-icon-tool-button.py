"""Test/demo script for IconToolButton — shows all usage patterns."""
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
)
from PyQt5.QtCore import Qt

from fibsem.ui.widgets.custom_widgets import IconToolButton
from fibsem.ui import stylesheets


def _section(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet("QGroupBox { color: #aaa; border: 1px solid #444; margin-top: 8px; padding: 8px; }"
                    "QGroupBox::title { subcontrol-origin: margin; left: 8px; }")
    return g


def _row(*widgets, label: str = "") -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    if label:
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #888; font-size: 11px; min-width: 220px;")
        row.addWidget(lbl)
    for w in widgets:
        row.addWidget(w)
    row.addStretch()
    return row


class IconToolButtonDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IconToolButton Demo")
        self.setStyleSheet("background-color: #262930; color: #d6d6d6;")
        self.resize(560, 560)

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setAlignment(Qt.AlignTop)

        # ── 1. Non-checkable action buttons ───────────────────────────────
        g1 = _section("Non-checkable action buttons")
        v1 = QVBoxLayout(g1)
        v1.setSpacing(6)

        btn_refresh = IconToolButton(icon="mdi:refresh", tooltip="Refresh")
        btn_add     = IconToolButton(icon="mdi:plus", tooltip="Add item")
        btn_trash   = IconToolButton(icon="mdi:trash-can-outline", tooltip="Delete")
        btn_edit    = IconToolButton(icon="mdi:pencil", tooltip="Edit")

        v1.addLayout(_row(btn_refresh, btn_add, btn_trash, btn_edit,
                          label="refresh / plus / trash / pencil"))
        root.addWidget(g1)

        # ── 2. Checkable — enable/disable (checkbox style) ─────────────────
        g2 = _section("Checkable — enable/disable (checkbox icon swap)")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(6)

        btn_en_off = IconToolButton(
            icon="mdi:checkbox-blank-outline",
            checked_icon="mdi:checkbox-marked-outline",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Enable feature",
            checked_tooltip="Disable feature",
            checked=False,
        )
        btn_en_on = IconToolButton(
            icon="mdi:checkbox-blank-outline",
            checked_icon="mdi:checkbox-marked-outline",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Enable feature",
            checked_tooltip="Disable feature",
            checked=True,
        )

        v2.addLayout(_row(btn_en_off, label="starts unchecked"))
        v2.addLayout(_row(btn_en_on,  label="starts checked"))
        root.addWidget(g2)

        # ── 3. Checkable — advanced/tune ──────────────────────────────────
        g3 = _section("Checkable — advanced/tune (icon + color swap)")
        v3 = QVBoxLayout(g3)
        v3.setSpacing(6)

        btn_adv = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )
        btn_adv_on = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
            checked=True,
        )

        v3.addLayout(_row(btn_adv,    label="starts unchecked"))
        v3.addLayout(_row(btn_adv_on, label="starts checked"))
        root.addWidget(g3)

        # ── 4. Checkable — collapse/expand (chevron) ──────────────────────
        g4 = _section("Checkable — collapse/expand (chevron, no color swap)")
        v4 = QVBoxLayout(g4)
        v4.setSpacing(6)

        btn_collapse = IconToolButton(
            icon="mdi:chevron-down",
            checked_icon="mdi:chevron-up",
            tooltip="Expand",
            checked_tooltip="Collapse",
            checked=False,
        )

        v4.addLayout(_row(btn_collapse, label="chevron toggle"))
        root.addWidget(g4)

        # ── 5. Fixed size + disabled ───────────────────────────────────────
        g5 = _section("Fixed size / display-only (disabled)")
        v5 = QVBoxLayout(g5)
        v5.setSpacing(6)

        btn_count = IconToolButton(icon="mdi:numeric-3-box-outline", size=32,
                                   tooltip="3 stages")
        btn_count.setEnabled(False)

        v5.addLayout(_row(btn_count, label="size=32, disabled (stage count)"))
        root.addWidget(g5)

        # ── 6. set_icon_state external sync ───────────────────────────────
        g6 = _section("set_icon_state() — external sync without emitting toggled")
        v6 = QVBoxLayout(g6)
        v6.setSpacing(6)

        self._btn_synced = IconToolButton(
            icon="mdi:checkbox-blank-outline",
            checked_icon="mdi:checkbox-marked-outline",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Enable",
            checked_tooltip="Disable",
        )
        btn_drive = IconToolButton(icon="mdi:swap-horizontal", tooltip="Toggle via set_icon_state()")
        btn_drive.clicked.connect(self._toggle_synced)
        self._synced_state = False

        v6.addLayout(_row(self._btn_synced, btn_drive,
                          label="left driven by set_icon_state()"))
        root.addWidget(g6)

    def _toggle_synced(self):
        self._synced_state = not self._synced_state
        # Drive icon without emitting toggled
        self._btn_synced.blockSignals(True)
        self._btn_synced.setChecked(self._synced_state)
        self._btn_synced.blockSignals(False)
        self._btn_synced.set_icon_state(self._synced_state)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = IconToolButtonDemo()
    w.show()
    sys.exit(app.exec_())
