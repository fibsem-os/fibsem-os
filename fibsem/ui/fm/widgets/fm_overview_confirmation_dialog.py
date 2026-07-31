"""Pre-flight summary for an FM overview acquisition.

Replaces `OverviewConfirmationDialog`. Same job, house style: one title, a muted meta
line, count chips, a duration broken down by where it goes, and a primary action in
the footer.
"""

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    OverviewParameters,
    ZParameters,
)
from fibsem.fm.timing import estimate_tileset_acquisition_time
from fibsem.structures import TileOrderStrategy
from fibsem.ui import stylesheets

BACKGROUND = "#262930"
PANEL = "#1e2027"
BORDER = "#3d4251"
TEXT = "#d6d6d6"
TEXT_STRONG = "#f0f1f2"
TEXT_MUTED = "#868e93"
ACCENT = "#50a6ff"

CHIP_ACQUIRE = "#4caf50"
CHIP_SKIP = "#868e93"
CHIP_CHANNEL = "#50a6ff"


def format_duration(seconds: float) -> str:
    """`2m 14s`, `1h 03m`, `45s` — whichever reads best at that magnitude."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def _chip(text: str, color: str) -> QWidget:
    """Pill label with a coloured dot, matching the app's other summaries."""
    chip = QFrame()
    chip.setStyleSheet(
        f"QFrame {{ background: {PANEL}; border: 1px solid {BORDER};"
        f" border-radius: 10px; }}"
    )
    layout = QHBoxLayout(chip)
    layout.setContentsMargins(9, 3, 10, 3)
    layout.setSpacing(6)

    dot = QLabel("●")
    dot.setStyleSheet(f"color: {color}; font-size: 10px; border: none;")
    label = QLabel(text)
    label.setStyleSheet(f"color: {TEXT}; font-size: 11px; border: none;")
    layout.addWidget(dot)
    layout.addWidget(label)
    return chip


class FMOverviewConfirmationDialog(QDialog):
    """Confirm an overview before it runs, with what it will cost."""

    def __init__(
        self,
        parameters: OverviewParameters,
        channel_settings: List[ChannelSettings],
        zparams: Optional[ZParameters] = None,
        tile_fov: Optional[tuple] = None,
        autofocus_settings: Optional[AutoFocusSettings] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.parameters = parameters
        self.channel_settings = channel_settings
        self.zparams = zparams if parameters.use_zstack else None
        self.tile_fov = tile_fov
        self.autofocus_settings = autofocus_settings

        self.setWindowTitle("Start Overview Acquisition")
        self.setMinimumWidth(430)
        self.setStyleSheet(f"QDialog {{ background: {BACKGROUND}; }}")
        self._init_ui()

    # ── content ──────────────────────────────────────────────────────────

    def _estimate(self) -> dict:
        return estimate_tileset_acquisition_time(
            self.channel_settings,
            (self.parameters.rows, self.parameters.cols),
            self.zparams,
            self.parameters.autofocus_mode,
            tile_mask=self.parameters.tile_mask,
        )

    def _meta_line(self) -> str:
        p = self.parameters
        total = p.rows * p.cols
        bits = [f"{p.rows} × {p.cols} grid", f"{p.overlap:.0%} overlap"]
        if self.tile_fov is not None:
            fov_x, fov_y = self.tile_fov
            width = ((p.cols - 1) * (1 - p.overlap) + 1) * fov_x * constants.SI_TO_MICRO
            height = ((p.rows - 1) * (1 - p.overlap) + 1) * fov_y * constants.SI_TO_MICRO
            bits.append(f"{width:.0f} × {height:.0f} µm")
        bits.append(f"{p.tile_order.value} order")
        if p.n_enabled_tiles != total:
            bits.append("sparse")
        return " · ".join(bits)

    def _sweep_rows(self) -> List[tuple]:
        """How the sweep is configured, now that it is configurable.

        Reports only the enabled passes, and says so plainly when there are none --
        that combination schedules focusing that then does nothing, and it looks
        correct from either control alone.
        """
        if self.autofocus_settings is None:
            return []

        rows = [("Focus method", self.autofocus_settings.method.value.title())]
        if self.autofocus_settings.channel_name:
            rows.append(("Focus channel", self.autofocus_settings.channel_name))

        enabled = [p for p in self.autofocus_settings.passes if p.enabled]
        if not enabled:
            rows.append(("Sweep", "no passes enabled — nothing will be focused"))
        else:
            # `search_range` is the total span, not a half-width -- positions cover
            # +/- range/2 -- so it is reported as a span rather than as +/-.
            rows.append((
                "Sweep",
                "  ·  ".join(
                    f"{p.search_range * constants.SI_TO_MICRO:.0f} µm span, "
                    f"{p.step_size * constants.SI_TO_MICRO:.1f} µm steps "
                    f"({p.n_steps} images)"
                    for p in enabled
                ),
            ))
        return rows

    def _rows(self) -> List[tuple]:
        """Label/value pairs for the detail block."""
        p = self.parameters
        estimate = self._estimate()

        detail = [
            ("Channels", ", ".join(ch.name for ch in self.channel_settings) or "—"),
        ]
        if p.use_zstack and self.zparams is not None:
            detail.append(("Z-stack", f"{self.zparams.num_planes} planes per tile"))
        else:
            detail.append(("Z-stack", "off"))

        if p.autofocus_mode is AutoFocusMode.NONE:
            detail.append(("Auto-focus", "off"))
        else:
            mode = p.autofocus_mode.name.replace("_", " ").lower()
            note = ""
            if (p.autofocus_mode is AutoFocusMode.EACH_ROW
                    and p.tile_order is TileOrderStrategy.SPIRAL):
                # The runner promotes this; say so before it happens rather than
                # leaving the user to find it in the log afterwards.
                note = "  → per tile, for a spiral"
            detail.append(("Auto-focus", f"{mode}{note}"))
            detail.extend(self._sweep_rows())

        detail.append(("Images", f"{estimate['total_images']:,}"))
        detail.append((
            "Estimated time",
            f"{format_duration(estimate['total_time'])}"
            f"   ({format_duration(estimate['image_acquisition_time'])} imaging"
            f" · {format_duration(estimate['stage_movement_time'])} moving"
            + (f" · {format_duration(estimate['autofocus_time'])} focusing"
               if estimate["autofocus_time"] else "")
            + ")",
        ))
        return detail

    # ── layout ───────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        p = self.parameters
        total = p.rows * p.cols
        acquired = p.n_enabled_tiles

        title = QLabel("Start overview acquisition")
        title.setStyleSheet(
            f"color: {TEXT_STRONG}; font-size: 15px; font-weight: 600;"
        )
        meta = QLabel(self._meta_line())
        meta.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        meta.setWordWrap(True)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(_chip(f"{acquired} to acquire", CHIP_ACQUIRE))
        if acquired != total:
            chips.addWidget(_chip(f"{total - acquired} skipped", CHIP_SKIP))
        chips.addWidget(_chip(
            f"{len(self.channel_settings)} channel"
            f"{'s' if len(self.channel_settings) != 1 else ''}", CHIP_CHANNEL))
        chips.addStretch()

        detail = QFrame()
        detail.setStyleSheet(
            f"QFrame {{ background: {PANEL}; border: 1px solid {BORDER};"
            f" border-radius: 4px; }}"
        )
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(12, 10, 12, 10)
        detail_layout.setSpacing(6)
        for label_text, value_text in self._rows():
            row = QHBoxLayout()
            row.setSpacing(12)
            label = QLabel(label_text)
            label.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 11px; border: none;")
            label.setFixedWidth(96)
            value = QLabel(value_text)
            value.setStyleSheet(f"color: {TEXT}; font-size: 11px; border: none;")
            value.setWordWrap(True)
            row.addWidget(label, alignment=Qt.AlignTop)
            row.addWidget(value, stretch=1)
            detail_layout.addLayout(row)

        self.button_start = QPushButton("Start Acquisition")
        self.button_start.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_start.setMinimumHeight(30)
        self.button_start.clicked.connect(self.accept)
        button_cancel = QPushButton("Cancel")
        button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        button_cancel.setMinimumHeight(30)
        button_cancel.clicked.connect(self.reject)

        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(button_cancel)
        footer.addWidget(self.button_start)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(meta)
        layout.addLayout(chips)
        layout.addWidget(detail)
        layout.addLayout(footer)

        if acquired == 0:
            # Nothing to do: the runner would reject this anyway, so say why here
            # rather than letting it fail after the dialog is dismissed.
            self.button_start.setEnabled(False)
            self.button_start.setToolTip("No tiles are selected.")
