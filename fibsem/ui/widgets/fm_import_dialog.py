"""Import a fluorescence image from other software: confirm what the file is (FIB-1030).

A file from another microscope may not say which axis is channels and which is z, how
big a pixel is, or what its channels are. :func:`fibsem.fm.reader.read_source`
suggests what the file does say; this dialog shows the result on the FM canvas -- the
same compositing, z slider and channel controls as the FM tab, so what the user sees
is how the image will look -- and lets them correct it: the role of each axis, the
pixel size, each channel's name and colour, and whether the image is mirrored.

The dialog does not place anything. The host asks it for the confirmed answers and
builds the image with the placement it chooses (:meth:`ImportImageDialog.build`).
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.fm.composite import AVAILABLE_COLORS
from fibsem.fm.reader import (
    ROLE_NAMES,
    ROLES,
    ImportSource,
    build_image,
    channel_count,
    check_roles,
    default_channels,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.ui import stylesheets
from fibsem.ui.tokens import CANVAS_BG, ERROR_COLOR, TEXT_COLOR, WARN_COLOR
from fibsem.ui.widgets.canvas.overlay_controls import panel_hint, panel_section

# More channels than this and the channel axis is probably z.
MANY_CHANNELS = 6
# What an unknown pixel size starts at: a guess, said to be one.
GUESSED_PIXEL_SIZE_UM = 1.0
_ROW_HEIGHT = 28


class ImportImageDialog(QDialog):
    """Confirm how to read an image file; preview it as it will look."""

    def __init__(self, source: ImportSource, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Import {source.name}")
        self.setStyleSheet(f"background: {CANVAS_BG}; color: {TEXT_COLOR};")
        self.resize(1150, 720)
        self.source = source
        self._channel_rows: List[tuple] = []

        from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget

        self.preview = FMCanvasWidget()

        panel = QFrame()
        panel.setObjectName("canvasPanel")
        panel.setStyleSheet(stylesheets.CANVAS_PANEL_STYLE)
        panel.setFixedWidth(380)
        form = QVBoxLayout(panel)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        described = f"{source.data.dtype} · {source.described_by}"
        self.label_file = QLabel(f"{source.name}\n{described}")
        self.label_file.setWordWrap(True)
        form.addWidget(self.label_file)

        form.addWidget(panel_section("Axes"))
        axes_row = QHBoxLayout()
        axes_row.setSpacing(4)
        self.combo_roles: List[QComboBox] = []
        for size, role in zip(source.data.shape, source.roles):
            column = QVBoxLayout()
            column.setSpacing(2)
            combo = QComboBox()
            for r in ROLES:
                combo.addItem(ROLE_NAMES[r], r)
            if role in ROLES:
                combo.setCurrentIndex(ROLES.index(role))
            else:
                combo.setCurrentIndex(-1)
            combo.currentIndexChanged.connect(self._on_roles_changed)
            self.combo_roles.append(combo)
            size_label = QLabel(str(size))
            size_label.setAlignment(Qt.AlignCenter)
            column.addWidget(combo)
            column.addWidget(size_label)
            axes_row.addLayout(column)
        form.addLayout(axes_row)
        self.btn_swap = QPushButton("Swap channel and z")
        self.btn_swap.clicked.connect(self.swap_channel_and_z)
        form.addWidget(self.btn_swap)
        self.label_axes = panel_hint()
        self.label_axes.setWordWrap(True)
        form.addWidget(self.label_axes)

        form.addWidget(panel_section("Pixel size"))
        size_row = QFormLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setDecimals(4)
        self.spin_pixel_size.setRange(0.0001, 1000.0)
        self.spin_pixel_size.setSuffix(" um")
        known = source.pixel_size is not None
        self.spin_pixel_size.setValue(
            source.pixel_size * constants.SI_TO_MICRO
            if known
            else GUESSED_PIXEL_SIZE_UM
        )
        size_row.addRow("Per pixel", self.spin_pixel_size)
        form.addLayout(size_row)
        self.label_pixel_size = panel_hint(
            "From the file."
            if known
            else "Not in the file: a guess. Enter it if you know it; otherwise "
            "Fit from points finds the scale (leave Lock scale off)."
        )
        self.label_pixel_size.setWordWrap(True)
        form.addWidget(self.label_pixel_size)

        form.addWidget(panel_section("Channels"))
        # Scrolled: read the wrong way round, a z-stack is dozens of channels.
        channels = QWidget()
        channels.setStyleSheet("background: transparent;")
        self._channels_widget = channels
        self.channels_box = QVBoxLayout(channels)
        self.channels_box.setContentsMargins(0, 0, 0, 0)
        self.channels_box.setSpacing(4)
        self.channels_box.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(channels)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll.setMinimumHeight(90)
        form.addWidget(scroll, 1)

        form.addWidget(panel_section("Orientation"))
        self.check_flip = QCheckBox("Mirror left to right")
        self.check_flip.setToolTip(
            "For an image recorded mirrored: a fit from points can turn an image, "
            "never mirror it"
        )
        self.check_flip.toggled.connect(self._refresh_preview)
        form.addWidget(self.check_flip)

        body = QHBoxLayout()
        body.addWidget(self.preview, 1)
        body.addWidget(panel)

        footer = QHBoxLayout()
        note = QLabel(
            "A corrected copy is saved in the grid's Aligned Images folder. It starts "
            "at the centre of the view; place it with Fit from points."
        )
        note.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        note.setWordWrap(True)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_import = QPushButton("Import")
        self.btn_import.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_import.clicked.connect(self.accept)
        footer.addWidget(note, 1)
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_import)

        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addLayout(footer)

        # The preview's own channel controls can recolour a channel too; what the
        # user picks there is what gets saved, so the form follows it.
        self.preview._panel.changed.connect(self._take_colours_from_preview)

        self._rebuild_channel_rows(source.channel_names, source.channel_colors)
        self._refresh_preview()

    # ── the answers ───────────────────────────────────────────────────────

    @property
    def roles(self) -> str:
        return "".join(combo.currentData() or "?" for combo in self.combo_roles)

    @property
    def pixel_size(self) -> float:
        """Metres."""
        return self.spin_pixel_size.value() * constants.MICRO_TO_SI

    @property
    def flip(self) -> bool:
        return self.check_flip.isChecked()

    @property
    def channel_names(self) -> List[str]:
        return [
            edit.text().strip() or edit.placeholderText()
            for edit, _ in self._channel_rows
        ]

    @property
    def channel_colors(self) -> List[str]:
        return [combo.currentText() for _, combo in self._channel_rows]

    def problem(self) -> Optional[str]:
        """Why the answers cannot be used as they stand, or None."""
        try:
            check_roles(self.roles, self.source.data.shape)
        except ValueError as e:
            return str(e)[:1].upper() + str(e)[1:] + "."
        return None

    def build(self, geometry=None, stage_position=None) -> FluorescenceImage:
        """The image as confirmed, placed as the host says."""
        return build_image(
            self.source,
            self.roles,
            self.pixel_size,
            self.channel_names,
            self.channel_colors,
            flip=self.flip,
            geometry=geometry,
            stage_position=stage_position,
        )

    def swap_channel_and_z(self) -> None:
        roles = self.roles
        if "C" not in roles or "Z" not in roles:
            return
        c, z = roles.index("C"), roles.index("Z")
        for combo in self.combo_roles:
            combo.blockSignals(True)
        self.combo_roles[c].setCurrentIndex(ROLES.index("Z"))
        self.combo_roles[z].setCurrentIndex(ROLES.index("C"))
        for combo in self.combo_roles:
            combo.blockSignals(False)
        self._on_roles_changed()

    # ── keeping the form and the preview in step ─────────────────────────────

    def _on_roles_changed(self, *_args) -> None:
        if self.problem() is None:
            count = channel_count(self.source.data.shape, self.roles)
            if count != len(self._channel_rows):
                # The channels are a different axis now: what the file said about
                # the old ones does not describe these.
                self._rebuild_channel_rows(*default_channels(count))
        self._refresh_preview()

    def _take_colours_from_preview(self) -> None:
        colours = {layer.name: layer.color for layer in self.preview.layers}
        for (_edit, combo), name in zip(self._channel_rows, self.channel_names):
            colour = colours.get(name)
            if colour and colour != combo.currentText():
                combo.blockSignals(True)
                if combo.findText(colour) < 0:
                    combo.addItem(colour)
                combo.setCurrentText(colour)
                combo.blockSignals(False)

    def _rebuild_channel_rows(self, names, colours) -> None:
        while self.channels_box.count() > 1:  # the stretch stays last
            item = self.channels_box.takeAt(0)
            if item.widget() is not None:
                # Off the screen now, not when the deferred delete comes round.
                item.widget().hide()
                item.widget().deleteLater()
        self._channel_rows = []
        for name, colour in zip(names, colours):
            row = QWidget()
            row.setFixedHeight(_ROW_HEIGHT)  # a fixed row, so many of them scroll
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(4)
            combo = QComboBox()
            combo.addItems(AVAILABLE_COLORS)
            if colour not in AVAILABLE_COLORS:
                combo.addItem(colour)
            combo.setCurrentText(colour)
            combo.setFixedWidth(90)
            combo.currentIndexChanged.connect(self._refresh_preview)
            edit = QLineEdit(name)
            edit.setPlaceholderText(name)
            edit.editingFinished.connect(self._refresh_preview)
            line.addWidget(combo)
            line.addWidget(edit, 1)
            self.channels_box.insertWidget(self.channels_box.count() - 1, row)
            self._channel_rows.append((edit, combo))
        # Tall enough for every row, so the scroll area scrolls them rather than
        # squeezing them into the space it has.
        spacing = self.channels_box.spacing()
        self._channels_widget.setMinimumHeight(
            len(self._channel_rows) * (_ROW_HEIGHT + spacing)
        )

    def _refresh_preview(self, *_args) -> None:
        problem = self.problem()
        roles = self.roles
        self.btn_swap.setEnabled("C" in roles and "Z" in roles)
        self.btn_import.setEnabled(problem is None)
        if problem is not None:
            self._say(problem, ERROR_COLOR)
            return
        count = channel_count(self.source.data.shape, roles)
        if count > MANY_CHANNELS:
            self._say(
                f"{count} channels. If these are z-slices, swap channel and z.",
                WARN_COLOR,
            )
        else:
            self._say("", None)
        try:
            image = self.build()
        except Exception as e:  # noqa: BLE001 - said in the form; raising in a slot aborts PyQt5
            self._say(f"Cannot read the image this way: {e}", ERROR_COLOR)
            self.btn_import.setEnabled(False)
            return
        self.preview.set_fm_image(image)

    def _say(self, text: str, colour: Optional[str]) -> None:
        self.label_axes.setText(text)
        self.label_axes.setStyleSheet(f"color: {colour};" if colour else "")
