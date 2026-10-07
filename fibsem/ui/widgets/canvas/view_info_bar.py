"""The bar under a view: what the image on it is, in the export bar's words (FIB-1186).

One row under each SEM, FIB and FM view, replacing the title row that used to sit
above it. It names the view, then reads the displayed image's metadata through
:func:`fibsem.imaging.export.image_fields`, so it says exactly what an export of the
same image would: the detector or objective unlabelled, then up to five short
labelled values, then when the image was taken. It describes the image as acquired,
not the microscope now -- the controls beside the view already show that.

A Qt child of the panel, below the canvas rather than drawn on it, so it can neither
cover the image nor collide with the scalebar, and costs nothing to repaint when the
canvas does. When the row is too narrow, whole fields drop from the right behind a
``+N`` chip that lists them on hover; a value is never cut off half way.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QWidget,
)

from fibsem.imaging.export import (
    FIELD_TITLES,
    MAX_FIELDS,
    ExportField,
    ImageFields,
)
from fibsem.ui.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    NUMBER_FONT,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

BAR_HEIGHT = 26
# A value reads a shade brighter than body text against the canvas background, as the
# canvas's own readouts do; the header (detector, objective) a shade dimmer.
_VALUE_COLOR = "#e8e8e8"
_HEADER_COLOR = "#aeb4b8"
_DIVIDER_COLOR = "#2c3038"
_FIELD_SPACING = 12

_BAR_STYLE = f"#viewInfoBar {{ background: {CANVAS_BG}; border-top: 1px solid {_DIVIDER_COLOR}; }}"
# Every part transparent over the bar: the app stylesheet gives a bare QLabel a
# background of its own, which drew a lighter box behind each one.
_CLEAR = "background: transparent;"
_KIND_STYLE = f"color: {TEXT_STRONG_COLOR}; font-size: 12px; font-weight: 700; {_CLEAR}"
_HEADER_STYLE = f"color: {_HEADER_COLOR}; font-size: 11px; {_CLEAR}"
_FIELD_STYLE = f"font-family: {NUMBER_FONT}; font-size: 11px; {_CLEAR}"
_TIME_STYLE = (
    f"color: {TEXT_MUTED_COLOR}; font-family: {NUMBER_FONT}; font-size: 11px; {_CLEAR}"
)
_CHIP_STYLE = (
    f"color: {_HEADER_COLOR}; font-size: 10px; padding: 0px 5px; {_CLEAR}"
    f" border: 1px solid {BORDER_COLOR}; border-radius: 7px;"
)
_TIME_TITLE = "Acquired at"


# What a new bar shows. The export's defaults less the pixel size: under a live view,
# HFW says the same thing more usefully, and the row has room for one fewer field
# than an exported figure. The pixel size is a tick away in the picker.
BAR_DEFAULT_FIELDS = ("detector", "objective", "hfw", "voltage", "current", "z")


def field_title(item: ExportField) -> str:
    """What a field's label stands for: `Horizontal field width` for `HFW`."""
    return FIELD_TITLES.get(item.key, item.name)


def _field_html(item: ExportField) -> str:
    return (
        f'<span style="color: {TEXT_MUTED_COLOR}">{item.label}</span>&nbsp;'
        f'<span style="color: {_VALUE_COLOR}">{item.value}</span>'
    )


class ViewInfoBar(QWidget):
    """The image's metadata, one row, under its view."""

    def __init__(self, kind: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("viewInfoBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(_BAR_STYLE)
        self.setFixedHeight(BAR_HEIGHT)
        # The bar must never be what decides how wide a view can get.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self._keys: Tuple[str, ...] = BAR_DEFAULT_FIELDS
        self._info: Optional[ImageFields] = None
        self._overrides: Dict[str, str] = {}
        self._shown: Optional[Tuple] = None  # what the labels were last built from
        self._labelled: List[ExportField] = []
        self._label_keys: List[str] = []  # the fields the labels were built for
        self._time: Optional[str] = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 6, 0)
        lay.setSpacing(10)
        # Nor what decides its own: fields drop to fit, so its content's width is not
        # a minimum (the default would make it one whenever the bar is a window).
        lay.setSizeConstraint(QLayout.SetNoConstraint)

        self.kind_label = QLabel(kind)
        self.kind_label.setStyleSheet(_KIND_STYLE)
        lay.addWidget(self.kind_label)

        self.header_label = QLabel()
        self.header_label.setStyleSheet(_HEADER_STYLE)
        lay.addWidget(self.header_label)

        self.divider = QFrame()
        self.divider.setFixedSize(1, 12)
        self.divider.setStyleSheet(f"background: {BORDER_COLOR};")
        lay.addWidget(self.divider)

        self._fields_box = QWidget()
        self._fields_box.setStyleSheet(_CLEAR)
        self._fields_layout = QHBoxLayout(self._fields_box)
        self._fields_layout.setContentsMargins(0, 0, 0, 0)
        self._fields_layout.setSpacing(_FIELD_SPACING)
        lay.addWidget(self._fields_box)
        self.field_labels: List[QLabel] = []

        self.more_chip = QLabel()
        self.more_chip.setStyleSheet(_CHIP_STYLE)
        lay.addWidget(self.more_chip)

        lay.addStretch(1)

        self.time_label = QLabel()
        self.time_label.setStyleSheet(_TIME_STYLE)
        self.time_label.setToolTip(_TIME_TITLE)
        lay.addWidget(self.time_label)

        self._rebuild()

    # ── what to show ───────────────────────────────────────────────────────
    @property
    def kind(self) -> str:
        return self.kind_label.text()

    def set_image_fields(self, info: Optional[ImageFields]) -> None:
        """Show what *info* says about the image now on the view; None clears it.

        Cheap to call on every frame of a live view: the labels are rebuilt only when
        what they would say has changed.
        """
        self._info = info
        self._overrides = {}
        self._rebuild()

    def set_field_value(self, key: str, value: Optional[str]) -> None:
        """Replace one field's value while the image stays: the FM's Z as it scrubs."""
        if value is None:
            self._overrides.pop(key, None)
        else:
            self._overrides[key] = value
        self._rebuild()

    def set_field_keys(self, keys: Sequence[str]) -> None:
        """Which fields to show, in order. Unlabelled ones (the detector, the
        objective) go in the header; at most :data:`MAX_FIELDS` labelled ones follow."""
        self._keys = tuple(keys)
        self._rebuild()

    def field_keys(self) -> Tuple[str, ...]:
        return self._keys

    def clear(self) -> None:
        self.set_image_fields(None)

    def visible_fields(self) -> List[ExportField]:
        """The labelled fields the bar has room for, in order."""
        return [
            item
            for item, label in zip(self._labelled, self.field_labels)
            if not label.isHidden()
        ]

    def hidden_fields(self) -> List[ExportField]:
        """The labelled fields dropped for want of room, behind the ``+N`` chip."""
        return [
            item
            for item, label in zip(self._labelled, self.field_labels)
            if label.isHidden()
        ]

    # ── building ───────────────────────────────────────────────────────────
    def _selected(self) -> Tuple[List[ExportField], List[ExportField], Optional[str]]:
        """(header fields, labelled fields, time) for the current image and keys."""
        if self._info is None:
            return [], [], None
        by_key = {item.key: item for item in self._info.fields}
        header: List[ExportField] = []
        labelled: List[ExportField] = []
        for key in self._keys:
            item = by_key.get(key)
            if item is None:
                continue
            if key in self._overrides:
                item = ExportField(
                    item.key, item.name, item.label, self._overrides[key]
                )
            if item.label:
                if len(labelled) < MAX_FIELDS:
                    labelled.append(item)
            else:
                header.append(item)
        time = next((p.value for p in self._info.provenance if p.key == "date"), None)
        return header, labelled, time

    def _rebuild(self) -> None:
        header, labelled, time = self._selected()
        signature = (
            tuple((f.key, f.value) for f in header),
            tuple((f.key, f.value) for f in labelled),
            time,
        )
        if signature == self._shown:
            return
        self._shown = signature
        self._labelled = labelled

        self.header_label.setText(" · ".join(f.value for f in header))
        self.header_label.setToolTip(
            "\n".join(f"{field_title(f)}: {f.value}" for f in header)
        )
        self.header_label.setVisible(bool(header))
        self.divider.setVisible(bool(labelled))

        same_fields = [f.key for f in labelled] == self._label_keys
        if not same_fields:
            for label in self.field_labels:
                self._fields_layout.removeWidget(label)
                # Hidden now: deleteLater waits for the event loop, and until then
                # the old label still paints under its replacement.
                label.hide()
                label.deleteLater()
            self.field_labels = []
            for _ in labelled:
                label = QLabel()
                label.setStyleSheet(_FIELD_STYLE)
                label.setTextFormat(Qt.RichText)
                self._fields_layout.addWidget(label)
                self.field_labels.append(label)
            self._label_keys = [f.key for f in labelled]
        # The same fields with new values -- a z step, the next live frame -- keep
        # their labels and only change what they say.
        for item, label in zip(labelled, self.field_labels):
            label.setText(_field_html(item))
            label.setToolTip(f"{field_title(item)}: {item.value}")

        self._time = time
        self.time_label.setText(time or "")
        self._fit()

    def _fit(self) -> None:
        """Make the row fit: drop the time first, then whole fields from the right.

        Everything dropped is counted in a ``+N`` chip and listed on its hover, so
        nothing leaves the bar unannounced.
        """
        labels = self.field_labels
        for label in labels:
            label.setVisible(True)
        self.time_label.setVisible(bool(self._time))
        self.more_chip.setVisible(False)

        widths = [label.sizeHint().width() for label in labels]
        total = sum(widths) + _FIELD_SPACING * max(len(widths) - 1, 0)
        if total <= self._room(with_time=bool(self._time)):
            return

        dropped_time = bool(self._time)
        self.time_label.setVisible(False)
        room = self._room(with_time=False)
        chip = self._chip_width(len(labels) + 1) + self.layout().spacing()
        used, keep = 0, 0
        for width in widths:
            step = width + (_FIELD_SPACING if keep else 0)
            if used + step + chip > room:
                break
            used += step
            keep += 1
        for label in labels[keep:]:
            label.setVisible(False)

        lines = [f"{field_title(f)}: {f.value}" for f in self._labelled[keep:]]
        if dropped_time:
            lines.append(f"{_TIME_TITLE}: {self._time}")
        if not lines:
            return
        self.more_chip.setText(f"+{len(lines)}")
        self.more_chip.setToolTip("\n".join(lines))
        self.more_chip.setVisible(True)

    def _room(self, with_time: bool) -> int:
        """The width left for the fields once the rest of the row has its share."""
        lay = self.layout()
        margins = lay.contentsMargins()
        shown = [
            w
            for w in (self.kind_label, self.header_label, self.divider)
            if not w.isHidden()
        ]
        if with_time:
            shown.append(self.time_label)
        fixed = margins.left() + margins.right()
        fixed += sum(w.sizeHint().width() for w in shown)
        # Spacing between every shown item in the outer row, the fields box included.
        fixed += lay.spacing() * len(shown)
        return self.width() - fixed

    def _chip_width(self, count: int) -> int:
        self.more_chip.setText(f"+{count}")
        return self.more_chip.sizeHint().width()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()
