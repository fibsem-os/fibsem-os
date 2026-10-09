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

The button at its end opens :class:`FieldPicker`, a checklist of what the bar shows.
The choice is a display preference per kind of view (``display.info_bar_fields``):
every bar of that kind follows it, now and after a restart.

The quad view's fourth cell has one too, with no image: the stage's position, pushed
by the microscope, under the chamber drawing (see ``MicroscopeViewController``).
"""

from __future__ import annotations

import logging
import weakref
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5 import sip
from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import config as cfg
from fibsem.imaging.export import (
    FIELD_CATALOGUE,
    FIELD_TITLES,
    MAX_FIELDS,
    ExportField,
    ImageFields,
    field_keys_for,
    z_value,
)
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    CAPTION_STYLE,
    NUMBER_FONT,
    PANEL_COLOR,
    PANEL_TITLE_STYLE,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

_logger = logging.getLogger(__name__)

BAR_HEIGHT = 26
# A value reads a shade brighter than body text against the canvas background, as the
# canvas's own readouts do; the header (detector, objective) a shade dimmer.
_VALUE_COLOR = "#e8e8e8"
_HEADER_COLOR = "#aeb4b8"
_DIVIDER_COLOR = "#2c3038"
# Between fields, with each field's own 3 px padding either side making up the 12 px.
_FIELD_SPACING = 6

_BAR_STYLE = f"#viewInfoBar {{ background: {CANVAS_BG}; border-top: 1px solid {_DIVIDER_COLOR}; }}"
# Every part transparent over the bar: the app stylesheet gives a bare QLabel a
# background of its own, which drew a lighter box behind each one.
_CLEAR = "background: transparent;"
_KIND_STYLE = f"color: {TEXT_STRONG_COLOR}; font-size: 12px; font-weight: 700; {_CLEAR}"
_HEADER_STYLE = f"color: {_HEADER_COLOR}; font-size: 11px; {_CLEAR}"
# The padding and radius are there whether or not the field is highlighted, so a
# highlight never changes a field's width and nothing beside it moves.
_FIELD_STYLE = (
    f"font-family: {NUMBER_FONT}; font-size: 11px; padding: 0px 3px;"
    f" border-radius: 3px; {_CLEAR}"
)
# A field whose value just changed: an accent tint behind it, fading out (FIB-1188).
_HIGHLIGHT_RGB = "80, 166, 255"  # ACCENT_COLOR
_HIGHLIGHT_ALPHA = 0.22
_HIGHLIGHT_VALUE_COLOR = "#8cc4ff"
_HIGHLIGHT_MS = 1500
_HIGHLIGHT_TICK_MS = 100
_TIME_STYLE = (
    f"color: {TEXT_MUTED_COLOR}; font-family: {NUMBER_FONT}; font-size: 11px; {_CLEAR}"
)
_CHIP_STYLE = (
    f"color: {_HEADER_COLOR}; font-size: 10px; padding: 0px 5px; {_CLEAR}"
    f" border: 1px solid {BORDER_COLOR}; border-radius: 7px;"
)
_TIME_TITLE = "Acquired at"
_TIME_KEY = "date"  # the provenance field the time comes from
_PICKER_STYLE = (
    f"#fieldPicker {{ background: {PANEL_COLOR}; border: 1px solid {BORDER_COLOR};"
    " border-radius: 4px; }"
    f" QCheckBox {{ color: {TEXT_COLOR}; font-size: 12px; background: transparent; }}"
    " QLabel { background: transparent; }"
)


# What a new bar shows. The export's defaults less the pixel size: under a live view,
# HFW says the same thing more usefully. Plus the working distance: every frame of a
# live view records it, so the bar follows focusing -- at the vendor's console too,
# which nothing announces, so a truly live readout would have to poll. Last, the time
# the image was taken. Anything else is a tick away in the picker.
BAR_DEFAULT_FIELDS = (
    "detector",
    "objective",
    "hfw",
    "voltage",
    "current",
    "working_distance",
    "z",
    _TIME_KEY,
)

# Every open bar, so a choice made on one reaches the others of its kind at once: the
# quad view and the lamella editor each have an SEM bar, and both should follow.
_BARS: "weakref.WeakSet[ViewInfoBar]" = weakref.WeakSet()


def stored_field_keys(kind: str) -> Optional[Tuple[str, ...]]:
    """The fields someone chose for *kind*'s bar, or None to take the defaults."""
    try:
        chosen = cfg.load_user_preferences().display.info_bar_fields.get(kind)
    except Exception:
        _logger.warning("could not read the view bar's fields", exc_info=True)
        return None
    return tuple(chosen) if chosen is not None else None


def choose_field_keys(kind: str, keys: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Save *keys* as *kind*'s fields -- None goes back to the defaults -- and show
    them on every open bar of that kind. Returns the keys now in effect."""

    def change(preferences) -> None:
        chosen = preferences.display.info_bar_fields
        if keys is None:
            chosen.pop(kind, None)
        else:
            chosen[kind] = list(keys)

    try:
        cfg.update_user_preferences(change)
    except Exception:
        _logger.warning("could not save the view bar's fields", exc_info=True)
    effective = BAR_DEFAULT_FIELDS if keys is None else tuple(keys)
    for bar in list(_BARS):
        # A bar whose window has closed can outlive its widget in this set; touching
        # it raises inside the picker's slot, which PyQt5 turns into an abort.
        if sip.isdeleted(bar) or bar.kind != kind:
            continue
        bar.set_field_keys(effective)
    return effective


def show_fm_plane(bar: "ViewInfoBar", fm_widget, stack: Optional[Tuple[int, float]]):
    """Make an FM bar's Z name the plane on screen: the projection, or `11 of 21`.

    *stack* is :func:`fibsem.imaging.export.z_stack` of the image shown; a single
    plane has no Z to follow.
    """
    if stack is None:
        return
    plane = None if fm_widget.max_projection else fm_widget.current_z
    bar.set_field_value("z", z_value(*stack, plane=plane))


def field_title(item: ExportField) -> str:
    """What a field's label stands for: `Horizontal field width` for `HFW`."""
    return FIELD_TITLES.get(item.key, item.name)


def _field_html(item: ExportField, value_color: str = _VALUE_COLOR) -> str:
    return (
        f'<span style="color: {TEXT_MUTED_COLOR}">{item.label}</span>&nbsp;'
        f'<span style="color: {value_color}">{item.value}</span>'
    )


class ViewInfoBar(QWidget):
    """The image's metadata, one row, under its view.

    *title* takes the bold view name's place: the fourth cell's page selector, which
    names its view as the others' labels do. A bar that is not *choosable* has no
    field button, for one whose fields are not the image's.

    With *highlight_changes*, a field whose value changes lights up and fades, so the
    next frame -- or a value the microscope pushed -- is seen catching up (FIB-1188).
    For a live view only: a bar whose image is chosen, in the lamella editor or the
    image viewer, would light up every field that differs from the last one picked.
    """

    def __init__(
        self,
        kind: str,
        parent: Optional[QWidget] = None,
        *,
        title: Optional[QWidget] = None,
        choosable: bool = True,
        highlight_changes: bool = False,
    ) -> None:
        super().__init__(parent)
        self._kind = kind
        self._choosable = choosable
        self._highlight_changes = highlight_changes
        self._highlights: Dict[str, float] = {}  # field key -> strength left, 1 to 0
        self._quiet = False  # the next rebuild is the user's doing, not news
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setInterval(_HIGHLIGHT_TICK_MS)
        self._highlight_timer.timeout.connect(self._fade_highlights)
        self.setObjectName("viewInfoBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(_BAR_STYLE)
        self.setFixedHeight(BAR_HEIGHT)
        # The bar must never be what decides how wide a view can get.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self._keys: Tuple[str, ...] = stored_field_keys(kind) or BAR_DEFAULT_FIELDS
        self._info: Optional[ImageFields] = None
        self._overrides: Dict[str, str] = {}
        # Values that come from the microscope, not the image: they outlive an image
        # change and a clear, and sit after the image's fields.
        self._live: Dict[str, ExportField] = {}
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
        self.title = title if title is not None else self.kind_label
        lay.addWidget(self.title)

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

        self.fields_button = QToolButton()
        self.fields_button.setIcon(
            fibsem_icon("mdi:tune-variant", color=TEXT_MUTED_COLOR)
        )
        self.fields_button.setAutoRaise(True)
        self.fields_button.setFixedSize(20, 20)
        self.fields_button.setToolTip("Choose fields")
        self.fields_button.setAccessibleName(f"Choose the {kind} bar's fields")
        self.fields_button.setStyleSheet(f"QToolButton {{ {_CLEAR} border: none; }}")
        self.fields_button.clicked.connect(self.open_picker)
        lay.addWidget(self.fields_button)
        self.fields_button.setVisible(choosable)
        self._picker: Optional["FieldPicker"] = None

        _BARS.add(self)
        self._rebuild()

    # ── what to show ───────────────────────────────────────────────────────
    @property
    def kind(self) -> str:
        return self._kind

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
        # Scrubbing the planes is the user's own doing: nothing to point out.
        self._quiet = True
        try:
            self._rebuild()
        finally:
            self._quiet = False

    def set_live_field(
        self, key: str, label: str, value: Optional[str], name: str
    ) -> None:
        """Show a value the microscope pushed, rather than one the image recorded:
        the FM objective's position now, the stage's. None removes it.

        Kept apart from the image's fields so that a new image, or a cleared view,
        leaves it be; *name* says on hover that it is the present value. With no
        *label* it goes in the header, as an image's detector does.
        """
        if value is None:
            self._live.pop(key, None)
        else:
            self._live[key] = ExportField(key=key, name=name, label=label, value=value)
        self._rebuild()

    def set_field_keys(self, keys: Sequence[str]) -> None:
        """Which fields to show, in order. Unlabelled ones (the detector, the
        objective) go in the header; at most :data:`MAX_FIELDS` labelled ones follow."""
        self._keys = tuple(keys)
        self._rebuild()

    def field_keys(self) -> Tuple[str, ...]:
        return self._keys

    def set_fields_button_visible(self, visible: bool) -> None:
        """Show the field button: the quad view shows it on the selected view only,
        as it does the canvas toolbar. Shown by default, for views without selection."""
        self.fields_button.setVisible(visible and self._choosable)
        self._fit()

    def open_picker(self) -> "FieldPicker":
        """Open the checklist of this bar's fields, just above the button.

        One picker per bar, kept and reopened: closing a popup must not leave the bar
        holding a deleted widget.
        """
        if self._picker is None:
            self._picker = FieldPicker(self.kind, self._keys, parent=self)
            self._picker.changed.connect(
                lambda keys: choose_field_keys(self.kind, keys)
            )
            self._picker.reset.connect(self._on_picker_reset)
        self._picker.set_keys(self._keys)
        self._picker.adjustSize()
        corner = self.fields_button.mapToGlobal(QPoint(self.fields_button.width(), 0))
        self._picker.move(corner - QPoint(self._picker.width(), self._picker.height()))
        self._picker.show()
        return self._picker

    def _on_picker_reset(self) -> None:
        keys = choose_field_keys(self.kind, None)
        if self._picker is not None:
            self._picker.set_keys(keys)

    def clear(self) -> None:
        """Forget the image. Live values stay: they did not come from it."""
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
        """(header fields, labelled fields, time) for the current image and keys,
        with the live values after the image's fields."""
        live = [f for f in self._live.values() if f.label]
        live_header = [f for f in self._live.values() if not f.label]
        if self._info is None:
            return live_header, live, None
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
        time = None
        if _TIME_KEY in self._keys:
            time = next(
                (p.value for p in self._info.provenance if p.key == _TIME_KEY), None
            )
        return header + live_header, labelled + live, time

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
        before = {f.key: f.value for f in self._labelled}
        self._labelled = labelled
        if self._highlight_changes and not self._quiet:
            # A value that changed, not one that arrived: a view's first image, or a
            # field just ticked on, has nothing to compare with.
            changed = [
                f.key for f in labelled if f.key in before and before[f.key] != f.value
            ]
            for key in changed:
                self._highlights[key] = 1.0
            if changed:
                self._highlight_timer.start()

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
            label.setToolTip(f"{field_title(item)}: {item.value}")
        self._paint_fields()

        self._time = time
        self.time_label.setText(time or "")
        self._fit()

    def _paint_fields(self) -> None:
        """Each field's text, with the highlight it has left, if any."""
        for item, label in zip(self._labelled, self.field_labels):
            strength = self._highlights.get(item.key, 0.0)
            if strength > 0:
                alpha = _HIGHLIGHT_ALPHA * strength
                label.setStyleSheet(
                    f"{_FIELD_STYLE} background: rgba({_HIGHLIGHT_RGB}, {alpha:.3f});"
                )
                label.setText(_field_html(item, _HIGHLIGHT_VALUE_COLOR))
            else:
                label.setStyleSheet(_FIELD_STYLE)
                label.setText(_field_html(item))

    def _fade_highlights(self) -> None:
        step = _HIGHLIGHT_TICK_MS / _HIGHLIGHT_MS
        self._highlights = {
            key: left - step
            for key, left in self._highlights.items()
            if left - step > 1e-6
        }
        if not self._highlights:
            self._highlight_timer.stop()
        self._paint_fields()

    def highlighted_fields(self) -> List[str]:
        """The keys of the fields lit up now."""
        return list(self._highlights)

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
            for w in (
                self.title,
                self.header_label,
                self.divider,
                self.fields_button,
            )
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


class FieldPicker(QFrame):
    """The checklist behind a bar's button: which fields it shows.

    Lists every field the kind of image can record, by the names the export dialog
    uses, then the time. Each tick saves at once; Defaults forgets the choice.
    """

    changed = pyqtSignal(object)  # the chosen keys, in display order
    reset = pyqtSignal()

    def __init__(
        self, kind: str, keys: Sequence[str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent, Qt.Popup)
        self.setObjectName("fieldPicker")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(_PICKER_STYLE)
        self._kind = kind

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        title = QLabel(f"{kind} bar")
        title.setStyleSheet(PANEL_TITLE_STYLE)
        lay.addWidget(title)
        caption = QLabel(f"Up to {MAX_FIELDS} values")
        caption.setStyleSheet(CAPTION_STYLE)
        lay.addWidget(caption)

        self.checkboxes: Dict[str, QCheckBox] = {}
        for key in field_keys_for(kind) + (_TIME_KEY,):
            name = _TIME_TITLE if key == _TIME_KEY else FIELD_CATALOGUE[key][0]
            label = FIELD_CATALOGUE[key][1] if key != _TIME_KEY else ""
            checkbox = QCheckBox(
                f"{name} ({label})" if label and label != name else name
            )
            checkbox.toggled.connect(self._on_toggled)
            self.checkboxes[key] = checkbox
            lay.addWidget(checkbox)

        self.defaults_button = QPushButton("Defaults")
        self.defaults_button.setToolTip("Show this bar's default fields")
        self.defaults_button.clicked.connect(self.reset)
        lay.addWidget(self.defaults_button, 0, Qt.AlignLeft)
        self.set_keys(keys)

    def set_keys(self, keys: Sequence[str]) -> None:
        """Tick *keys* without announcing it."""
        for key, checkbox in self.checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(key in keys)
            checkbox.blockSignals(False)

    def keys(self) -> Tuple[str, ...]:
        return tuple(k for k, cb in self.checkboxes.items() if cb.isChecked())

    def _on_toggled(self, _checked: bool) -> None:
        self.changed.emit(self.keys())
