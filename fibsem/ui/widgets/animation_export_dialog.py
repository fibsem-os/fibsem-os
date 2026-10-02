"""Export what a workflow did as an animated GIF: a frame per task.

The preview plays the frames ``fibsem.imaging.animation.render_animation`` draws, and
Save writes those same frames, so what plays here is what is saved.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.imaging.animation import (
    AnimationFrame,
    AnimationOptions,
    default_animation_name,
    frame_durations,
    included,
    render_animation,
    save_animation,
    webp_supported,
)
from fibsem.ui import notification_service
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import PRIMARY_BUTTON_STYLESHEET
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    CONTROL_STYLE,
    PANEL_COLOR,
    PANEL_TITLE_STYLE,
    SURFACE_COLOR,
    TEXT_COLOR,
)
from fibsem.ui.widgets.image_export_dialog import (
    _caption,
    _rgb_to_qimage,
    _row,
    _Segmented,
    _tab,
)

_BEAMS = (("SEM", "SEM"), ("FIB", "FIB"), ("Both", "both"))
_MAGNIFICATIONS = (("High mag", "high"), ("Low mag", "low"))
_FRAME_TIMES = (("0.6 s", 600), ("1.2 s", 1200), ("2 s", 2000))
_WIDTHS = (("768", 768), ("1024", 1024), ("1536", 1536))
_FORMATS = (("GIF", ".gif"), ("WebP", ".webp"))
_HOLD_MS = 2500
_THUMB_W, _THUMB_H = 96, 64


def _index(choices, value) -> int:
    return [v for _, v in choices].index(value)


class _FrameTile(QWidget):
    """A task in the strip: its image, and a checkbox to leave it out."""

    def __init__(self, frame: AnimationFrame, on_click, on_toggle) -> None:
        super().__init__()
        self.setFixedWidth(_THUMB_W + 4)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.thumb = QLabel()
        self.thumb.setFixedSize(_THUMB_W + 4, _THUMB_H + 4)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thumb.mousePressEvent = lambda _event: on_click()
        self.checkbox = QCheckBox(frame.title)
        self.checkbox.setChecked(True)
        self.checkbox.setToolTip(frame.title)
        self.checkbox.setStyleSheet(
            f"QCheckBox {{ font-size: 11px; color: {TEXT_COLOR}; }}"
        )
        self.checkbox.toggled.connect(on_toggle)
        layout.addWidget(self.thumb)
        layout.addWidget(self.checkbox)

    def set_image(self, rgb: Optional[np.ndarray]) -> None:
        if rgb is None:
            self.thumb.clear()
            return
        pixmap = QPixmap.fromImage(_rgb_to_qimage(rgb))
        self.thumb.setPixmap(
            pixmap.scaled(
                _THUMB_W,
                _THUMB_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_current(self, current: bool) -> None:
        colour = ACCENT_COLOR if current else "transparent"
        self.thumb.setStyleSheet(f"QLabel {{ border: 2px solid {colour}; }}")


class AnimationExportDialog(QDialog):
    """Preview and strip on the left, settings in tabs on the right, Save below."""

    def __init__(
        self,
        frames: Sequence[AnimationFrame],
        title: str,
        save_directory: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.frames = list(frames)
        self.title = title
        self.save_directory = save_directory
        self.options = AnimationOptions(hold_ms=_HOLD_MS)
        self._rendered: List[np.ndarray] = []
        self._indices: List[int] = []
        self._pixmaps: List[QPixmap] = []
        self._current = 0  # position in the rendered frames
        self._format = _FORMATS[0]  # (name, extension)
        self._playing = True

        self.setWindowTitle("Export GIF")
        self.setStyleSheet(f"QDialog {{ background-color: {SURFACE_COLOR}; }}")
        self.resize(940, 640)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_preview(), 1)
        body.addWidget(self._build_settings())
        root.addLayout(body, 1)
        root.addWidget(self._build_footer())

        self._refresh()

    # -- construction --------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 10)
        heading = QLabel("Export GIF")
        heading.setStyleSheet(PANEL_TITLE_STYLE)
        tasks = len(self.frames)
        meta = _caption(f"{self.title} · {tasks} task{'s' if tasks != 1 else ''}")
        layout.addWidget(heading)
        layout.addSpacing(8)
        layout.addWidget(meta, 1)
        return header

    def _build_preview(self) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {PANEL_COLOR}; }}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        self.label_preview = QLabel()
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setMinimumSize(480, 300)
        self.label_preview.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        # Rescale on the label's own resize, not the dialog's: the first frame is set
        # before the layout settles, and the dialog's resize comes too early to see
        # the label's final size -- the preview was cropped until the next refresh.
        self.label_preview.installEventFilter(self)
        layout.addWidget(self.label_preview, 1)

        controls = QHBoxLayout()
        self.button_play = QToolButton()
        self.button_play.setAutoRaise(True)
        self.button_play.clicked.connect(self.toggle_playing)
        self.label_position = _caption("")
        controls.addWidget(self.button_play)
        controls.addWidget(self.label_position, 1)
        layout.addLayout(controls)

        strip = QWidget()
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.setSpacing(8)
        self.tiles: List[_FrameTile] = []
        for i, f in enumerate(self.frames):
            tile = _FrameTile(
                f,
                on_click=lambda i=i: self.show_frame_of(i),
                on_toggle=self._on_changed,
            )
            self.tiles.append(tile)
            strip_layout.addWidget(tile)
        strip_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(strip)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(_THUMB_H + 40)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll)
        return frame

    def _build_settings(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(240)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setStyleSheet(
            f"QTabBar::tab {{ padding: 6px 10px; {CONTROL_STYLE} }}"
        )
        o = self.options

        self.segment_beam = _Segmented([n for n, _ in _BEAMS], _index(_BEAMS, o.beam))
        self.segment_magnification = _Segmented(
            [n for n, _ in _MAGNIFICATIONS], _index(_MAGNIFICATIONS, o.magnification)
        )
        self.checkbox_auto_contrast = QCheckBox("Auto contrast per frame")
        self.checkbox_auto_contrast.setToolTip(
            "Saved images vary in brightness; without this the GIF flickers"
        )
        self.checkbox_auto_contrast.setChecked(o.auto_contrast)
        self.tabs.addTab(
            _tab(
                _caption("Beam"),
                self.segment_beam,
                _caption("Image from each task"),
                self.segment_magnification,
                self.checkbox_auto_contrast,
            ),
            "Frames",
        )

        self.checkbox_title = QCheckBox("Task name")
        self.checkbox_lamella = QCheckBox("Lamella name")
        self.checkbox_counter = QCheckBox("Step counter")
        self.checkbox_scalebar = QCheckBox("Scalebar")
        self.checkbox_bar = QCheckBox("Acquisition values")
        self.checkbox_experiment = QCheckBox("Experiment and date")
        for checkbox, value in (
            (self.checkbox_title, o.title),
            (self.checkbox_lamella, o.lamella),
            (self.checkbox_counter, o.step_counter),
            (self.checkbox_scalebar, o.scalebar),
            (self.checkbox_bar, o.bar),
            (self.checkbox_experiment, o.experiment),
        ):
            checkbox.setChecked(value)
        self.tabs.addTab(
            _tab(
                _caption("On each frame"),
                self.checkbox_title,
                self.checkbox_lamella,
                self.checkbox_counter,
                self.checkbox_scalebar,
                _caption("Bar"),
                self.checkbox_bar,
                self.checkbox_experiment,
            ),
            "Labels",
        )

        self.segment_frame_time = _Segmented(
            [n for n, _ in _FRAME_TIMES], _index(_FRAME_TIMES, o.frame_ms)
        )
        self.checkbox_hold = QCheckBox("Hold the last frame")
        self.checkbox_hold.setChecked(o.hold_ms > o.frame_ms)
        self.segment_width = _Segmented(
            [n for n, _ in _WIDTHS], _index(_WIDTHS, o.width)
        )
        self.segment_format = _Segmented([n for n, _ in _FORMATS])
        self.segment_format.group.button(0).setToolTip(
            "Plays everywhere, PowerPoint included"
        )
        webp = self.segment_format.group.button(1)
        if webp_supported():
            webp.setToolTip("Full colour and far smaller; PowerPoint may not play it")
        else:
            webp.setEnabled(False)
            webp.setToolTip("This installation can't write animated WebP")
        self.tabs.addTab(
            _tab(
                _row("Frame time", self.segment_frame_time),
                self.checkbox_hold,
                _row("Width", self.segment_width),
                _row("Format", self.segment_format),
            ),
            "Output",
        )

        for segment in (
            self.segment_beam,
            self.segment_magnification,
            self.segment_frame_time,
            self.segment_width,
            self.segment_format,
        ):
            segment.changed.connect(self._on_changed)
        for checkbox in (
            self.checkbox_auto_contrast,
            self.checkbox_title,
            self.checkbox_lamella,
            self.checkbox_counter,
            self.checkbox_scalebar,
            self.checkbox_bar,
            self.checkbox_experiment,
            self.checkbox_hold,
        ):
            checkbox.toggled.connect(self._on_changed)
        return self.tabs

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setStyleSheet(
            f"QFrame {{ border-top: 1px solid {BORDER_COLOR}; }}"
            " QLabel, QPushButton { border-top: none; }"
        )
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 10, 16, 10)
        self.label_summary = _caption("")
        self.button_save = QPushButton("Save…")
        self.button_save.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.button_save.clicked.connect(self._on_save)
        layout.addWidget(self.label_summary, 1)
        layout.addWidget(self.button_save)
        return footer

    # -- state ---------------------------------------------------------------

    def _on_changed(self, *_) -> None:
        self._read_options()
        self._refresh()

    def _read_options(self) -> None:
        o = self.options
        o.beam = _BEAMS[self.segment_beam.index()][1]
        o.magnification = _MAGNIFICATIONS[self.segment_magnification.index()][1]
        o.auto_contrast = self.checkbox_auto_contrast.isChecked()
        o.title = self.checkbox_title.isChecked()
        o.lamella = self.checkbox_lamella.isChecked()
        o.step_counter = self.checkbox_counter.isChecked()
        o.scalebar = self.checkbox_scalebar.isChecked()
        o.bar = self.checkbox_bar.isChecked()
        o.experiment = self.checkbox_experiment.isChecked()
        o.frame_ms = _FRAME_TIMES[self.segment_frame_time.index()][1]
        o.hold_ms = _HOLD_MS if self.checkbox_hold.isChecked() else 0
        o.width = _WIDTHS[self.segment_width.index()][1]
        self._format = _FORMATS[self.segment_format.index()]
        o.skipped = [i for i, t in enumerate(self.tiles) if not t.checkbox.isChecked()]

    def _refresh(self) -> None:
        # Both sit on the title plate, so neither shows without it.
        self.checkbox_lamella.setEnabled(self.options.title)
        self.checkbox_counter.setEnabled(self.options.title)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._rendered = render_animation(self.frames, self.options, self.title)
        finally:
            QApplication.restoreOverrideCursor()
        self._indices = included(self.frames, self.options)
        self._pixmaps = [QPixmap.fromImage(_rgb_to_qimage(r)) for r in self._rendered]
        self._refresh_tiles()

        n = len(self._rendered)
        if n:
            h, w = self._rendered[0].shape[:2]
            seconds = sum(frame_durations(n, self.options)) / 1000
            frames = f"{n} frame{'s' if n != 1 else ''}"
            self.label_summary.setText(
                f"{w} × {h} px · {frames} · {seconds:.1f} s · {self._format[0]}"
            )
        else:
            self.label_summary.setText(
                "No frames: every task is left out or has no image"
            )
        self.button_save.setEnabled(n > 0)
        self._current = min(self._current, max(0, n - 1))
        self._show_current()

    def _refresh_tiles(self) -> None:
        """Each tile shows the image the chosen beam would use; a task without one
        is greyed, with the reason, rather than silently missing from the GIF."""
        kinds = ["SEM", "FIB"] if self.options.beam == "both" else [self.options.beam]
        for frame, tile in zip(self.frames, self.tiles):
            images = [frame.pick(k, self.options.magnification) for k in kinds]
            available = all(images)
            tile.set_image(images[-1].rgb if available else None)
            tile.checkbox.setEnabled(available)
            missing = [k for k, image in zip(kinds, images) if image is None]
            tile.setToolTip("" if available else f"No {' or '.join(missing)} image")

    def _show_current(self) -> None:
        self._timer.stop()
        for i, tile in enumerate(self.tiles):
            current = bool(self._indices) and self._indices[self._current] == i
            tile.set_current(current)
        if not self._pixmaps:
            self.label_preview.clear()
            self.label_position.setText("")
            self._update_play_button()
            return
        self._paint_preview()
        frame = self.frames[self._indices[self._current]]
        self.label_position.setText(
            f"{self._current + 1} / {len(self._pixmaps)} · {frame.title}"
        )
        self._update_play_button()
        if self._playing and len(self._pixmaps) > 1:
            durations = frame_durations(len(self._pixmaps), self.options)
            self._timer.start(durations[self._current])

    def _paint_preview(self) -> None:
        if not self._pixmaps:
            return
        self.label_preview.setPixmap(
            self._pixmaps[self._current].scaled(
                self.label_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        if obj is self.label_preview and event.type() == QEvent.Type.Resize:
            self._paint_preview()
        return super().eventFilter(obj, event)

    def _advance(self) -> None:
        if self._pixmaps:
            self._current = (self._current + 1) % len(self._pixmaps)
        self._show_current()

    def _update_play_button(self) -> None:
        icon = "mdi:pause" if self._playing else "mdi:play"
        self.button_play.setIcon(fibsem_icon(icon, color=TEXT_COLOR))
        self.button_play.setToolTip("Pause" if self._playing else "Play")

    def toggle_playing(self) -> None:
        self._playing = not self._playing
        self._show_current()

    def show_frame_of(self, index: int) -> None:
        """Jump to a task's frame, and stop there so it can be looked at."""
        if index in self._indices:
            self._current = self._indices.index(index)
            self._playing = False
            self._show_current()

    # -- output --------------------------------------------------------------

    def save(self, path: str) -> str:
        """Write the animation to ``path``, in the chosen format if it has no
        extension (an extension given wins: the writer goes by it)."""
        if not os.path.splitext(path)[1]:
            path += self._format[1]
        return save_animation(self._rendered, path, self.options)

    def _on_save(self) -> None:
        name, extension = self._format
        default = default_animation_name(self.title, self.save_directory, extension)
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {name}", default, f"{name} (*{extension})"
        )
        if not path:
            return
        try:
            written = self.save(path)
        except (OSError, ValueError) as e:
            logging.exception("Failed to save animation to %s", path)
            notification_service.show_toast(
                f"Couldn't save the {self._format[0]}: {e}", "error"
            )
            return
        notification_service.show_toast(f"Saved {os.path.basename(written)}", "success")
        self.accept()

    def done(self, result: int) -> None:
        self._timer.stop()
        super().done(result)
