"""Acquire a fluorescence overview: settings, run control, and live result.

Standalone, and embeddable as a tab. It owns nothing the acquisition needs -- it is
handed a microscope and drives `FMTiledAcquisitionRunner` -- so it can be dropped into
the AutoLamella UI or opened on its own against a simulator.

Layout follows the house convention: canvas on the left, controls on the right,
actions along the bottom.
"""

import logging
import threading
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.acquisition import FMTiledAcquisitionRunner, stitch_tileset
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    FluorescenceImage,
    OverviewParameters,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.ui import stylesheets
from fibsem.ui.fm.widgets.channel_list_widget import ChannelListWidget
from fibsem.ui.fm.widgets.fm_overview_confirmation_dialog import (
    FMOverviewConfirmationDialog,
    format_duration,
)
from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.widgets.canvas.fm_canvas import FMCanvasWidget

TEXT_MUTED = "#868e93"


class FMOverviewWidget(QWidget):
    """Configure, run and view a fluorescence overview acquisition."""

    overview_acquired = pyqtSignal(FluorescenceImage)

    # Internal hop from the acquisition thread to the GUI thread. The microscope's
    # progress signal is a psygnal, which calls its callbacks synchronously on
    # whichever thread emitted -- here, the worker. Touching widgets from there is a
    # cross-thread GUI access (Qt says so: "Cannot set parent, new parent is in a
    # different thread"). Re-emitting as a Qt signal gets it queued onto the GUI
    # thread, because this widget lives there.
    _progress_received = pyqtSignal(dict)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        channel_settings: Optional[List[ChannelSettings]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if microscope.fm is None:
            raise ValueError("This microscope has no fluorescence detector.")
        self.microscope = microscope
        self.fm = microscope.fm

        self._stop_event = threading.Event()
        self._worker: Optional[FunctionWorker] = None
        self._runner: Optional[FMTiledAcquisitionRunner] = None
        self._mosaic: Optional[FluorescenceImage] = None
        self._enabled_channels: Optional[List[ChannelSettings]] = None

        self._init_ui(channel_settings or self._default_channels())
        self._sync_tile_fov()
        self._on_settings_changed()

        self._progress_received.connect(self._apply_progress)
        self.fm.acquisition_progress_signal.connect(self._on_progress)

    def _default_channels(self) -> List[ChannelSettings]:
        """The saved FM configuration if there is one, otherwise a single channel."""
        try:
            from fibsem.fm.config import load_fm_configuration

            config = load_fm_configuration()
            if config is not None and config.channel_settings:
                return list(config.channel_settings)
        except Exception as e:
            logging.debug(f"Could not load the saved FM configuration: {e}")
        return [ChannelSettings(name="Channel-01")]

    # ── layout ───────────────────────────────────────────────────────────

    def _init_ui(self, channels: List[ChannelSettings]) -> None:
        self.canvas = FMCanvasWidget()

        self.channel_widget = ChannelListWidget(self.fm, channels)
        # Every overview setting lives in one widget, z-stack included, so their order
        # is decided in one place rather than split across two.
        self.settings_widget = FMOverviewSettingsWidget(
            channel_names=[ch.name for ch in channels]
        )

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self._section("Channels", self.channel_widget))
        controls_layout.addWidget(self.settings_widget)
        controls_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(480)
        # Vertical only: a horizontal bar here means a control is refusing to shrink,
        # and scrolling sideways to reach a spinbox is worse than a cramped one.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.progress.setFixedHeight(18)
        self.progress.hide()

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        self.button_acquire = QPushButton("Acquire Overview")
        self.button_acquire.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_acquire.setMinimumHeight(30)
        self.button_acquire.clicked.connect(self.acquire)

        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_cancel.setMinimumHeight(30)
        self.button_cancel.clicked.connect(self.cancel)
        self.button_cancel.setEnabled(False)

        actions = QHBoxLayout()
        actions.setContentsMargins(8, 0, 8, 8)
        actions.addWidget(self.status, stretch=1)
        actions.addWidget(self.button_cancel)
        actions.addWidget(self.button_acquire)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.progress)
        layout.addLayout(actions)

        self.channel_widget.settings_changed.connect(self._on_channels_changed)
        self.channel_widget.enabled_changed.connect(self._on_enabled_changed)
        self.settings_widget.changed.connect(self._on_settings_changed)

    def _section(self, title: str, widget: QWidget) -> QWidget:
        from PyQt5.QtWidgets import QGroupBox
        box = QGroupBox(title)
        inner = QVBoxLayout(box)
        inner.setContentsMargins(6, 6, 6, 6)
        inner.addWidget(widget)
        return box

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_acquiring(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def channels(self) -> List[ChannelSettings]:
        """The channels that will be acquired: those ticked, or all of them.

        `ChannelListWidget` reports ticks through `enabled_changed` rather than
        exposing them, so they are tracked here. Before any tick has been touched the
        list is every channel, which is what the widget shows.
        """
        if self._enabled_channels is None:
            return list(self.channel_widget.channel_settings)
        return list(self._enabled_channels)

    def _sync_tile_fov(self) -> None:
        """Tell the settings widget one tile's field of view, for the area readout."""
        try:
            pixel_size_x, pixel_size_y = self.fm.camera.pixel_size
            width, height = self.fm.camera.resolution
            self.settings_widget.set_tile_fov(width * pixel_size_x, height * pixel_size_y)
        except Exception as e:
            logging.debug(f"Could not read the camera field of view: {e}")

    def _on_channels_changed(self, channels: List[ChannelSettings]) -> None:
        self.settings_widget.set_channel_names([ch.name for ch in channels])
        self._on_settings_changed()

    def _on_enabled_changed(self, channels: List[ChannelSettings]) -> None:
        self._enabled_channels = list(channels)
        self._on_settings_changed()

    def _on_settings_changed(self) -> None:
        if self.is_acquiring:
            return
        parameters = self.settings_widget.parameters
        enabled = parameters.n_enabled_tiles > 0
        self.button_acquire.setEnabled(enabled)
        # Only own the status line while it has something of its own to say. Re-enabling
        # the controls at the end of a run makes children emit `changed`, which landed
        # here and wiped the result message the moment it was set.
        if not enabled:
            self.status.setText("No tiles selected.")
        elif self.status.text() == "No tiles selected.":
            self.status.setText("")

    # ── acquisition ──────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Confirm, then run the overview on a worker thread."""
        if self.is_acquiring:
            logging.warning("An overview acquisition is already running.")
            return
        if self.fm.is_acquiring:
            logging.warning("Stop live acquisition before acquiring an overview.")
            return

        parameters = self.settings_widget.parameters
        channels = self.channels
        if not channels:
            self.status.setText("No channels enabled.")
            return

        zparams = self.settings_widget.z_parameters

        dialog = FMOverviewConfirmationDialog(
            parameters=parameters,
            channel_settings=channels,
            zparams=zparams,
            tile_fov=self.settings_widget._tile_fov,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            logging.info("Overview acquisition cancelled before starting")
            return

        autofocus_settings = None
        if parameters.autofocus_mode is not AutoFocusMode.NONE:
            autofocus_settings = AutoFocusSettings(
                channel_name=self.settings_widget.autofocus_channel_name
            )

        self._stop_event.clear()
        self._set_running(True)
        self._runner = FMTiledAcquisitionRunner(
            microscope=self.microscope,
            channel_settings=channels,
            overview_parameters=parameters,
            zparams=zparams,
            autofocus_settings=autofocus_settings,
            stop_event=self._stop_event,
        )
        self._worker = FunctionWorker(self._acquire_worker)
        self._worker.start()

    def _acquire_worker(self) -> None:
        """Runs off the GUI thread. Only signals may cross back."""
        from fibsem.cancellation import OperationCancelledError

        try:
            runner = self._runner
            runner.run()
            mosaic = stitch_tileset(
                runner.tileset,
                runner.overview_parameters.overlap,
                centre_position=runner._initial_position,
                objective_position=runner._initial_objective_position,
            )
            self._mosaic = mosaic
            self.overview_acquired.emit(mosaic)
            self.fm.acquisition_progress_signal.emit(
                {"state": "overview-finished", "task": "tileset"}
            )
        except OperationCancelledError:
            logging.info("Overview acquisition cancelled")
            self.fm.acquisition_progress_signal.emit(
                {"state": "overview-cancelled", "task": "tileset"}
            )
        except Exception as e:
            logging.error(f"Overview acquisition failed: {e}", exc_info=True)
            self.fm.acquisition_progress_signal.emit(
                {"state": "overview-failed", "task": "tileset", "error": str(e)}
            )

    def cancel(self) -> None:
        if not self.is_acquiring:
            return
        logging.info("Cancelling overview acquisition")
        self._stop_event.set()
        self.button_cancel.setEnabled(False)
        self.status.setText("Cancelling…")

    def _set_running(self, running: bool) -> None:
        self.button_acquire.setEnabled(not running)
        self.button_cancel.setEnabled(running)
        self.settings_widget.setEnabled(not running)
        self.channel_widget.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)
            self.progress.setFormat("Starting…")

    # ── progress ─────────────────────────────────────────────────────────

    def _on_progress(self, payload: dict) -> None:
        """Called by psygnal, on whichever thread emitted. Touches no widgets."""
        self._progress_received.emit(payload)

    def _apply_progress(self, payload: dict) -> None:
        """Runs on the GUI thread, queued via `_progress_received`."""
        state = payload.get("state")

        if state == "tile":
            self._show_preview(payload)
            current, total = payload.get("current", 0), payload.get("total", 1)
            self.progress.setValue(int(current / max(1, total) * 100))
            remaining = payload.get("estimated_remaining_time")
            suffix = f" · {format_duration(remaining)} left" if remaining else ""
            self.progress.setFormat(f"Tile {current}/{total}{suffix}")
        elif state == "moving":
            self.progress.setFormat("Moving stage…")
        elif state in ("overview-finished", "overview-cancelled", "overview-failed"):
            self._finish(state, payload.get("error"))

    def _show_preview(self, payload: dict) -> None:
        """Paint the mosaic-so-far onto the canvas.

        The runner publishes the whole preview canvas each tile, so this stays
        stateless -- it redisplays what it is given rather than accumulating tiles of
        its own, which is also what makes it correct if a frame is dropped.
        """
        image = payload.get("image")
        if image is None:
            return
        try:
            planes = np.asarray(image)
            if planes.ndim == 2:
                planes = planes[np.newaxis]
            for channel, plane in zip(self.channels, planes):
                self.canvas.set_channel(channel.name, plane, channel.color)
        except Exception as e:
            logging.debug(f"Could not display the overview preview: {e}")

    def _finish(self, state: str, error: Optional[str]) -> None:
        self._set_running(False)
        self._worker = None

        if state == "overview-finished" and self._mosaic is not None:
            # Swap the decimated preview for the real thing.
            self.canvas.set_fm_image(self._mosaic)
            shape = self._mosaic.data.shape
            self.status.setText(f"Overview acquired — {shape[-1]} × {shape[-2]} px")
        elif state == "overview-cancelled":
            self.status.setText("Cancelled. Tiles acquired so far are still shown.")
        else:
            self.status.setText(f"Failed: {error}" if error else "Failed.")

    # ── lifecycle ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self.is_acquiring:
            self._stop_event.set()
        try:
            self.fm.acquisition_progress_signal.disconnect(self._on_progress)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)
