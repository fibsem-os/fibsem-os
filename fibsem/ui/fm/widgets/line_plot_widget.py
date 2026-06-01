import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QEvent, QObject, QSize
from PyQt5.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

_OVERLAY_BTN_STYLE = (
    "QPushButton { background: rgba(40,41,48,180); border: 1px solid #555;"
    " border-radius: 3px; padding: 0px; }"
    "QPushButton:hover { background: rgba(74,74,74,200); }"
    "QPushButton:pressed { background: rgba(30,30,30,220); }"
    "QPushButton:checked { background: rgba(90,92,100,200); border-color: #FFFFFF; }"
)
_OVERLAY_ICON_SIZE = QSize(14, 14)
_OVERLAY_BTN_SIZE = 22


class _CanvasOverlayFilter(QObject):
    """Repositions a list of overlay buttons on the canvas when it is resized.

    Buttons are stacked right-to-left in the top-right corner with a 4 px margin.
    """

    _MARGIN = 4
    _GAP = 2

    def __init__(self, canvas: QWidget, buttons: list) -> None:
        super().__init__(canvas)
        self._canvas = canvas
        self._buttons = buttons
        self._reposition()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._canvas and event.type() == QEvent.Type.Resize:
            self._reposition()
        return False

    def _reposition(self) -> None:
        w = self._canvas.width()
        x = w - self._MARGIN
        for btn in self._buttons:
            bw, bh = btn.width(), btn.height()
            x -= bw
            btn.move(x, self._MARGIN)
            x -= self._GAP


class LinePlotWidget(QWidget):
    """Widget for displaying line plots with rolling statistics."""

    INITIAL_XLIM_SECONDS = 30  # Initial time window in seconds
    INITIAL_XLIM_SAMPLES = 100  # Initial sample window

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        max_length: int = 1000,
        rolling_mean_window: int = 10,
    ):
        super().__init__(parent)
        self.max_length = max_length
        self.rolling_mean_window = max(1, int(rolling_mean_window))
        self.data_buffer = deque(maxlen=self.max_length)
        self.time_buffer = deque(maxlen=self.max_length)
        self.last_update_time = 0
        self.update_interval = 0.25
        self.updates_paused = False
        self.use_datetime_axis = True

        # Blitting optimization variables
        self.background = None
        self.line_artist = None
        self.mean_line_artist = None
        self.rolling_mean_line_artist = None
        self.peak_rolling_mean_artist = None
        self.threshold_artist = None
        self.scrub_line_artist = None
        self.legend = None
        self.axes_setup_complete = False
        self.last_legend_update = 0
        self.legend_update_interval = 1.0

        # Pan/zoom state (None = auto-follow live data)
        self._user_xlim: Optional[tuple] = None
        self._user_ylim: Optional[tuple] = None
        self._pan_start_display: Optional[tuple] = None
        self._pan_start_xlim: Optional[tuple] = None
        self._pan_start_ylim: Optional[tuple] = None

        self.initUI()

    def initUI(self):
        """Initialize the line plot widget UI."""
        self.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()

        # Create matplotlib figure and canvas with napari-style dark theme
        self.figure = Figure(figsize=(4, 3), dpi=80)
        self.figure.patch.set_facecolor("#262930")

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(200, 150)

        # Connect pan/zoom mouse events
        self._mpl_cids = [
            self.canvas.mpl_connect("scroll_event", self._on_scroll),
            self.canvas.mpl_connect("button_press_event", self._on_mouse_press),
            self.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion),
            self.canvas.mpl_connect("button_release_event", self._on_mouse_release),
        ]

        # ── Overlay buttons (parented to canvas, stacked right-to-left) ──────
        def _obtn(icon_name: str, tooltip: str, checkable: bool = False) -> QPushButton:
            btn = QPushButton(self.canvas)
            btn.setIcon(QIconifyIcon(icon_name, color="#aaaaaa"))
            btn.setIconSize(_OVERLAY_ICON_SIZE)
            btn.setFixedSize(_OVERLAY_BTN_SIZE, _OVERLAY_BTN_SIZE)
            btn.setToolTip(tooltip)
            btn.setCheckable(checkable)
            btn.setStyleSheet(_OVERLAY_BTN_STYLE)
            btn.raise_()
            return btn

        self._btn_reset_zoom = _obtn("mdi:fit-to-screen-outline", "Reset zoom / snap to live")
        self._btn_reset_zoom.clicked.connect(self._reset_view)

        self.reset_button = _obtn("mdi:trash-can-outline", "Reset chart (clear data)")
        self.reset_button.clicked.connect(self.reset_chart)

        self.xaxis_button = _obtn("mdi:clock-outline", "Toggle X axis: Time / Sample index")
        self.xaxis_button.clicked.connect(self.toggle_xaxis_mode)

        self.stats_button = _obtn("mdi:chart-bar", "Show / hide statistics", checkable=True)
        self.stats_button.clicked.connect(
            lambda checked: self.label_stats.setVisible(checked)
        )

        self.pause_button = _obtn("mdi:pause", "Pause live updates")
        self.pause_button.clicked.connect(self.toggle_updates)

        # Order: rightmost first → [reset_zoom, reset, xaxis, stats, pause]
        self._canvas_resize_filter = _CanvasOverlayFilter(
            self.canvas,
            [self._btn_reset_zoom, self.reset_button, self.xaxis_button,
             self.stats_button, self.pause_button],
        )
        self.canvas.installEventFilter(self._canvas_resize_filter)

        # Create the subplot with dark styling
        self.ax = self.figure.add_subplot(111)

        self._apply_dark_theme()

        # Initial empty plot
        self._plot_empty_chart()

        layout.addWidget(self.canvas)

        # Stats label (below canvas — text, not a button)
        self.label_stats = QLabel("No data", self)
        self.label_stats.setStyleSheet(
            "QLabel { color: #bbbbbb; font-size: 10px; background-color: transparent; }"
        )
        self.label_stats.setWordWrap(True)
        self.label_stats.setVisible(False)
        layout.addWidget(self.label_stats)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _plot_empty_chart(self):
        """Plot an empty chart with placeholder text."""
        self.ax.clear()
        self._apply_dark_theme()

        self.ax.text(
            0.5,
            0.5,
            "No data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=12,
            color="#bbbbbb",
        )
        self.ax.set_xlabel("Time" if self.use_datetime_axis else "Sample")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Line Plot")
        self.figure.tight_layout()
        self.canvas.draw()
        self.axes_setup_complete = False

    def _apply_dark_theme(self):
        """Apply napari-style dark theme to the axes."""
        self.ax.set_facecolor("#262930")
        self.ax.tick_params(colors="white", which="both")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

    def _setup_axes_for_blitting(self):
        """Set up the axes with static elements for blitting optimization."""
        self.ax.clear()
        self._apply_dark_theme()

        # Set up static elements
        self.ax.set_xlabel("Time" if self.use_datetime_axis else "Sample")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Line Plot")
        self.ax.set_ylim(0, 1)  # Will be updated dynamically

        if self.use_datetime_axis:
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            self.ax.tick_params(axis="x", rotation=30)
            now = datetime.now()
            self.ax.set_xlim(
                float(mdates.date2num(now)),
                float(
                    mdates.date2num(now + timedelta(seconds=self.INITIAL_XLIM_SECONDS))
                ),
            )
        else:
            self.ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            self.ax.tick_params(axis="x", rotation=0)
            self.ax.set_xlim(0, self.INITIAL_XLIM_SAMPLES)

        # Create empty line artists for blitting
        (self.line_artist,) = self.ax.plot(
            [],
            [],
            "o-",
            color="#0f7aad",
            linewidth=2,
            markersize=2,
            alpha=0.8,
            label="Values",
            animated=True,
        )
        self.mean_line_artist = self.ax.axhline(
            0,
            color="#ff6600",
            linestyle="--",
            alpha=0.9,
            animated=True,
            label="Average: 0.00",
        )
        self.rolling_mean_line_artist = self.ax.axhline(
            0,
            color="#00ff66",
            linestyle=":",
            alpha=0.9,
            animated=True,
            label=f"Last {self.rolling_mean_window} Mean: 0.00",
        )
        self.peak_rolling_mean_artist = self.ax.axhline(
            0,
            color="#ff9800",
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            animated=True,
            label="Peak Rolling Mean: —",
            visible=False,
        )
        self.threshold_artist = self.ax.axhline(
            0,
            color="#ef5350",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            animated=True,
            label="Threshold: —",
            visible=False,
        )

        # Set up legend in fixed top left corner
        self.legend = self.ax.legend(loc="upper left", fontsize=8)
        self.legend.get_frame().set_facecolor("#262930")
        self.legend.get_frame().set_edgecolor("white")
        for text in self.legend.get_texts():
            text.set_color("white")

        self.figure.tight_layout()
        self.canvas.draw()

        # Save background for blitting
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.axes_setup_complete = True

    def append_value(self, value: float, title: str = ""):
        """Append a new value to the chart with rate limiting.

        Args:
            value: The new value to add to the plot
            title: Optional title for the plot
        """
        # Always store the data even if updates are paused
        self.data_buffer.append(value)
        self.time_buffer.append(datetime.now())

        # Skip plot update if paused
        if self.updates_paused:
            return

        # Skip update if widget is not visible to save CPU
        if not self.isVisible():
            return

        current_time = time.time()

        # Rate limiting: only update plot if enough time has passed
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time
        self._append_value_internal(value, title)

    def _append_value_internal(self, value: float, title: str = ""):
        """Internal method to actually update the plot display using blitting."""
        if len(self.data_buffer) == 0:
            self._plot_empty_chart()
            self.label_stats.setText("No data")
            return

        try:
            # Convert buffers to arrays
            data_array = np.array(list(self.data_buffer))
            time_array = list(self.time_buffer)

            # Set up axes for blitting if not done yet
            if not self.axes_setup_complete:
                self._setup_axes_for_blitting()

            # Calculate statistics
            current_val = data_array[-1]
            mean_val = np.mean(data_array)

            window = self.rolling_mean_window
            rolling_window_data = (
                data_array[-window:] if len(data_array) >= window else data_array
            )
            rolling_mean = np.mean(rolling_window_data)

            # Update y-axis limits if needed
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            y_margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            new_ylim = (min_val - y_margin, max_val + y_margin)

            # Compute x values and desired xlim based on axis mode
            if self.use_datetime_axis:
                x_values = time_array
                t_min = time_array[0]
                t_max = time_array[-1]
                initial_end = t_min + timedelta(seconds=self.INITIAL_XLIM_SECONDS)
                desired_xlim_num = (
                    float(mdates.date2num(t_min)),
                    float(mdates.date2num(max(t_max, initial_end))),
                )
            else:
                x_values = np.arange(len(data_array))
                desired_xlim_num = (
                    0.0,
                    float(max(self.INITIAL_XLIM_SAMPLES, len(data_array) - 1)),
                )

            # Respect user pan/zoom — don't override manual view limits
            if self._user_xlim is not None:
                desired_xlim_num = self._user_xlim
            if self._user_ylim is not None:
                new_ylim = self._user_ylim

            current_xlim = self.ax.get_xlim()
            xlim_changed = (
                current_xlim[0] != desired_xlim_num[0]
                or current_xlim[1] != desired_xlim_num[1]
            )

            if self.ax.get_ylim() != new_ylim or xlim_changed:
                if xlim_changed:
                    self.ax.set_xlim(desired_xlim_num[0], desired_xlim_num[1])
                self.ax.set_ylim(new_ylim)
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.ax.bbox)

            # Restore background
            self.canvas.restore_region(self.background)

            # Update line data
            if self.line_artist is not None:
                self.line_artist.set_data(x_values, data_array)

            # Update horizontal lines
            if self.mean_line_artist is not None:
                self.mean_line_artist.set_ydata([mean_val, mean_val])
                self.mean_line_artist.set_label(f"Average: {mean_val:.2f}")

            if self.rolling_mean_line_artist is not None:
                self.rolling_mean_line_artist.set_ydata([rolling_mean, rolling_mean])
                self.rolling_mean_line_artist.set_label(
                    f"Last {self.rolling_mean_window} Mean: {rolling_mean:.2f}"
                )

            # Check if legend needs updating (less frequent to avoid performance hit)
            current_time = time.time()
            legend_needs_update = (
                current_time - self.last_legend_update
            ) > self.legend_update_interval

            if legend_needs_update and self.legend is not None:
                self.legend.remove()
                self.legend = self.ax.legend(loc="upper left", fontsize=8)
                self.legend.get_frame().set_facecolor("#262930")
                self.legend.get_frame().set_edgecolor("white")
                for text in self.legend.get_texts():
                    text.set_color("white")
                self.last_legend_update = current_time

                # Redraw background to include updated legend
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.ax.bbox)
                self.canvas.restore_region(self.background)

            # Draw artists
            if self.line_artist is not None:
                self.ax.draw_artist(self.line_artist)
            if self.mean_line_artist is not None:
                self.ax.draw_artist(self.mean_line_artist)
            if self.rolling_mean_line_artist is not None:
                self.ax.draw_artist(self.rolling_mean_line_artist)
            if (
                self.peak_rolling_mean_artist is not None
                and self.peak_rolling_mean_artist.get_visible()
            ):
                self.ax.draw_artist(self.peak_rolling_mean_artist)
            if (
                self.threshold_artist is not None
                and self.threshold_artist.get_visible()
            ):
                self.ax.draw_artist(self.threshold_artist)
            if (
                self.scrub_line_artist is not None
                and self.scrub_line_artist.get_visible()
            ):
                self.ax.draw_artist(self.scrub_line_artist)

            # Blit the updated artists
            self.canvas.blit(self.ax.bbox)

            # Update statistics label
            std_val = np.std(data_array)
            last_updated = datetime.now().strftime("%I:%M:%S %p")

            stats_text = (
                f"Current: {current_val:.2f} | Count: {len(data_array)}\n"
                f"Min: {min_val:.2f} | Max: {max_val:.2f}\n"
                f"Average: {mean_val:.2f} | Std: {std_val:.2f}\n"
                f"Last {self.rolling_mean_window} Mean: {rolling_mean:.2f}\n"
                f"Updated: {last_updated}"
            )
            self.label_stats.setText(stats_text)

        except Exception as e:
            logging.error(f"Error updating line plot: {e}")
            self.axes_setup_complete = False
            self._plot_empty_chart()
            self.label_stats.setText(f"Error: {str(e)}")

    def toggle_updates(self):
        """Toggle pause/resume state for plot updates."""
        self.updates_paused = not self.updates_paused

        if self.updates_paused:
            self.pause_button.setIcon(QIconifyIcon("mdi:play", color="#ff9800"))
            self.pause_button.setToolTip("Resume live updates")
        else:
            self.pause_button.setIcon(QIconifyIcon("mdi:pause", color="#aaaaaa"))
            self.pause_button.setToolTip("Pause live updates")
            # Trigger a plot update with the current data when resuming
            if len(self.data_buffer) > 0:
                self._append_value_internal(self.data_buffer[-1], "")

    def toggle_xaxis_mode(self):
        """Toggle between datetime and sample index x-axis."""
        self.use_datetime_axis = not self.use_datetime_axis
        if self.use_datetime_axis:
            self.xaxis_button.setIcon(QIconifyIcon("mdi:clock-outline", color="#aaaaaa"))
            self.xaxis_button.setToolTip("X axis: Time — click to switch to Sample index")
        else:
            self.xaxis_button.setIcon(QIconifyIcon("mdi:numeric", color="#aaaaaa"))
            self.xaxis_button.setToolTip("X axis: Sample index — click to switch to Time")

        # Force axes re-setup and replot
        self.axes_setup_complete = False
        self.background = None
        self.line_artist = None
        self.mean_line_artist = None
        self.rolling_mean_line_artist = None
        self.peak_rolling_mean_artist = None
        self.threshold_artist = None
        self.scrub_line_artist = None
        self.legend = None
        self.last_legend_update = 0

        if len(self.data_buffer) > 0:
            self._append_value_internal(self.data_buffer[-1], "")
        else:
            self._plot_empty_chart()

    def reset_chart(self):
        """Reset the chart by clearing all data."""
        self.data_buffer.clear()
        self.time_buffer.clear()
        self._user_xlim = None
        self._user_ylim = None
        self.axes_setup_complete = False
        self.background = None
        self.line_artist = None
        self.mean_line_artist = None
        self.rolling_mean_line_artist = None
        self.peak_rolling_mean_artist = None
        self.threshold_artist = None
        self.scrub_line_artist = None
        self.legend = None
        self.last_legend_update = 0
        self._plot_empty_chart()
        self.label_stats.setText("No data")

    # ------------------------------------------------------------------
    # Pan / zoom interaction
    # ------------------------------------------------------------------

    def _redraw_after_interaction(self) -> None:
        """Full canvas redraw (needed when axis limits change) + save blit background."""
        self.canvas.draw()
        if self.axes_setup_complete:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def _reset_view(self) -> None:
        """Clear manual pan/zoom and return to auto-follow live data."""
        self._user_xlim = None
        self._user_ylim = None
        if len(self.data_buffer) > 0:
            self.axes_setup_complete = False  # force limit recalc on next update
            self._append_value_internal(self.data_buffer[-1])
        else:
            self._redraw_after_interaction()

    def _on_scroll(self, event) -> None:
        """Zoom x-axis on scroll wheel, centred on cursor position."""
        if not self.axes_setup_complete or event.inaxes is not self.ax:
            return
        factor = 0.8 if event.step > 0 else 1.25  # scroll-up = zoom in
        x0, x1 = self.ax.get_xlim()
        xc = event.xdata if event.xdata is not None else (x0 + x1) / 2
        new_x0 = xc - (xc - x0) * factor
        new_x1 = xc + (x1 - xc) * factor
        self._user_xlim = (new_x0, new_x1)
        self.ax.set_xlim(new_x0, new_x1)
        self._redraw_after_interaction()

    def _on_mouse_press(self, event) -> None:
        """Start pan on left-button press inside axes."""
        if event.button != 1 or event.inaxes is not self.ax:
            return
        self._pan_start_display = (event.x, event.y)
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()

    def _on_mouse_motion(self, event) -> None:
        """Pan axes while left button is held."""
        if self._pan_start_display is None or event.inaxes is not self.ax:
            return
        # Convert pixel delta to data-space delta using the *original* transform
        inv = self.ax.transData.inverted()
        start_data = inv.transform(self._pan_start_display)
        curr_data = inv.transform((event.x, event.y))
        dx = start_data[0] - curr_data[0]
        dy = start_data[1] - curr_data[1]
        x0, x1 = self._pan_start_xlim
        y0, y1 = self._pan_start_ylim
        new_xlim = (x0 + dx, x1 + dx)
        new_ylim = (y0 + dy, y1 + dy)
        self._user_xlim = new_xlim
        self._user_ylim = new_ylim
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._redraw_after_interaction()

    def _on_mouse_release(self, event) -> None:
        """End pan."""
        if event.button == 1:
            self._pan_start_display = None
            self._pan_start_xlim = None
            self._pan_start_ylim = None

    def update_stats_lines(
        self, peak_rolling_mean: float, threshold_value: float, warmup_complete: bool
    ) -> None:
        """Update peak-rolling-mean and threshold horizontal lines from live strategy stats.

        Called every FM frame during milling. Only redraws background when values change
        enough to affect visibility; the blit pipeline picks up the rest.
        """
        if not self.axes_setup_complete:
            return

        changed = False

        if self.peak_rolling_mean_artist is not None and peak_rolling_mean > 0:
            self.peak_rolling_mean_artist.set_ydata(
                [peak_rolling_mean, peak_rolling_mean]
            )
            self.peak_rolling_mean_artist.set_label(
                f"Peak Rolling Mean: {peak_rolling_mean:.2f}"
            )
            visible = warmup_complete
            if self.peak_rolling_mean_artist.get_visible() != visible:
                self.peak_rolling_mean_artist.set_visible(visible)
                changed = True

        if self.threshold_artist is not None:
            show = warmup_complete and threshold_value > 0
            self.threshold_artist.set_ydata([threshold_value, threshold_value])
            self.threshold_artist.set_label(f"Threshold: {threshold_value:.2f}")
            if self.threshold_artist.get_visible() != show:
                self.threshold_artist.set_visible(show)
                changed = True

        if changed:
            # Invalidate blit background so legend picks up updated labels
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def set_scrub_timestamp(self, ts: Optional[float]) -> None:
        """Show or hide a yellow dashed vertical marker at Unix timestamp *ts*.

        Pass ``None`` to hide the marker and return to normal live view.
        Has no effect when the axes are not yet set up (no data received yet).
        """
        if not self.axes_setup_complete:
            return

        if ts is None:
            if self.scrub_line_artist is not None:
                self.scrub_line_artist.set_visible(False)
            # Invalidate blit background so the hidden line is erased
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            return

        # Compute x position in axis coordinates
        if self.use_datetime_axis:
            x_pos = float(mdates.date2num(datetime.fromtimestamp(ts)))
        else:
            # Map to nearest sample index by timestamp proximity
            if not self.time_buffer:
                return
            target = datetime.fromtimestamp(ts)
            diffs = [abs((t - target).total_seconds()) for t in self.time_buffer]
            x_pos = float(diffs.index(min(diffs)))

        if self.scrub_line_artist is None:
            self.scrub_line_artist = self.ax.axvline(
                x=x_pos,
                color="yellow",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                animated=True,
            )
        else:
            self.scrub_line_artist.set_xdata([x_pos, x_pos])
            self.scrub_line_artist.set_visible(True)

        # Invalidate blit background — next blit cycle will draw the line on top
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def update_chart(self, data_array: Optional[np.ndarray], title: str = ""):
        """Update the chart with a new array of data (replaces current data).

        Args:
            data_array: 1D numpy array containing the data to plot
            title: Optional title for the plot
        """
        if data_array is None or data_array.size == 0:
            self.reset_chart()
            return

        # Clear buffers and add new data
        self.data_buffer.clear()
        self.time_buffer.clear()

        # Take only the last max_length values if array is too long
        if len(data_array) > self.max_length:
            data_array = data_array[-self.max_length :]

        # Generate timestamps spaced 1 second apart ending at now
        now = datetime.now()
        start = now - timedelta(seconds=len(data_array) - 1)

        for i, value in enumerate(data_array):
            self.data_buffer.append(float(value))
            self.time_buffer.append(start + timedelta(seconds=i))

        # Update the plot
        self._append_value_internal(self.data_buffer[-1], title)
