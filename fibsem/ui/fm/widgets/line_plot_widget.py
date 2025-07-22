import logging
import time
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Line plot widget will be disabled.")


class LinePlotWidget(QWidget):
    """Widget for displaying line plots with rolling statistics."""

    def __init__(self, parent: Optional[QWidget] = None, max_length: int = 50):
        super().__init__(parent)
        self.max_length = max_length
        self.data_buffer = deque(maxlen=self.max_length)
        self.last_update_time = 0
        self.update_interval = 0.25  # 100ms rate limiting for line plots
        self.updates_paused = False  # Track if updates are paused
        
        # Blitting optimization variables
        self.background = None
        self.line_artist = None
        self.mean_line_artist = None
        self.rolling_mean_line_artist = None
        self.legend = None
        self.axes_setup_complete = False
        self.last_legend_update = 0
        self.legend_update_interval = 1.0  # Update legend every 1 second

        self.initUI()

    def initUI(self):
        """Initialize the line plot widget UI."""
        self.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()

        if not MATPLOTLIB_AVAILABLE:
            # Show error message if matplotlib is not available
            error_label = QLabel(
                "Matplotlib not available.\nInstall with: pip install matplotlib", self
            )
            error_label.setStyleSheet("color: red; font-size: 10px;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)
        else:
            # Create matplotlib figure and canvas with napari-style dark theme
            self.figure = Figure(figsize=(4, 3), dpi=80)
            self.figure.patch.set_facecolor("#262930")  # napari dark background

            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumSize(200, 150)

            # Create the subplot with dark styling
            self.ax = self.figure.add_subplot(111)

            self._apply_dark_theme()

            # Initial empty plot
            self._plot_empty_chart()

            layout.addWidget(self.canvas)

            # Button layout
            button_layout = QHBoxLayout()
            
            # Pause/Resume button
            self.pause_button = QPushButton("Pause", self)
            self.pause_button.setStyleSheet("""
                QPushButton {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #555;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:pressed {
                    background-color: #2a2a2a;
                }
            """)
            self.pause_button.clicked.connect(self.toggle_updates)
            button_layout.addWidget(self.pause_button)
            
            # Reset button
            self.reset_button = QPushButton("Reset", self)
            self.reset_button.setStyleSheet("""
                QPushButton {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #555;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:pressed {
                    background-color: #2a2a2a;
                }
            """)
            self.reset_button.clicked.connect(self.reset_chart)
            button_layout.addWidget(self.reset_button)
            
            layout.addLayout(button_layout)

            # Info label for statistics with napari-style colors
            self.label_stats = QLabel("No data", self)
            self.label_stats.setStyleSheet(
                "QLabel { color: #bbbbbb; font-size: 10px; background-color: transparent; }"
            )
            self.label_stats.setWordWrap(True)
            layout.addWidget(self.label_stats)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _plot_empty_chart(self):
        """Plot an empty chart with placeholder text."""
        if not MATPLOTLIB_AVAILABLE:
            return

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
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Line Plot")
        self.ax.set_xlim(0, self.max_length)
        self.figure.tight_layout()
        self.canvas.draw()
        self.axes_setup_complete = False

    def _apply_dark_theme(self):
        """Apply napari-style dark theme to the axes."""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.ax.set_facecolor("#262930")  # napari dark background
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
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.ax.clear()
        self._apply_dark_theme()
        
        # Set up static elements
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Line Plot")
        self.ax.set_xlim(0, max(1, self.max_length - 1))
        self.ax.set_ylim(0, 1)  # Will be updated dynamically
        
        # Create empty line artists for blitting
        self.line_artist, = self.ax.plot([], [], "o-", color="#0f7aad", 
                                        linewidth=2, markersize=4, alpha=0.8, 
                                        label="Values", animated=True)
        self.mean_line_artist = self.ax.axhline(0, color="#ff6600", linestyle="--", 
                                               alpha=0.9, animated=True, 
                                               label="Average: 0.00")
        self.rolling_mean_line_artist = self.ax.axhline(0, color="#00ff66", linestyle=":", 
                                                       alpha=0.9, animated=True,
                                                       label="Last 10 Mean: 0.00")
        
        # Set up legend in fixed top left corner (not animated, so it won't be blitted)
        self.legend = self.ax.legend(loc='upper left', fontsize=8)
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
        if not MATPLOTLIB_AVAILABLE:
            return

        # Always store the data even if updates are paused
        self.data_buffer.append(value)

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
        # Note: value is already added to buffer in append_value method

        if len(self.data_buffer) == 0:
            self._plot_empty_chart()
            self.label_stats.setText("No data")
            return

        try:
            # Convert buffer to numpy array for calculations
            data_array = np.array(list(self.data_buffer))

            # Set up axes for blitting if not done yet
            if not self.axes_setup_complete:
                self._setup_axes_for_blitting()

            # Calculate statistics
            current_val = data_array[-1]
            mean_val = np.mean(data_array)

            # Calculate rolling mean of last 10 values
            last_10_data = data_array[-10:] if len(data_array) >= 10 else data_array
            rolling_mean = np.mean(last_10_data)

            # Create x-axis values
            x_values = np.arange(len(data_array))

            # Update y-axis limits if needed
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            y_margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            new_ylim = (min_val - y_margin, max_val + y_margin)
            
            if self.ax.get_ylim() != new_ylim:
                self.ax.set_ylim(new_ylim)
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.ax.bbox)

            # Restore background
            self.canvas.restore_region(self.background)

            # Update line data (with safety checks)
            if self.line_artist is not None:
                self.line_artist.set_data(x_values, data_array)
            
            # Update horizontal lines
            if self.mean_line_artist is not None:
                self.mean_line_artist.set_ydata([mean_val, mean_val])
                self.mean_line_artist.set_label(f"Average: {mean_val:.2f}")
                
            if self.rolling_mean_line_artist is not None:
                self.rolling_mean_line_artist.set_ydata([rolling_mean, rolling_mean])
                self.rolling_mean_line_artist.set_label(f"Last 10 Mean: {rolling_mean:.2f}")

            # Check if legend needs updating (less frequent to avoid performance hit)
            current_time = time.time()
            legend_needs_update = (current_time - self.last_legend_update) > self.legend_update_interval
            
            if legend_needs_update and self.legend is not None:
                # Update legend with new values in fixed top left corner
                self.legend.remove()
                self.legend = self.ax.legend(loc='upper left', fontsize=8)
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

            # Blit the updated artists
            self.canvas.blit(self.ax.bbox)

            # Update statistics label
            std_val = np.std(data_array)
            last_updated = datetime.now().strftime("%I:%M:%S %p")

            stats_text = (
                f"Current: {current_val:.2f} | Count: {len(data_array)}\n"
                f"Min: {min_val:.2f} | Max: {max_val:.2f}\n"
                f"Average: {mean_val:.2f} | Std: {std_val:.2f}\n"
                f"Last 10 Mean: {rolling_mean:.2f}\n"
                f"Updated: {last_updated}"
            )
            self.label_stats.setText(stats_text)

        except Exception as e:
            logging.error(f"Error updating line plot: {e}")
            # Fall back to non-blitted mode on error
            self.axes_setup_complete = False
            self._plot_empty_chart()
            self.label_stats.setText(f"Error: {str(e)}")

    def toggle_updates(self):
        """Toggle pause/resume state for plot updates."""
        self.updates_paused = not self.updates_paused
        
        if self.updates_paused:
            self.pause_button.setText("Resume")
            self.pause_button.setStyleSheet("""
                QPushButton {
                    background-color: #ff6600;
                    color: white;
                    border: 1px solid #555;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #ff7700;
                }
                QPushButton:pressed {
                    background-color: #ee5500;
                }
            """)
        else:
            self.pause_button.setText("Pause")
            self.pause_button.setStyleSheet("""
                QPushButton {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #555;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:pressed {
                    background-color: #2a2a2a;
                }
            """)
            # Trigger a plot update with the current data when resuming
            if len(self.data_buffer) > 0:
                self._append_value_internal(self.data_buffer[-1], "")

    def reset_chart(self):
        """Reset the chart by clearing all data."""
        self.data_buffer.clear()
        self.axes_setup_complete = False
        self.background = None
        self.line_artist = None
        self.mean_line_artist = None
        self.rolling_mean_line_artist = None
        self.legend = None
        self.last_legend_update = 0
        if MATPLOTLIB_AVAILABLE:
            self._plot_empty_chart()
        self.label_stats.setText("No data")

    def update_chart(self, data_array: Optional[np.ndarray], title: str = ""):
        """Update the chart with a new array of data (replaces current data).

        Args:
            data_array: 1D numpy array containing the data to plot
            title: Optional title for the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if data_array is None or data_array.size == 0:
            self.reset_chart()
            return

        # Clear buffer and add new data
        self.data_buffer.clear()

        # Take only the last max_length values if array is too long
        if len(data_array) > self.max_length:
            data_array = data_array[-self.max_length :]

        # Add all values to buffer
        for value in data_array:
            self.data_buffer.append(float(value))

        # Update the plot
        self._append_value_internal(self.data_buffer[-1], title)
