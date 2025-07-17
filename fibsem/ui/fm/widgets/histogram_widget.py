import logging
import time
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Histogram widget will be disabled.")


class HistogramWidget(QWidget):
    """Widget for displaying image intensity histograms."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_data = None
        self.last_update_time = 0
        self.update_interval = 0.5  # 500ms rate limiting

        self.initUI()
    
    def initUI(self):
        """Initialize the histogram widget UI."""
        self.setContentsMargins(0, 0, 0, 0)
        
        layout = QVBoxLayout()
               
        if not MATPLOTLIB_AVAILABLE:
            # Show error message if matplotlib is not available
            error_label = QLabel("Matplotlib not available.\nInstall with: pip install matplotlib", self)
            error_label.setStyleSheet("color: red; font-size: 10px;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)
        else:
            # Create matplotlib figure and canvas with napari-style dark theme
            self.figure = Figure(figsize=(4, 3), dpi=80)
            self.figure.patch.set_facecolor('#262930')  # napari dark background
            
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumSize(200, 150)
            
            # Create the subplot with dark styling
            self.ax = self.figure.add_subplot(111)

            self._apply_dark_theme()

            # Initial empty plot
            self._plot_empty_histogram()
            
            layout.addWidget(self.canvas)
            
            # Info label for statistics with napari-style colors
            self.label_stats = QLabel("No image selected", self)
            self.label_stats.setStyleSheet("QLabel { color: #bbbbbb; font-size: 10px; background-color: transparent; }")
            self.label_stats.setWordWrap(True)
            layout.addWidget(self.label_stats)
        
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
    
    def _plot_empty_histogram(self):
        """Plot an empty histogram with placeholder text."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.ax.clear()
        self._apply_dark_theme()
        
        self.ax.text(0.5, 0.5, 'No image selected', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, color='#bbbbbb')
        self.ax.set_xlabel('Intensity')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title('Image Histogram')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _apply_dark_theme(self):
        """Apply napari-style dark theme to the axes."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.ax.set_facecolor('#262930')  # napari dark background
        self.ax.tick_params(colors='white', which='both')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
    
    def update_histogram(self, image_data: Optional[np.ndarray], layer_name: str = ""):
        """Update the histogram with new image data using 1-second rate limiting.
        
        Args:
            image_data: 2D or 3D numpy array containing image data
            layer_name: Name of the image layer for display purposes
        """
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Skip update if widget is not visible to save CPU
        if not self.isVisible():
            return
            
        current_time = time.time()
        
        # Simple rate limiting: only update if 1 second has passed
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        self._update_histogram_internal(image_data, layer_name)

    def _update_histogram_internal(self, image_data: Optional[np.ndarray], layer_name: str = ""):
        """Internal method to actually update the histogram display."""
        if image_data is None or image_data.size == 0:
            self._plot_empty_histogram()
            self.label_stats.setText("No image data available")
            return
        
        try:
            # Handle different image dimensions
            if image_data.ndim == 3:
                # For 3D data (e.g., z-stack), use the middle slice or flatten
                if image_data.shape[0] == 1:
                    data_to_plot = image_data[0]  # Single z-slice
                else:
                    data_to_plot = image_data[image_data.shape[0] // 2]  # Middle slice
            elif image_data.ndim == 2:
                data_to_plot = image_data
            else:
                # Flatten higher dimensional data
                data_to_plot = image_data.reshape(-1)
                
            # Subsample large images for performance (use every nth pixel)
            if data_to_plot.size > 100000:  # 100k pixels
                step = max(1, data_to_plot.size // 50000)  # Max 50k samples
                data_to_plot = data_to_plot.flat[::step]
                
            # Remove any NaN or infinite values
            valid_data = data_to_plot[np.isfinite(data_to_plot)]
            
            if valid_data.size == 0:
                self._plot_empty_histogram()
                self.label_stats.setText("No valid pixel data")
                return
            
            # Calculate statistics
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            
            # Clear and plot histogram
            self.ax.clear()
            self._apply_dark_theme()
            
            # Calculate number of bins (max 64 for performance during live imaging)
            n_bins = min(64, int(np.sqrt(valid_data.size)))
            n_bins = max(16, n_bins)  # At least 16 bins
            
            self.ax.hist(valid_data.flatten(), bins=n_bins, 
                        alpha=0.8, color='#0f7aad', edgecolor='#ffffff')
            
            # Set labels and title
            self.ax.set_xlabel('Intensity')
            self.ax.set_ylabel('Frequency')
            title = f'Histogram: {layer_name}' if layer_name else 'Image Histogram'
            self.ax.set_title(title, fontsize=10)
            
            # Add mean line
            self.ax.axvline(float(mean_val), color='#ff6600', linestyle='--', alpha=0.9, label=f'Mean: {mean_val:.2f}')
            legend = self.ax.legend(fontsize=8)
            legend.get_frame().set_facecolor('#262930')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update statistics label
            stats_text = (f"Min: {min_val:.2f} | Max: {max_val:.2f}\n"
                         f"Mean: {mean_val:.2f} | Std: {std_val:.2f}\n"
                         f"Pixels: {valid_data.size:,}")
            self.label_stats.setText(stats_text)
            
            # Store current data for potential future use
            self.current_data = image_data
            
        except Exception as e:
            logging.error(f"Error updating histogram: {e}")
            self._plot_empty_histogram()
            self.label_stats.setText(f"Error: {str(e)}")
    
    def clear_histogram(self):
        """Clear the histogram display."""
        if MATPLOTLIB_AVAILABLE:
            self._plot_empty_histogram()
        self.label_stats.setText("No image selected")
        self.current_data = None