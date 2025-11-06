import logging
from typing import List, Optional, TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from fibsem import conversions
from fibsem.structures import FibsemImage, FibsemStagePosition, Point
from fibsem.imaging.tiled import reproject_stage_positions_onto_image2
from fibsem.ui.napari.utilities import is_inside_image_bounds
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Minimap plot widget will be disabled.")

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI


class MinimapPlotWidget(QWidget):
    """Widget for displaying a minimap with lamella positions and current stage position."""

    def __init__(self, parent: Optional['AutoLamellaUI'] = None):
        super().__init__(parent)
        self.image: Optional[FibsemImage] = None
        self.lamella_positions: List[FibsemStagePosition] = []
        self.current_position: Optional[FibsemStagePosition] = None
        self.grid_positions: Optional[List[FibsemStagePosition]] = None
        self.selected_name: Optional[str] = None
        self.parent_widget = parent

        # Store the original view limits for reset
        self.original_xlim = None
        self.original_ylim = None

        # Cache for image orientation
        self.cached_orientation: Optional[str] = None

        # Display options
        self.show_names = False
        self.show_grid_boundary = True
        self.show_grid_positions = True
        self.show_current_fov = True
        self.show_rotation_reference = False

        # Field of view settings (in meters)
        self.fov_width = 150e-6  # 150 microns default FOV width
        self.grid_boundary_radius = 1000e-6  # 1000 microns default grid boundary
        self.default_zoom_box = 2000e-6  # 2000 microns default zoom box

        # Pan state
        self.pan_active = False
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_xlim = None
        self.pan_ylim = None

        # Artist references for incremental updates
        # Store matplotlib artist objects for current position to enable incremental updates
        self.current_position_artists: dict = {
            'marker': None,          # The + marker (Line2D)
            'text': None,            # Text label (Text)
            'fov_rect': None,        # FOV rectangle (Rectangle patch)
            'rotation_triangle': None # Rotation indicator (RegularPolygon patch)
        }

        self.initUI()

    def initUI(self):
        """Initialize the minimap plot widget UI."""
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
            self.figure = Figure(figsize=(6, 6), dpi=80)
            self.figure.patch.set_facecolor("#262930")  # napari dark background

            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumSize(300, 300)

            # Create the subplot with dark styling
            self.ax = self.figure.add_subplot(111)

            self._apply_dark_theme()

            # Enable scroll wheel zoom (without toolbar)
            self.canvas.mpl_connect('scroll_event', self._on_scroll)

            # Enable click handlers (double-click, alt+click, pan)
            self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
            self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

            # Initial empty plot
            self._plot_empty_minimap()

            layout.addWidget(self.canvas)

            # Button layout
            button_layout = QHBoxLayout()

            # Load Image button
            self.load_image_button = QToolButton(self)
            self.load_image_button.setText("Load Image")
            self.load_image_button.setToolTip("Load an image file for the minimap")
            self.load_image_button.setStyleSheet("""
                QToolButton {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #555;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QToolButton:hover {
                    background-color: #4a4a4a;
                }
                QToolButton:pressed {
                    background-color: #2a2a2a;
                }
            """)
            self.load_image_button.clicked.connect(self._on_load_image_clicked)
            button_layout.addWidget(self.load_image_button)

            # Refresh button
            self.refresh_button = QPushButton("Refresh", self)
            self.refresh_button.setStyleSheet("""
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
            self.refresh_button.clicked.connect(self.update_minimap)
            button_layout.addWidget(self.refresh_button)

            # Reset Zoom button
            self.reset_zoom_button = QPushButton("Reset Zoom", self)
            self.reset_zoom_button.setStyleSheet("""
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
            self.reset_zoom_button.clicked.connect(self.reset_zoom)
            button_layout.addWidget(self.reset_zoom_button)

            # Clear button
            self.clear_button = QPushButton("Clear", self)
            self.clear_button.setStyleSheet("""
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
            self.clear_button.clicked.connect(self.clear_minimap)
            button_layout.addWidget(self.clear_button)

            layout.addLayout(button_layout)

            # Checkbox options layout - Row 1
            checkbox_layout1 = QHBoxLayout()

            # Show names checkbox
            self.show_names_checkbox = QCheckBox("Show Names", self)
            self.show_names_checkbox.setChecked(self.show_names)
            self.show_names_checkbox.setStyleSheet("""
                QCheckBox {
                    color: #bbbbbb;
                    font-size: 10px;
                }
                QCheckBox::indicator {
                    width: 13px;
                    height: 13px;
                }
            """)
            self.show_names_checkbox.stateChanged.connect(self._on_show_names_changed)
            checkbox_layout1.addWidget(self.show_names_checkbox)

            # Show current FOV checkbox
            self.show_current_fov_checkbox = QCheckBox("Current FOV", self)
            self.show_current_fov_checkbox.setChecked(self.show_current_fov)
            self.show_current_fov_checkbox.setStyleSheet("""
                QCheckBox {
                    color: #bbbbbb;
                    font-size: 10px;
                }
                QCheckBox::indicator {
                    width: 13px;
                    height: 13px;
                }
            """)
            self.show_current_fov_checkbox.stateChanged.connect(self._on_show_current_fov_changed)
            checkbox_layout1.addWidget(self.show_current_fov_checkbox)

            checkbox_layout1.addStretch()
            layout.addLayout(checkbox_layout1)

            # Checkbox options layout - Row 2 (Grid options)
            checkbox_layout2 = QHBoxLayout()

            # Show grid positions checkbox
            self.show_grid_positions_checkbox = QCheckBox("Grid Positions", self)
            self.show_grid_positions_checkbox.setChecked(self.show_grid_positions)
            self.show_grid_positions_checkbox.setStyleSheet("""
                QCheckBox {
                    color: #bbbbbb;
                    font-size: 10px;
                }
                QCheckBox::indicator {
                    width: 13px;
                    height: 13px;
                }
            """)
            self.show_grid_positions_checkbox.stateChanged.connect(self._on_show_grid_positions_changed)
            checkbox_layout2.addWidget(self.show_grid_positions_checkbox)

            # Show grid boundary checkbox
            self.show_grid_boundary_checkbox = QCheckBox("Grid Boundary", self)
            self.show_grid_boundary_checkbox.setChecked(self.show_grid_boundary)
            self.show_grid_boundary_checkbox.setStyleSheet("""
                QCheckBox {
                    color: #bbbbbb;
                    font-size: 10px;
                }
                QCheckBox::indicator {
                    width: 13px;
                    height: 13px;
                }
            """)
            self.show_grid_boundary_checkbox.stateChanged.connect(self._on_show_grid_boundary_changed)
            checkbox_layout2.addWidget(self.show_grid_boundary_checkbox)

            checkbox_layout2.addStretch()
            layout.addLayout(checkbox_layout2)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _plot_empty_minimap(self):
        """Plot an empty minimap with placeholder text."""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.ax.clear()
        self._apply_dark_theme()

        self.ax.text(
            0.5,
            0.5,
            "No minimap image loaded",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=12,
            color="#bbbbbb",
        )
        self.ax.set_title("Minimap")
        self.ax.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()

    def _apply_dark_theme(self):
        """Apply napari-style dark theme to the axes."""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.ax.set_facecolor("black")  # Black background for plot area
        self.ax.tick_params(colors="white", which="both")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

    def set_minimap_image(self, image: FibsemImage):
        """Set the base minimap image.

        Args:
            image: The FibsemImage to use as the minimap base
        """
        self.image = image
        # Reset zoom limits when new image is loaded
        self.original_xlim = None
        self.original_ylim = None
        # Clear cached orientation since we have a new image
        self.cached_orientation = None
        self.update_minimap()

    def set_lamella_positions(self, positions: List[FibsemStagePosition]):
        """Set the lamella positions to display on the minimap.

        Args:
            positions: List of FibsemStagePosition objects for each lamella
        """
        self.lamella_positions = positions
        self.update_minimap()

    def set_current_position(self, position: Optional[FibsemStagePosition], force_full_redraw: bool = False, auto_zoom: bool = False):
        """Set the current stage position to display on the minimap.

        Args:
            position: FibsemStagePosition object for the current stage position
            force_full_redraw: If True, perform a full redraw instead of incremental update
            auto_zoom: If True, automatically zoom to center on the current position
        """
        self.current_position = position

        # Use incremental update if we have an image and artists are initialized
        if not force_full_redraw and self.image is not None and MATPLOTLIB_AVAILABLE:
            self._update_current_position_incremental()
        else:
            self.update_minimap()

        # Auto-zoom to 1000um box around current position if requested
        if auto_zoom and position is not None and self.image is not None:
            self._zoom_to_current_position()

    def set_grid_positions(self, positions: Optional[List[FibsemStagePosition]]):
        """Set grid positions to display on the minimap.

        Args:
            positions: List of FibsemStagePosition objects for grid positions
        """
        self.grid_positions = positions
        self.update_minimap()

    def set_selected_name(self, name: Optional[str]):
        """Set the selected lamella name to highlight in lime color.

        Args:
            name: Name of the selected lamella position to highlight (None to clear selection)
        """
        self.selected_name = name
        self.update_minimap()

    def set_fov_width(self, width: float):
        """Set the field of view width for the current position indicator.

        Args:
            width: FOV width in meters (e.g., 150e-6 for 150 microns)
        """
        self.fov_width = width
        self.update_minimap()

    def set_grid_boundary_radius(self, radius: float):
        """Set the grid boundary circle radius.

        Args:
            radius: Radius in meters (e.g., 1000e-6 for 1000 microns)
        """
        self.grid_boundary_radius = radius
        self.update_minimap()

    def set_default_zoom_box(self, size: float):
        """Set the default zoom box size when current position is set.

        Args:
            size: Box size in meters (e.g., 1000e-6 for 1000 microns)
        """
        self.default_zoom_box = size

    def update_minimap(self):
        """Update the minimap plot with current data."""
        if not MATPLOTLIB_AVAILABLE:
            return

        if self.image is None:
            self._plot_empty_minimap()
            self.label_info.setText("No minimap image loaded")
            return

        try:
            # Save current zoom state before clearing
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()

            # Clear the current axes (this will remove all artists)
            self.ax.clear()

            # Reset current position artist references since ax.clear() removed them
            self._clear_current_position_artists()

            # Display the base image
            self.ax.set_title("Minimap")
            self.ax.imshow(self.image.data, cmap="gray", origin="upper")

            # Plot current position separately (if exists)
            if self.current_position is not None:
                current_points = reproject_stage_positions_onto_image2(
                    image=self.image, positions=[self.current_position]
                )
                if len(current_points) > 0:
                    pt = current_points[0]
                    if is_inside_image_bounds(
                        (pt.y, pt.x),
                        (self.image.data.shape[0], self.image.data.shape[1]),
                    ):
                        if pt.name is None:
                            pt.name = "Current Position"
                        self._plot_current_position(pt.x, pt.y, name=pt.name, color="yellow")

            # Plot lamella positions
            if len(self.lamella_positions) > 0:
                points = reproject_stage_positions_onto_image2(
                    image=self.image, positions=self.lamella_positions
                )

                for i, pt in enumerate(points):
                    # Skip points outside image bounds
                    if not is_inside_image_bounds(
                        (pt.y, pt.x),
                        (self.image.data.shape[0], self.image.data.shape[1]),
                    ):
                        continue

                    if pt.name is None:
                        pt.name = f"Position {i:02d}"

                    # Determine color - lime for selected, cyan for others
                    if self.selected_name is not None and pt.name == self.selected_name:
                        c = "lime"
                    else:
                        c = "cyan"

                    # Plot the lamella position
                    self._plot_point_with_label(pt.x, pt.y, pt.name, color=c)

            # Plot grid positions (if enabled)
            if self.show_grid_positions and self.grid_positions is not None and len(self.grid_positions) > 0:
                points = reproject_stage_positions_onto_image2(
                    image=self.image, positions=self.grid_positions
                )

                for i, pt in enumerate(points):
                    # Skip points outside image bounds
                    if not is_inside_image_bounds(
                        (pt.y, pt.x),
                        (self.image.data.shape[0], self.image.data.shape[1]),
                    ):
                        continue

                    if pt.name is None:
                        pt.name = f"Grid {i:02d}"

                    # Plot the grid position with boundary circle
                    self._plot_grid_position(pt.x, pt.y, pt.name, color="red")

            # Add scalebar if available
            scalebar = self._create_scalebar()
            if scalebar is not None:
                self.ax.add_artist(scalebar)

            # Add orientation label in bottom left corner (in axes coordinates, not data coordinates)
            self._add_orientation_label()

            # Turn off axis
            self.ax.axis("off")

            # Store original view limits for reset zoom functionality (only first time)
            if self.original_xlim is None:
                # Get the default view limits after plotting
                default_xlim = self.ax.get_xlim()
                default_ylim = self.ax.get_ylim()
                self.original_xlim = default_xlim
                self.original_ylim = default_ylim

            # Update the figure
            self.figure.tight_layout()

            # Restore the saved zoom state (preserves user's zoom level)
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)

            self.canvas.draw()

        except Exception as e:
            logging.error(f"Error updating minimap: {e}")
            import traceback
            traceback.print_exc()
            self._plot_empty_minimap()

    def _update_current_position_incremental(self):
        """Update only the current position marker and related overlays incrementally."""
        if not MATPLOTLIB_AVAILABLE or self.image is None:
            return

        try:


            # Remove old current position artists
            self._clear_current_position_artists()

            # If no current position, just redraw and return
            if self.current_position is None:
                self.canvas.draw_idle()
                return

            # Reproject current position onto image
            points = reproject_stage_positions_onto_image2(
                image=self.image, positions=[self.current_position]
            )

            if len(points) == 0:
                self.canvas.draw_idle()
                return

            pt = points[0]

            # Skip if point is outside image bounds
            if not is_inside_image_bounds(
                (pt.y, pt.x),
                (self.image.data.shape[0], self.image.data.shape[1]),
            ):
                self.canvas.draw_idle()
                return

            # Set point name
            if pt.name is None:
                pt.name = "Current Position"

            # Plot the current position with all overlays
            self._plot_current_position(pt.x, pt.y, name=pt.name, color="yellow")

            # Use draw_idle for better performance (deferred draw)
            self.canvas.draw_idle()

        except Exception as e:
            logging.error(f"Error updating current position incrementally: {e}")
            # Fallback to full redraw on error
            self.update_minimap()

    def _clear_current_position_artists(self):
        """Remove all current position artists from the plot."""
        for key, artist in self.current_position_artists.items():
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass  # Artist may already be removed
                self.current_position_artists[key] = None

    def _create_fov_rectangle(self, pt_x: float, pt_y: float, color: str = 'yellow'):
        """Create a field of view rectangle indicator.

        Args:
            pt_x: X coordinate in image pixels
            pt_y: Y coordinate in image pixels
            color: Color for the rectangle

        Returns:
            Rectangle patch object or None if FOV display is disabled or metadata unavailable
        """
        if not self.show_current_fov or self.image is None:
            return None

        # Field of view in meters (width)
        fov_width = self.fov_width

        # Aspect ratio: width=1536, height=1024
        aspect_ratio = 1024 / 1536  # height / width
        fov_height = fov_width * aspect_ratio

        # Convert to pixels
        if self.image.metadata is not None and self.image.metadata.pixel_size is not None:
            pixel_size = self.image.metadata.pixel_size.x
            fov_width_px = fov_width / pixel_size
            fov_height_px = fov_height / pixel_size

            # Create rectangle (centered on point)
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (pt_x - fov_width_px / 2, pt_y - fov_height_px / 2),
                fov_width_px,
                fov_height_px,
                color=color,
                fill=False,
                linewidth=2,
                linestyle='--',
                alpha=0.7
            )
            return rect

        return None

    def _plot_point_with_label(self, pt_x: float, pt_y: float, name: str, color: str = 'cyan',
                               marker: str = '+', marker_size: int = 15, show_label: Optional[bool] = None):
        """Plot a point marker with optional text label.

        Args:
            pt_x: X coordinate in image pixels
            pt_y: Y coordinate in image pixels
            name: Name/label for the point
            color: Color for the marker and text
            marker: Marker style (default '+')
            marker_size: Size of the marker
            show_label: Whether to show the text label (defaults to self.show_names)

        Returns:
            Tuple of (line artist, text artist or None)
        """
        # Plot the point marker
        line, = self.ax.plot(
            pt_x,
            pt_y,
            ms=marker_size,
            c=color,
            marker=marker,
            markeredgewidth=2,
            label=name,
            alpha=0.7
        )

        # Plot the text label if enabled
        text_artist = None
        if show_label is None:
            show_label = self.show_names

        if show_label:
            text_artist = self.ax.text(
                pt_x + 10,
                pt_y - 10,
                name,
                fontsize=8,
                color=color,
                alpha=0.75,
            )

        return line, text_artist

    def _create_grid_boundary_circle(self, pt_x: float, pt_y: float, color: str = 'red'):
        """Create a grid boundary circle indicator.

        Args:
            pt_x: X coordinate in image pixels
            pt_y: Y coordinate in image pixels
            color: Color for the circle

        Returns:
            Circle patch object or None if grid boundary display is disabled or metadata unavailable
        """
        if not self.show_grid_boundary or self.image is None:
            return None

        # Grid area radius in meters
        grid_radius = self.grid_boundary_radius

        # Convert to pixels
        if self.image.metadata is not None and self.image.metadata.pixel_size is not None:
            pixel_size = self.image.metadata.pixel_size.x
            grid_radius_px = grid_radius / pixel_size

            # Create circle
            from matplotlib.patches import Circle
            circle = Circle(
                (pt_x, pt_y),
                grid_radius_px,
                color=color,
                fill=False,
                linewidth=2,
                linestyle='-',
                alpha=0.7
            )
            return circle

        return None

    def _plot_current_position(self, pt_x: float, pt_y: float, name: str = "Current Position",
                               color: str = 'yellow'):
        """Plot the current position marker with all associated overlays (FOV, rotation).

        Args:
            pt_x: X coordinate in image pixels
            pt_y: Y coordinate in image pixels
            name: Name/label for the position
            color: Color for the marker and overlays

        Returns:
            None (stores artist references in self.current_position_artists)
        """
        # Plot the position marker (with label if show_names is enabled)
        line, text_artist = self._plot_point_with_label(pt_x, pt_y, name, color=color)
        self.current_position_artists['marker'] = line
        self.current_position_artists['text'] = text_artist

        # Draw field of view indicator if enabled
        fov_rect = self._create_fov_rectangle(pt_x, pt_y, color=color)
        if fov_rect is not None:
            self.ax.add_patch(fov_rect)
            self.current_position_artists['fov_rect'] = fov_rect

        # Add rotation reference indicator if enabled
        triangle = self._create_rotation_triangle(pt_x, pt_y, color=color)
        if triangle is not None:
            self.ax.add_patch(triangle)
            self.current_position_artists['rotation_triangle'] = triangle

    def _plot_grid_position(self, pt_x: float, pt_y: float, name: str, color: str = 'red'):
        """Plot a grid position marker with boundary circle overlay.

        Args:
            pt_x: X coordinate in image pixels
            pt_y: Y coordinate in image pixels
            name: Name/label for the position
            color: Color for the marker and boundary circle
        """
        # Plot the position marker (with label if show_names is enabled)
        self._plot_point_with_label(pt_x, pt_y, name, color=color)

        # Draw grid boundary circle if enabled
        grid_circle = self._create_grid_boundary_circle(pt_x, pt_y, color=color)
        if grid_circle is not None:
            self.ax.add_patch(grid_circle)

    def _add_orientation_label(self):
        """Add an orientation label in the bottom left corner of the plot.

        The label uses axes coordinates (transform=self.ax.transAxes) so it stays
        in the same position regardless of zoom level. The orientation is cached
        to avoid recomputing on every update.
        """
        if self.image is None or self.image.metadata is None:
            return

        # Use cached orientation if available
        if self.cached_orientation is None:
            # Get orientation from metadata using microscope.get_stage_orientation
            orientation = "Unknown"
            try:
                if (self.parent_widget is not None and
                    hasattr(self.parent_widget, 'microscope') and
                    self.parent_widget.microscope is not None and
                    self.image.metadata.microscope_state is not None and
                    self.image.metadata.microscope_state.stage_position is not None):

                    stage_position = self.image.metadata.microscope_state.stage_position
                    orientation = self.parent_widget.microscope.get_stage_orientation(stage_position)

            except Exception as e:
                # If we can't get orientation, just show "Unknown"
                logging.debug(f"Could not get orientation: {e}")

            # Cache the orientation
            self.cached_orientation = orientation

        # Add text in bottom left corner using axes coordinates
        # transform=self.ax.transAxes means (0,0) is bottom-left, (1,1) is top-right
        # This stays fixed regardless of zoom/pan
        self.ax.text(
            0.02, 0.02,  # 2% from left, 2% from bottom
            f"Orientation: {self.cached_orientation}",
            transform=self.ax.transAxes,
            fontsize=9,
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6, edgecolor='white', linewidth=1),
            verticalalignment='bottom',
            horizontalalignment='left',
            zorder=1000  # High z-order to keep it on top
        )

    def _create_scalebar(self):
        """Create a scalebar for the minimap image.

        Returns:
            ScaleBar artist object or None if not available
        """
        if self.image is None or self.image.metadata is None or self.image.metadata.pixel_size is None:
            return None

        try:
            from matplotlib_scalebar.scalebar import ScaleBar

            scalebar = ScaleBar(
                dx=self.image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )
            return scalebar
        except Exception:
            # Silently skip if scalebar fails (e.g., matplotlib_scalebar not installed)
            return None

    def _create_rotation_triangle(self, pt_x: float, pt_y: float, color: str = 'yellow'):
        """Create a rotation reference triangle indicator.

        Args:
            pt_x: X coordinate in image pixels
            pt_y: Y coordinate in image pixels
            color: Color for the triangle edge

        Returns:
            RegularPolygon patch object or None if rotation data unavailable
        """
        if not self.show_rotation_reference or self.image is None or self.image.metadata is None:
            return None

        try:
            if (self.image.metadata.system is not None and
                self.image.metadata.microscope_state is not None and
                self.image.metadata.microscope_state.stage_position is not None):

                reference_rotation = self.image.metadata.system.stage.rotation_reference
                current_rotation = self.image.metadata.microscope_state.stage_position.r

                if reference_rotation is not None and current_rotation is not None:
                    # Calculate rotation difference
                    import numpy as np
                    reference_rotation_rad = np.deg2rad(reference_rotation)
                    rotation_diff = current_rotation - reference_rotation_rad

                    # Draw rotation indicator triangle
                    from matplotlib.patches import RegularPolygon

                    triangle_size = 30
                    triangle = RegularPolygon(
                        (pt_x, pt_y),
                        3,  # 3 sides = triangle
                        radius=triangle_size,
                        orientation=(rotation_diff + np.radians(180)),
                        facecolor='none',
                        edgecolor=color,
                        linewidth=2,
                        alpha=0.8
                    )
                    return triangle
        except Exception:
            pass  # Silently skip if metadata is incomplete

        return None

    def _on_show_names_changed(self, state):
        """Handle show names checkbox state change."""
        self.show_names = bool(state)
        self.update_minimap()

    def _on_show_grid_boundary_changed(self, state):
        """Handle show grid boundary checkbox state change."""
        self.show_grid_boundary = bool(state)
        self.update_minimap()

    def _on_show_grid_positions_changed(self, state):
        """Handle show grid positions checkbox state change."""
        self.show_grid_positions = bool(state)
        self.update_minimap()

    def _on_show_current_fov_changed(self, state):
        """Handle show current FOV checkbox state change."""
        self.show_current_fov = bool(state)
        self.update_minimap()

    def _on_scroll(self, event):
        """Handle mouse scroll events for zooming.

        Args:
            event: Matplotlib scroll event
        """
        if not MATPLOTLIB_AVAILABLE or event.inaxes != self.ax:
            return

        # Get the current x and y limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Get event location
        xdata = event.xdata
        ydata = event.ydata

        # Increased zoom factor for faster zooming
        base_scale = 1.5

        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Zoom out
            scale_factor = base_scale
        else:
            return

        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        # Center zoom on mouse position
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        new_xlim = (xdata - new_width * (1 - relx), xdata + new_width * relx)
        new_ylim = (ydata - new_height * (1 - rely), ydata + new_height * rely)

        # Set new limits
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        # Use blit for faster redrawing
        self.canvas.draw()

    def _on_mouse_press(self, event):
        """Handle mouse button press for double-click, alt+click, and pan.

        Args:
            event: Matplotlib mouse button press event
        """
        if not MATPLOTLIB_AVAILABLE or event.inaxes != self.ax:
            return

        # Handle left mouse button
        if event.button == 1:
            # Check if Alt key is pressed using Qt's keyboard modifiers
            modifiers = QApplication.keyboardModifiers()
            alt_pressed = modifiers == Qt.KeyboardModifier.AltModifier

            # Handle double-click for movement
            if event.dblclick:
                self._handle_double_click(event)

            # Handle Alt+click to add lamella
            elif alt_pressed:
                self._handle_alt_click(event)

            # Handle regular click for panning
            else:
                self._handle_pan_start(event)

    def _validate_minimap_state(self) -> bool:
        """Validate that the minimap has required data for coordinate conversion.

        Returns:
            bool: True if minimap state is valid, False otherwise
        """
        return (self.image is not None and
                self.image.metadata is not None and
                self.parent_widget is not None)

    def _check_stage_orientation_match(self) -> bool:
        """Check if current stage orientation matches the minimap image orientation.

        Returns:
            bool: True if orientations match, False otherwise

        Raises:
            RuntimeError: If minimap state is invalid (should be validated before calling)
        """
        # Type guards - should be validated by _validate_minimap_state() before calling
        if (self.image is None or
            self.image.metadata is None or
            self.parent_widget is None or
            self.parent_widget.microscope is None):
            raise RuntimeError("Invalid minimap state: missing required data")

        image_stage_position = self.image.metadata.stage_position
        image_stage_orientation = self.parent_widget.microscope.get_stage_orientation(image_stage_position)
        current_stage_orientation = self.parent_widget.microscope.get_stage_orientation()
        return image_stage_orientation == current_stage_orientation

    def _handle_double_click(self, event):
        """Handle double-click event to move stage to clicked position.

        Args:
            event: Matplotlib mouse button press event
        """
        if not self._validate_minimap_state():
            return

        if event.xdata is None or event.ydata is None:
            return

        # Type guard for parent_widget
        if self.parent_widget is None or self.parent_widget.microscope is None or self.parent_widget.movement_widget is None:
            raise RuntimeError("Invalid minimap state: missing parent_widget")

        logging.info(f"Double-click at image coordinates: x={event.xdata:.2f}, y={event.ydata:.2f}")

        try:
            stage_position = self._event_to_stage_position(event)
            logging.info(f"Double click at microscope stage coordinates: {stage_position.pretty}")

            # Check if the current stage orientation and image orientation are the same
            if not self._check_stage_orientation_match():
                logging.warning("Current stage orientation and image orientation do not match. Movement may be inaccurate.")
                return

            self.parent_widget.movement_widget.move_to_position(stage_position)

        except Exception as e:
            logging.warning(f"An error occurred when calculating stable movement...{e}")

    def _handle_alt_click(self, event):
        """Handle Alt+click event to add a new lamella position.

        Args:
            event: Matplotlib mouse button press event
        """
        if not self._validate_minimap_state():
            return

        if event.xdata is None or event.ydata is None:
            return

        # Type guard for parent_widget
        if self.parent_widget is None:
            raise RuntimeError("Invalid minimap state: missing parent_widget")

        logging.info(f"Alt+click at image coordinates: x={event.xdata:.2f}, y={event.ydata:.2f}")
        logging.info(f"Alt+click at display coordinates: x={event.x}, y={event.y}")

        try:
            stage_position = self._event_to_stage_position(event)
            logging.info(f"Alt+click at microscope stage coordinates: {stage_position.pretty}")

            # Check if the current stage orientation and image orientation are the same
            if not self._check_stage_orientation_match():
                logging.warning("Current stage orientation and image orientation do not match. Position may be inaccurate.")
                return

            self.parent_widget.add_new_lamella(stage_position)

        except Exception as e:
            logging.warning(f"An error occurred when adding new lamella...{e}")

    def _handle_pan_start(self, event):
        """Handle regular click to start panning.

        Args:
            event: Matplotlib mouse button press event
        """
        self.pan_active = True
        self.pan_start_x = event.xdata
        self.pan_start_y = event.ydata
        self.pan_xlim = self.ax.get_xlim()
        self.pan_ylim = self.ax.get_ylim()
        # Change cursor to hand
        self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _event_to_stage_position(self, event) -> FibsemStagePosition:
        """Convert a matplotlib click event to a microscope stage position.

        Args:
            event: Matplotlib mouse button press event with xdata and ydata

        Returns:
            FibsemStagePosition: The corresponding stage position for the clicked location
        """
        if self.image is None or self.image.metadata is None:
            raise ValueError("Minimap image or metadata is not set.")

        if self.parent_widget is None or self.parent_widget.microscope is None:
            raise ValueError("Parent widget or microscope is not set.")

        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=event.xdata, y=event.ydata),
            image=self.image.data,
            pixelsize=self.image.metadata.pixel_size.x,
        )

        stage_position = self.parent_widget.microscope.project_stable_move(
                    dx=point.x, dy=point.y,
                    beam_type=self.image.metadata.beam_type,
                    base_position=self.image.metadata.stage_position)

        return stage_position

    def _on_mouse_release(self, event):
        """Handle mouse button release for pan end.

        Args:
            event: Matplotlib mouse button release event
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if event.button == 1:
            self.pan_active = False
            self.pan_start_x = None
            self.pan_start_y = None
            self.pan_xlim = None
            self.pan_ylim = None
            # Reset cursor
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_mouse_move(self, event):
        """Handle mouse move for panning.

        Args:
            event: Matplotlib mouse motion event
        """
        if not MATPLOTLIB_AVAILABLE or not self.pan_active:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.pan_start_x is None or self.pan_start_y is None:
            return

        # Calculate the drag distance from current limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        dx = event.xdata - self.pan_start_x
        dy = event.ydata - self.pan_start_y

        # Update the view limits (inverse direction for intuitive dragging)
        new_xlim = (cur_xlim[0] - dx, cur_xlim[1] - dx)
        new_ylim = (cur_ylim[0] - dy, cur_ylim[1] - dy)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        # Update the pan start position for next move
        self.pan_start_x = event.xdata - dx
        self.pan_start_y = event.ydata - dy

        # Fast redraw
        self.canvas.draw()

    def _zoom_to_current_position(self):
        """Zoom to a 1000um box around the current position."""
        if not MATPLOTLIB_AVAILABLE or self.current_position is None or self.image is None:
            return

        try:

            # Reproject current position to image coordinates
            points = reproject_stage_positions_onto_image2(
                image=self.image, positions=[self.current_position]
            )

            if len(points) == 0:
                return

            pt = points[0]

            # Calculate zoom box in pixels
            if self.image.metadata is not None and self.image.metadata.pixel_size is not None:
                pixel_size = self.image.metadata.pixel_size.x
                box_size = self.default_zoom_box
                box_size_px = box_size / pixel_size / 2  # Half size for centering

                # Set zoom to center on current position with 1000um box
                new_xlim = (pt.x - box_size_px, pt.x + box_size_px)
                new_ylim = (pt.y + box_size_px, pt.y - box_size_px)

                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)

                # Update original limits so Reset Zoom returns to this view
                self.original_xlim = new_xlim
                self.original_ylim = new_ylim

                self.canvas.draw()

        except Exception as e:
            logging.debug(f"Could not zoom to current position: {e}")

    def reset_zoom(self):
        """Reset the zoom to center on the current position, or show full image if no current position."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # If there's a current position, zoom to it
        if self.current_position is not None and self.image is not None:
            self._zoom_to_current_position()
        # Otherwise, reset to full view
        elif self.original_xlim is not None and self.original_ylim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.canvas.draw()

    def clear_minimap(self):
        """Clear all positions and reset the minimap."""
        self.lamella_positions = []
        self.current_position = None
        self.grid_positions = None
        self.original_xlim = None
        self.original_ylim = None
        self.update_minimap()

    def _on_load_image_clicked(self):
        """Handle load image button click to select and load an image file."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Determine starting directory from experiment path if available
        start_dir = ""
        if (self.parent_widget is not None and self.parent_widget.experiment is not None):
            start_dir = str(self.parent_widget.experiment.path)

        # Open file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Minimap Image",
            start_dir,
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;TIFF Files (*.tif *.tiff);;All Files (*.*)"
        )

        if not file_path:
            # User cancelled the dialog
            return

        try:
            # Load the image using FibsemImage.load
            image = FibsemImage.load(file_path)

            # Set the minimap image
            self.set_minimap_image(image)

            logging.info(f"Loaded minimap image from: {file_path}")
            logging.info(f"Image shape: {image.data.shape}, dtype: {image.data.dtype}")

        except Exception as e:
            logging.error(f"Error loading minimap image: {e}")
            import traceback
            traceback.print_exc()
