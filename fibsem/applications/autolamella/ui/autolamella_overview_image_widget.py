"""Widget for generating final overview images with customizable markers and text."""

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import DefectType, Experiment
from fibsem.constants import DATETIME_DISPLAY_SHORT as DATETIME_DISPLAY
from fibsem.imaging.tiled import (
    DEFECT_FAILURE_COLOUR,
    DEFECT_REWORK_COLOUR,
    plot_minimap,
)
from fibsem.structures import FibsemImage
from fibsem.ui.tokens import (
    NEUTRAL_400,
    NEUTRAL_750,
    NEUTRAL_800,
    NEUTRAL_850,
    SURFACE_COLOR,
)
from fibsem.ui.widgets.custom_widgets import (
    IntegerValueSpinBox,
    TitledPanel,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


# Stylesheet constants
LINEEDIT_STYLESHEET = f"""
    QLineEdit {{
        background-color: {NEUTRAL_800};
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
    }}
    QLineEdit:hover {{
        background-color: {NEUTRAL_750};
    }}
"""

SPINBOX_STYLESHEET = f"""
    QSpinBox {{
        background-color: {NEUTRAL_800};
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
    }}
    QSpinBox:hover {{
        background-color: {NEUTRAL_750};
    }}
"""

COMBOBOX_STYLESHEET = f"""
    QComboBox {{
        background-color: {NEUTRAL_800};
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
        min-width: 150px;
    }}
    QComboBox:hover {{
        background-color: {NEUTRAL_750};
    }}
    QComboBox::drop-down {{
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background-color: {NEUTRAL_800};
        color: white;
        selection-background-color: {NEUTRAL_750};
    }}
"""

BUTTON_STYLESHEET = f"""
    QPushButton {{
        background-color: {NEUTRAL_800};
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
    }}
    QPushButton:hover {{
        background-color: {NEUTRAL_750};
    }}
    QPushButton:pressed {{
        background-color: {NEUTRAL_850};
    }}
"""

CHECKBOX_STYLESHEET = f"""
    QCheckBox {{
        color: white;
        font-size: 10px;
    }}
    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border: 1px solid #555;
        background-color: {NEUTRAL_800};
    }}
    QCheckBox::indicator:checked {{
        background-color: #4a90e2;
    }}
"""


ZOOM_SCALE_FACTOR = 1.2


class OverviewImageWidget(QWidget):
    """Widget for generating final overview images with customizable markers and text.

    This widget allows users to:
    - Select an overview image file
    - Specify output filename
    - Customize marker colors, text size, and names
    - Preview the generated image
    - Export the final image
    """

    def __init__(self, parent: Optional["AutoLamellaUI"] = None):
        """Initialize the overview image widget.

        Args:
            parent: Parent AutoLamellaUI widget (optional)
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.experiment: Optional["Experiment"] = None
        self.overview_image: Optional["FibsemImage"] = None
        self.current_figure: Optional[Figure] = None
        self.marker_color = QColor("cyan")
        self.stage_positions = []
        self._scroll_cid: Optional[int] = None
        self._pan_press_cid: Optional[int] = None
        self._pan_release_cid: Optional[int] = None
        self._pan_motion_cid: Optional[int] = None
        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None
        self._initial_axes_limits = []

        self.initUI()

    def initUI(self):
        """Initialize the widget UI components."""

        # Display Options Panel
        display_content = QWidget()
        display_layout = QFormLayout(display_content)
        display_layout.setContentsMargins(0, 0, 0, 0)

        # Which overview to draw on. Populated from the experiment; the Load Image
        # button still reaches anything else on disk.
        self.overview_combo = QComboBox()
        self.overview_combo.setStyleSheet(COMBOBOX_STYLESHEET)
        self.overview_combo.currentIndexChanged.connect(self._on_overview_selected)

        # Title Text
        self.title_textbox = QLineEdit()
        self.title_textbox.setStyleSheet(LINEEDIT_STYLESHEET)
        self.title_textbox.setText("Overview Image")

        # Marker color
        self.color_label = QLabel("●")
        self.color_label.setStyleSheet(
            f"color: {self.marker_color.name()}; font-size: 20px;"
        )
        self.color_button = QPushButton("Choose Color")
        self.color_button.setStyleSheet(BUTTON_STYLESHEET)
        self.color_button.clicked.connect(self._on_color_button_clicked)
        color_layout = QHBoxLayout()
        color_layout.addWidget(self.color_label)
        color_layout.addWidget(self.color_button)
        color_layout.addStretch()

        # Text size
        self.text_size_spinbox = IntegerValueSpinBox()
        self.text_size_spinbox.setStyleSheet(SPINBOX_STYLESHEET)
        self.text_size_spinbox.setRange(6, 48)
        self.text_size_spinbox.setValue(10)
        self.text_size_spinbox.setKeyboardTracking(False)

        # Marker size
        self.markersize_spinbox = IntegerValueSpinBox()
        self.markersize_spinbox.setStyleSheet(SPINBOX_STYLESHEET)
        self.markersize_spinbox.setRange(5, 100)
        self.markersize_spinbox.setValue(20)
        self.markersize_spinbox.setKeyboardTracking(False)

        # Show names checkbox
        self.show_names_checkbox = QCheckBox("")
        self.show_names_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.show_names_checkbox.setChecked(True)

        # Show descriptions checkbox (drawn as a subtitle under each name)
        self.show_descriptions_checkbox = QCheckBox("")
        self.show_descriptions_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.show_descriptions_checkbox.setChecked(False)

        # Show scalebar checkbox
        self.show_scalebar_checkbox = QCheckBox("")
        self.show_scalebar_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.show_scalebar_checkbox.setChecked(True)

        # Colour the flagged lamellae differently from the rest. On by default: a
        # defective lamella drawn identically to a good one is the map's most
        # consequential omission, since the recipient plans their session from it.
        self.show_defects_checkbox = QCheckBox("")
        self.show_defects_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.show_defects_checkbox.setChecked(True)

        # Instrument, operator, date and version along the bottom. An exported map gets
        # forwarded; without this it is a picture of some grid, somewhere, at some point.
        self.show_provenance_checkbox = QCheckBox("")
        self.show_provenance_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.show_provenance_checkbox.setChecked(True)

        # display layout
        display_layout.addRow("Overview", self.overview_combo)
        display_layout.addRow("Title", self.title_textbox)
        display_layout.addRow("Marker Color", color_layout)
        display_layout.addRow("Text Size", self.text_size_spinbox)
        display_layout.addRow("Marker Size", self.markersize_spinbox)
        display_layout.addRow("Show Names", self.show_names_checkbox)
        display_layout.addRow("Show Descriptions", self.show_descriptions_checkbox)
        display_layout.addRow("Show Scalebar", self.show_scalebar_checkbox)
        display_layout.addRow("Mark Defects", self.show_defects_checkbox)
        display_layout.addRow("Show Provenance", self.show_provenance_checkbox)

        display_group = TitledPanel(
            "Display Options", content=display_content, collapsible=False
        )

        # Connect signals to update preview on change
        self.text_size_spinbox.valueChanged.connect(self._on_preview_clicked)
        self.markersize_spinbox.valueChanged.connect(self._on_preview_clicked)
        self.show_names_checkbox.stateChanged.connect(self._on_preview_clicked)
        self.show_descriptions_checkbox.stateChanged.connect(self._on_preview_clicked)
        self.show_scalebar_checkbox.stateChanged.connect(self._on_preview_clicked)
        self.show_defects_checkbox.stateChanged.connect(self._on_preview_clicked)
        self.show_provenance_checkbox.stateChanged.connect(self._on_preview_clicked)
        self.title_textbox.editingFinished.connect(self._on_preview_clicked)

        # Preview canvas
        self.canvas = None
        self.figure = None

        # Create a placeholder widget for the canvas
        self.canvas_container = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Create initial empty figure
        self._create_empty_canvas()

        # Action buttons
        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.setStyleSheet(BUTTON_STYLESHEET)
        self.load_image_button.clicked.connect(self._on_browse_image_clicked)
        self.load_image_button.setAutoDefault(False)
        self.load_image_button.setDefault(False)

        self.reset_view_button = QPushButton("Reset View", self)
        self.reset_view_button.setStyleSheet(BUTTON_STYLESHEET)
        self.reset_view_button.clicked.connect(self._on_reset_view_clicked)

        self.save_button = QPushButton("Save Image", self)
        self.save_button.setStyleSheet(BUTTON_STYLESHEET)
        self.save_button.clicked.connect(self._on_save_clicked)

        # The preview takes the slack; the options column keeps its natural height at the
        # top. Until the header stopped stretching (see TitledPanel) the column absorbed
        # the spare space instead, which made the row look full while the canvas -- the
        # thing anyone actually looks at -- was the part being squeezed.
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.canvas_container, stretch=1)
        hlayout.addWidget(display_group, stretch=0, alignment=Qt.AlignmentFlag.AlignTop)

        # button layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.reset_view_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()

        # Info label
        self.info_label = QLabel("No experiment loaded")
        self.info_label.setStyleSheet(f"color: {NEUTRAL_400}; font-size: 10px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(hlayout, stretch=1)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.info_label)

        self.setLayout(main_layout)

    def _create_empty_canvas(self):
        """Create an empty canvas with placeholder text."""
        # Create empty figure
        self.figure = Figure(figsize=(10, 6), dpi=80)
        self.figure.patch.set_facecolor(SURFACE_COLOR)

        # Create canvas from figure
        if self.canvas is not None:
            self._disconnect_canvas_events()
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_layout.addWidget(self.canvas)

        # Add placeholder text
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("black")
        ax.text(
            0.5,
            0.5,
            "No preview available\nSelect image and generate preview",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color=NEUTRAL_400,
        )
        ax.set_title("Overview Image Preview", color="white")
        ax.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()
        self._enable_canvas_events()
        self._capture_initial_view_limits()

    def _replace_canvas_with_figure(self, new_figure):
        """Replace the current canvas with a new figure.

        Args:
            new_figure: Matplotlib Figure object to display
        """
        # Remove old canvas
        if self.canvas is not None:
            self._disconnect_canvas_events()
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        # Store the new figure
        if self.figure is not None and self.figure != new_figure:
            plt.close(self.figure)

        self.figure = new_figure

        # Apply dark theme to the figure
        self.figure.patch.set_facecolor(SURFACE_COLOR)

        # Create new canvas with the figure
        self.canvas = FigureCanvas(self.figure)
        # A FigureCanvas asks for exactly the figure's own size and will not grow past
        # it on its own, so a preview replaced mid-session would otherwise shrink the
        # row to whatever inches the renderer chose.
        self.canvas.setMinimumSize(400, 300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_layout.addWidget(self.canvas)

        # Draw the canvas
        self.canvas.draw()
        self._enable_canvas_events()
        self._capture_initial_view_limits()

    def _disconnect_canvas_events(self):
        """Disconnect zoom and pan handlers from the current canvas."""
        if self.canvas is None:
            return

        for attr in (
            "_scroll_cid",
            "_pan_press_cid",
            "_pan_release_cid",
            "_pan_motion_cid",
        ):
            cid = getattr(self, attr)
            if cid is None:
                continue
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
            setattr(self, attr, None)

        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None

    def _enable_canvas_events(self):
        """Attach scroll, press, release, and motion handlers to the canvas."""
        if self.canvas is None:
            return

        self._scroll_cid = self.canvas.mpl_connect(
            "scroll_event", self._on_canvas_scroll
        )
        self._pan_press_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_press
        )
        self._pan_release_cid = self.canvas.mpl_connect(
            "button_release_event", self._on_canvas_release
        )
        self._pan_motion_cid = self.canvas.mpl_connect(
            "motion_notify_event", self._on_canvas_motion
        )

    def _on_canvas_scroll(self, event):
        """Handle scroll-wheel events to zoom in/out."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        ax = event.inaxes
        if event.button == "up":
            scale_factor = 1 / ZOOM_SCALE_FACTOR
        elif event.button == "down":
            scale_factor = ZOOM_SCALE_FACTOR
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        x_range = cur_xlim[1] - cur_xlim[0]
        y_range = cur_ylim[1] - cur_ylim[0]

        if x_range == 0 or y_range == 0:
            return

        new_width = x_range * scale_factor
        new_height = y_range * scale_factor

        relx = (event.xdata - cur_xlim[0]) / x_range
        rely = (event.ydata - cur_ylim[0]) / y_range

        new_xlim = (
            event.xdata - relx * new_width,
            event.xdata + (1 - relx) * new_width,
        )
        new_ylim = (
            event.ydata - rely * new_height,
            event.ydata + (1 - rely) * new_height,
        )

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _on_canvas_press(self, event):
        """Start a pan gesture when the left mouse button is pressed."""
        if (
            event.button != 1
            or event.inaxes is None
            or event.xdata is None
            or event.ydata is None
        ):
            return

        self._pan_active = True
        self._pan_axes = event.inaxes
        self._pan_start = (event.xdata, event.ydata)
        self._pan_axes_limits = (
            event.inaxes.get_xlim(),
            event.inaxes.get_ylim(),
        )

    def _on_canvas_motion(self, event):
        """Update axes while the mouse moves to create panning."""
        if not self._pan_active or self._pan_axes is None:
            return
        if event.inaxes != self._pan_axes or event.xdata is None or event.ydata is None:
            return
        if self._pan_axes_limits is None or self._pan_start is None:
            return

        start_xlim, start_ylim = self._pan_axes_limits
        dx = event.xdata - self._pan_start[0]
        dy = event.ydata - self._pan_start[1]

        new_xlim = (start_xlim[0] - dx, start_xlim[1] - dx)
        new_ylim = (start_ylim[0] - dy, start_ylim[1] - dy)

        self._pan_axes.set_xlim(new_xlim)
        self._pan_axes.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _on_canvas_release(self, event):
        """End the pan gesture."""
        if event.button != 1:
            return

        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None

    def _capture_initial_view_limits(self):
        """Store axes limits to allow resetting the view."""
        self._initial_axes_limits = []
        if self.figure is None:
            return

        for ax in self.figure.axes:
            self._initial_axes_limits.append((ax, ax.get_xlim(), ax.get_ylim()))

    def _on_reset_view_clicked(self):
        """Reset all axes to their initial limits."""
        if not self.figure or not self.canvas or not self._initial_axes_limits:
            return

        for ax, xlim, ylim in self._initial_axes_limits:
            if ax in self.figure.axes:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        self.canvas.draw_idle()

    def set_experiment(self, experiment: "Experiment"):
        """Set the experiment to use for generating overview images.

        Args:
            experiment: The Experiment object containing lamella positions
        """
        self.experiment = experiment

        if experiment is not None:
            self.info_label.setText(
                f"Experiment: {experiment.name} | {len(experiment.positions)} lamellae"
            )
            self.title_textbox.setText(f"Experiment: {self.experiment.name}")
            self.title_textbox.setCursorPosition(0)

            # Collect positions (always use MILLING state)
            self.stage_positions.clear()
            self.stage_positions = self.experiment.get_milling_positions()

            self._populate_overview_picker()
        else:
            self.info_label.setText("No experiment loaded")

    def _populate_overview_picker(self):
        """List every overview in the experiment, and select the most recent.

        Selecting the most recent rather than an arbitrary one: the dialog used to glob
        for itself and take `filenames[-1]`, which is the last in *name* order and had no
        UI behind it at all -- an experiment with three overviews reported on one of them
        and gave no way to see the others.
        """
        self.overview_combo.blockSignals(True)
        self.overview_combo.clear()
        paths = self.experiment.find_overview_images()
        for path in paths:
            self.overview_combo.addItem(os.path.basename(path), path)
        self.overview_combo.blockSignals(False)

        # Fluorescence overviews are a different view of the sample and cannot be drawn
        # on a beam overview's axes, so they are counted here rather than offered. Saying
        # how many exist stops "my FM overview is missing" being a mystery.
        fm_count = len(self.experiment.find_fluorescence_overview_images())
        self._fm_overview_count = fm_count

        if paths:
            self.overview_combo.setCurrentIndex(len(paths) - 1)
            self._load_overview_image(paths[-1])
        else:
            self.info_label.setText(
                "No overview image found in the experiment - use Load Image."
            )

    def _on_overview_selected(self, index: int):
        """A different overview was picked from the list."""
        path = self.overview_combo.itemData(index)
        if path:
            self._load_overview_image(path)

    def _on_browse_image_clicked(self):
        """Handle browse button click for selecting and loading overview image."""
        start_dir = ""
        if self.experiment is not None:
            start_dir = str(self.experiment.path)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Overview Image",
            start_dir,
            "TIFF Files (*.tif *.tiff);;All Files (*.*)",
        )

        if file_path:
            self._load_overview_image(file_path)

    def _load_overview_image(self, file_path: str):
        """Load the overview image from the specified file path."""
        try:
            # Load the overview image
            self.overview_image = FibsemImage.load(file_path)
            self.info_label.setText(f"Loaded: {os.path.basename(file_path)}")

            # Automatically generate preview when image is selected
            self._on_preview_clicked()
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            self.info_label.setText(f"Error loading image: {str(e)}")
            self.overview_image = None

    def _on_color_button_clicked(self):
        """Handle color button click to select marker color."""
        color = QColorDialog.getColor(self.marker_color, self, "Select Marker Color")

        if color.isValid():
            self.marker_color = color
            self.color_label.setStyleSheet(f"color: {color.name()}; font-size: 20px;")
            self._on_preview_clicked()

    def _build_figure(self, title_color: str) -> Figure:
        """Render the overview and its markers into a new figure.

        The one place the figure is drawn, so the preview and the export cannot drift
        apart in anything but the colour of the title -- which is the only thing that
        legitimately differs between a dark canvas and a white page.
        """
        # lamella name -> free-text description, for the optional subtitle
        descriptions = {lam.name: lam.description for lam in self.experiment.positions}

        fig = plot_minimap(
            self.overview_image,
            self.stage_positions,
            current_position=None,
            grid_positions=None,
            color=self.marker_color.name(),
            fontsize=self.text_size_spinbox.value(),
            markersize=self.markersize_spinbox.value(),
            show_scalebar=self.show_scalebar_checkbox.isChecked(),
            show_names=self.show_names_checkbox.isChecked(),
            show_descriptions=self.show_descriptions_checkbox.isChecked(),
            descriptions=descriptions,
            colors=self._defect_colors(),
            # None sizes the figure to the overview's aspect. Only the starting point
            # for the preview, whose figure is then resized to the canvas widget -- but
            # it is what the export is built at, and there it decides everything.
            figsize=None,
        )

        title_text = self.title_textbox.text().strip()
        if title_text:
            fig.suptitle(title_text, fontsize=14, color=title_color)

        bottom = 0.0
        if self.show_provenance_checkbox.isChecked():
            provenance = self._provenance_text()
            if provenance:
                fig.text(
                    0.5,
                    0.012,
                    provenance,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=title_color,
                    alpha=0.75,
                )
                bottom = 0.045

        fig.tight_layout(rect=(0, bottom, 1, 0.98))
        return fig

    def _defect_colors(self) -> Dict[str, str]:
        """Lamella name -> marker colour, for the ones a human has flagged.

        Only the flagged ones. Everything else is absent from the mapping and keeps
        whatever the Marker Color control is set to, so choosing a colour still does what
        it says for the lamellae that have nothing wrong with them.

        Driven by `defect.state` alone -- the one field a person actually set. Nothing
        here is derived from task history: a lamella whose polishing task never ran is
        unfinished, which is not the same claim as defective, and a report that quietly
        promotes one to the other is inventing a judgement nobody made.
        """
        if not self.show_defects_checkbox.isChecked():
            return {}
        colours: Dict[str, str] = {}
        for lamella in self.experiment.positions:
            state = lamella.defect.state
            if state is DefectType.FAILURE:
                colours[lamella.name] = DEFECT_FAILURE_COLOUR
            elif state is DefectType.REWORK:
                colours[lamella.name] = DEFECT_REWORK_COLOUR
        return colours

    def _provenance_text(self) -> str:
        """One line naming the instrument, the operator, and when.

        An exported map gets forwarded, and three months later a picture of a grid with
        no instrument, date or software version on it cannot be tied back to anything.
        `Experiment.session` carries all of it already.

        Absent for an experiment no session has adopted, and for every experiment written
        before the session record existed. That returns an empty string rather than a
        line of "unknown"s -- a caption that admits nothing is better than one that
        asserts blanks.
        """
        session = getattr(self.experiment, "session", None)
        parts: List[str] = []
        if session is not None:
            system = getattr(session, "system", None)
            user = getattr(session, "user", None)
            model = getattr(system, "model", "") or ""
            serial = getattr(system, "serial_number", "") or ""
            if model:
                parts.append(f"{model} ({serial})" if serial else model)
            operator = getattr(user, "name", "") or ""
            if operator:
                parts.append(operator)
            version = getattr(system, "fibsem_version", "") or ""
            if version:
                parts.append(f"fibsemOS {version}")

        created = getattr(self.experiment, "created_at", None)
        if created:
            try:
                parts.append(
                    f"milled {datetime.fromtimestamp(created).strftime(DATETIME_DISPLAY)}"
                )
            except (ValueError, OSError, TypeError):
                pass

        if not parts:
            return ""
        parts.append(f"map {datetime.now().strftime(DATETIME_DISPLAY)}")
        return "  |  ".join(parts)

    def _on_preview_clicked(self):
        """Generate and display preview of the overview image."""
        if self.experiment is None:
            self.info_label.setText("Error: No experiment loaded")
            return

        if self.overview_image is None:
            self.info_label.setText("Error: No overview image loaded")
            return

        try:
            if not self.stage_positions:
                self.info_label.setText("Warning: No positions found for MILLING state")
                self._create_empty_canvas()
                return

            fig = self._build_figure(title_color="white")

            # Store and display figure
            self.current_figure = fig
            self._replace_canvas_with_figure(fig)

            self.info_label.setText(
                f"Preview generated | {len(self.stage_positions)} positions shown"
            )

        except Exception as e:
            logging.error(f"Error generating preview: {e}")
            import traceback

            traceback.print_exc()
            self.info_label.setText(f"Error: {str(e)}")
            self._create_empty_canvas()

    def _on_save_clicked(self):
        """Save the overview image by opening a save file dialog."""
        if self.current_figure is None:
            self.info_label.setText(
                "Error: No preview to save. Generate preview first."
            )
            return

        if self.overview_image is None or self.experiment is None:
            self.info_label.setText("Error: Missing required data")
            return

        # Open save file dialog
        default_name = f"{self.experiment.name}_overview_final.png"
        start_filename = os.path.join(str(self.experiment.path), default_name)

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Overview Image", start_filename, "PNG Files (*.png);;"
        )

        if not output_path:
            # User cancelled
            return

        export_figure = None
        try:
            # Rendered fresh rather than saved from the preview. The preview's figure is
            # resized to the canvas *widget* the moment it is shown, so its shape is the
            # dialog's, not the overview's -- and a wide overview in a tall widget leaves
            # a band of empty paper between the title and the image that no amount of
            # `bbox_inches="tight"` removes, because a bounding box is a rectangle and
            # the gap is inside it. Measured on a 3:1 overview: the axes held 0.35 of the
            # exported page.
            #
            # It also retires the trick this replaces, which recoloured the preview's
            # title black, saved, and put it back. That worked only for callers who went
            # through this method, and left the figure briefly in a state no reader
            # expected.
            export_figure = self._build_figure(title_color="black")
            self._match_view_limits(source=self.current_figure, target=export_figure)
            export_figure.savefig(
                output_path, dpi=300, bbox_inches="tight", facecolor="white"
            )

            logging.info(f"Saved overview image to: {output_path}")
            self.info_label.setText(f"Saved to: {output_path}")

        except Exception as e:
            logging.error(f"Error saving image: {e}")
            import traceback

            traceback.print_exc()
            self.info_label.setText(f"Error saving: {str(e)}")
        finally:
            if export_figure is not None:
                plt.close(export_figure)

    @staticmethod
    def _match_view_limits(source: Optional[Figure], target: Figure) -> None:
        """Copy the panned/zoomed view from one figure's axes onto another's.

        So that exporting after zooming into a corner of the grid exports that corner,
        which is the whole reason the preview can be zoomed.
        """
        if source is None:
            return
        for src_ax, dst_ax in zip(source.axes, target.axes):
            dst_ax.set_xlim(src_ax.get_xlim())
            dst_ax.set_ylim(src_ax.get_ylim())


def create_overview_image_widget(
    experiment: "Experiment", parent: Optional["AutoLamellaUI"] = None
) -> QDialog:
    """Create and initialize an OverviewImageWidget wrapped in a dialog.

    Args:
        experiment: The Experiment object to use
        parent: Optional parent AutoLamellaUI widget

    Returns:
        QDialog: Dialog containing the initialized widget with the experiment loaded
    """
    # Create dialog
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Generate Overview Image - {experiment.name}")
    dialog.setMinimumSize(800, 800)

    # Create layout
    layout = QVBoxLayout()

    # Create and add widget
    widget = OverviewImageWidget(parent=parent)
    widget.set_experiment(experiment)
    layout.addWidget(widget)

    dialog.setLayout(layout)

    return dialog


# TODO: add more options for customization (scalebar size, font type, etc.)
# - show only completed lamellae
# - show defected lamellae in different color


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    layout = QVBoxLayout()
    widget = OverviewImageWidget()
    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-11-07-12-32/experiment.yaml"
    exp = Experiment.load(PATH)
    dialog = create_overview_image_widget(experiment=exp)

    dialog.show()
    sys.exit(app.exec_())
