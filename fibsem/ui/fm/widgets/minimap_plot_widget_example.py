"""Example script demonstrating how to use the MinimapPlotWidget."""

import os
import sys

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget

from fibsem.applications.autolamella.structures import Experiment
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui.fm.widgets.minimap_plot_widget import MinimapPlotWidget

# Example paths - update these to match your data
PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-10-20-09-58"
OVERVIEW_PATH = os.path.join(PATH, "overview-image-2025-10-20_09-58-14.tif")


def load_experiment_data():
    """Load experiment data from the example path."""
    # Load experiment
    exp = Experiment.load(os.path.join(PATH, "experiment.yaml"))

    # Load overview image
    image = FibsemImage.load(OVERVIEW_PATH)

    # Extract lamella positions from experiment
    positions = []
    for p in exp.positions:
        pstate = p.poses.get("MILLING", p.state.microscope_state)
        if pstate is None or pstate.stage_position is None:
            continue
        pos = pstate.stage_position
        pos.name = p.name
        positions.append(pos)

    # Create example current position
    current_position = FibsemStagePosition(
        name="Current Position", x=250e-6, y=150e-6, z=0, r=0, t=0
    )

    # Create example grid position
    grid_position = FibsemStagePosition(name="Grid 01", x=0, y=0, z=0, r=0, t=0)

    return image, positions, current_position, [grid_position]


def main():
    """Run the minimap widget example."""
    app = QApplication(sys.argv)

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Minimap Plot Widget Example")
    window.setGeometry(100, 100, 800, 900)

    # Create central widget
    central_widget = QWidget()
    layout = QVBoxLayout()

    # Create and add the minimap widget
    minimap_widget = MinimapPlotWidget()
    layout.addWidget(minimap_widget)

    # Add control buttons
    control_layout = QHBoxLayout()

    reload_button = QPushButton("Reload Data")
    reload_button.setStyleSheet("""
        QPushButton {
            background-color: #0f7aad;
            color: white;
            border: none;
            padding: 8px 16px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #1a8abe;
        }
    """)

    select_button = QPushButton("Select First Lamella")
    select_button.setStyleSheet("""
        QPushButton {
            background-color: #00cc00;
            color: white;
            border: none;
            padding: 8px 16px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #00dd00;
        }
    """)

    def reload_data():
        """Reload and display data from experiment."""
        try:
            image, positions, current_position, grid_positions = load_experiment_data()

            # Set the data
            minimap_widget.set_minimap_image(image)
            minimap_widget.set_lamella_positions(positions)
            minimap_widget.set_current_position(current_position)
            minimap_widget.set_grid_positions(grid_positions)

            print(f"Loaded {len(positions)} lamella positions")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using demo data instead...")
            # Could add fallback demo data here

    def select_first_lamella():
        """Select the first lamella position."""
        try:
            image, positions, current_position, grid_positions = load_experiment_data()
            if positions:
                minimap_widget.set_selected_name(positions[0].name)
                print(f"Selected: {positions[0].name}")
        except Exception as e:
            print(f"Error selecting lamella: {e}")

    reload_button.clicked.connect(reload_data)
    select_button.clicked.connect(select_first_lamella)

    control_layout.addWidget(reload_button)
    control_layout.addWidget(select_button)
    control_layout.addStretch()

    layout.addLayout(control_layout)

    central_widget.setLayout(layout)
    window.setCentralWidget(central_widget)

    # Try to load data on startup
    reload_data()

    # Show the window
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
