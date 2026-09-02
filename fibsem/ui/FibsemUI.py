"""The standalone fibsemOS window: a microscope, and an overview of the grid it holds.

Two top-level tabs, the same shape AutoLamella's window has:

* **Microscope** — the quad view beside the control panel, which is what this whole
  window used to be.
* **Overview** — the tiled overview on the real-space canvas.

The Overview tab replaces the napari minimap this window used to open in a viewer of
its own, and it is the last napari overview anyone had as a default (FIB-821). It is
hosted *bare*: `FibsemOverviewWidget` knows nothing about lamellae by design -- it says
*where* a user pointed and leaves the meaning to whoever is listening -- and this
application has no experiment to listen with. So there is no lamella list and there are
no markers, which is the whole of what `AutoLamellaOverviewTab` adds on top of it.

Written by hand rather than loaded from `qtdesigner_files/FibsemUI.ui`. The generated
window contributed a title label this file deleted on the next line, a Tools menu with
one live entry, and a tab widget -- and it could not express the tab-in-a-tab this
layout is. Both it and the `.ui` it came from are gone.
"""

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QLabel,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.tokens import GRAY_ICON_COLOR
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget
from fibsem.versioning import get_version_string

NO_MICROSCOPE_MESSAGE = "Connect to a microscope to acquire an overview."


class FibsemUI(QMainWindow):
    def __init__(self, viewer=None):
        super().__init__()

        # napari-style dark theme, matching AutoLamellaMainUI. The napari window used to
        # supply this; a standalone window has to set it itself.
        self.setStyleSheet(NAPARI_STYLE)

        # get_version_string (not fibsem.__version__) so the running git revision still
        # shows once napari's viewer.title is gone — see FIB-349 / #202.
        self.setWindowTitle(f"fibsemOS v{get_version_string()}")

        # Viewer-less by default: the quad-view controller is the display. `viewer` is
        # retained so a caller that still has a napari Viewer gets the old layer path —
        # the image/movement widgets read it off this attribute.
        self.viewer = viewer

        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None

        self.image_widget: Optional[FibsemImageSettingsWidget] = None
        self.movement_widget: Optional[FibsemMovementWidget] = None
        self.milling_widget: Optional[MillingTaskViewerWidget] = None
        self.sample_widget: Optional[MillingTaskViewerWidget] = None
        self.manipulator_widget: Optional[FibsemManipulatorWidget] = None
        self.overview_widget: Optional[FibsemOverviewWidget] = None

        # The control tabs that exist only while a microscope is connected, in the order
        # they were added. Held so that adding and removing them is by identity: the
        # disconnect path used to remove indices 4..1, which is only correct for as long
        # as that set never changes size — and one of them is already conditional.
        self._microscope_tabs: List[QWidget] = []

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self._create_menu_bar()
        self._create_microscope_tab()
        self._create_overview_tab()

        self.setup_connections()
        self.update_ui()

    # ── layout ───────────────────────────────────────────────────────────

    def _create_menu_bar(self) -> None:
        tools_menu = self.menuBar().addMenu("Tools")
        self.action_manipulator_calibration = QAction("Manipulator Calibration", self)
        tools_menu.addAction(self.action_manipulator_calibration)

    def _create_microscope_tab(self) -> None:
        """The quad view beside the control panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # The controller's SEM/FIB/FM canvases are the display and drive the control
        # widgets, which resolve it through their parent.
        self.view_controller = MicroscopeViewController(parent=self)

        self.control_tabs = QTabWidget()
        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.control_tabs.addTab(self.system_widget, "Connection")

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.view_controller.widget)
        splitter.addWidget(self.control_tabs)
        splitter.setSizes([720, 460])
        splitter.widget(1).setMinimumWidth(420)
        layout.addWidget(splitter)

        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR),
            "Microscope",
        )

    def _create_overview_tab(self) -> None:
        """Reserve the Overview tab, and say why it is empty until it is not.

        Created empty on every system, and never removed, so the tab bar keeps the same
        shape whether or not anything is connected. `FibsemOverviewWidget` requires a
        microscope at construction — every scale on its canvas comes from the instrument
        — so there is nothing to build until there is one.
        """
        self.overview_container = QWidget()
        self._overview_layout = QVBoxLayout(self.overview_container)
        self._overview_layout.setContentsMargins(0, 0, 0, 0)

        self._overview_placeholder = QLabel(NO_MICROSCOPE_MESSAGE)
        self._overview_placeholder.setAlignment(Qt.AlignCenter)
        self._overview_layout.addWidget(self._overview_placeholder)

        self.tab_widget.addTab(
            self.overview_container,
            fibsem_icon("mdi:map-search-outline", color=GRAY_ICON_COLOR),
            "Overview",
        )

    # ── connections ──────────────────────────────────────────────────────

    def setup_connections(self) -> None:
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)
        # Connected unconditionally, and resolved when it fires. It used to be wired
        # only `if self.manipulator_widget is not None` — which runs before any
        # microscope has connected, so the widget is always None and the menu entry was
        # never actually connected to anything.
        self.action_manipulator_calibration.triggered.connect(
            self._calibrate_manipulator
        )

    def _calibrate_manipulator(self) -> None:
        if self.manipulator_widget is None:
            return
        self.manipulator_widget.calibrate_manipulator_positions()

    def update_ui(self) -> None:
        is_microscope_connected = self.microscope is not None
        self.tab_widget.setTabEnabled(
            self.tab_widget.indexOf(self.overview_container), is_microscope_connected
        )
        self.action_manipulator_calibration.setVisible(is_microscope_connected)

    def connect_to_microscope(self) -> None:
        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings
        self.update_microscope_ui()
        self.update_ui()

    def disconnect_from_microscope(self) -> None:
        self.microscope = None
        self.settings = None
        self.update_microscope_ui()
        self.update_ui()
        self.image_widget = None
        self.movement_widget = None
        self.milling_widget = None
        self.sample_widget = None
        self.overview_widget = None

    # ── the widgets a connection brings with it ──────────────────────────

    def _add_control_tab(self, widget: QWidget, label: str) -> None:
        """Add a control tab that lives only for as long as this connection does."""
        self.control_tabs.addTab(widget, label)
        self._microscope_tabs.append(widget)

    def update_microscope_ui(self) -> None:
        if self.microscope is not None:
            # reusable components
            self.image_widget = FibsemImageSettingsWidget(
                microscope=self.microscope,
                image_settings=self.settings.image,
                parent=self,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                parent=self,
            )
            self.milling_widget = MillingTaskViewerWidget(
                microscope=self.microscope,
                parent=self,
                image_widget=self.image_widget,  # lets it resolve the quad controller
            )
            if self.microscope.system.manipulator.enabled:
                self.manipulator_widget = FibsemManipulatorWidget(
                    microscope=self.microscope,
                    settings=self.settings,
                    viewer=self.viewer,
                    image_widget=self.image_widget,
                    parent=self,
                )
            else:
                self.manipulator_widget = None

            # The hardware view of the grids: the holder, and the magazine when
            # there is one. Slot moves go through the Movement widget, the same
            # route as a saved position, so the readout and post-move images follow.
            self.sample_widget = FibsemSampleWidget(
                microscope=self.microscope, parent=self
            )
            self.sample_widget.move_to_requested.connect(
                self.movement_widget.move_to_position
            )

            # add widgets to tabs
            self._add_control_tab(self.image_widget, "Image")
            self._add_control_tab(self.movement_widget, "Movement")
            self._add_control_tab(self.milling_widget, "Milling")
            self._add_control_tab(self.sample_widget, "Sample")

            if self.microscope.system.manipulator.enabled:
                self._add_control_tab(self.manipulator_widget, "Manipulator")

            self.system_widget.image_widget = self.image_widget
            self.system_widget.milling_widget = self.milling_widget

            self._build_overview()

        else:
            if not self._microscope_tabs:
                return

            # remove tabs
            for widget in reversed(self._microscope_tabs):
                index = self.control_tabs.indexOf(widget)
                if index != -1:
                    self.control_tabs.removeTab(index)
            self._microscope_tabs.clear()

            self.image_widget.clear_viewer()
            # Drop the widgets' controller/canvas connections before deleteLater — the
            # canvases persist across a reconnect, so a leaked slot would fire on a dead
            # widget (deleteLater fires neither closeEvent nor close).
            self.image_widget._teardown_connections()
            self.image_widget.deleteLater()
            self.movement_widget._teardown_connections()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()
            self.sample_widget.deleteLater()
            if self.manipulator_widget is not None:
                self.manipulator_widget.deleteLater()

            self._drop_overview()

    def _build_overview(self) -> None:
        """Fill the Overview tab in, now that there is an instrument to scale it."""
        self._drop_overview()
        self.overview_widget = FibsemOverviewWidget(
            microscope=self.microscope,
            parent=self.overview_container,
        )
        # Where the rest of this application already writes. Left unset, a run falls
        # back to `os.getcwd()` and scatters tilesets through whatever directory the app
        # happened to be launched from. Shown in the Output panel, so it is a default
        # rather than a decision — the user can point it elsewhere.
        self.overview_widget.set_save_directory(self.settings.image.path)

        self._overview_placeholder.hide()
        self._overview_layout.addWidget(self.overview_widget)

    def _drop_overview(self) -> None:
        """Retire the overview widget, if there is one.

        `close()` before `deleteLater()`, and not one or the other: the widget's
        `closeEvent` is what releases the psygnal subscriptions it holds on the
        microscope. Those outlive the widget — they belong to the microscope — so one
        left connected emits into a Qt object that has already been torn down on the C++
        side. That was a segfault rather than an exception. `deleteLater` fires neither
        `closeEvent` nor `close`.
        """
        if self.overview_widget is None:
            return
        self.overview_widget.close()
        self.overview_widget.deleteLater()
        self.overview_widget = None
        self._overview_placeholder.show()


def main():
    # Fully viewer-less: the quad-view controller is the display, and the overview is a
    # tab beside it rather than a napari viewer of its own.
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    fibsem_ui = FibsemUI()
    fibsem_ui.show()
    app.exec_()


if __name__ == "__main__":
    main()
