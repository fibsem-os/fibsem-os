from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.ui import notification_service
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.qtdesigner_files import FibsemUI as FibsemUIMainWindow
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget
from fibsem.versioning import get_version_string


class FibsemUI(FibsemUIMainWindow.Ui_MainWindow, QtWidgets.QMainWindow):

    def __init__(self, viewer=None):
        super().__init__()
        self.setupUi(self)

        # napari-style dark theme, matching AutoLamellaMainUI. The napari window used to
        # supply this; a standalone window has to set it itself.
        self.setStyleSheet(NAPARI_STYLE)

        # Title now lives in the window titlebar; drop the in-panel label.
        # get_version_string (not fibsem.__version__) so the running git revision still
        # shows once napari's viewer.title is gone — see FIB-349 / #202.
        self.setWindowTitle(f"fibsemOS v{get_version_string()}")
        self.gridLayout.removeWidget(self.label_title)
        self.label_title.deleteLater()

        # Viewer-less by default: the quad-view controller is the display. `viewer` is
        # retained so a caller that still has a napari Viewer gets the old layer path —
        # the image/movement widgets read it off this attribute.
        self.viewer = viewer

        # Quad-view display: the controller's SEM/FIB canvases are the left pane; the
        # existing tab panel becomes the right pane.
        self.view_controller = MicroscopeViewController(parent=self)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.view_controller.widget)
        splitter.addWidget(self.centralwidget)
        splitter.setSizes([720, 460])
        self.setCentralWidget(splitter)

        self.microscope: FibsemMicroscope = None
        self.settings: MicroscopeSettings = None

        self.image_widget: FibsemImageSettingsWidget = None
        self.movement_widget: FibsemMovementWidget = None
        self.milling_widget: MillingTaskViewerWidget = None
        self.manipulator_widget: FibsemManipulatorWidget = None

        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.tabWidget.addTab(self.system_widget, "Connection")
        self.setup_connections()
        self.update_ui()

    def setup_connections(self):
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)
        if self.manipulator_widget is not None:
            self.actionManipulator_Positions_Calibration.triggered.connect(self.manipulator_widget.calibrate_manipulator_positions)
        self.actionOpen_Minimap.triggered.connect(self.open_minimap_widget)

    def open_minimap_widget(self):
        if self.microscope is None:
            notification_service.show_toast("Please connect to a microscope first... [No Microscope Connected]", "warning")
            return

        if self.movement_widget is None:
            notification_service.show_toast("Please connect to a microscope first... [No Movement Widget]", "warning")
            return

        # NOTE: the minimap still opens its own napari viewer. It is being rebuilt on the
        # FM-overview (multi-image) model in Phase 3 / FIB-405, so it is deliberately left
        # alone here rather than ported twice.
        import napari

        from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget

        self.viewer_minimap = napari.Viewer(ndisplay=2)
        self.minimap_widget = FibsemMinimapWidget(viewer=self.viewer_minimap, parent=self)
        self.viewer_minimap.window.add_dock_widget(
            widget=self.minimap_widget, 
            area="right", 
            add_vertical_stretch=True, 
            name="fibsemOS Minimap"
        )
        napari.run(max_loop_level=2)

    def update_ui(self):

        is_microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(1, is_microscope_connected)
        self.tabWidget.setTabVisible(2, is_microscope_connected)
        self.tabWidget.setTabVisible(3, is_microscope_connected)
        self.tabWidget.setTabVisible(4, is_microscope_connected)
        self.actionOpen_Minimap.setVisible(is_microscope_connected)
        self.actionManipulator_Positions_Calibration.setVisible(is_microscope_connected)

    def connect_to_microscope(self):
        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings
        self.update_microscope_ui()
        self.update_ui()

    def disconnect_from_microscope(self):
    
        self.microscope = None
        self.settings = None
        self.update_microscope_ui()
        self.update_ui()
        self.image_widget = None
        self.movement_widget = None
        self.milling_widget = None

    def update_microscope_ui(self):

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
  


            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.tabWidget.addTab(self.milling_widget, "Milling")

            if self.microscope.system.manipulator.enabled:
                self.tabWidget.addTab(self.manipulator_widget, "Manipulator")

            self.system_widget.image_widget = self.image_widget
            self.system_widget.milling_widget = self.milling_widget

        else:
            if self.image_widget is None:
                return
            
            # remove tabs
            self.tabWidget.removeTab(4)
            self.tabWidget.removeTab(3)
            self.tabWidget.removeTab(2)
            self.tabWidget.removeTab(1)
            self.image_widget.clear_viewer()
            # Drop the widgets' controller/canvas connections before deleteLater — the
            # canvases persist across a reconnect, so a leaked slot would fire on a dead
            # widget (deleteLater fires neither closeEvent nor close).
            self.image_widget._teardown_connections()
            self.image_widget.deleteLater()
            self.movement_widget._teardown_connections()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()
            if self.manipulator_widget is not None:
                self.manipulator_widget.deleteLater() 



def main():

    # Fully viewer-less: the quad-view controller is the display, so there is no napari
    # main viewer. (The minimap still opens its own, on demand — see open_minimap_widget.)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    fibsem_ui = FibsemUI()
    fibsem_ui.show()
    app.exec_()


if __name__ == "__main__":
    main()



