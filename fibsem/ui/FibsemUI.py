"""The standalone fibsemOS window: a microscope, and an overview of the grid it holds.

Two top-level tabs, the same shape AutoLamella's window has:

* **Microscope** — the quad view beside the control panel.
* **Overview** — the tiled overview on the real-space canvas.

The Overview tab is hosted *bare*: `FibsemOverviewWidget` knows nothing about lamellae
by design -- it says *where* a user pointed and leaves the meaning to whoever is
listening -- and this application has no experiment to listen with. So there is no lamella list and there are
no markers, which is the whole of what `BeamOverviewTab` adds on top of it.
"""

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.imaging.overview_session import OverviewSession
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.theme.tokens import GRAY_ICON_COLOR
from fibsem.tools.apps import FIBSEM_UI
from fibsem.tools.desktop_launch import Splash
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.icon import fibsem_icon, fibsemos_icon
from fibsem.ui.qt.app import start_qt_app
from fibsem.ui.stylesheets import NAPARI_STYLE
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.microscope_panel import MicroscopeControls, add_microscope_tab
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget
from fibsem.ui.widgets.status_bar import FibsemStatusBar
from fibsem.versioning import get_version_string

NO_MICROSCOPE_MESSAGE = "Connect to a microscope to acquire an overview."


class FibsemUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # The app's dark theme, matching AutoLamellaWindow; a standalone window sets it
        # itself.
        self.setStyleSheet(NAPARI_STYLE)

        # get_version_string (not fibsem.__version__) so the running git revision shows.
        self.setWindowTitle(f"fibsemOS v{get_version_string()}")

        self.microscope: Optional[FibsemMicroscope] = None
        self.settings: Optional[MicroscopeSettings] = None

        self.overview_widget: Optional[FibsemOverviewWidget] = None

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self._create_microscope_tab()
        self._create_overview_tab()

        # What the instrument is doing, in words, while it does it.
        self.status_bar = FibsemStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.follow(self.view_controller)

        self.setup_connections()
        self.update_ui()

    # ── layout ───────────────────────────────────────────────────────────

    def _create_microscope_tab(self) -> None:
        """The quad view beside the control panel."""
        # The controller's SEM/FIB/FM canvases are the display and drive the control
        # widgets, which resolve it through their parent.
        self.view_controller = MicroscopeViewController(parent=self)

        # The same control tabs AutoLamella and Volume show, after the Connection tab.
        self.controls = MicroscopeControls(
            view_controller=lambda: self.view_controller,
            overview_acquiring=self._overview_is_acquiring,
            parent=self,
        )
        self.system_widget = FibsemSystemSetupWidget(parent=self)
        self.controls.tabWidget.addTab(self.system_widget, "Connection")

        add_microscope_tab(
            self.tab_widget,
            self.view_controller,
            self.controls,
            sizes=(720, 460),
            controls_min_width=420,
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

    def update_ui(self) -> None:
        is_microscope_connected = self.microscope is not None
        self.tab_widget.setTabEnabled(
            self.tab_widget.indexOf(self.overview_container), is_microscope_connected
        )

    def connect_to_microscope(self) -> None:
        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings
        self.status_bar.set_microscope(self.microscope)
        self._show_microscope()
        self._build_overview()
        self.update_ui()

    def disconnect_from_microscope(self) -> None:
        self.microscope = None
        self.settings = None
        self.status_bar.set_microscope(None)
        self.view_controller.clear()
        self._show_microscope()
        self._drop_overview()
        self.update_ui()

    def _show_microscope(self) -> None:
        """The control tabs for the connected microscope, or none."""
        self.controls.show_microscope(self.microscope, self.settings)
        image_widget = self.controls.image_widget
        # The defaults' "Read from Acquire Tab" reads this tab.
        self.system_widget.set_current_imaging(
            image_widget._get_image_settings_from_ui
            if image_widget is not None
            else None
        )

    def _overview_is_acquiring(self) -> bool:
        return self.overview_widget is not None and self.overview_widget.is_acquiring

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Let the instrument go rather than leave the client open until the
        process ends."""
        if self.microscope is not None:
            self.microscope.try_disconnect()
        super().closeEvent(event)

    # ── the overview a connection brings with it ─────────────────────────

    def _build_overview(self) -> None:
        """Fill the Overview tab in, now that there is an instrument to scale it."""
        self._drop_overview()
        self.overview_widget = FibsemOverviewWidget(
            OverviewSession(self.microscope),
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


def main(argv: Optional[List[str]] = None, splash: Optional[Splash] = None):
    """Run the fibsemOS UI.

    ``argv`` is accepted for the launcher's common signature; the UI takes no
    flags. ``splash`` is the launcher's splash screen (fibsem.tools.desktop_launch),
    closed once the main window is up.
    """
    app = start_qt_app(FIBSEM_UI, fibsemos_icon())
    fibsem_ui = FibsemUI()
    fibsem_ui.show()
    if splash is not None:
        splash.finish(fibsem_ui)
    app.exec_()


if __name__ == "__main__":
    main()
