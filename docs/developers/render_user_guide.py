# ruff: noqa: E402  (the repo import comes before the fibsem imports on purpose)
"""Render the user guide's screenshots from the running application.

Every image in the user guide on fibsemos.org is produced here, from the real
AutoLamella window driven against the Demo microscope with the sample scene on,
so the guide can be re-shot after any UI change instead of rotting. Widget
callouts (the red boxes) are drawn from the widgets' own geometry, so a renamed
or removed widget fails the run rather than leaving a stale box.

Run from the repo root, with the docs site checked out beside this repository
(or pass --site):

    python docs/developers/render_user_guide.py                 # every page
    python docs/developers/render_user_guide.py getting-started # one page
    python docs/developers/render_user_guide.py --list

Renders on Qt's offscreen platform by default, so it needs no display and the
output does not depend on the screen it ran on; set QT_QPA_PLATFORM to override.

User state is redirected to a temporary directory for the run, so the machine's
own configuration registry and preferences are neither read nor written, and
nothing in a screenshot names the machine or the person who ran it.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

# render the code this file lives with, not an installed copy elsewhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# before Qt is imported: the platform is read when the QApplication is built
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop, QPoint, QRect, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget

import fibsem.config as cfg
from fibsem import guided_setup
from fibsem.guided_setup import LOCATION_MICROSCOPE, LOCATION_SUPPORT


def _find_site() -> Path:
    """The docs site checkout: a sibling of this repo, or of a parent of it
    (a git worktree under .claude/worktrees is two levels down)."""
    for parent in (ROOT, *ROOT.parents):
        candidate = parent / "fibsem-os.github.io"
        if (candidate / "package.json").exists():
            return candidate
    return ROOT.parent / "fibsem-os.github.io"


DEFAULT_SITE = _find_site()
IMG_SUBDIR = Path("public") / "doc" / "img"
# A typical microscope-PC window; there is no screen to clamp it. The control
# panel's tab bar clips at this size, as it does for real users.
WINDOW_SIZE = (1600, 1000)

# The shipped development configurations, registered in memory for the run
# under the names the guide uses. Neither touches the machine's registry.
SIM_CONFIGURATIONS = {
    "sim-arctis": "sim-arctis-configuration.yaml",
    "sim-iflm": "sim-iflm-configuration.yaml",
}

# The worked example's folders on a Windows support PC. Nothing in a screenshot
# may show the path of the machine the harness ran on.
EXAMPLE_CONFIG_DIR = r"C:\fibsemOS\config"
EXAMPLE_EXPERIMENT_DIR = r"D:\fibsemOS\experiments"

CALLOUT_COLOUR = QColor(230, 57, 70)  # the guide's red: the badges
CALLOUT_LINE = QColor(230, 57, 70, 170)  # the boxes, lighter so they do not shout
CALLOUT_WIDTH = 2
CALLOUT_PAD = 4


class Box:
    """A callout drawn as a region: a light box round the widget plus its badge.

    A bare widget in a callout list gets the badge alone. Regions (a panel, a
    tab bar) want the box, because it says "this area"; a single control does
    not, because the box only repeats the control's own edge.
    """

    def __init__(self, widget: QWidget):
        self.widget = widget


Callout = Union[QWidget, Box]
Widgets = Union[Callout, Sequence[Callout]]

# page name -> render function, in the order the guide lists them
PAGES: Dict[str, Callable[["Harness"], None]] = {}


def page(name: str):
    """Register a page renderer. One function per guide page, in nav order."""

    def register(fn):
        PAGES[name] = fn
        return fn

    return register


class Harness:
    """One application, one window, driven to named states and photographed."""

    def __init__(self, site: Path, scale: int = 1):
        self.site = site
        self.img_root = site / IMG_SUBDIR
        self.scale = scale
        self.manifest: Dict[str, List[str]] = {}
        self._page: Optional[str] = None
        self._tmp = tempfile.TemporaryDirectory(prefix="fibsem-user-guide-")
        self._redirect_user_state(Path(self._tmp.name))

        self.app = QApplication.instance() or QApplication(sys.argv)
        self.app.setStyle("Fusion")
        from fibsem.ui.qt.gc import install_main_thread_gc

        self._gc = install_main_thread_gc(parent=self.app)

        from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
            AutoLamellaSingleWindowUI,
        )

        self.window = AutoLamellaSingleWindowUI()
        self.window.resize(*WINDOW_SIZE)
        self.window.show()
        self.pump(500)

    # -- state isolation ----------------------------------------------------

    @staticmethod
    def _redirect_user_state(tmp: Path) -> None:
        """Point every user-state file at a scratch directory for this run.

        These are module globals read at call time, so rebinding them here is
        enough: is_first_run() sees no registry file (a fresh install), the
        wizard's save lands in the scratch directory, and preferences written
        during the run never reach the machine's own file.
        """
        cfg.USER_CONFIGURATIONS_PATH = str(tmp / "user-configurations.yaml")
        cfg.USER_PREFERENCES_PATH = str(tmp / "user-preferences.yaml")
        cfg.POSITION_PATH = str(tmp / "saved-positions.yaml")
        for name, filename in SIM_CONFIGURATIONS.items():
            cfg.USER_CONFIGURATIONS[name] = {
                "path": os.path.join(cfg.CONFIG_PATH, filename)
            }

    # -- driving ------------------------------------------------------------

    def pump(self, ms: int = 100) -> None:
        """Run the event loop for a while: paints, timers, queued signals."""
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec_()

    @property
    def ui(self):
        return self.window.autolamella_ui

    @property
    def connection(self):
        return self.ui.system_widget

    def select_configuration(self, name: str) -> None:
        combo = self.connection.comboBox_configuration
        if combo.findText(name) == -1:
            combo.addItem(name)
        combo.setCurrentText(name)
        self.pump()

    def connect(self, name: str = "sim-arctis") -> None:
        """Connect to a simulator configuration the way the Connection tab does.

        Idempotent: a page that finds the previous page's connection still up
        keeps it if it is the right one. The Connection button toggles, so
        pressing it while connected would disconnect.
        """
        if self.connection.microscope is not None:
            if self.connection.comboBox_configuration.currentText() == name:
                return
            self.disconnect()
        self.select_configuration(name)
        self.connection.connect_to_microscope()
        # the connect builds the microscope tabs; let them lay out and paint,
        # and let the connect toast expire before anything is photographed
        self.pump(4000)
        if self.connection.microscope is None:
            raise RuntimeError(f"connect to {name} failed")

    def disconnect(self) -> None:
        if self.connection.microscope is not None:
            self.connection.connect_to_microscope()  # toggles
            self.pump(500)

    def first_run(self, on: bool) -> None:
        """Show or hide the first-run offer on the Connection tab."""
        guided_setup.is_first_run = lambda: on
        self.connection.refresh_first_run_offer()
        self.pump()

    def wait_acquisition(self, widget, timeout_ms: int = 30000) -> None:
        """Pump until the image widget's acquisition worker has finished."""
        waited = 0
        while getattr(widget, "is_acquiring", False) and waited < timeout_ms:
            self.pump(100)
            waited += 100
        if getattr(widget, "is_acquiring", False):
            raise RuntimeError("acquisition did not finish")
        self.pump(300)

    def wait_move(self, control, image_widget, timeout_ms: int = 60000) -> None:
        """Pump until a stage move and the images it retakes have finished."""
        waited = 0
        while waited < timeout_ms:
            self.pump(200)
            waited += 200
            if control.pushButton_move.isEnabled() and not getattr(
                image_widget, "is_acquiring", False
            ):
                break
        else:
            raise RuntimeError("stage move did not finish")
        self.pump(500)

    @staticmethod
    def image_point_rect(
        canvas, panel: QWidget, x: int, y: int, radius: int = 16
    ) -> QRect:
        """A box around image pixel (x, y) of ``canvas``, in ``panel`` coordinates.

        matplotlib's display coordinates are physical pixels with y upwards;
        Qt's are logical with y downwards.
        """
        ratio = canvas.devicePixelRatioF()
        dx, dy = canvas._ax.transData.transform((x, y))
        qx, qy = dx / ratio, canvas.height() - dy / ratio
        origin = canvas.mapTo(panel, QPoint(int(qx), int(qy)))
        return QRect(origin.x() - radius, origin.y() - radius, 2 * radius, 2 * radius)

    def show_tab(self, index: int) -> None:
        self.window.tab_widget.setCurrentIndex(index)
        self.pump()

    # -- photographing ------------------------------------------------------

    def begin_page(self, name: str) -> None:
        self._page = name
        self.manifest[name] = []
        (self.img_root / name).mkdir(parents=True, exist_ok=True)

    def shot(
        self,
        name: str,
        target: Optional[QWidget] = None,
        callouts: Optional[Widgets] = None,
        numbered: bool = False,
        crop: bool = False,
        callout_rects: Optional[Sequence[QRect]] = None,
    ) -> Path:
        """Grab ``target`` (the window by default) and write ``<page>/<name>.png``.

        ``callouts`` are widgets to mark, in ``target``'s coordinate space: a
        bare widget gets a numbered badge, a ``Box(widget)`` a light box as
        well. With ``numbered`` each carries its 1-based index for the prose
        to refer to. A callout that is not visible raises: the guide must not
        describe a control the reader cannot see. ``crop`` trims a panel to
        the area its visible children occupy, so a tall tab with three rows
        of controls is not mostly empty.
        """
        assert self._page, "call begin_page first"
        target = target or self.window
        self.pump(150)
        region = QRect(QPoint(0, 0), target.size())
        if crop:
            region = (
                self._occupied(target).adjusted(
                    -CALLOUT_PAD * 2, -CALLOUT_PAD * 2, CALLOUT_PAD * 2, CALLOUT_PAD * 2
                )
                & region
            )
        pixmap = target.grab(region)
        marks: List[Tuple[QRect, bool]] = []
        if callouts:
            marks += self._callout_rects(target, callouts)
        if callout_rects:
            marks += [(r, True) for r in callout_rects]
        if marks:
            self._draw_callouts(
                pixmap,
                [(r.translated(-region.topLeft()), boxed) for r, boxed in marks],
                numbered,
            )
        if self.scale == 1 and pixmap.devicePixelRatio() != 1:
            # a retina grab is 2x; the guide serves 1x
            image = pixmap.toImage().scaled(
                region.width(),
                region.height(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
            image.setDevicePixelRatio(1)
            pixmap = QPixmap.fromImage(image)
        path = self.img_root / self._page / f"{name}.png"
        if not pixmap.save(str(path)):
            raise RuntimeError(f"could not write {path}")
        self.manifest[self._page].append(path.name)
        print(f"  {self._page}/{path.name}  {pixmap.width()}x{pixmap.height()}")
        return path

    @staticmethod
    def _occupied(target: QWidget) -> QRect:
        """The bounding box of the target's visible descendants, in its coordinates."""
        rect = QRect()
        for child in target.findChildren(QWidget):
            if child.isVisible() and child.width() > 0 and child.height() > 0:
                origin = child.mapTo(target, QPoint(0, 0))
                rect = rect.united(QRect(origin, child.size()))
        return rect if rect.isValid() else QRect(QPoint(0, 0), target.size())

    @staticmethod
    def _callout_rects(target: QWidget, callouts: Widgets) -> List[Tuple[QRect, bool]]:
        """Each callout's rectangle in ``target``'s coordinates, and whether
        it is boxed.

        A callout that is not visible raises: the guide must not describe a
        control the reader cannot see.
        """
        if isinstance(callouts, (QWidget, Box)):
            callouts = [callouts]
        marks = []
        for i, callout in enumerate(callouts, start=1):
            boxed = isinstance(callout, Box)
            widget = callout.widget if boxed else callout
            if not widget.isVisible():
                raise RuntimeError(
                    f"callout {i} ({type(widget).__name__}) is not visible"
                )
            marks.append(
                (QRect(widget.mapTo(target, QPoint(0, 0)), widget.size()), boxed)
            )
        return marks

    @staticmethod
    def _draw_callouts(
        pixmap: QPixmap, marks: Sequence[Tuple[QRect, bool]], numbered: bool
    ) -> None:
        """Badge each mark at its top-left corner, and box the ones flagged."""
        # QPainter paints a high-DPI pixmap in logical coordinates already
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(CALLOUT_LINE, CALLOUT_WIDTH)
        painter.setPen(pen)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(16)
        painter.setFont(font)
        # logical size: a high-DPI pixmap is painted in logical coordinates
        ratio = pixmap.devicePixelRatio()
        bounds = QRect(0, 0, int(pixmap.width() / ratio), int(pixmap.height() / ratio))
        inset = CALLOUT_WIDTH // 2 + 1
        bounds = bounds.adjusted(inset, inset, -inset, -inset)
        for i, (box, boxed) in enumerate(marks, start=1):
            rect = box.adjusted(-CALLOUT_PAD, -CALLOUT_PAD, CALLOUT_PAD, CALLOUT_PAD)
            # a box that runs off the grab (a menu item filling its menu) is
            # pulled inside it, so all four sides stay visible
            rect = rect.intersected(bounds)
            if boxed:
                painter.drawRoundedRect(rect, 3, 3)
            if numbered:
                badge = QRect(rect.left() - 12, rect.top() - 12, 24, 24)
                # a box on the grab's edge would put its badge off the image
                badge.moveLeft(max(badge.left(), bounds.left()))
                badge.moveTop(max(badge.top(), bounds.top()))
                painter.setBrush(CALLOUT_COLOUR)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(badge)
                painter.setPen(QColor("white"))
                painter.drawText(badge, Qt.AlignCenter, str(i))
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
        painter.end()

    # -- lifecycle ----------------------------------------------------------

    def write_manifest(self) -> None:
        path = self.img_root / "manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2) + "\n")
        print(f"manifest: {path}")

    def close(self) -> None:
        self.disconnect()
        self.window.close()
        self.pump(200)
        self._tmp.cleanup()


# -- pages ------------------------------------------------------------------


@page("installation")
def render_installation(h: Harness) -> None:
    """The window as it opens on a fresh install: the proof that it runs."""
    h.first_run(True)
    h.show_tab(0)
    h.shot("first-launch")

    # Tools > Create Desktop Shortcut: the menu, open, with the item boxed.
    # A QMenu is its own top-level window, so it is grabbed on its own; the
    # action is not a widget, so its box comes from the menu's own geometry.
    tools = next(a.menu() for a in h.window.menuBar().actions() if a.text() == "Tools")
    tools.popup(h.window.mapToGlobal(QPoint(0, 0)))
    h.pump(300)
    shortcut = next(
        a for a in tools.actions() if a.text().startswith("Create Desktop Shortcut")
    )
    h.shot("tools-menu", target=tools, callout_rects=[tools.actionGeometry(shortcut)])
    tools.close()
    h.pump(200)


@page("getting-started")
def render_getting_started(h: Harness) -> None:
    """Connection tab, the Guided Setup wizard, and the first connection."""
    from fibsem.ui.widgets.guided_setup_dialog import (
        STEP_TITLES,
        GuidedSetupDialog,
    )

    conn = h.connection
    h.first_run(True)
    h.show_tab(0)

    # the offer, and where the manual route starts
    h.shot(
        "connection-tab",
        target=conn,
        callouts=[conn._button_run_wizard, conn.comboBox_configuration],
        numbered=True,
        crop=True,
    )

    # the wizard, one step at a time (modeless, so it can be photographed),
    # filled in as the worked examples the page walks through: an Arctis
    # (compustage) and a pre-tilted-shuttle instrument. The folders are the
    # example's, not this machine's: the defaults would show the checkout and
    # the home directory of whoever ran the harness.
    # The Arctis example runs on the microscope PC (the usual arrangement,
    # address localhost); the shuttle example on a support PC.
    for prefix, model, location, config_name in (
        ("standard", "tfs-hydra", LOCATION_SUPPORT, "Hydra Bay 1"),
        ("arctis", "tfs-arctis", LOCATION_MICROSCOPE, "Arctis Bay 2"),
    ):
        dialog = GuidedSetupDialog(parent=h.window)
        dialog._select_model(model)
        dialog._select_location(location)
        dialog._configuration_dir.setText(EXAMPLE_CONFIG_DIR)
        dialog._experiment_dir.setText(EXAMPLE_EXPERIMENT_DIR)
        dialog.choices.configuration_directory = EXAMPLE_CONFIG_DIR
        dialog.choices.experiment_directory = EXAMPLE_EXPERIMENT_DIR
        dialog._name_edit.setText(config_name)
        dialog.show()
        h.pump(300)
        for index, title in enumerate(STEP_TITLES):
            dialog._show_step(index)
            h.pump(200)
            slug = title.lower().replace(" ", "-")
            h.shot(f"guided-setup-{prefix}-{index + 1}-{slug}", target=dialog)
        dialog.close()
        h.pump(200)

    # the manual route: pick a shipped configuration and connect
    h.first_run(False)
    h.select_configuration("sim-arctis")
    h.shot(
        "select-configuration",
        target=conn,
        callouts=[conn.comboBox_configuration, conn.pushButton_connect_to_microscope],
        numbered=True,
        crop=True,
    )
    h.connect("sim-arctis")
    h.shot(
        "connected",
        callouts=[Box(conn._frame_status), Box(h.ui.tabWidget.tabBar())],
        numbered=True,
    )

    # the lie of the land, once connected: every region the guide will name
    h.shot(
        "around-the-window",
        callouts=[
            Box(h.window.menuBar()),
            Box(h.window.tab_widget.tabBar()),
            Box(h.window.view_controller.widget),
            Box(h.ui.tabWidget.tabBar()),
            Box(h.window.status_bar),
            h.window.run_workflow_btn,
        ],
        numbered=True,
    )


@page("imaging")
def render_imaging(h: Harness) -> None:
    """The Image tab: the two panels, the first images, field of view, saving, live."""
    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    iw = h.ui.image_widget
    h.ui.tabWidget.setCurrentWidget(iw)
    h.pump(300)
    panels = h.window.view_controller.widget._all_panels
    sem_panel, fib_panel = panels[0], panels[2]
    settings = iw.image_settings_widget
    # the save path defaults to this checkout's log directory; the example's
    settings.path_edit.setText(EXAMPLE_EXPERIMENT_DIR)
    h.pump()

    # the tab before anything is acquired: beam panel, image panel, the buttons
    h.shot(
        "image-tab",
        target=iw,
        callouts=[
            Box(iw.dual_beam_widget),
            Box(iw.image_group),
            iw.pushButton_take_all_images,
            iw.pushButton_start_acquisition,
        ],
        numbered=True,
        crop=True,
    )

    # first images of the grid, both beams, at the shipped field of view
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    h.shot("first-images")

    # the view's toolbar, left to right, and the contrast popover it opens
    sem_canvas = h.window.view_controller.sem_canvas
    h.shot(
        "view-toolbar",
        target=sem_panel,
        callouts=[
            sem_canvas.btn_toggle_ruler,
            sem_canvas.btn_contrast,
            sem_canvas.btn_toggle_crosshair,
            sem_canvas.btn_toggle_scalebar,
            sem_canvas.btn_reset_view,
        ],
        numbered=True,
    )
    sem_canvas.btn_contrast.setChecked(True)
    sem_canvas.toggle_contrast()
    h.pump(300)
    h.shot("view-contrast", target=sem_panel)
    sem_canvas.btn_contrast.setChecked(False)
    sem_canvas.toggle_contrast()
    h.pump()

    # the same place at three fields of view, SEM only
    for hfw in (400.0, 150.0, 50.0):
        settings.hfw_spinbox.setValue(hfw)
        iw.acquire_sem_image()
        h.wait_acquisition(iw)
        h.shot(f"sem-{int(hfw)}um", target=sem_panel)
    settings.hfw_spinbox.setValue(150.0)
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    h.shot("fib-150um", target=fib_panel)

    # the two panels on their own
    h.shot("beam-panel", target=iw.dual_beam_widget)
    h.shot("image-panel", target=iw.image_group)

    # saving: tick save, name the file, box the three controls
    settings.save_image_check.setChecked(True)
    settings.filename_edit.setText("grid-overview")
    h.pump()
    h.shot(
        "save-settings",
        target=iw.image_group,
        callouts=[
            settings.save_image_check,
            settings.path_edit,
            settings.filename_edit,
        ],
        numbered=True,
    )
    settings.save_image_check.setChecked(False)
    h.pump()

    # live acquisition: the green border and the LIVE badge
    iw.toggle_live_acquisition()
    h.pump(2500)
    h.shot("live", callouts=[iw.pushButton_start_acquisition])
    iw.toggle_live_acquisition()
    h.pump(500)


@page("movement")
def render_movement(h: Harness) -> None:
    """The Movement tab, the orientations, click-to-move, and coincidence."""
    import numpy as np

    from fibsem import utils
    from fibsem.structures import BeamType
    from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    mw = h.ui.movement_widget
    iw = h.ui.image_widget
    ctrl = mw.control_widget
    h.ui.tabWidget.setCurrentWidget(mw)
    h.pump(300)
    panels = h.window.view_controller.widget._all_panels
    sem_panel, fib_panel = panels[0], panels[2]
    sem_canvas = h.window.view_controller.sem_canvas
    fib_canvas = h.window.view_controller.fib_canvas
    microscope = h.connection.microscope

    # the tab
    h.shot(
        "movement-tab",
        target=mw,
        callouts=[
            Box(mw.position_widget),
            ctrl.pushButton_move,
            ctrl.pushButton_move_to_sem_orientation,
            ctrl.pushButton_move_to_fib_orientation,
            ctrl.doubleSpinBox_milling_angle,
            ctrl.pushButton_move_to_milling_angle,
            Box(mw.saved_positions_panel),
        ],
        numbered=True,
        crop=True,
    )

    # the three orientations, drawn by the app's own stage diagram at the
    # poses the app itself derives from the shipped configurations: a
    # pre-tilted shuttle and a compustage
    for kind, filename in (
        ("shuttle", "sim-iflm-configuration.yaml"),
        ("compustage", "sim-arctis-configuration.yaml"),
    ):
        demo, _ = utils.setup_session(
            config_path=os.path.join(cfg.CONFIG_PATH, filename),
            manufacturer="Demo",
            setup_logging=False,
        )
        pre_tilt = float(demo.system.stage.shuttle_pre_tilt)
        reference = demo.get_orientation("SEM")
        for name in ("SEM", "MILLING", "FIB"):
            pose = demo.get_orientation(name)
            half_turn = abs(abs(np.degrees(pose.r - reference.r)) % 360 - 180) < 1
            diagram = StageDiagram(pre_tilt=pre_tilt, orientation=name)
            diagram.set_orientation(
                name, stage_tilt=float(np.degrees(pose.t)), mirrored=half_turn
            )
            diagram.resize(560, 240)
            diagram.show()
            h.pump(200)
            h.shot(f"orientation-{kind}-{name.lower()}", target=diagram)
            diagram.close()
        demo.disconnect()

    # click to move: a feature off-centre in the SEM, double-clicked, is centred
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    image = iw.eb_image
    hgt, wid = image.data.shape[:2]
    x, y = int(wid * 0.70), int(hgt * 0.30)
    h.shot(
        "click-before",
        target=sem_panel,
        callout_rects=[h.image_point_rect(sem_canvas, sem_panel, x, y)],
    )
    ctrl._on_canvas_double_click(BeamType.ELECTRON, x, y, [])
    h.wait_move(ctrl, iw)
    h.shot("click-after", target=sem_panel)

    # coincidence: the fiducial cross, centred in the SEM, sits off-centre in
    # the FIB by the boot height error; Alt + double-click on it in the FIB
    # brings the two views into coincidence
    microscope.system.sim["sample"]["fiducial"] = True
    microscope._setup_sample_scene()
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    h.shot("coincidence-before")
    # where the cross sits in the FIB view: the boot height error, seen at the
    # ion column's angle (vertical_move inverts exactly this), below centre
    ib_image = iw.ib_image
    hgt, wid = ib_image.data.shape[:2]
    dy = microscope._sample_scene.coincidence_offset * np.sin(
        np.radians(microscope.system.ion.column_tilt)
    )
    y_cross = int(hgt / 2 + dy / ib_image.metadata.pixel_size.y)
    h.shot(
        "coincidence-click",
        target=fib_panel,
        callout_rects=[h.image_point_rect(fib_canvas, fib_panel, wid // 2, y_cross)],
    )
    ctrl._on_canvas_double_click(BeamType.ION, wid // 2, y_cross, ["Alt"])
    h.wait_move(ctrl, iw)
    h.shot("coincidence-after")
    microscope.system.sim["sample"]["fiducial"] = False
    microscope._setup_sample_scene()


# -- entry point --------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pages", nargs="*", help="pages to render (default: all)")
    parser.add_argument(
        "--site", type=Path, default=DEFAULT_SITE, help="docs site checkout"
    )
    parser.add_argument(
        "--scale", type=int, choices=(1, 2), default=1, help="output pixel scale"
    )
    parser.add_argument("--list", action="store_true", help="list pages and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in PAGES:
            print(name)
        return 0
    unknown = [p for p in args.pages if p not in PAGES]
    if unknown:
        parser.error(f"unknown page(s): {', '.join(unknown)}; see --list")
    if not (args.site / "package.json").exists():
        parser.error(f"{args.site} is not the docs site checkout (pass --site)")

    selected = args.pages or list(PAGES)
    harness = Harness(site=args.site, scale=args.scale)
    try:
        for name in selected:
            print(f"{name}:")
            harness.begin_page(name)
            PAGES[name](harness)
        harness.write_manifest()
    finally:
        harness.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
