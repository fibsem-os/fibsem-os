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
# The worked example's experiment: what the AutoLamella pages run against
EXAMPLE_EXPERIMENT_NAME = "yeast-grid-a"
EXAMPLE_EXPERIMENT_METADATA = {
    "description": "Yeast on grid A, first session",
    "user": "Operator",
    "project": "CLEM pilot",
    "organisation": "Example Institute",
}

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
        # acquisitions made with no experiment open (the FM's, for one) save
        # into the working directory; keep them out of the checkout
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)

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
        # the stage module imports the holder paths by name, so rebinding
        # cfg's is not enough: a calibration would land in the real file
        from fibsem.microscopes import _stage as stage_module

        holder_path = str(tmp / "sample-holder.yaml")
        occupancy_path = str(tmp / "sample-holder-occupancy.yaml")
        cfg.SAMPLE_HOLDER_CONFIGURATION_PATH = holder_path
        cfg.SAMPLE_HOLDER_OCCUPANCY_PATH = occupancy_path
        stage_module.SAMPLE_HOLDER_CONFIGURATION_PATH = holder_path
        stage_module.SAMPLE_HOLDER_OCCUPANCY_PATH = occupancy_path
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

    def wait_fm(self, fm_control, timeout_ms: int = 300000) -> None:
        """Pump until the fluorescence widget's acquisition has finished."""
        waited = 0
        self.pump(500)
        while (
            fm_control.is_acquisition_active or fm_control._has_worker
        ) and waited < timeout_ms:
            self.pump(200)
            waited += 200
        if fm_control.is_acquisition_active or fm_control._has_worker:
            raise RuntimeError("fluorescence acquisition did not finish")
        self.pump(800)

    def ensure_experiment(self):
        """Open the worked example's experiment, creating it on the first call.

        Created the way the Create Experiment dialog does it, in the scratch
        directory, with the shipped task protocol copied in; then adopted by
        the app, which is what enables the AutoLamella tabs.
        """
        if self.ui.experiment is not None:
            return self.ui.experiment
        from fibsem.applications.autolamella import config as al_cfg
        from fibsem.applications.autolamella.structures import (
            AutoLamellaTaskProtocol,
            Experiment,
        )

        path = os.path.join(self._tmp.name, EXAMPLE_EXPERIMENT_NAME)
        experiment = Experiment.create(
            path=path,
            name=EXAMPLE_EXPERIMENT_NAME,
            metadata=dict(EXAMPLE_EXPERIMENT_METADATA),
        )
        experiment.task_protocol = AutoLamellaTaskProtocol.load(
            al_cfg.TASK_PROTOCOL_PATH
        )
        experiment.task_protocol.save(os.path.join(experiment.path, "protocol.yaml"))
        self.ui._adopt_experiment(experiment)
        self.pump(1500)
        return experiment

    def show_main_tab(self, title: str) -> None:
        """Select a main-window tab by its label (Microscope, Overview, ...)."""
        tabs = self.window.tab_widget
        for i in range(tabs.count()):
            if tabs.tabText(i).strip() == title:
                tabs.setCurrentIndex(i)
                self.pump(300)
                return
        raise RuntimeError(f"no main tab named {title!r}")

    def cell_positions(self, count: int = 3, reach: float = 300e-6):
        """Stage positions of ``count`` cells on film near the scene's anchor.

        Read from the simulator's own feature list and checked against its
        support masks, so a marked position is on a cell and never on a grid
        bar, a hole or a rip: what a user would pick on an overview.
        """
        import numpy as np

        from fibsem.projection import BeamStageProjection
        from fibsem.structures import BeamType

        microscope = self.connection.microscope
        scene = microscope._sample_scene
        projection = BeamStageProjection.from_microscope(
            microscope, beam_type=BeamType.ELECTRON
        )
        reference = scene.reference_position
        chosen = []
        cells = [f for f in scene.features if f.kind == "cell"]
        for cell in sorted(cells, key=lambda f: -f.sigma):
            if abs(cell.x) > reach or abs(cell.y) > reach * 0.7:
                continue
            # the cell and a ring 20 um round it must all be plain film
            ring = 20e-6
            xs = np.array([cell.x, cell.x - ring, cell.x + ring, cell.x, cell.x])
            ys = np.array([cell.y, cell.y, cell.y, cell.y - ring, cell.y + ring])
            bars, holes, rips, rim, beyond = scene.film_masks(xs, ys)
            if bars.any() or holes.any() or rips.any() or rim.any() or beyond.any():
                continue
            if any(
                (cell.x - c.x) ** 2 + (cell.y - c.y) ** 2 < (80e-6) ** 2 for c in chosen
            ):
                continue
            chosen.append(cell)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            raise RuntimeError(f"only {len(chosen)} cells on film within reach")
        return [projection.from_plane(c.x, c.y, reference) for c in chosen]

    def ensure_lamellae(self, count: int = 3):
        """The worked example's lamellae, one on each of ``count`` cells."""
        experiment = self.ensure_experiment()
        if len(experiment.positions) >= count:
            return list(experiment.positions)
        for position in self.cell_positions(count)[len(experiment.positions) :]:
            self.ui.add_new_lamella(stage_position=position)
            self.pump(400)
        self.pump(500)
        return list(experiment.positions)

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
        clicks: Optional[Sequence[Tuple[QRect, str]]] = None,
        height: Optional[int] = None,
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
        pad = QPoint(0, 0)
        if crop:
            # room for a badge outside the outermost box; a target with no
            # such room of its own is padded with its background instead
            margin = CALLOUT_PAD + 14
            wanted = self._occupied(target).adjusted(-margin, -margin, margin, margin)
            if height is not None:
                # a list that stretches to fill its tab is mostly empty rows
                wanted.setHeight(min(wanted.height(), height))
            region = wanted & region
            pad = region.topLeft() - wanted.topLeft()
            region_size = wanted.size()
        pixmap = target.grab(region)
        if crop and (pad.x() or pad.y() or region.size() != region_size):
            ratio = pixmap.devicePixelRatio()
            padded = QPixmap(
                int(region_size.width() * ratio), int(region_size.height() * ratio)
            )
            padded.setDevicePixelRatio(ratio)
            padded.fill(QColor(pixmap.toImage().pixel(0, 0)))
            painter = QPainter(padded)
            painter.drawPixmap(pad, pixmap)
            painter.end()
            pixmap = padded
            region = wanted
        shift = -region.topLeft()
        marks: List[Tuple[QRect, bool]] = []
        if callouts:
            marks += self._callout_rects(target, callouts)
        if callout_rects:
            marks += [(r, True) for r in callout_rects]
        if marks:
            self._draw_callouts(
                pixmap, [(r.translated(shift), boxed) for r, boxed in marks], numbered
            )
        if clicks:
            self._draw_clicks(pixmap, [(r.translated(shift), t) for r, t in clicks])
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

    @staticmethod
    def _draw_clicks(pixmap: QPixmap, clicks: Sequence[Tuple[QRect, str]]) -> None:
        """Mark where the mouse went: a box, a crosshair at the click itself,
        and the gesture written beneath ("Double click", "Alt + Double click")."""
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(13)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for box, text in clicks:
            painter.setPen(QPen(CALLOUT_LINE, CALLOUT_WIDTH))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(box, 3, 3)
            c = box.center()
            arm = box.width() // 4
            painter.setPen(QPen(CALLOUT_COLOUR, CALLOUT_WIDTH))
            painter.drawLine(c.x() - arm, c.y(), c.x() + arm, c.y())
            painter.drawLine(c.x(), c.y() - arm, c.x(), c.y() + arm)
            # the gesture, on a pill under the box (above it if there is no room)
            width = metrics.horizontalAdvance(text) + 16
            height = metrics.height() + 6
            top = box.bottom() + 8
            if top + height > int(pixmap.height() / pixmap.devicePixelRatio()) - 4:
                top = box.top() - 8 - height
            pill = QRect(c.x() - width // 2, top, width, height)
            painter.setPen(Qt.NoPen)
            painter.setBrush(CALLOUT_COLOUR)
            painter.drawRoundedRect(pill, height // 2, height // 2)
            painter.setPen(QColor("white"))
            painter.drawText(pill, Qt.AlignCenter, text)
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
        os.chdir(self._cwd)
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
        clicks=[(h.image_point_rect(sem_canvas, sem_panel, x, y), "Double click")],
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
        clicks=[
            (
                h.image_point_rect(fib_canvas, fib_panel, wid // 2, y_cross),
                "Alt + Double click",
            )
        ],
    )
    ctrl._on_canvas_double_click(BeamType.ION, wid // 2, y_cross, ["Alt"])
    h.wait_move(ctrl, iw)
    h.shot("coincidence-after")

    # eucentricity: coincident, but with the surface 30 um above the tilt
    # axis, a tilt to the MILLING orientation swings the cross off centre
    scene = microscope._sample_scene
    scene.tilt_axis_offset = 30e-6
    ctrl.move_to_orientation("MILLING")
    h.wait_move(ctrl, iw)
    h.shot("eucentric-after-tilt")
    scene.tilt_axis_offset = 0.0
    ctrl.move_to_orientation("SEM")
    h.wait_move(ctrl, iw)
    microscope.system.sim["sample"]["fiducial"] = False
    microscope._setup_sample_scene()


@page("milling")
def render_milling(h: Harness) -> None:
    """The Milling tab: a stage, its pattern on the FIB view, a run, the trench."""
    from fibsem import conversions
    from fibsem.structures import Point

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    iw = h.ui.image_widget
    ctrl = h.ui.movement_widget.control_widget
    mv = h.ui.milling_task_config_widget
    cw = mv.config_widget
    stages_w = cw.milling_stages_widget
    stage_list = stages_w._list
    runner = mv.milling_widget
    panels = h.window.view_controller.widget._all_panels
    fib_panel = panels[2]
    fib_canvas = h.window.view_controller.fib_canvas

    # at the milling angle, with fresh images at the field of view the task
    # will mill at: lamella-sized patterns are hard to see at 150 um
    ctrl.move_to_orientation("MILLING")
    h.wait_move(ctrl, iw)
    iw.image_settings_widget.hfw_spinbox.setValue(80.0)
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    h.ui.tabWidget.setCurrentWidget(mv)
    cw.field_of_view_spinbox.setValue(80.0)
    h.pump(300)

    # the tab before any stage exists
    h.shot(
        "milling-tab",
        target=mv,
        callouts=[
            Box(cw.core_panel),
            Box(cw.alignment_panel),
            Box(cw.acquisition_panel),
            Box(stages_w),
            runner.pushButton_run_milling,
        ],
        numbered=True,
        crop=True,
    )

    # one stage, added with the + button: a rectangle at the beam centre
    stage_list._header.btn_add.click()
    h.pump(500)
    row = stage_list._list.itemWidget(stage_list._list.item(0))
    stage = stage_list.get_stages()[0]
    # a rectangle big enough to see at this field of view
    stage.pattern.width = 20e-6
    stage.pattern.height = 8e-6
    stage_list.select_stage(stage)
    h.pump(300)
    stages_w._pattern_widget.set_pattern(stage.pattern)
    stages_w._on_pattern_changed(stage.pattern)
    h.pump(500)
    from fibsem.ui.widgets.custom_widgets import TitledPanel

    def panel_of(widget):
        while widget is not None and not isinstance(widget, TitledPanel):
            widget = widget.parentWidget()
        return widget

    editor_panels = [
        panel_of(stages_w._milling_widget),
        panel_of(stages_w._pattern_widget),
        panel_of(stages_w._strategy_widget),
    ]
    for panel in editor_panels:
        panel.expand()
    h.pump(300)
    h.shot(
        "stage-row",
        target=stages_w,
        callouts=[
            row.checkbox,
            row.name_edit,
            row.depth_spin,
            row.current_combo,
            row.strategy_combo,
            row.btn_remove,
            stage_list._header.btn_add,
        ],
        numbered=True,
        crop=True,
    )
    h.shot("pattern-on-fib", target=fib_panel)

    # a few of the shapes, each at its shipped size, at the beam centre
    from fibsem.milling.patterning import get_pattern

    for name in ("Rectangle", "Trench", "Fiducial", "MicroExpansion"):
        pattern = get_pattern(name)
        stages_w._pattern_widget.set_pattern(pattern)
        stages_w._on_pattern_changed(pattern)
        h.pump(400)
        h.shot(f"pattern-{name.lower()}", target=fib_panel)
    # a task with two stages: trenches, then micro-expansion cuts beside them
    stage_list._header.btn_add.click()
    h.pump(400)
    stages = stage_list.get_stages()
    for stage_, name, label in (
        (stages[0], "Trench", "Trench"),
        (stages[1], "MicroExpansion", "Micro-expansion"),
    ):
        stage_.name = label
        stage_list.select_stage(stage_)
        h.pump(300)
        pattern = get_pattern(name)
        stages_w._pattern_widget.set_pattern(pattern)
        stages_w._on_pattern_changed(pattern)
        h.pump(400)
    stage_list.refresh_stage(stages[0])
    stage_list.refresh_stage(stages[1])
    h.pump(400)
    h.shot("multi-stage-list", target=stage_list, crop=True)
    h.shot("multi-stage-fib", target=fib_panel)

    # back to the single rectangle for the move and the run
    stage_list.remove_stage(stages[1])
    stage = stages[0]
    stage.name = "Milling Stage 1"
    stage_list.select_stage(stage)
    h.pump(300)
    pattern = get_pattern("Rectangle")
    pattern.width, pattern.height = 20e-6, 8e-6
    stages_w._pattern_widget.set_pattern(pattern)
    stages_w._on_pattern_changed(pattern)
    h.pump(400)
    h.shot(
        "stage-editor",
        target=stages_w._detail_widget,
        callouts=[Box(panel) for panel in editor_panels],
        numbered=True,
        crop=True,
    )

    # moving the pattern: right-click > Move All Patterns Here
    image = iw.ib_image
    hgt, wid = image.data.shape[:2]
    x, y = int(wid * 0.62), int(hgt * 0.42)
    h.shot(
        "pattern-move-click",
        target=fib_panel,
        clicks=[(h.image_point_rect(fib_canvas, fib_panel, x, y), "Right click")],
    )
    point = conversions.image_to_microscope_image_coordinates(
        coord=Point(x=x, y=y), image=image.data, pixelsize=image.metadata.pixel_size.x
    )
    mv._move_patterns(point, move_all=True)
    h.pump(500)
    h.shot("pattern-moved", target=fib_panel)

    # run it: the progress bars and Stop while it mills, then the trench
    runner.run_milling()
    h.pump(2500)
    h.shot("milling-running", callouts=[runner.pushButton_stop_milling])
    waited = 0
    while runner.is_milling and waited < 120000:
        h.pump(250)
        waited += 250
    if runner.is_milling:
        raise RuntimeError("milling did not finish")
    h.pump(500)
    # the trench itself: pattern overlay hidden (the eye button), fresh images
    stage_list._header.btn_eye.setChecked(True)
    h.pump(300)
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    h.shot("after-milling")
    h.shot("after-milling-fib", target=fib_panel)
    stage_list._header.btn_eye.setChecked(False)
    h.pump(300)

    # the two-stage task run for real, on clean film to the left
    stage_list._header.btn_add.click()
    h.pump(400)
    stages = stage_list.get_stages()
    for stage_, name, label in (
        (stages[0], "Trench", "Trench"),
        (stages[1], "MicroExpansion", "Micro-expansion"),
    ):
        stage_.name = label
        stage_list.select_stage(stage_)
        h.pump(300)
        pattern = get_pattern(name)
        stages_w._pattern_widget.set_pattern(pattern)
        stages_w._on_pattern_changed(pattern)
        stage_list.refresh_stage(stage_)
        h.pump(400)
    stage_list.select_stage(stages[0])
    h.pump(300)
    x, y = int(wid * 0.35), int(hgt * 0.55)
    point = conversions.image_to_microscope_image_coordinates(
        coord=Point(x=x, y=y), image=image.data, pixelsize=image.metadata.pixel_size.x
    )
    mv._move_patterns(point, move_all=True)
    h.pump(500)
    h.shot("multi-stage-before", target=fib_panel)
    runner.run_milling()
    waited = 0
    while (runner.is_milling or waited < 1000) and waited < 240000:
        h.pump(250)
        waited += 250
    if runner.is_milling:
        raise RuntimeError("milling did not finish")
    stage_list._header.btn_eye.setChecked(True)
    h.pump(300)
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    h.shot("multi-stage-after", target=fib_panel)
    h.shot("multi-stage-after-window")
    stage_list._header.btn_eye.setChecked(False)
    h.pump(300)

    # back to a clean state for whoever renders next
    for stage_ in list(stage_list.get_stages()):
        stage_list.remove_stage(stage_)
    iw.image_settings_widget.hfw_spinbox.setValue(150.0)
    h.pump()
    ctrl.move_to_orientation("SEM")
    h.wait_move(ctrl, iw)


@page("sample-holder")
def render_sample_holder(h: Harness) -> None:
    """The Sample tab: the holder's slots, naming grids, calibrating, moving."""
    from fibsem.structures import FibsemStagePosition

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-iflm")
    iw = h.ui.image_widget
    ctrl = h.ui.movement_widget.control_widget
    sw = h.ui.sample_widget
    hw = sw.holder_widget
    microscope = h.connection.microscope
    h.ui.tabWidget.setCurrentWidget(sw)
    h.pump(300)

    # the shuttle as shipped: two slots, nothing loaded, nothing calibrated
    row0, row1 = hw._row_widget(0), hw._row_widget(1)
    h.shot(
        "sample-tab",
        target=sw,
        callouts=[hw.facts_label, row0.name_edit, row0.btn_move, hw.btn_calibrate],
        numbered=True,
        crop=True,
        height=220,
    )

    # name the grids in each slot
    for row, name in ((row0, "Grid A"), (row1, "Grid B")):
        row.name_edit.setText(name)
        row.name_edit.editingFinished.emit()
        h.pump(200)
    h.shot("grids-named", target=sw, crop=True, height=220)

    # the calibration wizard, step by step
    hw._on_calibrate()
    dialog = hw._calibration_dialog
    h.pump(300)
    h.shot("calibrate-1-holder", target=dialog)
    dialog._on_next()
    h.pump(200)
    dialog._on_move_to_orientation()
    waited = 0
    while dialog._worker is not None and dialog._worker.is_alive() and waited < 30000:
        h.pump(200)
        waited += 200
    h.pump(500)
    dialog._refresh_orientation_status()
    h.pump(200)
    h.shot("calibrate-2-orientation", target=dialog)
    dialog._on_next()
    h.pump(300)
    dialog._on_capture()
    h.pump(300)
    h.shot("calibrate-3-slot-1", target=dialog)
    dialog._on_next()
    h.pump(300)
    # the second grid sits 4 mm along the shuttle
    microscope.move_stage_relative(FibsemStagePosition(x=4e-3))
    h.pump(500)
    dialog._on_capture()
    h.pump(300)
    h.shot("calibrate-4-slot-2", target=dialog)
    dialog._on_next()
    h.pump(300)
    h.shot("calibrate-5-review", target=dialog)
    dialog._on_next()  # save
    h.pump(500)
    h.shot("slots-calibrated", target=sw, crop=True, height=220)

    # with a grid at every calibrated, occupied slot, a slot move shows the
    # other grid (and the holder between them) at a wide field of view
    microscope._sample_scene.grids_from_holder = True
    iw.image_settings_widget.hfw_spinbox.setValue(2000.0)
    holder = hw.current_holder
    slots = sorted(holder.slots.values(), key=lambda s_: s_.index)
    for slot, name in ((slots[0], "slot-1"), (slots[1], "slot-2")):
        hw._on_move_slot(slot)
        h.wait_move(ctrl, iw)
        iw.acquire_reference_images()
        h.wait_acquisition(iw)
        h.shot(f"at-{name}")
    microscope._sample_scene.grids_from_holder = False
    iw.image_settings_widget.hfw_spinbox.setValue(150.0)
    hw._on_move_slot(slots[0])
    h.wait_move(ctrl, iw)

    # the Arctis: holder plus the autoloader magazine
    h.connect("sim-arctis")
    sw = h.ui.sample_widget
    h.ui.tabWidget.setCurrentWidget(sw)
    h.pump(300)
    h.shot(
        "sample-tab-arctis",
        target=sw,
        callouts=[Box(sw.holder_widget), Box(sw.loader_widget)],
        numbered=True,
        crop=True,
        height=760,
    )


@page("fluorescence")
def render_fluorescence(h: Harness) -> None:
    """The Fluorescence tab and the FM view, on the simulated Arctis."""
    from fibsem.fm.structures import ChannelSettings

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    iw = h.ui.image_widget
    ctrl = h.ui.movement_widget.control_widget
    fmc = h.ui.fm_control_widget
    fm = h.connection.microscope.fm
    ocw = fmc.objectiveControlWidget
    quad = h.window.view_controller.widget
    fm_panel = quad._all_panels[1]
    fm_widget = quad.fm_widget

    ctrl.move_to_orientation("SEM")
    h.wait_move(ctrl, iw)
    iw.acquire_reference_images()
    h.wait_acquisition(iw)
    h.ui.tabWidget.setCurrentWidget(fmc)
    h.pump(300)

    # the tab as it opens: objective retracted, one channel. The panels live
    # in a scroll area, so they are grabbed from its content widget (which
    # lays out at full height) and the buttons from the tab itself.
    panels_widget = fmc.objectivePanel.parentWidget()
    for panel in (fmc.cameraPanel, fmc.autofocusPanel, fmc.histogramPanel):
        panel._btn_collapse.setChecked(False)
    fmc.zParametersPanel._btn_collapse.setChecked(True)
    h.pump(300)
    h.shot(
        "fluorescence-tab",
        target=panels_widget,
        callouts=[
            Box(fmc.objectivePanel),
            Box(fmc.channelPanel),
            Box(fmc.zParametersPanel),
        ],
        numbered=True,
        crop=True,
    )
    h.shot(
        "fluorescence-buttons",
        target=fmc,
        callouts=[
            fmc.pushButton_acquire_single_image,
            fmc.pushButton_toggle_acquisition,
            fmc.pushButton_acquire_zstack,
            fmc.pushButton_run_autofocus,
        ],
        numbered=True,
    )

    # insert the objective, as the button does once its dialog is answered
    # (the stage is at 0 deg tilt, so no move is needed first)
    ocw._set_objective_actions_enabled(False)
    worker = ocw._insert_objective_worker(None)
    worker.returned.connect(ocw._on_objective_action_finished)
    worker.errored.connect(ocw._on_objective_action_error)
    worker.start()
    waited = 0
    while worker.is_alive() and waited < 30000:
        h.pump(200)
        waited += 200
    h.pump(500)
    # and to its focus position, as Move to Focus Position does (that button
    # confirms a move this large in a dialog)
    fm.objective.move_absolute(fm.objective.focus_position)
    ocw.update_objective_position_labels()
    h.pump(400)
    h.shot("objective-inserted", target=fmc.objectivePanel, crop=True)

    # four channels: reflection and the three lines the simulator's dyes answer
    lines = sorted(fm.filter_set.available_excitation_wavelengths)

    def nearest(target):
        return min(lines, key=lambda w: abs(w - target))

    channels = [
        ChannelSettings(
            name="Reflection",
            excitation_wavelength=nearest(550),
            emission_wavelength=None,
            color="gray",
            exposure_time=0.05,
            power=0.1,
        ),
        ChannelSettings(
            name="DAPI",
            excitation_wavelength=nearest(405),
            emission_wavelength="Fluorescence",
            color="blue",
            exposure_time=0.1,
            power=0.2,
        ),
        ChannelSettings(
            name="GFP",
            excitation_wavelength=nearest(488),
            emission_wavelength="Fluorescence",
            color="green",
            exposure_time=0.1,
            power=0.2,
        ),
        ChannelSettings(
            name="mCherry",
            excitation_wavelength=nearest(561),
            emission_wavelength="Fluorescence",
            color="red",
            exposure_time=0.1,
            power=0.2,
        ),
    ]
    fmc.channelSettingsWidget.channel_settings = channels
    h.pump(400)
    h.shot("channels", target=fmc.channelPanel, crop=True)

    # Acquire Image takes the selected channel only; Acquire Z-Stack takes
    # every channel at every plane, so a two-plane stack (the smallest it
    # accepts) is the way to one image with all four channels in it
    import glob

    from fibsem.fm.structures import FluorescenceImage
    from fibsem.ui.fm.widgets.fm_image_viewer_widget import FMImageViewerWidget

    quad.set_selected("fm")
    zp = fmc.zParametersWidget
    zp.doubleSpinBox_zstep.setValue(2.0)
    zp.doubleSpinBox_zmin.setValue(-1.0)
    zp.doubleSpinBox_zmax.setValue(1.0)
    h.pump(200)
    fmc.pushButton_acquire_zstack.click()
    h.wait_fm(fmc)
    # the Microscope tab's FM view shows the frame just taken, one channel
    h.shot("fm-view-live", target=fm_panel)

    # the standalone viewer is where a multi-channel image is composed: load
    # the file the acquisition wrote
    def newest(pattern):
        files = sorted(
            glob.glob(os.path.join(h._tmp.name, pattern)), key=os.path.getmtime
        )
        return files[-1]

    viewer = FMImageViewerWidget(start_directory=h._tmp.name)
    viewer.resize(1180, 700)
    viewer.show()
    h.pump(300)
    viewer.add_image(FluorescenceImage.load(newest("z-stack-*.ome.tiff")))
    h.pump(800)
    vcanvas = viewer.canvas
    h.shot(
        "viewer",
        target=viewer,
        callouts=[
            Box(vcanvas),
            Box(viewer.listWidget_images),
            viewer.pushButton_load_image,
        ],
        numbered=True,
    )
    h.shot("composite", target=vcanvas)
    layers = vcanvas.layers
    for layer in layers:
        for other in layers:
            other.visible = other is layer
        vcanvas._recomposite()
        h.pump(300)
        h.shot(f"channel-{layer.name.lower()}", target=vcanvas)
    for layer in layers:
        layer.visible = True
    vcanvas._recomposite()
    h.pump(300)

    # the view's own controls: the channels button and its panel, with colour,
    # opacity, gamma and contrast for the selected channel
    vcanvas._btn_layers.setChecked(True)
    vcanvas._toggle_layers_panel()
    h.pump(400)
    h.shot(
        "fm-view-toolbar", target=vcanvas, callouts=[vcanvas._btn_layers], numbered=True
    )
    h.shot("fm-layers-panel", target=vcanvas._panel)
    vcanvas._btn_layers.setChecked(False)
    vcanvas._toggle_layers_panel()
    h.pump(300)

    # a z-stack: the parameters, then the slice controls under the view
    zp.doubleSpinBox_zmin.setValue(-10.0)
    zp.doubleSpinBox_zmax.setValue(10.0)
    zp.doubleSpinBox_zstep.setValue(2.0)
    h.pump(200)
    h.shot("z-parameters", target=fmc.zParametersPanel, crop=True)
    fmc.channelSettingsWidget.channel_settings = channels[1:3]
    h.pump(300)
    fmc.pushButton_acquire_zstack.click()
    h.wait_fm(fmc)
    stack = FluorescenceImage.load(newest("z-stack-*.ome.tiff"))
    viewer.add_image(stack)
    viewer.display_image(stack)
    # a loaded stack opens in max projection; the slice controls show
    # only with it off
    vcanvas.set_max_projection(False)
    h.pump(800)
    h.shot(
        "z-stack",
        target=vcanvas,
        callouts=[
            vcanvas._z_prev,
            vcanvas._z_slider,
            vcanvas._z_next,
            vcanvas._btn_mip,
        ],
        numbered=True,
    )
    viewer.close()
    h.pump(300)

    # retract, and leave the tab as found
    ocw._set_objective_actions_enabled(False)
    worker = ocw._retract_objective_worker()
    worker.returned.connect(ocw._on_objective_action_finished)
    worker.errored.connect(ocw._on_objective_action_error)
    worker.start()
    waited = 0
    while worker.is_alive() and waited < 30000:
        h.pump(200)
        waited += 200
    h.pump(300)


@page("experiments")
def render_experiments(h: Harness) -> None:
    """Create Experiment, and the window once an experiment is open."""
    from fibsem.applications.autolamella.ui.autolamella_create_experiment_widget import (
        AutoLamellaCreateExperimentWidget,
    )

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")

    # the dialog, filled in as the worked example (shown, not accepted: the
    # experiment itself is created in the scratch directory below)
    dialog = AutoLamellaCreateExperimentWidget(parent=h.window)
    dialog.show()
    h.pump(500)
    dialog.lineEdit_experiment_name.setText(EXAMPLE_EXPERIMENT_NAME)
    dialog.lineEdit_experiment_description.setText(
        EXAMPLE_EXPERIMENT_METADATA["description"]
    )
    dialog.lineEdit_experiment_user.setText(EXAMPLE_EXPERIMENT_METADATA["user"])
    dialog.lineEdit_experiment_project.setText(EXAMPLE_EXPERIMENT_METADATA["project"])
    dialog.lineEdit_experiment_organisation.setText(
        EXAMPLE_EXPERIMENT_METADATA["organisation"]
    )
    dialog.lineEdit_experiment_directory.setText(EXAMPLE_EXPERIMENT_DIR)
    # the shipped protocol loaded by default; its path is this checkout's, so
    # show where it would be on the example machine
    dialog.lineEdit_protocol_path.setText(EXAMPLE_CONFIG_DIR + r"\task-protocol.yaml")
    h.pump(300)
    h.shot(
        "create-experiment",
        target=dialog,
        callouts=[
            Box(dialog.lineEdit_experiment_name.parentWidget()),
            dialog.lineEdit_experiment_directory,
            dialog.btn_select_protocol,
            dialog.btn_ok,
        ],
        numbered=True,
    )
    dialog.close()
    h.pump(300)

    h.ensure_experiment()
    h.show_tab(0)
    h.pump(300)
    h.shot(
        "experiment-open",
        callouts=[Box(h.window.tab_widget.tabBar()), Box(h.window.status_bar)],
        numbered=True,
    )
    h.ui.tabWidget.setCurrentWidget(h.ui.tab)
    h.pump(300)
    h.shot("experiment-tab", target=h.ui.tab, crop=True)


@page("protocols")
def render_protocols(h: Harness) -> None:
    """The Protocol tab: the task list, a task's parameters, Global Edit."""
    from fibsem.applications.autolamella.ui.autolamella_global_task_editor_dialog import (
        AutoLamellaGlobalTaskEditDialog,
    )

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    h.ensure_experiment()
    h.show_main_tab("Protocol")
    editor = h.window.task_widget
    editor.task_list_widget.select("Rough Milling")
    h.pump(500)
    h.shot(
        "protocol-tab",
        callouts=[
            Box(editor.protocol_header),
            Box(editor.task_list_widget),
            Box(editor.task_parameters_config_widget),
            Box(editor.milling_task_editor),
            editor.pushButton_sync_to_lamella,
            editor.pushButton_open_global_editor,
            editor.pushButton_open_lamella_defaults,
        ],
        numbered=True,
    )

    # the File menu's protocol entries
    file_menu = next(
        a.menu() for a in h.window.menuBar().actions() if a.text().strip() == "File"
    )
    file_menu.popup(h.window.mapToGlobal(QPoint(0, 0)))
    h.pump(300)
    rects = [
        file_menu.actionGeometry(a)
        for a in file_menu.actions()
        if a.text() in ("Load Protocol", "Save Protocol")
    ]
    h.shot("file-menu", target=file_menu, callout_rects=rects, numbered=True)
    file_menu.close()
    h.pump(200)

    # Global Edit, shown rather than run
    dialog = AutoLamellaGlobalTaskEditDialog(h.ui.experiment, parent=h.window)
    dialog.resize(640, 900)
    dialog.show()
    h.pump(400)
    h.shot("global-edit", target=dialog)
    dialog.close()
    h.pump(200)


@page("overview")
def render_overview(h: Harness) -> None:
    """The Overview tab: a tiled SEM overview, positions marked on it, FIB
    overviews at two orientations, and the FM overview tab."""
    from fibsem.applications.autolamella.ui.overview_container_tab import (
        MODALITY_FLUORESCENCE as MODALITY_FM,
    )
    from fibsem.structures import BeamType

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    h.ensure_experiment()
    iw = h.ui.image_widget
    ctrl = h.ui.movement_widget.control_widget
    microscope = h.connection.microscope
    ctrl.move_to_orientation("SEM")
    h.wait_move(ctrl, iw)
    h.show_main_tab("Overview")
    container = h.window.overview_tab
    beam_tab = container.beam_tab
    ow = beam_tab.overview
    settings = ow.settings_widget

    def acquire(beam, rows=3, cols=3, hfw_um=400.0):
        settings.combo_beam.set_value(beam)
        settings.grid.spin_rows.setValue(rows)
        settings.grid.spin_cols.setValue(cols)
        settings.spin_hfw.setValue(hfw_um)
        h.pump(200)
        ow._confirm = lambda *_: True  # the confirmation dialog, answered
        ow.acquire()
        waited = 0
        h.pump(500)
        while ow.is_acquiring and waited < 300000:
            h.pump(250)
            waited += 250
        if ow.is_acquiring:
            raise RuntimeError("overview did not finish")
        h.pump(1000)

    # the tab before anything is acquired
    settings.combo_beam.set_value(BeamType.ELECTRON)
    settings.grid.spin_rows.setValue(3)
    settings.grid.spin_cols.setValue(3)
    settings.spin_hfw.setValue(400.0)
    h.pump(300)
    h.shot(
        "overview-tab",
        callouts=[
            Box(settings),
            ow.button_acquire,
            Box(ow.view_strip),
            Box(ow.overview_list),
            Box(beam_tab.lamella_list),
        ],
        numbered=True,
    )

    # a 3 x 3 SEM overview of the grid centre
    acquire(BeamType.ELECTRON)
    h.shot("sem-overview")

    # three positions marked on it, on cells (what "Add New Position Here"
    # does at a right-click)
    for position in h.cell_positions(3):
        ow.position_add_requested.emit(position)
        h.pump(400)
    h.pump(500)
    h.shot("positions-marked")

    # the ion beam's view of the same place, at the SEM orientation and at the
    # MILLING orientation: two more entries in the view strip
    # (smaller tiles: the ion beam's tiles step further across the stage
    # than their width, and a 3 x 3 at 400 um runs past the stage limits)
    acquire(BeamType.ION, hfw_um=250.0)
    h.shot("fib-overview-sem-orientation")
    ctrl.move_to_orientation("MILLING")
    h.wait_move(ctrl, iw)
    # at the milling angle the beam grazes the surface, so a row of tiles
    # spans a long way along the stage: one row of three is the strip a
    # milling-orientation overview usually is
    acquire(BeamType.ION, rows=1, cols=3, hfw_um=250.0)
    h.shot("fib-overview-milling-orientation")
    h.shot("view-strip", target=ow.view_strip, crop=True)
    ctrl.move_to_orientation("SEM")
    h.wait_move(ctrl, iw)

    # the fluorescence overview tab, as it opens (Arctis only)
    container.set_modality(MODALITY_FM)
    h.pump(500)
    fow = container.fm_tab.overview
    h.shot(
        "fm-overview-tab",
        callouts=[fow.button_move_to_fm, fow.button_acquire],
        numbered=True,
    )
    container.set_modality("FIBSEM")
    h.pump(300)


@page("lamella")
def render_lamella(h: Harness) -> None:
    """A lamella: the Experiment tab's list, the Lamella tab's cards and
    editor, a card's actions, and the Review sub-tab."""
    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    ctrl = h.ui.movement_widget.control_widget
    iw = h.ui.image_widget
    ctrl.move_to_orientation("SEM")
    h.wait_move(ctrl, iw)
    lamellae = h.ensure_lamellae(3)

    # the Experiment tab: the list, + to add at the current position, and the
    # selected lamella's details
    h.show_main_tab("Microscope")
    h.ui.tabWidget.setCurrentWidget(h.ui.tab)
    h.ui.lamella_list.select(lamellae[0].name) if hasattr(
        h.ui.lamella_list, "select"
    ) else None
    h.pump(400)
    h.shot("experiment-tab", target=h.ui.tab, crop=True, height=520)

    # the Lamella tab: cards on the left, the selected lamella's own task
    # configuration and its images on the right
    h.show_main_tab("Lamella")
    cards = h.window.lamella_card_container
    editor = h.window.lamella_widget
    editor.select_lamella(lamellae[0].name)
    h.pump(600)
    editor.listWidget_selected_task.select("Rough Milling")
    h.pump(600)
    h.shot(
        "lamella-tab",
        callouts=[
            Box(cards),
            Box(editor.listWidget_selected_task),
            Box(editor.task_parameters_config_widget),
            editor.pushButton_apply_to_other,
        ],
        numbered=True,
    )

    # one card's actions menu
    from fibsem.applications.autolamella.ui.lamella_card_widget import LamellaCardWidget

    card = cards.findChildren(LamellaCardWidget)[0]
    menu = card._btn_actions.menu()
    menu.popup(card._btn_actions.mapToGlobal(QPoint(0, card._btn_actions.height())))
    h.pump(300)
    rects = [menu.actionGeometry(a) for a in menu.actions() if a.text()]
    h.shot("card-actions", target=menu)
    menu.close()
    h.pump(200)
    h.shot(
        "card",
        target=card,
        callouts=[card._btn_actions, card._btn_defect],
        numbered=True,
    )

    # the Review sub-tab: task images, empty before a task has run
    right_tabs = h.window.lamella_task_image_widget.parentWidget()
    while right_tabs is not None and not hasattr(right_tabs, "setCurrentWidget"):
        right_tabs = right_tabs.parentWidget()
    right_tabs.setCurrentWidget(h.window.lamella_task_image_widget)
    h.pump(400)
    h.shot(
        "review-tab", target=h.window.lamella_task_image_widget, crop=True, height=300
    )
    right_tabs.setCurrentIndex(0)
    h.pump(200)


@page("tasks")
def render_tasks(h: Harness) -> None:
    """The task types on offer, and the shipped workflow's task list."""
    from fibsem.applications.autolamella.ui.autolamella_task_config_editor import (
        AddTaskDialog,
    )

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    h.ensure_lamellae(3)

    # the Add Task dialog, with its list of task types open
    dialog = AddTaskDialog(h.ui.experiment.task_protocol.task_config, parent=h.window)
    # the example plugin installed for development is not part of the product
    combo = dialog.comboBox_task_type
    for i in reversed(range(combo.count())):
        if "EXAMPLE" in str(combo.itemData(i)):
            combo.removeItem(i)
    dialog.show()
    h.pump(400)
    h.shot("add-task", target=dialog)
    dialog.comboBox_task_type.showPopup()
    h.pump(400)
    popup = dialog.comboBox_task_type.view().window()
    h.shot("task-types", target=popup)
    dialog.comboBox_task_type.hidePopup()
    dialog.close()
    h.pump(200)

    # the shipped protocol's workflow as the Workflow tab lists it
    h.show_main_tab("Workflow")
    ww = h.window.lamella_workflow_widget
    h.pump(400)
    h.shot("starting-workflow", target=ww.workflow, crop=True)


@page("workflows")
def render_workflows(h: Harness) -> None:
    """The Workflow tab, a supervised run with its prompts, and the Grids view."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as main_module

    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    ctrl = h.ui.movement_widget.control_widget
    iw = h.ui.image_widget
    ctrl.move_to_orientation("SEM")
    h.wait_move(ctrl, iw)
    lamellae = h.ensure_lamellae(3)
    h.show_main_tab("Workflow")
    ww = h.window.lamella_workflow_widget

    # one lamella, the first two tasks
    ww.lamella_list.set_all_selected(False)
    ww.lamella_list._row(0).checkbox.setChecked(True)
    ww.workflow.set_all_selected(False)
    for i in (0, 1):
        ww.workflow._row(i).checkbox.setChecked(True)
    h.pump(400)
    h.shot(
        "workflow-tab",
        callouts=[
            Box(ww.lamella_list),
            Box(ww.info),
            Box(ww.workflow),
            h.window.run_workflow_btn,
        ],
        numbered=True,
    )
    row = ww.workflow._row(2)
    h.shot(
        "task-row",
        target=row,
        callouts=[
            row.checkbox,
            row.btn_schedule,
            row.btn_supervise,
            row.btn_edit,
            row.btn_remove,
        ],
        numbered=True,
    )

    # run it, supervised: the preflight dialog is answered, each prompt is
    # photographed on the way and then answered
    main_module.confirm_run_workflow_dialog = lambda *_args, **_kwargs: True
    # the completion summary is modal; it is shown by hand below instead
    h.ui._show_workflow_summary = lambda: None
    seen = set()

    def run_selected(task_indices, queue_shots=False):
        # the lists rebuild after a run, so both selections are made afresh
        ww.lamella_list.set_all_selected(False)
        ww.lamella_list._row(0).checkbox.setChecked(True)
        ww.workflow.set_all_selected(False)
        for i in task_indices:
            ww.workflow._row(i).checkbox.setChecked(True)
        h.pump(300)
        h.show_main_tab("Workflow")
        h.window.run_workflow_btn.click()
        answered_milling = set()
        waited = 0
        h.pump(1500)
        if not h.ui.is_workflow_running:
            raise RuntimeError(f"workflow did not start for tasks {task_indices}")
        while (h.ui.is_workflow_running or waited < 2000) and waited < 1200000:
            h.pump(250)
            waited += 250
            if not h.ui.WAITING_FOR_USER_INTERACTION:
                continue
            h.pump(1200)
            if queue_shots:
                # the queue mid-run: one item active, one still to come, and a
                # pending item's actions
                queue_shots = False
                h.show_main_tab("Workflow")
                h.pump(400)
                h.shot(
                    "queue-running",
                    callouts=[Box(h.window.workflow_timeline)],
                    numbered=True,
                )
                progress = h.window.workflow_timeline
                pending = None
                for i in range(len(progress._outer._steps)):
                    menu = progress.build_row_menu(i)
                    if menu is not None and any(
                        a.text() == "Remove from queue" and a.isEnabled()
                        for a in menu.actions()
                    ):
                        pending = (i, menu)
                        break
                if pending is not None:
                    i, menu = pending
                    menu.popup(h.window.mapToGlobal(QPoint(900, 300)))
                    h.pump(300)
                    h.shot("queue-row-menu", target=menu)
                    menu.close()
                    h.pump(200)
            question = h.ui.ui_responder.pending_question
            if callable(question):
                question = question()
            kind = type(question).__name__ if question is not None else "Confirm"
            key = getattr(getattr(question, "config", None), "name", None)
            # a milling prompt is one image per task, the others one each
            shot_name = f"prompt-{kind.lower()}"
            if kind == "RunMillingTask" and key:
                shot_name += "-" + key.lower().replace(" ", "-")
            if shot_name not in seen:
                if not seen:
                    h.shot(
                        "attention-required",
                        callouts=[h.window.user_attention_btn],
                        numbered=True,
                    )
                seen.add(shot_name)
                h.show_main_tab("Microscope")
                h.ui.tabWidget.setCurrentWidget(h.ui.tab)
                h.pump(500)
                buttons = [
                    b
                    for b in (h.ui.pushButton_yes, h.ui.pushButton_no)
                    if b.isVisible()
                ]
                h.shot(
                    shot_name,
                    callouts=[Box(h.ui.label_instructions), *buttons],
                    numbered=True,
                )
            # a milling prompt comes back after the run (Yes = Run Milling
            # again, No = Continue): Yes once per task, then Continue
            if kind == "RunMillingTask" and key in answered_milling:
                h.ui.pushButton_no.click()
            else:
                if kind == "RunMillingTask":
                    answered_milling.add(key)
                h.ui.pushButton_yes.click()
            h.pump(800)
        if h.ui.is_workflow_running:
            raise RuntimeError("workflow did not finish")
        h.pump(1500)

    def lamella_tab_shot(name, task):
        h.show_main_tab("Lamella")
        editor = h.window.lamella_widget
        editor.select_lamella(lamellae[0].name)
        h.pump(500)
        editor.listWidget_selected_task.select(task)
        h.pump(800)
        h.shot(name)

    # the first two tasks: position and fiducial
    run_selected((0, 1))
    from fibsem.ui.widgets.workflow_summary_dialog import WorkflowSummaryDialog

    summary_dialog = WorkflowSummaryDialog(h.ui._last_run_summary, parent=h.window)
    summary_dialog.show()
    h.pump(500)
    h.shot("workflow-summary", target=summary_dialog)
    summary_dialog.close()
    h.pump(300)
    lamella_tab_shot("lamella-after-fiducial", "Mill Fiducial")

    # then rough milling and polishing
    run_selected((2, 3), queue_shots=True)
    lamella_tab_shot("lamella-after-polishing", "Polishing")
    h.show_main_tab("Workflow")
    h.pump(300)
    h.shot("workflow-finished")

    # the Grids view, behind the grid-workflow flag
    h.window._preferences.features.grid_workflow = True
    h.window._apply_grid_workflow_visibility()
    h.window.workflow_left_tabs.setCurrentWidget(h.window.grid_workflow_widget)
    h.pump(500)
    h.shot("grids-view", target=h.window.grid_workflow_widget, crop=True)
    h.window.workflow_left_tabs.setCurrentIndex(0)
    h.window._preferences.features.grid_workflow = False
    h.window._apply_grid_workflow_visibility()
    h.pump(300)


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
