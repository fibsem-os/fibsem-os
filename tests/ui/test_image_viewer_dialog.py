"""The image viewer: a saved image on the real canvas, at full resolution (FIB-1189).

It replaces a dialog that shrank every image to 1024 px before you could zoom. These
tests read real files from disk, through the same off-thread load the app uses.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget

sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_view_info_bar import (  # noqa: E402
    _beam_image,
    _fm_image,
    _preferences,  # noqa: F401 - autouse: bars read a temporary preferences file
)

from fibsem.imaging.export import image_fields  # noqa: E402
from fibsem.structures import BeamType  # noqa: E402
from fibsem.ui.widgets import image_viewer_dialog  # noqa: E402
from fibsem.ui.widgets.image_viewer_dialog import (  # noqa: E402
    ImageViewer,
    ImageViewerDialog,
    open_image_viewer,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _wait_until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)


def _loaded(viewer: ImageViewer, path: str) -> None:
    _wait_until(lambda: viewer.path == path)
    assert viewer.path == path, "the image never arrived"


@pytest.fixture
def beam_path(tmp_path):
    return _beam_image(BeamType.ION, voltage=30e3).save(str(tmp_path / "fib.tif"))


@pytest.fixture
def fm_path(tmp_path):
    return _fm_image(slices=5).save(str(tmp_path / "stack.ome.tiff"))


def test_a_beam_image_opens_at_full_resolution(beam_path):
    viewer = ImageViewer()
    viewer.show_path(beam_path)
    _loaded(viewer, beam_path)
    assert viewer.stack.currentWidget() is viewer.canvas.parentWidget()
    extent = viewer.canvas._content_extent()
    assert extent is not None
    width = extent[1] - extent[0]
    assert round(width) == 1536, "every pixel of the file, not a 1024 px copy"


def test_it_has_the_bar_of_its_kind(beam_path):
    viewer = ImageViewer()
    viewer.show()
    viewer.show_path(beam_path)
    _loaded(viewer, beam_path)
    assert not viewer.fib_bar.isHidden() and viewer.sem_bar.isHidden()
    viewer.fib_bar.resize(2000, viewer.fib_bar.height())
    shown = {f.label: f.value for f in viewer.fib_bar.visible_fields()}
    expected = {f.label: f.value for f in image_fields(viewer.image).fields if f.label}
    assert shown["HV"] == expected["HV"] == "30 kV"


def test_an_image_that_does_not_say_what_it_is_gets_no_bar():
    viewer = ImageViewer()
    image = _beam_image()
    image.metadata = None
    viewer.show_image(image)
    assert viewer.sem_bar.isHidden() and viewer.fib_bar.isHidden()


def test_a_stack_opens_in_the_fm_viewer_and_up_and_down_step_its_planes(fm_path):
    viewer = ImageViewer()
    viewer.show_path(fm_path)
    _loaded(viewer, fm_path)
    assert viewer.stack.currentWidget() is viewer.fm_widget.parentWidget()
    fm = viewer.fm_widget
    fm.set_max_projection(False)  # a plane, not the projection
    start = fm.current_z

    def press(key):
        viewer.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))

    press(Qt.Key_Up)
    assert fm.current_z == start + 1
    shown = {f.key: f.value for f in viewer.fm_bar.visible_fields()}
    assert shown["z"].startswith(f"{start + 2} of 5"), "the bar's Z follows the plane"
    press(Qt.Key_Down)
    assert fm.current_z == start


def test_the_clicked_tile_shows_until_the_file_arrives(beam_path, monkeypatch):
    held = []
    monkeypatch.setattr(
        image_viewer_dialog._Loader, "start", lambda self: held.append(self)
    )
    viewer = ImageViewer()
    tile = QPixmap(64, 40)
    tile.fill(Qt.red)
    viewer.show_path(beam_path, placeholder=tile)
    assert viewer.stack.currentWidget() is viewer.loading_label
    assert not viewer.loading_label.pixmap().isNull()
    assert not viewer.export_button.isEnabled(), "nothing to export yet"


def test_a_later_click_wins_over_a_slow_read(beam_path, fm_path, monkeypatch):
    """The first file is held back until the second has landed, so its late
    arrival is what is being tested, whatever the disk is doing."""
    import threading

    second_shown = threading.Event()
    real_load = image_viewer_dialog.load_viewer_image

    def load(path):
        if path == beam_path:
            second_shown.wait(10)
        return real_load(path)

    monkeypatch.setattr(image_viewer_dialog, "load_viewer_image", load)
    viewer = ImageViewer()
    viewer.show_path(beam_path)
    viewer.show_path(fm_path)
    _loaded(viewer, fm_path)
    second_shown.set()
    _wait_until(lambda: not viewer._loaders)
    assert not viewer._loaders, "the slow read never finished"
    assert viewer.path == fm_path, "a late read replaced the image clicked after it"


def test_a_file_that_cannot_be_read_says_so(tmp_path):
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not a tiff")
    viewer = ImageViewer()
    viewer.show_path(str(bad))
    _wait_until(lambda: "Couldn't open" in viewer.loading_label.text())
    assert "broken.tif" in viewer.loading_label.text()
    assert viewer.image is None


def test_export_opens_the_export_dialog_for_the_image_shown(beam_path, monkeypatch):
    from fibsem.ui.widgets import image_export_dialog

    opened = []
    monkeypatch.setattr(
        image_export_dialog.ImageExportDialog,
        "exec_",
        lambda self: opened.append(self.image),
    )
    viewer = ImageViewer()
    viewer.show_path(beam_path)
    _loaded(viewer, beam_path)
    viewer.export_button.click()
    assert len(opened) == 1
    assert opened[0].path == beam_path and opened[0].kind == "FIB"


def test_one_non_modal_window_per_app_window_reused(beam_path, fm_path):
    window = QWidget()
    first = open_image_viewer(window, beam_path, title="01-fancy-mite › Rough Milling")
    second = open_image_viewer(window, fm_path, title="01-fancy-mite › Fluorescence")
    assert first is second
    assert isinstance(first, ImageViewerDialog) and not first.isModal()
    assert first.windowTitle() == "01-fancy-mite › Fluorescence"
    _loaded(first.viewer, fm_path)
    first.close()
    window.deleteLater()


def test_the_history_tab_opens_its_tile_in_the_viewer(tmp_path, monkeypatch):
    from fibsem.applications.autolamella.ui import lamella_task_image_widget as history

    opened = []
    monkeypatch.setattr(
        history, "open_image_viewer", lambda *args, **kwargs: opened.append(kwargs)
    )
    widget = history.LamellaTaskImageWidget()

    class _Lamella:
        name = "01-fancy-mite"

    widget._lamella = _Lamella()
    path = str(tmp_path / "ref_final_ib.tif")
    widget._task_names[path] = "Rough Milling"
    tile = QPixmap(8, 8)
    widget._pixmap_cache[path] = tile
    widget._open_expanded(path)
    assert opened == [{"title": "01-fancy-mite › Rough Milling", "placeholder": tile}]


def test_the_fm_stack_count_is_shared_with_the_export():
    """One rule for a stack's planes, read the same by the export, the quad view and
    the viewer."""
    from fibsem.imaging.export import z_stack

    assert z_stack(_fm_image(slices=5)) == (5, 500e-9)
    assert z_stack(_fm_image(slices=1)) is None
    np.testing.assert_equal(_fm_image(slices=5).data.shape[-3], 5)
