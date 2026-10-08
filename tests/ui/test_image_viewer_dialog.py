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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtTest import QTest
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
    ViewerItem,
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
    viewer.show()
    viewer.show_path(fm_path)
    _loaded(viewer, fm_path)
    assert viewer.stack.currentWidget() is viewer.fm_widget.parentWidget()
    fm = viewer.fm_widget
    fm.set_max_projection(False)  # a plane, not the projection
    start = fm.current_z
    fm.canvas.setFocus()  # shortcuts follow the focus, as in the app

    def press(key):
        QTest.keyClick(fm.canvas, key)  # through the canvas, as a user's key goes

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
    first = open_image_viewer(
        window, [ViewerItem(beam_path, title="01-fancy-mite › Rough Milling")]
    )
    second = open_image_viewer(
        window, [ViewerItem(fm_path, title="01-fancy-mite › Fluorescence")]
    )
    assert first is second
    assert isinstance(first, ImageViewerDialog) and not first.isModal()
    assert first.windowTitle() == "01-fancy-mite › Fluorescence"
    _loaded(first.viewer, fm_path)
    first.close()
    window.deleteLater()


# --- stepping through a lamella's images -------------------------------------------


@pytest.fixture
def items(beam_path, fm_path, tmp_path):
    sem_path = _beam_image(BeamType.ELECTRON).save(str(tmp_path / "sem.tif"))
    return [
        ViewerItem(sem_path, title="01-fancy-mite › Rough Milling", label="SEM"),
        ViewerItem(beam_path, title="01-fancy-mite › Rough Milling", label="FIB"),
        ViewerItem(fm_path, title="01-fancy-mite › Acquire Fluorescence"),
    ]


def test_it_opens_on_the_clicked_image_and_says_where_it_is(items):
    viewer = ImageViewer()
    viewer.set_items(items, index=1)
    _loaded(viewer, items[1].path)
    assert viewer.position_label.text() == "FIB · 2 of 3"
    assert viewer.title_label.text() == "01-fancy-mite › Rough Milling"
    assert [b.isChecked() for b in viewer._film_buttons] == [False, True, False]


def test_left_and_right_step_through_and_swap_to_the_fm_viewer(items):
    viewer = ImageViewer()
    viewer.show()
    viewer.set_items(items, index=1)
    _loaded(viewer, items[1].path)
    viewer.canvas.setFocus()
    # Through the matplotlib canvas, which takes key presses for itself.
    QTest.keyClick(viewer.canvas, Qt.Key_Right)
    _loaded(viewer, items[2].path)
    assert viewer.stack.currentWidget() is viewer.fm_widget.parentWidget()
    assert viewer.title_label.text() == "01-fancy-mite › Acquire Fluorescence"
    QTest.keyClick(viewer.fm_widget, Qt.Key_Left)
    _loaded(viewer, items[1].path)
    assert viewer.stack.currentWidget() is viewer.canvas.parentWidget()
    viewer.close()


def test_it_stops_at_either_end(items):
    viewer = ImageViewer()
    viewer.set_items(items, index=0)
    assert not viewer.previous_button.isEnabled() and viewer.next_button.isEnabled()
    viewer.step(-1)
    assert viewer.index == 0
    viewer.go_to(2)
    viewer.step(1)
    assert viewer.index == 2
    assert not viewer.next_button.isEnabled()


def test_a_filmstrip_tile_goes_to_its_image(items):
    viewer = ImageViewer()
    viewer.set_items(items, index=0)
    viewer._film_buttons[2].click()
    assert viewer.index == 2
    _loaded(viewer, items[2].path)


def test_an_unnamed_image_is_named_by_the_file_once_read(items):
    viewer = ImageViewer()
    viewer.set_items(items, index=2)
    assert viewer._film_buttons[2].text() == "3", "a number until the file says"
    _loaded(viewer, items[2].path)
    assert viewer._film_buttons[2].text() == "FM"
    assert viewer.position_label.text() == "FM · 3 of 3"


def test_one_image_has_no_filmstrip_or_arrows(beam_path):
    viewer = ImageViewer()
    viewer.set_items([ViewerItem(beam_path, label="FIB")])
    assert viewer.filmstrip.isHidden() and viewer.next_button.isHidden()
    assert viewer.position_label.text() == "FIB"
    assert viewer.hint_label.text() == "Esc to close"


def test_the_window_title_follows_the_image(items):
    dialog = ImageViewerDialog()
    dialog.show_items(items, index=0)
    dialog.viewer.go_to(2)
    assert dialog.windowTitle() == "01-fancy-mite › Acquire Fluorescence"
    _loaded(dialog.viewer, items[2].path)
    dialog.close()


def test_the_history_tab_hands_over_every_image_in_task_order(tmp_path, monkeypatch):
    from fibsem.applications.autolamella.ui import lamella_task_image_widget as history

    opened = []
    monkeypatch.setattr(
        history,
        "open_image_viewer",
        lambda parent, items, index: opened.append((items, index)),
    )
    widget = history.LamellaTaskImageWidget()

    class _Lamella:
        name = "01-fancy-mite"

    widget._lamella = _Lamella()
    sem, fib, fm = (str(tmp_path / n) for n in ("eb.tif", "ib.tif", "fm.ome.tiff"))
    widget._task_names.update({sem: "Rough Milling", fib: "Rough Milling"})
    widget._task_names[fm] = "Acquire Fluorescence"
    widget._kinds.update({sem: "SEM", fib: "FIB"})
    tile = QPixmap(8, 8)
    widget._pixmap_cache[fib] = tile
    widget._open_expanded(fib)

    ((items, index),) = opened
    assert index == 1
    assert [i.path for i in items] == [sem, fib, fm]
    assert [i.label for i in items] == ["SEM", "FIB", ""]
    assert items[1].title == "01-fancy-mite › Rough Milling"
    assert items[2].title == "01-fancy-mite › Acquire Fluorescence"
    assert items[1].thumbnail is tile and items[0].thumbnail is None


def test_the_fm_stack_count_is_shared_with_the_export():
    """One rule for a stack's planes, read the same by the export, the quad view and
    the viewer."""
    from fibsem.imaging.export import z_stack

    assert z_stack(_fm_image(slices=5)) == (5, 500e-9)
    assert z_stack(_fm_image(slices=1)) is None
    np.testing.assert_equal(_fm_image(slices=5).data.shape[-3], 5)


# --- files dropped from the desktop ------------------------------------------------


def _drag(paths):
    """(mime, enter, drop) for dragging *paths*. Keep the mime data: the events hold
    only a pointer to it, and Qt reads freed memory if Python collects it."""
    from PyQt5.QtCore import QMimeData, QPoint, QUrl
    from PyQt5.QtGui import QDragEnterEvent, QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    enter = QDragEnterEvent(
        QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier
    )
    drop = QDropEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    return mime, enter, drop


def test_a_viewer_takes_no_drops_unless_asked(beam_path):
    """History and Grids › Results show a lamella's or a grid's own images: nothing
    dropped may join them."""
    viewer = ImageViewer()
    assert not viewer.acceptDrops()
    mime, enter, _ = _drag([beam_path])
    viewer.dragEnterEvent(enter)
    assert not enter.isAccepted()


def test_dropped_images_join_the_filmstrip_and_the_last_is_shown(beam_path, fm_path):
    viewer = ImageViewer()
    viewer.show()
    viewer.set_accepts_drops(True)
    mime, enter, drop = _drag([beam_path, fm_path])

    viewer.dragEnterEvent(enter)
    assert enter.isAccepted()
    assert viewer.drop_hint.isVisible(), "the window says it will take them"

    viewer.dropEvent(drop)
    assert not viewer.drop_hint.isVisible()
    assert [i.path for i in viewer.items] == [beam_path, fm_path]
    assert viewer.index == 1
    _loaded(viewer, fm_path)
    assert viewer.title_label.text() == os.path.basename(fm_path)
    viewer.close()


def test_only_image_files_are_taken(beam_path, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image")
    viewer = ImageViewer()
    viewer.set_accepts_drops(True)
    mime, enter, _ = _drag([notes])
    viewer.dragEnterEvent(enter)
    assert not enter.isAccepted()
    mime, _, drop = _drag([notes, beam_path])
    viewer.dropEvent(drop)
    assert [i.path for i in viewer.items] == [beam_path]


def test_the_canvases_leave_drops_to_the_viewer():
    """Qt hands a drop to the nearest widget that accepts drops; a canvas that took
    them would swallow every file dropped on the image."""
    viewer = ImageViewer()
    viewer.set_accepts_drops(True)
    for child in (viewer.canvas, viewer.fm_widget, viewer.fm_widget.canvas):
        assert not child.acceptDrops()


def test_the_fm_image_viewer_takes_drops(fm_path):
    from fibsem.ui.fm.widgets.fm_image_viewer_widget import FMImageViewerWidget

    widget = FMImageViewerWidget()
    assert widget.image_viewer.acceptDrops()
    image = _fm_image()
    widget.add_image(image)
    mime, _, drop = _drag([fm_path])
    widget.image_viewer.dropEvent(drop)
    assert widget.image_viewer.index == 1
    _loaded(widget.image_viewer, fm_path)
    # A held image is found among the viewer's items, dropped files included.
    widget.display_image(image)
    assert widget.image_viewer.index == 0


def test_a_viewer_closed_while_it_reads_does_not_take_the_app_down(
    beam_path, monkeypatch
):
    """The FM Image Viewer is replaced on each reopen, perhaps while a dropped file is
    still being read. Qt aborts on a thread destroyed while it runs, so a read must
    outlive its viewer; its result then goes nowhere."""
    import gc
    import threading

    from PyQt5 import sip

    release = threading.Event()
    real_load = image_viewer_dialog.load_viewer_image

    def held(path):
        release.wait(10)
        return real_load(path)

    monkeypatch.setattr(image_viewer_dialog, "load_viewer_image", held)
    viewer = ImageViewer()
    before = set(image_viewer_dialog._RUNNING)
    viewer.show_path(beam_path)
    (read,) = image_viewer_dialog._RUNNING - before
    # Destroyed now: deleteLater waits for an event loop a test never returns to.
    sip.delete(viewer)
    del viewer
    gc.collect()
    release.set()
    _wait_until(lambda: read not in image_viewer_dialog._RUNNING)
    assert read not in image_viewer_dialog._RUNNING, "the read never finished"
