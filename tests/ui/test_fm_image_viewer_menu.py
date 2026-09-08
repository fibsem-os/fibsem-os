"""The Fluorescence Image Viewer opens from the View menu, with or without an
experiment (FIB-942).

It used to sit on the Development menu, hidden unless dev mode was on, and refused to
open without an experiment loaded, so the user guide could not say where it was. It
is a file viewer: the experiment only chooses which folder Load opens in.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    yield window
    # closeEvent ends in app.quit(); on the shared test QApplication that latches an
    # interrupt every later QEventLoop.exec_() returns from. Stub it for the close.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def _menu_titles(window, title):
    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == title)
    return [a.text() for a in menu.actions() if a.text()]


def test_the_viewer_is_on_the_view_menu(main_ui):
    assert "Fluorescence Image Viewer..." in _menu_titles(main_ui, "View")


def test_and_no_longer_on_the_development_menu(main_ui):
    assert not any("Image Viewer" in t for t in _menu_titles(main_ui, "Development"))


def test_it_opens_without_an_experiment(main_ui, qapp, monkeypatch):
    ui = main_ui.autolamella_ui
    assert ui.experiment is None
    toasts = []
    from fibsem.applications.autolamella.ui import AutoLamellaUI as module

    monkeypatch.setattr(
        module.notification_service, "show_toast", lambda *a, **k: toasts.append(a)
    )
    main_ui.action_open_fm_image_viewer.trigger()
    qapp.processEvents()
    window = ui._fm_image_viewer_window
    try:
        assert window is not None and window.isVisible()
        assert toasts == []
        assert os.path.isdir(window.start_directory)
    finally:
        window.close()
