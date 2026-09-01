"""Tools -> Create Desktop Shortcut... on the real main window.

Window construction follows test_mainui_workflow_status.py (the minimap stub is
the only non-real piece; see that module's docstring). The location dialog is
the one modal the handler owns, so the tests stand in for the user there —
QFileDialog.getExistingDirectory returns tmp_path instead of blocking on a
native dialog — and everything else is the real QAction, handler, and file
writer. No microscope connection is needed: the action acts on the install,
not a session.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QFileDialog, QMessageBox


@pytest.fixture(scope="module")
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    original = module.AutoLamellaSingleWindowUI.add_minimap_tab
    module.AutoLamellaSingleWindowUI.add_minimap_tab = lambda self: None
    try:
        window = module.AutoLamellaSingleWindowUI()
    finally:
        module.AutoLamellaSingleWindowUI.add_minimap_tab = original
    yield window
    # closeEvent ends in app.quit(); on the shared test QApplication that latches
    # and breaks later QEventLoop.exec_() calls — run close with quit stubbed.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture()
def chosen_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path)
    )
    return tmp_path


def test_the_action_lives_in_the_tools_menu(main_ui):
    tools = [
        a.menu()
        for a in main_ui.menuBar().actions()
        if a.menu() is not None and a.text() == "Tools"
    ]
    assert len(tools) == 1
    assert main_ui.action_create_desktop_shortcut in tools[0].actions()


def test_triggering_the_action_writes_the_shortcut_to_the_chosen_directory(
    main_ui, chosen_directory
):
    from fibsem.tools import desktop_shortcut

    main_ui.action_create_desktop_shortcut.trigger()

    created = desktop_shortcut.shortcut_path(chosen_directory)
    assert created.is_file()
    assert str(desktop_shortcut.find_entry_point()) in created.read_text()


def test_cancelling_the_location_dialog_writes_nothing(main_ui, monkeypatch, tmp_path):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "")
    main_ui.action_create_desktop_shortcut.trigger()
    assert list(tmp_path.iterdir()) == []


def test_declining_the_overwrite_question_leaves_the_file(
    main_ui, chosen_directory, monkeypatch
):
    from fibsem.tools import desktop_shortcut

    existing = desktop_shortcut.shortcut_path(chosen_directory)
    existing.write_text("pre-existing")

    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
    main_ui.action_create_desktop_shortcut.trigger()
    assert existing.read_text() == "pre-existing"

    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )
    main_ui.action_create_desktop_shortcut.trigger()
    assert str(desktop_shortcut.find_entry_point()) in existing.read_text()
