"""The imaging hotkeys reach the Microscope tab's widgets.

F2, F6, F9 and F11 find the image and fluorescence widgets on the panel. Looked up
by name, a move of those widgets left every key a silent no-op.
"""

import os

from tests.fixtures.main_window import main_window_fixture

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.structures.settings import BeamType

main_ui = main_window_fixture(connect=True)


def _select(monkeypatch, window, view):
    monkeypatch.setattr(
        type(window.view_controller), "selected_view", property(lambda self: view)
    )


def test_the_em_hotkeys_find_the_image_widget(main_ui, monkeypatch):
    _select(monkeypatch, main_ui, BeamType.ELECTRON)
    image_widget = main_ui.panel.image_widget

    assert image_widget is not None
    assert main_ui.menu_bar._selected_em_image_widget() is image_widget


def test_f2_acquires_on_the_selected_beam(main_ui, monkeypatch):
    image_widget = main_ui.panel.image_widget
    acquired = []
    monkeypatch.setattr(image_widget, "acquire_fib_image", lambda: acquired.append(1))
    _select(monkeypatch, main_ui, BeamType.ION)

    main_ui.menu_bar._hotkey_acquire()

    assert acquired == [1]
