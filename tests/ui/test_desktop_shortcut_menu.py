"""Tools → Create Desktop Shortcut: the apps, then the fibsemOS Gateway."""

import os

from tests.fixtures.main_window import main_window_fixture

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.tools import desktop_shortcut
from fibsem.tools.apps import GATEWAY

window = main_window_fixture(connect=False, name="window")


def test_the_gateway_shortcut_comes_after_the_apps(window):
    actions = window.menu_bar.shortcut_actions
    assert list(actions)[-1] == "gateway"
    assert actions["gateway"].text() == "fibsemOS Gateway..."


def test_it_writes_the_gateway_shortcut_where_chosen(window, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    made = []
    monkeypatch.setattr(
        desktop_shortcut,
        "create_desktop_shortcut",
        lambda **kwargs: made.append(kwargs) or tmp_path / "x",
    )
    monkeypatch.setattr(window.menu_bar, "_show_toast", lambda *a, **k: None)
    window.menu_bar.shortcut_actions["gateway"].trigger()
    assert made == [{"directory": tmp_path, "app": GATEWAY}]
