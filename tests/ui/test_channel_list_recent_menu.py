"""Headless tests for the recent-channels quick-select menu in ChannelListWidget.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
The real ``_on_add_channel`` opens a modal ``QMenu.exec_()``; tests stub that
to capture the menu, and exercise the row/selection/removal helpers directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QMenu, QWidgetAction

from fibsem import config as cfg
from fibsem.fm import config as fm_config
from fibsem.fm.structures import ChannelSettings
from fibsem.ui.fm.widgets import channel_list_widget as clw


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def recents_env(tmp_path, monkeypatch):
    """Point the recent-channels store at an isolated temp file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        cfg, "FM_RECENT_CHANNELS_PATH", str(config_dir / "fm-recent-channels.yaml")
    )
    return tmp_path


class _FakeFilterSet:
    available_excitation_wavelengths = [405.0, 488.0, 550.0]
    available_emission_wavelengths = [450.0, 520.0]


class _FakeFM:
    filter_set = _FakeFilterSet()
    is_acquiring = False


def _widget(qapp):
    channel = ChannelSettings(
        name="Channel-01", excitation_wavelength=405.0, emission_wavelength=450.0
    )
    return clw.ChannelListWidget(fm=_FakeFM(), channel_settings=[channel])


def _recent_rows(menu: QMenu):
    return [
        a
        for a in menu.actions()
        if isinstance(a, QWidgetAction)
        and isinstance(a.defaultWidget(), clw._RecentChannelRow)
    ]


# --- availability --------------------------------------------------------


def test_channel_available_matches_filter_set(qapp):
    w = _widget(qapp)
    assert w._channel_available(
        ChannelSettings(name="GFP", excitation_wavelength=488.0, emission_wavelength=520.0)
    )
    # emission None (reflection) is always allowed
    assert w._channel_available(
        ChannelSettings(name="Refl", excitation_wavelength=405.0, emission_wavelength=None)
    )
    # excitation not on the filter set
    assert not w._channel_available(
        ChannelSettings(name="Bad", excitation_wavelength=999.0, emission_wavelength=520.0)
    )
    # emission not on the filter set
    assert not w._channel_available(
        ChannelSettings(name="Bad", excitation_wavelength=405.0, emission_wavelength=999.0)
    )


# --- menu construction ---------------------------------------------------


def test_no_recents_adds_directly_without_menu(qapp, monkeypatch):
    w = _widget(qapp)
    monkeypatch.setattr(clw, "load_recent_channels", lambda: [])
    monkeypatch.setattr(
        QMenu, "exec_", lambda *a, **k: pytest.fail("menu must not open with no recents")
    )
    before = w._list.count()
    w._on_add_channel()
    assert w._list.count() == before + 1


def test_menu_lists_new_channel_then_recents(qapp, monkeypatch):
    w = _widget(qapp)
    recents = [
        ChannelSettings(name="GFP", excitation_wavelength=488.0, emission_wavelength=520.0),
        ChannelSettings(name="Bad", excitation_wavelength=999.0, emission_wavelength=520.0),
    ]
    monkeypatch.setattr(clw, "load_recent_channels", lambda: recents)

    captured = {}

    def fake_exec(self, *a, **k):
        captured["menu"] = self
        return None

    monkeypatch.setattr(QMenu, "exec_", fake_exec)
    w._on_add_channel()

    menu = captured["menu"]
    widget_actions = [a for a in menu.actions() if isinstance(a, QWidgetAction)]
    rows = [a.defaultWidget() for a in widget_actions]
    # first row is New channel, then one row per recent
    assert isinstance(rows[0], clw._NewChannelRow)
    recent_rows = [r for r in rows if isinstance(r, clw._RecentChannelRow)]
    assert [r.channel.name for r in recent_rows] == ["GFP", "Bad"]
    # the unavailable recent is greyed out / not selectable
    assert recent_rows[0]._available is True
    assert recent_rows[1]._available is False


# --- selection -----------------------------------------------------------


def test_new_channel_selected_adds_default(qapp):
    w = _widget(qapp)
    before = w._list.count()
    w._on_new_channel_selected(QMenu())
    assert w._list.count() == before + 1


def test_recent_selected_adds_copy_and_uniquifies_name(qapp):
    w = _widget(qapp)  # already contains "Channel-01"
    recent = ChannelSettings(
        name="Channel-01", excitation_wavelength=405.0, emission_wavelength=450.0
    )
    before = w._list.count()
    w._on_recent_selected(QMenu(), recent)

    assert w._list.count() == before + 1
    names = [w._row(i).channel.name for i in range(w._list.count())]
    assert "Channel-01 (2)" in names
    # a copy was added, not the passed-in object
    added = w._row(w._list.count() - 1).channel
    assert added is not recent


# --- removal -------------------------------------------------------------


def test_recent_removed_updates_store(qapp, recents_env):
    w = _widget(qapp)
    fm_config.record_recent_channels(
        [
            ChannelSettings(name="GFP", excitation_wavelength=488.0, emission_wavelength=520.0),
            ChannelSettings(name="DAPI", excitation_wavelength=405.0, emission_wavelength=450.0),
        ]
    )
    menu = QMenu()
    for recent in fm_config.load_recent_channels():
        w._add_recent_menu_row(menu, recent)
    assert len(_recent_rows(menu)) == 2

    # click the x on the first recent row
    _recent_rows(menu)[0].defaultWidget().btn_remove.click()

    remaining = [c.name for c in fm_config.load_recent_channels()]
    assert len(remaining) == 1
    assert len(_recent_rows(menu)) == 1


def test_removing_last_recent_empties_store(qapp, recents_env):
    w = _widget(qapp)
    fm_config.record_recent_channels(
        ChannelSettings(name="GFP", excitation_wavelength=488.0, emission_wavelength=520.0)
    )
    menu = QMenu()
    for recent in fm_config.load_recent_channels():
        w._add_recent_menu_row(menu, recent)

    _recent_rows(menu)[0].defaultWidget().btn_remove.click()

    assert fm_config.load_recent_channels() == []
    assert _recent_rows(menu) == []
