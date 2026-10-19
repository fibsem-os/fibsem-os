"""The development menu's coincidence entries."""

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMenu  # noqa: E402

from fibsem.ui.widgets.coincidence_menu import CoincidenceMenu  # noqa: E402


def test_the_entries_sit_in_their_own_submenu_and_need_a_microscope(qapp):
    menu = QMenu()
    toasts = []
    tools = CoincidenceMenu(
        menu,
        microscope=lambda: None,
        experiment_path=lambda: None,
        show_toast=lambda message, kind="info", **_: toasts.append((message, kind)),
    )
    (submenu_action,) = menu.actions()
    assert submenu_action.text() == "Coincidence Alignment"
    assert [a.text() for a in submenu_action.menu().actions()] == [
        a.text() for a in tools.actions
    ]
    assert len(tools.actions) == 4

    for action in tools.actions:
        action.trigger()
    assert toasts == [("Not connected to a microscope.", "warning")] * 4
    assert all(action.isEnabled() for action in tools.actions)
