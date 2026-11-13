"""fibsemOS and the apps show one set of microscope tabs, built by one
`MicroscopeControls`: the union of what each window used to build."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy.QtWidgets")

from tests.fixtures.connecting import connect_and_wait
from tests.fixtures.main_window import close_window

CONNECTION = "Connection"


def _tabs(tab_widget, without=()):
    return [
        tab_widget.tabText(i)
        for i in range(tab_widget.count())
        if tab_widget.tabText(i) not in without
    ]


def test_fibsemos_and_autolamella_show_the_same_microscope_tabs(qapp):
    from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
    from fibsem.ui.FibsemUI import FibsemUI

    fibsem_ui = FibsemUI()
    connect_and_wait(fibsem_ui.system_widget)
    app = AutoLamellaWindow()
    connect_and_wait(app.system_widget)
    qapp.processEvents()
    try:
        ours = _tabs(fibsem_ui.controls.tabWidget, without=(CONNECTION,))
        theirs = _tabs(
            app.panel.tabWidget,
            without=(CONNECTION, app.panel.experiment_tab.title),
        )
        assert ours == theirs
        # The default configuration has a manipulator; AutoLamella had no tab for it.
        assert "Manipulator" in theirs
    finally:
        close_window(app)
        close_window(fibsem_ui)


def test_fibsemos_drops_the_microscope_tabs_on_disconnect(qapp):
    from fibsem.ui.FibsemUI import FibsemUI

    window = FibsemUI()
    connect_and_wait(window.system_widget)
    microscope = window.microscope
    window.disconnect_from_microscope()
    qapp.processEvents()
    try:
        assert _tabs(window.controls.tabWidget) == [CONNECTION]
        assert window.controls.image_widget is None
    finally:
        microscope.disconnect()
        close_window(window)
