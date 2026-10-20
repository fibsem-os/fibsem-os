"""The Sample control tab comes and goes with the connection, like the others."""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate


@pytest.fixture
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import window as module

    window = module.AutoLamellaWindow()
    yield window
    if window.microscope is not None:
        window.microscope.disconnect()
    # closeEvent ends in app.quit(); on the shared test QApplication that latches
    # an interrupt. Run the real close cleanup with quit stubbed out.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def _tab_labels(ui):
    tabs = ui.panel.tabWidget
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_the_sample_tab_lives_for_the_connection(main_ui):
    ui = main_ui
    assert "Sample" not in _tab_labels(ui)
    assert ui.panel.sample_widget is None

    ui.system_widget.connect_to_microscope()
    labels = _tab_labels(ui)
    assert labels[labels.index("Milling") + 1] == "Sample"
    assert (
        ui.panel.sample_widget.holder_widget.current_holder
        is ui.microscope.sample_handling.holder
    )
    # the holder panel left the Movement sub-tab for good
    assert not hasattr(ui.panel.movement_widget, "sample_holder_widget")

    ui.microscope.disconnect()
    ui.microscope = None
    ui._show_microscope()
    assert "Sample" not in _tab_labels(ui)
    assert ui.panel.sample_widget is None
