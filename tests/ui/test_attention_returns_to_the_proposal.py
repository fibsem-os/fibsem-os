"""The attention button is the way back to a proposal you navigated away from.

A proposal decided on a tab of its own -- a detection, a mill -- disappears
from view the moment the operator looks at something else. The button says
"Attention Required" and the run is stopped on that proposal, so pressing it
has to land on the proposal's tab, not merely on the Microscope tab.

The real ``FibsemEmbeddedDetectionWidget`` needs the ``ml`` extra, which no
test environment installs, so the stand-in below is a real ``QWidget`` on the
real tab bar exposing the methods the window drives.
"""

import os

from tests.fixtures.main_window import main_window_fixture

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QWidget

from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
from fibsem.detection.detection import DetectedFeatures, LamellaCentre
from fibsem.workflows.proposals import DETECTION, Proposal
from tests.fixtures.main_window import close_window

ITEM, TASK = "item-1", "Setup"


class _DetWidget(QWidget):
    """The methods the window drives, on a real widget in the tab bar."""

    def __init__(self):
        super().__init__()
        self.det = None

    def set_detected_features(self, det):
        self.det = det

    def _get_detected_features(self):
        return self.det

    def clear_layers(self):
        self.det = None


@pytest.fixture
def ui(qapp):
    widget = AutoLamellaWindow()
    widget.system_widget.connect_to_microscope()
    det_widget = _DetWidget()
    widget.panel.det_widget = det_widget
    index = widget.panel.tabWidget.addTab(det_widget, "Detection")
    widget.panel.tabWidget.setTabVisible(index, False)
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    close_window(widget)


def _detection() -> DetectedFeatures:
    return DetectedFeatures(
        features=[LamellaCentre()],
        image=np.zeros((8, 8), dtype=np.uint8),
        mask=np.zeros((8, 8), dtype=np.uint8),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        pixelsize=1e-9,
    )


def _put_up(ui):
    """Put a detection proposal up, as the window does when a task asks it."""
    ui.panel.det_widget.set_detected_features(_detection())
    ui.proposals.show_widget_proposal(ITEM, TASK, Proposal(kind=DETECTION))


def _decided(ui):
    ui.proposals.clear_widget_proposal(ITEM, TASK)


def test_the_window_says_which_tab_its_proposal_is_on(ui, qapp):
    _put_up(ui)

    assert ui.proposals.proposal_host() is ui.panel.det_widget

    _decided(ui)


def test_there_is_no_host_when_nothing_is_being_asked(ui, qapp):
    assert ui.proposals.proposal_host() is None


def test_the_host_is_forgotten_once_the_proposal_is_decided(ui, qapp):
    """Otherwise the button would keep returning to a tab whose proposal is
    down, long after the run moved on."""
    _put_up(ui)
    _decided(ui)

    assert ui.proposals.proposal_host() is None


def test_navigating_away_and_coming_back_lands_on_the_proposal(ui, qapp):
    """The whole point: the proposal's tab is fronted when it is asked, the
    operator leaves, and ``front_proposal`` brings it back."""
    _put_up(ui)
    assert ui.panel.tabWidget.currentWidget() is ui.panel.det_widget, (
        "asked on its own tab"
    )

    ui.panel.tabWidget.setCurrentIndex(0)
    qapp.processEvents()
    assert ui.panel.tabWidget.currentWidget() is not ui.panel.det_widget, (
        "looked elsewhere"
    )

    ui.proposals.front_proposal()

    assert ui.panel.tabWidget.currentWidget() is ui.panel.det_widget
    _decided(ui)


def test_coming_back_un_hides_a_tab_that_hides_itself(ui, qapp):
    """Detection's tab is hidden until it is asked for, and fronting a hidden
    tab is a silent no-op -- so the way back has to un-hide it first."""
    _put_up(ui)
    index = ui.panel.tabWidget.indexOf(ui.panel.det_widget)
    ui.panel.tabWidget.setCurrentIndex(0)
    ui.panel.tabWidget.setTabVisible(index, False)
    qapp.processEvents()

    ui.proposals.front_proposal()

    assert ui.panel.tabWidget.isTabVisible(index)
    assert ui.panel.tabWidget.currentWidget() is ui.panel.det_widget
    _decided(ui)


def test_it_does_nothing_when_no_proposal_is_up(ui, qapp):
    """A hold with no proposal on a tab of its own -- an agent working, a
    proposal on the shared bar -- leaves the tab where the caller put it."""
    ui.panel.tabWidget.setCurrentIndex(0)
    before = ui.panel.tabWidget.currentWidget()

    ui.proposals.front_proposal()

    assert ui.panel.tabWidget.currentWidget() is before


# ---------------------------------------------------------------------------
# The window's own tabs
# ---------------------------------------------------------------------------


main_ui = main_window_fixture(connect=True)


def test_a_proposal_on_one_of_the_windows_own_tabs_is_fronted_there(main_ui, qapp):
    """The destination comes from the proposal, not from the hold's kind:
    ``HoldKind.proposal`` covers one decided on the Microscope tab and one
    decided on a tab of its own alike. Stood up here with the Grids tab
    standing in for a proposal's host."""
    host = main_ui.grids_tab
    main_ui.tab_widget.setTabEnabled(main_ui.tab_widget.indexOf(host), True)
    main_ui.proposals.proposal_host = lambda: host
    main_ui.tab_widget.setCurrentIndex(0)

    main_ui.attention.attention_clicked()

    assert main_ui.tab_widget.currentWidget() is host


def test_a_proposal_elsewhere_still_lands_on_the_microscope_tab(main_ui, qapp):
    """A proposal on the shared prompt bar has no tab of its own and must not
    be dragged into whatever the last host was."""
    main_ui.proposals.proposal_host = lambda: None
    main_ui.tab_widget.setCurrentWidget(main_ui.grids_tab)

    main_ui.attention.attention_clicked()

    assert main_ui.tab_widget.currentIndex() == 0


def test_while_the_agent_decides_the_proposal_stays_off_the_operators_tab(ui, qapp):
    """At Drive the agent decides: the proposal is still shown on its widget,
    but the operator's tab is left where they put it."""
    from fibsem.acting import AGENT

    ui.session.deciders = lambda: (AGENT,)
    ui.panel.tabWidget.setCurrentIndex(0)
    before = ui.panel.tabWidget.currentWidget()

    _put_up(ui)

    assert ui.panel.tabWidget.currentWidget() is before
    assert ui.proposals.proposal_host() is ui.panel.det_widget
    assert ui.panel.prompt_bar.message.text().startswith("Agent deciding: ")
    _decided(ui)
