"""When a coincidence run ends, the viewer must come back to the front.

Reported from a session: "when the coincidence milling stops the AutoLamella GUI comes
to the front of everything". The viewer is a deliberately parentless top-level window
(see ``open_coincidence_viewer_window``), so nothing holds it above the main window --
whatever the main window's own end-of-run refresh does to the stacking order wins, and
the post-milling FIB image ends up hidden behind it.

The end of ``_finalize_milling_ui`` is the last thing the milling flow runs, so that is
where the viewer restacks itself. Pinned here because the call is invisible in normal
use (nothing else covers the window in a test) and would be an easy line to drop.

Driven through the real widget rather than a stub: the raise has to happen after the
result is recorded and the finished signal is emitted, and only the real method has
that order in it.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def viewer_widget(qapp, tmp_path):
    """The whole viewer on a simulated FM microscope.

    The FM configuration is not incidental: ``_connect_signals`` returns immediately
    when ``microscope.fm is None``, so on a plain demo session the viewer constructs
    with none of its signals wired.
    """
    from fibsem import config as cfg
    from fibsem import utils
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget,
    )

    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    )
    assert microscope.fm is not None, "expected an FM on the simulated Arctis"
    return FluorescenceCoincidenceViewerWidget(
        microscope=microscope,
        experiment=Experiment(path=str(tmp_path), name="coincidence-layering-test"),
    )


def test_the_viewer_restacks_itself_when_a_run_ends(viewer_widget, monkeypatch):
    calls = []
    monkeypatch.setattr(
        type(viewer_widget), "raise_", lambda self: calls.append(self), raising=False
    )

    viewer_widget._finalize_milling_ui()

    assert calls == [viewer_widget.window()], (
        "the viewer did not raise its own window when the run ended -- the main "
        "window can sit in front of it and hide the result"
    )


def test_the_raise_comes_after_the_run_is_recorded_and_announced(
    viewer_widget, monkeypatch
):
    """Ordering, not just the call: the main window reacts to both the recording and
    the finished signal, so a raise that ran before them could be undone by what they
    set off."""
    order = []
    monkeypatch.setattr(
        type(viewer_widget),
        "_record_coincidence_result",
        lambda self: order.append("record"),
    )
    monkeypatch.setattr(
        type(viewer_widget), "raise_", lambda self: order.append("raise"), raising=False
    )
    viewer_widget.milling_finished_signal.connect(lambda: order.append("finished"))

    viewer_widget._finalize_milling_ui()

    assert order == ["record", "finished", "raise"], f"unexpected order: {order}"


def test_the_viewer_does_not_take_focus(viewer_widget, monkeypatch):
    """raise_() restacks inside the application; activateWindow() would take keyboard
    focus and pull the app in front of the microscope software. Only the first is
    wanted -- the complaint being fixed is a window stealing the front."""
    monkeypatch.setattr(type(viewer_widget), "raise_", lambda self: None, raising=False)
    stolen = []
    monkeypatch.setattr(
        type(viewer_widget),
        "activateWindow",
        lambda self: stolen.append(self),
        raising=False,
    )

    viewer_widget._finalize_milling_ui()

    assert not stolen, "the viewer activated itself: that steals focus, it must not"


def test_it_leaves_alone_a_window_the_operator_moved_to(viewer_widget, monkeypatch):
    """Otherwise the backstop is the same complaint in reverse: click over to the main
    window mid-run and this window jumps in front of it the moment milling stops.

    A window pulled forward as a side effect does not take focus, so the case this
    exists for still restacks -- only a deliberate switch holds the active window.
    """
    from PyQt5.QtWidgets import QWidget

    elsewhere = QWidget()
    elsewhere.setWindowTitle("some other window of ours")
    calls = []
    monkeypatch.setattr(
        type(viewer_widget), "raise_", lambda self: calls.append(self), raising=False
    )
    monkeypatch.setattr(
        QApplication, "activeWindow", staticmethod(lambda: elsewhere)
    )

    viewer_widget._finalize_milling_ui()

    assert not calls, "restacked over a window the operator had deliberately moved to"


def test_an_embedded_viewer_does_not_restack_its_host(viewer_widget, monkeypatch):
    """`window()` on an embedded widget is somebody else's window, and raising it is
    somebody else's call. Only a top-level viewer restacks itself."""
    from PyQt5.QtWidgets import QVBoxLayout, QWidget

    calls = []
    monkeypatch.setattr(
        type(viewer_widget), "raise_", lambda self: calls.append(self), raising=False
    )
    host = QWidget()
    QVBoxLayout(host).addWidget(viewer_widget)  # reparented: no longer a window
    assert not viewer_widget.isWindow()

    viewer_widget._finalize_milling_ui()

    assert not calls, "an embedded viewer restacked something it does not own"
