"""Headless smoke for the quad-view *selected view* (PR1).

Verifies selection state, the panel border, the "only the selected canvas shows its
toolbar" rule (incl. the btn_mode exemption + FM's hidden contrast button), click-to-select
via the event filter, and controller forwarding.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_selected_view.py
"""
import sys

from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication

from fibsem.structures import BeamType
from fibsem.ui.widgets.canvas.quad_view import (
    _SELECT_ACCENT,
    MicroscopeViewController,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _press(canvas) -> None:
    """Dispatch a left mouse-press to *canvas* so the widget's event filter runs."""
    ev = QMouseEvent(
        QEvent.MouseButtonPress, QPointF(5, 5), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier
    )
    QApplication.sendEvent(canvas, ev)


def test_default_selection_is_sem():
    c = MicroscopeViewController()
    assert c.selected_view is BeamType.ELECTRON
    # only the SEM toolbar is shown
    assert c.sem_canvas.btn_reset_view.isHidden() is False
    assert c.fib_canvas.btn_reset_view.isHidden() is True
    assert c.fm_canvas.btn_reset_view.isHidden() is True
    # SEM panel is bordered, the others are not
    w = c.widget
    assert _SELECT_ACCENT in w._panels[BeamType.ELECTRON].styleSheet()
    assert _SELECT_ACCENT not in w._panels[BeamType.ION].styleSheet()
    assert _SELECT_ACCENT not in w._panels["fm"].styleSheet()


def test_set_selected_switches_toolbar_and_border():
    c = MicroscopeViewController()
    c.widget.set_selected("fm")
    assert c.selected_view == "fm"
    # toolbars follow selection
    assert c.fm_canvas.btn_reset_view.isHidden() is False
    assert c.sem_canvas.btn_reset_view.isHidden() is True
    # borders follow selection
    w = c.widget
    assert _SELECT_ACCENT in w._panels["fm"].styleSheet()
    assert _SELECT_ACCENT not in w._panels[BeamType.ELECTRON].styleSheet()


def test_click_selects_via_event_filter():
    c = MicroscopeViewController()
    _press(c.fib_canvas)
    assert c.selected_view is BeamType.ION
    _press(c.fm_canvas)
    assert c.selected_view == "fm"


def test_controller_forwards_view_selected():
    c = MicroscopeViewController()
    seen = []
    c.view_selected.connect(seen.append)
    c.widget.set_selected(BeamType.ION)
    assert seen == [BeamType.ION]
    # re-selecting the same view is a no-op (no echo)
    c.widget.set_selected(BeamType.ION)
    assert seen == [BeamType.ION]


def test_active_mode_toggle_survives_deselect():
    """A non-selected canvas mid overlay-edit keeps its contextual mode toggle."""
    c = MicroscopeViewController()  # SEM selected
    c.sem_canvas.enter_overlay_mode(object(), "Edit")
    assert c.sem_canvas.btn_mode.isHidden() is False
    c.widget.set_selected("fm")  # deselect SEM -> hide its generic toolbar
    assert c.sem_canvas.btn_reset_view.isHidden() is True  # generic buttons hidden
    assert c.sem_canvas.btn_mode.isHidden() is False  # ...but the mode toggle stays


def test_fm_contrast_stays_hidden_across_cycle():
    """FM hides btn_contrast; a hide/show toolbar cycle must not resurrect it."""
    c = MicroscopeViewController()
    assert c.fm_canvas.btn_contrast.isHidden() is True  # hidden by FMCanvasWidget
    c.widget.set_selected("fm")  # show the FM toolbar
    assert c.fm_canvas.btn_reset_view.isHidden() is False  # generic buttons back
    assert c.fm_canvas.btn_contrast.isHidden() is True  # contrast stays hidden


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
