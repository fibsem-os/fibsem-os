"""Headless smoke for the quad-view *full screen* (PR2).

Verifies that full-screening a view hides the sibling cell + the other vertical splitter,
that toggle targets the selected view, that None restores the grid, and that the controller
forwards it (no-op on the lamella editor view).

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_fullscreen.py
"""
import sys

from PyQt5.QtWidgets import QApplication

from fibsem.structures import BeamType
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _shown(panel) -> bool:
    """Not explicitly hidden (reliable before the window is shown, unlike isVisible)."""
    return not panel.isHidden()


def test_default_is_grid():
    w = MicroscopeViewController().widget
    assert w.fullscreen is None
    assert all(_shown(p) for p in w._all_panels)
    assert all(_shown(s) for s in w._splitters.values())


def test_fullscreen_sem_hides_sibling_and_other_splitter():
    w = MicroscopeViewController().widget
    w.set_fullscreen(BeamType.ELECTRON)
    assert w.fullscreen is BeamType.ELECTRON
    panels = w._panels
    assert _shown(panels[BeamType.ELECTRON])       # SEM kept
    assert not _shown(panels["fm"])                # FM sibling hidden
    assert not _shown(panels[BeamType.ION])        # FIB hidden
    assert _shown(w._splitters["left"])            # left splitter (holds SEM) kept
    assert not _shown(w._splitters["right"])       # right splitter hidden


def test_fullscreen_fib_uses_right_splitter():
    w = MicroscopeViewController().widget
    w.set_fullscreen(BeamType.ION)
    assert _shown(w._panels[BeamType.ION])
    assert _shown(w._splitters["right"])
    assert not _shown(w._splitters["left"])


def test_fullscreen_selects_target():
    """Full-screening a specific cell also selects it, so its toolbar is the one shown."""
    c = MicroscopeViewController()  # SEM selected by default
    c.widget.set_fullscreen(BeamType.ION)
    assert c.selected_view is BeamType.ION
    assert c.fib_canvas.btn_reset_view.isHidden() is False  # FIB toolbar visible
    assert c.sem_canvas.btn_reset_view.isHidden() is True


def test_switch_target_directly():
    w = MicroscopeViewController().widget
    w.set_fullscreen(BeamType.ELECTRON)
    w.set_fullscreen(BeamType.ION)  # switch without exiting first
    assert w.fullscreen is BeamType.ION
    assert _shown(w._panels[BeamType.ION])
    assert not _shown(w._panels[BeamType.ELECTRON])
    assert _shown(w._splitters["right"])
    assert not _shown(w._splitters["left"])


def test_toggle_targets_selection_and_restores():
    w = MicroscopeViewController().widget  # SEM selected by default
    w.toggle_fullscreen()
    assert w.fullscreen is BeamType.ELECTRON
    assert w._saved_sizes  # grid sizes captured on entry
    w.toggle_fullscreen()  # exit
    assert w.fullscreen is None
    assert all(_shown(p) for p in w._all_panels)
    assert all(_shown(s) for s in w._splitters.values())


def test_set_fullscreen_none_restores_grid():
    w = MicroscopeViewController().widget
    w.set_fullscreen("fm")
    assert not _shown(w._panels[BeamType.ELECTRON])
    w.set_fullscreen(None)
    assert w.fullscreen is None
    assert all(_shown(p) for p in w._all_panels)


def test_unknown_key_is_noop():
    w = MicroscopeViewController().widget
    w.set_fullscreen("nope")
    assert w.fullscreen is None
    assert all(_shown(p) for p in w._all_panels)


def test_controller_forwards_fullscreen():
    c = MicroscopeViewController()
    c.toggle_fullscreen()
    assert c.fullscreen is BeamType.ELECTRON
    c.set_fullscreen(None)
    assert c.fullscreen is None


def test_editor_view_fullscreen_is_noop():
    c = MicroscopeViewController(view=LamellaEditorView())
    c.toggle_fullscreen()      # must not raise
    c.set_fullscreen(BeamType.ION)  # must not raise
    assert c.fullscreen is None


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
