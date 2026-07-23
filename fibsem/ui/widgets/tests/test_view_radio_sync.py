"""Headless smoke for the two-way quad-view-selection <-> beam-radio sync.

Wires a REAL MicroscopeViewController + REAL QRadioButtons (in a QButtonGroup) through the
actual FibsemImageSettingsWidget slots (called unbound against a lightweight fake self), so
the loop-safety relies on the real set_selected re-select guard + real setChecked semantics.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_view_radio_sync.py
"""
import sys
import types

from PyQt5.QtWidgets import QApplication, QButtonGroup, QRadioButton

from fibsem.structures import BeamType
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

_app = QApplication.instance() or QApplication(sys.argv)


def _rig():
    """Controller + two radios wired exactly like setup_connections, via a fake self."""
    ctrl = MicroscopeViewController()  # defaults to SEM (BeamType.ELECTRON) selected
    sem, fib = QRadioButton(), QRadioButton()
    grp = QButtonGroup()
    grp.addButton(sem)
    grp.addButton(fib)
    sem.setChecked(True)  # matches the controller default (before wiring, no echo)

    fake = types.SimpleNamespace(
        dual_beam_widget=types.SimpleNamespace(sem_radio=sem, fib_radio=fib),
    )
    fake._view_controller = lambda: ctrl

    ctrl.view_selected.connect(lambda k: FibsemImageSettingsWidget._on_view_selected(fake, k))
    sem.toggled.connect(lambda c: FibsemImageSettingsWidget._on_beam_radio_toggled(fake, c))
    # keep a ref to the group so it isn't GC'd
    fake._grp = grp
    return ctrl, sem, fib


def test_view_selection_checks_radio():
    ctrl, sem, fib = _rig()
    ctrl.widget.set_selected(BeamType.ION)
    assert fib.isChecked() and not sem.isChecked()
    assert ctrl.selected_view is BeamType.ION
    ctrl.widget.set_selected(BeamType.ELECTRON)
    assert sem.isChecked() and not fib.isChecked()
    assert ctrl.selected_view is BeamType.ELECTRON


def test_radio_check_selects_view():
    ctrl, sem, fib = _rig()
    fib.setChecked(True)
    assert ctrl.selected_view is BeamType.ION
    sem.setChecked(True)
    assert ctrl.selected_view is BeamType.ELECTRON


def test_fm_selection_leaves_radios_untouched():
    ctrl, sem, fib = _rig()  # SEM checked
    ctrl.widget.set_selected("fm")
    assert ctrl.selected_view == "fm"
    assert sem.isChecked() and not fib.isChecked()  # no fm radio -> unchanged


def test_bidirectional_terminates_no_loop():
    # If the sync recursed, these would raise RecursionError.
    ctrl, sem, fib = _rig()
    for _ in range(5):
        ctrl.widget.set_selected(BeamType.ION)
        ctrl.widget.set_selected(BeamType.ELECTRON)
        fib.setChecked(True)
        sem.setChecked(True)
    assert ctrl.selected_view is BeamType.ELECTRON


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
