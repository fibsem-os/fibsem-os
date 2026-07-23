"""Headless smoke for the Microscope-tab hotkeys.

Exercises the AutoLamellaMainUI hotkey handlers in isolation (unbound methods called with a
lightweight fake ``self``), so we test the F2 / F5 / Esc / F6 / F9 / F11 routing + guards
without building the whole main window.

    F2  acquire selected view (SEM/FIB/FM)
    F6  toggle live acquisition (SEM/FIB only)
    F9  auto contrast          (SEM/FIB only)
    F11 auto focus             (SEM/FIB only)

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_hotkeys.py
"""
import sys
import types

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.ui.AutoLamellaMainUI import AutoLamellaSingleWindowUI
from fibsem.structures import BeamType
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

_app = QApplication.instance() or QApplication(sys.argv)


class _FakeImageWidget:
    def __init__(self):
        self.calls = []
        self.is_acquiring = False

    def acquire_sem_image(self):
        self.calls.append("sem")

    def acquire_fib_image(self):
        self.calls.append("fib")

    def toggle_live_acquisition(self):
        self.calls.append("live")

    def run_autocontrast(self):
        self.calls.append("autocontrast")

    def run_autofocus(self):
        self.calls.append("autofocus")


class _FakeFM:
    def __init__(self):
        self.calls = []
        self.is_acquiring = False

    def acquire_image(self):
        self.calls.append("fm")


def _fake_self(selected, image_widget=None, fm_widget=None):
    """A stand-in with just the attributes the hotkey handlers touch."""
    ctrl = MicroscopeViewController()
    if selected is not None:
        ctrl.widget.set_selected(selected)
    ui = types.SimpleNamespace(
        image_widget=image_widget if image_widget is not None else _FakeImageWidget(),
        fm_control_widget=fm_widget if fm_widget is not None else _FakeFM(),
    )
    s = types.SimpleNamespace(view_controller=ctrl, autolamella_ui=ui)
    # the EM-only handlers route through this helper; bind the real one to the fake self
    s._selected_em_image_widget = lambda: AutoLamellaSingleWindowUI._selected_em_image_widget(s)
    return s


# ── F2: acquire selected ─────────────────────────────────────────────────────

def test_f2_acquires_sem_when_sem_selected():
    s = _fake_self(BeamType.ELECTRON)
    AutoLamellaSingleWindowUI._hotkey_acquire(s)
    assert s.autolamella_ui.image_widget.calls == ["sem"]


def test_f2_acquires_fib_when_fib_selected():
    s = _fake_self(BeamType.ION)
    AutoLamellaSingleWindowUI._hotkey_acquire(s)
    assert s.autolamella_ui.image_widget.calls == ["fib"]


def test_f2_acquires_fm_when_fm_selected():
    s = _fake_self("fm")
    AutoLamellaSingleWindowUI._hotkey_acquire(s)
    assert s.autolamella_ui.fm_control_widget.calls == ["fm"]


def test_f2_noop_when_no_image_widget():
    ctrl = MicroscopeViewController()
    ctrl.widget.set_selected(BeamType.ELECTRON)
    ui = types.SimpleNamespace(image_widget=None, fm_control_widget=None)
    s = types.SimpleNamespace(view_controller=ctrl, autolamella_ui=ui)
    AutoLamellaSingleWindowUI._hotkey_acquire(s)  # must not raise


def test_f2_noop_when_already_acquiring():
    s = _fake_self(BeamType.ELECTRON)
    s.autolamella_ui.image_widget.is_acquiring = True
    AutoLamellaSingleWindowUI._hotkey_acquire(s)
    assert s.autolamella_ui.image_widget.calls == []


# ── F6 / F9 / F11: EM-only (live / autocontrast / autofocus) ─────────────────

def test_f6_live_toggles_on_selected_em_beam():
    s = _fake_self(BeamType.ELECTRON)
    AutoLamellaSingleWindowUI._hotkey_toggle_live(s)
    assert s.autolamella_ui.image_widget.calls == ["live"]


def test_f9_autocontrast_on_selected_em_beam():
    s = _fake_self(BeamType.ION)
    AutoLamellaSingleWindowUI._hotkey_autocontrast(s)
    assert s.autolamella_ui.image_widget.calls == ["autocontrast"]


def test_f11_autofocus_on_selected_em_beam():
    s = _fake_self(BeamType.ELECTRON)
    AutoLamellaSingleWindowUI._hotkey_autofocus(s)
    assert s.autolamella_ui.image_widget.calls == ["autofocus"]


def test_em_hotkeys_noop_when_fm_selected():
    # FM is selected -> the EM-only hotkeys must not touch the image widget.
    s = _fake_self("fm")
    AutoLamellaSingleWindowUI._hotkey_toggle_live(s)
    AutoLamellaSingleWindowUI._hotkey_autocontrast(s)
    AutoLamellaSingleWindowUI._hotkey_autofocus(s)
    assert s.autolamella_ui.image_widget.calls == []


# ── View menu (fullscreen) ───────────────────────────────────────────────────

class _FakeAction:
    def __init__(self):
        self.checked = None
        self.enabled = None

    def setChecked(self, v):
        self.checked = v

    def setEnabled(self, v):
        self.enabled = v


def test_sync_view_menu_reflects_fullscreen():
    ctrl = MicroscopeViewController()
    s = types.SimpleNamespace(
        view_controller=ctrl,
        action_toggle_fullscreen=_FakeAction(),
        action_exit_fullscreen=_FakeAction(),
    )
    AutoLamellaSingleWindowUI._sync_view_menu(s)  # grid
    assert s.action_toggle_fullscreen.checked is False
    assert s.action_exit_fullscreen.enabled is False
    ctrl.toggle_fullscreen()  # SEM selected -> full screen
    AutoLamellaSingleWindowUI._sync_view_menu(s)
    assert s.action_toggle_fullscreen.checked is True
    assert s.action_exit_fullscreen.enabled is True


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
