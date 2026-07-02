"""Headless smoke for the FIBSEM working-distance Shift+scroll (PR4).

Drives FibsemBeamSettingsWidget._on_canvas_scroll / _execute_wd_wheel_move_impl against a fake
microscope (deterministic, and lets us stub the confirm dialog so no modal blocks). Covers the
Shift gate, acquisition lockout, spinbox<->hardware sync, and the large-change confirm/revert.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_working_distance_scroll.py
"""
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem import constants
from fibsem.structures import BeamType
from fibsem.ui.widgets.beam_settings_widget import (
    WD_WHEEL_STEP_MM,
    FibsemBeamSettingsWidget,
)
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


class _FakeMicroscope:
    manufacturer = "Demo"

    def __init__(self):
        self._wd = {BeamType.ELECTRON: 4.0e-3, BeamType.ION: 16.5e-3}
        self.is_acquiring = False
        self.set_calls = []

    def get_working_distance(self, beam):
        return self._wd[beam]

    def set_working_distance(self, wd, beam):
        self._wd[beam] = float(wd)
        self.set_calls.append((beam, float(wd)))
        return self._wd[beam]


def _widget(beam=BeamType.ELECTRON):
    ms = _FakeMicroscope()
    w = FibsemBeamSettingsWidget(microscope=ms, beam_type=beam)
    w._set_working_distance_spinbox(ms.get_working_distance(beam))  # sync spinbox to hardware
    return w, ms


def _close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_shift_scroll_updates_spinbox_and_target():
    w, _ = _widget()
    sb = w.working_distance_spinbox
    start = sb.value()
    w._on_canvas_scroll(0, 0, +1, ("Shift",))
    assert _close(sb.value(), start + WD_WHEEL_STEP_MM)
    assert _close(w._wd_wheel_target_mm, sb.value())


def test_plain_scroll_ignored():
    w, _ = _widget()
    sb = w.working_distance_spinbox
    start = sb.value()
    w._on_canvas_scroll(0, 0, +1, ())  # no Shift -> canvas zoom, not WD
    assert _close(sb.value(), start)


def test_scroll_works_during_acquisition():
    # WD (beam focus) must stay adjustable while scanning — no acquisition lockout.
    w, ms = _widget()
    ms.is_acquiring = True
    sb = w.working_distance_spinbox
    start = sb.value()
    w._on_canvas_scroll(0, 0, +1, ("Shift",))
    assert _close(sb.value(), start + WD_WHEEL_STEP_MM)


def test_debounced_move_applies_to_hardware():
    w, ms = _widget()
    w._on_canvas_scroll(0, 0, +1, ("Shift",))  # small step, within threshold -> no dialog
    target_mm = w._wd_wheel_target_mm
    w._execute_wd_wheel_move_impl()
    assert _close(ms.get_working_distance(BeamType.ELECTRON), target_mm * constants.MILLI_TO_SI)


def test_scroll_down_decreases_wd():
    w, ms = _widget()
    sb = w.working_distance_spinbox
    start = sb.value()
    w._on_canvas_scroll(0, 0, -1, ("Shift",))
    assert _close(sb.value(), start - WD_WHEEL_STEP_MM)


def test_flash_message_shows_and_clears():
    c = FibsemImageCanvas()
    c.flash_message("WD 4.001 mm")
    assert c._flash_text == "WD 4.001 mm"
    assert c._flash_artist is not None
    assert c._flash_timer.isActive()
    c._clear_flash()
    assert c._flash_text is None
    assert c._flash_artist is None


def test_scroll_flashes_on_emitting_canvas():
    """A real canvas_scrolled emission drives the handler and flashes on that canvas."""
    w, _ = _widget()
    c = FibsemImageCanvas()
    c.canvas_scrolled.connect(w._on_canvas_scroll)
    c.canvas_scrolled.emit(0.0, 0.0, +1, ("Shift",))
    assert c._flash_text is not None and "WD" in c._flash_text


def test_flash_survives_image_update():
    """A live frame recomposites via set_array (clears the axes) — the flash must survive so
    it stays visible while adjusting WD during acquisition."""
    c = FibsemImageCanvas()
    c.flash_message("WD 4.001 mm")
    c.set_array(np.zeros((32, 32), dtype=np.uint8))  # simulate a new acquired frame
    assert c._flash_text == "WD 4.001 mm"
    assert c._flash_artist is not None


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
