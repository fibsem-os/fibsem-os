import gc
import os

import pytest


@pytest.fixture(scope="module")
def qapp():
    """Shared offscreen QApplication for widget tests.

    Keep CPython's cyclic garbage collector from running while these tests build
    and paint real Qt widgets. Constructing a parameter form ends in widget.show()
    (e.g. AutoLamellaTaskParametersConfigWidget._update_from_config), and Qt's C++
    paint of the qtawesome toolbutton icon re-enters Python; an automatic gen-0
    collection landing in that window finalises objects mid-paint and segfaults the
    interpreter — an intermittent EXC_BAD_ACCESS inside QCommonStyle::drawControl
    that takes the whole `pytest` process down rather than failing one test. It only
    bites when a prior test has left cyclic garbage for that collection to reclaim,
    which is why a single test in isolation always passes.

    This is the single-threaded, re-entrant cousin of the Windows vispy/gloo crash
    (PR #168): the application never lets automatic GC run for this reason and
    collects on the Qt main thread instead (fibsem/ui/qt/gc.py). The tests build
    widgets without that machinery, so adopt the same contract here — disable
    automatic collection while the widgets are alive and drain it once at a safe
    point (no Qt paint on the stack) on teardown. Module scope keeps the disable
    confined to each UI module and restores it in between.

    test_main_thread_gc.py deliberately keeps its own qapp fixture (which shadows
    this one by name) so it can exercise stock GC behaviour.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - environment without Qt
        pytest.skip(f"PyQt5 not available: {exc}")
    app = QApplication.instance() or QApplication([])

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield app
    finally:
        gc.collect()
        if gc_was_enabled:
            gc.enable()
