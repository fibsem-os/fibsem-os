import gc
import os

import pytest

# The Demo microscope and the simulated fluorescence microscope sleep to emulate
# real hardware timing, which otherwise makes the suite spend most of its
# wall-clock asleep. Skip those simulated delays for the whole test run; real
# drivers override the methods and are unaffected (see fibsem._timing.sim_sleep).
# Use FIBSEM_SIM_NO_DELAY=0 to run with the delays enabled.
os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test from its own tmp dir.

    Some code paths write next to the current working directory — e.g. alignment
    plotting creates an ``Alignment/`` directory under the reference image's path
    and falls back to the cwd when it is unset (as it is for the demo images used
    across the alignment and milling tests). Chdir into the test's tmp_path so
    those artifacts land there (auto-cleaned) instead of the repo root, and never
    collide between workers under ``pytest -n``.
    """
    monkeypatch.chdir(tmp_path)


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
