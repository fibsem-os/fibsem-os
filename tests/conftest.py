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
def _isolate_sample_holder_config(tmp_path, monkeypatch):
    """Point every test at a private copy of the shipped sample holder config.

    ``fibsem/config/sample-holder.yaml`` is the operator's calibration, written by
    the calibration wizard and ignored by git. Reading it in tests would make the
    suite depend on whichever holder was last calibrated on this machine, and the
    holder widget's auto-save would overwrite that calibration with test data --
    which it did, before this fixture existed. Both readers resolve the path at
    call time, so patching the two module attributes is enough.
    """
    import shutil

    import fibsem.config as cfg
    import fibsem.microscopes._stage as stage_module

    path = tmp_path / "sample-holder.yaml"
    shutil.copy(cfg.DEFAULT_SAMPLE_HOLDER_CONFIGURATION_PATH, path)
    monkeypatch.setattr(cfg, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))
    monkeypatch.setattr(stage_module, "SAMPLE_HOLDER_CONFIGURATION_PATH", str(path))


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


@pytest.fixture
def destroy_widgets_after_test(qapp):
    """Destroy every top-level widget the test creates, once it is over.

    ``close()`` is not destruction -- it hides. A test that builds a window, closes it
    in teardown and drops its reference leaves the window alive for the rest of the
    session, because the Qt object outlives the Python wrapper and nothing has asked
    for it to go. ``tests/ui/`` reaches 2541 live top-level widgets in a full run this
    way, and the files that leak most are the ones that *do* call ``close()``.

    That matters less for memory than for what the next test sees.
    ``QApplication.activeWindow()`` is process-global, so an abandoned window is still
    a candidate answer to it -- which is how three tests in
    test_coincidence_viewer_layering.py came to assert about the window stack rather
    than about the code (#583).

    Two things are needed and neither is obvious. ``deleteLater()`` posts a
    ``DeferredDelete`` event rather than deleting anything, and ``processEvents()``
    does **not** deliver that event -- only ``sendPostedEvents`` with the type named
    explicitly does. Using ``processEvents`` here measures as a complete no-op: the
    widget count does not move, which reads as "the fix does not work" rather than as
    "the deletion never ran".

    Only widgets that appear during the test are touched, so a module-scoped fixture's
    widget built by an earlier test is left alone.
    """
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication

    before = set(QApplication.topLevelWidgets())
    yield
    for widget in QApplication.topLevelWidgets():
        if widget not in before:
            widget.close()
            widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
