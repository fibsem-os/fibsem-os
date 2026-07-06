"""Migration smoke: the quad-view widgets' ``@thread_worker`` methods now build the napari-free
``FunctionWorker`` and still run off the GUI thread, delivering ``finished`` / ``returned`` /
``errored`` back on it.

After swapping ``from napari.qt.threading import thread_worker`` for
``from fibsem.ui.qt.threading import thread_worker`` in ``FibsemImageSettingsWidget``,
``FibsemMovementWidget`` and ``FibsemSpotBurnWidget``, each decorated method must:

* return a :class:`fibsem.ui.qt.threading.FunctionWorker` (the swap took effect), and
* run its *real* body off-thread against a Demo microscope and emit ``finished``.

The widgets themselves need a live napari GL canvas to construct, which aborts under the
offscreen platform, so we drive each real decorated body against a minimal QObject stub as
``self`` — the decorator binds ``self`` through exactly as it does on the real widget, so this
exercises the migrated code path (decorator -> FunctionWorker -> off-thread body -> GUI-thread
delivery) without the canvas.

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_qt_worker_migration.py
"""
from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEventLoop, QObject, QTimer, pyqtSignal

# QApplication before the napari/vispy-heavy fibsem imports below.
_QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from fibsem import acquire, utils
from fibsem.structures import BeamType
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSpotBurnWidget import FibsemSpotBurnWidget
from fibsem.ui.qt.threading import FunctionWorker

_MAIN_THREAD = threading.current_thread()


def _spin(signal, timeout_ms: int = 20000) -> None:
    loop = QEventLoop()
    signal.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec_()
    _QAPP.processEvents()


def _drive(worker: FunctionWorker):
    """Start a worker, spin until finished, return (error_or_None, ran_off_main_thread)."""
    assert isinstance(worker, FunctionWorker), "decorated method did not build a FunctionWorker"
    err, fin, off_main = [], [], []
    worker.errored.connect(lambda e: err.append(e))
    # DirectConnection: this slot runs synchronously in the emitting thread, so it captures the
    # thread the body runs on. ``started`` is the first line of the worker body's execution.
    worker.started.connect(
        lambda: off_main.append(threading.current_thread() is not _MAIN_THREAD),
        Qt.DirectConnection,
    )
    worker.finished.connect(lambda: fin.append(True))
    worker.start()
    _spin(worker.finished)
    assert fin == [True], "worker never emitted finished"
    return (err[0] if err else None), (off_main[0] if off_main else None)


class _ImageStub(QObject):
    """Just what ``FibsemImageSettingsWidget``'s workers touch on ``self``."""

    acquisition_progress_signal = pyqtSignal(dict)

    def __init__(self, microscope, image_settings):
        super().__init__()
        self.microscope = microscope
        self.image_settings = image_settings
        self.eb_image = None
        self.ib_image = None


class _MoveStub(QObject):
    """Just what ``FibsemMovementWidget.move_to_orientation_worker`` touches on ``self``."""

    movement_progress_signal = pyqtSignal(dict)

    def __init__(self, microscope):
        super().__init__()
        self.microscope = microscope

    def update_ui_after_movement(self):
        pass


class _SpotStub(QObject):
    """Just what ``FibsemSpotBurnWidget._run_spot_burn`` touches on ``self``."""

    _spot_burn_finished_signal = pyqtSignal(object)
    _spot_burn_errored_signal = pyqtSignal(object)

    def __init__(self, microscope):
        super().__init__()
        self.microscope = microscope
        self.stop_event = threading.Event()


def test_image_settings_workers_migrated():
    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    stub = _ImageStub(microscope, settings.image)

    # autocontrast: a bare microscope call, off-thread, no error.
    err, off_main = _drive(FibsemImageSettingsWidget._autocontrast_worker(stub, BeamType.ELECTRON))
    assert err is None
    assert off_main is True  # body genuinely ran on the worker thread

    # acquisition (both): populates eb/ib images off-thread.
    err, _ = _drive(FibsemImageSettingsWidget.acquisition_worker(stub, stub.image_settings, both=True))
    assert err is None
    assert stub.eb_image is not None and stub.ib_image is not None


def test_movement_worker_migrated():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    stub = _MoveStub(microscope)

    err, off_main = _drive(FibsemMovementWidget.move_to_orientation_worker(stub, "SEM"))
    assert err is None
    assert off_main is True


def test_spot_burn_worker_migrated():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    stub = _SpotStub(microscope)

    # The current widget builds the worker directly as ``FunctionWorker(self._run_spot_burn,
    # settings)`` (see ``run_spot_burn_worker``); the burn itself is ``settings.run(...)``, so a
    # fake no-op settings keeps this about the worker plumbing, not a real burn.
    class _FakeSettings:
        def run(self, **kwargs):
            return None

    finished = []
    stub._spot_burn_finished_signal.connect(lambda r: finished.append(r))
    worker = FunctionWorker(FibsemSpotBurnWidget._run_spot_burn, stub, _FakeSettings())
    err, off_main = _drive(worker)
    assert err is None
    assert off_main is True  # body genuinely ran on the worker thread
    assert finished == [None]  # the widget's own finished signal fired on success


if __name__ == "__main__":
    test_image_settings_workers_migrated()
    test_movement_worker_migrated()
    test_spot_burn_worker_migrated()
    print("PASS")
