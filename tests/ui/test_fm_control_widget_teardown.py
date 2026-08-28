"""What must stop reaching FMControlWidget once the host has torn it down (FIB-550).

The three signals it takes from the microscope are psygnal, not Qt, so nothing
disconnects them on its behalf: psygnal's weak reference is to the *Python* object, and
``deleteLater`` destroys the C++ one underneath while that stays alive. The host
(``AutoLamellaUI``) tears down without stopping the acquisition first, so a disconnect
mid-run leaves a worker emitting into slots whose widgets are gone.

The failure this pins is not a wrong value -- it is a `RuntimeError: wrapped C/C++
object ... has been deleted` raised inside somebody else's worker thread, which reads
as a broken acquisition rather than as a teardown bug.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import sip
from PyQt5.QtWidgets import QApplication

from fibsem.fm.progress import (  # noqa: E402
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget, _DemoHost

# A report an emitter actually produces: the within-tile z-stack progress from
# `acquire_z_stack`. It reaches the bar and the label, which is what these tests need
# a live slot to touch.
PROGRESS = FluorescenceAcquisitionProgress(
    status=FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
    channel="DAPI",
    channel_index=1,
    total_channels=1,
    zlevel=1,
    total_zlevels=3,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    host = _DemoHost()
    w = FMControlWidget(microscope=build_microscope(), parent=host)
    # Keep the host alive for the test's duration; the widget only weakly needs it,
    # but a collected parent would destroy the child and mask what we are testing.
    w._test_host = host
    return w


def _fm_slots(signal):
    return [repr(s) for s in signal._slots]


def test_teardown_disconnects_every_microscope_signal(widget):
    """All three, not just `acquisition_signal` -- the other two were missed."""
    fm = widget.fm
    widget._teardown_connections()

    for name, signal, slot_name in (
        ("acquisition_signal", fm.acquisition_signal, "update_image"),
        (
            "acquisition_progress_signal",
            fm.acquisition_progress_signal,
            "_on_acquisition_progress",
        ),
        ("acquiring_changed", fm.acquiring_changed, "_on_fm_acquiring_changed"),
    ):
        assert not any(slot_name in s for s in _fm_slots(signal)), (
            f"{name} still connected to {slot_name} after teardown"
        )


def test_emitting_after_teardown_and_delete_does_not_raise(widget):
    """The real sequence, verbatim: teardown -> the C++ object goes -> a worker emits.

    ``sip.delete`` is the synchronous stand-in for ``deleteLater``, so this needs no
    event loop. Before the fix each of these raised RuntimeError out of the emit.
    """
    fm = widget.fm
    widget._current_acquisition_type = "image"

    widget._teardown_connections()
    sip.delete(widget)
    assert sip.isdeleted(widget)

    # Would raise psygnal's EmitLoopError (from RuntimeError) if either slot survived.
    fm.acquisition_progress_signal.emit(PROGRESS)
    fm.acquiring_changed.emit(False)
    fm.acquisition_signal.emit(None)


def test_teardown_is_idempotent(widget):
    """Called from `closeEvent` AND the host's `deleteLater` path, so it runs twice."""
    for _ in range(3):
        widget._teardown_connections()


def test_progress_before_any_local_acquisition_does_not_raise(widget):
    """A workflow task drives the FM while the tab sits open.

    The widget is connected from construction, so it receives progress for an
    acquisition it never started -- `_current_acquisition_type` is still None, and the
    handler runs anyway. Left deliberately unset here: that *is* the condition.
    """
    assert widget._current_acquisition_type is None
    widget.fm.acquisition_progress_signal.emit(PROGRESS)
