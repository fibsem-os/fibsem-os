"""The Responder seam, driven end to end: workflow thread in, GUI thread out.

First conversion off the hand-rolled RPC: ``set_images_ui`` sends a ``SetImages``
request through ``QtResponder`` and waits on a future of its own, instead of
setting ``WAITING_FOR_UI_UPDATE``, emitting a dict, and sleep-polling a flag any
other emitter could clear.

Everything here calls ``set_images_ui`` from a real worker thread while the test
spins the GUI event loop — the same shape as production, where the workflow
thread blocks and the GUI thread answers.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from copy import deepcopy

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows import ui as workflow_ui
from fibsem.applications.autolamella.workflows.interaction import (
    EditAlignmentArea,
    ask,
)
from fibsem.applications.autolamella.workflows.ui import set_images_ui
from fibsem.structures import FibsemRectangle


@pytest.fixture
def ui(qapp, monkeypatch):
    """A real AutoLamellaUI, connected (Demo), same harness as the payload tests.

    Pinned to the simulated Arctis configuration rather than whatever this
    machine's default is: `DemoMicroscope` only builds a fluorescence microscope
    on a compustage config, and the fluorescence test needs `fm_control_widget`
    to exist. Resolving the default made the test pass on a machine whose
    default is a compustage and fail on CI, whose default is not.
    """
    import fibsem.config as fibsem_config

    arctis_config = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    widget = AutoLamellaUI(parent_ui=None)
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: arctis_config,
    )
    widget.system_widget.connect_to_microscope()
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _call_on_worker_thread(qapp, fn, timeout_s=10.0):
    """Run ``fn`` on a worker thread while spinning the GUI loop; return its outcome."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "worker thread never finished"
    return outcome


def test_images_land_in_the_widget_from_a_worker_thread(ui, qapp):
    sem = deepcopy(ui.image_widget.eb_image)
    fib = deepcopy(ui.image_widget.ib_image)

    outcome = _call_on_worker_thread(
        qapp, lambda: set_images_ui(ui, eb_image=sem, ib_image=fib)
    )

    assert "error" not in outcome
    assert ui.image_widget.eb_image is sem
    assert ui.image_widget.ib_image is fib
    # The old mechanism must stay untouched: nothing set or cleared the flag.
    assert ui.WAITING_FOR_UI_UPDATE is False


def test_the_widget_update_runs_on_the_gui_thread(ui, qapp):
    threads = []
    original = ui.image_widget._on_acquire

    def recording(image):
        threads.append(threading.current_thread().name)
        return original(image)

    ui.image_widget._on_acquire = recording
    sem = deepcopy(ui.image_widget.eb_image)

    _call_on_worker_thread(qapp, lambda: set_images_ui(ui, eb_image=sem))

    assert threads == ["MainThread"]


def test_a_gui_side_failure_reraises_on_the_workflow_thread(ui, qapp):
    # Under the old signal this exception escaped a queued slot, which PyQt5
    # turns into a process abort (FIB-329) — the pytest process would be gone.
    # Now it is this instruction's failure, on the thread that asked.
    def broken(image):
        raise RuntimeError("display fell over")

    ui.image_widget._on_acquire = broken
    sem = deepcopy(ui.image_widget.eb_image)

    outcome = _call_on_worker_thread(qapp, lambda: set_images_ui(ui, eb_image=sem))

    assert isinstance(outcome.get("error"), RuntimeError)
    assert "display fell over" in str(outcome["error"])


def test_an_unhandled_request_type_fails_fast_not_by_timeout(ui, qapp):
    # EditAlignmentArea has no QtResponder handler yet (it converts in a later
    # step). Asking for it must fail the caller immediately with a TypeError,
    # not leave it hanging until the instruction timeout.
    started = time.monotonic()
    outcome = _call_on_worker_thread(
        qapp,
        lambda: ask(ui.ui_responder, EditAlignmentArea(initial=FibsemRectangle())),
    )

    assert isinstance(outcome.get("error"), TypeError)
    assert "EditAlignmentArea" in str(outcome["error"])
    assert time.monotonic() - started < 5


def _call_with_the_gui_wedged(fn, timeout_s=10.0):
    """Run ``fn`` on a worker thread and never spin the GUI loop.

    From the main thread the responder's connection is direct and dispatch is
    synchronous — nothing can wedge. From a worker thread the dispatch is queued,
    and with nobody pumping events it never runs: exactly a wedged GUI thread.
    """
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    assert not thread.is_alive(), "the caller is stuck: neither abort nor timeout"
    return outcome


def test_a_stop_interrupts_the_wait(ui):
    # Stop must get the caller out of a wait the GUI will never answer, and as
    # InterruptedError — the abort path's own exception, one unwind path. The
    # event is set only after the call is under way, so the entry abort check
    # cannot be what raises: this pins the wait loop's own abort.
    sem = deepcopy(ui.image_widget.eb_image)
    outcome = {}

    def target():
        try:
            set_images_ui(ui, eb_image=sem)
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    time.sleep(0.3)  # past the entry check and into the wait
    ui._workflow_stop_event.set()  # legacy event: no task manager in this harness
    try:
        thread.join(timeout=10.0)
        assert not thread.is_alive(), "Stop did not interrupt the wait"
    finally:
        ui._workflow_stop_event.clear()

    assert isinstance(outcome.get("error"), InterruptedError)


def test_a_wedged_gui_thread_times_out_instead_of_hanging_forever(ui, monkeypatch):
    monkeypatch.setattr(workflow_ui, "INSTRUCTION_TIMEOUT_S", 0.4)
    sem = deepcopy(ui.image_widget.eb_image)
    started = time.monotonic()

    outcome = _call_with_the_gui_wedged(lambda: set_images_ui(ui, eb_image=sem))

    assert isinstance(outcome.get("error"), TimeoutError)
    assert time.monotonic() - started < 5


# ── milling config instructions ─────────────────────────────────────────────────


def test_a_milling_config_lands_in_the_editor_and_fronts_its_tab(ui, qapp):
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    config = FibsemMillingTaskConfig(name="from-the-workflow")

    from fibsem.applications.autolamella.workflows.interaction import (
        SetMillingConfig,
    )

    outcome = _call_on_worker_thread(
        qapp,
        lambda: ask(ui.ui_responder, SetMillingConfig(config=config)),
    )

    assert "error" not in outcome
    assert ui.milling_task_config_widget.get_config().name == "from-the-workflow"
    assert ui.tabWidget.currentWidget() is ui.milling_task_config_widget
    assert ui.WAITING_FOR_UI_UPDATE is False


def test_clearing_the_milling_config_resets_the_editor(ui, qapp):
    from fibsem.applications.autolamella.workflows.interaction import (
        ClearMillingConfig,
        SetMillingConfig,
    )
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    _call_on_worker_thread(
        qapp,
        lambda: ask(
            ui.ui_responder,
            SetMillingConfig(config=FibsemMillingTaskConfig(name="to-be-cleared")),
        ),
    )
    assert ui.milling_task_config_widget.get_config().name == "to-be-cleared"

    outcome = _call_on_worker_thread(
        qapp, lambda: ask(ui.ui_responder, ClearMillingConfig())
    )

    assert "error" not in outcome
    assert (
        ui.milling_task_config_widget.get_config().name
        == FibsemMillingTaskConfig().name
    )


# ── fluorescence + spot-burn instructions ───────────────────────────────────────


def test_spot_burn_settings_land_in_the_widget(ui, qapp):
    from fibsem.applications.autolamella.workflows.ui import (
        update_spot_burn_parameters,
    )
    from fibsem.imaging.spot import SpotBurnSettings

    settings = SpotBurnSettings(exposure_time=17.0)

    outcome = _call_on_worker_thread(
        qapp, lambda: update_spot_burn_parameters(ui, settings=settings)
    )

    assert "error" not in outcome
    assert ui.spot_burn_widget.get_settings().exposure_time == 17.0
    assert ui.WAITING_FOR_UI_UPDATE is False


def test_clearing_spot_burn_leaves_workflow_mode(ui, qapp):
    from fibsem.applications.autolamella.workflows.ui import clear_spot_burn_ui

    ui.spot_burn_widget.set_workflow_mode(True)

    outcome = _call_on_worker_thread(qapp, lambda: clear_spot_burn_ui(ui))

    assert "error" not in outcome
    assert ui.spot_burn_widget._workflow_mode is False


def test_fluorescence_channels_reach_the_fm_widget(ui, qapp):
    from fibsem.fm.structures import ChannelSettings

    channels = [ChannelSettings(name="channel-from-the-workflow")]

    class _Task:
        parent_ui = ui
        _check_for_abort = staticmethod(lambda *a, **k: False)

    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    outcome = _call_on_worker_thread(
        qapp,
        lambda: AutoLamellaTask.set_fluorescence_channels_ui(_Task(), channels),
    )

    assert "error" not in outcome
    got = ui.fm_control_widget.channelSettingsWidget.channel_settings
    assert [c.name for c in got] == ["channel-from-the-workflow"]
