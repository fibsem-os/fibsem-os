"""The spot-burn question over the Responder seam: RunSpotBurn.

The old supervised loop mirrored milling's hand-rolled RPC: ``ask_user(spot_burn=
True)`` parked on the polled flag, the workflow thread emitted
``start_spot_burn_signal`` over a BlockingQueuedConnection, sleep-polled
``widget.is_burning`` across the seam, then read ``get_settings()`` back and sent
a clear instruction. Now the whole loop lives in ``QtResponder``, and the answer
is the settings as actually used — or None without a spot-burn widget (optional
hardware must not fail a workflow).

``run_spot_burn`` is stubbed at the widget's import site: runs still go through
the widget's real worker thread, button states and new finished signal — only
the beam time is gone.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.workflows.interaction import RunSpotBurn, ask
from fibsem.imaging.spot import SpotBurnSettings
from fibsem.structures import Point

MSG = "Place the points, then run."


def _settings(with_point: bool = True) -> SpotBurnSettings:
    return SpotBurnSettings(
        coordinates=[Point(x=0.5, y=0.5)] if with_point else [],
        exposure_time=7.0,
    )


@pytest.fixture(scope="module")
def ui(qapp):
    """The embedded AutoLamellaUI of a real main window, with the burn stubbed.

    The full main window, because entering spot-burn mode arms the coordinate
    overlay on the quad-view controller, which lives there — same reason the
    alignment-area and POI tests use it.
    """
    import importlib

    # Not `import fibsem.ui.FibsemSpotBurnWidget`: the package __init__ rebinds
    # that attribute to the widget CLASS, shadowing the submodule.
    sbw = importlib.import_module("fibsem.ui.FibsemSpotBurnWidget")
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    burns = []

    def fake_run_spot_burn(microscope, settings, beam_type, stop_event=None, **kwargs):
        burns.append(settings)
        time.sleep(0.05)

    original_run = sbw.run_spot_burn
    sbw.run_spot_burn = fake_run_spot_burn
    original_minimap = module.AutoLamellaSingleWindowUI.add_minimap_tab
    module.AutoLamellaSingleWindowUI.add_minimap_tab = lambda self: None
    try:
        window = module.AutoLamellaSingleWindowUI()
    finally:
        module.AutoLamellaSingleWindowUI.add_minimap_tab = original_minimap
    window.autolamella_ui.system_widget.connect_to_microscope()
    # The main window's _on_workflow_update reads _border_state unconditionally,
    # first assigned on the Run click (the latent init gap documented in
    # test_mainui_workflow_status; its fix rides the typed-status PR).
    window._set_border_state("idle")
    window.autolamella_ui._burn_runs = burns  # for the tests to inspect
    yield window.autolamella_ui
    sbw.run_spot_burn = original_run
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    # closeEvent ends in app.quit(); on the shared test QApplication that latches
    # and breaks every later QEventLoop.exec_() — close with quit stubbed out.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture(autouse=True)
def _fresh_runs(ui):
    ui._burn_runs.clear()
    yield


def _ask_on_worker_thread(ui, qapp, request, wait_for_prompt=True):
    outcome = {}

    def target():
        try:
            outcome["settings"] = ask(
                ui.ui_responder, request, abort=ui._workflow_stop_event.is_set
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    if wait_for_prompt:
        _wait_for_prompt(ui, qapp, request.message)
    return thread, outcome


def _wait_for_prompt(ui, qapp, msg, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if ui.label_instructions.text() == msg and ui.pushButton_yes.isEnabled():
            return
        time.sleep(0.01)
    raise AssertionError(f"the prompt {msg!r} never appeared")


def _finish(thread, qapp, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "the asker never returned"


def test_run_then_continue_burns_once_and_answers_the_settings(ui, qapp):
    request = RunSpotBurn(settings=_settings(), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    assert ui.pushButton_yes.text() == "Run Spot Burn"
    assert ui.pushButton_no.text() == "Continue"
    assert ui.spot_burn_widget._workflow_mode is True
    assert ui.WAITING_FOR_USER_INTERACTION is True

    ui.pushButton_yes.click()  # run
    # Prompt down while burning; back when the widget's finished signal fires.
    _wait_for_prompt(ui, qapp, MSG)
    assert len(ui._burn_runs) == 1

    ui.pushButton_no.click()  # continue
    _finish(thread, qapp)

    assert "error" not in outcome
    settings = outcome["settings"]
    assert isinstance(settings, SpotBurnSettings)
    assert settings.exposure_time == 7.0
    assert len(settings.coordinates) == 1
    # The question cleared the widget on its way out.
    assert ui.spot_burn_widget._workflow_mode is False
    assert ui.WAITING_FOR_USER_INTERACTION is False
    assert ui.WAITING_FOR_UI_UPDATE is False


def test_continue_without_running_answers_without_a_burn(ui, qapp):
    request = RunSpotBurn(settings=_settings(), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert "error" not in outcome
    assert isinstance(outcome["settings"], SpotBurnSettings)
    assert ui._burn_runs == []


def test_a_refused_run_reasks_instead_of_hanging(ui, qapp):
    # No in-bounds points: run_spot_burn_worker refuses before starting a worker,
    # so no finished signal will ever come. The old is_burning poll fell straight
    # through and re-asked; the question must do the same, not hang forever.
    request = RunSpotBurn(settings=_settings(with_point=False), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    ui.pushButton_yes.click()  # run — refused
    _wait_for_prompt(ui, qapp, MSG)
    assert ui._burn_runs == []

    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert "error" not in outcome


def test_no_spot_burn_widget_answers_none_not_a_failure(ui, qapp):
    real_widget = ui.spot_burn_widget
    ui.spot_burn_widget = None
    outcome = {}

    def target():
        try:
            outcome["settings"] = ask(
                ui.ui_responder, RunSpotBurn(settings=_settings(), message=MSG)
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        _finish(thread, qapp)
    finally:
        ui.spot_burn_widget = real_widget

    assert "error" not in outcome
    assert outcome["settings"] is None


def test_a_stop_interrupts_the_prompt(ui, qapp):
    request = RunSpotBurn(settings=_settings(), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()

    assert isinstance(outcome.get("error"), InterruptedError)
