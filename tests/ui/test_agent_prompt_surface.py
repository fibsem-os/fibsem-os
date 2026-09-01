"""Supervision answering end to end: a real workflow-thread question, seen and
answered over HTTP through the mounted server — the killer-app data path.

Three threads, like production: the worker asks through ``ask()``, the GUI spins,
and the 'agent' speaks HTTP from the test thread via TestClient."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from fibsem.applications.autolamella.server import AgentContext
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.interaction import Confirm, ask
from fibsem.server import AuthConfig, build_server

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def ui(qapp, monkeypatch):
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
    widget.deleteLater()
    qapp.processEvents()


def _client(ui, arm_control):
    app = build_server(
        ui.microscope,
        app_context=AgentContext(ui),
        auth=AuthConfig.generate(arm_control=arm_control, token=TOKEN),
    )
    return TestClient(app, raise_server_exceptions=False)


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _ask_on_worker(ui, qapp):
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(
                ui.ui_responder,
                Confirm("Continue to polishing?", positive="Continue", negative="Stop"),
            )
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _spin_until(qapp, lambda: ui.ui_responder.pending_question() is not None)
    return thread, outcome


def test_agent_sees_and_answers_the_question_over_http(ui, qapp):
    with _client(ui, arm_control=True) as client:
        # Nothing pending yet: available, but no question.
        body = client.get("/app/prompt", headers=AUTH).json()
        assert body == {"available": True, "pending": None}

        thread, outcome = _ask_on_worker(ui, qapp)
        pending = client.get("/app/prompt", headers=AUTH).json()["pending"]
        assert pending["type"] == "Confirm"
        assert pending["message"] == "Continue to polishing?"
        assert pending["positive"] == "Continue"
        assert isinstance(pending["nonce"], int)
        # A Confirm has no live half — no GUI peek happens, no "current" key.
        assert "current" not in pending

        # POST from a separate thread, as production does: the answer applies on
        # the GUI thread, which in this test is the thread running the loop —
        # a blocking post from here would deadlock into the 10 s timeout.
        posted = {}

        def do_post():
            posted["response"] = client.post(
                "/app/prompt/answer",
                headers=AUTH,
                json={"response": True, "nonce": pending["nonce"]},
            )

        poster = threading.Thread(target=do_post, daemon=True)
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        answered = posted["response"]
        assert answered.status_code == 200
        assert answered.json()["applied"] is True
        _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
        thread.join(timeout=5)
        assert outcome.get("answer") is True


def test_a_stale_nonce_is_refused_and_the_question_stands(ui, qapp):
    with _client(ui, arm_control=True) as client:
        thread, outcome = _ask_on_worker(ui, qapp)
        nonce = client.get("/app/prompt", headers=AUTH).json()["pending"]["nonce"]

        posted = {}

        def do_post():
            posted["response"] = client.post(
                "/app/prompt/answer",
                headers=AUTH,
                json={"response": True, "nonce": nonce - 1},
            )

        poster = threading.Thread(target=do_post, daemon=True)
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        refused = posted["response"]
        assert refused.status_code == 409
        assert refused.json()["detail"]["error_type"] == "stale_prompt"
        # Nothing was clicked: the question still stands and the asker waits.
        assert ui.ui_responder.pending_question() is not None
        assert outcome == {}
        # Clean up: answer for real so the worker thread can finish.
        ui.ui_responder.answer_confirm(False)
        _spin_until(qapp, lambda: "answer" in outcome)
        thread.join(timeout=5)


def test_display_images_are_served_read_scope(ui, qapp):
    with _client(ui, arm_control=False) as client:
        body = client.get("/app/images", headers=AUTH).json()
        assert body["available"] is True
        # A connected UI displays the seeded placeholder images on both sides.
        for beam in ("sem", "fib"):
            assert body[beam]["image_b64_jpeg"]


def test_display_images_follow_a_one_shot_acquisition(ui, qapp):
    """The post-mill inspect image reaches /app/images, pixels AND acquired_at.

    A milling task's inspect acquisition is a one-shot emit on
    fib_acquisition_signal with no live stream running. On the 2026-08-31
    supervised run, /app/images kept serving the pre-mill reference through
    the whole inspect prompt — the supervising agent refuses images whose
    acquired_at predates the mill, so staleness here blocks supervision."""
    from datetime import datetime

    from fibsem.structures import BeamType, FibsemImage, MicroscopeState

    inspect_image = FibsemImage.generate_blank_image(
        resolution=(1536, 1024), hfw=100e-6, random=True
    )
    inspect_image.metadata.image_settings.beam_type = BeamType.ION
    stamp = datetime(2026, 9, 1, 12, 34, 56)
    inspect_image.metadata.microscope_state = MicroscopeState(
        timestamp=stamp.timestamp()
    )

    with _client(ui, arm_control=False) as client:
        assert not ui.microscope.is_acquiring, "this test pins the one-shot path"
        ui.microscope.fib_acquisition_signal.emit(inspect_image)
        qapp.processEvents()
        fib = client.get("/app/images", headers=AUTH).json()["fib"]
        assert fib["acquired_at"] == stamp.isoformat()
        # The display cache serves the same scale facts as the acquisition
        # endpoints: overlays drawn on the preview must not guess the FOV.
        assert fib["full_width"] == 1536
        assert fib["hfw"] == pytest.approx(100e-6)
        assert fib["pixelsize"]["x"] == pytest.approx(100e-6 / 1536)


def test_an_answer_without_a_nonce_is_rejected(ui, qapp):
    with _client(ui, arm_control=True) as client:
        rejected = client.post(
            "/app/prompt/answer", headers=AUTH, json={"response": True}
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["error_type"] == "missing_nonce"


def test_prompt_lifecycle_reaches_the_event_stream_with_attribution(ui, qapp):
    # Wired the way hosting wires it: responder feeds the same buffer the
    # events endpoint serves — the long-poll replaces polling /app/prompt.
    from fibsem.applications.autolamella.server.events import EventBuffer

    buffer = EventBuffer()
    app = build_server(
        ui.microscope,
        app_context=AgentContext(ui, event_buffer=buffer),
        auth=AuthConfig.generate(arm_control=True, token=TOKEN),
    )
    dispose = ui.ui_responder.add_question_observer(buffer.append)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            thread, outcome = _ask_on_worker(ui, qapp)
            raised = client.get("/app/events?since=0", headers=AUTH).json()["events"]
            assert raised[-1]["kind"] == "prompt_raised"
            nonce = raised[-1]["payload"]["nonce"]

            posted = {}

            def do_post():
                posted["response"] = client.post(
                    "/app/prompt/answer",
                    headers=AUTH,
                    json={"response": True, "nonce": nonce},
                )

            poster = threading.Thread(target=do_post, daemon=True)
            poster.start()
            _spin_until(qapp, lambda: "response" in posted)
            assert posted["response"].status_code == 200
            thread.join(timeout=5)

            events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
            answered = [e for e in events if e["kind"] == "prompt_answered"][-1]
            assert answered["payload"]["answered_by"] == "agent"
            assert answered["payload"]["response"] is True
            assert answered["payload"]["nonce"] == nonce
    finally:
        dispose()


def test_stop_workflow_rides_the_read_scope(ui, qapp):
    # The safety action: available without arming, like stop_milling.
    with _client(ui, arm_control=False) as client:
        body = client.post("/app/workflow/stop", headers=AUTH)
        assert body.status_code == 200
        assert body.json() == {
            "available": True,
            "stopped": False,
            "reason": "no workflow is running",
        }


def test_start_workflow_validates_and_starts_on_the_gui_thread(ui, qapp):
    from psygnal.containers import EventedDict

    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskDescription,
        AutoLamellaTaskProtocol,
        Experiment,
    )
    from fibsem.structures import MicroscopeState

    experiment = Experiment(path="/tmp/agent-start", name="agent-start")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(name="Mill Fiducial", supervise=True, required=True)
    )
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    ui.experiment = experiment

    calls = []

    class _AliveThread:
        @staticmethod
        def is_alive():
            return True

    def fake_start(task_names, item_names):
        calls.append((task_names, item_names))
        ui._task_worker_thread = _AliveThread()

    ui._start_run_workflow_thread = fake_start

    with _client(ui, arm_control=True) as client:
        posted = {}

        def do_post(body):
            posted["response"] = client.post(
                "/app/workflow/start", headers=AUTH, json=body
            )

        # Unknown task: structured refusal carrying the valid names.
        poster = threading.Thread(
            target=do_post, args=({"task_names": ["No Such Task"]},), daemon=True
        )
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        refused = posted["response"].json()
        assert refused["started"] is False
        assert refused["task_names"] == ["Mill Fiducial"]
        assert calls == []

        # Valid start; omitted items means every item in the experiment.
        posted.clear()
        poster = threading.Thread(
            target=do_post, args=({"task_names": ["Mill Fiducial"]},), daemon=True
        )
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        assert posted["response"].json() == {"available": True, "started": True}
        assert calls == [(["Mill Fiducial"], [experiment.positions[0].name])]

        # And now that it is "running", a second start is refused.
        posted.clear()
        poster = threading.Thread(
            target=do_post, args=({"task_names": ["Mill Fiducial"]},), daemon=True
        )
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        assert posted["response"].json()["started"] is False
        ui._task_worker_thread = None

    ui.experiment = None


def test_start_workflow_requires_control_and_a_valid_body(ui, qapp):
    with _client(ui, arm_control=False) as client:
        refused = client.post(
            "/app/workflow/start", headers=AUTH, json={"task_names": ["x"]}
        )
        assert refused.status_code == 403
        assert refused.json()["detail"]["scope"] == "control"
    with _client(ui, arm_control=True) as client:
        rejected = client.post("/app/workflow/start", headers=AUTH, json={})
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["error_type"] == "missing_field"


def test_supervision_and_requeue_require_the_control_scope(ui, qapp):
    with _client(ui, arm_control=False) as client:
        for path, payload in (
            ("/app/supervision", {"task_name": "Rough Milling", "supervise": True}),
            ("/app/queue/requeue", {"item_name": "01", "task_name": "x"}),
        ):
            refused = client.post(path, headers=AUTH, json=payload)
            assert refused.status_code == 403
            assert refused.json()["detail"]["scope"] == "control"
    with _client(ui, arm_control=True) as client:
        for path in ("/app/supervision", "/app/queue/requeue"):
            rejected = client.post(path, headers=AUTH, json={})
            assert rejected.status_code == 422
            assert rejected.json()["detail"]["error_type"] == "missing_field"


def test_answering_requires_the_control_scope(ui, qapp):
    with _client(ui, arm_control=False) as client:
        # Reading the prompt is observation: read scope suffices.
        assert client.get("/app/prompt", headers=AUTH).status_code == 200
        refused = client.post(
            "/app/prompt/answer", headers=AUTH, json={"response": True, "nonce": 1}
        )
        assert refused.status_code == 403
        assert refused.json()["detail"]["scope"] == "control"


def test_capabilities_reflect_the_armed_control_scope(ui, qapp):
    with _client(ui, arm_control=True) as client:
        scopes = client.get("/capabilities", headers=AUTH).json()["scopes"]
        assert scopes["control"] is True
        from fibsem.server.catalog import tools_for_capabilities

        names = {
            t.name
            for t in tools_for_capabilities(
                client.get("/capabilities", headers=AUTH).json()
            )
        }
        assert "answer_prompt" in names
    with _client(ui, arm_control=False) as client:
        from fibsem.server.catalog import tools_for_capabilities

        names = {
            t.name
            for t in tools_for_capabilities(
                client.get("/capabilities", headers=AUTH).json()
            )
        }
        assert "get_pending_prompt" in names
        assert "answer_prompt" not in names
