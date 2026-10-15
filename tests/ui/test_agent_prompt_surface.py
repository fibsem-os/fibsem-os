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

from fibsem.applications.autolamella.server import AutoLamellaAppContext
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
        app_context=AutoLamellaAppContext(ui),
        auth=AuthConfig.generate(
            level="drive" if arm_control else "watch", token=TOKEN
        ),
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
        body = client.get("/app/workflow/prompt", headers=AUTH).json()
        assert body == {"available": True, "pending": None}

        thread, outcome = _ask_on_worker(ui, qapp)
        pending = client.get("/app/workflow/prompt", headers=AUTH).json()["pending"]
        assert pending["type"] == "Confirm"
        assert pending["message"] == "Continue to polishing?"
        assert pending["positive"] == "Continue"
        assert isinstance(pending["question_id"], int)
        # A Confirm has no live half — no GUI peek happens, no "current" key.
        assert "current" not in pending

        # POST from a separate thread, as production does: the answer applies on
        # the GUI thread, which in this test is the thread running the loop —
        # a blocking post from here would deadlock into the 10 s timeout.
        posted = {}

        def do_post():
            posted["response"] = client.post(
                "/app/workflow/prompt/answer",
                headers=AUTH,
                json={"response": True, "question_id": pending["question_id"]},
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


def test_a_stale_question_id_is_refused_and_the_question_stands(ui, qapp):
    with _client(ui, arm_control=True) as client:
        thread, outcome = _ask_on_worker(ui, qapp)
        question_id = client.get("/app/workflow/prompt", headers=AUTH).json()[
            "pending"
        ]["question_id"]

        posted = {}

        def do_post():
            posted["response"] = client.post(
                "/app/workflow/prompt/answer",
                headers=AUTH,
                json={"response": True, "question_id": question_id - 1},
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
        body = client.get("/app/experiment/images", headers=AUTH).json()
        assert body["available"] is True
        # A connected UI displays the seeded placeholder images on both sides.
        for beam in ("sem", "fib"):
            assert body[beam]["image_b64_jpeg"]


def test_display_images_follow_a_one_shot_acquisition(ui, qapp):
    """The post-mill inspect image reaches /app/experiment/images, pixels AND acquired_at.

    A milling task's inspect acquisition is a one-shot emit on
    fib_acquisition_signal with no live stream running. On the 2026-08-31
    supervised run, /app/experiment/images kept serving the pre-mill reference through
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
        fib = client.get("/app/experiment/images", headers=AUTH).json()["fib"]
        # With its UTC offset, as the event stream's times carry theirs (FIB-1190).
        assert fib["acquired_at"] == stamp.astimezone().isoformat()
        # The display cache serves the same scale facts as the acquisition
        # endpoints: overlays drawn on the preview must not guess the FOV.
        assert fib["full_width"] == 1536
        assert fib["hfw"] == pytest.approx(100e-6)
        assert fib["pixelsize"]["x"] == pytest.approx(100e-6 / 1536)


def test_alignment_prompt_carries_the_fib_display_frame(ui, qapp):
    """The area is fractions of the FIB frame; the frame itself is display
    state, attached by the server seam (not carried in the request, and not
    grabbed by the workflow) — so a remote responder can judge the area
    without pairing a second request, and sees exactly what the operator
    sees."""
    from fibsem.applications.autolamella.workflows.interaction import (
        EditAlignmentArea,
    )
    from fibsem.structures import BeamType, FibsemImage, FibsemRectangle

    fib_frame = FibsemImage.generate_blank_image(
        resolution=(1536, 1024), hfw=80e-6, random=True
    )
    fib_frame.metadata.image_settings.beam_type = BeamType.ION
    ui.image_widget.ib_image = fib_frame

    area = FibsemRectangle(left=0.4, top=0.4, width=0.2, height=0.2)
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(ui.ui_responder, EditAlignmentArea(initial=area))
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _spin_until(qapp, lambda: ui.ui_responder.pending_question() is not None)

    with _client(ui, arm_control=False) as client:
        fetched = {}

        def do_get():
            fetched["pending"] = client.get(
                "/app/workflow/prompt", headers=AUTH
            ).json()["pending"]

        # GET from a separate thread: the live-half peek marshals to the GUI
        # thread, which in this test is the thread spinning the loop.
        getter = threading.Thread(target=do_get, daemon=True)
        getter.start()
        _spin_until(qapp, lambda: "pending" in fetched)

    pending = fetched["pending"]
    assert pending["type"] == "EditAlignmentArea"
    entry = pending["image"]
    assert entry["full_width"] == 1536
    assert entry["hfw"] == pytest.approx(80e-6)
    assert entry["pixelsize"]["x"] == pytest.approx(80e-6 / 1536)

    ui.pushButton_yes.click()
    _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
    thread.join(timeout=5)
    assert "error" not in outcome


def test_a_value_on_a_valueless_question_is_refused(ui, qapp):
    """Only EditAlignmentArea, PickPOI, ConfirmDetection and RunSpotBurn answers
    can carry a value; a value on anything else — or a value that is not an object — is a 422 refusal that
    clicks nothing, and the question stands."""
    with _client(ui, arm_control=True) as client:
        thread, outcome = _ask_on_worker(ui, qapp)
        question_id = client.get("/app/workflow/prompt", headers=AUTH).json()[
            "pending"
        ]["question_id"]

        malformed = client.post(
            "/app/workflow/prompt/answer",
            headers=AUTH,
            json={"response": True, "question_id": question_id, "value": [1, 2]},
        )
        assert malformed.status_code == 422
        assert malformed.json()["detail"]["error_type"] == "invalid_value"

        posted = {}

        def do_post():
            posted["response"] = client.post(
                "/app/workflow/prompt/answer",
                headers=AUTH,
                json={
                    "response": True,
                    "question_id": question_id,
                    "value": {"x": 0, "y": 0},
                },
            )

        poster = threading.Thread(target=do_post, daemon=True)
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        refused = posted["response"]
        assert refused.status_code == 422
        assert refused.json()["detail"]["error_type"] == "invalid_value"
        assert "Confirm" in refused.json()["detail"]["message"]

        # The refusals clicked nothing: the same question id still answers.
        posted.clear()

        def do_plain_post():
            posted["response"] = client.post(
                "/app/workflow/prompt/answer",
                headers=AUTH,
                json={"response": True, "question_id": question_id},
            )

        poster = threading.Thread(target=do_plain_post, daemon=True)
        poster.start()
        _spin_until(qapp, lambda: "response" in posted)
        assert posted["response"].status_code == 200
        _spin_until(qapp, lambda: "answer" in outcome or "error" in outcome)
        thread.join(timeout=5)
        assert outcome.get("answer") is True


def test_an_answer_without_a_question_id_is_rejected(ui, qapp):
    with _client(ui, arm_control=True) as client:
        rejected = client.post(
            "/app/workflow/prompt/answer", headers=AUTH, json={"response": True}
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["error_type"] == "missing_question_id"


def test_prompt_lifecycle_reaches_the_event_stream_with_attribution(ui, qapp):
    # Wired the way hosting wires it: responder feeds the same buffer the
    # events endpoint serves — the long-poll replaces polling /app/workflow/prompt.
    from fibsem.applications.autolamella.server.events import EventBuffer

    buffer = EventBuffer()
    app = build_server(
        ui.microscope,
        app_context=AutoLamellaAppContext(ui, event_buffer=buffer),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    dispose = ui.ui_responder.add_question_observer(buffer.append)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            thread, outcome = _ask_on_worker(ui, qapp)
            raised = client.get("/app/events?since=0", headers=AUTH).json()["events"]
            assert raised[-1]["kind"] == "prompt_raised"
            question_id = raised[-1]["payload"]["question_id"]

            posted = {}

            def do_post():
                posted["response"] = client.post(
                    "/app/workflow/prompt/answer",
                    headers=AUTH,
                    json={"response": True, "question_id": question_id},
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
            assert answered["payload"]["question_id"] == question_id
    finally:
        dispose()


def test_stop_workflow_needs_the_control_scope(ui, qapp):
    # A watching agent stops nothing (FIB-1035); one that can act can stop.
    with _client(ui, arm_control=False) as client:
        assert client.post("/app/workflow/stop", headers=AUTH).status_code == 403
    with _client(ui, arm_control=True) as client:
        body = client.post(
            "/app/workflow/stop", headers=AUTH, json={"reason": "test stop"}
        )
        assert body.status_code == 200
        assert body.json() == {
            "available": True,
            "stopped": False,
            "reason": "no workflow is running",
        }


def test_start_workflow_validates_and_starts_on_the_gui_thread(ui, qapp):
    from psygnal.containers import EventedDict

    from fibsem.applications.autolamella.structures import (
        Attention,
        AutoLamellaTaskDescription,
        AutoLamellaTaskProtocol,
        Experiment,
    )
    from fibsem.structures import MicroscopeState

    experiment = Experiment(path="/tmp/agent-start", name="agent-start")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(
            name="Mill Fiducial", attention=Attention.supervised, required=True
        )
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
        assert refused.json()["detail"]["level"] == "watch"
    with _client(ui, arm_control=True) as client:
        rejected = client.post("/app/workflow/start", headers=AUTH, json={})
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["error_type"] == "missing_field"


def test_supervision_and_requeue_require_the_control_scope(ui, qapp):
    with _client(ui, arm_control=False) as client:
        for path, payload in (
            (
                "/app/agent/supervision",
                {"task_name": "Rough Milling", "supervise": True},
            ),
            ("/app/workflow/queue/requeue", {"item_name": "01", "task_name": "x"}),
        ):
            refused = client.post(path, headers=AUTH, json=payload)
            assert refused.status_code == 403
            assert refused.json()["detail"]["level"] == "watch"
    with _client(ui, arm_control=True) as client:
        for path in ("/app/agent/supervision", "/app/workflow/queue/requeue"):
            rejected = client.post(path, headers=AUTH, json={})
            assert rejected.status_code == 422
            assert rejected.json()["detail"]["error_type"] == "missing_field"


def test_answering_requires_the_control_scope(ui, qapp):
    with _client(ui, arm_control=False) as client:
        # Reading the prompt is observation: read scope suffices.
        assert client.get("/app/workflow/prompt", headers=AUTH).status_code == 200
        refused = client.post(
            "/app/workflow/prompt/answer",
            headers=AUTH,
            json={"response": True, "question_id": 1},
        )
        assert refused.status_code == 403
        assert refused.json()["detail"]["level"] == "watch"


def test_capabilities_reflect_the_level(ui, qapp):
    with _client(ui, arm_control=True) as client:
        level = client.get("/capabilities", headers=AUTH).json()["level"]
        assert level == "drive"
        from fibsem.server.catalog import tools_for_capabilities

        names = {
            t.name
            for t in tools_for_capabilities(
                client.get("/capabilities", headers=AUTH).json()
            )
        }
        assert "prompt_answer" in names
    with _client(ui, arm_control=False) as client:
        from fibsem.server.catalog import tools_for_capabilities

        names = {
            t.name
            for t in tools_for_capabilities(
                client.get("/capabilities", headers=AUTH).json()
            )
        }
        assert "prompt_get" in names
        # Offered at Watch too: the server refuses it (FIB-1035).
        assert "prompt_answer" in names
