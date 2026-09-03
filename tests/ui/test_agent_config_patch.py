"""Config patches end to end: HTTP → configure scope → GUI-thread apply.

Same three-thread shape as the prompt-surface tests: the 'agent' speaks HTTP
from a worker thread, the apply marshals to the GUI thread the test spins.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AgentContext
from fibsem.applications.autolamella.server.events import EventBuffer
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTaskConfig,
)
from fibsem.server import AuthConfig, build_server
from fibsem.structures import MicroscopeState

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TASK = "Rough Milling"
DEPTH = "milling.mill_rough.stages.0.pattern.depth"


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
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
    exp = Experiment(path=tmp_path / "exp", name="config-patch-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.positions[0].task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    exp.task_protocol.task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    widget.experiment = exp
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _client(ui, buffer=None, arm_configure=True):
    app = build_server(
        ui.microscope,
        app_context=AgentContext(ui, event_buffer=buffer),
        auth=AuthConfig.generate(arm_configure=arm_configure, token=TOKEN),
    )
    return TestClient(app, raise_server_exceptions=False)


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _post_on_worker(qapp, client, path, body):
    """POST from a worker thread while the GUI loop spins (the apply marshals)."""
    posted = {}

    def target():
        posted["response"] = client.post(path, headers=AUTH, json=body)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _spin_until(qapp, lambda: "response" in posted)
    return posted["response"]


def test_a_patch_lands_with_diff_version_and_event(ui, qapp):
    buffer = EventBuffer()
    item = ui.experiment.positions[0].name
    with _client(ui, buffer=buffer) as client:
        doc = client.get(f"/app/items/{item}/task_config/{TASK}", headers=AUTH).json()
        old_depth = (
            ui.experiment.positions[0]
            .task_config[TASK]
            .milling["mill_rough"]
            .stages[0]
            .pattern.depth
        )

        resp = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}/task_config/{TASK}",
            {"patch": {DEPTH: 2.5e-6}, "version": doc["version"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied"] is True
        assert body["changes"] == [{"path": DEPTH, "old": old_depth, "new": 2.5e-6}]
        # The object actually changed, and the new version names the new state.
        config = ui.experiment.positions[0].task_config[TASK]
        assert config.milling["mill_rough"].stages[0].pattern.depth == 2.5e-6
        assert body["version"] != doc["version"]
        reread = client.get(
            f"/app/items/{item}/task_config/{TASK}", headers=AUTH
        ).json()
        assert reread["version"] == body["version"]
        # And the record shows it: config_edited on the event stream.
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        edited = [e for e in events if e["kind"] == "config_edited"]
        assert edited and edited[-1]["payload"]["changes"][0]["path"] == DEPTH


def test_a_stale_version_is_refused_and_nothing_changes(ui, qapp):
    item = ui.experiment.positions[0].name
    config = ui.experiment.positions[0].task_config[TASK]
    with _client(ui) as client:
        doc = client.get(f"/app/items/{item}/task_config/{TASK}", headers=AUTH).json()
        # The operator edits in between: the agent's version is now stale.
        config.milling["mill_rough"].stages[0].pattern.depth = 1.9e-6
        resp = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}/task_config/{TASK}",
            {"patch": {DEPTH: 2.5e-6}, "version": doc["version"]},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "stale_config"
        assert config.milling["mill_rough"].stages[0].pattern.depth == 1.9e-6


def test_invalid_patches_and_unknown_names_refuse_structuredly(ui, qapp):
    item = ui.experiment.positions[0].name
    with _client(ui) as client:
        doc = client.get(f"/app/items/{item}/task_config/{TASK}", headers=AUTH).json()
        bad = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}/task_config/{TASK}",
            {"patch": {DEPTH: "deep"}, "version": doc["version"]},
        )
        assert bad.status_code == 422
        assert bad.json()["detail"]["error_type"] == "invalid_patch"
        assert bad.json()["detail"]["path"] == DEPTH

        missing = _post_on_worker(
            qapp,
            client,
            f"/app/items/no-such-item/task_config/{TASK}",
            {"patch": {DEPTH: 2e-6}, "version": doc["version"]},
        )
        assert missing.status_code == 404
        assert missing.json()["detail"]["error_type"] == "not_found"

        malformed = client.post(
            f"/app/items/{item}/task_config/{TASK}",
            headers=AUTH,
            json={"patch": {}, "version": doc["version"]},
        )
        assert malformed.status_code == 422
        assert malformed.json()["detail"]["error_type"] == "missing_field"


def test_the_configure_scope_gates_the_route(ui, qapp):
    item = ui.experiment.positions[0].name
    with _client(ui, arm_configure=False) as client:
        # Reads stay read-scope; the write needs configure armed.
        doc = client.get(f"/app/items/{item}/task_config/{TASK}", headers=AUTH).json()
        assert doc["available"] is True
        resp = client.post(
            f"/app/items/{item}/task_config/{TASK}",
            headers=AUTH,
            json={"patch": {DEPTH: 2e-6}, "version": doc["version"]},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["scope"] == "configure"


def test_a_protocol_level_patch_lands_and_is_recorded(ui, qapp):
    buffer = EventBuffer()
    with _client(ui, buffer=buffer) as client:
        doc = client.get(f"/app/protocol/task_config/{TASK}", headers=AUTH).json()
        resp = _post_on_worker(
            qapp,
            client,
            f"/app/protocol/task_config/{TASK}",
            {"patch": {DEPTH: 2.8e-6}, "version": doc["version"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied"] is True and "item_name" not in body
        config = ui.experiment.task_protocol.task_config[TASK]
        assert config.milling["mill_rough"].stages[0].pattern.depth == 2.8e-6
        # The item's own copy is untouched: protocol edits reach future items.
        item_config = ui.experiment.positions[0].task_config[TASK]
        assert item_config.milling["mill_rough"].stages[0].pattern.depth != 2.8e-6
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        edited = [e for e in events if e["kind"] == "config_edited"][-1]
        assert edited["payload"]["level"] == "protocol"

        stale = _post_on_worker(
            qapp,
            client,
            f"/app/protocol/task_config/{TASK}",
            {"patch": {DEPTH: 3.1e-6}, "version": doc["version"]},
        )
        assert stale.status_code == 409


def test_a_patch_is_durable_on_disk(ui, qapp):
    import yaml

    item = ui.experiment.positions[0].name
    with _client(ui) as client:
        doc = client.get(f"/app/items/{item}/task_config/{TASK}", headers=AUTH).json()
        resp = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}/task_config/{TASK}",
            {"patch": {DEPTH: 2.2e-6}, "version": doc["version"]},
        )
        assert resp.json()["saved"] is True

    # The write survives the process: read the file, not the object.
    with open(os.path.join(ui.experiment.path, "experiment.yaml")) as f:
        saved = yaml.safe_load(f)
    lamella = next(p for p in saved["positions"] if p["petname"] == item)
    depth = lamella["task_config"][TASK]["milling"]["mill_rough"]["stages"][0][
        "pattern"
    ]["depth"]
    assert depth == 2.2e-6


def test_item_fields_patch_moves_poi_and_sets_defect(ui, qapp):
    buffer = EventBuffer()
    lamella = ui.experiment.positions[0]
    from fibsem.structures import Point

    lamella.poi = Point(x=1e-6, y=2e-6)
    item = lamella.name
    with _client(ui, buffer=buffer) as client:
        doc = client.get(f"/app/items/{item}", headers=AUTH).json()
        assert "version" in doc and doc["defect"]["state"] == "NONE"
        resp = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}",
            {
                "patch": {
                    "poi.x": 5e-6,
                    "defect.state": "REWORK",
                    "defect.description": "curtained face",
                },
                "version": doc["version"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied"] is True and body["saved"] is True
        assert lamella.poi.x == 5e-6
        assert lamella.defect.state.name == "REWORK"
        assert lamella.defect.updated_at is not None
        # New version serves on the next read; the event stream records it.
        reread = client.get(f"/app/items/{item}", headers=AUTH).json()
        assert reread["version"] == body["version"] != doc["version"]
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        edited = [e for e in events if e["kind"] == "config_edited"][-1]
        assert edited["payload"]["level"] == "item_fields"


def test_item_fields_allowlist_and_alignment_validity(ui, qapp):
    lamella = ui.experiment.positions[0]
    from fibsem.structures import FibsemRectangle

    lamella.alignment_area = FibsemRectangle(left=0.6, top=0.3, width=0.3, height=0.4)
    item = lamella.name
    with _client(ui) as client:
        doc = client.get(f"/app/items/{item}", headers=AUTH).json()
        off_list = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}",
            {"patch": {"path": "/tmp/evil"}, "version": doc["version"]},
        )
        assert off_list.status_code == 422
        assert "editable" in off_list.json()["detail"]["message"]

        # milling_angle is an outcome of Setup, not an input — not editable.
        angle = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}",
            {"patch": {"milling_angle": 17.0}, "version": doc["version"]},
        )
        assert angle.status_code == 422

        # A rectangle pushed out of the frame is reverted whole.
        bad = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}",
            {"patch": {"alignment_area.left": 0.9}, "version": doc["version"]},
        )
        assert bad.status_code == 422
        assert lamella.alignment_area.left == 0.6


def test_apply_protocol_recopies_configs_and_repoints_paths(ui, qapp):
    from fibsem.applications.autolamella.server.context import config_version

    buffer = EventBuffer()
    lamella = ui.experiment.positions[0]
    protocol_config = ui.experiment.task_protocol.task_config[TASK]
    # Diverge the two, as real runs do.
    lamella.task_config[TASK].milling["mill_rough"].stages[0].pattern.depth = 9e-6
    assert config_version(lamella.task_config[TASK]) != config_version(protocol_config)

    item = lamella.name
    with _client(ui, buffer=buffer) as client:
        resp = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}/apply_protocol",
            {"task_names": [TASK]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied"] is True and body["task_names"] == [TASK]
        # The item's copy now matches the protocol (fresh deepcopy, not shared)...
        copied = lamella.task_config[TASK]
        assert copied is not protocol_config
        # ...except the imaging paths, which point at the lamella's own dir.
        for mtc in copied.milling.values():
            assert str(mtc.acquisition.imaging.path) == str(lamella.path)
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        assert [e for e in events if e["payload"].get("level") == "protocol_applied"]

        unknown = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}/apply_protocol",
            {"task_names": ["Nope"]},
        )
        assert unknown.status_code == 404
        assert TASK in unknown.json()["detail"]["task_names"]


def test_a_poi_patch_moves_the_attached_patterns_with_it(ui, qapp):
    """The GUI's POI move syncs sync_to_poi patterns; the patch path must too
    (found live: a patched POI left rough patterns detached)."""
    from fibsem.structures import Point

    lamella = ui.experiment.positions[0]
    lamella.poi = Point(x=1e-6, y=2e-6)
    stage = lamella.task_config[TASK].milling["mill_rough"].stages[0]
    pattern_before = Point(x=stage.pattern.point.x, y=stage.pattern.point.y)
    assert lamella.task_config[TASK].sync_to_poi is True

    item = lamella.name
    with _client(ui) as client:
        doc = client.get(f"/app/items/{item}", headers=AUTH).json()
        resp = _post_on_worker(
            qapp,
            client,
            f"/app/items/{item}",
            {"patch": {"poi.x": 6e-6, "poi.y": 2e-6}, "version": doc["version"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert TASK in body["synced_tasks"]
        # The pattern followed the POI: the first stage anchors ON the point
        # (inter-stage offsets preserved) — the same semantics as the GUI's
        # move, via the same sync_tasks_to_poi call.
        assert (pattern_before.x, pattern_before.y) != (6e-6, 2e-6)
        assert stage.pattern.point.x == pytest.approx(6e-6)
        assert stage.pattern.point.y == pytest.approx(2e-6)


def test_stage_reorder_is_same_set_versioned_and_recorded(ui, qapp):
    buffer = EventBuffer()
    lamella = ui.experiment.positions[0]
    stages = lamella.task_config[TASK].milling["mill_rough"].stages
    names = [s.name for s in stages]
    assert len(names) >= 2
    new_order = list(reversed(names))

    item = lamella.name
    base = f"/app/items/{item}/task_config/{TASK}"
    with _client(ui, buffer=buffer) as client:
        doc = client.get(base, headers=AUTH).json()
        resp = _post_on_worker(
            qapp,
            client,
            f"{base}/stages/reorder",
            {
                "milling_key": "mill_rough",
                "order": new_order,
                "version": doc["version"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied"] is True and body["order"] == new_order
        assert [
            s.name for s in lamella.task_config[TASK].milling["mill_rough"].stages
        ] == new_order
        assert body["version"] != doc["version"]
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        assert [e for e in events if (e["payload"].get("reorder") or {}).get("order")]

        # Same-set enforcement: dropping a stage is refused.
        fresh = client.get(base, headers=AUTH).json()
        bad = _post_on_worker(
            qapp,
            client,
            f"{base}/stages/reorder",
            {
                "milling_key": "mill_rough",
                "order": new_order[:-1],
                "version": fresh["version"],
            },
        )
        assert bad.status_code == 422
        # Stale version refused.
        stale = _post_on_worker(
            qapp,
            client,
            f"{base}/stages/reorder",
            {"milling_key": "mill_rough", "order": names, "version": doc["version"]},
        )
        assert stale.status_code == 409
        # Unknown milling key refused with the known ones.
        wrong = _post_on_worker(
            qapp,
            client,
            f"{base}/stages/reorder",
            {"milling_key": "nope", "order": new_order, "version": fresh["version"]},
        )
        assert wrong.status_code == 422
