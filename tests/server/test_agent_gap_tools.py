"""The agent's tools for what only the app's windows did before (2026-10-16):
stopping one task, grid records, the workflow's and the grid protocol's tasks,
run options, points on saved images, the microscope state, device queries,
and the milling and alignment checks.

The Demo is an Arctis here, with an experiment open on the default protocol.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.ai.agent.catalog.describe import describe_action  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from tests.run_state import stub_session  # noqa: E402
from tests.server.test_agent_requests import RecordingContext  # noqa: E402
from tests.server.test_grid_tools import AUTH, TOKEN, _call, arctis  # noqa: E402, F401


@pytest.fixture
def driving(arctis, tmp_path):  # noqa: F811
    """Drive, with an experiment on the default protocol open over the Arctis."""
    from fibsem.applications.autolamella import experiment_setup
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskProtocol,
        Experiment,
    )

    host = stub_session()
    host.microscope = arctis
    (tmp_path / "gaps-exp").mkdir()
    host.experiment = Experiment(path=tmp_path, name="gaps-exp")
    host.experiment.task_protocol = AutoLamellaTaskProtocol.load(
        experiment_setup.default_protocol_path()
    )
    auth = AuthConfig.generate(level="drive", token=TOKEN)
    context = RecordingContext(host)
    app = build_server(arctis, app_context=context, auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        client.host = host
        client.context = context
        yield client


def _ok(answer):
    assert answer.status_code == 200, answer.text
    return answer.json()


# --- grid records ---------------------------------------------------------------


def test_the_agent_judges_notes_and_removes_a_grid(driving):
    _ok(_call(driving, "grid_add"))
    experiment = driving.host.experiment

    judged = _ok(
        _call(
            driving,
            "grid_set_quality",
            grid_name="G1",
            verdict="rework",
            reason="cracked",
        )
    )
    assert judged["quality"]["verdict"] == "REWORK"
    quality = experiment.get_grid_by_name("G1").quality
    assert quality.state.name == "REWORK" and quality.description == "cracked"

    _ok(_call(driving, "grid_set_note", grid_name="G1", note="thin ice in the centre"))
    assert experiment.get_grid_by_name("G1").description == "thin ice in the centre"

    wrong = _call(driving, "grid_set_quality", grid_name="G1", verdict="SUPERB")
    assert wrong.status_code == 422
    missing = _call(driving, "grid_set_note", grid_name="G9", note="x")
    assert missing.status_code == 404

    from tests.run_state import run_working_on

    with run_working_on(driving.host.runner, "G2"):
        busy = _call(driving, "grid_remove", grid_name="G2")
    assert busy.status_code == 409
    with run_working_on(driving.host.runner, "G1"):
        removed = _ok(_call(driving, "grid_remove", grid_name="G2"))
    assert removed["items_without_grid"] == []
    assert experiment.get_grid_by_name("G2") is None


def test_the_agent_s_grid_note_is_its_note_on_the_record(driving):
    from fibsem.acting import AGENT
    from fibsem.workflows.actions import ACTION, NOTE

    _ok(_call(driving, "grid_add"))
    grid = driving.host.experiment.get_grid_by_name("G1")
    acted = []

    def record(kind, payload):
        acted.append(kind)

    driving.host.microscope.session.record_signal.connect(record)
    try:
        _ok(_call(driving, "grid_set_note", grid_name="G1", note="thin ice"))
        noted = [(p, actor) for k, p, actor in driving.context.recorded if k == NOTE]
        assert noted == [
            (
                {"text": "thin ice", "item_id": grid.id, "item_name": "G1"},
                AGENT,
            )
        ]
        # The note is the record of the call: no action besides it.
        assert ACTION not in acted

        # Clearing it is not a note; the call is recorded as an act.
        _ok(_call(driving, "grid_set_note", grid_name="G1", note=""))
    finally:
        driving.host.microscope.session.record_signal.disconnect(record)
    assert grid.description == ""
    assert len([k for k, *_ in driving.context.recorded if k == NOTE]) == 1
    assert acted.count(ACTION) == 1


# --- the workflow's tasks ---------------------------------------------------------


def _workflow(driving):
    return driving.host.experiment.task_protocol.workflow_config.workflow


def test_the_agent_adds_reorders_requires_and_removes_a_task(driving):
    added = _ok(
        _call(driving, "protocol_add_task", task_type="MILL_TRENCH", task_name="Trench")
    )
    assert added["task_names"][-1] == "Trench"
    lamella_config = driving.host.experiment.task_protocol.task_config["Trench"]
    assert lamella_config.task_name == "Trench"

    again = _call(
        driving, "protocol_add_task", task_type="MILL_TRENCH", task_name="Trench"
    )
    assert again.status_code == 422
    unknown = _call(driving, "protocol_add_task", task_type="BAKE", task_name="Bake")
    assert unknown.status_code == 404
    assert "MILL_TRENCH" in unknown.json()["detail"]["task_type_names"]

    order = ["Trench"] + [n for n in _workflow(driving) if n != "Trench"]
    _ok(_call(driving, "protocol_reorder_tasks", order=order))
    assert _workflow(driving)[0] == "Trench"

    # Mill Fiducial requires Setup, so it cannot go first.
    bad = ["Mill Fiducial"] + [n for n in _workflow(driving) if n != "Mill Fiducial"]
    assert _call(driving, "protocol_reorder_tasks", order=bad).status_code == 422
    assert _call(driving, "protocol_reorder_tasks", order=["Trench"]).status_code == 422

    # Requirements: earlier tasks only; optional is a flag.
    later = _call(
        driving,
        "protocol_set_task_requirements",
        task_name="Trench",
        requires=["Polishing"],
    )
    assert later.status_code == 422
    set_ = _ok(
        _call(
            driving,
            "protocol_set_task_requirements",
            task_name="Polishing",
            requires=["Trench", "Rough Milling"],
            required=False,
        )
    )
    assert set_["requires"] == ["Trench", "Rough Milling"] and set_["required"] is False

    _ok(_call(driving, "protocol_remove_task", task_name="Trench"))
    assert "Trench" not in _workflow(driving)
    assert "Trench" not in driving.host.experiment.task_protocol.task_config


def test_the_workflow_is_not_edited_while_it_runs(driving):
    driving.host.runner.is_running = True
    refused = _call(driving, "protocol_reorder_tasks", order=list(_workflow(driving)))
    assert refused.status_code == 409


def test_lamella_defaults_change_only_what_is_given(driving):
    protocol = driving.host.experiment.task_protocol
    answer = _ok(
        _call(
            driving,
            "protocol_set_lamella_defaults",
            name_prefix="Lyso",
            poi={"x": 0.1, "y": -0.2},
        )
    )
    assert answer["lamella_defaults"]["name_prefix"] == "Lyso"
    assert protocol.lamella_defaults.poi.x == pytest.approx(0.1)
    _ok(_call(driving, "protocol_set_lamella_defaults", poi={}))
    assert protocol.lamella_defaults.poi is None
    assert protocol.lamella_defaults.name_prefix == "Lyso"
    bad = driving.post("/app/experiment/protocol/lamella_defaults", json={"colour": 1})
    assert bad.status_code == 422
    outside = _call(
        driving,
        "protocol_set_lamella_defaults",
        alignment_area={"left": 0.8, "top": 0.0, "width": 0.5, "height": 0.5},
    )
    assert outside.status_code == 422
    assert "inside the image" in outside.text
    assert protocol.lamella_defaults.alignment_area is None


def test_the_protocol_is_saved_to_the_protocol_folder(driving, tmp_path, monkeypatch):
    from fibsem.applications.autolamella import experiment_setup

    folder = tmp_path / "protocols"
    monkeypatch.setattr(experiment_setup, "protocol_directory", lambda: str(folder))
    saved = _ok(_call(driving, "protocol_save", name="lyso-trench"))
    assert saved["name"] == "lyso-trench.yaml"
    assert os.path.isfile(folder / "lyso-trench.yaml")
    exists = _call(driving, "protocol_save", name="lyso-trench")
    assert exists.status_code == 422
    _ok(_call(driving, "protocol_save", name="lyso-trench", overwrite=True))
    assert _call(driving, "protocol_save", name="../out").status_code == 422


# --- the grid protocol ------------------------------------------------------------


def test_the_agent_edits_the_grid_tasks(driving):
    grid_tasks = driving.host.experiment.grid_protocol
    read = _ok(
        _call(driving, "protocol_get_grid_task_config", task_name="SEM Overview")
    )
    assert read["config"]["attention"] == "automated"

    patched = _ok(
        _call(
            driving,
            "protocol_update_grid_task_config",
            task_name="SEM Overview",
            patch={"attention": "supervised"},
            version=read["version"],
        )
    )
    assert patched["applied"] and patched["version"] != read["version"]
    assert grid_tasks.task_config["SEM Overview"].attention.value == "supervised"
    stale = _call(
        driving,
        "protocol_update_grid_task_config",
        task_name="SEM Overview",
        patch={"attention": "automated"},
        version=read["version"],
    )
    assert stale.status_code == 409

    _ok(
        _call(
            driving,
            "protocol_reorder_grid_tasks",
            order=["FIB Overview", "SEM Overview"],
        )
    )
    assert grid_tasks.ordered_task_names == ["FIB Overview", "SEM Overview"]

    _ok(
        _call(
            driving,
            "protocol_set_grid_task_requirements",
            task_name="FIB Overview",
            requires=["SEM Overview"],
        )
    )
    assert grid_tasks.task_config["FIB Overview"].requires == ["SEM Overview"]
    itself = _call(
        driving,
        "protocol_set_grid_task_requirements",
        task_name="FIB Overview",
        requires=["FIB Overview"],
    )
    assert itself.status_code == 422


# --- run control and options ------------------------------------------------------


def test_run_options_are_set_and_saved(driving):
    options = driving.host.experiment.task_protocol.options
    _ok(_call(driving, "workflow_set_options", turn_beams_off=True))
    assert options.turn_beams_off is True
    assert driving.post("/app/workflow/options", json={}).status_code == 422
    assert (
        driving.post("/app/workflow/options", json={"turn_beams_off": 1}).status_code
        == 422
    )


class _Queue:
    def __init__(self, active):
        self.active = active


class _Manager:
    RUN_KIND = "items"
    RUN_DESCRIPTION = "a lamella run"

    def __init__(self, active):
        self.queue = _Queue(active)

    def stop_task(self):
        pass


class _Item:
    item_name = "01-live-raptor"
    task_name = "Rough Milling"


def test_stop_task_stops_only_the_task_named(driving):

    nothing = _ok(
        _call(
            driving,
            "workflow_stop_task",
            item_name="01-live-raptor",
            task_name="Rough Milling",
            reason="the trench is off target",
        )
    )
    assert nothing["stopped"] is False and driving.host.runner.stops == []

    driving.host.runner.manager = _Manager(_Item())
    other = _ok(
        _call(
            driving,
            "workflow_stop_task",
            item_name="01-live-raptor",
            task_name="Polishing",
            reason="too thin",
        )
    )
    assert other["stopped"] is False
    assert other["running"] == {
        "item_name": "01-live-raptor",
        "task_name": "Rough Milling",
    }
    assert driving.host.runner.stops == []

    stopped = _ok(
        _call(
            driving,
            "workflow_stop_task",
            item_name="01-live-raptor",
            task_name="Rough Milling",
            reason="the trench is off target",
        )
    )
    assert stopped["stopped"] is True and driving.host.runner.stops == ["task"]

    no_reason = driving.post(
        "/app/workflow/stop_task",
        json={"item_name": "01-live-raptor", "task_name": "Rough Milling"},
    )
    assert no_reason.status_code == 422


def test_grid_tasks_join_only_a_running_grid_run(driving):
    answer = _ok(
        _call(
            driving,
            "workflow_queue_grids",
            task_names=["SEM Overview"],
            grid_names=["G1"],
        )
    )
    assert answer["queued"] is False
    driving.host.runner.manager = _Manager(None)
    lamella_run = _ok(
        _call(
            driving,
            "workflow_queue_grids",
            task_names=["SEM Overview"],
            grid_names=["G1"],
        )
    )
    assert "lamella run" in lamella_run["reason"]


def test_grid_tasks_are_queued_after_what_waits(driving):
    from fibsem.workflows.grid.manager import (
        GridTaskManager,
    )
    from fibsem.workflows.queue import TaskQueue

    _ok(_call(driving, "grid_add"))
    manager = GridTaskManager.__new__(GridTaskManager)
    manager.queue = TaskQueue()
    manager.notify_queue_changed = lambda: None
    manager.experiment = driving.host.experiment
    manager.microscope = driving.host.microscope
    manager.queue.add("G1", "Load Grid")
    driving.host.runner.manager = manager

    answer = _ok(
        _call(
            driving,
            "workflow_queue_grids",
            task_names=["SEM Overview"],
            grid_names=["G1", "G2"],
        )
    )
    added = [(a["grid_name"], a["step"]) for a in answer["added"]]
    assert ("G1", "SEM Overview") in added and ("G2", "SEM Overview") in added
    unknown = _ok(
        _call(driving, "workflow_queue_grids", task_names=["Bake"], grid_names=["G1"])
    )
    assert unknown["queued"] is False and "SEM Overview" in unknown["task_names"]


# --- points on saved images ---------------------------------------------------------

STAGE_POSITION = "/app/experiment/images/stage_position"


def _saved_image(driving, arctis, name="sem.tif"):  # noqa: F811
    from fibsem.structures import BeamType, ImageSettings

    image = arctis.acquire_image(ImageSettings(beam_type=BeamType.ELECTRON, hfw=150e-6))
    path = os.path.join(str(driving.host.experiment.path), name)
    image.save(path)
    return path


def test_a_point_on_any_saved_image_is_a_stage_position_and_back(driving, arctis):  # noqa: F811
    path = _saved_image(driving, arctis)
    centre = _ok(
        driving.get(STAGE_POSITION, params={"path": path, "fx": 0.5, "fy": 0.5})
    )
    position = centre["stage_position"]

    pixels = _ok(
        _call(
            driving,
            "experiment_get_image_pixels",
            path=path,
            stage_positions=[position],
        )
    )
    (point,) = pixels["points"]
    assert point["inside"] is True
    assert point["px"][0] == pytest.approx(pixels["image"]["width"] / 2, abs=1)
    assert point["px"][1] == pytest.approx(pixels["image"]["height"] / 2, abs=1)

    elsewhere = driving.get(
        STAGE_POSITION, params={"path": "/etc/hosts", "fx": 0.5, "fy": 0.5}
    )
    assert elsewhere.status_code == 403
    both = driving.get(STAGE_POSITION, params={"path": path})
    assert both.status_code == 422


# --- the microscope ----------------------------------------------------------------


def test_the_microscope_state_goes_back_to_a_saved_one(driving, arctis):  # noqa: F811
    saved = _ok(driving.get("/microscope/state"))["microscope_state"]
    _ok(_call(driving, "stage_move_relative", dx=20e-6))
    _ok(_call(driving, "microscope_set_state", microscope_state=saved))
    now = arctis.get_stage_position()
    assert now.x == pytest.approx(saved["stage_position"]["x"], abs=1e-9)


def test_a_device_query_answers(driving):
    answer = _ok(
        _call(
            driving,
            "device_call_query",
            device="stage",
            query="pose_table",
            kwargs={
                "rotation_reference": 0.0,
                "shuttle_pre_tilt": 0.0,
                "fib_column_tilt": 0.907,
            },
        )
    )
    assert "SEM" in answer["result"]


def test_milling_stages_are_checked_before_drawing(driving):
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.patterning.patterns.rectangle import RectanglePattern

    inside = FibsemMillingStage(
        name="inside", pattern=RectanglePattern(width=10e-6, height=5e-6)
    )
    wide = FibsemMillingStage(
        name="wide", pattern=RectanglePattern(width=500e-6, height=5e-6)
    )
    answer = _ok(
        _call(
            driving, "milling_check_stages", stages=[inside.to_dict(), wide.to_dict()]
        )
    )
    by_name = {stage["name"]: stage for stage in answer["result"]}
    assert by_name["inside"]["outside_field"] == []
    assert by_name["inside"]["estimated_seconds"] > 0
    assert len(by_name["wide"]["outside_field"]) == 1

    options = _ok(_call(driving, "milling_list_options"))["result"]
    assert "Rectangle" in options["patterns"] and options["strategies"]


def test_alignment_methods_are_compared_on_saved_images(driving, arctis):  # noqa: F811
    first = _saved_image(driving, arctis, "a.tif")
    second = _saved_image(driving, arctis, "b.tif")
    answer = _ok(
        _call(
            driving,
            "alignment_compare_methods",
            reference_path=first,
            image_path=second,
        )
    )
    result = answer["result"]
    assert set(result) >= {"shifts_px", "max_disagreement_px", "agreement"}


# --- what the operator reads ---------------------------------------------------------


@pytest.mark.parametrize(
    "tool, arguments, change",
    [
        (
            "workflow_stop_task",
            {"item_name": "01", "task_name": "Polishing", "reason": "x"},
            "Stop this task; the run carries on",
        ),
        (
            "grid_set_quality",
            {"grid_name": "G1", "verdict": "REWORK", "reason": "cracked"},
            "Mark it rework: cracked",
        ),
        ("grid_remove", {"grid_name": "G1"}, "Stop tracking it"),
        (
            "protocol_add_task",
            {"task_type": "MILL_TRENCH", "task_name": "Trench"},
            "Add a MILL_TRENCH task named 'Trench'",
        ),
        (
            "workflow_set_options",
            {"turn_beams_off": True},
            "Turn the beams off when a run ends",
        ),
        (
            "workflow_start_grids",
            {"task_names": ["SEM Overview"], "unload_at_end": True},
            ", then unload the last grid",
        ),
        ("microscope_set_state", {"microscope_state": {}}, "the stage moves"),
    ],
)
def test_the_operator_reads_each_new_action_plainly(tool, arguments, change):
    line = describe_action(tool, arguments, "")
    assert change in line["change"]
