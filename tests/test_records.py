import json
import os

import yaml

from fibsem import records
from fibsem.ai.agent import manager


def _session(root, name, record, managed=None, link=None):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / manager.RECORD).write_text(json.dumps(record))
    if managed is not None:
        (folder / manager.MANAGED).write_text(json.dumps(managed))
    if link is not None:
        (folder / "agent").mkdir()
        (folder / "agent" / "experiment.json").write_text(json.dumps(link))
    return folder


def _experiment(folder, name, experiment_id=None):
    folder.mkdir(parents=True)
    saved = {"name": name}
    if experiment_id:
        saved["_id"] = experiment_id
    (folder / "experiment.yaml").write_text(yaml.safe_dump(saved))
    return folder


def _status(runs, managed=None, alive=False):
    return records.session_status({"runs": runs}, managed or {}, lambda run: alive)


def test_status_of_a_run_still_going():
    assert _status([{"pid": 1}], alive=True) == ("running", None)


def test_status_of_a_run_that_died_without_its_end():
    assert _status([{"pid": 1}]) == ("crashed", None)


def test_status_of_a_session_left_by_its_gateway():
    assert _status([{"pid": 1}], {"left": 100.0})[0] == "left"


def test_the_managers_end_is_the_last_word():
    ended = {"ended": "did not finish (the gateway closed)"}
    assert _status([{"pid": 1}], ended) == ("ended", ended["ended"])
    run = {"ended": "t", "exit_code": manager.GONE, "reason": "the app went"}
    assert _status([run], ended) == ("ended", ended["ended"])


def test_a_managed_session_whose_app_went_waits():
    run = {"ended": "t", "exit_code": manager.GONE, "reason": "the app went"}
    assert _status([run], {"about": {}}) == ("waiting", "the app went")
    # by hand nothing resumes it
    assert _status([run]) == ("ended", "the app went")


def test_status_of_a_run_that_ended():
    run = {"ended": "t", "exit_code": manager.STOPPED, "reason": "stopped"}
    assert _status([run], {"about": {}}) == ("ended", "stopped")


def test_sessions_newest_first_with_their_experiment(tmp_path):
    root = tmp_path / "sessions"
    _session(
        root,
        "2026-11-20-090000-a",
        {
            "session_id": "a",
            "playbook": {"name": "autolamella", "hash": "h"},
            "model": "m",
            "runs": [
                {"pid": 1, "started": "s1", "ended": "e1", "spent_usd": 1.5},
                {"pid": 2, "started": "s2", "ended": "e2", "spent_usd": 2.25},
            ],
        },
        managed={"about": {"level_now": "drive"}},
        link={"session_id": "a", "path": "/x/exp", "name": "exp", "id": "e-1"},
    )
    _session(root, "2026-11-21-090000-b", {"session_id": "b", "runs": [{"pid": 3}]})

    rows = records.list_sessions(root=root, alive=lambda run: True)

    assert [row.session_id for row in rows] == ["b", "a"]
    b, a = rows
    assert b.status == "running" and b.experiment is None and not b.by_gateway
    assert a.status == "ended" and a.ended_at == "e2" and a.started_at == "s1"
    assert a.playbook == "autolamella" and a.level == "drive"
    assert a.spent_usd == 3.75 and a.by_gateway
    assert a.experiment == records.ExperimentLink("/x/exp", "exp", "e-1")
    assert len(records.list_sessions(limit=1, root=root)) == 1


def test_a_folder_without_a_record_is_not_a_session(tmp_path):
    root = tmp_path / "sessions"
    (root / "2026-11-21-090000-x").mkdir(parents=True)
    assert records.list_sessions(root=root) == []


def test_experiments_recent_and_worked_on(tmp_path):
    recent = _experiment(tmp_path / "recent", "Recent one", "r-1")
    worked = _experiment(tmp_path / "worked", "Worked on", "w-1")
    os.utime(recent / "experiment.yaml", (1_000_000, 1_000_000))
    (worked / "events.jsonl").write_text("{}\n")
    gone = str(tmp_path / "gone")
    sessions = [
        records.SessionRow(
            "a", "f", "ended", experiment=records.ExperimentLink(str(worked))
        ),
        records.SessionRow(
            "b", "f", "ended", experiment=records.ExperimentLink(str(worked))
        ),
        records.SessionRow("c", "f", "ended", experiment=records.ExperimentLink(gone)),
    ]

    rows = records.list_experiments(
        sessions, recent=lambda: [str(recent), str(worked), gone]
    )

    assert [row.name for row in rows] == ["Worked on", "Recent one"]
    assert rows[0].id == "w-1" and rows[0].session_ids == ["a", "b"]
    assert rows[1].session_ids == [] and rows[1].updated_at.startswith("1970-01-")


def test_an_unreadable_experiment_is_listed_by_its_folder(tmp_path):
    folder = tmp_path / "Broken"
    folder.mkdir()
    (folder / "experiment.yaml").write_text("name: [unclosed")
    rows = records.list_experiments([], recent=lambda: [str(folder)])
    assert [(row.name, row.id) for row in rows] == [("Broken", None)]
