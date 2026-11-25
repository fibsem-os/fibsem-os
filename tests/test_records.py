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
    assert _status([run]) == ("ended", "stopped")
    ended = {"about": {}, "ended": "ended by the operator"}
    assert _status([run], ended) == ("ended", "ended by the operator")


def test_a_parked_agent_waits():
    # The app was stopped: its run stopped, and its manager hasn't ended it.
    run = {"ended": "t", "exit_code": manager.STOPPED, "reason": "stopped"}
    assert _status([run], {"about": {}}) == ("waiting", "stopped")


def test_a_start_waiting_for_its_app():
    status = records.session_status({}, {"about": {}}, lambda run: False)
    assert status[0] == "waiting"


def test_a_run_that_cannot_resume_has_ended():
    run = {"ended": "t", "exit_code": manager.GONE, "resumable": False}
    assert _status([run], {"about": {}})[0] == "ended"


def test_a_reused_pid_is_not_the_run(monkeypatch):
    seen = []
    monkeypatch.setattr(
        records, "same_process", lambda pid, started: seen.append((pid, started))
    )
    records._alive({"pid": 7, "pid_started": 12.5})
    assert seen == [(7, 12.5)]


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
                # each run writes the session's running total
                {"pid": 1, "started": "s1", "ended": "e1", "spent_usd": 1.5},
                {"pid": 2, "started": "s2", "ended": "e2", "spent_usd": 3.75},
            ],
        },
        managed={"about": {"level_now": "drive"}, "ended": "stopped"},
        link={"session_id": "a", "path": "/x/exp", "name": "exp", "id": "e-1"},
    )
    _session(root, "2026-11-21-090000-b", {"session_id": "b", "runs": [{"pid": 3}]})

    rows = records.list_sessions(root=root, alive=lambda run: True)

    assert [row.session_id for row in rows] == ["b", "a"]
    b, a = rows
    assert b.status == "running" and b.experiment is None and not b.by_gateway
    assert a.status == "ended" and a.ended_at == "e2" and a.started_at == "s1"
    assert a.playbook == "autolamella" and a.level == "drive"
    assert a.spent_usd == 3.75 and a.by_gateway and a.last_active_at
    assert a.experiment == records.ExperimentLink("/x/exp", "exp", "e-1")
    assert len(records.list_sessions(limit=1, root=root)) == 1


def test_a_folder_without_a_record_is_not_a_session(tmp_path):
    root = tmp_path / "sessions"
    (root / "2026-11-21-090000-x").mkdir(parents=True)
    assert records.list_sessions(root=root) == []


def test_a_start_waiting_for_its_app_is_listed(tmp_path):
    root = tmp_path / "sessions"
    folder = root / "2026-11-21-090000-0f-1e"
    folder.mkdir(parents=True)
    (folder / manager.MANAGED).write_text(json.dumps({"about": {"level": "watch"}}))
    [row] = records.list_sessions(root=root)
    assert (row.session_id, row.status, row.level) == ("0f-1e", "waiting", "watch")


def test_the_limit_counts_sessions_only(tmp_path):
    root = tmp_path / "sessions"
    _session(root, "2026-11-20-090000-a", {"session_id": "a", "runs": [{"pid": 1}]})
    _session(root, "2026-11-21-090000-x", {"runs": []})
    rows = records.list_sessions(limit=1, root=root, alive=lambda run: True)
    assert [row.session_id for row in rows] == ["a"]


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

    assert [row.name for row in rows] == ["Worked on", "Recent one", "gone"]
    assert rows[0].id == "w-1" and rows[0].session_ids == ["a", "b"]
    assert rows[1].session_ids == [] and rows[1].updated_at.startswith("1970-01-")
    # linked but no longer on disk: known by its link
    assert not rows[2].on_this_pc and rows[2].session_ids == ["c"]


def test_an_experiment_is_one_row_by_its_id(tmp_path):
    here = _experiment(tmp_path / "moved", "Moved", "m-1")
    sessions = [
        records.SessionRow(
            "a",
            "f",
            "ended",
            experiment=records.ExperimentLink("/old/place", "Moved", "m-1"),
        ),
        records.SessionRow(
            "b",
            "f",
            "ended",
            experiment=records.ExperimentLink(str(here), "Moved", "m-1"),
        ),
    ]
    rows = records.list_experiments(sessions, recent=lambda: [])
    assert [(row.path, row.on_this_pc) for row in rows] == [(str(here), True)]
    assert rows[0].session_ids == ["a", "b"]


def test_a_linked_experiment_not_on_this_pc_is_kept(tmp_path):
    on_hub = records.ExperimentLink("/microscope-pc/exp", "Far", "far-1")
    here = _experiment(tmp_path / "here", "Here", "h-1")
    sessions = [
        records.SessionRow("a", "f", "ended", experiment=on_hub),
        records.SessionRow("b", "f", "ended", experiment=on_hub),
    ]
    rows = records.list_experiments(sessions, recent=lambda: [str(here)])
    assert [(row.name, row.on_this_pc) for row in rows] == [
        ("Here", True),
        ("Far", False),
    ]
    assert rows[1].id == "far-1" and rows[1].session_ids == ["a", "b"]


def test_an_unreadable_experiment_is_listed_by_its_folder(tmp_path):
    folder = tmp_path / "Broken"
    folder.mkdir()
    (folder / "experiment.yaml").write_text("name: [unclosed")
    rows = records.list_experiments([], recent=lambda: [str(folder)])
    assert [(row.name, row.id) for row in rows] == [("Broken", None)]
