"""The local index: made from the folders, and made again from them."""

import json
import os

import yaml

from fibsem.index import Index


def experiment(root, folder, **saved):
    path = root / folder / "experiment.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"name": folder, "positions": [], **saved}))
    return path


def session(root, folder, **record):
    path = root / folder / "session.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return path


def test_the_index_lists_experiments_and_the_sessions_that_worked_on_them(tmp_path):
    data, sessions = tmp_path / "data", tmp_path / "sessions"
    experiment(
        data,
        "grid-1",
        id="exp-1",
        created_at="2026-11-12T09:00:00+11:00",
        positions=[{}, {}],
        session={"system": {"serial_number": "SN-1"}, "user": {"name": "sam"}},
    )
    experiment(
        data, "nested/grid-2", id="exp-2", created_at="2026-11-13T09:00:00+11:00"
    )
    session(
        sessions,
        "s1",
        session_id="s1",
        experiment_id="exp-1",
        instrument_id="i-1",
        goal="Mill grid 1",
        runs=[{"started": "2026-11-12T10:00:00+11:00"}],
    )
    session(sessions, "s2", session_id="s2", experiment_id="exp-2")

    index = Index(tmp_path / "state" / "index.sqlite")
    assert index.refresh([data], [sessions]) == {"experiments": 2, "sessions": 2}

    newest, oldest = index.experiments()
    assert (newest["id"], oldest["id"]) == ("exp-2", "exp-1")
    assert oldest["num_items"] == 2 and oldest["operator"] == "sam"
    assert [row["id"] for row in index.experiments(instrument_serial="SN-1")] == [
        "exp-1"
    ]
    assert [row["id"] for row in index.experiments(text="grid-2")] == ["exp-2"]
    (found,) = index.sessions(experiment_id="exp-1")
    assert found["goal"] == "Mill grid 1" and found["instrument_id"] == "i-1"
    assert index.sessions(instrument_id="nobody") == []


def test_a_refresh_reads_what_changed_and_forgets_what_is_gone(tmp_path):
    data, other = tmp_path / "data", tmp_path / "other"
    kept = experiment(data, "a", id="a")
    gone = experiment(data, "b", id="b")
    experiment(other, "c", id="c")
    index = Index(tmp_path / "index.sqlite")
    index.refresh([data, other])

    kept.write_text(yaml.safe_dump({"name": "renamed", "id": "a", "positions": []}))
    os.utime(kept, (1, 1))
    gone.unlink()
    # Only *data* is read again: what the other folder gave is kept.
    assert index.refresh([data])["experiments"] == 2
    assert {row["name"] for row in index.experiments()} == {"renamed", "c"}


def test_the_index_is_made_again_from_the_folders(tmp_path):
    data = tmp_path / "data"
    experiment(data, "a", id="a")
    index = Index(tmp_path / "index.sqlite")
    index.refresh([data])
    index.path.unlink()
    assert index.experiments() == []
    assert index.rebuild([data]) == {"experiments": 1, "sessions": 0}


def test_an_unreadable_file_is_listed_as_unavailable_and_stops_nothing(tmp_path):
    data, sessions = tmp_path / "data", tmp_path / "sessions"
    bad = data / "bad" / "experiment.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not: [yaml")
    experiment(data, "good", id="good")
    session(sessions, "s1")
    (sessions / "s1" / "session.json").write_text("not json")
    index = Index(tmp_path / "index.sqlite")
    assert index.refresh([data, tmp_path / "missing"], [sessions]) == {
        "experiments": 2,
        "sessions": 1,
    }
    assert {row["name"]: row["available"] for row in index.experiments()} == {
        "bad": 0,
        "good": 1,
    }
    assert index.sessions()[0]["session_id"] is None


def test_an_experiment_folder_is_not_searched_for_more_experiments(tmp_path):
    data = tmp_path / "data"
    experiment(data, "a", id="a")
    experiment(data, "a/copy", id="copy")
    assert Index(tmp_path / "index.sqlite").refresh([data])["experiments"] == 1
