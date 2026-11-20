import json

import pytest

from fibsem.ai.evals.__main__ import main
from fibsem.ai.evals.core import episode


def _events(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _experiment(tmp_path, sessions=("s1",)):
    exp = tmp_path / "runs" / "grid-01"
    (exp / "Lam-01").mkdir(parents=True)
    (exp / "Lam-01" / "ref_ib.tif").write_bytes(b"tif")
    rows = [
        {"actor": "operator", "kind": "agent_level_set", "timestamp": 1, "payload": {}},
    ]
    for i, sid in enumerate(sessions):
        rows.append(
            {
                "actor": "agent",
                "kind": "agent_session_started",
                "timestamp": 2 + i,
                "payload": {"session_id": sid},
            }
        )
    rows.append(
        {
            "actor": "app",
            "kind": "image_acquired",
            "timestamp": 10,
            "payload": {
                "path": "C:\\Users\\op\\experiments\\grid-01\\Lam-01\\ref_ib.tif"
            },
        }
    )
    _events(exp / "events.jsonl", rows)
    return exp


def _session(root, sid, folder_name=None):
    folder = root / (folder_name or f"2026-11-20-101500-{sid}")
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(
        json.dumps({"session_id": sid, "fibsem": "0.7.0", "playbook_hash": "abc"})
    )
    (folder / "transcript.jsonl").write_text("{}\n")
    return folder


def test_pack_copies_experiment_and_named_sessions(tmp_path):
    exp = _experiment(tmp_path, sessions=("s1", "s2"))
    store = tmp_path / "sessions"
    _session(store, "s1")
    _session(store, "other")

    ep = episode.pack(exp, tmp_path / "ep", [store], note="sim run")

    assert (ep.experiment / "Lam-01" / "ref_ib.tif").read_bytes() == b"tif"
    assert (ep.session("s1") / "transcript.jsonl").is_file()
    assert ep.session("s2") is None
    assert ep.session("other") is None
    data = episode.Episode.load(ep.path).data
    assert data["note"] == "sim run"
    by_id = {s["id"]: s for s in data["sessions"]}
    assert by_id["s1"]["playbook_hash"] == "abc"
    assert by_id["s2"]["missing"] is True


def test_pack_finds_a_session_folder_given_directly(tmp_path):
    exp = _experiment(tmp_path)
    folder = _session(tmp_path, "s1", folder_name="session")
    ep = episode.pack(exp, tmp_path / "ep", [folder])
    assert ep.session("s1") is not None


def test_pack_refuses_a_non_experiment_and_an_existing_out(tmp_path):
    with pytest.raises(episode.EpisodeError):
        episode.pack(tmp_path, tmp_path / "ep")
    exp = _experiment(tmp_path)
    (tmp_path / "ep").mkdir()
    with pytest.raises(episode.EpisodeError):
        episode.pack(exp, tmp_path / "ep")


def test_events_in_time_order(tmp_path):
    ep = episode.pack(_experiment(tmp_path), tmp_path / "ep")
    times = [e["timestamp"] for e in ep.events()]
    assert times == sorted(times)


def test_local_finds_recorded_paths_in_the_copy(tmp_path):
    ep = episode.pack(_experiment(tmp_path), tmp_path / "ep")
    image = ep.experiment / "Lam-01" / "ref_ib.tif"
    assert ep.local("C:\\Users\\op\\experiments\\grid-01\\Lam-01\\ref_ib.tif") == image
    assert ep.local("/Users/op/experiments/grid-01/Lam-01/ref_ib.tif") == image
    assert ep.local("Lam-01/ref_ib.tif") == image
    assert ep.local("/Users/op/elsewhere/x.tif") is None
    assert ep.local("Lam-01/missing.tif") is None
    assert ep.local(None) is None


def test_link_shares_files(tmp_path):
    exp = _experiment(tmp_path)
    ep = episode.pack(exp, tmp_path / "ep", link=True)
    copied = ep.experiment / "Lam-01" / "ref_ib.tif"
    assert copied.stat().st_ino == (exp / "Lam-01" / "ref_ib.tif").stat().st_ino


def test_cli_pack(tmp_path, capsys):
    exp = _experiment(tmp_path)
    store = tmp_path / "sessions"
    _session(store, "s1")
    code = main(
        [
            "episode",
            "pack",
            str(exp),
            "--out",
            str(tmp_path / "ep"),
            "--sessions",
            str(store),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "session s1: kept" in out
    assert main(["episode", "pack", str(exp), "--out", str(tmp_path / "ep")]) == 2
