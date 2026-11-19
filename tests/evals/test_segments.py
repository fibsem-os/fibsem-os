import json

import pytest

from fibsem.ai.evals.__main__ import main
from fibsem.ai.evals.core import catalogue, episode, segments


def _episode(tmp_path):
    exp = tmp_path / "grid-01"
    (exp / "Lam-01").mkdir(parents=True)
    (exp / "Lam-01" / "setup_ib.tif").write_bytes(b"a")
    (exp / "Lam-01" / "later_ib.tif").write_bytes(b"b")
    (exp / "Lam-01" / "unnamed_ib.tif").write_bytes(b"c")
    rows = [
        {
            "id": "s-1",
            "timestamp": 1,
            "kind": "image_acquired",
            "actor": "agent",
            "payload": {"path": "/Users/op/grid-01/Lam-01/setup_ib.tif"},
        },
        {
            "id": "s-2",
            "timestamp": 2,
            "kind": "action",
            "actor": "agent",
            "payload": {},
        },
        {
            "id": "s-3",
            "timestamp": 3,
            "kind": "image_acquired",
            "actor": "agent",
            "payload": {"path": "/Users/op/grid-01/Lam-01/later_ib.tif"},
        },
        {
            "id": "s-4",
            "timestamp": 4,
            "kind": "edit",
            "actor": "operator",
            "payload": {},
        },
    ]
    (exp / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return episode.pack(exp, tmp_path / "ep")


def _segment(**changes):
    segment = {
        "id": "seg-01",
        "task": "setup_placement",
        "item": "Lam-01",
        "start_event": "s-2",
        "end_event": "s-4",
        "inputs": {"fib_image": "Lam-01/setup_ib.tif"},
        "decision": "lamella centred on the target",
        "status": "proposed",
    }
    segment.update(changes)
    return segment


def _data(*segs):
    return {"format": 0, "segments": list(segs)}


def test_catalogue_lists_tasks_cut_from_runs():
    names = {e["task"] for e in catalogue.entries()}
    assert {"setup_placement", "polishing_stop"} <= names
    assert not any(n.startswith("live_") for n in names)
    for entry in catalogue.entries():
        assert entry["decision"] and entry["inputs"]


def test_a_good_segment_passes(tmp_path):
    result = segments.check(_episode(tmp_path), _data(_segment()))
    assert result.ok, result.problems


def test_an_input_saved_after_the_cut_is_refused(tmp_path):
    seg = _segment(inputs={"fib_image": "Lam-01/later_ib.tif"})
    result = segments.check(_episode(tmp_path), _data(seg))
    assert any("saved after" in p for p in result.problems)


def test_an_input_no_event_names_is_a_note(tmp_path):
    seg = _segment(inputs={"fib_image": "Lam-01/unnamed_ib.tif"})
    result = segments.check(_episode(tmp_path), _data(seg))
    assert result.ok
    assert any("not on the record" in n for n in result.notes)


def test_bad_segments(tmp_path):
    ep = _episode(tmp_path)
    cases = [
        (_segment(task="nope"), "not in the catalogue"),
        (_segment(task="live_polishing"), "not in the catalogue"),
        (_segment(start_event="x-9"), "not an event"),
        (_segment(start_event="s-4", end_event="s-2"), "before start_event"),
        (_segment(inputs={}), "no input 'fib_image'"),
        (_segment(inputs={"fib_image": "Lam-01/x.tif"}), "not in the episode"),
        (_segment(decision=""), "no decision"),
        (_segment(status="maybe"), "status"),
        (_segment(task=None, new_task={"name": "x"}), "no name or question"),
    ]
    for seg, words in cases:
        result = segments.check(ep, _data(seg))
        assert any(words in p for p in result.problems), (words, result.problems)
    twice = segments.check(ep, _data(_segment(), _segment()))
    assert any("used twice" in p for p in twice.problems)


def test_a_new_task_segment(tmp_path):
    seg = _segment(
        task=None,
        new_task={
            "name": "milling_current",
            "question": "Which current?",
            "answer_shape": "choice",
        },
        inputs={"fib_image": "Lam-01/setup_ib.tif"},
    )
    result = segments.check(_episode(tmp_path), _data(seg))
    assert result.ok, result.problems


def test_cli_check(tmp_path, capsys):
    ep = _episode(tmp_path)
    segments.save(ep, _data(_segment()))
    assert main(["episode", "check", str(ep.path)]) == 0
    segments.save(ep, _data(_segment(start_event="nope")))
    assert main(["episode", "check", str(ep.path)]) == 1
    assert "PROBLEM" in capsys.readouterr().out


def test_cli_catalogue(capsys):
    assert main(["catalogue"]) == 0
    assert "setup_placement" in capsys.readouterr().out


def test_segmenter_prepare(tmp_path):
    import yaml

    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    folder = segmenter.prepare(ep)
    digest = (folder / segmenter.EVENTS_DIGEST).read_text().splitlines()
    assert len(digest) == 4
    assert digest[0].startswith("s-1\t")
    assert "setup_ib.tif" in digest[0]
    cat = yaml.safe_load((folder / segmenter.CATALOGUE).read_text())
    assert any(t["task"] == "setup_placement" for t in cat["tasks"])
    assert "episode: ep" in (folder / segmenter.PROMPT).read_text()


def test_segmenter_may_only_write_segments(tmp_path):
    import asyncio

    sdk = pytest.importorskip("claude_agent_sdk")
    PermissionResultAllow, PermissionResultDeny = (
        sdk.PermissionResultAllow,
        sdk.PermissionResultDeny,
    )

    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    check = segmenter._may_use(ep)
    ok = asyncio.run(
        check("Write", {"file_path": str(ep.path / "segments.yaml")}, None)
    )
    assert isinstance(ok, PermissionResultAllow)
    for tool, data in (
        ("Write", {"file_path": str(ep.experiment / "events.jsonl")}),
        ("Bash", {"command": "ls"}),
    ):
        assert isinstance(asyncio.run(check(tool, data, None)), PermissionResultDeny)


def test_cli_segment_dry_run(tmp_path):
    ep = _episode(tmp_path)
    assert main(["episode", "segment", str(ep.path), "--dry-run"]) == 0
    assert (ep.path / "segmenter" / "prompt.md").is_file()


class _FakeClient:
    """Stands in for ClaudeSDKClient: each turn runs *act* and ends with
    the given result."""

    replies = []
    asked = []

    def __init__(self, options):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, message):
        _FakeClient.asked.append(message)

    async def receive_response(self):
        act, result = _FakeClient.replies.pop(0)
        if act:
            act()
        yield result


def _result(sdk, is_error=False, text="done", errors=None):
    return sdk.ResultMessage(
        subtype="error" if is_error else "success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="x",
        total_cost_usd=0.5,
        result=text,
        errors=errors,
    )


def test_segmenter_reports_a_failed_turn(tmp_path, monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    monkeypatch.setattr(sdk, "ClaudeSDKClient", _FakeClient)
    _FakeClient.asked = []
    _FakeClient.replies = [
        (None, _result(sdk, True, "Failed to authenticate", ["OAuth expired"]))
    ]
    with pytest.raises(segmenter.SegmenterError, match="OAuth expired"):
        segmenter.run(ep)
    assert len(_FakeClient.asked) == 1


def test_segmenter_with_no_file_written_fails(tmp_path, monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    monkeypatch.setattr(sdk, "ClaudeSDKClient", _FakeClient)
    _FakeClient.asked = []
    _FakeClient.replies = [(None, _result(sdk, text="I could not"))]
    with pytest.raises(segmenter.SegmenterError, match="I could not"):
        segmenter.run(ep)
    assert len(_FakeClient.asked) == 1


def test_segmenter_gets_one_turn_to_fix_problems(tmp_path, monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    monkeypatch.setattr(sdk, "ClaudeSDKClient", _FakeClient)
    _FakeClient.asked = []
    _FakeClient.replies = [
        (lambda: segments.save(ep, _data(_segment(start_event="nope"))), _result(sdk)),
        (lambda: segments.save(ep, _data(_segment())), _result(sdk, text="fixed")),
    ]
    result = segmenter.run(ep)
    assert result.ok
    assert len(_FakeClient.asked) == 2 and "nope" in _FakeClient.asked[1]
    info = segments.load(ep)["segmenter"]
    assert info["spent_usd"] == 0.5 and info["said"] == "fixed"
