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
    assert "(ep)" in (folder / segmenter.GOAL).read_text()


def test_segmenter_runs_on_its_playbook(tmp_path):
    from fibsem.ai.evals.core import segmenter

    text = segmenter.instructions()
    assert "instrument: none" in text and "segments.yaml" in text
    assert "episode: ep" not in text
    assert "(ep)" in segmenter.goal(_episode(tmp_path))


def test_cli_segment_dry_run(tmp_path):
    ep = _episode(tmp_path)
    assert main(["episode", "segment", str(ep.path), "--dry-run"]) == 0
    assert (ep.path / "segmenter" / "goal.md").is_file()


# Stand-ins for fibsem-agent-host and fibsem-agent: the host answers /health
# and keeps the operator's message; the agent writes segments.yaml as
# FAKE_SEGMENTER says and prints the harness's "Turn ended" line.
FAKE_HOST = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
folder = sys.argv[sys.argv.index("--folder") + 1]
os.makedirs(folder, exist_ok=True)
token = os.environ["FIBSEM_HOST_OPERATOR_TOKEN"]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def do_POST(self):
        assert self.headers["Authorization"] == "Bearer " + token
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        open(os.path.join(folder, "message.txt"), "w").write(body["text"])
        self.send_response(201); self.end_headers(); self.wfile.write(b"{}")
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""

FAKE_AGENT = """
import json, os, sys, time
args = sys.argv[1:]
episode = args[args.index("--files") + 1]
mode = os.environ["FAKE_SEGMENTER"]
assert "FIBSEM_HOST_OPERATOR_TOKEN" not in os.environ
assert args[args.index("--playbook") + 1] == "eval-segmenter"
def write(start):
    seg = {"id": "seg-01", "task": "setup_placement", "item": "Lam-01",
           "start_event": start, "end_event": "s-4",
           "inputs": {"fib_image": "Lam-01/setup_ib.tif"},
           "decision": "centred", "status": "proposed"}
    data = {"format": 0, "segments": [seg]}
    open(os.path.join(episode, "segments.yaml"), "w").write(json.dumps(data))
    print("Turn ended: success, 3 steps, $0.10", flush=True)
if mode == "fail":
    print("Turn ended: error. The model can't sign in: OAuth expired", flush=True)
    sys.exit(1)
if mode == "tamper":
    with open(os.path.join(episode, "episode.yaml"), "a") as f:
        f.write("# changed")
write("nope" if mode == "fix" else "s-2")
if mode == "fix":
    message = os.path.join(episode, "segmenter", "host", "message.txt")
    while not os.path.exists(message):
        time.sleep(0.1)
    assert "nope" in open(message).read()
    write("s-2")
"""


def _fakes(tmp_path, monkeypatch, mode):
    import sys

    host, agent = tmp_path / "fake_host.py", tmp_path / "fake_agent.py"
    host.write_text(FAKE_HOST)
    agent.write_text(FAKE_AGENT)
    monkeypatch.setenv("FAKE_SEGMENTER", mode)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return {"agent": [sys.executable, str(agent)], "host": [sys.executable, str(host)]}


def test_segmenter_reports_a_failed_agent(tmp_path, monkeypatch):
    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    fakes = _fakes(tmp_path, monkeypatch, "fail")
    with pytest.raises(segmenter.SegmenterError, match="OAuth expired"):
        segmenter.run(ep, timeout_s=30, **fakes)


def test_segmenter_gets_one_message_to_fix_problems(tmp_path, monkeypatch):
    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    fakes = _fakes(tmp_path, monkeypatch, "fix")
    result = segmenter.run(ep, timeout_s=30, **fakes)
    assert result.ok, result.problems
    info = segments.load(ep)["segmenter"]
    assert info["fix_turn"] is True and info["playbook"] == "eval-segmenter"
    assert info["harness"] == "fibsem-agent"


def test_segmenter_with_good_segments_needs_no_message(tmp_path, monkeypatch):
    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    fakes = _fakes(tmp_path, monkeypatch, "good")
    assert segmenter.run(ep, timeout_s=30, **fakes).ok
    assert segments.load(ep)["segmenter"]["fix_turn"] is False
    assert not (ep.path / "segmenter" / "host" / "message.txt").exists()


def test_segmenter_may_not_change_the_episode(tmp_path, monkeypatch):
    from fibsem.ai.evals.core import segmenter

    ep = _episode(tmp_path)
    fakes = _fakes(tmp_path, monkeypatch, "tamper")
    with pytest.raises(segmenter.SegmenterError, match="episode.yaml"):
        segmenter.run(ep, timeout_s=30, **fakes)
