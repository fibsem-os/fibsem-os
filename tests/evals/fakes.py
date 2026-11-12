"""Stand-ins for the App Server and fibsem-agent, for the eval runner's tests.

``server --create EXP ...`` makes EXP/experiment.yaml, writes the discovery
file and waits to be ended. ``agent --session-id ID --files WORK ...`` writes
the answer in ``EVAL_FAKE_ANSWER`` (JSON) to WORK/answer.json and stops; with
none, it waits for its session's stop file.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def server(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", type=Path)
    parser.add_argument("--experiments", type=Path)
    parser.add_argument("--manufacturer")
    parser.add_argument("--port", type=int)
    parser.add_argument("--discovery", type=Path)
    args = parser.parse_args(argv)
    args.create.mkdir(parents=True)
    (args.create / "experiment.yaml").write_text("app: fake\n")
    found = args.discovery or Path.home() / ".fibsem" / "app-server.json"
    found.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port or 1}"
    found.write_text(json.dumps({"url": url, "connect_token": "x", "pid": os.getpid()}))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.2)


def agent(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id")
    parser.add_argument("--files", type=Path)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--playbook")
    parser.add_argument("--model")
    parser.add_argument("--goal")
    parser.add_argument("--max-budget-usd")
    args, _ = parser.parse_known_args(argv)
    code = args.code is not None
    args.files = args.code if code else args.files
    sessions = Path.home() / ".fibsem" / "agent" / "sessions"
    folder = sessions / f"2026-01-01-000000-{args.session_id}"
    folder.mkdir(parents=True)
    task = (args.files / "task.md").read_text()
    record = {
        "session_id": args.session_id,
        "goal": args.goal,
        "task_chars": len(task),
        "code": code,
        "playbook": args.playbook,
        "model": args.model,
        "server": json.loads(Path(os.environ["FIBSEM_DISCOVERY_FILE"]).read_text())[
            "url"
        ],
    }
    answer = os.environ.get("EVAL_FAKE_ANSWER")
    if answer is not None:
        (args.files / "answer.json").write_text(answer)
    else:
        while not (folder / "stop").exists():
            time.sleep(0.2)
    record["runs"] = [{"spent_usd": 0.25, "turns": 3, "reason": "stopped"}]
    (folder / "session.json").write_text(json.dumps(record))


if __name__ == "__main__":
    {"server": server, "agent": agent}[sys.argv[1]](sys.argv[2:])
