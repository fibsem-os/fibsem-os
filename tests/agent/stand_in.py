"""A stand-in for ``fibsem-agent``, run by the manager's tests: it keeps its
session's record the way the harness does and ends as ``STAND_IN_PLAN`` says,
one entry per run: an exit code, ``crash`` (exits without recording), ``fatal``
(1, not resumable), ``wait`` (until its stop file, then 0) or ``ignore``
(never stops by itself)."""

import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from fibsem.ai.agent import manager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id")
    parser.add_argument("--resume")
    parser.add_argument("--goal")
    args, rest = parser.parse_known_args()
    session_id = args.resume or args.session_id or str(uuid.uuid4())
    root = Path(os.environ["STAND_IN_ROOT"])
    session = manager.SessionFolder.open(
        session_id, {"cwd": os.getcwd(), "goal": args.goal, "args": rest}, root
    )
    runs = len(session.record()["runs"])
    plan = os.environ.get("STAND_IN_PLAN", "wait").split(",")
    step = plan[min(runs, len(plan)) - 1]
    if step == "crash":
        os._exit(9)
    if step == "fatal":
        session.end(
            1, "the model can't answer (authentication_failed)", resumable=False
        )
        return 1
    if step == "ignore":
        time.sleep(600)
        return 0
    if step == "wait":
        stopped = threading.Event()
        manager.watch_for_stop(session, stopped.set, poll_s=0.05)
        stopped.wait(600)
        session.end(0, "stopped at fibsem-agent")
        return 0
    code = int(step)
    session.end(
        code,
        {0: "stopped", 1: "the app could not be watched", 2: "its budget is spent"}[
            code
        ],
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
