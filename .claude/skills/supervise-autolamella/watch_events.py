#!/usr/bin/env python3
"""The supervision watch: long-poll the agent server's events, print one line
per thing worth waking for.

Run as a BACKGROUND monitor (each printed line wakes the supervising agent);
run in the foreground it would block you from acting on what it prints —
the watcher must never be the actor.

Stdlib only, and Python because the instrument PCs run Windows: a shell
watcher needs Git Bash there, this file runs anywhere fibsem does.

    python .claude/skills/supervise-autolamella/watch_events.py

Exits 0 when the workflow ends, 1 when the server cannot be reached (start
AutoLamella with the agent feature enabled, then rerun). The first fresh
agent to write its own watcher matched the wrong JSON key and slept through
a question — hence this file: run the tested one.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DISCOVERY = Path.home() / ".fibsem" / "agent-server.json"
INTERESTING = {
    "workflow_started",
    "task_started",
    "task_completed",
    "task_failed",
    "task_cancelled",
    "workflow_completed",
    "workflow_cancelled",
}


def _connection():
    info = json.loads(DISCOVERY.read_text())
    return info["url"], {"Authorization": "Bearer " + info["token"]}


def _get(url, headers, timeout):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def main():
    try:
        base, headers = _connection()
    except (OSError, KeyError, ValueError):
        print(
            "NO SERVER: no discovery file — start AutoLamella with the agent "
            "feature enabled and a microscope connected, then rerun."
        )
        return 1

    since = 0
    failures = 0
    while True:
        try:
            payload = _get(
                f"{base}/app/events?since={since}&timeout=25", headers, timeout=35
            )
            failures = 0
        except (urllib.error.URLError, OSError, TimeoutError):
            failures += 1
            if failures >= 4:
                print("SERVER GONE: events unreachable — the app has stopped.")
                return 1
            time.sleep(5)
            continue

        for event in payload.get("events", []):
            kind = event.get("kind", "")
            data = event.get("payload", {})
            since = max(since, event.get("seq", since))
            if kind == "prompt_raised":
                print(
                    "PROMPT nonce=%s type=%s" % (data.get("nonce"), data.get("type")),
                    flush=True,
                )
            elif kind == "prompt_answered":
                print(
                    "ANSWERED nonce=%s by=%s response=%s"
                    % (
                        data.get("nonce"),
                        data.get("answered_by"),
                        data.get("response"),
                    ),
                    flush=True,
                )
            elif kind == "prompt_cancelled":
                print("PROMPT WITHDRAWN nonce=%s" % data.get("nonce"), flush=True)
            elif kind in INTERESTING:
                print("EVENT %s %s" % (kind, data.get("item_name", "")), flush=True)
            if kind in ("workflow_completed", "workflow_cancelled"):
                print("RUN ENDED", flush=True)
                return 0


if __name__ == "__main__":
    sys.exit(main())
