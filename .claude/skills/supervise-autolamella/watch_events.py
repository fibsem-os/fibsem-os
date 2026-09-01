#!/usr/bin/env python3
"""The supervision watch: long-poll the agent server's events, print one line
per thing worth waking for.

Run as a BACKGROUND monitor (each printed line wakes the supervising agent);
run in the foreground it would block you from acting on what it prints —
the watcher must never be the actor.

Stdlib only, and Python because the instrument PCs run Windows: a shell
watcher needs Git Bash there, this file runs anywhere fibsem does.

    python .claude/skills/supervise-autolamella/watch_events.py

Exits 0 when the workflow ends, 1 when the server cannot be watched (start
AutoLamella with the agent feature enabled, then rerun). The first fresh
agent to write its own watcher matched the wrong JSON key and slept through
a question — hence this file: run the tested one.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path

DISCOVERY = Path.home() / ".fibsem" / "agent-server.json"
# Kinds that end the watch: the run is over, report and stop.
TERMINAL = {"workflow_completed", "workflow_cancelled"}
# Kinds worth a wake-up line of their own (prompts are handled separately).
INTERESTING = TERMINAL | {
    "workflow_started",
    "task_started",
    "task_completed",
    "task_failed",
    "task_cancelled",
    "task_skipped",
}


def _connection():
    info = json.loads(DISCOVERY.read_text())
    return info["url"], {"Authorization": "Bearer " + info["token"]}


def _get(url, headers, timeout):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _print_events(events, since):
    """Print wake-up lines; returns (new_since, run_ended)."""
    run_ended = False
    for event in events:
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
        elif kind in TERMINAL:
            print("RUN ENDED (%s)" % kind, flush=True)
            run_ended = True
        elif kind in INTERESTING:
            print("EVENT %s %s" % (kind, data.get("item_name", "")), flush=True)
    return since, run_ended


def main():
    # Item names can carry operator-typed text; never let an exotic character
    # kill the watch on a cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

    try:
        base, headers = _connection()
    except (OSError, KeyError, ValueError):
        print(
            "NO SERVER: no discovery file — start AutoLamella with the agent "
            "feature enabled and a microscope connected, then rerun."
        )
        return 1

    # Seed the cursor at the tail: the buffer outlives workflow runs, so
    # starting at 0 would replay history — stale PROMPT lines, and a previous
    # run's workflow_completed would end this watch before it began.
    try:
        snapshot = _get(f"{base}/app/events?since=0&timeout=0", headers, timeout=10)
    except urllib.error.HTTPError as error:
        print(
            "SERVER REFUSED: HTTP %s from /app/events — check the token in "
            "%s and that an application is hosted." % (error.code, DISCOVERY)
        )
        return 1
    except (OSError, HTTPException, ValueError):
        print("SERVER GONE: events unreachable — is the app running?")
        return 1
    if not snapshot.get("available"):
        print(
            "NO EVENTS: the server has no event stream wired — nothing to "
            "watch. Is an application hosted (not just a bench server)?"
        )
        return 1
    since = snapshot.get("latest_seq") or 0

    failures = 0
    while True:
        try:
            payload = _get(
                f"{base}/app/events?since={since}&timeout=25", headers, timeout=35
            )
            failures = 0
        except urllib.error.HTTPError as error:
            # An HTTP status is the server talking, not the server gone. The
            # common cause is an app restart rotating the token: re-read the
            # discovery file and retry with fresh credentials.
            failures += 1
            if failures >= 4:
                print(
                    "SERVER REFUSED: HTTP %s from /app/events — the token or "
                    "hosting changed; reconnect from the discovery file." % error.code
                )
                return 1
            try:
                base, headers = _connection()
            except (OSError, KeyError, ValueError):
                pass
            time.sleep(5)
            continue
        except (OSError, HTTPException, ValueError):
            # Connection-level failure, a response cut off mid-read, or a
            # truncated body: retry, then report the server gone.
            failures += 1
            if failures >= 4:
                print("SERVER GONE: events unreachable — the app has stopped.")
                return 1
            time.sleep(5)
            continue

        if not payload.get("available"):
            print("NO EVENTS: the server's event stream went away — stopping.")
            return 1

        oldest = payload.get("oldest_available")
        if oldest is not None and oldest > since + 1:
            # Eviction ate events between polls; continuity is broken.
            print(
                "GAP: events were evicted before they could be read — "
                "re-read /app/prompt and /app/status rather than assuming "
                "nothing happened.",
                flush=True,
            )

        since, run_ended = _print_events(payload.get("events", []), since)
        since = max(since, payload.get("latest_seq") or 0)
        if run_ended:
            return 0


if __name__ == "__main__":
    sys.exit(main())
