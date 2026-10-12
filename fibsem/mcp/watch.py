"""``fibsem-mcp watch``: long-poll a running app's events, one line per wake-up.

MCP is request-response: the App Server cannot wake an idle agent, and a
looping ``get_events`` tool blocks the agent's turn while it waits. So an
agent runs this as a background process its harness monitors, and each line
it prints wakes the agent. It must never be the actor: run in the foreground,
it would block the agent from acting on what it prints.

    fibsem-mcp watch
    python -m fibsem.mcp.sidecar watch   # where fibsem-mcp is not on PATH

The lines: ``PROMPT``, ``ANSWERED``, ``PROMPT WITHDRAWN``, ``QUESTION``,
``QUESTION RELEASED``, task ``EVENT`` lines, ``GAP`` and ``RUN ENDED``.
Exits 0 when the run ends, 1 when the server cannot be watched.

It finds the server the way the sidecar does (``--url``/``--token``, the
environment, the discovery file) and follows a restarted server the same way
(FIB-1269). Standard library only: watching needs nothing but HTTP, so it
works wherever fibsem is installed, without the [mcp] extra.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException

# Kinds that end the watch: the run is over, report and stop.
TERMINAL = {"workflow_completed", "workflow_cancelled"}
# Kinds worth a wake-up line of their own (prompts and questions have theirs).
INTERESTING = TERMINAL | {
    "workflow_started",
    "task_started",
    "task_completed",
    "task_failed",
    "task_cancelled",
    "task_skipped",
}

POLL_S = 25.0  # the server caps a park at 30 s
FAILURES_BEFORE_GIVING_UP = 4
RETRY_DELAY_S = 5.0

GAP_LINE = (
    "GAP: events were lost before they could be read -- re-read /app/prompt "
    "and /app/status rather than assuming nothing happened."
)


class _Response:
    """The few httpx.Response members the sidecar's helpers read."""

    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)


class _UrllibClient:
    """An httpx-style client over urllib, so the sidecar's reconnecting client
    can wrap it. An HTTP status comes back as a response, as httpx's does;
    only a failure to talk to the server raises."""

    def __init__(self, base_url, token):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    def request(self, method, path, json=None, timeout=10.0):
        data = None
        headers = dict(self._headers)
        if json is not None:
            data = _json_bytes(json)
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _Response(response.status, response.read())
        except urllib.error.HTTPError as error:
            return _Response(error.code, error.read())

    def close(self):
        pass


def _json_bytes(payload):
    return json.dumps(payload).encode()


def _poll(client, since, timeout):
    """One long-poll: (payload, None), or (None, why it failed)."""
    try:
        response = client.request(
            "GET",
            f"/app/events?since={since}&timeout={timeout}",
            timeout=timeout + 10.0,
        )
    except (OSError, HTTPException, ValueError) as error:
        return None, f"unreachable ({error})"
    if response.status_code >= 400:
        return None, f"HTTP {response.status_code}"
    try:
        return response.json(), None
    except ValueError:
        return None, "a reply that is not JSON"


def _print_events(events, since, out):
    """Print wake-up lines; returns (new_since, run_ended)."""
    run_ended = False
    for event in events:
        kind = event.get("kind", "")
        data = event.get("payload") or {}
        since = max(since, event.get("seq", since))
        if kind == "question_asked":
            out(
                f"QUESTION item={data.get('item_name')} task={data.get('task_name')} "
                "-- the run is stopped on a recorded question: find it in "
                "get_pending_reviews (holding_the_run) and decide it"
            )
        elif kind == "question_released":
            out(
                f"QUESTION RELEASED item={data.get('item_name')} "
                f"task={data.get('task_name')}"
            )
        elif kind == "prompt_raised":
            out(f"PROMPT question={data.get('question_id')} type={data.get('type')}")
        elif kind == "prompt_answered":
            out(
                f"ANSWERED question={data.get('question_id')} by={data.get('answered_by')} "
                f"response={data.get('response')}"
            )
        elif kind == "prompt_cancelled":
            out(f"PROMPT WITHDRAWN question={data.get('question_id')}")
        elif kind in TERMINAL:
            out(f"RUN ENDED ({kind})")
            run_ended = True
        elif kind in INTERESTING:
            out(f"EVENT {kind} {data.get('item_name', '')}")
    return since, run_ended


def watch(client, out=None, sleep_fn=None, poll_s=POLL_S):
    """Watch until the run ends (0) or the server cannot be watched (1).

    ``client`` is httpx-style (``request(method, path, timeout=...)``), in
    practice the sidecar's reconnecting client: a 401 or a lost connection
    re-reads the discovery file once per request, so a restarted app is
    followed without this loop knowing.
    """
    out = out or _print_line
    sleep_fn = sleep_fn or time.sleep

    # Seed the cursor at the tail: the buffer outlives runs, so starting at 0
    # would replay history -- stale PROMPT lines, and a previous run's
    # workflow_completed would end this watch before it began.
    snapshot, failure = _poll(client, since=0, timeout=0.0)
    if failure is not None:
        out(f"SERVER UNAVAILABLE: /app/events gave {failure}. Is the app running?")
        return 1
    if not snapshot.get("available"):
        out(
            "NO EVENTS: the server has no event stream, so there is nothing to "
            "watch. Is an application hosted, not only a bench server?"
        )
        return 1
    since = snapshot.get("latest_seq") or 0

    failures = 0
    while True:
        payload, failure = _poll(client, since=since, timeout=poll_s)
        if failure is not None:
            # A refusal is usually a restart that has not rewritten the
            # discovery file yet: the next request re-reads it.
            failures += 1
            if failures >= FAILURES_BEFORE_GIVING_UP:
                out(f"SERVER GONE: /app/events gave {failure}. The app has stopped.")
                return 1
            sleep_fn(RETRY_DELAY_S)
            continue
        failures = 0

        if not payload.get("available"):
            out("NO EVENTS: the server's event stream went away. Stopping.")
            return 1

        latest = payload.get("latest_seq") or 0
        if latest < since:
            # A new stream (the app restarted with a fresh buffer): its seqs
            # start again, and this cursor would wait past all of them. All
            # of it is news, since it began after this watch did.
            out(GAP_LINE)
            since = 0
            continue
        oldest = payload.get("oldest_available")
        if oldest is not None and oldest > since + 1:
            # Eviction ate events between polls; continuity is broken.
            out(GAP_LINE)

        since, run_ended = _print_events(payload.get("events", []), since, out)
        since = max(since, latest)
        if run_ended:
            return 0


def _print_line(line):
    print(line, flush=True)


def connect(url=None, token=None, client_factory=_UrllibClient):
    """The sidecar's reconnecting client over ``client_factory``, or None when
    no server is known."""
    from fibsem.mcp.sidecar import _ReconnectingClient, _try_resolve

    resolved = _try_resolve(url, token)
    if resolved is None:
        return None
    return _ReconnectingClient(
        client_factory, url, token, client_factory(*resolved), resolved
    )


def main(argv=None):
    """``fibsem-mcp watch``; returns the exit code."""
    from fibsem.mcp.sidecar import _NO_SERVER_MSG

    parser = argparse.ArgumentParser(
        prog="fibsem-mcp watch",
        description="Print one line per app event worth waking an agent for.",
    )
    parser.add_argument("--url", default=None, help="Server URL")
    parser.add_argument("--token", default=None, help="Bearer token")
    args = parser.parse_args(argv)

    # Item names can carry operator-typed text; never let an exotic character
    # kill the watch on a cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

    client = connect(args.url, args.token)
    if client is None:
        _print_line(f"NO SERVER: {_NO_SERVER_MSG}")
        return 1
    try:
        return watch(client)
    except KeyboardInterrupt:
        return 1
