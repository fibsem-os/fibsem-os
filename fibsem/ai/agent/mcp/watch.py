"""``fibsem-mcp watch``: long-poll a running app's events, one line per wake-up.

MCP is request-response: the App Server cannot wake an idle agent, and a
looping ``app_get_events`` tool blocks the agent's turn while it waits. So an
agent runs this as a background process its harness monitors, and each line
it prints wakes the agent. It must never be the actor: run in the foreground,
it would block the agent from acting on what it prints.

    fibsem-mcp watch
    fibsem-mcp watch --since 42          # resume after event 42
    python -m fibsem.ai.agent.mcp.sidecar watch   # where fibsem-mcp is not on PATH

The lines: ``PROMPT``, ``ANSWERED``, ``PROMPT WITHDRAWN``, ``QUESTION``,
``QUESTION RELEASED``, ``REQUEST`` (the agent's request decided, finished or
cancelled, FIB-1283), ``COMMAND ENDED``, ``MILL STAGE STARTED`` and
``MILL ENDED`` (every mill, whoever started it), task ``EVENT`` lines,
``WORKFLOW STARTED``, ``WORKFLOW ENDED``, ``MESSAGE`` (the operator wrote to
the agent from its card, FIB-1423), ``GAP``, ``RESTARTED`` and
``AGENT OFF``/``AGENT BACK``.

It follows the whole session, not one workflow: the agent starts and stops
workflows itself, and between them it still has to hear the app (FIB-1431). A
workflow's end is a line, not the end of the watch, and so is one that stops
without an end event: ``WORKFLOW ENDED (stopped without an end event)``, once
``/app/workflow/status`` says nothing is running. It exits only when the
server cannot be watched (1), or when it is stopped. While the operator has the
agent level at Off the app refuses the agent, but it is still running: the
watch says so once and waits for the level to come back (FIB-1429).

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

# Kinds that end a workflow: report it, and keep watching. A stalled workflow
# drained with decisions pending; a failed one died outside a task (FIB-1037).
WORKFLOW_ENDS = {
    "workflow_completed",
    "workflow_cancelled",
    "workflow_stalled",
    "workflow_failed",
}
# Kinds worth a wake-up line of their own (prompts and questions have theirs).
INTERESTING = {
    "task_started",
    "task_completed",
    "task_failed",
    "task_cancelled",
    "task_skipped",
}

ENDED = "WORKFLOW ENDED"
NO_END_EVENT_LINE = f"{ENDED} (stopped without an end event)"

POLL_S = 25.0  # the server caps a park at 30 s
FAILURES_BEFORE_GIVING_UP = 4
RETRY_DELAY_S = 5.0

GAP_LINE = (
    "GAP: events were lost before they could be read -- re-read /app/workflow/prompt "
    "and /app/workflow/status rather than assuming nothing happened."
)
# A new event stream: its seqs start again. Nothing was lost, and everything in
# it is news (FIB-1421).
RESTARTED_LINE = (
    "RESTARTED: the app restarted (new session); following its events from the start."
)
# The operator set the level to Off: the app is running but refuses the agent
# (403 agent_off), so the watch waits for the level to come back (FIB-1429).
AGENT_OFF = "agent_off"
AGENT_OFF_LINE = (
    "AGENT OFF: the operator set the agent level to Off; waiting for it to be raised."
)
AGENT_BACK_LINE = (
    "AGENT BACK: the operator raised the agent level; following events again."
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
        if _error_type(response) == AGENT_OFF:
            return None, AGENT_OFF
        return None, f"HTTP {response.status_code}"
    try:
        return response.json(), None
    except ValueError:
        return None, "a reply that is not JSON"


def _error_type(response):
    """A refusal's ``error_type`` (``agent_off``, ``level``), or None."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail.get("error_type") if isinstance(detail, dict) else None


def _print_events(events, since, out):
    """Print wake-up lines; returns (new_since, running).

    ``running`` is True when the last workflow event among them started one,
    False when it ended one, and None when there was neither."""
    running = None
    for event in events:
        kind = event.get("kind", "")
        data = event.get("payload") or {}
        since = max(since, event.get("seq", since))
        if kind == "question_asked":
            out(_question_line(data.get("item_name"), data.get("task_name")))
        elif kind == "question_released":
            out(
                f"QUESTION RELEASED item={data.get('item_name')} "
                f"task={data.get('task_name')}"
            )
        elif kind == "prompt_raised":
            out(f"PROMPT question={data.get('question_id')} type={data.get('type')}")
        elif kind == "prompt_answered":
            line = (
                f"ANSWERED question={data.get('question_id')} by={event.get('actor')} "
                f"response={data.get('response')}"
            )
            # The operator took the agent's suggestion (FIB-1283).
            if data.get("suggested_by"):
                line += f" suggested_by={data['suggested_by']}"
            out(line)
        elif kind == "prompt_cancelled":
            out(f"PROMPT WITHDRAWN question={data.get('question_id')}")
        elif kind == "prompt_suggestion_dismissed":
            # The operator declined the agent's suggestion (FIB-1424).
            line = f"SUGGESTION DISMISSED question={data.get('question_id')}"
            out(f"{line} reason={data['reason']}" if data.get("reason") else line)
        elif kind in ("request_decided", "request_finished", "request_cancelled"):
            out(_request_line(kind, data))
        elif kind == "command_ended":
            # A long command the agent started (a mill, a spot burn) ended.
            line = (
                f"COMMAND ENDED {data.get('service')}.{data.get('command')} "
                f"run={data.get('run_id')} state={data.get('state')}"
            )
            out(f"{line} ({data['error']})" if data.get("error") else line)
        elif kind == "milling_stage_started":
            out(_mill_stage_line(event, data))
        elif kind == "milling":
            out(_mill_ended_line(event, data))
        elif kind == "operator_message":
            message_id = data.get("id")
            text = " ".join(str(data.get("text", "")).split())
            out(
                f"MESSAGE id={message_id} from the operator: {text} -- answer with "
                f"experiment_add_note (reply_to={message_id}). It is not an "
                "approval: your level still decides what you may do"
            )
        elif kind == "workflow_started":
            out("WORKFLOW STARTED")
            running = True
        elif kind in WORKFLOW_ENDS:
            detail = data.get("error") or _counts(data.get("task_counts"))
            out(f"{ENDED} ({kind}: {detail})" if detail else f"{ENDED} ({kind})")
            running = False
        elif kind in INTERESTING:
            task = data.get("task_name")
            line = f"EVENT {kind} {data.get('item_name', '')}"
            line = f"{line} {task}" if task else line
            reason = _skip_reason(data) if kind == "task_skipped" else None
            out(f"{line} ({reason})" if reason else line)
    return since, running


def _item_of(event):
    """`` item=<name>`` for the lamella or grid the event happened on, or ''."""
    item = event.get("item")
    name = item.get("name") if isinstance(item, dict) else None
    return f" item={name}" if name else ""


def _mill_stage_line(event, data):
    """A mill's stage started, from ``milling_stage_started``: which of how
    many, with the current and depth it mills at, to check against the plan."""
    stage = data.get("stage") or {}
    index, total = data.get("stage_index"), data.get("total_stages")
    place = f"{index + 1}/{total}" if isinstance(index, int) else "?"
    line = (
        f"MILL STAGE STARTED {place} '{stage.get('name')}'{_item_of(event)} "
        f"task='{data.get('milling_task_name')}'"
    )
    current = (stage.get("milling") or {}).get("milling_current")
    if current is not None:
        line += f" current={_si(current)}A"
    depth = (stage.get("pattern") or {}).get("depth")
    if depth is not None:
        line += f" depth={_si(depth)}m"
    return f"{line} run={data.get('milling_run_id')}"


def _si(value):
    """A number in SI units, short (``2e-11``); anything else as it came."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _mill_ended_line(event, data):
    """A mill ended, from its ``milling`` event: its status, how each stage
    ended, and the error of a failed one."""
    status = data.get("status")
    if data.get("reason"):
        status = f"{status} ({data['reason']})"
    line = (
        f"MILL ENDED '{data.get('milling_task_name')}'{_item_of(event)} "
        f"status={status} run={data.get('milling_run_id')}"
    )
    stages = [
        f"{stage.get('name')}={stage.get('status')}"
        + (f" ({stage['reason']})" if stage.get("reason") else "")
        for stage in data.get("stages") or []
    ]
    if stages:
        line += " stages: " + ", ".join(stages)
    return f"{line} ({data['error']})" if data.get("error") else line


def _request_line(kind, data):
    """The agent's request moved on (FIB-1283): decided, finished, cancelled."""
    head = f"REQUEST {data.get('request_id')}"
    if kind == "request_cancelled":
        return f"{head} cancelled: {data.get('why')}"
    if kind == "request_finished":
        counts = f"{data.get('applied')} of {data.get('of')}"
        # An applied action stopped part way, a mill say (FIB-1398).
        if data.get("cancelled"):
            return f"{head} {data.get('status')} ({counts}; run cancelled)"
        return f"{head} {data.get('status')} ({counts} applied)"
    if data.get("outcome") == "rejected":
        reason = data.get("reason")
        return f"{head} rejected" + (f": {reason}" if reason else "")
    return f"{head} approved"


def _skip_reason(data):
    """``missing_prereqs: Mill Fiducial``: why a task was skipped, and for a
    missing requirement which one (FIB-1357)."""
    reason = data.get("skip_reason")
    missing = data.get("missing_prereqs")
    if reason and missing:
        return f"{reason}: {', '.join(str(name) for name in missing)}"
    return reason


def _counts(counts):
    """``nothing ran, 2 skipped`` or ``3 completed, 1 failed``: how a completed
    run's tasks ended, so a run that skipped everything does not read as done."""
    if not isinstance(counts, dict):
        return None
    ran = [
        f"{counts[status]} {status}"
        for status in ("completed", "failed")
        if counts.get(status)
    ]
    skipped = [f"{counts['skipped']} skipped"] if counts.get("skipped") else []
    return ", ".join((ran or ["nothing ran"]) + skipped)


def _question_line(item_name, task_name):
    return (
        f"QUESTION item={item_name} task={task_name} "
        "-- the run is stopped on a recorded question: find it in "
        "review_list (holding_the_run) and decide it"
    )


def _held_questions(client):
    """The questions a run is stopped on now, as (item_name, task_name), from
    /app/workflow/reviews; empty when it cannot say."""
    try:
        response = client.request("GET", "/app/workflow/reviews", timeout=10.0)
        if response.status_code >= 400:
            return []
        reviews = response.json().get("reviews") or []
    except (OSError, HTTPException, ValueError, AttributeError, TypeError):
        return []
    return [
        (r.get("item_name"), r.get("task_name"))
        for r in reviews
        if isinstance(r, dict) and r.get("holding_the_run")
    ]


def _run_is_live(client):
    """Whether /app/workflow/status says a run is going; None when it cannot say."""
    try:
        response = client.request("GET", "/app/workflow/status", timeout=10.0)
        if response.status_code >= 400:
            return None
        return bool(response.json()["workflow"]["running"])
    except (OSError, HTTPException, ValueError, KeyError, TypeError):
        return None


def watch(client, out=None, sleep_fn=None, poll_s=POLL_S, clock=None, since=None):
    """Watch until the server cannot be watched (1). Workflows starting and
    ending are lines along the way, not the end of the watch (FIB-1431).

    ``since`` resumes after that event seq, so a watch restarted after the
    last one returned misses nothing in between; None starts at the tail.

    ``client`` is httpx-style (``request(method, path, timeout=...)``), in
    practice the sidecar's reconnecting client: a 401 or a lost connection
    re-reads the discovery file once per request, so a restarted app is
    followed without this loop knowing.
    """
    out = out or _print_line
    sleep_fn = sleep_fn or time.sleep
    clock = clock or time.monotonic

    # Seed the cursor at the tail: the buffer outlives workflows, so starting at
    # 0 would replay history -- stale PROMPT lines, and a previous workflow's
    # end reported as news.
    snapshot, failure = _poll(client, since=0, timeout=0.0)
    if failure == AGENT_OFF:
        out(AGENT_OFF_LINE)
        while failure == AGENT_OFF:
            sleep_fn(RETRY_DELAY_S)
            snapshot, failure = _poll(client, since=0, timeout=0.0)
        if failure is None:
            out(AGENT_BACK_LINE)
    if failure is not None:
        out(f"SERVER UNAVAILABLE: /app/events gave {failure}. Is the app running?")
        return 1
    if not snapshot.get("available"):
        out("NO EVENTS: the server has no event stream, so there is nothing to watch.")
        return 1
    stream = snapshot.get("stream")
    # Started at the tail, a question asked before this watch began is behind
    # the cursor: said below, first, while a run is live.
    held_before = since is None
    if since is None:
        since = snapshot.get("latest_seq") or 0
    elif (snapshot.get("latest_seq") or 0) < since:
        # Resumed with a seq from before the app restarted.
        out(RESTARTED_LINE)
        since = 0
    # A workflow can end without an end event: one that dies outside a task
    # fires none (FIB-1331). So while one is live, /app/workflow/status is the
    # backstop.
    workflow_live = _run_is_live(client) is True
    if workflow_live and held_before:
        # An agent that reconnects to a held run hears what it is held on.
        for item_name, task_name in _held_questions(client):
            out(_question_line(item_name, task_name))
    checked_at = clock()

    failures = 0
    agent_off = False
    while True:
        payload, failure = _poll(client, since=since, timeout=poll_s)
        if failure == AGENT_OFF:
            # The app is up; its events wait in the buffer for the level to
            # come back, so the cursor holds and nothing is missed.
            if not agent_off:
                out(AGENT_OFF_LINE)
                agent_off = True
            failures = 0
            sleep_fn(RETRY_DELAY_S)
            continue
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
        if agent_off:
            out(AGENT_BACK_LINE)
            agent_off = False

        if not payload.get("available"):
            out("NO EVENTS: the server's event stream went away. Stopping.")
            return 1

        latest = payload.get("latest_seq") or 0
        restarted = payload.get("stream") != stream
        stream = payload.get("stream")
        if latest < since or restarted:
            # A new stream (the app restarted with a fresh buffer): its seqs
            # start again, and this cursor would wait past all of them. All
            # of it is news, since it began after this watch did. (A server
            # that names no stream is told by its seqs going back.)
            out(RESTARTED_LINE)
            since = 0
            continue
        oldest = payload.get("oldest_available")
        if oldest is not None and oldest > since + 1:
            # Eviction ate events between polls; continuity is broken.
            out(GAP_LINE)

        events = payload.get("events", [])
        since, running = _print_events(events, since, out)
        since = max(since, latest)
        if running is not None:
            workflow_live = running
            checked_at = clock()

        # Asked on a quiet poll, and at least once a poll on a busy one.
        if workflow_live and (not events or clock() - checked_at >= poll_s):
            checked_at = clock()
            if _run_is_live(client) is False:
                # An end event may have landed since the poll: report that
                # one rather than the fallback.
                last, failure = _poll(client, since=since, timeout=0.0)
                running = None
                if failure is None and last.get("available"):
                    since, running = _print_events(last.get("events", []), since, out)
                if running is None:
                    out(NO_END_EVENT_LINE)
                    running = False
                workflow_live = running


def _print_line(line):
    print(line, flush=True)


def connect(url=None, token=None, client_factory=_UrllibClient):
    """The sidecar's reconnecting client over ``client_factory``, or None when
    no server is known."""
    from fibsem.ai.agent.mcp.sidecar import _ReconnectingClient, _try_resolve

    resolved = _try_resolve(url, token)
    if resolved is None:
        return None
    return _ReconnectingClient(
        client_factory, url, token, client_factory(*resolved), resolved
    )


def main(argv=None):
    """``fibsem-mcp watch``; returns the exit code."""
    from fibsem.ai.agent.mcp.sidecar import _NO_SERVER_MSG

    parser = argparse.ArgumentParser(
        prog="fibsem-mcp watch",
        description="Print one line per app event worth waking an agent for.",
    )
    parser.add_argument("--url", default=None, help="Server URL")
    parser.add_argument("--token", default=None, help="Bearer token")
    parser.add_argument(
        "--since",
        type=int,
        default=None,
        help="Print events after this seq (default: only new ones)",
    )
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
        return watch(client, since=args.since)
    except KeyboardInterrupt:
        return 1
