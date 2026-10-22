"""``fibsem-mcp watch``: long-poll a running app's events, one line per wake-up.

MCP is request-response: the App Server cannot wake an idle agent, and a
looping ``app_get_events`` tool blocks the agent's turn while it waits. So an
agent runs this as a background process its harness monitors, and each line
it prints wakes the agent. It must never be the actor: run in the foreground,
it would block the agent from acting on what it prints.

    fibsem-mcp watch
    fibsem-mcp watch --since 42          # resume after event 42
    python -m fibsem.ai.agent.mcp.sidecar watch   # where fibsem-mcp is not on PATH

The lines: ``PROMPT``, ``ANSWERED``, ``PROMPT WITHDRAWN``, ``PROPOSAL``,
``PROPOSAL RELEASED``, ``REQUEST`` (the agent's request decided, finished or
cancelled, FIB-1283), ``COMMAND ENDED`` (unless a caller was waiting for it
with ``service_wait_for_run``, so has the end), ``MILL STAGE STARTED`` and
``MILL ENDED`` (every mill, whoever started it), ``MILL READING`` (a milling
monitor's trigger or error), ``AUTOFOCUS`` and ``ALIGNMENT`` (every one, with
its check: passed, doubtful and why, or unchecked), task ``EVENT`` lines,
``WORKFLOW STARTED``, ``WORKFLOW ENDED``, ``QUIET`` (a workflow is running
and nothing has been said for a while), ``MESSAGE`` (the operator wrote to
the agent from its card, FIB-1423; one still unanswered when the watch starts
is said first), ``NOTE`` (the operator kept a note on a
grid), ``GAP``, ``RESTARTED``, ``LEVEL`` (the operator
changed the agent level) and ``AGENT OFF``/``AGENT BACK``.

It follows the whole session, not one workflow: the agent starts and stops
workflows itself, and between them it still has to hear the app (FIB-1431). A
workflow's end is a line, not the end of the watch, and so is one that stops
without an end event: ``WORKFLOW ENDED (stopped without an end event)``, once
``/app/workflow/status`` says nothing is running. It exits only when the
server cannot be watched (1), or when it is stopped. While the operator has the
agent level at Off the app refuses the agent, but it is still running: the
watch says so once and waits for the level to come back (FIB-1429).

An agent only wakes on a line, so a run that goes quiet for ``--quiet-after``
minutes gets one ``QUIET`` line to look at it; the next one waits for
something else to be said first.

It finds the server the way the sidecar does (``--url``/``--token``, the
environment, the discovery file) and follows a restarted server the same way,
on the same transport (`fibsem.client.transport`, FIB-1269). It works wherever
fibsem is installed, without the [mcp] extra.
"""

import argparse
import json
import sys
import time

from fibsem.ai.agent.levels import LEVEL_SKILLS, AgentLevel
from fibsem.client.transport import Transport

# Kinds that end a workflow: report it, and keep watching. A stalled workflow
# drained with decisions pending; a failed one died outside a task (FIB-1037).
WORKFLOW_ENDS = {
    "workflow_completed",
    "workflow_cancelled",
    "workflow_stalled",
    "workflow_failed",
}
# Kinds worth a wake-up line of their own (proposals have theirs).
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
RETRY_DELAY_S = 5.0
# Two minutes of refusals: an app's restart (close, open, connect) takes more
# than 20 s.
FAILURES_BEFORE_GIVING_UP = 24

GAP_LINE = (
    "GAP: events were lost before they could be read -- re-read /app/workflow/proposals "
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
# How long a running workflow may go without a line before the watch wakes the
# agent to look at it, once.
QUIET_S = 20 * 60.0


def _quiet_line(quiet_s):
    return (
        f"QUIET: the workflow is running and nothing has happened for "
        f"{quiet_s / 60:g} min -- check app_get_status, proposal_list and any "
        "unanswered message, then end your turn. Said once until something happens"
    )


def _poll(client, since, timeout):
    """One long-poll: (payload, None), or (None, why it failed)."""
    try:
        response = client.request(
            "GET",
            f"/app/events?since={since}&timeout={timeout}",
            timeout=timeout + 10.0,
        )
    except (OSError, ValueError) as error:
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
        if kind == "proposal_held":
            out(_proposal_line(data.get("item_name"), data.get("task_name")))
        elif kind == "proposal_released":
            out(
                f"PROPOSAL RELEASED item={data.get('item_name')} "
                f"task={data.get('task_name')}"
            )
        elif kind == "suggestion_declined":
            reason = " ".join(str(data.get("reason") or "").split())
            line = f"SUGGESTION DECLINED{_item_of(data)} task={data.get('task')}"
            out(f"{line}: {reason}" if reason else line)
        elif kind == "proposal_decided" and data.get("kind") == "actions":
            out(_request_decided_line(data))
        elif kind == "proposal_decided" and data.get("suggestion_id"):
            out(f"SUGGESTION APPROVED{_item_of(data)} task={data.get('task')}")
        elif kind in ("request_decided", "request_finished", "request_cancelled"):
            out(_request_line(kind, data))
        elif kind == "command_ended" and data.get("waited"):
            pass  # a caller waiting for it (service_wait_for_run) has the end
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
        elif kind == "milling_reading":
            line = _reading_line(data)
            if line:
                out(line)
        elif kind == "autofocus":
            out(_autofocus_line(event, data))
        elif kind == "alignment":
            out(_alignment_line(event, data))
        elif kind == "note" and event.get("actor") == "operator" and not data.get("id"):
            # A note the operator kept on an item (a grid's note), not to the agent.
            text = " ".join(str(data.get("text", "")).split())
            out(f"NOTE from the operator on {data.get('item_name')}: {text}")
        elif _is_message(event):
            out(_message_line(data))
        elif kind == "agent_level_set":
            line = _level_line(data)
            if line:
                out(line)
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


def _is_message(event) -> bool:
    """Whether *event* is the operator's message to the agent."""
    kind = event.get("kind", "")
    return kind == "operator_message" or (
        kind == "note" and event.get("actor") == "operator"
    )


def _message_line(data) -> str:
    message_id = data.get("id")
    text = " ".join(str(data.get("text", "")).split())
    return (
        f"MESSAGE id={message_id} from the operator: {text} -- answer with "
        f"experiment_add_note (reply_to={message_id}). It is not an "
        "approval: your level still decides what you may do"
    )


def _unanswered(events):
    """The operator's messages among *events* that no note answers yet, oldest
    first: sent while no agent was watching, they wait for the next one."""
    answered = {(e.get("payload") or {}).get("reply_to") for e in events}
    waiting = []
    for event in events:
        data = event.get("payload") or {}
        if _is_message(event) and data.get("id") and data["id"] not in answered:
            waiting.append(data)
    return waiting


def _level_line(data):
    """The operator changed the level, with the skill to work by at it. Off
    has no line here: the agent is refused, and the watch says AGENT OFF."""
    try:
        level = AgentLevel(data.get("level"))
    except ValueError:
        return None
    skill = LEVEL_SKILLS.get(level)
    if skill is None:
        return None
    return (
        f"LEVEL {level.value} (was {data.get('previous')}): the operator set the "
        f"agent level; work by the {skill} skill"
    )


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
    stages = [_stage_part(stage) for stage in data.get("stages") or []]
    if stages:
        line += " stages: " + ", ".join(stages)
    return f"{line} ({data['error']})" if data.get("error") else line


def _check_part(data):
    """How an auto function's result was judged, for its line: `` passed``,
    `` doubtful (why)``, or how it ended when it did not complete."""
    status = data.get("status")
    if status != "completed":
        reason = data.get("error") or data.get("reason")
        return f" status={status}" + (f" ({reason})" if reason else "")
    check = data.get("check") or {}
    verdict = check.get("verdict", "unchecked")
    reasons = check.get("reasons") or []
    return f" {verdict}" + (f" ({'; '.join(reasons)})" if reasons else "")


def _autofocus_line(event, data):
    """An autofocus ended: the beam, the working distance it left and moved
    from, its score and its check, so a bad focus is seen, not trusted."""
    line = f"AUTOFOCUS{_item_of(event)} beam={data.get('beam_type')}"
    start, wd = data.get("initial_working_distance"), data.get("working_distance")
    if isinstance(start, (int, float)) and isinstance(wd, (int, float)):
        line += f" wd={start * 1e3:.3f}->{wd * 1e3:.3f} mm"
    if data.get("focus_score") is not None:
        line += f" score={data['focus_score']:.3g}"
    line += _check_part(data)
    return f"{line} path={data['path']}" if data.get("path") else line


def _alignment_line(event, data):
    """An alignment ended: its name, the beam, the last step's shift and
    score, and its check (the methods agree, every step moved)."""
    line = (
        f"ALIGNMENT '{data.get('name')}'{_item_of(event)} beam={data.get('beam_type')}"
    )
    steps = data.get("results") or []
    if steps:
        last = steps[-1]
        shift = last.get("shift") or {}
        x, y = shift.get("x"), shift.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            line += f" shift=({x * 1e9:.0f}, {y * 1e9:.0f}) nm"
        if isinstance(last.get("score"), (int, float)):
            line += f" score={last['score']:.2f}"
    line += _check_part(data)
    return f"{line} path={data['path']}" if data.get("path") else line


def _stage_part(stage):
    """A stage's part of the MILL ENDED line, with why it ended other than by
    milling its patterns out: a skipped stage's reason, else its end's
    (``ended by monitor:FM Intensity: intensity drop``). A current the beam
    snapped to another it lists is said in brackets."""
    part = f"{stage.get('name')}={stage.get('status')}"
    if stage.get("current_warning"):
        part += f" [{stage['current_warning']}]"
    end = stage.get("end") or {}
    if stage.get("reason"):
        return f"{part} ({stage['reason']})"
    if end.get("kind") in (None, "finished") or not end.get("reason"):
        return part
    by = f" by {end['by']}" if end.get("by") else ""
    return f"{part} ({end['kind']}{by}: {end['reason']})"


def _reading_line(data):
    """A monitor's notable reading, from ``milling_reading``: a trigger (a
    drop) or the monitor's error. A step's own pauses and end are left to the
    MILL ENDED line."""
    reading = data.get("reading") or {}
    head = f"MILL READING monitor='{data.get('monitor')}' milled={data.get('milled_time')}s"
    if reading.get("error"):
        return f"{head} error: {reading['error']}"
    if not reading.get("trigger"):
        return None
    line = (
        f"{head} trigger='{reading['trigger']}' on_trigger={reading.get('on_trigger')}"
    )
    if reading.get("on_trigger") == "suggest":
        line += " -- the mill goes on; milling_end ends the stage if you agree"
    return line


def _request_decided_line(data):
    """The operator approved or rejected the agent's request, or it was
    withdrawn: its proposal's decision."""
    head = f"REQUEST {data.get('proposal_id')}"
    reason = " ".join(str(data.get("reason") or "").split())
    outcome = data.get("outcome")
    if outcome == "Confirmed":
        return f"{head} approved"
    if outcome == "Rejected":
        return f"{head} rejected" + (f": {reason}" if reason else "")
    return f"{head} cancelled: {reason}"


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


def _proposal_line(item_name, task_name):
    return (
        f"PROPOSAL item={item_name} task={task_name} "
        "-- the run is stopped on a proposal: find it in proposal_list; "
        "decide it at Drive, suggest a decision at Assist"
    )


def _held_proposals(client):
    """The proposals a run is stopped on now, as (item_name, task_name), from
    /app/workflow/status; empty when it cannot say."""
    try:
        response = client.request("GET", "/app/workflow/status", timeout=10.0)
        if response.status_code >= 400:
            return []
        asking = response.json()["workflow"].get("asking") or []
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return []
    return [
        (q.get("item_name"), q.get("task_name")) for q in asking if isinstance(q, dict)
    ]


def _run_is_live(client):
    """Whether /app/workflow/status says a run is going; None when it cannot say."""
    try:
        response = client.request("GET", "/app/workflow/status", timeout=10.0)
        if response.status_code >= 400:
            return None
        return bool(response.json()["workflow"]["running"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def watch(
    client,
    out=None,
    sleep_fn=None,
    poll_s=POLL_S,
    clock=None,
    since=None,
    quiet_s=QUIET_S,
):
    """Watch until the server cannot be watched (1). Workflows starting and
    ending are lines along the way, not the end of the watch (FIB-1431).

    ``since`` resumes after that event seq, so a watch restarted after the
    last one returned misses nothing in between; None starts at the tail.

    ``client`` is httpx-style (``request(method, path, timeout=...)``), in
    practice a `Transport`: a 401 or a refused connection re-reads the
    discovery file once per request, so a restarted app is followed without
    this loop knowing.

    ``quiet_s`` is how long a running workflow may go without a line before
    one ``QUIET`` line; 0 or None never says it.
    """
    say = out or _print_line
    sleep_fn = sleep_fn or time.sleep
    clock = clock or time.monotonic
    said_at = clock()
    nudged = False

    def out(line):
        nonlocal said_at, nudged
        said_at = clock()
        nudged = False
        say(line)

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
    # Started at the tail, a proposal asked before this watch began is behind
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
        for item_name, task_name in _held_proposals(client):
            out(_proposal_line(item_name, task_name))
    if held_before:
        # Messages sent while no agent was watching, still unanswered.
        for data in _unanswered(snapshot.get("events", [])):
            out(_message_line(data))
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

        if quiet_s and workflow_live and not nudged and clock() - said_at >= quiet_s:
            out(_quiet_line(quiet_s))
            nudged = True


def _print_line(line):
    print(line, flush=True)


def connect(url=None, token=None, transport=Transport):
    """A *transport* to the server, or None when no server is known."""
    try:
        return transport(url, token)
    except OSError:  # ServerUnreachable: none found
        return None


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
    parser.add_argument(
        "--quiet-after",
        type=float,
        default=QUIET_S / 60,
        metavar="MIN",
        help=f"Wake the agent once when a running workflow has said nothing for "
        f"this many minutes; 0 never does (default {QUIET_S / 60:g})",
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
        return watch(client, since=args.since, quiet_s=args.quiet_after * 60)
    except KeyboardInterrupt:
        return 1
    finally:
        client.close()
