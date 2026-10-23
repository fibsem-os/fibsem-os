"""The agent's requests: actions the operator approves before they run.

At Assist the agent acts by asking. It files a request -- one or more of its
own tool calls, and why -- and the operator approves or rejects it in the app.
Approve runs each action through the App Server's own route, in order, with
Drive lent for that call only (``auth.lending_drive``). So an
approved action is validated, versioned, takes control of the microscope and is
refused exactly as the same call made at Drive would be, and a refusal comes
back as the route's own error.

The agent files and reads requests over HTTP (:func:`build_request_router`).
Approve and Reject are the operator's: the window, or the operator's routes
behind the operator token (`fibsem.server.operator`), so the agent cannot
approve its own request.

One request is pending at a time; a batch covers "several things at once".
There is no expiry at Assist -- silence means no -- but a request may carry one,
with what expiry does, for Drive's countdown.

Each request is an ``actions`` proposal on the experiment, so it is saved with
who approved or rejected it and why; this store keeps what only this session
knows (applying, each action's result).
"""

import asyncio
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.acting import AGENT, Actor, acting
from fibsem.ai.agent.calls import AgentCall, calling, current_call, join_line
from fibsem.ai.agent.catalog import CATALOG, ToolSpec, route_call
from fibsem.ai.agent.catalog.describe import (
    Describer,
    describe_action,
    describe_actions,
)
from fibsem.ai.agent.levels import AgentLevel
from fibsem.codec import encode
from fibsem.devices.wire import command_body
from fibsem.server.auth import AuthConfig, lending_drive, require_level
from fibsem.server.responses import (
    ActionView,
    AgentRequestView,
    RequestList,
    answer,
)
from fibsem.workflows.actions import ACTION, action_record
from fibsem.workflows.evidence import EvidenceRefused, frame_path
from fibsem.workflows.evidence import load as load_evidence
from fibsem.workflows.proposals import ACTIONS, Decision, DecisionOutcome, Proposal

PENDING = "pending"
APPLYING = "applying"
APPLIED = "applied"
FAILED = "failed"
REJECTED = "rejected"
CANCELLED = "cancelled"
_DONE = (APPLIED, FAILED, REJECTED, CANCELLED)

MAX_ACTIONS = 20
# How long one read of a running step's run waits for it to end, in seconds.
RUN_POLL_S = 30
MAX_REASON = 2000
# Finished requests kept for GET: enough for an agent to read back its last few.
KEEP = 50


class RequestRefused(Exception):
    """A request that cannot be filed or decided as given."""

    def __init__(self, status_code: int, error_type: str, message: str, **extra):
        super().__init__(message)
        self.status_code = status_code
        self.detail = {"error_type": error_type, "message": message, **extra}


@dataclass
class Action:
    """One tool call, and the route call it stands for."""

    tool: str
    arguments: Dict[str, Any]
    method: str
    path: str
    body: Optional[Dict[str, Any]]
    # The operator's two lines for it, written by the server (catalog.describe).
    what: str = ""
    change: str = ""
    # Every value, for the operator's "Show details": {columns, rows}, or None.
    details: Optional[Dict[str, Any]] = None

    def view(self) -> ActionView:
        return ActionView(
            self.tool,
            self.arguments,
            self.method,
            self.path,
            self.what,
            self.change,
            self.details,
        )

    def to_dict(self) -> Dict[str, Any]:
        return encode(self.view())


@dataclass
class AgentRequest:
    id: str
    actions: List[Action]
    reason: str
    created_at: float = field(default_factory=time.time)
    status: str = PENDING
    decided_by: Optional[str] = None
    decided_at: Optional[float] = None
    # The operator's words on a reject, for the agent.
    reject_reason: str = ""
    # Why it was cancelled: the level changed, Stop agent, or the agent withdrew it.
    cancel_reason: str = ""
    # One per action run: tool, status_code, ok, and the route's answer.
    results: List[Dict[str, Any]] = field(default_factory=list)
    # What the agent judged from (`fibsem.workflows.evidence`, its record), and
    # the beam image of it the card draws the actions' values on.
    evidence: Optional[Dict[str, Any]] = None
    image: Optional[str] = None
    # The experiment its proposal is on.
    experiment: Any = field(default=None, repr=False)
    # Set when the level changes or the operator stops the agent mid-batch: the
    # actions not yet run are not run.
    _abort: Optional[str] = field(default=None, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def view(self) -> AgentRequestView:
        return AgentRequestView(
            request_id=self.id,
            status=self.status,
            reason=self.reason,
            actions=[a.view() for a in self.actions],
            created_at=self.created_at,
            decided_by=self.decided_by,
            decided_at=self.decided_at,
            reject_reason=self.reject_reason,
            cancel_reason=self.cancel_reason,
            results=list(self.results),
            image=self.image or None,
            evidence=self.evidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return encode(self.view())


async def _asgi_call(app, method: str, path: str, body, token: str, query: str = ""):
    """One call into the ASGI app in this process, as an HTTP client would make
    it: (status, decoded JSON or None)."""
    payload = json.dumps(body).encode() if body is not None else b""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": quote(path).encode(),
        "root_path": "",
        "query_string": query.encode(),
        "headers": [
            (b"authorization", f"Bearer {token}".encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("app-server", 0),
        "server": ("app-server", 80),
    }
    sent = False
    finished = asyncio.Event()
    status = 500
    chunks: List[bytes] = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                finished.set()

    await app(scope, receive, send)
    raw = b"".join(chunks)
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, None


def _call_named(spec: ToolSpec, tool: str, arguments: Dict[str, Any]) -> str:
    """A tool call as the record names it: the tool and what its route names
    ("device_set_parameter electron.on"), so a refusal at filing says what was
    refused rather than "requests"."""
    target = ".".join(
        str(arguments[name])
        for name in re.findall(r"{(\w+)}", spec.path)
        if name in arguments
    )
    return f"{tool} {target}" if target else tool


def _argument_problem(spec: ToolSpec, unknown: List[str], missing: List[str]) -> str:
    """What is wrong with a request's arguments for *spec*, in the agent's terms:
    the names it does not take, the ones it needs, and what it takes.
    Checked at filing, so the operator never reads an action with a value left
    out ("Load 'None'")."""
    parts = []
    if unknown:
        parts.append(f"{spec.name} takes no {', '.join(unknown)}")
    if missing:
        parts.append(f"{spec.name} needs {', '.join(missing)}")
    takes = ", ".join(
        f"{name} (optional)" if name in spec.optional else name for name in spec.params
    )
    return f"{'; '.join(parts)}. It takes: {takes or 'no arguments'}."


def _started_run(answer: Any) -> Optional[Dict[str, Any]]:
    """The run when *answer* is a long command's start, which goes on after
    the answer (`fibsem.devices.runs`); None for any other answer."""
    result = answer.get("result") if isinstance(answer, dict) else None
    if isinstance(result, dict) and isinstance(result.get("run_id"), str):
        if "state" in result and "command" in result:
            return result
    return None


def _cut_short(answer: Any) -> bool:
    """Whether an action ran but was stopped part way: a mill answers
    ``cancelled`` rather than failing. Applied, but the operator and
    the agent are told it was cut short. A long command answers its
    run, whose ``state`` says so when it ended before the answer."""
    if not isinstance(answer, dict):
        return False
    result = answer.get("result")
    if isinstance(result, dict):
        result = result.get("status") or result.get("state")
    return isinstance(result, str) and result.lower() == CANCELLED


# The flags a route answers 2xx with to say whether it did what it was asked:
# workflow_start answers ``started: false`` and a reason when it is turned away,
# a context edit ``applied: false``, a stop with nothing to stop ``stopped: false``.
_OUTCOME_FLAGS = ("ok", "started", "applied", "stopped")


def turned_away(answer: Any) -> Optional[Dict[str, Any]]:
    """The refusal in an answer that came back 2xx but says the action did
    nothing (one of its outcome flags is False), as a route's error detail;
    None for an answer that did what it was asked. Such a step failed, with
    its reason, rather than reading "applied"."""
    if not isinstance(answer, dict):
        return None
    flag = next((name for name in _OUTCOME_FLAGS if answer.get(name) is False), None)
    if flag is None:
        return None
    reason = next(
        (
            answer[key]
            for key in ("reason", "error", "message")
            if isinstance(answer.get(key), str) and answer[key].strip()
        ),
        f"Answered {flag}: false.",
    )
    return {"error_type": f"not_{flag}", "message": reason}


class RequestStore:
    """The agent's requests, and the worker that runs approved ones.

    ``emit(kind, payload)`` puts each step on the experiment's record;
    ``subscribe`` tells the app when a request changes, from whichever thread
    changed it.
    """

    def __init__(
        self,
        app,
        auth: AuthConfig,
        mounted: frozenset,
        emit: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        describe: Optional[Describer] = None,
        check: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
        not_allowed: Optional[Callable[[str, str], Optional[str]]] = None,
        experiment_path: Optional[Callable[[], Optional[str]]] = None,
        experiment: Optional[Callable[[], Any]] = None,
        path_refused: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    ) -> None:
        self._app = app
        # Why a command's or query's path would be refused when it runs (outside
        # the open experiment), or None: refused at filing.
        self._path_refused = path_refused
        # The experiment a request is recorded on (``recording_experiment``).
        self._experiment = experiment or (lambda: None)
        # Why a request's image path is refused (outside the experiment), or None.
        # The open experiment's folder: an image the agent shows is inside it.
        self._experiment_path = experiment_path or (lambda: None)
        self._describe = describe
        # Why the app would refuse an action when it runs (a config edit that
        # does not apply), or None: refused at filing, as the blocklist is.
        self._check = check
        self._auth = auth
        self._mounted = mounted
        self._emit_fn = emit
        # Why a route call is never the agent's (the blocklist), or None: refused
        # at filing, so the operator is never asked about what cannot run.
        self._not_allowed = not_allowed
        self._lock = threading.Lock()
        self._requests: Dict[str, AgentRequest] = {}
        self._listeners: List[Callable[[AgentRequest], None]] = []
        self._tools = {spec.name: spec for spec in CATALOG}
        # A level change or Stop agent cancels what is waiting, and the rest of
        # what is running.
        auth.on_level_change(self._level_changed)

    # --- reading ---

    def get(self, request_id: str) -> Optional[AgentRequest]:
        with self._lock:
            return self._requests.get(request_id)

    def pending(self) -> Optional[AgentRequest]:
        with self._lock:
            return self._pending_locked()

    def recent(self) -> List[AgentRequest]:
        """Newest first."""
        with self._lock:
            return sorted(self._requests.values(), key=lambda r: -r.created_at)

    def subscribe(self, listener: Callable[[AgentRequest], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def wait(self, request_id: str, timeout: Optional[float] = None) -> AgentRequest:
        """Block until the request is finished (applied, failed, rejected or
        cancelled), or the timeout passes."""
        request = self.get(request_id)
        if request is None:
            raise KeyError(request_id)
        request._done.wait(timeout)
        return request

    # --- the agent files one ---

    def _evidence(self, evidence_id: Any) -> Optional[Dict[str, Any]]:
        """The record of the evidence a request cites, or None for none."""
        if not evidence_id:
            return None
        try:
            return load_evidence(self._experiment_path, evidence_id)
        except EvidenceRefused as refusal:
            raise RequestRefused(
                refusal.status_code, refusal.error_type, refusal.message
            )

    def create(
        self,
        actions: Any,
        reason: Any,
        evidence: Any = None,
    ) -> AgentRequest:
        resolved = self._resolve(actions)
        cited = self._evidence(evidence)
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > MAX_REASON
        ):
            raise RequestRefused(
                422,
                "missing_field",
                f"Pass a reason (at most {MAX_REASON} characters): it is what the "
                "operator reads before approving.",
            )
        experiment = self._recording()
        for index, line in enumerate(self._describe_actions(resolved)):
            if line.get("refused"):
                raise RequestRefused(
                    422,
                    "cannot_run",
                    f"Action {index + 1}: {line['refused']}.",
                    tool=resolved[index].tool,
                )
        with self._lock:
            waiting = self._pending_locked()
            if waiting is not None:
                raise RequestRefused(
                    409,
                    "request_pending",
                    "A request is already waiting for the operator; one at a time. "
                    "Wait for its outcome, or put everything in one request.",
                    request_id=waiting.id,
                )
            request = AgentRequest(
                id=secrets.token_hex(4),
                actions=resolved,
                reason=reason.strip(),
                image=frame_path(cited),
                evidence=cited,
            )
            self._requests[request.id] = request
            self._trim_locked()
        request.experiment = experiment
        provenance = {"reason": request.reason}
        if cited:
            provenance["evidence"] = cited["evidence_id"]
        if request.image:
            provenance["reference_image"] = request.image
        experiment.ask_proposal(
            None,
            "",
            Proposal(
                kind=ACTIONS,
                id=request.id,
                # With the card's own words, so a line about it later (the
                # app's Recent list) reads as the card did.
                values={
                    "actions": [
                        {
                            "tool": a.tool,
                            "arguments": a.arguments,
                            "what": a.what,
                            "change": a.change,
                        }
                        for a in resolved
                    ]
                },
                provenance=provenance,
                asked_by=self._agent(),
            ),
        )
        self._changed(request)
        return request

    def _resolve(self, actions: Any) -> List[Action]:
        if (
            not isinstance(actions, list)
            or not actions
            or len(actions) > MAX_ACTIONS
            or not all(isinstance(a, dict) for a in actions)
        ):
            raise RequestRefused(
                422,
                "missing_field",
                f"Pass actions: a list of 1 to {MAX_ACTIONS} objects, each "
                "{tool, arguments}, with the arguments you would give that tool.",
            )
        resolved = []
        for index, action in enumerate(actions):
            tool = action.get("tool")
            arguments = action.get("arguments", {}) or {}
            spec = self._tools.get(tool) if isinstance(tool, str) else None
            if spec is None:
                raise RequestRefused(
                    422, "unknown_tool", f"Action {index + 1}: no tool named {tool!r}."
                )
            if tool == "proposal_decide":
                raise RequestRefused(
                    422,
                    "not_an_action",
                    f"Action {index + 1}: at Assist the operator decides "
                    "proposals, so proposal_decide can't go in a request: "
                    "suggest the decision with proposal_suggest instead.",
                )
            if spec.level == "assist":
                raise RequestRefused(
                    422,
                    "not_an_action",
                    f"Action {index + 1}: {tool} needs no approval at Assist, "
                    "so call it directly.",
                )
            if spec.level != "drive":
                raise RequestRefused(
                    422,
                    "not_an_action",
                    f"Action {index + 1}: {tool} only reads; call it directly.",
                )
            if spec.router not in self._mounted:
                raise RequestRefused(
                    422,
                    "not_available",
                    f"Action {index + 1}: {tool} is not served here (the App "
                    "Server's devices setting is off).",
                )
            if spec.part and spec.part not in self._mounted:
                raise RequestRefused(
                    422,
                    "not_available",
                    f"Action {index + 1}: {tool} is not served here (this app "
                    f"has no {spec.part.replace('_', ' ')}).",
                )
            if not isinstance(arguments, dict):
                raise RequestRefused(
                    422,
                    "invalid_arguments",
                    f"Action {index + 1}: arguments must be an object.",
                )
            unknown = sorted(set(arguments) - set(spec.params))
            missing = [
                name
                for name in spec.params
                if name not in spec.optional and arguments.get(name) is None
            ]
            if unknown or missing:
                raise RequestRefused(
                    422,
                    "invalid_arguments",
                    f"Action {index + 1}: {_argument_problem(spec, unknown, missing)}",
                    tool=tool,
                    unknown=unknown,
                    missing=missing,
                    takes=list(spec.params),
                )
            try:
                method, path, body = route_call(spec, arguments)
            except ValueError as exc:
                raise RequestRefused(
                    422, "invalid_arguments", f"Action {index + 1}: {exc}"
                ) from None
            why = self._not_allowed(method, path) if self._not_allowed else None
            if why is not None:
                raise RequestRefused(
                    403,
                    "not_allowed",
                    f"Action {index + 1}: {why}",
                    call=_call_named(spec, tool, arguments),
                )
            kwargs = body.get("kwargs") if isinstance(body, dict) else None
            why = (
                self._path_refused(kwargs)
                if self._path_refused is not None
                and isinstance(kwargs, dict)
                and ("/commands/" in path or "/queries/" in path)
                else None
            )
            if why is not None:
                raise RequestRefused(
                    403,
                    "not_allowed",
                    f"Action {index + 1}: {tool} refused: {why}.",
                    call=_call_named(spec, tool, arguments),
                )
            why = self._check(tool, arguments) if self._check else None
            if why is not None:
                raise RequestRefused(
                    422,
                    "invalid_patch",
                    f"Action {index + 1}: {why}",
                    tool=tool,
                )
            resolved.append(Action(tool, dict(arguments), method, path, body))
        return resolved

    # --- the operator decides ---

    def approve(self, request_id: str, by: str = "operator") -> AgentRequest:
        """Approve a pending request and start running its actions, on a worker
        thread: never the GUI thread, which the actions marshal onto."""
        with self._lock:
            request = self._pending_or_refuse(request_id)
            assisting = self._assisting()
        if not assisting:
            self._close(request, CANCELLED, "the agent is no longer at Assist")
            raise RequestRefused(
                409,
                "request_cancelled",
                "The agent is no longer at Assist, so its request was cancelled.",
            )
        self._decide(request, DecisionOutcome.Confirmed, by)
        with self._lock:
            request.status = APPLYING
        self._changed(request)
        threading.Thread(
            target=self._apply,
            args=(request,),
            name=f"agent-request-{request.id}",
            daemon=True,
        ).start()
        return request

    def reject(
        self, request_id: str, reason: str = "", by: str = "operator"
    ) -> AgentRequest:
        with self._lock:
            request = self._pending_or_refuse(request_id)
        request.reject_reason = str(reason or "").strip()
        self._decide(request, DecisionOutcome.Rejected, by, request.reject_reason)
        with self._lock:
            self._finish_locked(request, REJECTED)
        self._changed(request)
        return request

    def withdraw(self, request_id: str, reason: Any) -> AgentRequest:
        """The agent takes back its pending request, with a reason: the operator
        changed their mind, or the agent did. Only while it waits; once
        decided, it is the operator's."""
        if not isinstance(reason, str) or not reason.strip():
            raise RequestRefused(
                422,
                "missing_field",
                "Pass a reason: the operator reads why the request was withdrawn.",
            )
        with self._lock:
            request = self._pending_or_refuse(request_id)
        self._close(
            request,
            CANCELLED,
            f"withdrawn by the agent: {reason.strip()}",
            actor=self._agent(),
        )
        return request

    def cancel(self, why: str) -> Optional[AgentRequest]:
        """Cancel the pending request, if any: the level changed, or the operator
        stopped the agent. A request already running stops before its next
        action; the one running finishes, or is stopped by Stop agent."""
        with self._lock:
            for running in self._requests.values():
                if running.status == APPLYING:
                    running._abort = why
            request = self._pending_locked()
        if request is None:
            return None
        # The operator's doing: they changed the level, stopped the agent or
        # the App Server.
        operator = request.experiment.actor() if request.experiment else None
        self._close(request, CANCELLED, why, actor=operator)
        return request

    def _level_changed(self, level: AgentLevel, stopped: bool) -> None:
        self.cancel(
            "the operator stopped the agent"
            if stopped
            else f"the agent level changed to {level.value.title()}"
        )

    # --- running an approved request ---

    def _apply(self, request: AgentRequest) -> None:
        status = APPLIED
        with acting(AGENT):
            for index, action in enumerate(request.actions):
                if request._abort is not None or not self._assisting():
                    status = CANCELLED
                    break
                call = AgentCall(
                    tool=action.tool,
                    arguments=action.arguments,
                    action=join_line(action.what, action.change),
                    level=AgentLevel.ASSIST.value,
                )
                with lending_drive(), calling(call):
                    code, answer = self._call(action.method, action.path, action.body)
                    run = _started_run(answer) if code < 400 else None
                    if run is not None:
                        # The next step waits for this one's run to end, and a
                        # run that fails fails the step.
                        code, answer = self._wait_for_run(request, action, run)
                refused = turned_away(answer) if code < 400 else None
                ok = code < 400 and refused is None
                result = {
                    "index": index,
                    "tool": action.tool,
                    "status_code": code,
                    "ok": ok,
                }
                if code < 400:
                    result["result"] = answer
                if ok and _cut_short(answer):
                    result["cancelled"] = True
                if not ok:
                    result["error"] = refused or (answer or {}).get("detail", answer)
                request.results.append(result)
                self._emit(
                    ACTION,
                    action_record(
                        join_line(action.what, action.change) or action.tool,
                        ok=ok,
                        request_id=request.id,
                        **{k: v for k, v in result.items() if k != "ok"},
                    ),
                )
                self._changed(request)
                if not ok:
                    status = FAILED
                    break
            with self._lock:
                self._finish_locked(request, status)
            self._emit(
                "request_finished",
                {
                    "request_id": request.id,
                    "status": status,
                    "applied": sum(1 for r in request.results if r["ok"]),
                    "of": len(request.actions),
                    "cancelled": sum(1 for r in request.results if r.get("cancelled")),
                },
            )
        self._changed(request)

    def _call(self, method: str, path: str, body: Any, query: str = ""):
        """One route call in this process: (status, answer). What it raises is
        answered as a 500, its type and message the error."""
        try:
            return asyncio.run(
                _asgi_call(self._app, method, path, body, self._auth.token, query)
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return 500, {
                "detail": {"error_type": type(exc).__name__, "message": str(exc)}
            }

    def _wait_for_run(self, request: AgentRequest, action: Action, run: Dict):
        """Wait for the *run* *action* started to end: (status, answer), the
        answer the ended run as the command's own would be. A run that raised
        is its error, with the status the route answers it with; one
        that failed otherwise is a 500. A stopped one is cut short. Stops
        waiting, the run still going, when the request is cancelled."""
        run_id = run["run_id"]
        run_path = action.path.rsplit("/commands/", 1)[0] + "/run"
        query = f"run_id={quote(run_id)}&wait={RUN_POLL_S}"
        # Read through the run route at least once: a run that ended before
        # its command answered says how it failed only there.
        read = False
        while run.get("state") == "running" or not read:
            stopping = request._abort is not None or not self._assisting()
            if stopping and run.get("state") == "running":
                return 200, {"result": run}
            read = True
            code, answer = self._call("GET", run_path, None, query)
            if code >= 400:
                return code, answer
            run = (answer or {}).get("run") or {}
            if run.get("run_id") != run_id:
                return 500, {
                    "detail": {
                        "error_type": "run_replaced",
                        "message": f"Its run {run_id} was replaced by another.",
                    }
                }
        if run.get("state") == CANCELLED:
            return 200, {"result": run}  # stopped: applied, cut short
        error = run.get("error")
        if error is None and run.get("state") == FAILED:
            error = {"error_type": "failed", "message": f"{run.get('command')} failed."}
        if error is not None:
            return run.get("status_code") or 500, {"detail": error}
        return 200, {"result": run}

    def refresh(self, request_id: str) -> bool:
        """Describe a waiting request's actions again: the values now, read when
        the operator looks, not when it was filed. An edit against a
        version keeps the line it was filed with: it applies only to the values
        it was read against, so those are its "now". Reads the routes
        it describes, so never on the GUI thread. False when it no longer
        waits."""
        request = self.get(request_id)
        if request is None or request.status != PENDING:
            return False
        before = [(a.what, a.change, a.details) for a in request.actions]
        self._describe_actions(request.actions, refresh=True)
        if [(a.what, a.change, a.details) for a in request.actions] != before:
            self._changed(request)
        return True

    # --- helpers ---

    def _describe_actions(
        self, actions: List[Action], refresh: bool = False
    ) -> List[Dict[str, Any]]:
        lines = describe_actions(
            [(a.tool, a.arguments, a.path) for a in actions],
            read=self._read,
            describe=self._describe,
        )
        for action, line in zip(actions, lines):
            if refresh and "version" in action.arguments:
                continue
            action.what, action.change = line["what"], line["change"]
            action.details = line.get("details")
        return lines

    def describe_call(
        self, tool: str, arguments: Dict[str, Any], path: str
    ) -> Optional[str]:
        """A direct call as its request's card would say it, ``what · change``,
        reading the state it changes: "Set Field of View 150 µm → 300 µm", not
        the bare "Set hfw to 0.0003" a line without a reader falls back to."""
        line = describe_action(
            tool, arguments, path, read=self._read, describe=self._describe
        )
        return join_line(line.get("what"), line.get("change"))

    def _read(
        self, path: str, kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Read a route in this process: the document a config edit patches, or
        a service query's answer with *kwargs* (a query moves nothing), such as
        which grid an exchange unloads or how long an overview takes."""
        if "/queries/" in path:
            method, body = "POST", command_body(kwargs)
        elif path.endswith("/workflow/grids/plan"):
            # The grid run's plan: a POST that only reads, its body the run's.
            method, body = "POST", dict(kwargs or {})
        else:
            method, body = "GET", None
        code, answer = asyncio.run(
            _asgi_call(self._app, method, path, body, self._auth.token)
        )
        return answer if code < 400 and isinstance(answer, dict) else None

    def _assisting(self) -> bool:
        return self._auth.level is AgentLevel.ASSIST and not self._auth.stopped

    def _pending_locked(self) -> Optional[AgentRequest]:
        for request in self._requests.values():
            if request.status == PENDING:
                return request
        return None

    def _pending_or_refuse(self, request_id: str) -> AgentRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise RequestRefused(404, "not_found", f"No request {request_id!r}.")
        if request.status != PENDING:
            raise RequestRefused(
                409,
                "not_pending",
                f"Request {request_id} is {request.status}, not waiting.",
            )
        return request

    def _finish_locked(self, request: AgentRequest, status: str) -> None:
        request.status = status
        request._done.set()

    def _trim_locked(self) -> None:
        done = [r for r in self._requests.values() if r.status in _DONE]
        for request in sorted(done, key=lambda r: r.created_at)[
            : max(0, len(done) - KEEP)
        ]:
            del self._requests[request.id]

    def _recording(self) -> Any:
        """The experiment a new request is recorded on, or refused."""
        try:
            experiment = self._experiment()
        except OSError as exc:
            experiment, why = None, str(exc)
        else:
            why = "no experiment is open"
        if experiment is None:
            raise RequestRefused(
                409,
                "no_experiment",
                f"A request is recorded on the experiment, and {why}.",
            )
        return experiment

    def _agent(self) -> Actor:
        return Actor.agent(self._auth.agent_client or "agent")

    def _decide(
        self,
        request: AgentRequest,
        outcome: DecisionOutcome,
        by: str,
        reason: str = "",
    ) -> None:
        """The operator's (or expiry's) decision on the request's proposal."""
        experiment = request.experiment
        if by == "operator" and experiment is not None:
            actor = experiment.actor()
        else:
            actor = Actor.parse(by) if ":" in by else Actor.task(by)
        decided = experiment.decide(
            None,
            "",
            Decision(
                outcome=outcome,
                actor=actor,
                reason=reason,
                via="request",
                proposal_id=request.id,
            ),
        )
        if not decided.applied:
            raise RequestRefused(
                409, decided.error_type or "not_pending", decided.reason
            )
        with self._lock:
            request.decided_by = str(actor)
            request.decided_at = time.time()

    def _close(
        self,
        request: AgentRequest,
        status: str,
        why: str,
        actor: Optional[Actor] = None,
    ) -> None:
        """End a request nobody decided, its proposal withdrawn with ``why``."""
        with self._lock:
            if request.status != PENDING:
                return
            request.cancel_reason = why
            self._finish_locked(request, status)
        if request.experiment is not None:
            request.experiment.withdraw_experiment_proposal(request.id, why, actor)
        self._changed(request)

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._emit_fn is not None:
            self._emit_fn(kind, payload)

    def _changed(self, request: AgentRequest) -> None:
        for listener in list(self._listeners):
            try:
                listener(request)
            except Exception:  # noqa: BLE001 - a listener never costs the request
                pass


def build_request_router(store: RequestStore) -> APIRouter:
    """The agent's side of requests: file one, withdraw it, read them back."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def _refused(refusal: RequestRefused) -> HTTPException:
        return HTTPException(status_code=refusal.status_code, detail=refusal.detail)

    @router.post("/requests", status_code=202, response_model=AgentRequestView)
    def create_request(request: Request, body: Dict[str, Any]):
        auth: AuthConfig = request.app.state.auth
        if auth.level is not AgentLevel.ASSIST:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_type": "level",
                    "level": auth.level.value,
                    "message": "Requests are how the agent acts at Assist. "
                    + (
                        "At Drive, make the call itself."
                        if auth.level is AgentLevel.DRIVE
                        else "The operator sets the level from the Agent chip."
                    ),
                },
            )
        try:
            call = current_call()
            created = store.create(
                body.get("actions"),
                body.get("reason"),
                evidence=call.evidence if call is not None else None,
            )
        except RequestRefused as refusal:
            raise _refused(refusal)
        return answer(created.view(), status_code=202)

    @router.post("/requests/{request_id}/withdraw", response_model=AgentRequestView)
    def withdraw_request(request_id: str, body: Dict[str, Any]):
        try:
            withdrawn = store.withdraw(request_id, body.get("reason"))
        except RequestRefused as refusal:
            raise _refused(refusal)
        return answer(withdrawn.view())

    @router.get("/requests", response_model=RequestList)
    def list_requests():
        return answer(RequestList([r.view() for r in store.recent()]))

    @router.get("/requests/{request_id}", response_model=AgentRequestView)
    def get_request(request_id: str):
        found = store.get(request_id)
        if found is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_type": "not_found",
                    "message": f"No request {request_id!r}.",
                },
            )
        return answer(found.view())

    return router
