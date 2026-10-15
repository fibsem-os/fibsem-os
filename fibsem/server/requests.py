"""The agent's requests: actions the operator approves before they run (FIB-1283).

At Assist the agent acts by asking. It files a request -- one or more of its
own tool calls, and why -- and the operator approves or rejects it in the app.
Approve runs each action through the App Server's own route, in order, with
Drive lent for that call only (``auth.lending_drive``). So an
approved action is validated, versioned, takes control of the microscope and is
refused exactly as the same call made at Drive would be, and a refusal comes
back as the route's own error.

The agent files and reads requests over HTTP (:func:`build_request_router`).
Approve and Reject are not routes: only the app calls them, so the agent cannot
approve its own request.

One request is pending at a time; a batch covers "several things at once".
There is no expiry at Assist -- silence means no -- but a request may carry one,
with what expiry does, for Drive's countdown (FIB-1267).
"""

import asyncio
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.acting import AGENT, acting
from fibsem.server.auth import AgentLevel, AuthConfig, lending_drive, require_level
from fibsem.server.catalog import CATALOG, ToolSpec, route_call
from fibsem.server.request_text import describe_action

PENDING = "pending"
APPLYING = "applying"
APPLIED = "applied"
FAILED = "failed"
REJECTED = "rejected"
CANCELLED = "cancelled"
_DONE = (APPLIED, FAILED, REJECTED, CANCELLED)

MAX_ACTIONS = 20
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
    # The operator's two lines for it, written by the server (request_text).
    what: str = ""
    change: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "method": self.method,
            "path": self.path,
            "what": self.what,
            "change": self.change,
        }


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
    # One per action run: tool, status_code, ok, and the route's answer.
    results: List[Dict[str, Any]] = field(default_factory=list)
    # For Drive's countdown (FIB-1267): when it lapses, and what lapsing does.
    expires_at: Optional[float] = None
    on_expiry: str = "reject"
    # Set when the level changes or the operator stops the agent mid-batch: the
    # actions not yet run are not run.
    _abort: Optional[str] = field(default=None, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.id,
            "status": self.status,
            "reason": self.reason,
            "actions": [a.to_dict() for a in self.actions],
            "created_at": self.created_at,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "reject_reason": self.reject_reason,
            "results": list(self.results),
            "expires_at": self.expires_at,
            "on_expiry": self.on_expiry if self.expires_at is not None else None,
        }


async def _asgi_call(app, method: str, path: str, body, token: str):
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
        "query_string": b"",
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


def _cut_short(answer: Any) -> bool:
    """Whether an action ran but was stopped part way: a mill answers
    ``cancelled`` rather than failing (FIB-1377). Applied, but the operator and
    the agent are told it was cut short (FIB-1398)."""
    if not isinstance(answer, dict):
        return False
    result = answer.get("result")
    if isinstance(result, dict):
        result = result.get("status")
    return isinstance(result, str) and result.lower() == CANCELLED


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
    ) -> None:
        self._app = app
        self._auth = auth
        self._mounted = mounted
        self._emit_fn = emit
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

    def create(
        self,
        actions: Any,
        reason: Any,
        expires_at: Optional[float] = None,
        on_expiry: str = "reject",
    ) -> AgentRequest:
        resolved = self._resolve(actions)
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
        if on_expiry not in ("approve", "reject"):
            raise ValueError("on_expiry is 'approve' or 'reject'")
        for action in resolved:
            described = describe_action(
                action.tool, action.arguments, action.path, read=self._read
            )
            action.what, action.change = described["what"], described["change"]
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
                expires_at=expires_at,
                on_expiry=on_expiry,
            )
            self._requests[request.id] = request
            self._trim_locked()
        self._emit(
            "request_raised",
            {
                "request_id": request.id,
                "reason": request.reason,
                "actions": [
                    {"tool": a.tool, "arguments": a.arguments} for a in resolved
                ],
            },
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
            if not isinstance(arguments, dict):
                raise RequestRefused(
                    422,
                    "invalid_arguments",
                    f"Action {index + 1}: arguments must be an object.",
                )
            try:
                method, path, body = route_call(spec, arguments)
            except ValueError as exc:
                raise RequestRefused(
                    422, "invalid_arguments", f"Action {index + 1}: {exc}"
                ) from None
            resolved.append(Action(tool, dict(arguments), method, path, body))
        return resolved

    # --- the operator decides ---

    def approve(self, request_id: str, by: str = "operator") -> AgentRequest:
        """Approve a pending request and start running its actions, on a worker
        thread: never the GUI thread, which the actions marshal onto."""
        with self._lock:
            request = self._pending_or_refuse(request_id)
            if not self._assisting():
                self._finish_locked(request, CANCELLED)
                cancelled = True
            else:
                request.status = APPLYING
                request.decided_by = by
                request.decided_at = time.time()
                cancelled = False
        if cancelled:
            self._cancelled(request, "the agent is no longer at Assist")
            raise RequestRefused(
                409,
                "request_cancelled",
                "The agent is no longer at Assist, so its request was cancelled.",
            )
        self._emit(
            "request_decided",
            {"request_id": request.id, "outcome": "approved", "decided_by": by},
        )
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
            request.decided_by = by
            request.decided_at = time.time()
            request.reject_reason = str(reason or "").strip()
            self._finish_locked(request, REJECTED)
        payload = {"request_id": request.id, "outcome": "rejected", "decided_by": by}
        if request.reject_reason:
            payload["reason"] = request.reject_reason
        self._emit("request_decided", payload)
        self._changed(request)
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
            self._finish_locked(request, CANCELLED)
        self._cancelled(request, why)
        return request

    def _level_changed(self, level: AgentLevel, stopped: bool) -> None:
        self.cancel(
            "the operator stopped the agent"
            if stopped
            else f"the agent level changed to {level.value.title()}"
        )

    def expire_due(self, now: Optional[float] = None) -> Optional[AgentRequest]:
        """Lapse the pending request if its time is up, as its ``on_expiry``
        says. Driven by whoever shows the countdown (FIB-1267)."""
        now = time.time() if now is None else now
        request = self.pending()
        if request is None or request.expires_at is None or now < request.expires_at:
            return None
        if request.on_expiry == "approve":
            return self.approve(request.id, by="expiry")
        return self.reject(request.id, by="expiry")

    # --- running an approved request ---

    def _apply(self, request: AgentRequest) -> None:
        status = APPLIED
        with acting(AGENT):
            for index, action in enumerate(request.actions):
                if request._abort is not None or not self._assisting():
                    status = CANCELLED
                    break
                with lending_drive():
                    try:
                        code, answer = asyncio.run(
                            _asgi_call(
                                self._app,
                                action.method,
                                action.path,
                                action.body,
                                self._auth.token,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - reported, not raised
                        code, answer = (
                            500,
                            {
                                "detail": {
                                    "error_type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            },
                        )
                ok = code < 400
                result = {
                    "index": index,
                    "tool": action.tool,
                    "status_code": code,
                    "ok": ok,
                }
                if ok:
                    result["result"] = answer
                    if _cut_short(answer):
                        result["cancelled"] = True
                else:
                    result["error"] = (answer or {}).get("detail", answer)
                request.results.append(result)
                self._emit("request_applied", {"request_id": request.id, **result})
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

    # --- helpers ---

    def _read(self, path: str) -> Optional[Dict[str, Any]]:
        """GET a route in this process: the document a config edit patches."""
        code, answer = asyncio.run(
            _asgi_call(self._app, "GET", path, None, self._auth.token)
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

    def _cancelled(self, request: AgentRequest, why: str) -> None:
        self._emit("request_cancelled", {"request_id": request.id, "why": why})
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
    """The agent's side of requests: file one, read them back."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def _refused(refusal: RequestRefused) -> HTTPException:
        return HTTPException(status_code=refusal.status_code, detail=refusal.detail)

    @router.post("/requests", status_code=202)
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
            created = store.create(body.get("actions"), body.get("reason"))
        except RequestRefused as refusal:
            raise _refused(refusal)
        return created.to_dict()

    @router.get("/requests")
    def list_requests():
        return {"requests": [r.to_dict() for r in store.recent()]}

    @router.get("/requests/{request_id}")
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
        return found.to_dict()

    return router
