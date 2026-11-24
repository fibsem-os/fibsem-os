"""The agent's review list: its finished work, what it wants the operator to
look at, and what it waits on, each with its pictures.

The agent adds an item (``review_add``) at one of three levels: ``done`` (a
record of finished work), ``check`` (it wants a look, and carries on) or
``wait`` (it holds that piece of work until someone answers). The operator
reads the list on the Agent page and answers an item with one of its options
(Accept or Reject unless the agent gave its own), a note and marks on a
picture; the answer reaches the agent as the operator's message
(`fibsem.server.agent.operator`). The agent may close an item it no longer
needs answered. Nothing is enforced: a wait is the agent's own promise.

An item may be a scored result (``score``: sites, target or polishing, about
the grid or lamella it scores): the operator's Accept is a pass and Reject a
fail, with the note saying why. The run's scores are counted from these.

Kept at ``agent/review-<session id>.jsonl`` beside the plan (`AgentFiles`):
one line per item added, answered or closed, read back in order.
"""

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.agent.files import AgentFiles
from fibsem.server.agent.plan import running_session
from fibsem.server.agent.record import cited_evidence
from fibsem.server.auth import (
    AuthConfig,
    is_operator_token,
    refusal_error,
    require_level,
)
from fibsem.server.bodies import ReviewAddBody, ReviewCloseBody

logger = logging.getLogger(__name__)

ADDED = "agent_review_added"
ANSWERED = "agent_review_answered"
CLOSED = "agent_review_closed"
LEVELS = ("done", "check", "wait")
OPTIONS = ("Accept", "Reject")
# What a scored item scores: a grid's site list, a lamella's target, a
# lamella's polishing. Accept is a pass, Reject a fail.
SCORES = ("sites", "target", "polishing")
MAX_TITLE = 120
MAX_TEXT = 2000
MAX_ABOUT = 60
MAX_OPTION = 40
MAX_OPTIONS = 4
MAX_IMAGES = 4
MAX_FACTS = 4
MAX_NOTE = 4000


def _invalid(message: str) -> Exception:
    return refusal_error(422, "invalid", message)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _line(raw: Any, name: str, most: int, required: bool = False) -> str:
    text = raw.strip() if isinstance(raw, str) else ""
    if (required and not text) or len(text) > most or "\n" in text:
        need = "a line" if required else "at most a line"
        raise _invalid(f"{name} is {need} of at most {most} characters.")
    return text


def _images(raw: Optional[List[str]]) -> List[str]:
    found = [p.strip().replace("\\", "/") for p in raw or [] if p and p.strip()]
    if len(found) > MAX_IMAGES:
        raise _invalid(f"images is at most {MAX_IMAGES} paths.")
    for path in found:
        if path.startswith("/") or ".." in path.split("/") or ":" in path:
            raise _invalid(
                f"{path!r}: images are paths inside the experiment, as the "
                "image tools answer them."
            )
    return found


def _facts(raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    found = []
    for fact in raw or []:
        if not isinstance(fact, dict):
            raise _invalid("facts is a list of {label, value}.")
        found.append(
            {
                "label": _line(fact.get("label"), "A fact's label", 30, True),
                "value": _line(fact.get("value"), "A fact's value", 80, True),
            }
        )
    if len(found) > MAX_FACTS:
        raise _invalid(f"facts is at most {MAX_FACTS}.")
    return found


def _options(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return list(OPTIONS)
    found = [_line(o, "An option", MAX_OPTION, True) for o in raw]
    if not 2 <= len(found) <= MAX_OPTIONS or len(set(found)) < len(found):
        raise _invalid(f"options is 2 to {MAX_OPTIONS} different answers.")
    return found


def state(item: Dict[str, Any]) -> str:
    """waiting, to_check, answered, closed or done."""
    if item.get("answer"):
        return "answered"
    if item.get("closed"):
        return "closed"
    return {"wait": "waiting", "check": "to_check"}.get(item["level"], "done")


class ReviewStore:
    """One session's review list, in memory and in the session's folder."""

    def __init__(
        self,
        experiment_path: Callable[[], Optional[str]],
        files: Optional[AgentFiles] = None,
    ) -> None:
        self._files = files or AgentFiles(experiment_path)
        self.session_id: Optional[str] = None
        self.items: List[Dict[str, Any]] = []
        # The agent adds and the operator answers on different threads: one
        # change at a time, each on disk before it is held.
        self.lock = threading.RLock()

    def _file(self, session_id: str) -> Optional[Path]:
        folder = self._files.folder(session_id)
        return folder / f"review-{session_id}.jsonl" if folder is not None else None

    def current(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        """The list of *session_id*: what is held, else read from its file."""
        with self.lock:
            return self._current(session_id)

    def _current(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        if session_id != self.session_id:
            self.session_id, self.items = session_id, []
            path = self._file(session_id) if session_id else None
            if path is not None and path.is_file():
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            self._apply(json.loads(line))
                except (OSError, ValueError):
                    logger.warning("could not read the review %s", path, exc_info=True)
        return self.items

    def _apply(self, change: Dict[str, Any]) -> None:
        op = change.get("op")
        if op == "add":
            self.items.append({k: v for k, v in change.items() if k != "op"})
            return
        item = self.find(change.get("id"))
        if item is None:
            return
        if op == "answer":
            item["answer"] = change.get("answer")
        elif op == "close":
            item["closed"] = change.get("closed")
        elif op == "told" and item.get("answer"):
            item["answer"]["message_id"] = change.get("message_id")

    def _write(self, change: Dict[str, Any]) -> None:
        """*change* on disk, then held; refused (503 not_saved) when it can't
        be written, so nothing is held that the file doesn't have."""
        path = self._file(self.session_id) if self.session_id else None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as out:
                    out.write(json.dumps(change) + "\n")
            except OSError as error:
                logger.warning("could not write the review %s", path, exc_info=True)
                raise refusal_error(
                    503,
                    "not_saved",
                    f"The review list could not be saved ({error}): nothing was changed.",
                )
        self._apply(change)

    def find(self, item_id: Any) -> Optional[Dict[str, Any]]:
        return next((i for i in self.items if i["id"] == item_id), None)

    def add(self, session_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self._current(session_id)
            item = {"id": f"r{len(self.items) + 1}", "at": now(), **item}
            self._write({"op": "add", **item})
            return self.find(item["id"])

    def answer(
        self, session_id: str, item_id: str, answer: Dict[str, Any]
    ) -> Dict[str, Any]:
        with self.lock:
            open_item(self, session_id, item_id)
            self._write({"op": "answer", "id": item_id, "answer": answer})
            return self.find(item_id)

    def told(self, session_id: str, item_id: str, message_id: str) -> None:
        """The message that told the agent of the answer to *item_id*."""
        with self.lock:
            self._current(session_id)
            self._write({"op": "told", "id": item_id, "message_id": message_id})

    def close(self, session_id: str, item_id: str, reason: str) -> Dict[str, Any]:
        with self.lock:
            open_item(self, session_id, item_id)
            closed = {"reason": reason, "at": now()}
            self._write({"op": "close", "id": item_id, "closed": closed})
            return self.find(item_id)

    def view(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {**item, "state": state(item)}

    def counts(self, reached: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        found: Dict[str, Any] = {
            "waiting": 0,
            "to_check": 0,
            "rejected": 0,
            "all": len(self.items),
            "scores": scores(self.items, reached),
        }
        for item in self.items:
            said = state(item)
            if said in found:
                found[said] += 1
            choice = (item.get("answer") or {}).get("choice")
            if choice == "Reject":
                found["rejected"] += 1
        return found


# The tasks after which a lamella's polishing is scored: polishing, with or
# without the FM between passes.
POLISHING_TASKS = ("MILL_POLISHING", "MILL_COINCIDENT")


def reached_in(positions) -> Dict[str, List[str]]:
    """Per kind scored, the lamellae of *positions* the run took through that
    step: polishing, finished or failed. Sites and targets aren't read from
    the record, so only an item filed for them counts."""
    polished = []
    for position in positions:
        for run in getattr(position, "task_history", None) or []:
            status = getattr(getattr(run, "status", None), "name", "")
            if run.task_type in POLISHING_TASKS and status in ("Completed", "Failed"):
                polished.append(position.name)
                break
    return {"polishing": polished}


def same(name: str) -> str:
    """A grid's or lamella's name as compared: letters and digits, lower case
    ("Lamella 01" and "lamella-01" are one)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def scores(
    items: List[Dict[str, Any]], reached: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, Any]]:
    """Per kind scored: passed, failed and still open, one score per grid or
    lamella. Its first answered item is the score; a later item for the same
    one (a redo after a Reject) is counted as after_help, not again. A name in
    *reached* (per kind, the grids or lamellae the run took through that step)
    with no item is missing, and counts as a fail. A closed item without an
    answer is left out."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for item in items:
        if item.get("score") and not (item.get("closed") and not item.get("answer")):
            groups.setdefault((item["score"], same(item.get("about", ""))), []).append(
                item
            )
    found: Dict[str, Dict[str, Any]] = {}

    def tally(kind: str) -> Dict[str, Any]:
        return found.setdefault(
            kind, {"passed": 0, "failed": 0, "open": 0, "after_help": 0, "missing": []}
        )

    for (kind, _), group in groups.items():
        answered = [i for i in group if i.get("answer")]
        counts = tally(kind)
        if answered:
            choice = answered[0]["answer"].get("choice")
            counts["passed" if choice == "Accept" else "failed"] += 1
            counts["after_help"] += len(answered) - 1
        else:
            counts["open"] += 1
    for kind, names in (reached or {}).items():
        for name in names:
            if (kind, same(name)) not in groups:
                counts = tally(kind)
                counts["missing"].append(name)
                counts["failed"] += 1
    return found


def store_of(
    request: Request, experiment_path: Callable[[], Optional[str]]
) -> ReviewStore:
    """The app's review store, made on first use."""
    store = getattr(request.app.state, "agent_review", None)
    if store is None:
        store = request.app.state.agent_review = ReviewStore(
            experiment_path, files=getattr(request.app.state, "agent_files", None)
        )
    return store


def open_item(store: ReviewStore, session_id: Optional[str], item_id: str):
    """The item *item_id* of the session's list, refused when it is not
    there or no longer waits for an answer."""
    store.current(session_id)
    item = store.find(item_id)
    if item is None:
        raise refusal_error(404, "not_found", f"No review item {item_id}.")
    if item.get("answer") or item.get("closed"):
        raise refusal_error(
            409, "not_open", f"Review item {item_id} is already {state(item)}."
        )
    return item


def build_agent_review_router(
    experiment_path: Callable[[], Optional[str]],
    emit: Optional[Callable[..., None]] = None,
    reached: Optional[Callable[[], Dict[str, List[str]]]] = None,
) -> APIRouter:
    """``/app/agent/review``: the agent adds and closes items, both read
    them; the operator's answer is on the operator's router."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def _record(kind: str, payload: Dict[str, Any]) -> None:
        if emit is None:
            return
        try:
            emit(kind, payload, actor="agent")
        except Exception:  # noqa: BLE001 - the record never costs the item
            logger.debug("could not record %s", kind, exc_info=True)

    def _agent_session(request: Request) -> str:
        auth: AuthConfig = request.app.state.auth
        if is_operator_token(auth, request.headers.get("Authorization")):
            raise refusal_error(
                403, "agent_only", "The agent adds and closes review items."
            )
        found = running_session(auth)
        if found is None:
            raise refusal_error(
                409,
                "no_session",
                "No agent session is running: the review list is the session's.",
            )
        return found

    def _reached() -> Dict[str, List[str]]:
        if reached is None:
            return {}
        try:
            return reached()
        except Exception:  # noqa: BLE001 - the counts never cost the list
            logger.debug("could not read which lamellae were scored", exc_info=True)
            return {}

    @router.get("/review")
    def get_review(request: Request, show: str = "all", about: str = ""):
        session = request.app.state.auth.agent_session or {}
        store = store_of(request, experiment_path)
        items = store.current(session.get("session_id"))
        shown = [store.view(i) for i in items]
        if show != "all":
            shown = [i for i in shown if i["state"] == show]
        if about:
            shown = [i for i in shown if about.lower() in i.get("about", "").lower()]
        return {
            "session_id": store.session_id,
            "items": shown,
            "counts": store.counts(_reached()),
        }

    @router.post("/review")
    def add_review(request: Request, body: ReviewAddBody):
        session_id = _agent_session(request)
        if body.level not in LEVELS:
            raise _invalid(f"level is one of {list(LEVELS)}.")
        if len(body.text) > MAX_TEXT:
            raise _invalid(f"text is at most {MAX_TEXT} characters.")
        score = body.score.strip()
        about = _line(body.about, "about", MAX_ABOUT)
        if score:
            if score not in SCORES:
                raise _invalid(f"score is one of {list(SCORES)}.")
            if not about:
                raise _invalid(
                    "A scored item says what it scores in about: the grid for "
                    "sites, the lamella for target and polishing."
                )
            if body.options:
                raise _invalid(
                    "A scored item is answered Accept (pass) or Reject (fail): "
                    "leave options out."
                )
        cited = cited_evidence(
            SimpleNamespace(experiment_path=experiment_path()), request
        )
        item = {
            "level": body.level,
            "title": _line(body.title, "title", MAX_TITLE, True),
            "text": body.text.strip(),
            "about": about,
            "images": _images(body.images),
            "evidence": cited["evidence_id"] if cited else None,
            "facts": _facts(body.facts),
            "options": _options(body.options),
        }
        if score:
            item["score"] = score
        store = store_of(request, experiment_path)
        added = store.add(session_id, item)
        _record(
            ADDED,
            {
                "id": added["id"],
                "level": added["level"],
                "title": added["title"],
                "about": added["about"],
                "score": added.get("score", ""),
                "session_id": session_id,
            },
        )
        return store.view(added)

    @router.post("/review/{item_id}/close")
    def close_review(request: Request, item_id: str, body: ReviewCloseBody):
        session_id = _agent_session(request)
        reason = _line(body.reason, "reason", 300, True)
        store = store_of(request, experiment_path)
        open_item(store, session_id, item_id)
        closed = store.close(session_id, item_id, reason)
        _record(CLOSED, {"id": item_id, "reason": reason, "session_id": session_id})
        return store.view(closed)

    return router


def answered_text(item: Dict[str, Any], choice: str, note: str) -> str:
    """The operator's answer as the message the agent reads."""
    said = f'Review {item["id"]} "{item["title"]}": {choice}.'
    return f"{said} {note}" if note else said
