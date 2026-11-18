"""The agent's record, as routes: the event long-poll it wakes on, and the
notes and messages it puts on the record.

One router over a *source*: the App Server's app context
(`fibsem.server.app_context.AppContext`), or the agent host's
(`fibsem.server.agent.host.AgentHost`). Each source answers

- ``events(since, timeout, after=None)``: the long-poll's answer;
- ``add_note(text, item_name=None, evidence=None)``,
  ``add_message(text, item_name=None, reply_to=None, evidence=None)`` and
  ``add_evidence(item_name, save)``: the record, or a `Refusal`;
- ``experiment_path``: where evidence is kept and read from; None for none.
"""

from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Type

from fastapi import APIRouter, HTTPException, Request
from fastapi.routing import APIRoute

from fibsem.ai.agent.calls import current_call, from_headers
from fibsem.server.auth import refusal_error
from fibsem.server.bodies import EvidenceBody, NoteBody, OperatorMessageBody
from fibsem.server.responses import Events, Recorded, Refusal
from fibsem.workflows import evidence
from fibsem.workflows.events.text import note_asked, with_text

# The long-poll's ceiling: a parked poll holds a threadpool worker, so the
# transport owns it.
MAX_POLL_S = 30.0
# A note's or a message's longest text.
MAX_NOTE = 4000


class RecordSource(Protocol):
    experiment_path: Optional[str]

    def events(
        self, since: int = 0, timeout: float = 0.0, after: Optional[str] = None
    ) -> Dict[str, Any]: ...

    def add_note(
        self,
        text: str,
        item_name: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> Any: ...

    def add_message(
        self,
        text: str,
        item_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> Any: ...

    def add_evidence(
        self, item_name: Optional[str], save: Callable[[], Dict[str, Any]]
    ) -> Any: ...


def described(answer: Dict[str, Any]) -> Dict[str, Any]:
    """*answer* with each record's ``text`` and ``group`` (`events.text`)."""
    events = answer.get("events")
    if not isinstance(events, list):
        return answer
    asked: Dict[str, Any] = {}
    rows = []
    for record in events:
        note_asked(asked, record)
        rows.append(with_text(record, asked))
    return {**answer, "events": rows}


def evidence_refused(refusal: evidence.EvidenceRefused) -> HTTPException:
    return HTTPException(status_code=refusal.status_code, detail=refusal.detail)


def cited_evidence(
    source: Any, request: Optional[Request] = None
) -> Optional[Dict[str, Any]]:
    """The record of the evidence the agent's call cites
    (`calls.EVIDENCE_HEADER`); None when it cites none. The App Server names
    the call for every route; a server that doesn't (the agent host) passes
    the *request*, whose headers name it."""
    call = current_call()
    if call is None and request is not None:
        call = from_headers(request.headers)
    if call is None or not call.evidence:
        return None
    try:
        return evidence.load(lambda: source.experiment_path, call.evidence)
    except evidence.EvidenceRefused as refusal:
        raise evidence_refused(refusal)


def _note_body(body: NoteBody) -> Tuple[str, Optional[str]]:
    """A note's or a message's text and item name, or the 422 when the text
    is blank or too long."""
    if not body.text.strip() or len(body.text) > MAX_NOTE:
        raise refusal_error(
            422,
            "missing_field",
            f"Pass text (a non-empty string, at most {MAX_NOTE} characters).",
        )
    return body.text, body.item_name


def _answered(result: Any) -> Any:
    if isinstance(result, Refusal):
        raise refusal_error(
            result.status, result.error_type, result.message, **result.names
        )
    return result


def build_record_router(
    source: RecordSource, route_class: Type[APIRoute] = APIRoute
) -> APIRouter:
    """``/events``, ``/experiment/notes``, ``/experiment/messages`` and
    ``/experiment/evidence`` over
    *source*, for the including router's prefix (``/app``) and gate."""
    router = APIRouter(route_class=route_class)

    @router.get("/events", response_model=Events)
    def events(
        since: int = 0,
        timeout: float = 0.0,
        after: Optional[str] = None,
        text: bool = False,
    ):
        # A long-poll: parks up to `timeout` seconds waiting for news.
        # `after` is an event id, in place of `since`, and reaches back past
        # a restart. `text` adds what each record says, in words, for a page.
        timeout = min(max(timeout, 0.0), MAX_POLL_S)
        if after:
            answer = source.events(since=since, timeout=timeout, after=after)
        else:
            answer = source.events(since=since, timeout=timeout)
        return described(answer) if text else answer

    @router.post("/experiment/notes", response_model=Recorded)
    def add_note(request: Request, body: NoteBody):
        # Observations for the record. A POST the dashboard token is refused:
        # a watching agent notes what it sees, and a note changes nothing.
        text, item_name = _note_body(body)
        cited = cited_evidence(source, request)
        result = source.add_note(
            text,
            item_name,
            **({"evidence": cited["evidence_id"]} if cited else {}),
        )
        return _answered(result)

    @router.post("/experiment/messages", response_model=Recorded)
    def add_message(request: Request, body: OperatorMessageBody):
        # The agent's message to the operator, on the record as a note to
        # them. Like a note, it changes nothing.
        text, item_name = _note_body(body)
        cited = cited_evidence(source, request)
        result = source.add_message(
            text,
            item_name,
            body.reply_to,
            **({"evidence": cited["evidence_id"]} if cited else {}),
        )
        return _answered(result)

    @router.post("/experiment/evidence", response_model=Recorded)
    def add_evidence(body: EvidenceBody):
        # What the agent judged from, kept where the source keeps its record:
        # an image of the experiment's or one it made (uploaded, saved
        # there), with marks on it. Changes nothing, like a note.
        item_name = body.item_name

        def save() -> Dict[str, Any]:
            return evidence.save(
                lambda: source.experiment_path,
                image=body.image,
                upload=body.upload,
                marks=body.marks,
                source=body.source,
                item_name=item_name,
            )

        try:
            result = source.add_evidence(item_name, save)
        except evidence.EvidenceRefused as refusal:
            raise evidence_refused(refusal)
        return _answered(result)

    return router
