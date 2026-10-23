"""What someone did, and what they said, on the session's record.

One shape for every act, whoever did it -- an agent's call, an approved
request's step, a stop, the operator's Stop button -- and one for every note:
the agent's notes, the messages between the operator and the agent, a grid's
note.
Who did it is the record's actor (``fibsem.acting``), stamped as it is written;
these build the payload.

Old records (``agent_called``, ``agent_refused``, ``agent_stop``,
``request_applied``, ``workflow_stop``, ``agent_stopped``, ``agent_note``,
``operator_message``) are still read by what reads the record.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fibsem.acting import OPERATOR, acting

ACTION = "action"
NOTE = "note"

# The reason on a stop nobody gave one for: the operator's buttons ask none, and
# an agent's stop must give one.
OPERATOR_STOP = "Stopped by the operator"


def action_record(
    what: str,
    *,
    ok: bool = True,
    reason: str = "",
    stops: Optional[str] = None,
    **fields: Any,
) -> Dict[str, Any]:
    """An act: ``what`` in the operator's words, whether it worked, and why it
    was done. ``stops`` names what a stop stopped (workflow, task, milling,
    agent). Other fields say how: tool, arguments, route, status, level, via,
    request_id and index for a request's step, error_type and message for one
    that failed or was refused. Empty fields are left out; a stop always has a
    reason, `OPERATOR_STOP` when none was given (FIB-1495)."""
    payload: Dict[str, Any] = {"what": what, "ok": ok}
    if stops and not reason:
        reason = OPERATOR_STOP
    if reason:
        payload["reason"] = reason
    if stops:
        payload["stops"] = stops
    payload.update({k: v for k, v in fields.items() if v is not None and v != ""})
    return payload


def note_record(
    text: str,
    *,
    item_id: Optional[str] = None,
    item_name: Optional[str] = None,
    **fields: Any,
) -> Dict[str, Any]:
    """A note: its text, the item it is about if any, and for a message who it
    is ``to`` (`OPERATOR` or `AGENT`), its ``id`` and the ``reply_to`` it answers."""
    payload: Dict[str, Any] = {"text": text}
    if item_id:
        payload.update(item_id=item_id, item_name=item_name)
    payload.update({k: v for k, v in fields.items() if v is not None and v != ""})
    return payload


def is_message(record: Dict[str, Any]) -> bool:
    """Whether the note *record* is a message between the operator and the
    agent rather than a note for the record. Older records mark no ``to``: the
    operator's message had an ``id``, the agent's a ``reply_to``."""
    p = record.get("payload") or {}
    return bool(
        p.get("to")
        or p.get("reply_to")
        or (record.get("actor") == OPERATOR and p.get("id"))
    )


def record_operator_action(microscope: Any, what: str, **fields: Any) -> None:
    """The operator's act at a control, on the microscope's record."""
    with acting(OPERATOR):
        microscope.record_event(ACTION, action_record(what, **fields))
