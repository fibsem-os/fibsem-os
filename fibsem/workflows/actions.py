"""What someone did, and what they said, on the session's record.

One shape for every act, whoever did it -- an agent's call, an approved
request's step, a stop, the operator's Stop button -- and one for every note:
the agent's notes and replies, the operator's messages to it, a grid's note.
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
    that failed or was refused. Empty fields are left out."""
    payload: Dict[str, Any] = {"what": what, "ok": ok}
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
    """A note: its text, the item it is about if any, and how it threads
    (``id`` for a message to answer, ``reply_to`` for the answer)."""
    payload: Dict[str, Any] = {"text": text}
    if item_id:
        payload.update(item_id=item_id, item_name=item_name)
    payload.update({k: v for k, v in fields.items() if v is not None and v != ""})
    return payload


def record_operator_action(microscope: Any, what: str, **fields: Any) -> None:
    """The operator's act at a control, on the microscope's record."""
    with acting(OPERATOR):
        microscope.record_event(ACTION, action_record(what, **fields))
