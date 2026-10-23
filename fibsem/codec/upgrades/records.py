"""The records an experiment keeps on its items, as older versions wrote them, in
the current shape.

- A verdict was a ``DefectState`` (``state``, ``description``,
  ``last_completed_task``) and, before that, ``has_defect`` / ``requires_rework``
  flags. Its names ``NONE``, ``FAILURE`` and ``POOR`` are ``UNASSESSED`` and
  ``FAILED``. A grid's verdict that is a bare name reads as unassessed.
- A task run named its item ``lamella_id``. ``AwaitingDecision`` (0.6 development
  builds) was a run that had finished.
- An item's proposals were keyed by the task that asked them, and a decision
  named who made it ``author``.

Presence, not a version, decides each step. Each step copies what it is given.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# The verdict names older files wrote.
_OLD_VERDICTS = {"NONE": "UNASSESSED", "FAILURE": "FAILED", "POOR": "FAILED"}


def quality(data: Union[Dict[str, Any], str, None]) -> Dict[str, Any]:
    """A ``QualityRecord`` from any version, in the current shape."""
    if not isinstance(data, dict):
        return {}
    data = dict(data)
    if "has_defect" in data:
        if not data.pop("has_defect"):
            verdict = "UNASSESSED"
        elif data.pop("requires_rework", False):
            verdict = "REWORK"
        else:
            verdict = "FAILED"
        data.setdefault("verdict", verdict)
    for old, new in (
        ("state", "verdict"),
        ("description", "reason"),
        ("last_completed_task", "at_task"),
    ):
        if old in data:
            data.setdefault(new, data.pop(old))
    verdict = data.get("verdict")
    if isinstance(verdict, str) and verdict in _OLD_VERDICTS:
        data["verdict"] = _OLD_VERDICTS[verdict]
    return data


def task_state(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A ``TaskState`` from any version, in the current shape."""
    data = dict(data or {})
    if "lamella_id" in data:
        data.setdefault("item_id", data.pop("lamella_id"))
    if data.get("status") == "AwaitingDecision":
        data["status"] = "Completed"
    return data


def proposals(
    data: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], None],
) -> List[Dict[str, Any]]:
    """An item's proposals as a list, each saying who asked it."""
    if isinstance(data, dict):
        data = [
            {**proposal, "asked_by": proposal.get("asked_by") or f"task:{name}"}
            for name, saved in data.items()
            for proposal in saved
        ]
    return [proposal(p) for p in data or []]


def proposal(data: Dict[str, Any]) -> Dict[str, Any]:
    """A ``Proposal`` from any version, in the current shape."""
    if not data.get("decisions"):
        return data
    return {**data, "decisions": [decision(d) for d in data["decisions"]]}


def decision(data: Dict[str, Any]) -> Dict[str, Any]:
    """A ``Decision`` from any version, in the current shape."""
    if "author" not in data:
        return data
    data = dict(data)
    data.setdefault("actor", data.pop("author"))
    return data


def item(data: Dict[str, Any], verdict: str = "quality") -> Dict[str, Any]:
    """An item's records (its verdict under ``verdict``, its runs and its
    proposals) from any version, in the current shape."""
    data = dict(data)
    if verdict in data:
        data[verdict] = quality(data[verdict])
    if "task_state" in data:
        data["task_state"] = task_state(data["task_state"])
    if "task_history" in data:
        data["task_history"] = [task_state(t) for t in data["task_history"] or []]
    if "proposals" in data:
        data["proposals"] = proposals(data["proposals"])
    return data
