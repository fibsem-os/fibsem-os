"""What each requested action does, in the operator's words (FIB-1283).

The approval popover shows every action of a request as two lines: what it acts
on (an item and task, the stage, a beam) and the change. The server writes them
from the tool and its arguments, never from the agent's own words, so an agent
cannot word an action to look like something else. Only the request's reason is
the agent's.

A config edit shows each dotted path as old -> new, the old value read from the
same document the edit patches, when ``read`` can supply it.
"""

import json
import math
from typing import Any, Callable, Dict, List, Optional

# GET a route's JSON (the document an edit patches), or None.
Reader = Callable[[str], Optional[Dict[str, Any]]]

_MISSING = object()


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _um(metres: Any) -> str:
    try:
        return f"{float(metres) * 1e6:.3g} µm"
    except (TypeError, ValueError):
        return _short(metres)


def _deg(radians: Any) -> str:
    try:
        return f"{math.degrees(float(radians)):.3g}°"
    except (TypeError, ValueError):
        return _short(radians)


def _beam(arguments: Dict[str, Any]) -> str:
    beam = str(arguments.get("beam_type", "ELECTRON")).upper()
    return "Ion beam" if beam == "ION" else "Electron beam"


def _names(values: Any, every: str) -> str:
    if not values:
        return every
    if isinstance(values, (list, tuple)):
        return ", ".join(str(v) for v in values)
    return str(values)


def _walk(document: Any, dotted: str) -> Any:
    node = document
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return _MISSING
    return node


def _patch_lines(patch: Any, document: Optional[Dict[str, Any]]) -> str:
    if not isinstance(patch, dict) or not patch:
        return "No changes"
    lines: List[str] = []
    for path, new in patch.items():
        old = _MISSING if document is None else _walk(document, str(path))
        if old is _MISSING:
            lines.append(f"{path}: {_short(new)}")
        else:
            lines.append(f"{path}: {_short(old)} → {_short(new)}")
    return "\n".join(lines)


def _stage_relative(a: Dict[str, Any]) -> str:
    moves = [
        f"{axis} {_um(a[k])}"
        for axis, k in (("x", "dx"), ("y", "dy"), ("z", "dz"))
        if a.get(k)
    ]
    return "Move by " + (", ".join(moves) if moves else "nothing")


def _stage_absolute(a: Dict[str, Any]) -> str:
    return (
        f"Move to x {_um(a.get('x'))}, y {_um(a.get('y'))}, z {_um(a.get('z'))}, "
        f"rotation {_deg(a.get('r'))}, tilt {_deg(a.get('t'))}"
    )


def describe_action(
    tool: str, arguments: Dict[str, Any], path: str, read: Optional[Reader] = None
) -> Dict[str, str]:
    """``{"what": ..., "change": ...}`` for one action. ``path`` is the route the
    action calls; ``read`` GETs a route, for the old values of a config edit."""
    a = arguments or {}
    item, task = a.get("item_name"), a.get("task_name")

    def document(key: Optional[str]) -> Optional[Dict[str, Any]]:
        if read is None:
            return None
        try:
            answer = read(path)
        except Exception:  # noqa: BLE001 - no old value is shown, nothing more
            return None
        if not isinstance(answer, dict):
            return None
        return answer.get(key) if key else answer

    if tool == "move_stage_relative":
        return {"what": "Stage", "change": _stage_relative(a)}
    if tool == "move_stage_absolute":
        return {"what": "Stage", "change": _stage_absolute(a)}
    if tool == "move_to_milling_angle":
        return {
            "what": "Stage",
            "change": f"Tilt to the milling angle, {_short(a.get('milling_angle_deg'))}°",
        }
    if tool == "acquire_image_preview":
        return {"what": _beam(a), "change": "Take an image"}
    if tool == "last_image_preview":
        return {"what": _beam(a), "change": "Show the last image"}
    if tool == "autocontrast":
        return {"what": _beam(a), "change": "Autocontrast"}
    if tool == "set_parameter":
        return {
            "what": str(a.get("device")),
            "change": f"Set {a.get('parameter')} to {_short(a.get('value'))}",
        }
    if tool in ("call_command", "call_service_command"):
        owner = a.get("device") if tool == "call_command" else a.get("service")
        kwargs = a.get("kwargs") or {}
        extra = ", ".join(f"{k} {_short(v)}" for k, v in kwargs.items())
        return {
            "what": str(owner),
            "change": f"Run {a.get('command')}" + (f": {extra}" if extra else ""),
        }
    if tool == "update_item_task_config":
        return {
            "what": f"{item} · {task}",
            "change": _patch_lines(a.get("patch"), document("config")),
        }
    if tool == "update_protocol_task_config":
        return {
            "what": f"Protocol · {task}",
            "change": _patch_lines(a.get("patch"), document("config")),
        }
    if tool == "update_item_detail":
        return {
            "what": str(item),
            "change": _patch_lines(a.get("patch"), document(None)),
        }
    if tool == "apply_protocol_to_item":
        tasks = _names(a.get("task_names"), "every task")
        return {
            "what": str(item),
            "change": f"Copy the protocol's settings for {tasks}",
        }
    if tool == "reorder_item_milling_stages":
        return {
            "what": f"{item} · {task}",
            "change": f"Reorder {a.get('milling_key')} stages: {_short(a.get('order'))}",
        }
    if tool == "reorder_protocol_milling_stages":
        return {
            "what": f"Protocol · {task}",
            "change": f"Reorder {a.get('milling_key')} stages: {_short(a.get('order'))}",
        }
    if tool == "set_task_schedule":
        when = a.get("scheduled_at")
        return {
            "what": str(task),
            "change": f"Schedule for {when}" if when else "Clear its schedule",
        }
    if tool == "start_workflow":
        return {
            "what": "Workflow",
            "change": f"Run {_names(a.get('task_names'), 'no tasks')} on "
            f"{_names(a.get('item_names'), 'every lamella')}",
        }
    if tool == "start_grid_workflow":
        return {
            "what": "Grids",
            "change": f"Run {_names(a.get('task_names'), 'no tasks')} on "
            f"{_names(a.get('grid_names'), 'every grid')}",
        }
    if tool == "set_task_supervision":
        return {
            "what": str(task),
            "change": "Supervised: ask at each decision"
            if a.get("supervise")
            else "Automated: run without asking",
        }
    if tool == "requeue_task":
        return {
            "what": str(item),
            "change": f"Queue {task} to run again"
            + (", first" if a.get("front") else ""),
        }
    if tool == "decide_review":
        outcome = str(a.get("outcome", "")).lower()
        if outcome.startswith("reject"):
            change = f"Reject the proposal: {a.get('reason', '')}".rstrip(": ")
        else:
            change = "Confirm the proposal"
            if a.get("values"):
                change += f" with {_short(a.get('values'))}"
        return {"what": f"{a.get('item_id')} · {task}", "change": change}
    if tool == "answer_prompt":
        change = "Answer yes" if a.get("response") else "Answer no"
        if a.get("value"):
            change += f" with {_short(a.get('value'))}"
        return {"what": "The run's question", "change": change}
    return {"what": tool, "change": _short(a) if a else "Run"}
