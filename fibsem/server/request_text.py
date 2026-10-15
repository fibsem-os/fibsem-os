"""What each requested action does, in the operator's words (FIB-1283).

The approval popover shows every action of a request as two lines: what it acts
on (an item and task, the stage, a beam) and the change. The server writes them
from the tool and its arguments, never from the agent's own words, so an agent
cannot word an action to look like something else. Only the request's reason is
the agent's.

A config edit shows each dotted path as old -> new, the old value read from the
same document the edit patches, when ``read`` can supply it. Where the app can
reach the typed config (``describe``), each path reads as its form names it, in
display units: "Rough Mill 01 · trench depth 2 µm → 2.5 µm" (FIB-1403).

An action may also carry ``details``, a small table the operator can open
("Show details"): every value, so the one line never has to hold them all.
"""

import json
import math
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from fibsem.structures import FieldSpec

# GET a route's JSON (the document an edit patches), or None.
Reader = Callable[[str], Optional[Dict[str, Any]]]
# The app's own description of an action it can say better (a config edit, from
# the typed config): a partial ``{"what", "change", "details"}``, or None.
Describer = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]

# A long value in a line is cut here; the details table holds it whole.
_LINE_VALUE = 40

_MISSING = object()


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _clip(text: str, limit: int = _LINE_VALUE) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: float) -> str:
    """3 significant figures, no trailing zeros: 2.5, 20, 0.2."""
    return f"{value:.3g}"


def _distance(metres: Any) -> str:
    try:
        metres = float(metres)
    except (TypeError, ValueError):
        return _short(metres)
    if abs(metres) >= 1e-3:
        return f"{_number(metres * 1e3)} mm"
    if abs(metres) >= 1e-8 or metres == 0:
        return f"{_number(metres * 1e6)} µm"
    return f"{_number(metres * 1e9)} nm"


def _current(amps: Any) -> str:
    try:
        amps = float(amps)
    except (TypeError, ValueError):
        return _short(amps)
    if abs(amps) >= 1e-9:
        return f"{_number(amps * 1e9)} nA"
    return f"{_number(amps * 1e12)} pA"


def _seconds(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    return f"{round(seconds / 60)} min"


def _words(name: str) -> str:
    """``milling_current`` -> ``milling current``."""
    return str(name).replace("_", " ")


def _table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    return {"columns": list(columns), "rows": [[str(c) for c in row] for row in rows]}


def _kwargs_table(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not kwargs:
        return None
    return _table(
        ("Argument", "Value"), [(_words(k), _short(v)) for k, v in kwargs.items()]
    )


# --- a typed config's fields, as its form shows them ----------------------------------


def _display(value: Any, spec: Optional["FieldSpec"]) -> str:
    """``value`` as the field's form shows it: 2e-06 with scale 1e6 and unit m
    reads "2 µm"."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, bool):
        return "on" if value else "off"
    if not isinstance(value, (int, float)):
        return _short(value)
    if spec is None:
        shown, unit = value, ""
    elif spec.unit and not spec.scale and not spec.display.unit:
        # no scale declared: the prefix the value's size calls for (60 pA)
        from fibsem.utils import _get_display_unit, _get_scale_from_value

        scale = _get_scale_from_value(value)
        shown, unit = value * scale, _get_display_unit(scale, spec.unit)
    else:
        shown, unit = value * (spec.scale or 1), spec.suffix
    number = _number(shown) if isinstance(shown, float) else str(shown)
    if unit in ("°", "%"):
        return f"{number}{unit}"
    return f"{number} {unit}".rstrip()


def _child(node: Any, part: str) -> Any:
    if isinstance(node, dict):
        return node.get(part, _MISSING)
    if isinstance(node, (list, tuple)):
        return node[int(part)] if part.isdigit() and int(part) < len(node) else _MISSING
    if is_dataclass(node):
        return getattr(node, part, _MISSING)
    return _MISSING


def _shape_name(node: Any) -> Optional[str]:
    """A milling pattern's own name ("Trench"), for the field it holds."""
    from fibsem.milling.patterning.base import BasePattern

    if not isinstance(node, BasePattern):
        return None
    name = getattr(type(node), "name", None)
    if isinstance(name, str):
        return name
    return type(node).__name__.replace("Pattern", "") or None


def describe_field(root: Any, path: str) -> Tuple[str, Optional["FieldSpec"], Any]:
    """``(setting, spec, value)`` for a dotted ``path`` into the typed config
    ``root``: the setting as its form names it, under the milling stage it is in
    ("Rough Mill 01 · trench depth"), the field as its form reads it (None for
    none), and the value now. A path that doesn't resolve reads as itself."""
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.structures import field_specs

    node, parent, stage = root, None, None
    parts = str(path).split(".")
    for part in parts:
        if isinstance(node, FibsemMillingStage):
            stage = node.name
        parent, node = node, _child(node, part)
        if node is _MISSING:
            return str(path), None, _MISSING
    leaf = parts[-1]
    spec = None
    if is_dataclass(parent) and leaf in {f.name for f in dataclass_fields(parent)}:
        spec = field_specs(type(parent)).get(leaf)
    name = str((spec.label if spec is not None else None) or _words(leaf)).lower()
    shape = _shape_name(parent)
    if shape:
        name = f"{shape.lower()} {name}"
    return (
        (f"{stage} · {name}" if stage else name[:1].upper() + name[1:]),
        spec,
        node,
    )


def describe_patch(root: Any, patch: Any) -> Dict[str, Any]:
    """A config edit against the typed config ``root``: one line per setting,
    old -> new in display units, and the same as a table."""
    if not isinstance(patch, dict) or not patch:
        return {"change": "No changes"}
    lines: List[str] = []
    rows: List[Tuple[str, str, str]] = []
    for path, new in patch.items():
        setting, spec, old = describe_field(root, str(path))
        after = _display(new, spec)
        if old is _MISSING:
            lines.append(f"{setting}: {after}")
            rows.append((setting, "", after))
            continue
        before = _display(old, spec)
        lines.append(f"{setting} {before} → {after}")
        rows.append((setting, before, after))
    return {
        "change": "\n".join(lines),
        "details": _table(("Setting", "Now", "New"), rows),
    }


# --- the milling service, in the milling panel's words --------------------------------


def parse_patterns(values: Any) -> List[Any]:
    """Pattern dicts as the milling service takes them, as pattern settings;
    one that does not parse is left out."""
    from fibsem.structures import FibsemPatternSettings

    patterns = []
    for value in values or []:
        try:
            patterns.append(FibsemPatternSettings.from_dict(dict(value)))
        except Exception:  # noqa: BLE001 - the route refuses it; describe the rest
            continue
    return patterns


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _pattern_kind(pattern: Any) -> str:
    return type(pattern).__name__.replace("Fibsem", "").replace("Settings", "").lower()


def _size(pattern: Any) -> str:
    width, height = getattr(pattern, "width", None), getattr(pattern, "height", None)
    if width is not None and height is not None:
        return f"{_number(width * 1e6)} × {_number(height * 1e6)} µm"
    radius = getattr(pattern, "radius", None)
    if radius is not None:
        return f"radius {_distance(radius)}"
    return ""


def _centre(pattern: Any) -> str:
    x, y = getattr(pattern, "centre_x", None), getattr(pattern, "centre_y", None)
    if x is None or y is None:
        return ""
    return f"({x * 1e6:+.1f}, {y * 1e6:+.1f}) µm"


def _estimate(patterns: List[Any], current: Optional[float]) -> Optional[float]:
    """Seconds to mill ``patterns`` at ``current``, from the sputter-rate table."""
    if not patterns or not current:
        return None
    from fibsem.milling.base import estimate_milling_time

    try:
        return sum(estimate_milling_time(p, current) for p in patterns)
    except Exception:  # noqa: BLE001 - no estimate, nothing more
        return None


def _prepare(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    patterns = parse_patterns(kwargs.get("patterns"))
    settings = kwargs.get("settings") or {}
    current = settings.get("milling_current") if isinstance(settings, dict) else None
    voltage = settings.get("milling_voltage") if isinstance(settings, dict) else None
    kinds = {_pattern_kind(p) for p in patterns}
    if len(kinds) == 1:
        parts = [_plural(len(patterns), kinds.pop())]
    else:
        parts = [_plural(len(patterns), "pattern")]
    sizes = {_size(p) for p in patterns}
    if len(sizes) == 1 and "" not in sizes:
        parts.append(sizes.pop())
    depths = {getattr(p, "depth", None) for p in patterns}
    if len(depths) == 1 and None not in depths:
        parts.append(f"{_distance(depths.pop())} deep")
    if current is not None:
        beam = _current(current)
        if voltage:
            beam += f" at {_number(float(voltage) / 1e3)} kV"
        parts.append(beam)
    name = kwargs.get("name")
    change = (f"Prepare '{name}'" if name else "Prepare") + ": " + ", ".join(parts)
    rows = [
        (
            f"{_pattern_kind(p).capitalize()} {i + 1}",
            _size(p),
            _distance(getattr(p, "depth", None)) if hasattr(p, "depth") else "",
            _centre(p),
        )
        for i, p in enumerate(patterns)
    ]
    return {
        "what": "Milling",
        "change": change,
        "details": _table(("Pattern", "Size", "Depth", "Centre"), rows),
        "_estimate": _estimate(patterns, current),
    }


_MILLING = {
    "start": "Start milling what is drawn",
    "stop": "Stop the mill now running",
    "pause": "Pause the mill",
    "resume": "Resume the paused mill",
    "clear": "Clear the drawn patterns",
    "restore": "Put the beam back as it was before milling",
    "finish": "Finish milling: clear the patterns and put the beam back",
}


def _milling_command(command: str, kwargs: Dict[str, Any], estimate: Optional[float]):
    if command == "prepare":
        return _prepare(kwargs)
    if command == "run":
        about = f", about {_seconds(estimate)}" if estimate else ""
        return {"what": "Milling", "change": f"Run{about}"}
    if command in _MILLING:
        return {"what": "Milling", "change": _MILLING[command]}
    return None


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


def _beam_words(beam_type: Any, default: str = "ELECTRON") -> str:
    beam = str(beam_type or default).upper()
    return "the ion beam" if beam in ("ION", "2") else "the electron beam"


def _offset(dx: Any, dy: Any) -> str:
    """An image offset, x right and y up, in the operator's words."""
    parts = []
    try:
        x, y = float(dx or 0.0), float(dy or 0.0)
    except (TypeError, ValueError):
        return f"x {_short(dx)}, y {_short(dy)}"
    if x:
        parts.append(f"{_distance(abs(x))} {'right' if x > 0 else 'left'}")
    if y:
        parts.append(f"{_distance(abs(y))} {'up' if y > 0 else 'down'}")
    return ", ".join(parts) or "no offset"


def _stage_absolute(a: Dict[str, Any]) -> str:
    return (
        f"Move to x {_um(a.get('x'))}, y {_um(a.get('y'))}, z {_um(a.get('z'))}, "
        f"rotation {_deg(a.get('r'))}, tilt {_deg(a.get('t'))}"
    )


def describe_actions(
    actions: Sequence[Tuple[str, Dict[str, Any], str]],
    read: Optional[Reader] = None,
    describe: Optional[Describer] = None,
) -> List[Dict[str, Any]]:
    """Each of a request's ``(tool, arguments, path)`` described, in order. An
    action can read what came before it: a milling run says how long the
    patterns an earlier action prepares take."""
    described = []
    estimate: Optional[float] = None
    for tool, arguments, path in actions:
        line = _describe(tool, arguments, path, read, describe, estimate)
        if "_estimate" in line:
            estimate = line.pop("_estimate")
        described.append(line)
    return described


def describe_action(
    tool: str,
    arguments: Dict[str, Any],
    path: str,
    read: Optional[Reader] = None,
    describe: Optional[Describer] = None,
) -> Dict[str, Any]:
    """``{"what": ..., "change": ...}`` for one action, with ``details`` where
    there is more to show. ``path`` is the route the action calls; ``read`` GETs
    a route, for the old values of a config edit; ``describe`` is the app's own
    description, for what it can say better."""
    line = _describe(tool, arguments, path, read, describe, None)
    line.pop("_estimate", None)
    return line


def _describe(
    tool: str,
    arguments: Dict[str, Any],
    path: str,
    read: Optional[Reader],
    describe: Optional[Describer],
    estimate: Optional[float],
) -> Dict[str, Any]:
    a = arguments or {}
    item, task = a.get("item_name"), a.get("task_name")
    if describe is not None:
        try:
            own = describe(tool, a)
        except Exception:  # noqa: BLE001 - the server's own lines stand
            own = None
        if own:
            line = {"what": tool, "change": "Run"}
            line.update(_describe(tool, a, path, read, None, estimate))
            line.update(own)
            return line

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
    if tool == "safe_move_stage_absolute":
        return {"what": "Stage", "change": "Safe " + _stage_absolute(a).lower()}
    if tool == "stable_move":
        return {
            "what": "Stage",
            "change": f"Stable move in {_beam_words(a.get('beam_type'))}: "
            f"{_offset(a.get('dx'), a.get('dy'))}",
        }
    if tool == "vertical_move":
        return {
            "what": "Stage",
            "change": "Vertical move, to centre a feature "
            f"{_offset(a.get('dx'), a.get('dy'))} in {_beam_words(a.get('beam_type'), 'ION')}",
        }
    if tool == "move_to_orientation":
        return {
            "what": "Stage",
            "change": f"Move to the {str(a.get('orientation', '')).upper()} orientation",
        }
    if tool == "stop_milling":
        return {"what": "Milling", "change": _MILLING["stop"]}
    if tool == "move_to_milling_angle":
        return {
            "what": "Stage",
            "change": f"Tilt to the milling angle, {_short(a.get('milling_angle_deg'))}°"
            + (
                f", rotation {_short(a['rotation_deg'])}°"
                if a.get("rotation_deg") is not None
                else ""
            ),
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
        if not isinstance(kwargs, dict):
            kwargs = {}
        if tool == "call_service_command" and owner == "milling":
            line = _milling_command(str(a.get("command")), kwargs, estimate)
            if line is not None:
                return line
        extra = ", ".join(f"{_words(k)} {_clip(_short(v))}" for k, v in kwargs.items())
        if len(extra) > 2 * _LINE_VALUE:
            extra = _plural(len(kwargs), "argument")
        line = {
            "what": _words(owner).capitalize() if owner else "",
            "change": f"{_words(a.get('command', '')).capitalize()}"
            + (f": {extra}" if extra else ""),
        }
        details = _kwargs_table(kwargs)
        if details is not None and _short(kwargs) not in line["change"]:
            line["details"] = details
        return line
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
