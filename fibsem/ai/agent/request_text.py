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

import math
import os
import re
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from fibsem.structures import Parameter

# GET a route's JSON (the document an edit patches), or None.
Reader = Callable[[str], Optional[Dict[str, Any]]]
# The app's own description of an action it can say better (a config edit, from
# the typed config): a partial ``{"what", "change", "details"}``, or None.
Describer = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]

# A long value in a line is cut here; the details table holds it whole.
_LINE_VALUE = 40

_MISSING = object()


def _short(value: Any) -> str:
    """A value as words: never JSON, which an operator should not have to read."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        return value
    if value is None:
        return "none"
    if isinstance(value, dict):
        return ", ".join(f"{_words(k)} {_short(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_short(v) for v in value)
    return str(value)


def _clip(text: str, limit: int = _LINE_VALUE) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: float) -> str:
    """3 significant figures, no trailing zeros: 2.5, 20, 0.2, −3."""
    return _minus(f"{value:.3g}")


def _minus(text: str) -> str:
    """A typeset minus sign, as the operator reads one: −3, not -3."""
    return "−" + text[1:] if text.startswith("-") else text


def _coordinate(metres: Any) -> str:
    """A place on the stage to 0.1 µm, which 3 significant figures of a
    millimetre lose: 1.2345 mm, −9.5 µm."""
    try:
        metres = float(metres)
    except (TypeError, ValueError):
        return _short(metres)
    if abs(metres) >= 1e-3:
        text, unit = f"{metres * 1e3:.4f}", "mm"
    else:
        text, unit = f"{metres * 1e6:.1f}", "µm"
    text = text.rstrip("0").rstrip(".")
    if float(text) == 0:
        text = "0"
    return f"{_minus(text)} {unit}"


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


def _display(value: Any, spec: Optional["Parameter"]) -> str:
    """``value`` as the field's form shows it: 2e-06 with scale 1e6 and unit m
    reads "2 µm". No value reads "(none)"."""
    if value is None:
        return "(none)"
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, bool):
        return "on" if value else "off"
    if not isinstance(value, (int, float)):
        return _short(value)
    if spec is None:
        shown, unit = value, ""
    elif spec.unit and not spec.scale and not spec._shown.unit:
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


def describe_field(root: Any, path: str) -> Tuple[str, Optional["Parameter"], Any]:
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
        # A switch reads "Re-align during milling: off → on".
        joint = ": " if isinstance(old, bool) else " "
        lines.append(f"{setting}{joint}{before} → {after}")
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


def _past_field(patterns: List[Any], settings: Any) -> str:
    """ "1 of them past the 150 µm field": the patterns a real instrument would
    refuse, on the step's line and not only in the picture's legend. The field
    is an image's 2:3, as the picture checks it."""
    from fibsem.milling.patterning.utils import outside_field

    try:
        hfw = float(settings["hfw"])
    except (KeyError, TypeError, ValueError):
        return ""
    past = [p for p in patterns if outside_field(p, hfw, hfw * 2 / 3)]
    if not past:
        return ""
    which = "past" if len(patterns) == 1 else f"{len(past)} of them past"
    return f"{which} the {_distance(hfw)} field"


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
    past = _past_field(patterns, settings)
    if past:
        parts.append(past)
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
        "_estimate": (_estimate(patterns, current), current),
    }


_MILLING = {
    "start": "Start milling what is drawn",
    "stop": "Stop the mill now running",
    "end": "End the milling stage now running, keeping what it milled",
    "pause": "Pause the mill",
    "resume": "Resume the paused mill",
    "clear": "Clear the drawn patterns",
    "restore": "Put the beam back as it was before milling",
    "finish": "Finish milling: clear the patterns and put the beam back",
}

# Whose estimate a mill's time is: the sputter-rate table's, at the mill's
# current, not the instrument's (FIB-1412).
_ESTIMATED = "estimated from the sputter rate"


def _stages(values: Any) -> List[Any]:
    """Milling stage dicts as stages; one that does not parse is left out."""
    from fibsem.milling.base import FibsemMillingStage

    stages = []
    for value in values or []:
        try:
            stages.append(FibsemMillingStage.from_dict(dict(value)))
        except Exception:  # noqa: BLE001 - the route refuses it; describe the rest
            continue
    return stages


def _stage_rows(stages: List[Any]) -> Dict[str, Any]:
    rows = []
    for stage in stages:
        pattern = stage.pattern
        point = getattr(pattern, "point", None)
        depth = getattr(pattern, "depth", None)
        rows.append(
            (
                stage.name,
                str(getattr(pattern, "name", "") or type(pattern).__name__),
                _current(stage.milling.milling_current),
                _distance(depth) if depth is not None else "",
                (
                    f"({point.x * 1e6:+.1f}, {point.y * 1e6:+.1f}) µm"
                    if point is not None
                    else ""
                ),
            )
        )
    return _table(("Stage", "Pattern", "Current", "Depth", "Position"), rows)


def _stages_estimate(stages: List[Any]) -> Optional[float]:
    try:
        return sum(stage.estimated_time for stage in stages)
    except Exception:  # noqa: BLE001 - no estimate, nothing more
        return None


def _names_cut(names: List[str], shown: int = 3) -> str:
    if len(names) <= shown:
        return ", ".join(names)
    return ", ".join(names[:shown]) + " …"


def _run_task(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A milling task, by its stages: what each mills, at what current, where."""
    config = kwargs.get("config")
    config = config if isinstance(config, dict) else {}
    name = config.get("name") or "milling task"
    stages = [s for s in _stages(config.get("stages")) if s.enabled]
    if not stages:
        raw = [s.get("name") for s in config.get("stages") or [] if isinstance(s, dict)]
        listed = f": {_plural(len(raw), 'stage')}" if raw else ""
        return {"what": "Milling", "change": f"Run task '{name}'{listed}"}
    names = [stage.name for stage in stages]
    change = f"Run task '{name}': {_plural(len(stages), 'stage')} ({_names_cut(names)})"
    estimate = _stages_estimate(stages)
    if estimate:
        change += f", about {_seconds(estimate)} ({_ESTIMATED})"
    return {"what": "Milling", "change": change, "details": _stage_rows(stages)}


def _run_stage(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    stages = _stages([kwargs.get("stage")])
    if not stages:
        stage = kwargs.get("stage")
        name = stage.get("name") if isinstance(stage, dict) else None
        return {"what": "Milling", "change": f"Run stage '{name or 'milling stage'}'"}
    stage = stages[0]
    pattern = stage.pattern
    change = (
        f"Run stage '{stage.name}': "
        f"{str(getattr(pattern, 'name', '') or 'pattern').lower()}, "
        f"{_current(stage.milling.milling_current)}"
    )
    estimate = _stages_estimate(stages)
    if estimate:
        change += f", about {_seconds(estimate)} ({_ESTIMATED})"
    return {"what": "Milling", "change": change, "details": _stage_rows(stages)}


def _milling_command(command: str, kwargs: Dict[str, Any], estimate):
    if command == "prepare":
        return _prepare(kwargs)
    if command == "run":
        seconds, current = estimate or (None, None)
        about = ""
        if seconds:
            at = f" at {_current(current)}" if current else ""
            about = f", about {_seconds(seconds)} ({_ESTIMATED}{at})"
        return {"what": "Milling", "change": f"Run{about}"}
    if command == "run_task":
        return _run_task(kwargs)
    if command == "run_stage":
        return _run_stage(kwargs)
    if command in _MILLING:
        return {"what": "Milling", "change": _MILLING[command]}
    return None


def _deg(radians: Any) -> str:
    try:
        return f"{_number(math.degrees(float(radians)))}°"
    except (TypeError, ValueError):
        return _short(radians)


def _beam(arguments: Dict[str, Any]) -> str:
    beam = str(arguments.get("beam_type", "ELECTRON")).upper()
    return "Ion beam" if beam == "ION" else "Electron beam"


def _beam_short(beam_type: Any) -> str:
    return "FIB" if str(beam_type or "ELECTRON").upper() in ("ION", "2") else "SEM"


def _options_line(a: Dict[str, Any]) -> str:
    """The workflow options a call sets, in one line."""
    parts = []
    if "show_run_summary" in a and a["show_run_summary"] is not None:
        parts.append(
            "Open the run summary when a run ends"
            if a["show_run_summary"]
            else "Don't open the run summary when a run ends; a toast says it ended"
        )
    if "turn_beams_off" in a and a["turn_beams_off"] is not None:
        parts.append(
            "Turn the beams off when a run ends"
            if a["turn_beams_off"]
            else "Leave the beams on when a run ends"
        )
    return "; ".join(parts) or "No change"


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


def _item_setting(path: str) -> str:
    """An item's dotted path as words: "poi.x" reads "POI x"."""
    words = _words(str(path).replace(".", " "))
    return "POI" + words[3:] if words.startswith("poi") else words


def _item_value(path: str, value: Any) -> str:
    """An item's value in display units: the POI in metres reads in µm, the
    alignment area's fractions of the frame as percent."""
    if isinstance(value, dict):
        return ", ".join(
            f"{_words(k)} {_item_value(f'{path}.{k}', v)}" for k, v in value.items()
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parts = str(path).split(".")
        if "area" in str(path):
            return f"{_number(value * 100)}%"
        if parts[-1] in ("x", "y", "z") or "poi" in parts or "point" in parts:
            return _distance(value)
    return _short(value)


# An item's places, each one line however many of its fields the edit sets:
# (field, words, the fields in the order they read).
_ITEM_PLACES = (
    ("poi", "Point of interest", ("x", "y")),
    ("alignment_area", "Alignment area", ("left", "top", "width", "height")),
)


def _item_patch_lines(patch: Any, document: Optional[Dict[str, Any]]) -> str:
    """An item edit as lines: its point of interest and its alignment area one
    line each ("Point of interest: (1, 0) µm → (2, 0) µm"), however the edit
    sets them; anything else a line per setting."""
    if not isinstance(patch, dict) or not patch:
        return "No changes"
    flat: Dict[str, Any] = {}
    for path, new in patch.items():
        if isinstance(new, dict) and any(path == f for f, _, _ in _ITEM_PLACES):
            flat.update({f"{path}.{k}": v for k, v in new.items()})
        else:
            flat[str(path)] = new
    lines: List[str] = []
    for field, words, parts in _ITEM_PLACES:
        if not any(f"{field}.{p}" in flat for p in parts):
            continue
        old = {p: _walk(document or {}, f"{field}.{p}") for p in parts}
        new = {p: flat.pop(f"{field}.{p}", old[p]) for p in parts}
        unit = " µm" if field == "poi" else ""
        line = f"{words}: {_item_place(field, new)}{unit}"
        if all(v is not _MISSING for v in old.values()):
            before = (
                "(none)"
                if all(v is None for v in old.values())
                else _item_place(field, old)
            )
            line = f"{words}: {before} → {_item_place(field, new)}{unit}"
        lines.append(line)
    rest = _patch_lines(flat, document, _item_setting, _item_value) if flat else ""
    return "\n".join(lines + ([rest] if rest else []))


def _item_place(field: str, values: Dict[str, Any]) -> str:
    """A point of interest as "(+1, −3.2)", in µm from the image's centre to
    0.1 µm; an alignment area as "30 × 20% at (25, 50)%". A value not known
    reads "?"."""

    def number(value: Any, scale: float) -> str:
        try:
            return _number(float(value) * scale)
        except (TypeError, ValueError):
            return "?"

    def offset(value: Any) -> str:
        try:
            text = f"{float(value) * 1e6:+.1f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return "?"
        return "0" if float(text) == 0 else _minus(text)

    if field == "poi":
        return f"({offset(values['x'])}, {offset(values['y'])})"
    return (
        f"{number(values['width'], 100)} × {number(values['height'], 100)}% at "
        f"({number(values['left'], 100)}, {number(values['top'], 100)})%"
    )


def _patch_lines(
    patch: Any,
    document: Optional[Dict[str, Any]],
    name: Callable[[str], str] = str,
    value: Callable[[str, Any], str] = lambda _path, v: _short(v),
) -> str:
    if not isinstance(patch, dict) or not patch:
        return "No changes"
    lines: List[str] = []
    for path, new in patch.items():
        old = _MISSING if document is None else _walk(document, str(path))
        after = value(str(path), new)
        if old is _MISSING:
            lines.append(f"{name(str(path))}: {after}")
        else:
            before = "(none)" if old is None else value(str(path), old)
            lines.append(f"{name(str(path))}: {before} → {after}")
    return "\n".join(lines)


def _values_text(values: Any) -> str:
    """A decision's or an answer's values as words: {"poi": {"x", "y"}} reads
    "POI x 1.2 µm, y -3 µm"."""
    if isinstance(values, dict):
        return ", ".join(
            f"{_item_setting(k)} {_item_value(k, v)}" for k, v in values.items()
        )
    return _short(values)


def _stage_relative(a: Dict[str, Any], now: Optional[Dict[str, Any]] = None) -> str:
    """ "Move 2.5 µm in x, from x 8 µm": by how much along which stage axis, and
    from where on that axis when the stage's position is known."""
    axes = [
        (axis, a[k]) for axis, k in (("x", "dx"), ("y", "dy"), ("z", "dz")) if a.get(k)
    ]
    if not axes:
        return "Move nothing"
    text = "Move " + ", ".join(f"{_minus(_distance(v))} in {axis}" for axis, v in axes)
    if now:
        known = [
            f"{axis} {_coordinate(now[axis])}"
            for axis, _ in axes
            if now.get(axis) is not None
        ]
        if known:
            text += ", from " + ", ".join(known)
    return text


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


def _position(p: Any) -> str:
    """A stage position in metres and radians, in display units."""
    if not isinstance(p, dict):
        return _short(p)
    parts = [
        f"{axis} {_coordinate(p[axis])}" for axis in "xyz" if p.get(axis) is not None
    ]
    for axis, word in (("t", "tilt"), ("r", "rotation")):
        if p.get(axis) is not None:
            parts.append(f"{word} {_deg(p[axis])}")
    return ", ".join(parts)


def _dwell(seconds: Any) -> str:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return _short(seconds)
    if seconds >= 1e-3:
        return f"{_number(seconds * 1e3)} ms"
    if seconds >= 1e-6:
        return f"{_number(seconds * 1e6)} µs"
    return f"{_number(seconds * 1e9)} ns"


def _resolution(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]} × {value[1]}"
    return _short(value)


# A fresh image's settings when the agent leaves one out: the tool's defaults,
# not the beam's current values (catalog: beam_acquire_image).
_IMAGE_DEFAULTS = {"hfw": 150e-6, "resolution": [1536, 1024], "dwell_time": 1e-6}


# An image's settings as the imaging panel labels them, in display units.
_IMAGE_LABELS = {
    "beam_type": "Beam",
    "hfw": "Field of view",
    "resolution": "Resolution",
    "dwell_time": "Dwell time",
    "autocontrast": "Autocontrast",
    "autogamma": "Auto gamma",
    "save": "Save",
    "path": "Saved in",
    "filename": "File name",
    "line_integration": "Line integration",
    "frame_integration": "Frame integration",
    "scan_interlacing": "Scan interlacing",
    "drift_correction": "Drift correction",
    "reduced_area": "Reduced area",
    "scan_rotation": "Scan rotation",
    "nrows": "Rows",
    "ncols": "Columns",
    "overlap": "Overlap",
}

# The beam's settings an image changes, by the beam device's parameter names.
_BEAM_PARAMETERS = ("hfw", "resolution", "dwell_time")


def _image_value(key: str, value: Any) -> str:
    if key == "hfw":
        return _distance(value)
    if key == "dwell_time":
        return _dwell(value)
    if key == "resolution":
        return _resolution(value)
    if key == "scan_rotation":
        return _deg(value)
    if key == "beam_type":
        return _beam_short(value)
    if key == "overlap" and isinstance(value, (int, float)):
        return f"{_number(value * 100)}%"
    if key == "path":
        return _folder(value)
    return _short(value)


def _image_label(key: str) -> str:
    return _IMAGE_LABELS.get(key) or _words(key).capitalize()


def _folder(path: Any) -> str:
    """A folder as its last part: the experiment's folders are long."""
    text = str(path or "").rstrip("/\\").replace("\\", "/")
    return os.path.basename(text) or text or "none"


def _image_settings(settings: Dict[str, Any]) -> str:
    """An image's field of view, resolution and dwell, as one phrase."""
    hfw = settings.get("hfw")
    resolution = settings.get("resolution")
    dwell = settings.get("dwell_time")
    return ", ".join(
        part
        for part in (
            f"{_distance(hfw)} wide" if hfw is not None else "",
            _resolution(resolution) if resolution is not None else "",
            f"{_dwell(dwell)} dwell" if dwell is not None else "",
        )
        if part
    )


def _image_rows(settings: Dict[str, Any], skip: Sequence[str] = ()) -> List[tuple]:
    """An image's settings as (label, value) rows, in display units."""
    return [
        (_image_label(key), _image_value(key, value))
        for key, value in settings.items()
        if key not in skip and value is not None
    ]


def _acquire_image(a: Dict[str, Any], get: Callable[[str], Any]) -> Dict[str, Any]:
    settings = a.get("image_settings")
    if not isinstance(settings, dict) or not settings:
        return {"what": _beam(a), "change": "Take an image, as the beam is set now"}
    settings = {**_IMAGE_DEFAULTS, **settings}
    beam = "ion" if str(a.get("beam_type", "")).upper() == "ION" else "electron"
    rows = []
    for key, value in settings.items():
        if key == "beam_type" or value is None:
            continue
        new = _image_value(key, value)
        now = (
            get(f"/microscope/devices/{beam}/{key}")
            if key in _BEAM_PARAMETERS
            else None
        )
        if isinstance(now, dict) and "value" in now:
            rows.append((_image_label(key), _image_value(key, now["value"]), new))
        else:
            rows.append((_image_label(key), "", new))
    return {
        "what": _beam(a),
        # The caveat on a line of its own: a line about it after the fact
        # (Recent) reads the first only.
        "change": f"Take an image: {_image_settings(settings)}\n"
        "These settings stay on the beam",
        "details": _table(("Setting", "Now", "New"), rows),
    }


def _overview(a: Dict[str, Any]) -> Dict[str, Any]:
    settings = a.get("settings") if isinstance(a.get("settings"), dict) else {}
    image = settings.get("image_settings") or {}
    rows, cols = settings.get("nrows"), settings.get("ncols")
    beam = _beam_short(image.get("beam_type"))
    hfw = image.get("hfw")
    grid = f"{rows} × {cols} " if rows and cols else ""
    change = f"Take a {grid}{beam} overview"
    size = _overview_size(settings)
    if hfw is not None and size:
        change += f", {_distance(hfw)} tiles ({size})"
    change += ", here"
    table = _image_rows(
        {
            k: v
            for k, v in settings.items()
            if not isinstance(v, dict) and k != "tile_mask"
        }
    ) + _image_rows(image)
    if a.get("path"):
        table.append(
            ("Saved as", f"{a.get('name') or 'overview'} in {_folder(a['path'])}")
        )
    return {
        "what": "Overview",
        "change": change,
        "details": _table(("Setting", "Value"), table),
    }


def _overview_size(settings: Dict[str, Any]) -> Optional[str]:
    """What an overview covers, as its estimate says: "about 420 × 280 µm"."""
    from fibsem.structures import OverviewAcquisitionSettings

    try:
        parsed = OverviewAcquisitionSettings.from_dict(settings)
        across, down = float(parsed.total_fov_x), float(parsed.total_fov_y)
    except Exception:  # noqa: BLE001 - a partial settings dict says less
        return None
    width, height = _distance(across), _distance(down)
    if width.split()[-1] == height.split()[-1]:
        width = width.split()[0]
    return f"about {width} × {height}"


def _focus_stack(a: Dict[str, Any]) -> Dict[str, Any]:
    image = a.get("image_settings") if isinstance(a.get("image_settings"), dict) else {}
    steps = a.get("n_steps") or 3
    change = f"Take an image focused in {steps} strips"
    if image.get("hfw") is not None:
        change += f", {_distance(image['hfw'])} wide"
    if a.get("auto_focus", True):
        change += ", each strip focused first"
    return {"what": _beam(image), "change": change}


def _auto_focus(a: Dict[str, Any]) -> Dict[str, Any]:
    """The sweep's passes, as the operator reads them: "50 µm in 5 µm steps"."""
    if str(a.get("method") or "sweep").lower() == "instrument":
        return {"what": _beam(a), "change": "Autofocus with the instrument's routine"}
    settings = a.get("settings") if isinstance(a.get("settings"), dict) else {}
    passes = settings.get("passes")
    if not isinstance(passes, list) or not passes:
        from fibsem.autofunctions.autofocus import AutoFocusSettings

        passes = [vars(p) for p in AutoFocusSettings().passes]
    steps = [
        f"{_distance(p.get('search_range'))} in {_distance(p.get('step_size'))} steps"
        for p in passes
        if isinstance(p, dict) and p.get("enabled", True)
    ]
    change = "Focus by sweeping the working distance"
    if steps:
        change += ": " + ", then ".join(steps)
    if a.get("hfw") is not None:
        change += f", images {_distance(a['hfw'])} wide"
    return {"what": _beam(a), "change": change}


def _channel_setting(key: str, value: Any) -> str:
    """One ChannelSettings field as the FM panel shows it: wavelengths in nm
    (as stored), the emission filter by name, exposure in ms, power and gain as
    % of full."""
    if isinstance(value, dict) and "name" in value:
        return f"{_words(key)} {_short(value['name'])}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{_words(key)} {_short(value)}"
    if key.endswith("wavelength"):
        shown = f"{_number(number)} nm"
    elif key == "exposure_time":
        shown = f"{_number(number * 1e3)} ms"
    elif key in ("power", "gain"):
        shown = f"{_number(number * 100)}%"
    else:
        shown = _short(value)
    return f"{_words(key).replace('time', '').strip()} {shown}"


def _z_range(z: Dict[str, Any]) -> str:
    """A z-stack's range about the objective, "−10 µm to +10 µm in 1 µm steps
    (21 planes)"; "" when the call doesn't say."""
    try:
        zmin, zmax, zstep = float(z["zmin"]), float(z["zmax"]), float(z["zstep"])
    except (KeyError, TypeError, ValueError):
        return ""
    sign = "" if zmax < 0 else "+"
    text = f"{_minus(_distance(zmin))} to {sign}{_minus(_distance(zmax))}"
    text += f" in {_distance(zstep)} steps"
    if zstep > 0 and zmax > zmin:
        text += f" ({int(round((zmax - zmin) / zstep)) + 1} planes)"
    return text


def _fm_channels(a: Dict[str, Any], change: str) -> Dict[str, Any]:
    """*change* in the channels the call names, with their settings as details."""
    channels = [c for c in a.get("channels") or [] if isinstance(c, dict)]
    names = [str(c.get("name") or "channel") for c in channels]
    change += f" in {_plural(len(channels), 'channel')}"
    if names:
        change += f" ({_names_cut(names)})"
    change += ", here"
    table = [
        (
            str(c.get("name") or ""),
            ", ".join(_channel_setting(k, v) for k, v in c.items() if k != "name"),
        )
        for c in channels
    ]
    return {
        "what": "Fluorescence",
        "change": change,
        "details": _table(("Channel", "Settings"), table) if table else None,
    }


def _fm_overview(a: Dict[str, Any]) -> Dict[str, Any]:
    parameters = a.get("parameters") if isinstance(a.get("parameters"), dict) else {}
    rows, cols = parameters.get("rows"), parameters.get("cols")
    grid = f"{rows} × {cols} " if rows and cols else ""
    return _fm_channels(a, f"Take a {grid}overview")


def _fm_image(a: Dict[str, Any]) -> Dict[str, Any]:
    zparams = a.get("zparams") if isinstance(a.get("zparams"), dict) else None
    if not zparams:
        return _fm_channels(a, "Take an image")
    z = _z_range(zparams)
    return _fm_channels(a, f"Take a z-stack, {z}," if z else "Take a z-stack")


_ALIGN_WITH = {
    "beam-shift": "the beam shift",
    "stage": "stage moves",
    "stage-vertical": "vertical stage moves",
}


def _parameter_spec(info: Any) -> Optional["Parameter"]:
    """A device parameter's listing (``describe_device``) as a ``Parameter``:
    its unit and how it is shown."""
    if not isinstance(info, dict):
        return None
    from fibsem.structures import Display, Parameter

    display = info.get("display")
    shown = None
    if isinstance(display, dict) and {"label", "scale", "unit"} & set(display):
        known = {f.name for f in dataclass_fields(Display)}
        try:
            shown = Display(**{k: v for k, v in display.items() if k in known})
        except TypeError:
            shown = None
    return Parameter(unit=info.get("unit"), display=shown)


def _set_parameter(
    a: Dict[str, Any], owner_key: str, path: str, get: Callable[[str], Any]
) -> Dict[str, Any]:
    owner, parameter = a.get(owner_key), str(a.get("parameter"))
    if parameter == "on" and isinstance(a.get("value"), bool):
        # "Electron · Set on to off" said nothing (sim session 2026-10-16).
        name = _words(owner) + (" beam" if owner in ("electron", "ion") else "")
        state = "on" if a["value"] else "off"
        return {
            "what": _words(owner).capitalize(),
            "change": f"Turn the {name} {state}",
        }
    spec = None
    listing = get(path.rsplit("/", 1)[0]) if "/" in path else None
    if isinstance(listing, dict):
        spec = _parameter_spec((listing.get("parameters") or {}).get(parameter))
    name = (spec.label if spec is not None else None) or _words(parameter)
    new = _display(a.get("value"), spec)
    current = get(path)
    if isinstance(current, dict) and "value" in current:
        change = f"Set {name} {_display(current['value'], spec)} → {new}"
    else:
        change = f"Set {name} to {new}"
    return {"what": _words(owner).capitalize() if owner else "", "change": change}


def _item_route(item: Any) -> str:
    from urllib.parse import quote

    return f"/app/experiment/items/{quote(str(item), safe='')}"


def _pose_of(detail: Any, pose: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(detail, dict):
        return None
    position = (detail.get("poses") or {}).get(pose)
    return position if isinstance(position, dict) else None


def _overview_name(overview: Dict[str, Any]) -> str:
    """ "the Grid-01 overview (04:58)": whose, and when it was taken, which the
    file's stamp says; the file name only when it has no stamp."""

    grid = overview.get("grid_name")
    filename = str(overview.get("filename") or "")
    stamp = re.search(r"(\d{2})-(\d{2})-\d{2}", os.path.basename(filename))
    if stamp:
        name = f"the {grid} overview" if grid else "the overview"
        return f"{name} ({stamp.group(1)}:{stamp.group(2)})"
    name = filename or "the overview"
    return f"{name} of {grid}" if grid else name


def _add_item(a: Dict[str, Any], stage_now: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    named = f" '{a['name']}'" if a.get("name") else ""
    overview = a.get("overview")
    position = a.get("stage_position")
    if isinstance(overview, dict):
        where = _overview_name(overview)
        if overview.get("fraction") is not None:
            fx, fy = (list(overview["fraction"]) + [None, None])[:2]
            at = f"{_number(float(fx) * 100)}% across, {_number(float(fy) * 100)}% down"
        else:
            at = f"pixel {_short(overview.get('px'))}"
        change = f"Add{named} on {where}, {at}"
    elif isinstance(position, dict):
        change = f"Add{named} at {_position(position)}"
    else:
        change = f"Add{named} at the stage position now"
        if stage_now:
            change += f": {_position(stage_now)}"
    return {"what": "Lamella", "change": change}


def _open_experiment(get) -> Optional[str]:
    """The name of the experiment open now, or None."""
    status = get("/app/workflow/status")
    current = (status or {}).get("experiment") if isinstance(status, dict) else None
    return (current or {}).get("name") if isinstance(current, dict) else None


def _experiment_name(path: str) -> str:
    """An experiment's name from its path: the folder, not experiment.yaml."""
    path = path.rstrip("/\\")
    name = os.path.basename(path.replace("\\", "/"))
    if os.path.splitext(name)[1].lower() in (".yaml", ".yml"):
        name = os.path.basename(os.path.dirname(path.replace("\\", "/")))
    return name or "an experiment"


def _recent_experiment_about(get, path: str) -> Optional[str]:
    """'5 lamellae, created 15 Oct' for a path on the recents list, else None."""
    recent = get("/app/experiment/recent")
    items = recent.get("items") if isinstance(recent, dict) else None
    wanted = path.replace("\\", "/").rstrip("/")
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        entry_path = str(entry.get("path") or "").replace("\\", "/").rstrip("/")
        if wanted not in (entry_path, os.path.dirname(entry_path)):
            continue
        count = entry.get("num_items")
        parts = []
        if isinstance(count, int):
            parts.append(f"{count} lamell{'a' if count == 1 else 'ae'}")
        try:
            created = datetime.fromisoformat(str(entry.get("created_at")))
            parts.append(f"created {created.day} {created:%b}")
        except ValueError:
            pass
        return ", ".join(parts) or None
    return None


def _proposal_item(proposals: Any, item_id: Any) -> Optional[str]:
    if not isinstance(proposals, dict):
        return None
    for proposal in proposals.get("proposals") or []:
        if isinstance(proposal, dict) and proposal.get("item_id") == item_id:
            return proposal.get("item_name")
    return None


# --- grids on the hardware: what an exchange will do ---------------------------------

_SAMPLE_HANDLING = "/microscope/services/sample_handling/queries/"


def _result(answer: Any) -> Any:
    return answer.get("result") if isinstance(answer, dict) else None


def _loaded(inventory: List[Dict[str, Any]]) -> Optional[str]:
    """The grid on the stage now, by name."""
    return next((e.get("name") for e in inventory if e.get("state") == "loaded"), None)


def _grid_load(name: str, inventory: Any) -> str:
    """``Load grid 'G2' from Slot-02 (unloads 'G1' first)``: the exchange the
    loader will make, from the inventory as it reads now."""
    change = f"Load grid '{name}'"
    if not isinstance(inventory, list) or not inventory:
        return change
    entry = next((e for e in inventory if e.get("name") == name), None)
    if entry is None:
        return change + ": it is not in the inventory"
    if entry.get("source") != "magazine":
        return change + f": it is in {entry.get('slot_name')} already, nothing moves"
    if entry.get("state") == "loaded":
        return change + ": it is on the stage already, nothing moves"
    change += f" from {entry.get('slot_name')}"
    current = _loaded(inventory)
    return change + (f" (unloads '{current}' first)" if current else "")


def _grid_unload(inventory: Any) -> str:
    if not isinstance(inventory, list) or not inventory:
        return "Return the grid on the stage to the magazine"
    if inventory[0].get("source") != "magazine":
        return "Unload: nothing to do on a fixed holder"
    current = _loaded(inventory)
    if current is None:
        return "Unload: no grid is on the stage"
    return f"Return grid '{current}' to the magazine"


def _grid_add(get: Callable[[str], Any]) -> str:
    """Which grids the inventory read will record: those present and not yet
    in the experiment."""
    inventory = _result(get(_SAMPLE_HANDLING + "get_inventory"))
    records = get("/app/experiment/grids")
    if not isinstance(inventory, list) or not isinstance(records, dict):
        return "Record the inventory's grids the experiment has no record of"
    known = {g.get("name") for g in records.get("items") or []}
    new = [
        e.get("name")
        for e in inventory
        if e.get("name")
        and e.get("state") in ("loaded", "occupied")
        and e.get("name") not in known
    ]
    if not new:
        return "Record the inventory's grids: every one is recorded already"
    return f"Record {_names_cut(new)} from the inventory"


def _slot_move(change: str, slot: Optional[Dict[str, Any]], now: Any) -> str:
    """A move to a holder slot, with where that is and where the stage is now."""
    target = slot.get("position") if slot else None
    if target:
        change += f": {_position(target)}"
    elif slot is not None:
        change += ": the slot is not calibrated, so it will be refused"
    if target and now:
        change += f"\nfrom {_position(now)}"
    return change


# The sample handling commands that have a tool of their own: called either way,
# a request reads the same.
_GRID_COMMANDS = {
    "ensure_loaded": "grid_load",
    "unload": "grid_unload",
    "run_inventory": "grid_run_inventory",
    "move_to_grid": "grid_move_to",
    "move_to_slot": "grid_move_to_slot",
}


def _grid_line(
    tool: str,
    a: Dict[str, Any],
    get: Callable[[str], Any],
    stage_now: Callable[[], Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    if tool == "grid_load":
        inventory = _result(get(_SAMPLE_HANDLING + "get_inventory"))
        return {
            "what": "Grids",
            "change": _grid_load(str(a.get("grid_name")), inventory),
        }
    if tool == "grid_unload":
        inventory = _result(get(_SAMPLE_HANDLING + "get_inventory"))
        return {"what": "Grids", "change": _grid_unload(inventory)}
    if tool == "grid_run_inventory":
        return {
            "what": "Grids",
            "change": "Scan the magazine slot by slot, to read which grids it holds",
        }
    slots = _result(get(_SAMPLE_HANDLING + "holder_slots"))
    slots = slots if isinstance(slots, list) else []
    if tool == "grid_move_to":
        name = str(a.get("grid_name"))
        change = f"Move to grid '{name}'"
        slot = next(
            (s for s in slots if (s.get("loaded_grid") or {}).get("name") == name),
            None,
        )
        if slot is not None:
            change += f", in {slot.get('name')}"
        elif slots:
            # As of now: an earlier step of the same request may load it.
            change += " (not on the stage yet)"
    else:
        name = str(a.get("slot_name"))
        change = f"Move to slot '{name}'"
        slot = next((s for s in slots if s.get("name") == name), None)
    return {"what": "Stage", "change": _slot_move(change, slot, stage_now())}


def describe_actions(
    actions: Sequence[Tuple[str, Dict[str, Any], str]],
    read: Optional[Reader] = None,
    describe: Optional[Describer] = None,
) -> List[Dict[str, Any]]:
    """Each of a request's ``(tool, arguments, path)`` described, in order. An
    action can read what came before it: a milling run says how long the
    patterns an earlier action prepares take."""
    described = []
    estimate: Optional[Tuple[Optional[float], Any]] = None
    unloaded = False
    for tool, arguments, path in actions:
        line = _describe(tool, arguments, path, read, describe, estimate)
        if "_estimate" in line:
            estimate = line.pop("_estimate")
        if unloaded and _grid_tool(tool, arguments) == "grid_load":
            # An earlier step unloads it: the load's own "(unloads 'G1'
            # first)", read from the stage as it is now, says it twice.
            line["change"] = _UNLOADS_FIRST.sub("", str(line.get("change")))
        unloaded = unloaded or _grid_tool(tool, arguments) == "grid_unload"
        described.append(line)
    return described


_UNLOADS_FIRST = re.compile(r" \(unloads '[^']*' first\)")


def _grid_tool(tool: str, arguments: Dict[str, Any]) -> str:
    """A grid command by its tool's name, called either way."""
    if tool == "service_call_command" and arguments.get("service") == "sample_handling":
        return _GRID_COMMANDS.get(str(arguments.get("command")), tool)
    return tool


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
    estimate,
) -> Dict[str, Any]:
    line = _line(tool, arguments or {}, path, read, describe, estimate)
    if line.get("details") is None:
        line.pop("details", None)
    return line


def _line(
    tool: str,
    a: Dict[str, Any],
    path: str,
    read: Optional[Reader],
    describe: Optional[Describer],
    estimate,
) -> Dict[str, Any]:
    item, task = a.get("item_name"), a.get("task_name")
    if describe is not None:
        try:
            own = describe(tool, a)
        except Exception:  # noqa: BLE001 - the server's own lines stand
            own = None
        if own:
            line = {"what": tool, "change": "Run"}
            line.update(_line(tool, a, path, read, None, estimate))
            line.update(own)
            return line

    def get(route: str) -> Any:
        """GET *route*, or None: no old value is shown, nothing more."""
        if read is None:
            return None
        try:
            return read(route)
        except Exception:  # noqa: BLE001
            return None

    def document(key: Optional[str]) -> Optional[Dict[str, Any]]:
        answer = get(path)
        if not isinstance(answer, dict):
            return None
        return answer.get(key) if key else answer

    def stage_now() -> Optional[Dict[str, Any]]:
        answer = get("/microscope/devices/stage/position")
        value = answer.get("value") if isinstance(answer, dict) else None
        return value if isinstance(value, dict) else None

    if tool == "stage_move_relative":
        return {"what": "Stage", "change": _stage_relative(a, stage_now())}
    if tool == "stage_move_absolute":
        change = f"Safe move to {_position(a)}"
        now = stage_now()
        if now:
            change += f"\nfrom {_position(now)}"
        return {"what": "Stage", "change": change}
    if tool == "stage_stable_move":
        return {
            "what": "Stage",
            "change": f"Stable move in {_beam_words(a.get('beam_type'))}: "
            f"{_offset(a.get('dx'), a.get('dy'))}",
        }
    if tool == "stage_vertical_move":
        change = (
            "Vertical move, to centre a feature "
            f"{_offset(a.get('dx'), a.get('dy'))} in {_beam_words(a.get('beam_type'), 'ION')}"
        )
        try:
            relaxation = float(a["relaxation"])
        except (KeyError, TypeError, ValueError):
            relaxation = 1.0
        if relaxation < 1:
            change += f", {_number(relaxation * 100)}% of the way"
        return {"what": "Stage", "change": change}
    if tool == "stage_move_to_orientation":
        return {
            "what": "Stage",
            "change": f"Move to the {str(a.get('orientation', '')).upper()} orientation",
        }
    if tool == "grid_add":
        return {"what": "Grids", "change": _grid_add(get)}
    if tool == "grid_rename":
        return {
            "what": f"Grid '{a.get('grid_name')}'",
            "change": f"Rename it '{a.get('name')}', on its slot and in the experiment",
        }
    if tool == "grid_set_quality":
        verdict = str(a.get("verdict", "")).upper()
        reason = a.get("reason")
        return {
            "what": f"Grid '{a.get('grid_name')}'",
            "change": "Clear its verdict"
            if verdict == "UNASSESSED"
            else f"Mark it {verdict.lower()}" + (f": {reason}" if reason else ""),
        }
    if tool == "grid_set_note":
        note = str(a.get("note") or "")
        return {
            "what": f"Grid '{a.get('grid_name')}'",
            "change": f"Note: {_clip(note)}" if note else "Clear its note",
        }
    if tool == "grid_remove":
        return {
            "what": f"Grid '{a.get('grid_name')}'",
            "change": "Stop tracking it; its files stay, its lamellae keep no grid",
        }
    if tool in _GRID_COMMANDS.values():
        return _grid_line(tool, a, get, stage_now)
    if tool == "milling_stop":
        return {"what": "Milling", "change": _MILLING["stop"]}
    if tool == "milling_end":
        return {"what": "Milling", "change": _MILLING["end"]}
    if tool == "stage_move_to_milling_angle":
        return {
            "what": "Stage",
            "change": f"Tilt to the milling angle, {_short(a.get('milling_angle_deg'))}°"
            + (
                f", rotation {_short(a['rotation_deg'])}°"
                if a.get("rotation_deg") is not None
                else ""
            ),
        }
    if tool == "beam_acquire_image":
        return _acquire_image(a, get)
    if tool == "last_image_preview":
        return {"what": _beam(a), "change": "Show the last image"}
    if tool == "beam_auto_contrast":
        return {"what": _beam(a), "change": "Autocontrast"}
    if tool == "beam_auto_brightness_contrast":
        return {"what": _beam(a), "change": "Auto brightness and contrast"}
    if tool == "beam_auto_focus":
        return _auto_focus(a)
    if tool == "beam_acquire_overview":
        return _overview(a)
    if tool == "beam_acquire_focus_stack":
        return _focus_stack(a)
    if tool == "fm_auto_focus":
        on = (
            f"on {a['channel_name']}"
            if a.get("channel_name")
            else "on the current channel"
        )
        return {"what": "Fluorescence", "change": f"Focus the objective {on}"}
    if tool == "fm_acquire_overview":
        return _fm_overview(a)
    if tool == "fm_acquire_image":
        return _fm_image(a)
    if tool == "milling_run_stage":
        return _run_stage(a)
    if tool == "milling_run_task":
        return _run_task(a)
    if tool == "alignment_align_to_reference":
        reference = (
            os.path.basename(str(a.get("reference_path") or "")) or "a reference"
        )
        steps = a.get("steps") or 3
        method = _ALIGN_WITH.get(
            str(a.get("subsystem") or "beam-shift"), "the beam shift"
        )
        return {
            "what": "Alignment",
            "change": f"Align to {reference}: {_plural(int(steps), 'step')} with {method}",
        }
    if tool == "microscope_set_state":
        return {
            "what": "Microscope",
            "change": "Set both beams and the stage to a saved state; the stage moves",
        }
    if tool == "alignment_check_coincidence":
        return {
            "what": "Alignment",
            "change": "Take an SEM and a FIB image to measure coincidence (moves nothing)",
        }
    if tool == "request_create":
        actions = [x for x in a.get("actions") or [] if isinstance(x, dict)]
        change = "Ask you to approve " + _plural(len(actions), "action")
        if actions:
            first = actions[0]
            line = describe_action(
                str(first.get("tool")), first.get("arguments") or {}, ""
            )
            said = str(line.get("change") or "").split("\n", 1)[0]
            change += (
                f": {line.get('what')} · {said}" if line.get("what") else f": {said}"
            )
        return {"what": "Request", "change": change}
    if tool == "device_set_parameter":
        return _set_parameter(a, "device", path, get)
    if tool == "service_set_parameter":
        return _set_parameter(a, "service", path, get)
    if tool in ("device_call_command", "service_call_command"):
        owner = a.get("device") if tool == "device_call_command" else a.get("service")
        kwargs = a.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        if tool == "service_call_command" and owner == "milling":
            line = _milling_command(str(a.get("command")), kwargs, estimate)
            if line is not None:
                return line
        if tool == "service_call_command" and owner == "sample_handling":
            grid_tool = _GRID_COMMANDS.get(str(a.get("command")))
            if grid_tool is not None:
                return _grid_line(grid_tool, kwargs, get, stage_now)
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
    if tool == "item_update_task_config":
        return {
            "what": f"{item} · {task}",
            "change": _patch_lines(a.get("patch"), document("config")),
        }
    if tool == "protocol_update_task_config":
        return {
            "what": f"Protocol · {task}",
            "change": _patch_lines(a.get("patch"), document("config")),
        }
    if tool == "item_update":
        return {
            "what": str(item),
            "change": _item_patch_lines(a.get("patch"), document(None)),
        }
    if tool == "item_add":
        return _add_item(a, None if a.get("stage_position") else stage_now())
    if tool == "item_remove":
        return {
            "what": str(item),
            "change": "Remove it from the experiment. Its files stay on disk",
        }
    if tool == "item_move_to_pose":
        pose = a.get("pose")
        change = f"Move the stage to its {pose} pose"
        target = _pose_of(get(_item_route(item)), pose)
        if target:
            change += f": {_position(target)}"
        now = stage_now()
        if target and now:
            change += f"\nfrom {_position(now)}"
        return {"what": str(item), "change": change}
    if tool == "item_apply_correlation":
        run = str(a.get("run_path") or "")
        name = os.path.basename(os.path.normpath(run)) if run else run
        change = f"Move its point of interest to the correlation saved in {name}"
        try:
            from fibsem.correlation.history import load_run

            result = load_run(run).result
            poi = result.poi[0].px_m
            change += (
                f": ({poi.x * 1e6:.2f}, {poi.y * 1e6:.2f}) µm from the image centre"
            )
        except Exception:  # the request still reads without the run's numbers
            pass
        change += "\nThe patterns that follow the point move with it"
        return {"what": str(item), "change": change}
    if tool == "item_set_pose":
        pose = a.get("pose")
        change = f"Set its {pose} pose to where the stage is now"
        now = stage_now()
        if now:
            change += f": {_position(now)}"
        old = _pose_of(get(_item_route(item)), pose)
        if old:
            change += f"\nwas {_position(old)}"
        if str(pose).upper() == "MILLING":
            change += "\nIts milling angle and fluorescence pose follow"
        return {"what": str(item), "change": change}
    if tool == "item_set_objective_position":
        return {
            "what": str(item),
            "change": "Store the objective's position now on its fluorescence pose",
        }
    if tool == "item_move_to_objective_position":
        return {
            "what": str(item),
            "change": "Move the FM objective to its stored position",
        }
    if tool == "item_apply_protocol":
        tasks = _names(a.get("task_names"), "every task")
        return {
            "what": str(item),
            "change": f"Copy the protocol's settings for {tasks}",
        }
    if tool == "item_reorder_milling_stages":
        return {
            "what": f"{item} · {task}",
            "change": f"Reorder {a.get('milling_key')} stages: {_names(a.get('order'), 'as they are')}",
        }
    if tool == "protocol_reorder_milling_stages":
        return {
            "what": f"Protocol · {task}",
            "change": f"Reorder {a.get('milling_key')} stages: {_names(a.get('order'), 'as they are')}",
        }
    if tool == "protocol_set_schedule":
        when = a.get("scheduled_at")
        return {
            "what": str(task),
            "change": f"Schedule for {when}" if when else "Clear its schedule",
        }
    if tool == "workflow_set_options":
        return {"what": "Workflow", "change": _options_line(a)}
    if tool == "protocol_add_task":
        return {
            "what": "Protocol",
            "change": f"Add a {a.get('task_type')} task named '{task}'",
        }
    if tool == "protocol_remove_task":
        return {
            "what": "Protocol",
            "change": f"Remove {task} from the protocol and every lamella",
        }
    if tool == "protocol_reorder_tasks":
        return {
            "what": "Protocol",
            "change": f"Put the tasks in this order: {_names(a.get('order'), 'none')}",
        }
    if tool == "protocol_set_task_requirements":
        changes = []
        if a.get("requires") is not None:
            changes.append(f"needs {_names(a.get('requires'), 'no task')} first")
        if a.get("required") is not None:
            changes.append("required" if a.get("required") else "optional")
        return {
            "what": str(task),
            "change": ("; ".join(changes) or "No change").capitalize(),
        }
    if tool == "protocol_set_lamella_defaults":
        given = {
            k: a[k]
            for k in ("use_petname", "name_prefix", "poi", "alignment_area")
            if k in a
        }
        return {
            "what": "New lamellae",
            "change": f"Start with {_values_text(given)}" if given else "No change",
        }
    if tool == "protocol_save":
        return {
            "what": "Protocol",
            "change": f"Save it to the protocol folder as {a.get('name')}"
            + (", replacing that file" if a.get("overwrite") else ""),
        }
    if tool == "protocol_update_grid_task_config":
        return {
            "what": f"Grid protocol · {task}",
            "change": _patch_lines(a.get("patch"), document("config")),
        }
    if tool == "protocol_reorder_grid_tasks":
        return {
            "what": "Grid protocol",
            "change": f"Put the grid tasks in this order: {_names(a.get('order'), 'none')}",
        }
    if tool == "protocol_set_grid_task_requirements":
        return {
            "what": f"Grid protocol · {task}",
            "change": f"Needs {_names(a.get('requires'), 'no task')} first",
        }
    if tool == "workflow_start":
        return {
            "what": "Workflow",
            "change": f"Run {_names(a.get('task_names'), 'no tasks')} on "
            f"{_names(a.get('item_names'), 'every lamella')}",
        }
    if tool == "workflow_start_grids":
        ends = [
            words
            for key, words in (
                ("unload_at_end", "unload the last grid"),
                ("report_at_end", "write the report"),
            )
            if a.get(key)
        ]
        return {
            "what": "Grids",
            "change": f"Run {_names(a.get('task_names'), 'no tasks')} on "
            f"{_names(a.get('grid_names'), 'every grid')}"
            + (f", then {' and '.join(ends)}" if ends else ""),
        }
    if tool == "workflow_queue_grids":
        return {
            "what": "Grids",
            "change": f"Add {_names(a.get('task_names'), 'no tasks')} for "
            f"{_names(a.get('grid_names'), 'no grids')} to the end of the run",
        }
    if tool == "workflow_stop":
        return {
            "what": "Workflow",
            "change": "Stop the run now running. A mill in progress stops too",
        }
    if tool == "workflow_stop_task":
        return {
            "what": f"{item} · {task}",
            "change": "Stop this task; the run carries on. A mill in progress stops too",
        }
    if tool == "protocol_set_supervision":
        return {
            "what": str(task),
            "change": "Supervised: ask at each decision"
            if a.get("supervise")
            else "Automated: run without asking",
        }
    if tool == "workflow_requeue":
        return {
            "what": str(item),
            "change": f"Queue {task} to run again"
            + (", first" if a.get("front") else ""),
        }
    if tool == "workflow_edit_queue":
        return {
            "what": str(item),
            "change": {
                "remove": f"Take {task} out of the queue",
                "move_up": f"Move {task} up the queue",
                "move_down": f"Move {task} down the queue",
                "run_next": f"Run {task} next",
            }.get(str(a.get("action")), f"{a.get('action')} {task} in the queue"),
        }
    if tool == "proposal_decide":
        outcome = str(a.get("outcome", "")).lower()
        if outcome.startswith("reject"):
            change = f"Reject the proposal: {a.get('reason', '')}".rstrip(": ")
        else:
            change = (
                f"Decide the pass: {a.get('action')}"
                if a.get("action")
                else "Confirm the proposal"
            )
            if a.get("values"):
                change += f" with {_values_text(a.get('values'))}"
            if a.get("patch"):
                change += f", setting {_values_text(a.get('patch'))}"
        name = _proposal_item(get("/app/workflow/proposals"), a.get("item_id"))
        return {"what": f"{name or a.get('item_id')} · {task}", "change": change}
    if tool == "proposal_suggest":
        name = _proposal_item(get("/app/workflow/proposals"), a.get("item_id"))
        return {
            "what": f"{name or a.get('item_id')} · {task}",
            "change": f"Suggest a decision: {a.get('reason', '')}".rstrip(": "),
        }
    if tool == "experiment_create":
        protocol = a.get("protocol")
        replaces = _open_experiment(get)
        return {
            "what": "Experiment",
            "change": f"Create and open '{a.get('name')}', with "
            + (f"protocol {protocol}" if protocol else "the default protocol"),
            "details": _table(
                ("Setting", "Value"),
                [
                    ("Folder", _short(a.get("name"))),
                    ("Protocol", protocol or "the default"),
                    ("Description", a.get("description") or ""),
                    ("Replaces", replaces or "none open"),
                ],
            ),
        }
    if tool == "experiment_load":
        path = str(a.get("path") or "")
        replaces = _open_experiment(get)
        about = _recent_experiment_about(get, path)
        return {
            "what": "Experiment",
            "change": f"Open '{_experiment_name(path)}'"
            + (f" ({about})" if about else "")
            + (f" in place of '{replaces}'" if replaces else ""),
            "details": _table(
                ("Setting", "Value"),
                [("Path", path), ("Replaces", replaces or "none open")],
            ),
        }
    if tool == "protocol_load":
        return {
            "what": "Protocol",
            "change": f"Load '{a.get('protocol')}' into the open experiment, "
            "replacing its protocol",
        }
    line = {"what": _words(tool).capitalize(), "change": "Run"}
    details = _kwargs_table(a)
    if details is not None:
        line["details"] = details
    return line
