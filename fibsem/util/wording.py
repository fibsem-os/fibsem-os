"""Actions in the operator's words: the pieces every line of a request is made of.

The approval popover shows each action an agent asks for as two lines, what it
acts on and the change, written by the server from the tool and its arguments,
never from the agent's words. Each tool writes its own line
(``ToolSpec.line``), and a service command its own (``command(line=...)``),
from these pieces: values as words, distances and currents in their units,
stage positions, and a config edit's dotted paths as old -> new.

An action may also carry ``details``, a small table the operator can open
("Show details"): every value, so the one line never has to hold them all.
"""

import math
import os
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from fibsem.structures.params import Parameter

# GET a route's JSON (the document an edit patches), or a query's answer with
# its kwargs (the second argument); None for neither.
Reader = Callable[..., Optional[Dict[str, Any]]]

# A long value in a line is cut here; the details table holds it whole.
LINE_VALUE = 40

_MISSING = object()


def short(value: Any) -> str:
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
        return ", ".join(f"{words(k)} {short(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(short(v) for v in value)
    return str(value)


def clip(text: str, limit: int = LINE_VALUE) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def number(value: float) -> str:
    """3 significant figures, no trailing zeros: 2.5, 20, 0.2, −3."""
    return minus(f"{value:.3g}")


def minus(text: str) -> str:
    """A typeset minus sign, as the operator reads one: −3, not -3."""
    return "−" + text[1:] if text.startswith("-") else text


def coordinate(metres: Any) -> str:
    """A place on the stage to 0.1 µm, which 3 significant figures of a
    millimetre lose: 1.2345 mm, −9.5 µm."""
    try:
        metres = float(metres)
    except (TypeError, ValueError):
        return short(metres)
    if abs(metres) >= 1e-3:
        text, unit = f"{metres * 1e3:.4f}", "mm"
    else:
        text, unit = f"{metres * 1e6:.1f}", "µm"
    text = text.rstrip("0").rstrip(".")
    if float(text) == 0:
        text = "0"
    return f"{minus(text)} {unit}"


def distance(metres: Any) -> str:
    try:
        metres = float(metres)
    except (TypeError, ValueError):
        return short(metres)
    if abs(metres) >= 1e-3:
        return f"{number(metres * 1e3)} mm"
    if abs(metres) >= 1e-8 or metres == 0:
        return f"{number(metres * 1e6)} µm"
    return f"{number(metres * 1e9)} nm"


def current(amps: Any) -> str:
    try:
        amps = float(amps)
    except (TypeError, ValueError):
        return short(amps)
    if abs(amps) >= 1e-9:
        return f"{number(amps * 1e9)} nA"
    return f"{number(amps * 1e12)} pA"


def seconds(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    return f"{round(seconds / 60)} min"


def words(name: str) -> str:
    """``milling_current`` -> ``milling current``."""
    return str(name).replace("_", " ")


def table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    return {"columns": list(columns), "rows": [[str(c) for c in row] for row in rows]}


def kwargs_table(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not kwargs:
        return None
    return table(
        ("Argument", "Value"), [(words(k), short(v)) for k, v in kwargs.items()]
    )


def display(value: Any, spec: Optional["Parameter"]) -> str:
    """``value`` as the field's form shows it: 2e-06 with scale 1e6 and unit m
    reads "2 µm". No value reads "(none)"."""
    if value is None:
        return "(none)"
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, bool):
        return "on" if value else "off"
    if not isinstance(value, (int, float)):
        return short(value)
    if spec is None:
        shown, unit = value, ""
    elif spec.unit and not spec.scale and not spec._shown.unit:
        # no scale declared: the prefix the value's size calls for (60 pA)
        from fibsem.utils import _get_display_unit, _get_scale_from_value

        scale = _get_scale_from_value(value)
        shown, unit = value * scale, _get_display_unit(scale, spec.unit)
    else:
        shown, unit = value * (spec.scale or 1), spec.suffix
    text = number(shown) if isinstance(shown, float) else str(shown)
    if unit in ("°", "%"):
        return f"{text}{unit}"
    return f"{text} {unit}".rstrip()


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
    from fibsem.structures.params import field_specs

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
    name = str((spec.label if spec is not None else None) or words(leaf)).lower()
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
        after = display(new, spec)
        if old is _MISSING:
            lines.append(f"{setting}: {after}")
            rows.append((setting, "", after))
            continue
        before = display(old, spec) if old not in (None, "") else "(none)"
        # A switch reads "Re-align during milling: off → on".
        joint = ": " if isinstance(old, bool) else " "
        lines.append(f"{setting}{joint}{before} → {after}")
        rows.append((setting, before, after))
    return {
        "change": "\n".join(lines),
        "details": table(("Setting", "Now", "New"), rows),
    }


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def names_cut(names: List[str], shown: int = 3) -> str:
    if len(names) <= shown:
        return ", ".join(names)
    return ", ".join(names[:shown]) + " …"


def degrees(radians: Any) -> str:
    try:
        return f"{number(math.degrees(float(radians)))}°"
    except (TypeError, ValueError):
        return short(radians)


def beam_name(arguments: Dict[str, Any]) -> str:
    beam = str(arguments.get("beam_type", "ELECTRON")).upper()
    return "Ion beam" if beam == "ION" else "Electron beam"


def beam_short(beam_type: Any) -> str:
    return "FIB" if str(beam_type or "ELECTRON").upper() in ("ION", "2") else "SEM"


def names_text(values: Any, every: str) -> str:
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


def item_setting(path: str) -> str:
    """An item's dotted path as words: "poi.x" reads "POI x"."""
    text = words(str(path).replace(".", " "))
    return "POI" + text[3:] if text.startswith("poi") else text


def item_value(path: str, value: Any) -> str:
    """An item's value in display units: the POI in metres reads in µm, the
    alignment area's fractions of the frame as percent."""
    if isinstance(value, dict):
        # a Point's empty name reads as nothing, not "name none"
        return ", ".join(
            f"{words(k)} {item_value(f'{path}.{k}', v)}"
            for k, v in value.items()
            if v is not None
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parts = str(path).split(".")
        if "area" in str(path):
            return f"{number(value * 100)}%"
        if parts[-1] in ("x", "y", "z") or "poi" in parts or "point" in parts:
            return distance(value)
    return short(value)


# An item's places, each one line however many of its fields the edit sets:
# (field, words, the fields in the order they read).
_ITEM_PLACES = (
    ("poi", "Point of interest", ("x", "y")),
    ("alignment_area", "Alignment area", ("left", "top", "width", "height")),
)


def item_patch_lines(patch: Any, document: Optional[Dict[str, Any]]) -> str:
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
    for field, label, parts in _ITEM_PLACES:
        if not any(f"{field}.{p}" in flat for p in parts):
            continue
        old = {p: _walk(document or {}, f"{field}.{p}") for p in parts}
        new = {p: flat.pop(f"{field}.{p}", old[p]) for p in parts}
        unit = " µm" if field == "poi" else ""
        line = f"{label}: {_item_place(field, new)}{unit}"
        if all(v is not _MISSING for v in old.values()):
            before = (
                "(none)"
                if all(v is None for v in old.values())
                else _item_place(field, old)
            )
            line = f"{label}: {before} → {_item_place(field, new)}{unit}"
        lines.append(line)
    rest = patch_lines(flat, document, item_setting, item_value) if flat else ""
    return "\n".join(lines + ([rest] if rest else []))


def _item_place(field: str, values: Dict[str, Any]) -> str:
    """A point of interest as "(+1, −3.2)", in µm from the image's centre to
    0.1 µm; an alignment area as "30 × 20% at (25, 50)%". A value not known
    reads "?"."""

    def number(value: Any, scale: float) -> str:
        try:
            return number(float(value) * scale)
        except (TypeError, ValueError):
            return "?"

    def offset(value: Any) -> str:
        try:
            text = f"{float(value) * 1e6:+.1f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return "?"
        return "0" if float(text) == 0 else minus(text)

    if field == "poi":
        return f"({offset(values['x'])}, {offset(values['y'])})"
    return (
        f"{number(values['width'], 100)} × {number(values['height'], 100)}% at "
        f"({number(values['left'], 100)}, {number(values['top'], 100)})%"
    )


def patch_lines(
    patch: Any,
    document: Optional[Dict[str, Any]],
    name: Callable[[str], str] = str,
    value: Callable[[str, Any], str] = lambda _path, v: short(v),
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
            before = "(none)" if old in (None, "") else value(str(path), old)
            lines.append(f"{name(str(path))}: {before} → {after}")
    return "\n".join(lines)


def values_text(values: Any) -> str:
    """A decision's or an answer's values as words: {"poi": {"x", "y"}} reads
    "POI x 1.2 µm, y -3 µm"."""
    if isinstance(values, dict):
        return ", ".join(
            f"{item_setting(k)} {item_value(k, v)}" for k, v in values.items()
        )
    return short(values)


def stage_relative(a: Dict[str, Any], now: Optional[Dict[str, Any]] = None) -> str:
    """ "Move 2.5 µm in x, from x 8 µm": by how much along which stage axis, and
    from where on that axis when the stage's position is known."""
    axes = [
        (axis, a[k]) for axis, k in (("x", "dx"), ("y", "dy"), ("z", "dz")) if a.get(k)
    ]
    if not axes:
        return "Move nothing"
    text = "Move " + ", ".join(f"{minus(distance(v))} in {axis}" for axis, v in axes)
    if now:
        known = [
            f"{axis} {coordinate(now[axis])}"
            for axis, _ in axes
            if now.get(axis) is not None
        ]
        if known:
            text += ", from " + ", ".join(known)
    return text


def beam_words(beam_type: Any, default: str = "ELECTRON") -> str:
    beam = str(beam_type or default).upper()
    return "the ion beam" if beam in ("ION", "2") else "the electron beam"


def offset(dx: Any, dy: Any) -> str:
    """An image offset, x right and y up, in the operator's words."""
    parts = []
    try:
        x, y = float(dx or 0.0), float(dy or 0.0)
    except (TypeError, ValueError):
        return f"x {short(dx)}, y {short(dy)}"
    if x:
        parts.append(f"{distance(abs(x))} {'right' if x > 0 else 'left'}")
    if y:
        parts.append(f"{distance(abs(y))} {'up' if y > 0 else 'down'}")
    return ", ".join(parts) or "no offset"


def position_text(p: Any) -> str:
    """A stage position in metres and radians, in display units."""
    if not isinstance(p, dict):
        return short(p)
    parts = [
        f"{axis} {coordinate(p[axis])}" for axis in "xyz" if p.get(axis) is not None
    ]
    for axis, word in (("t", "tilt"), ("r", "rotation")):
        if p.get(axis) is not None:
            parts.append(f"{word} {degrees(p[axis])}")
    return ", ".join(parts)


def stage_place(p: Dict[str, Any]) -> str:
    """Where a stage position is, as a decision reads it: x and y (z when it
    is not 0), then "flat" or its tilt and rotation; no name or coordinate
    system. "x −35.5 µm, y −27.9 µm, flat"."""
    parts = [
        f"{axis} {coordinate(p[axis])}"
        for axis in "xyz"
        if p.get(axis) is not None and (axis != "z" or p[axis])
    ]
    t, r = p.get("t"), p.get("r")
    if not t and not r:
        parts.append("flat")
    else:
        parts += [f"tilt {degrees(t or 0)}", f"rotation {degrees(r or 0)}"]
    return ", ".join(parts)


def dwell(seconds: Any) -> str:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return short(seconds)
    if seconds >= 1e-3:
        return f"{number(seconds * 1e3)} ms"
    if seconds >= 1e-6:
        return f"{number(seconds * 1e6)} µs"
    return f"{number(seconds * 1e9)} ns"


def resolution(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]} × {value[1]}"
    return short(value)


def folder(path: Any) -> str:
    """A folder as its last part: the experiment's folders are long."""
    text = str(path or "").rstrip("/\\").replace("\\", "/")
    return os.path.basename(text) or text or "none"


def about(duration: Any) -> str:
    """ ", about 2 min": how long a run takes, as its estimate says; empty when
    there is none."""
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return ""
    return f", about {seconds(duration)}" if duration > 0 else ""


def result_of(answer: Any) -> Any:
    return answer.get("result") if isinstance(answer, dict) else None


@dataclass
class ActionContext:
    """What an action's line may read: its route (``path``), the server's
    reader, and the estimate an earlier action of the same request left (a
    milling prepare's, for the run after it)."""

    path: str = ""
    read: Optional[Reader] = None
    estimate: Optional[Tuple[Optional[float], Any]] = None

    def get(self, route: str, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """GET *route* (a query, with *kwargs*), or None: no old value or
        estimate is shown, nothing more."""
        if self.read is None:
            return None
        try:
            return self.read(route) if kwargs is None else self.read(route, kwargs)
        except Exception:  # noqa: BLE001
            return None

    def document(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        """The document the action's route reads, or its *key*."""
        answer = self.get(self.path)
        if not isinstance(answer, dict):
            return None
        return answer.get(key) if key else answer

    def stage_now(self) -> Optional[Dict[str, Any]]:
        """Where the stage is now, or None."""
        answer = self.get("/microscope/devices/stage/position")
        value = answer.get("value") if isinstance(answer, dict) else None
        return value if isinstance(value, dict) else None


# A tool's or command's line: its arguments and what it may read, to
# ``{"what", "change", "details"?}``.
Line = Callable[[Dict[str, Any], ActionContext], Dict[str, Any]]
