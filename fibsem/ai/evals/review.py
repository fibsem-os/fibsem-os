"""Reviewing a run's results against the keys: where the agent disagreed
with a case's key, and settling each one by changing the key or keeping it.

A disagreement is something the key could be wrong about: a site the agent
picked that the key doesn't name, a square label the key doesn't accept, a
polishing stop outside the key's span. Settling one either changes the key in
``case.yaml`` (add the site as good or bad, accept the label, widen the span)
or keeps it; either way the choice goes on the case's review log
(``reference/review.jsonl``) with the key as it was, and a kept one is not
asked again.
"""

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from fibsem.ai.evals import tasks
from fibsem.ai.evals.cases import (
    Case,
    load_case,
    pass_names,
    pixel_size_um,
    square_call,
    stop_passes,
)
from fibsem.ai.evals.scoring import score_case

LOG = Path("reference") / "review.jsonl"
# (result, answer) of one repeat, as visual.Run.
Run = Tuple[Dict[str, Any], Optional[Dict[str, Any]]]


def _repeat(result: Dict[str, Any]) -> int:
    return int(result.get("repeat") or 0)


def _label(value: Any) -> str:
    value = value.get("label") if isinstance(value, dict) else value
    return str(value).strip().lower().replace(" ", "_")


# --- what was settled --------------------------------------------------------


def read_log(case: Case) -> List[Dict[str, Any]]:
    path = case.path / LOG
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


def _undone(entries: List[Dict[str, Any]]) -> set:
    return {e.get("undoes") for e in entries if e.get("action") == "undo"}


def _kept(case: Case) -> List[Dict[str, Any]]:
    entries = read_log(case)
    undone = _undone(entries)
    return [
        e
        for n, e in enumerate(entries)
        if e.get("action") == "keep" and n not in undone
    ]


def last_settled(case: Case) -> Optional[Dict[str, Any]]:
    """The newest settle that hasn't been undone, with its place in the log."""
    entries = read_log(case)
    undone = _undone(entries)
    for n in range(len(entries) - 1, -1, -1):
        entry = entries[n]
        if entry.get("action") != "undo" and n not in undone:
            return dict(entry, n=n)
    return None


# --- disagreements -------------------------------------------------------------


def _xy(value: Any) -> Optional[Tuple[float, float]]:
    try:
        return float(value["x_px"]), float(value["y_px"])
    except (KeyError, TypeError, ValueError):
        return None


def _missed(case: Case, answer: Optional[Dict[str, Any]]) -> bool:
    scores = score_case(case, answer or {})
    return bool(scores) and scores[0].result != "hit"


def _cluster(points: List[Tuple[int, float, float, str]], radius: float) -> List[Dict]:
    """(repeat, x, y, reason) grouped within *radius* of a group's first one."""
    groups: List[Dict[str, Any]] = []
    for repeat, x, y, reason in points:
        for group in groups:
            if math.hypot(x - group["x0"], y - group["y0"]) <= radius:
                break
        else:
            group = {"x0": x, "y0": y, "points": [], "by": {}}
            groups.append(group)
        group["points"].append((x, y))
        group["by"].setdefault(repeat, reason)
    for group in groups:
        xs, ys = zip(*group["points"])
        group["x"], group["y"] = round(sum(xs) / len(xs)), round(sum(ys) / len(ys))
    return groups


def key_point(case: Case) -> Optional[Tuple[float, float]]:
    """The target the key names, in the image the task answers on."""
    task = tasks.get(case.task)
    return task.key_point(case.key) if task and task.key_point else None


def point_items(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    """A target the repeats placed away from the key's (or a usable call the
    key doesn't make): move the key to it, or click a better place."""
    kept = {e.get("id") for e in _kept(case)}
    size = float(pixel_size_um(case) or 1)
    tol_um = float(
        case.scoring.get("xy_tolerance_um")
        or case.scoring.get("across_tolerance_um")
        or 1.0
    )
    radius = max(tol_um / size, 5)
    key_xy = key_point(case)
    items = []
    usable_key = case.key.get("usable")
    if case.task == "fm_target" and usable_key is not None:
        # A call on whether there is a usable target at all.
        wanted = str(usable_key).lower() in ("true", "yes", "1")
        said = [
            _repeat(r)
            for r, a in runs
            if a is not None
            and "usable" in a
            and (str(a.get("usable")).lower() in ("true", "yes", "1")) != wanted
        ]
        item_id = f"usable:{'no' if wanted else 'yes'}"
        if said and item_id not in kept:
            items.append(
                {
                    "id": item_id,
                    "kind": "usable",
                    "usable": not wanted,
                    "repeats": said,
                    "title": "Agent said there is "
                    + ("no usable target" if wanted else "a usable target")
                    + ", the key says "
                    + ("there is one" if wanted else "there isn't"),
                    "key_text": f"Key: usable {usable_key}",
                    "answers": [
                        {
                            "repeat": _repeat(r),
                            "said": f"usable {(a or {}).get('usable')}",
                            "reason": str((a or {}).get("reason") or ""),
                        }
                        for r, a in runs
                    ],
                    "said": [],
                    "actions": [
                        {
                            "action": "accept",
                            "label": "Change the key: "
                            + ("no usable target" if wanted else "usable"),
                        }
                    ],
                }
            )
    points = []
    for result, answer in runs:
        xy = _xy(answer or {})
        if xy is None or not _missed(case, answer):
            continue
        points.append((_repeat(result), *xy, str((answer or {}).get("reason") or "")))
    for group in _cluster(points, radius):
        item_id = f"point:{group['x']},{group['y']}"
        if item_id in kept:
            continue
        away = (
            f", {math.hypot(group['x'] - key_xy[0], group['y'] - key_xy[1]) * size:.2f} um from the key's"
            if key_xy
            else ""
        )
        items.append(
            {
                "id": item_id,
                "kind": "point",
                "x_px": group["x"],
                "y_px": group["y"],
                "repeats": sorted(group["by"]),
                "title": f"Target placed at ({group['x']}, {group['y']}){away}",
                "key_text": "Key: "
                + (f"({key_xy[0]:.0f}, {key_xy[1]:.0f})" if key_xy else "no target"),
                "answers": [
                    {
                        "repeat": _repeat(r),
                        "said": "placed it here"
                        if _repeat(r) in group["by"]
                        else "elsewhere",
                        "reason": group["by"].get(_repeat(r), ""),
                    }
                    for r, _ in runs
                ],
                "said": [],
                "actions": [{"action": "move", "label": "Move the key here"}],
                "can_place": True,
            }
        )
    return items


def _squares_found(answer: Optional[Dict[str, Any]]) -> List[Tuple[float, float]]:
    return [xy for xy in (_xy(p) for p in (answer or {}).get("squares") or []) if xy]


def disagreements(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    """What the repeats said that the key doesn't accept, most repeats first.
    Empty for a task this page can't settle, or a case with parts."""
    task = tasks.get(case.task)
    if case.parts or task is None or task.disagreements is None:
        return []
    return task.disagreements(case, runs)


def can_settle(case: Case) -> bool:
    task = tasks.get(case.task)
    return bool(task and task.disagreements) and not case.parts


# --- settling ------------------------------------------------------------------


class SettleError(ValueError):
    pass


def _key_square(key: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    for square in key.get("squares") or []:
        if (round(float(square["x_px"])), round(float(square["y_px"]))) == (
            item["x_px"],
            item["y_px"],
        ):
            return square
    raise SettleError(f"no key square at ({item['x_px']}, {item['y_px']})")


def _set_point(case: Case, key: Dict[str, Any], x: float, y: float) -> None:
    tasks.get(case.task).set_point(key, round(float(x), 1), round(float(y), 1))


def _changed_key(
    case: Case,
    item: Dict[str, Any],
    action: str,
    at: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    key = json.loads(json.dumps(case.key))
    if action == "place":
        # A point clicked on the picture: where the item's thing really is.
        if at is None or not item.get("can_place"):
            raise SettleError(f"can't place a {item['kind']} disagreement")
        if item["kind"] == "point":
            _set_point(case, key, *at)
        elif item["kind"] == "extra":
            key.setdefault("squares", []).append(
                {"x_px": round(at[0]), "y_px": round(at[1]), "from": "review"}
            )
        elif item["kind"] == "missed":
            square = _key_square(key, item)
            square.update(x_px=round(at[0]), y_px=round(at[1]), moved_by="review")
        return key
    if item["kind"] == "point" and action == "move":
        _set_point(case, key, item["x_px"], item["y_px"])
    elif item["kind"] == "usable" and action == "accept":
        key["usable"] = item["usable"]
    elif item["kind"] == "extra" and action in ("add", "optional"):
        square = {"x_px": item["x_px"], "y_px": item["y_px"], "from": "review"}
        if action == "optional":
            square["optional"] = True
        key.setdefault("squares", []).append(square)
    elif item["kind"] == "missed" and action == "optional":
        _key_square(key, item)["optional"] = True
    elif item["kind"] == "missed" and action == "remove":
        key["squares"].remove(_key_square(key, item))
    elif item["kind"] == "site" and action in ("good", "bad"):
        key.setdefault(action, []).append(
            {"x_px": item["x_px"], "y_px": item["y_px"], "from": "review"}
        )
    elif item["kind"] == "square" and action in ("accept", "replace"):
        labels = key.setdefault("labels", {})
        name = next((k for k in labels if str(k) == item["square"]), item["square"])
        call = square_call(labels.get(name))
        sites = call["sites"] + [item["site"]] if action == "accept" else [item["site"]]
        labels[name] = {
            "site": sites if len(sites) > 1 else sites[0],
            "reasons": call["reasons"],
        }
    elif item["kind"] == "stop" and action == "accept":
        names = pass_names(case)
        wanted = [names.index(s) for s in stop_passes(key)] + [
            names.index(item["pass"])
        ]
        span = names[min(wanted) : max(wanted) + 1]
        key["stop_after"] = span if len(span) > 1 else span[0]
    else:
        raise SettleError(f"can't {action!r} a {item['kind']} disagreement")
    return key


def settle(
    case: Case,
    runs: List[Run],
    item_id: str,
    action: str,
    run: str = "",
    at: Optional[Tuple[float, float]] = None,
) -> Case:
    """Settle one disagreement: *action* is ``keep`` or one the item offers.
    The case as it is after; its review log has the choice and the old key."""
    item = next((i for i in disagreements(case, runs) if i["id"] == item_id), None)
    if item is None:
        raise SettleError(f"no open disagreement {item_id!r}")
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": run,
        "id": item_id,
        "kind": item["kind"],
        "action": action,
        "repeats": item["repeats"],
        "key_before": case.key,
    }
    if "x_px" in item:
        entry.update(x_px=item["x_px"], y_px=item["y_px"])
    if at is not None:
        entry["at_px"] = [float(at[0]), float(at[1])]
    if action != "keep":
        _write_key(case, _changed_key(case, item, action, at))
    _log(case, entry)
    return load_case(case.path)


def undo(case: Case, run: str = "") -> Case:
    """Undo the newest settle: put the key back as it was before it (a kept
    one is asked again). The undo goes on the log too."""
    last = last_settled(case)
    if last is None:
        raise SettleError("nothing to undo")
    if last.get("action") != "keep" and "key_before" in last:
        _write_key(case, last["key_before"])
    _log(
        case,
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run": run,
            "id": last.get("id"),
            "action": "undo",
            "undoes": last["n"],
            "key_before": case.key,
        },
    )
    return load_case(case.path)


def _log(case: Case, entry: Dict[str, Any]) -> None:
    log = case.path / LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _write_key(case: Case, key: Dict[str, Any]) -> None:
    """Rewrite only the top-level ``key:`` block of case.yaml, so comments
    elsewhere in the file are kept (ones inside the key block are not)."""
    path = case.path / "case.yaml"
    text = path.read_text(encoding="utf-8")
    block = yaml.safe_dump(
        {"key": key},
        sort_keys=False,
        allow_unicode=True,
        width=4096,  # long lines stay on one line, as written
        default_flow_style=None,
    )
    lines = text.splitlines(keepends=True)
    start = next(
        (n for n, line in enumerate(lines) if re.match(r"key\s*:", line)), None
    )
    if start is None:
        if text and not text.endswith("\n"):
            text += "\n"
        path.write_text(text + block, encoding="utf-8")
        return
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line[0].isspace() and not line.startswith("#"):
            break
        end += 1
    # Comments and blank lines just above the next key belong to it.
    while end - 1 > start and (
        not lines[end - 1].strip() or lines[end - 1].startswith("#")
    ):
        end -= 1
    path.write_text(
        "".join(lines[:start]) + block + "".join(lines[end:]), encoding="utf-8"
    )
