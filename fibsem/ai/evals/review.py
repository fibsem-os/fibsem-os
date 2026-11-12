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

from fibsem.ai.evals.cases import (
    Case,
    load_case,
    pass_names,
    pixel_size_um,
    square_call,
    stop_passes,
)
from fibsem.ai.evals.scoring import match_squares, score_case

LOG = Path("reference") / "review.jsonl"
# The tasks whose disagreements can be settled here.
SETTLED_TASKS = (
    "site_selection",
    "square_quality",
    "polishing_stop",
    "fm_target",
    "correlation",
    "square_finding",
)

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


def _site_items(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    key = case.key
    radius = float(key.get("hit_radius_px", case.scoring.get("hit_radius_px", 100)))
    keyed = (key.get("good") or []) + (key.get("bad") or [])
    most = int(case.scoring.get("max_picks", 10))
    kept = [e for e in _kept(case) if e.get("kind") == "site"]

    def near(x, y, points) -> bool:
        return any(math.hypot(x - p["x_px"], y - p["y_px"]) <= radius for p in points)

    groups: List[Dict[str, Any]] = []
    for result, answer in runs:
        for pick in ((answer or {}).get("picks") or [])[:most]:
            try:
                x, y = float(pick["x_px"]), float(pick["y_px"])
            except (KeyError, TypeError, ValueError):
                continue
            if near(x, y, keyed) or near(x, y, kept):
                continue
            for group in groups:
                if math.hypot(x - group["x_px"], y - group["y_px"]) <= radius:
                    break
            else:
                group = {
                    "x_px": x,
                    "y_px": y,
                    "points": [],
                    "repeats": [],
                    "reasons": [],
                    "by": {},
                }
                groups.append(group)
            group["points"].append((x, y))
            if _repeat(result) not in group["repeats"]:
                group["repeats"].append(_repeat(result))
            reason = str(pick.get("reason") or "") if isinstance(pick, dict) else ""
            if reason and reason not in group["reasons"]:
                group["reasons"].append(reason)
            group["by"].setdefault(_repeat(result), reason)
    items = []
    for group in groups:
        xs, ys = zip(*group["points"])
        x, y = round(sum(xs) / len(xs)), round(sum(ys) / len(ys))
        items.append(
            {
                "id": f"site:{x},{y}",
                "kind": "site",
                "x_px": x,
                "y_px": y,
                "repeats": sorted(group["repeats"]),
                "title": f"Site at ({x}, {y}) is not in the key",
                "key_text": "Not in the key as good or bad",
                "answers": [
                    {
                        "repeat": _repeat(result),
                        "said": "picked it"
                        if _repeat(result) in group["by"]
                        else "did not pick it",
                        "reason": group["by"].get(_repeat(result), ""),
                    }
                    for result, _ in runs
                ],
                "said": group["reasons"][:3],
                "actions": [
                    {"action": "good", "label": "Add as good"},
                    {"action": "bad", "label": "Add as bad"},
                ],
            }
        )
    return sorted(items, key=lambda i: -len(i["repeats"]))


def _square_items(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    """Squares where a repeat called the site (good or bad) other than the
    key does. The reasons are shown beside it, not settled."""
    labels = case.key.get("labels") or {}
    kept = {e.get("id") for e in _kept(case)}
    items = []
    for name, value in labels.items():
        name, want = str(name), square_call(value)
        said: Dict[str, List[int]] = {}
        answers = []
        for result, answer in runs:
            given = ((answer or {}).get("labels") or {}).get(name)
            call = square_call(given) if given is not None else None
            reason = given.get("reason", "") if isinstance(given, dict) else ""
            site = call["sites"][0] if call and call["sites"] else "-"
            answers.append(
                {
                    "repeat": _repeat(result),
                    "said": site
                    + (
                        f" ({', '.join(call['reasons'])})"
                        if call and call["reasons"]
                        else ""
                    ),
                    "reason": str(reason or ""),
                }
            )
            if call and call["sites"]:
                said.setdefault(site, []).append(_repeat(result))
        key_text = " or ".join(want["sites"]) + (
            f" ({', '.join(want['reasons'])})" if want["reasons"] else ""
        )
        for site, repeats in sorted(said.items(), key=lambda kv: -len(kv[1])):
            item_id = f"square:{name}:{site}"
            if site in want["sites"] or item_id in kept:
                continue
            items.append(
                {
                    "id": item_id,
                    "kind": "square",
                    "square": name,
                    "site": site,
                    "key": want["sites"],
                    "repeats": sorted(repeats),
                    "title": f"Square {name}: agent called it a {site} site, the key says {' or '.join(want['sites'])}",
                    "key_text": "Key: " + key_text,
                    "answers": answers,
                    "said": [],
                    "actions": [
                        {"action": "accept", "label": f"Accept {site} as well"},
                        {"action": "replace", "label": f"Change the key to {site}"},
                    ],
                }
            )
    return sorted(items, key=lambda i: -len(i["repeats"]))


def _stop(answer: Optional[Dict[str, Any]], names: List[str]) -> Optional[str]:
    stops = [
        names.index(p.get("name"))
        for p in (answer or {}).get("passes") or []
        if isinstance(p, dict)
        and str(p.get("decision", "")).lower().replace(" ", "_") == "stop"
        and p.get("name") in names
    ]
    return names[min(stops)] if stops else None


def _stop_items(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    names = pass_names(case)
    span = stop_passes(case.key)
    kept = {e.get("id") for e in _kept(case)}
    stops: Dict[str, Dict[str, Any]] = {}
    answers = []
    for result, answer in runs:
        stop = _stop(answer, names)
        reason = next(
            (
                str(p.get("reason") or "")
                for p in (answer or {}).get("passes") or []
                if isinstance(p, dict) and stop and p.get("name") == stop
            ),
            "",
        )
        answers.append(
            {
                "repeat": _repeat(result),
                "said": f"stopped after {stop}" if stop else "never stopped",
                "reason": reason,
            }
        )
        if stop is None or stop in span:
            continue
        entry = stops.setdefault(stop, {"repeats": [], "reasons": []})
        entry["repeats"].append(_repeat(result))
        for p in (answer or {}).get("passes") or []:
            reason = str(p.get("reason") or "") if isinstance(p, dict) else ""
            if reason and p.get("name") == stop and reason not in entry["reasons"]:
                entry["reasons"].append(reason)
    items = []
    for stop, entry in stops.items():
        item_id = f"stop:{stop}"
        if item_id in kept:
            continue
        early = span and names.index(stop) < min(names.index(s) for s in span)
        items.append(
            {
                "id": item_id,
                "kind": "stop",
                "pass": stop,
                "key": span,
                "repeats": sorted(entry["repeats"]),
                "title": f"Stopped after {stop}, the key's span is {', '.join(span)}"
                + (" (early)" if early else " (late)"),
                "key_text": "Key: stop after " + " or ".join(span),
                "answers": answers,
                "said": entry["reasons"][:3],
                "actions": [{"action": "accept", "label": f"Widen the span to {stop}"}],
            }
        )
    return sorted(items, key=lambda i: -len(i["repeats"]))


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
    if case.task == "fm_target":
        return _xy(case.key.get("target") or {})
    if case.task == "correlation":
        target = case.key.get("target_fib_px") or {}
        try:
            return float(target["x"]), float(target["y"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _point_items(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
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


def _square_finding_items(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    """Squares the repeats found that the key doesn't have, and key squares
    a repeat didn't find."""
    kept = {e.get("id") for e in _kept(case)}
    squares = case.key.get("squares") or []
    pitch = float(case.key.get("pitch_px") or 100)
    radius = float(case.scoring.get("match_radius_px", pitch / 3))
    extra, missed = [], {}
    for result, answer in runs:
        points = _squares_found(answer)
        matched = match_squares(points, squares, radius)
        extra += [
            (_repeat(result), x, y, "")
            for i, (x, y) in enumerate(points)
            if i not in matched
        ]
        found = set(matched.values())
        for j, sq in enumerate(squares):
            if j not in found and not sq.get("optional"):
                missed.setdefault(j, []).append(_repeat(result))
    items = []
    for group in _cluster(extra, radius):
        item_id = f"extra:{group['x']},{group['y']}"
        if item_id in kept:
            continue
        items.append(
            {
                "id": item_id,
                "kind": "extra",
                "x_px": group["x"],
                "y_px": group["y"],
                "repeats": sorted(group["by"]),
                "title": f"A square at ({group['x']}, {group['y']}) is not in the key",
                "key_text": "Not one of the key's squares",
                "answers": [
                    {
                        "repeat": _repeat(r),
                        "said": "found it" if _repeat(r) in group["by"] else "-",
                        "reason": "",
                    }
                    for r, _ in runs
                ],
                "said": [],
                "actions": [
                    {"action": "add", "label": "Add it to the key"},
                    {"action": "optional", "label": "Add it as optional"},
                ],
                "can_place": True,
            }
        )
    for j, repeats in missed.items():
        sq = squares[j]
        x, y = round(float(sq["x_px"])), round(float(sq["y_px"]))
        item_id = f"missed:{x},{y}"
        if item_id in kept:
            continue
        items.append(
            {
                "id": item_id,
                "kind": "missed",
                "x_px": x,
                "y_px": y,
                "repeats": sorted(repeats),
                "title": f"Key square at ({x}, {y}) was not found",
                "key_text": "A required square in the key",
                "answers": [
                    {
                        "repeat": _repeat(r),
                        "said": "missed it" if _repeat(r) in repeats else "found it",
                        "reason": "",
                    }
                    for r, _ in runs
                ],
                "said": [],
                "actions": [
                    {"action": "optional", "label": "Make it optional"},
                    {"action": "remove", "label": "Remove it from the key"},
                ],
                "can_place": True,
            }
        )
    return sorted(items, key=lambda i: -len(i["repeats"]))


def disagreements(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    """What the repeats said that the key doesn't accept, most repeats first.
    Empty for a task this page can't settle, or a case with parts."""
    if case.parts:
        return []
    if case.task == "site_selection":
        return _site_items(case, runs)
    if case.task == "square_quality":
        return _square_items(case, runs)
    if case.task == "polishing_stop":
        return _stop_items(case, runs)
    if case.task in ("fm_target", "correlation"):
        return _point_items(case, runs)
    if case.task == "square_finding":
        return _square_finding_items(case, runs)
    return []


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
    x, y = round(float(x), 1), round(float(y), 1)
    if case.task == "fm_target":
        key.setdefault("target", {}).update(x_px=x, y_px=y)
    else:
        key.setdefault("target_fib_px", {}).update(x=x, y=y)


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
