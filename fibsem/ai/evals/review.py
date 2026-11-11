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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from fibsem.ai.evals.cases import (
    Case,
    accepted_labels,
    load_case,
    pass_names,
    stop_passes,
)

LOG = Path("reference") / "review.jsonl"
# The tasks whose disagreements can be settled here.
SETTLED_TASKS = ("site_selection", "square_quality", "polishing_stop")

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


def _kept(case: Case) -> List[Dict[str, Any]]:
    return [e for e in read_log(case) if e.get("action") == "keep"]


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
                }
                groups.append(group)
            group["points"].append((x, y))
            if _repeat(result) not in group["repeats"]:
                group["repeats"].append(_repeat(result))
            reason = str(pick.get("reason") or "") if isinstance(pick, dict) else ""
            if reason and reason not in group["reasons"]:
                group["reasons"].append(reason)
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
                "said": group["reasons"][:3],
                "actions": [
                    {"action": "good", "label": "Add as good"},
                    {"action": "bad", "label": "Add as bad"},
                ],
            }
        )
    return sorted(items, key=lambda i: -len(i["repeats"]))


def _square_items(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    labels = case.key.get("labels") or {}
    kept = {e.get("id") for e in _kept(case)}
    items = []
    for name, want in labels.items():
        name, want = str(name), accepted_labels(want)
        said: Dict[str, List[int]] = {}
        for result, answer in runs:
            value = ((answer or {}).get("labels") or {}).get(name)
            if value is None:
                continue
            said.setdefault(_label(value), []).append(_repeat(result))
        for label, repeats in sorted(said.items(), key=lambda kv: -len(kv[1])):
            item_id = f"square:{name}:{label}"
            if label in want or item_id in kept:
                continue
            items.append(
                {
                    "id": item_id,
                    "kind": "square",
                    "square": name,
                    "label": label,
                    "key": want,
                    "repeats": sorted(repeats),
                    "title": f"Square {name}: agent said {label}, key says {' or '.join(want)}",
                    "said": [],
                    "actions": [
                        {"action": "accept", "label": f"Accept {label} as well"}
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
    for result, answer in runs:
        stop = _stop(answer, names)
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
                "said": entry["reasons"][:3],
                "actions": [{"action": "accept", "label": f"Widen the span to {stop}"}],
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
    return []


# --- settling ------------------------------------------------------------------


class SettleError(ValueError):
    pass


def _changed_key(case: Case, item: Dict[str, Any], action: str) -> Dict[str, Any]:
    key = json.loads(json.dumps(case.key))
    if item["kind"] == "site" and action in ("good", "bad"):
        key.setdefault(action, []).append(
            {"x_px": item["x_px"], "y_px": item["y_px"], "from": "review"}
        )
    elif item["kind"] == "square" and action == "accept":
        labels = key.setdefault("labels", {})
        name = next((k for k in labels if str(k) == item["square"]), item["square"])
        labels[name] = accepted_labels(labels.get(name, [])) + [item["label"]]
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
    case: Case, runs: List[Run], item_id: str, action: str, run: str = ""
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
    if item["kind"] == "site":
        entry.update(x_px=item["x_px"], y_px=item["y_px"])
    if action != "keep":
        key = _changed_key(case, item, action)
        path = case.path / "case.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["key"] = key
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
    log = case.path / LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return load_case(case.path)
