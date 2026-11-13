"""Polishing stop: when to stop polishing, pass by pass."""

from typing import Any, Dict, List, Optional

import numpy as np

from fibsem.ai.evals.cases import Case, pass_names, stop_passes
from fibsem.ai.evals.review import _kept, _repeat
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import KEY, _colour, _pass_strip, _stops


def score_polishing_stop(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    passes = answer.get("passes")
    if not isinstance(passes, list):
        raise InvalidAnswer("no passes list")
    names = pass_names(case)
    stop = None
    for entry in passes:
        if entry.get("name") not in names:
            raise InvalidAnswer(f"unknown pass {entry.get('name')!r}")
        decision = str(entry.get("decision", "")).lower().replace(" ", "_")
        if decision not in ("carry_on", "stop"):
            raise InvalidAnswer(
                f"decision {entry.get('decision')!r} is not carry_on or stop"
            )
        if decision == "stop":
            index = names.index(entry["name"])
            stop = index if stop is None else min(stop, index)
    wanted = [names.index(name) for name in stop_passes(key)]
    first, last = min(wanted), max(wanted)
    details = {
        "stopped_after": names[stop] if stop is not None else None,
        "key": key["stop_after"],
    }
    if stop is not None and first <= stop <= last:
        return "hit", details
    if stop is not None and stop < first:
        details["passes_early"] = first - stop
        return "early", details
    # Carrying on past the stop: thickness lost, or the target milled away.
    details["costly"] = 1
    return "late", details


def draw_polishing_stop(case: Case, runs: List[Run], key=None, part=None):
    key = key or case.key
    names = pass_names(case)
    passes = case.data.get("passes") or []
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(max(8, 1.1 * len(names)), 6))
    grid = fig.add_gridspec(2, len(names), height_ratios=(1, 2))
    _pass_strip(case, [fig.add_subplot(grid[0, i]) for i in range(len(names))])
    ax = fig.add_subplot(grid[1, :])
    xs = np.arange(len(names))
    med = [p.get("measured", {}).get("median_nm", np.nan) for p in passes]
    lo = [p.get("measured", {}).get("min_nm", np.nan) for p in passes]
    hi = [p.get("measured", {}).get("max_nm", np.nan) for p in passes]
    ax.fill_between(xs, lo, hi, color="#cccccc", label="min-max")
    ax.plot(xs, med, "k.-", label="median")
    wanted = [names.index(name) for name in stop_passes(key)]
    ax.axvspan(
        min(wanted) - 0.08,
        max(wanted) + 0.08,
        color=KEY,
        alpha=0.4,
        label="key: stop after " + " or ".join(stop_passes(key)),
    )
    for n, (result, answer) in enumerate(runs):
        stop = _stops(
            answer if part is None else ((answer or {}).get("parts") or {}).get(part),
            names,
        )
        if stop is not None:
            ax.axvline(
                stop + 0.04 * (n + 1),
                color=_colour(n),
                ls="--",
                label=f"r{result['repeat']} stops after {names[stop]}",
            )
        else:
            ax.text(
                len(names) - 1,
                np.nanmax(hi),
                f"r{result['repeat']}: never stops",
                color=_colour(n),
                ha="right",
                va="top",
                fontsize=8,
            )
    ax.set_xticks(xs, names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("thickness, nm")
    ax.legend(fontsize=7)
    return fig


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


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    names, stops = pass_names(case), stop_passes(key)
    if not stops or any(stop not in names for stop in stops):
        return [f"key stop_after {key.get('stop_after')!r} is not a pass"]
    return []


def summary(d: Dict[str, Any]) -> str:
    if "key" not in d:
        return ""
    stop = d.get("stopped_after")
    return f"stopped after {stop}" if stop else "never stopped"


def key_text(key: Dict[str, Any]) -> str:
    return "stop after " + " or ".join(stop_passes(key))


TASK = EvalTask(
    name="polishing_stop",
    title="Say when to stop polishing, pass by pass",
    skill="autolamella-polishing",
    answer_shape={
        "passes": [{"name": "<pass>", "decision": "carry_on | stop", "reason": "..."}]
    },
    score=score_polishing_stop,
    check_key=check_key,
    summary=summary,
    draw=draw_polishing_stop,
    key_text=key_text,
    disagreements=_stop_items,
    sims=(Sim("polishing", "fibsem.ai.evals.sim_polishing"),),
)
