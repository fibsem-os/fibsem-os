"""Square quality: each numbered square a good or bad site, with reasons."""

from typing import Any, Dict, List

import numpy as np

from fibsem.ai.evals.cases import SITES, SQUARE_REASONS, Case, square_call
from fibsem.ai.evals.review import _kept, _repeat
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.base import EvalTask, Run
from fibsem.ai.evals.visual import _figure, _input, _read_image, _show


def score_square_quality(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    """The headline is the site call, good or bad, against the key; the
    reasons are scored apart (recall and precision over every square) and
    never decide the headline."""
    labels = answer.get("labels")
    if not isinstance(labels, dict):
        raise InvalidAnswer("no labels map")
    said = {}
    for name, value in labels.items():
        call = square_call(value)
        old = not (isinstance(value, dict) and "site" in value)
        if old and not set(call["reasons"]) <= set(SQUARE_REASONS):
            raise InvalidAnswer(f"square {name}: {value!r} is not a label")
        if len(call["sites"]) != 1 or call["sites"][0] not in SITES:
            raise InvalidAnswer(
                f"square {name}: site {call['sites']} is not good or bad"
            )
        said[str(name)] = call
    # Calling these good sends a lamella to a square that can't hold one.
    never_good = set(
        case.scoring.get(
            "costly_if_called_good",
            case.scoring.get("unsafe_if_called_good", ["damaged", "thick"]),
        )
    )
    right, wrong, unanswered, bad_called_good = 0, {}, [], []
    found = expected_reasons = given_reasons = 0
    for name, entry in key["labels"].items():
        name, want = str(name), square_call(entry)
        if name not in said:
            unanswered.append(name)
            expected_reasons += len(want["reasons"])
            continue
        got = said[name]
        site = got["sites"][0]
        if site in want["sites"]:
            right += 1
        else:
            wrong[name] = {
                "said": site,
                "key": " | ".join(want["sites"]),
                "key_reasons": want["reasons"],
            }
            if site == "good" and set(want["reasons"]) & never_good:
                bad_called_good.append(name)
        found += len(set(got["reasons"]) & set(want["reasons"]))
        expected_reasons += len(set(want["reasons"]))
        given_reasons += len(set(got["reasons"]))
    total = len(key["labels"])
    accuracy = right / total if total else 0.0
    details = {
        "correct": right,
        "total": total,
        "accuracy": round(accuracy, 3),
        "wrong": wrong,
        "unanswered": unanswered,
        "bad_called_good": bad_called_good,
        "costly": len(bad_called_good),
        "reasons_recall": round(found / expected_reasons, 3)
        if expected_reasons
        else None,
        "reasons_precision": round(found / given_reasons, 3) if given_reasons else None,
    }
    hit = accuracy >= float(case.scoring.get("min_accuracy", 0.8))
    return ("hit" if hit else "miss"), details


def draw_square_quality(case: Case, runs: List[Run]):
    image = _read_image(_input(case, "overview"))
    squares = case.data.get("squares") or []
    labels = case.key.get("labels") or {}
    cols = 6
    rows = int(np.ceil(len(squares) / cols)) or 1
    fig, axes = _figure(cols * 2.2, rows * 2.6, nrows=rows, ncols=cols, squeeze=False)
    half = 200
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sq in zip(axes.ravel(), squares):
        x, y = int(sq["x_px"]), int(sq["y_px"])
        crop = image[max(0, y - half) : y + half, max(0, x - half) : x + half]
        _show(ax, crop)
        name = str(sq["square"])
        want = square_call(labels.get(name, "?"))
        said = []
        for result, answer in runs:
            value = ((answer or {}).get("labels") or {}).get(name)
            said.append(square_call(value)["sites"][0] if value else "-")
        wrong = any(s not in want["sites"] for s in said)
        why = ", ".join(r.replace("_", " ") for r in want["reasons"])
        ax.set_title(
            f"{name}: {' | '.join(want['sites'])}"
            + (f" ({why})" if why else "")
            + "\n"
            + " ".join(said),
            fontsize=7,
            color="red" if wrong else "black",
        )
    return fig


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


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    shown = {str(s.get("square")) for s in case.data.get("squares") or []}
    labels = key.get("labels") or {}
    problems = [] if labels else ["key has no labels"]
    for name, label in labels.items():
        call = square_call(label)
        for site in call["sites"]:
            if site not in SITES:
                problems.append(f"square {name}: site {site!r} is not good or bad")
        for reason in call["reasons"]:
            if reason not in SQUARE_REASONS:
                problems.append(f"square {name}: unknown reason {reason!r}")
        if str(name) not in shown:
            problems.append(f"square {name} is keyed but not in squares")
    return problems


def summary(d: Dict[str, Any]) -> str:
    if "total" not in d:
        return ""
    text = f"{d['correct']} of {d['total']} labels right"
    if d.get("bad_called_good"):
        text += f", bad called good: {', '.join(d['bad_called_good'])}"
    return text


def key_text(key: Dict[str, Any]) -> str:
    return f"{len(key.get('labels') or {})} squares labelled"


TASK = EvalTask(
    name="square_quality",
    title="Label each numbered grid square",
    skill="screen-grids",
    answer_shape={
        "labels": {
            "<square>": {"site": " | ".join(SITES), "reasons": list(SQUARE_REASONS)}
        },
        "reason": "...",
    },
    answer_notes=(
        "site is whether you would put a lamella in the square: good or bad. "
        "reasons lists what you see that matters, from those shown (any that "
        "apply, or none); a good site can have reasons too, such as cells "
        "near a bar with room left for a lamella."
    ),
    score=score_square_quality,
    check_key=check_key,
    summary=summary,
    draw=draw_square_quality,
    key_text=key_text,
    disagreements=_square_items,
)
