"""Site selection: pick squares for lamellae on a grid overview."""

import math
from typing import Any, Dict, List

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.review import _kept, _repeat
from fibsem.ai.evals.scoring import InvalidAnswer, _point
from fibsem.ai.evals.tasks.base import EvalTask, Run
from fibsem.ai.evals.visual import (
    KEY,
    _colour,
    _figure,
    _input,
    _legend,
    _read_image,
    _show,
    _shrink,
)


def score_site_selection(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    picks = answer.get("picks")
    if not isinstance(picks, list):
        raise InvalidAnswer("no picks list")
    radius = float(key.get("hit_radius_px", case.scoring.get("hit_radius_px", 100)))
    picks = picks[: int(case.scoring.get("max_picks", 10))]
    good, bad = key.get("good") or [], key.get("bad") or []

    def nearest(x, y, squares):
        best = None
        for i, square in enumerate(squares):
            d = math.hypot(x - square["x_px"], y - square["y_px"])
            if d <= radius and (best is None or d < best[1]):
                best = (i, d)
        return best

    found, bad_picks, unkeyed, per_pick = set(), [], 0, []
    for n, pick in enumerate(picks):
        x, y = _point(pick)
        if (hit := nearest(x, y, bad)) is not None:
            bad_picks.append(n)
            per_pick.append({"pick": n, "bad": bad[hit[0]].get("square", hit[0])})
        elif (hit := nearest(x, y, good)) is not None:
            found.add(hit[0])
            per_pick.append({"pick": n, "good": good[hit[0]].get("square", hit[0])})
        else:
            unkeyed += 1
            per_pick.append({"pick": n})
    # The share of picks that are distinct good sites: a square picked twice
    # is one site, and an unkeyed pick counts against.
    fraction = len(found) / len(picks) if picks else 0.0
    wanted = float(case.scoring.get("min_good_fraction", 0.5))
    details = {
        "good_fraction": round(fraction, 3),
        "min_good_fraction": wanted,
        "picks": len(picks),
        "good_found": len(found),
        "good_total": len(good),
        "bad_picks": len(bad_picks),
        "unkeyed_picks": unkeyed,
        "costly": len(bad_picks),
        "per_pick": per_pick,
    }
    return ("hit" if found and fraction >= wanted else "miss"), details


def draw_site_selection(case: Case, runs: List[Run]):
    image, step = _shrink(_read_image(_input(case, "overview")))
    fig, ax = _figure(12, 5.5)
    h, w = image.shape
    _show(ax, image, extent=(0, w * step, h * step, 0))
    radius = float(case.key.get("hit_radius_px", 100))
    from matplotlib.patches import Circle

    for square in case.key.get("good") or []:
        ax.add_patch(
            Circle(
                (square["x_px"], square["y_px"]), radius, fill=False, color=KEY, lw=1.5
            )
        )
    for square in case.key.get("bad") or []:
        ax.add_patch(
            Circle((square["x_px"], square["y_px"]), radius, fill=False, color="red")
        )
    for n, (_, answer) in enumerate(runs):
        for i, pick in enumerate((answer or {}).get("picks") or []):
            try:
                x, y = float(pick["x_px"]), float(pick["y_px"])
            except (KeyError, TypeError, ValueError):
                continue
            ax.plot(x, y, "x", color=_colour(n), ms=7, mew=2)
            ax.annotate(
                str(i + 1),
                (x, y),
                (4, 4),
                textcoords="offset points",
                color=_colour(n),
                fontsize=7,
            )
    _legend(ax, runs, "keyed good square")
    return fig


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


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    return [
        f"key square without x_px, y_px: {square}"
        for square in key.get("good", []) + key.get("bad", [])
        if not {"x_px", "y_px"} <= set(square)
    ]


def summary(d: Dict[str, Any]) -> str:
    if "picks" not in d:
        return ""
    on_good = sum(1 for p in d.get("per_pick", []) if "good" in p)
    text = f"{on_good} of {d['picks']} picks on good squares"
    if d.get("bad_picks"):
        text += f", {d['bad_picks']} on bad"
    return text


def key_text(key: Dict[str, Any]) -> str:
    return f"{len(key.get('good') or [])} good squares, {len(key.get('bad') or [])} bad"


TASK = EvalTask(
    name="site_selection",
    title="Pick sites for lamellae on a grid overview",
    skill="screen-grids",
    answer_shape={"picks": [{"x_px": 0, "y_px": 0, "reason": "..."}]},
    score=score_site_selection,
    check_key=check_key,
    summary=summary,
    draw=draw_site_selection,
    key_text=key_text,
    disagreements=_site_items,
)
