"""Square finding: every grid square on an overview, its pitch and angle."""

import itertools
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.review import _cluster, _kept, _repeat, _squares_found
from fibsem.ai.evals.scoring import InvalidAnswer, _point
from fibsem.ai.evals.tasks.base import EvalTask, Run
from fibsem.ai.evals.visual import KEY, _figure, _input, _read_image, _show, _shrink


def match_squares(points: List[tuple], squares: List[Dict[str, Any]], radius: float):
    """Each point paired with at most one square and each square with at most
    one point, nearest pairs first, within *radius*."""
    pairs = sorted(
        (math.hypot(x - sq["x_px"], y - sq["y_px"]), i, j)
        for i, (x, y) in enumerate(points)
        for j, sq in enumerate(squares)
    )
    used_points, used_squares, matched = set(), set(), {}
    for d, i, j in pairs:
        if d > radius:
            break
        if i in used_points or j in used_squares:
            continue
        used_points.add(i)
        used_squares.add(j)
        matched[i] = j
    return matched


def placed_scale(
    points: List[Tuple[float, float]],
    squares: List[Dict[str, Any]],
    matched: Dict[int, int],
    pitch: float,
) -> Optional[float]:
    """How far apart the agent put neighbouring squares, as a fraction of the
    key's spacing: the median, over pairs of neighbours in the key both
    matched, of their distance in the answer over their distance in the key.
    None when no such pair was matched."""
    at = {j: points[i] for i, j in matched.items()}
    ratios = []
    for j, k in itertools.combinations(sorted(at), 2):
        a, b = squares[j], squares[k]
        apart = math.hypot(a["x_px"] - b["x_px"], a["y_px"] - b["y_px"])
        if 0 < apart <= 1.2 * pitch:
            (x1, y1), (x2, y2) = at[j], at[k]
            ratios.append(math.hypot(x1 - x2, y1 - y2) / apart)
    return statistics.median(ratios) if ratios else None


def score_square_finding(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    found = answer.get("squares")
    if not isinstance(found, list):
        raise InvalidAnswer("no squares list")
    points = [_point(p) for p in found]
    squares = key["squares"]
    sc = case.scoring
    pitch = float(key["pitch_px"])
    radius = float(sc.get("match_radius_px", pitch / 3))
    matched = match_squares(points, squares, radius)
    # Optional squares (cut by a tile edge, or hidden) count when found and
    # are not missed when not.
    required = [j for j, sq in enumerate(squares) if not sq.get("optional")]
    found_required = len(set(matched.values()) & set(required))
    recall = found_required / len(required) if required else 1.0
    precision = len(matched) / len(points) if points else 0.0
    details: Dict[str, Any] = {
        "found": len(points),
        "matched": len(matched),
        "required_found": found_required,
        "required_total": len(required),
        "extra": len(points) - len(matched),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
    }
    ok = recall >= float(sc.get("min_recall", 0.8)) and precision >= float(
        sc.get("min_precision", 0.9)
    )
    if sc.get("lattice") is False:
        # a tilted view foreshortens the grid: one pitch doesn't fit it
        details["lattice"] = "not scored"
        return ("hit" if ok else "miss"), details
    try:
        # The pitch of the squares placed, not the number written: an agent
        # can place every square right and still write the pitch measured
        # along the image axis.
        written_error = abs(float(answer["pitch_px"]) - pitch) / pitch
        scale = placed_scale(points, squares, matched, pitch)
        pitch_error = written_error if scale is None else abs(scale - 1)
        # Lattice directions repeat every 90 degrees.
        turn = (float(answer["angle_deg"]) - float(key["angle_deg"])) % 90
        angle_error = min(turn, 90 - turn)
    except (KeyError, TypeError, ValueError):
        details["lattice"] = "no pitch_px or angle_deg"
        return "miss", details
    details.update(
        pitch_error=round(pitch_error, 3),
        written_pitch_error=round(written_error, 3),
        angle_error_deg=round(angle_error, 1),
    )
    ok = (
        ok
        and pitch_error <= float(sc.get("pitch_tolerance", 0.1))
        and angle_error <= float(sc.get("angle_tolerance_deg", 5))
    )
    return ("hit" if ok else "miss"), details


def draw_square_finding(case: Case, runs: List[Run]):
    image, step = _shrink(_read_image(_input(case, "overview")))
    h, w = image.shape
    fig, axes = _figure(
        12, 4.8 * len(runs) or 4.8, nrows=max(1, len(runs)), squeeze=False
    )
    squares = case.key["squares"]
    radius = float(case.scoring.get("match_radius_px", float(case.key["pitch_px"]) / 3))
    for ax, (result, answer) in zip(axes[:, 0], runs):
        _show(ax, image, extent=(0, w * step, h * step, 0))
        points = []
        for p in (answer or {}).get("squares") or []:
            try:
                points.append((float(p["x_px"]), float(p["y_px"])))
            except (KeyError, TypeError, ValueError):
                pass
        matched = match_squares(points, squares, radius)
        found = set(matched.values())
        for j, sq in enumerate(squares):
            colour = KEY if j in found else ("#999999" if sq.get("optional") else "red")
            ax.plot(sq["x_px"], sq["y_px"], "o", mfc="none", mec=colour, ms=9, mew=1.5)
        for i, (x, y) in enumerate(points):
            ax.plot(
                x, y, "+", color="#1f77b4" if i in matched else "orange", ms=8, mew=2
            )
        score = (result.get("scores") or [{}])[0]
        d = score.get("details", {})
        ax.set_title(
            f"r{result['repeat']}: {score.get('result')}  recall {d.get('recall')}  "
            f"precision {d.get('precision')}  pitch {(answer or {}).get('pitch_px')} "
            f"(key {case.key['pitch_px']})  angle {(answer or {}).get('angle_deg')} "
            f"(key {case.key['angle_deg']})",
            fontsize=9,
        )
    axes[0, 0].text(
        0.01,
        0.01,
        "circles: key (green found, red missed, grey optional); "
        "+: agent (blue matched, orange extra)",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6),
    )
    return fig


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


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    squares = key.get("squares") or []
    problems = [] if squares else ["key has no squares"]
    problems += [
        f"key square without x_px, y_px: {square}"
        for square in squares
        if not {"x_px", "y_px"} <= set(square)
    ]
    problems += [f"key has no {n}" for n in ("pitch_px", "angle_deg") if n not in key]
    return problems


def summary(d: Dict[str, Any]) -> str:
    if "recall" not in d:
        return ""
    return f"found {d['recall']:.0%} of squares, {d.get('extra', 0)} extra"


TASK = EvalTask(
    name="square_finding",
    title="Find every grid square on an overview",
    skill="screen-grids",
    answer_shape={
        "squares": [{"x_px": 0, "y_px": 0}],
        "pitch_px": 0,
        "angle_deg": 0,
        "reason": "...",
    },
    answer_notes=(
        "pitch_px is the centre-to-centre distance between neighbouring "
        "squares along a grid bar (not along the image's x or y); angle_deg "
        "is a bar's direction in the image."
    ),
    score=score_square_finding,
    check_key=check_key,
    summary=summary,
    draw=draw_square_finding,
    disagreements=_square_finding_items,
)
