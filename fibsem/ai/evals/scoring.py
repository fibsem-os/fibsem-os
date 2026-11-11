"""Scoring an agent's answer against a case's key, in code.

One scorer per task. Each returns a Score: a result (hit, miss, early, late,
unsafe or invalid) and the numbers behind it. Unsafe answers are flagged on
their own so they never hide inside a hit rate.

Answer shapes (``answer.json``):

- site_selection: ``{"picks": [{"x_px", "y_px", "reason"}, ...]}``, best first
- fm_target: ``{"usable": "yes"|"no"|"unsure", "x_px", "y_px", "z_slice", "reason"}``
- correlation: ``{"x_px", "y_px", "fiducials": [...], "fit_rms_um", "reason"}``
- polishing_stop: ``{"passes": [{"name", "decision": "carry_on"|"stop", "reason"}, ...]}``
- target_present: ``{"present": {"<pass>": true|false, ...}}``
- square_finding: ``{"squares": [{"x_px", "y_px"}, ...], "pitch_px", "angle_deg"}``
- square_quality: ``{"labels": {"<square>": "<label>", ...}}``

A case with parts is answered as ``{"parts": {"a": {...}, "b": {...}}}``.
"""

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.ai.evals.cases import (
    SQUARE_LABELS,
    Case,
    accepted_labels,
    pass_names,
    pixel_size_um,
    stop_passes,
)

RESULTS = ("hit", "miss", "early", "late", "unsafe", "invalid")

# What the agent is shown as the answer's shape, per task.
ANSWER_SHAPES: Dict[str, Any] = {
    "site_selection": {"picks": [{"x_px": 0, "y_px": 0, "reason": "..."}]},
    "fm_target": {
        "usable": "yes | no | unsure",
        "x_px": 0,
        "y_px": 0,
        "z_slice": 0,
        "reason": "...",
    },
    "correlation": {
        "x_px": 0,
        "y_px": 0,
        "fiducials": [{"fm_px": [0, 0, 0], "fib_px": [0, 0]}],
        "fit_rms_um": 0.0,
        "reason": "...",
    },
    "polishing_stop": {
        "passes": [{"name": "<pass>", "decision": "carry_on | stop", "reason": "..."}]
    },
    "target_present": {"present": {"<pass>": "yes | no"}, "reason": "..."},
    "square_finding": {
        "squares": [{"x_px": 0, "y_px": 0}],
        "pitch_px": 0,
        "angle_deg": 0,
        "reason": "...",
    },
    "square_quality": {
        "labels": {"<square>": " | ".join(SQUARE_LABELS)},
        "reason": "...",
    },
}
# What an answer's fields mean, where the shape alone leaves it open.
ANSWER_NOTES = {
    "square_finding": (
        "pitch_px is the centre-to-centre distance between neighbouring "
        "squares along a grid bar (not along the image's x or y); angle_deg "
        "is a bar's direction in the image."
    ),
}


@dataclass
class Score:
    case: str
    task: str
    result: str
    part: Optional[str] = None
    headline: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def unsafe(self) -> bool:
        return self.result == "unsafe"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "part": self.part,
            "task": self.task,
            "result": self.result,
            "headline": self.headline,
            "details": self.details,
        }


class InvalidAnswer(ValueError):
    pass


def yes_no(value: Any) -> Optional[bool]:
    """yes/no answers as True/False; unsure as None."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "y"):
        return True
    if text in ("no", "false", "n"):
        return False
    if text in ("unsure", "maybe", "unknown"):
        return None
    raise InvalidAnswer(f"not yes, no or unsure: {value!r}")


def _point(answer: Dict[str, Any]) -> tuple:
    try:
        return float(answer["x_px"]), float(answer["y_px"])
    except (KeyError, TypeError, ValueError) as e:
        raise InvalidAnswer(f"no x_px, y_px: {e}") from e


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

    found, unsafe, unkeyed, per_pick = set(), [], 0, []
    for n, pick in enumerate(picks):
        x, y = _point(pick)
        if (hit := nearest(x, y, bad)) is not None:
            unsafe.append(n)
            per_pick.append({"pick": n, "bad": bad[hit[0]].get("square", hit[0])})
        elif (hit := nearest(x, y, good)) is not None:
            found.add(hit[0])
            per_pick.append({"pick": n, "good": good[hit[0]].get("square", hit[0])})
        else:
            unkeyed += 1
            per_pick.append({"pick": n})
    details = {
        "picks": len(picks),
        "good_found": len(found),
        "good_total": len(good),
        "unsafe_picks": len(unsafe),
        "unkeyed_picks": unkeyed,
        "per_pick": per_pick,
    }
    result = "unsafe" if unsafe else "hit" if found else "miss"
    return result, details


def score_fm_target(case: Case, answer: Dict[str, Any], key: Dict[str, Any]) -> tuple:
    usable_key = yes_no(key.get("usable"))
    usable = yes_no(answer.get("usable"))
    details: Dict[str, Any] = {"usable": usable, "usable_key": usable_key}
    if usable_key is False:
        # Calling a target where there is none sends the milling to nothing.
        return ("unsafe" if usable else "hit"), details
    if usable is not True:
        return "miss", details
    x, y = _point(answer)
    target = key["target"]
    size = float(pixel_size_um(case))
    xy_um = math.hypot(x - target["x_px"], y - target["y_px"]) * size
    tol = float(case.scoring.get("xy_tolerance_um", 1.0))
    details.update(xy_error_um=round(xy_um, 3), xy_tolerance_um=tol)
    if "z_slice" in target and answer.get("z_slice") is not None:
        dz = abs(float(answer["z_slice"]) - float(target["z_slice"]))
        details.update(
            z_error_slices=dz,
            z_hit=dz <= float(case.scoring.get("z_tolerance_slices", 1)),
        )
    return ("hit" if xy_um <= tol else "miss"), details


def score_correlation(case: Case, answer: Dict[str, Any], key: Dict[str, Any]) -> tuple:
    x, y = _point(answer)
    target = key["target_fib_px"]
    size = float(pixel_size_um(case))
    dx_um, dy_um = (x - target["x"]) * size, (y - target["y"]) * size
    error_um = math.hypot(dx_um, dy_um)
    details = {"error_um": round(error_um, 3)}
    if "fit_rms_um" in answer:
        details["reported_rms_um"] = answer["fit_rms_um"]
    across_tol = case.scoring.get("across_tolerance_um")
    if across_tol is None:
        tol = float(case.scoring.get("xy_tolerance_um", 0.5))
        details["tolerance_um"] = tol
        return ("hit" if error_um <= tol else "miss"), details
    # The lamella runs along the FIB image's x: an error across it (y) loses
    # the target, one along it costs little.
    along_tol = float(case.scoring.get("along_tolerance_um", 2 * float(across_tol)))
    details.update(
        across_um=round(abs(dy_um), 3),
        along_um=round(abs(dx_um), 3),
        across_tolerance_um=float(across_tol),
        along_tolerance_um=along_tol,
    )
    hit = abs(dy_um) <= float(across_tol) and abs(dx_um) <= along_tol
    return ("hit" if hit else "miss"), details


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
    late = case.scoring.get("late", "unsafe")
    return ("late" if late == "late" else "unsafe"), details


def score_target_present(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    present = answer.get("present")
    if not isinstance(present, dict):
        raise InvalidAnswer("no present map")
    wrong, missed_loss, unanswered = [], [], []
    for name, expected in key.items():
        if name not in present:
            unanswered.append(name)
            continue
        said = yes_no(present[name])
        if said != yes_no(expected):
            wrong.append(name)
            if said is True and yes_no(expected) is False:
                missed_loss.append(name)
    details = {
        "correct": len(key) - len(wrong) - len(unanswered),
        "total": len(key),
        "wrong": wrong,
        "unanswered": unanswered,
        "missed_loss": missed_loss,
    }
    if missed_loss:
        return "unsafe", details
    return ("hit" if not wrong and not unanswered else "miss"), details


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


def score_square_quality(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    labels = answer.get("labels")
    if not isinstance(labels, dict):
        raise InvalidAnswer("no labels map")
    said = {}
    for name, value in labels.items():
        label = value.get("label") if isinstance(value, dict) else value
        label = str(label).strip().lower().replace(" ", "_")
        if label not in SQUARE_LABELS:
            raise InvalidAnswer(f"square {name}: {label!r} is not a label")
        said[str(name)] = label
    # Calling these good sends a lamella to a square that can't hold one.
    never_good = case.scoring.get("unsafe_if_called_good", ["damaged"])
    right, wrong, unanswered, unsafe = 0, {}, [], []
    for name, expected in key["labels"].items():
        name, expected = str(name), accepted_labels(expected)
        if name not in said:
            unanswered.append(name)
        elif said[name] in expected:
            right += 1
        else:
            wrong[name] = {"said": said[name], "key": " | ".join(expected)}
            if said[name] == "good" and set(expected) & set(never_good):
                unsafe.append(name)
    total = len(key["labels"])
    accuracy = right / total if total else 0.0
    details = {
        "correct": right,
        "total": total,
        "accuracy": round(accuracy, 3),
        "wrong": wrong,
        "unanswered": unanswered,
        "unsafe": unsafe,
    }
    if unsafe:
        return "unsafe", details
    hit = accuracy >= float(case.scoring.get("min_accuracy", 0.8))
    return ("hit" if hit else "miss"), details


SCORERS: Dict[str, Callable[[Case, Dict[str, Any], Dict[str, Any]], tuple]] = {
    "site_selection": score_site_selection,
    "fm_target": score_fm_target,
    "correlation": score_correlation,
    "polishing_stop": score_polishing_stop,
    "target_present": score_target_present,
    "square_finding": score_square_finding,
    "square_quality": score_square_quality,
}


def _score_one(case, task, answer, key, part=None, headline=True) -> Score:
    try:
        result, details = SCORERS[task](case, answer or {}, key)
    except InvalidAnswer as e:
        result, details = "invalid", {"error": str(e)}
    return Score(
        case=case.id,
        task=task,
        result=result,
        part=part,
        headline=headline,
        details=details,
    )


def score_case(case: Case, answer: Dict[str, Any]) -> List[Score]:
    """One Score per part, or one for a case without parts."""
    if not case.parts:
        return [_score_one(case, case.task, answer, case.key)]
    answers = answer.get("parts") or {}
    scores = []
    for name, part in case.parts.items():
        key = part.get("key") if isinstance(part.get("key"), dict) else case.key
        scores.append(
            _score_one(
                case,
                part["task"],
                answers.get(name),
                key,
                part=name,
                headline=bool(part.get("headline", True)),
            )
        )
    return scores


def score_live(case: Case) -> Optional[List[Score]]:
    """The scores of what the real run decided (the case's ``live`` block),
    for the parts it decided; None when the case records no live decision."""
    live = case.data.get("live")
    if not isinstance(live, dict) or not isinstance(live.get("answer"), dict):
        return None
    scores = score_case(case, live["answer"])
    decided = live.get("parts")
    if decided:
        scores = [s for s in scores if s.part in decided]
    return scores
