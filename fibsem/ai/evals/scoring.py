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

A case with parts is answered as ``{"parts": {"a": {...}, "b": {...}}}``.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fibsem.ai.evals.cases import Case, pass_names, pixel_size_um

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


def _yes(value: Any) -> Optional[bool]:
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
    usable_key = _yes(key.get("usable"))
    usable = _yes(answer.get("usable"))
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
    error_um = math.hypot(x - target["x"], y - target["y"]) * float(pixel_size_um(case))
    tol = float(case.scoring.get("xy_tolerance_um", 0.5))
    details = {"error_um": round(error_um, 3), "tolerance_um": tol}
    if "fit_rms_um" in answer:
        details["reported_rms_um"] = answer["fit_rms_um"]
    return ("hit" if error_um <= tol else "miss"), details


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
    want = names.index(key["stop_after"])
    details = {
        "stopped_after": names[stop] if stop is not None else None,
        "key": key["stop_after"],
    }
    if stop == want:
        return "hit", details
    if stop is not None and stop < want:
        details["passes_early"] = want - stop
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
        said = _yes(present[name])
        if said != _yes(expected):
            wrong.append(name)
            if said is True and _yes(expected) is False:
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


SCORERS: Dict[str, Callable[[Case, Dict[str, Any], Dict[str, Any]], tuple]] = {
    "site_selection": score_site_selection,
    "fm_target": score_fm_target,
    "correlation": score_correlation,
    "polishing_stop": score_polishing_stop,
    "target_present": score_target_present,
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
