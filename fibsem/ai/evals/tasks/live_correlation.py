"""Live correlation: on the simulator, correlate an FM target into the FIB
and put a lamella's point of interest on it
(``fibsem.ai.evals.live.sim_live_correlation``). Scored on the lamella's
point of interest as the run left it, against the target's true place in the
FIB's view from the lamella's milling pose: across the lamella (the image's
y) it must be within the across tolerance, along it within the along one.
Without a point of interest set, the answer's own point is scored. Moving
the stage voids the key, which is from the set-up's pose.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.base import EvalTask, Sim
from fibsem.ai.evals.visual import THEME, _figure

POSE = ("x", "y", "z")


def _item(answer: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for item in (answer.get("record") or {}).get("items") or []:
        if item.get("name") == name:
            return item
    return None


def _moved(item: Dict[str, Any], key: Dict[str, Any], tolerance_m: float) -> bool:
    """Whether none of the item's poses is still the set-up's milling pose."""
    want = key["milling_pose"]
    for pose in (item.get("poses") or {}).values():
        if not isinstance(pose, dict):
            continue
        try:
            far = max(abs(float(pose[k]) - float(want[k])) for k in POSE)
        except (KeyError, TypeError, ValueError):
            continue
        if far <= tolerance_m and abs(float(pose.get("t", 0)) - want["t"]) < 1e-3:
            return False
    return True


def _poi(item: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    poi = (item or {}).get("poi")
    if isinstance(poi, dict) and "x" in poi and "y" in poi:
        return float(poi["x"]), float(poi["y"])
    return None


def _answered(answer: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    point = answer.get("target_m")
    try:
        return float(point[0]), float(point[1])
    except (TypeError, IndexError, ValueError):
        return None


def score_live_correlation(
    case: Case, answer: Any, key: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(answer, dict):
        raise InvalidAnswer("the answer is not an object")
    if not (answer.get("record") or {}).get("items"):
        raise InvalidAnswer("the run has no record of its items: was it a live run?")
    item = _item(answer, key["item"])
    if item is None:
        raise InvalidAnswer(f"the run's record has no {key['item']}")
    tolerance = float(case.scoring.get("same_pose_um", 0.2)) * 1e-6
    details: Dict[str, Any] = {}
    if _moved(item, key, tolerance):
        details["verdict"] = "moved the stage off the lamella's pose"
        return "miss", details
    point, source = _poi(item), "point of interest"
    if point is None or point == (0.0, 0.0):
        point, source = _answered(answer), "answer"
    if point is None:
        raise InvalidAnswer(
            "no point of interest set and no target_m [x, y] (m) in the answer"
        )
    target = key["target_poi_m"]
    along_um = (point[0] - float(target[0])) * 1e6
    across_um = (point[1] - float(target[1])) * 1e6
    across_tol = float(case.scoring.get("across_tolerance_um", 0.75))
    along_tol = float(case.scoring.get("along_tolerance_um", 2 * across_tol))
    hit = abs(across_um) <= across_tol and abs(along_um) <= along_tol
    details.update(
        scored=source,
        across_um=round(abs(across_um), 3),
        along_um=round(abs(along_um), 3),
        across_tolerance_um=across_tol,
        along_tolerance_um=along_tol,
        error_um=round(float(np.hypot(across_um, along_um)), 3),
        verdict=f"{abs(across_um):.2f} um across, {abs(along_um):.2f} um along",
    )
    return ("hit" if hit else "miss"), details


def summary(d: Dict[str, Any]) -> str:
    if "across_um" in d:
        return f"{d['across_um']} um across, {d['along_um']} um along ({d['scored']})"
    return d.get("verdict", "")


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    problems = []
    for name in ("item", "target_poi_m", "milling_pose", "burns"):
        if not key.get(name):
            problems.append(f"key has no {name}")
    if len(key.get("target_poi_m") or []) != 2:
        problems.append("key's target_poi_m is not [x, y]")
    return problems


def key_text(key: Dict[str, Any]) -> str:
    x, y = (v * 1e6 for v in key.get("target_poi_m") or (0, 0))
    return (
        f"{key.get('target_kind')} at {x:+.2f}, {y:+.2f} um from the FIB view's "
        f"centre, {len(key.get('burns') or [])} burns"
    )


def draw_live_correlation(case: Case, runs: List[Any], **_: Any):
    """Each repeat's point against the target, in microns from the target,
    with the tolerance box."""
    from matplotlib.patches import Rectangle

    key = case.key
    fig, ax = _figure(5, 5)
    across = float(case.scoring.get("across_tolerance_um", 0.75))
    along = float(case.scoring.get("along_tolerance_um", 1.0))
    ax.add_patch(
        Rectangle(
            (-along, -across), 2 * along, 2 * across, fill=False, edgecolor=THEME.good
        )
    )
    ax.plot([0], [0], "+", color=THEME.good, markersize=14)
    target = key["target_poi_m"]
    for result, answer in runs:
        item = _item(answer or {}, key["item"])
        point = _poi(item) or _answered(answer or {})
        if point is None:
            continue
        dx = (point[0] - target[0]) * 1e6
        dy = (point[1] - target[1]) * 1e6
        ax.plot([dx], [dy], "x", markersize=10)
        ax.annotate(f"r{result.get('repeat')}", (dx, dy), fontsize=8)
    ax.set_xlabel("along the lamella (um)")
    ax.set_ylabel("across it (um)")
    ax.set_aspect("equal")
    ax.set_title(key_text(key), fontsize=8)
    return fig


TASK = EvalTask(
    name="live_correlation",
    title="Correlate an FM target into the FIB on the simulator, and place a lamella on it",
    skill="autolamella-correlative",
    answer_shape={
        "item": "Lam-01",
        "fib_image": "path of the FIB image you correlated on",
        "target_px": [768.0, 512.0],
        "target_m": [0.0, 0.0],
        "poi_set": True,
        "reason": "...",
    },
    answer_notes=(
        "target_m is the target in metres from the FIB image's centre, +y up, "
        "as a point of interest is. Scored on the lamella's point of interest "
        "as you leave it (target_m when none is set): across the lamella "
        "within 0.75 um, along it within 1 um. Do not move the stage."
    ),
    score=score_live_correlation,
    check_key=check_key,
    summary=summary,
    draw=draw_live_correlation,
    key_text=key_text,
    sims=(Sim("live-correlation", "fibsem.ai.evals.live.sim_live_correlation"),),
)
