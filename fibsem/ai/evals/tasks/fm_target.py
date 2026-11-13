"""FM target: whether an FM stack has a target at the site, and where."""

import math
from typing import Any, Dict, List, Optional, Tuple

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.review import _xy, point_items
from fibsem.ai.evals.scoring import _point, yes_no
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import KEY, _colour, _figure, _legend, _show, _stack_crop


def score_fm_target(case: Case, answer: Dict[str, Any], key: Dict[str, Any]) -> tuple:
    usable_key = yes_no(key.get("usable"))
    usable = yes_no(answer.get("usable"))
    details: Dict[str, Any] = {"usable": usable, "usable_key": usable_key}
    if usable_key is False:
        # Calling a target where there is none sends the milling to nothing.
        details["costly"] = int(bool(usable))
        return ("miss" if usable else "hit"), details
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


def draw_fm_target(case: Case, runs: List[Run]):
    target = case.key.get("target") or {}
    x, y = target.get("x_px"), target.get("y_px")
    if x is None:
        pts = [a for _, a in runs if a and "x_px" in a]
        if not pts:
            return None
        x, y = pts[0]["x_px"], pts[0]["y_px"]
    crop, extent = _stack_crop(case, float(x), float(y), 400)
    fig, ax = _figure(7, 7)
    _show(ax, crop, extent=extent)
    if "x_px" in target:
        ax.plot(target["x_px"], target["y_px"], "o", mfc="none", mec=KEY, ms=14, mew=2)
    for n, (_, answer) in enumerate(runs):
        try:
            ax.plot(
                float(answer["x_px"]),
                float(answer["y_px"]),
                "x",
                color=_colour(n),
                ms=10,
                mew=2,
            )
        except (KeyError, TypeError, ValueError):
            pass
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"max projection, key usable: {case.key.get('usable')}", fontsize=9)
    _legend(ax, runs)
    return fig


def pixel_size_um(case: Case) -> Optional[float]:
    return (case.data.get("stack") or {}).get("pixel_size_um")


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    problems = []
    target = key.get("target") or {}
    if "usable" not in key:
        problems.append("key has no usable")
    if key.get("usable") is True and not {"x_px", "y_px"} <= set(target):
        problems.append("key target has no x_px, y_px")
    if not pixel_size_um(case):
        problems.append("no stack pixel_size_um")
    return problems


def summary(d: Dict[str, Any]) -> str:
    if "xy_error_um" in d:
        return f"{d['xy_error_um']} um from the target"
    return f"usable: {d['usable']}" if "usable" in d else ""


def key_point(key: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    return _xy(key.get("target") or {})


def set_point(key: Dict[str, Any], x: float, y: float) -> None:
    key.setdefault("target", {}).update(x_px=x, y_px=y)


TASK = EvalTask(
    name="fm_target",
    title="Find the target in an FM stack",
    skill="autolamella-correlative",
    answer_shape={
        "usable": "yes | no | unsure",
        "x_px": 0,
        "y_px": 0,
        "z_slice": 0,
        "reason": "...",
    },
    score=score_fm_target,
    check_key=check_key,
    summary=summary,
    draw=draw_fm_target,
    disagreements=point_items,
    pixel_size_um=pixel_size_um,
    key_point=key_point,
    set_point=set_point,
    sims=(Sim("fm-target", "fibsem.ai.evals.sim_fm_target"),),
)
