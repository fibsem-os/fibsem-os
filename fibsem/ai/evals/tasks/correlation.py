"""Correlation: place the FM target on the FIB image."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.review import point_items
from fibsem.ai.evals.scoring import _point, score_case
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import (
    KEY,
    _colour,
    _figure,
    _fm_overview,
    _input,
    _legend,
    _mark_points,
    _read_image,
    _show,
    log,
)


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


def _pairs(answer: Dict[str, Any], side: str) -> List[Tuple[float, float]]:
    points = []
    for f in answer.get("fiducials") or []:
        try:
            points.append((float(f[side][0]), float(f[side][1])))
        except (KeyError, IndexError, TypeError, ValueError):
            points.append((np.nan, np.nan))
    return points


def draw_correlation(case: Case, runs: List[Run]):
    """The FM stack (when it can be read) beside the FIB image: the key's
    target and fiducials in green, each repeat's in its colour, fiducial
    pairs numbered alike on both sides."""
    fib = _read_image(_input(case, "fib_image"))
    key = case.key
    fm = None
    if case.data.get("inputs", {}).get("fm_stack"):
        try:
            step = 4
            fm = (_fm_overview(_input(case, "fm_stack"), step), step)
        except Exception as e:  # the FIB side still says most of it
            log.warning("No FM panel for %s: %s", case.id, e)
    fig, axes = _figure(16 if fm else 10, 7, ncols=2 if fm else 1, squeeze=False)
    axes = axes[0]
    if fm:
        image, step = fm
        ax = axes[0]
        h, w = image.shape[:2]
        ax.imshow(image, extent=(0, w * step, h * step, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "FM max projection (green: label, magenta: reflection)", fontsize=9
        )
        _mark_points(ax, [p[:2] for p in key.get("fiducials_fm_px") or []], KEY, "s", 9)
        target = key.get("target_fm") or {}
        if "x_px" in target:
            ax.plot(
                target["x_px"], target["y_px"], "o", mfc="none", mec=KEY, ms=16, mew=2.5
            )
        for n, (_, answer) in enumerate(runs):
            _mark_points(ax, _pairs(answer or {}, "fm_px"), _colour(n), "s", 7)
    ax = axes[-1]
    _show(ax, fib, extent=(0, fib.shape[1], fib.shape[0], 0))
    _mark_points(ax, key.get("fiducials_fib_px") or [], KEY, "s", 9)
    target = key["target_fib_px"]
    ax.plot(target["x"], target["y"], "o", mfc="none", mec=KEY, ms=16, mew=2.5)
    for n, (_, answer) in enumerate(runs):
        answer = answer or {}
        _mark_points(ax, _pairs(answer, "fib_px"), _colour(n), "s", 7)
        try:
            ax.plot(
                float(answer["x_px"]),
                float(answer["y_px"]),
                "x",
                color=_colour(n),
                ms=14,
                mew=3,
            )
        except (KeyError, TypeError, ValueError):
            pass
    errors = []
    for result, answer in runs:
        d = score_case(case, answer or {})[0].details
        if "across_um" in d:
            errors.append(
                f"r{result['repeat']} across {d['across_um']}, along {d['along_um']}"
            )
        elif "error_um" in d:
            errors.append(f"r{result['repeat']} {d['error_um']}")
    ax.set_title(
        "FIB image. circle/x: target; squares: fiducial pairs, numbered"
        + (f"\nerror (um): {'; '.join(errors)}" if errors else ""),
        fontsize=9,
    )
    _legend(ax, runs)
    return fig


def pixel_size_um(case: Case) -> Optional[float]:
    return (case.data.get("inputs") or {}).get("fib_pixel_size_um")


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    problems = []
    if not {"x", "y"} <= set(key.get("target_fib_px") or {}):
        problems.append("key has no target_fib_px x, y")
    if not pixel_size_um(case):
        problems.append("no inputs fib_pixel_size_um")
    return problems


def summary(d: Dict[str, Any]) -> str:
    if "across_um" in d:
        return f"{d['across_um']} um across, {d['along_um']} um along"
    return f"{d['error_um']} um off" if "error_um" in d else ""


def key_point(key: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    target = key.get("target_fib_px") or {}
    try:
        return float(target["x"]), float(target["y"])
    except (KeyError, TypeError, ValueError):
        return None


def set_point(key: Dict[str, Any], x: float, y: float) -> None:
    key.setdefault("target_fib_px", {}).update(x=x, y=y)


TASK = EvalTask(
    name="correlation",
    title="Place the FM target on the FIB image",
    skill="autolamella-correlative",
    answer_shape={
        "x_px": 0,
        "y_px": 0,
        "fiducials": [{"fm_px": [0, 0, 0], "fib_px": [0, 0]}],
        "fit_rms_um": 0.0,
        "reason": "...",
    },
    score=score_correlation,
    check_key=check_key,
    summary=summary,
    draw=draw_correlation,
    disagreements=point_items,
    pixel_size_um=pixel_size_um,
    key_point=key_point,
    set_point=set_point,
    sims=(Sim("correlation", "fibsem.ai.evals.sim_correlation"),),
)
