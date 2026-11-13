"""Setup placement: where the lamella goes on a FIB image at the milling angle."""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from fibsem.ai.evals.cases import Case, bad_mask
from fibsem.ai.evals.scoring import _point, score_case
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import (
    KEY,
    _colour,
    _figure,
    _input,
    _legend,
    _read_image,
    _show,
)


def _mask(case: Case, rel: str) -> np.ndarray:
    with Image.open(case.path / rel) as image:
        return np.asarray(image.convert("L")) > 0


def score_setup_placement(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    """Hit when the lamella's centre is where the whole lamella sits on the
    target with both trenches clear (``ok_mask``); else a miss, counted as
    costly where a trench cuts a bar or the lamella misses every cell
    (``bad_mask``)."""
    x, y = _point(answer)
    ok, bad = _mask(case, key["ok_mask"]), _mask(case, bad_mask(key))
    size = float(pixel_size_um(case))
    target = key["target_px"]
    details: Dict[str, Any] = {
        "to_target_um": round(math.hypot(x - target["x"], y - target["y"]) * size, 3)
    }
    col, row = int(round(x)), int(round(y))
    inside = 0 <= row < ok.shape[0] and 0 <= col < ok.shape[1]
    if not inside:
        details["outside_image"] = True
        return "miss", details
    rows, cols = np.nonzero(ok)
    if rows.size:
        nearest = np.hypot(cols - x, rows - y).min()
        details["to_ok_um"] = round(float(nearest) * size, 3)
    if ok[row, col]:
        return "hit", details
    details["costly"] = int(bool(bad[row, col]))
    return "miss", details


def draw_setup_placement(case: Case, runs: List[Run]):
    """The FIB image with the key's masks tinted (green: ok, red: bad) and
    each repeat's lamella centre as an x."""
    from PIL import Image

    image = _read_image(_input(case, "fib_image"))
    fig, ax = _figure(10, 7)
    h, w = image.shape
    _show(ax, image, extent=(0, w, h, 0))
    for names, colour in (
        (("ok_mask",), (0.17, 0.63, 0.17)),
        (("bad_mask", "unsafe_mask"), (0.84, 0.15, 0.16)),
    ):
        path = next((case.key[n] for n in names if case.key.get(n)), None)
        if not path or not (case.path / path).is_file():
            continue
        mask = np.array(Image.open(case.path / path)) > 0
        tint = np.zeros((*mask.shape, 4))
        tint[mask] = (*colour, 0.35)
        ax.imshow(tint, extent=(0, w, h, 0))
    target = case.key.get("target_px") or {}
    if "x" in target:
        ax.plot(target["x"], target["y"], "o", mfc="none", mec=KEY, ms=16, mew=2.5)
    results = []
    for n, (result, answer) in enumerate(runs):
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
        results.append(
            f"r{result['repeat']} {score_case(case, answer or {})[0].result}"
        )
    ax.set_title(
        "FIB image. green: ok centres; red: a trench or the lamella on a bar"
        + (f"\n{'; '.join(results)}" if results else ""),
        fontsize=9,
    )
    _legend(ax, runs)
    return fig


def pixel_size_um(case: Case) -> Optional[float]:
    return (case.data.get("inputs") or {}).get("fib_pixel_size_um")


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    problems = []
    for name in ("ok_mask", "bad_mask"):
        rel = key.get(name) if name == "ok_mask" else bad_mask(key)
        if not rel:
            problems.append(f"key has no {name}")
        elif not str(rel).startswith("reference/"):
            problems.append(f"key {name} must sit under reference/, out of sight")
        elif not (case.path / rel).is_file():
            problems.append(f"key {name} {rel} is missing")
    if not {"x", "y"} <= set(key.get("target_px") or {}):
        problems.append("key has no target_px x, y")
    if not pixel_size_um(case):
        problems.append("no inputs fib_pixel_size_um")
    return problems


def summary(d: Dict[str, Any]) -> str:
    return f"{d['to_target_um']} um from the target" if "to_target_um" in d else ""


TASK = EvalTask(
    name="setup_placement",
    title="Place the lamella on a FIB image at the milling angle",
    skill="autolamella",
    answer_shape={"x_px": 0, "y_px": 0, "reason": "..."},
    answer_notes=(
        "x_px, y_px is the lamella's centre on the FIB image; the trenches sit "
        "above and below it as the protocol sets."
    ),
    score=score_setup_placement,
    check_key=check_key,
    summary=summary,
    draw=draw_setup_placement,
    pixel_size_um=pixel_size_um,
    sims=(Sim("placement", "fibsem.ai.evals.sim_placement"),),
)
