"""Live site selection: on the simulator, take an overview of the loaded
grid, save the likely sites, go to each and say whether it is good for a
lamella. Scored where the sites really are, from the sample the simulator
drew (``fibsem.ai.evals.live.sim_sites``).

The sites' positions are the experiment's saved items, which the runner
puts in the answer as ``record`` after the run: what the agent saved, not
what it says it saved.
"""

import math
from typing import Any, Dict, List, Tuple

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import KEY, _colour, _figure, _legend

VERDICTS = ("good", "bad")
# How far a site may be from a cell body's edge and still be on it (m): a
# lamella is milled through the cell, and a site is saved at its centre.
ON_CELL_SLACK = 1e-6
BAD_SQUARES = ("damaged", "thick")
FLAT = math.radians(1.0)  # a pose this near no tilt is at the SEM orientation


def _frame_point(
    key: Dict[str, Any], x: float, y: float
) -> Tuple[int, int, float, float]:
    """The square a stage point is in and its offset from the square's
    centre (m), in the grid's frame."""
    frame = key["frame"]
    dx, dy = x - frame["origin"][0], y - frame["origin"][1]
    # u, v are the grid's axes as stage vectors per world metre
    (ux, uy), (vx, vy) = frame["u"], frame["v"]
    det = ux * vy - uy * vx
    lu = (dx * vy - dy * vx) / det
    lv = (ux * dy - uy * dx) / det
    pitch = frame["pitch"]
    i, j = int(round(lu / pitch)), int(round(lv / pitch))
    return i, j, lu - i * pitch, lv - j * pitch


def site_truth(key: Dict[str, Any], x: float, y: float) -> Dict[str, Any]:
    """What is really at stage (x, y): good when it is on a cell wholly on
    the film and clear of the bars, in a square not torn or iced; else bad,
    and why; unsure when its square's label is too close to call."""
    i, j, du, dv = _frame_point(key, x, y)
    frame = key["frame"]
    half = (frame["pitch"] - frame["bar_width"]) / 2
    squares = {tuple(q["index"]): q for q in key.get("squares") or []}
    square = squares.get((i, j))
    out: Dict[str, Any] = {"square": [i, j]}
    if square is None:
        return {**out, "truth": "unkeyed", "why": "outside the keyed squares"}
    out["square_label"] = square["label"]
    if max(abs(du), abs(dv)) > half:
        return {**out, "truth": "bad", "why": "on a grid bar"}
    cell = None
    for n, c in enumerate(key.get("cells") or []):
        d = math.hypot(c["x"] - x, c["y"] - y)
        if d <= c["radius_um"] * 1e-6 + ON_CELL_SLACK and (cell is None or d < cell[1]):
            cell = (n, d)
    if cell is not None:
        out["cell"] = cell[0]
    if square["label"] in BAD_SQUARES:
        truth = {**out, "truth": "bad", "why": f"the square is {square['label']}"}
    elif cell is None:
        truth = {**out, "truth": "bad", "why": "no cell there"}
    else:
        c = key["cells"][cell[0]]
        if not c["on_film"]:
            truth = {**out, "truth": "bad", "why": "the cell is mostly off the film"}
        elif not c["clear"]:
            truth = {**out, "truth": "bad", "why": "the cell is too near a bar"}
        else:
            truth = {**out, "truth": "good", "why": "a cell clear of the bars"}
    if not square.get("certain", True):
        truth["truth"], truth["was"] = "unsure", truth["truth"]
    return truth


def _positions(answer: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """The saved items' stage positions by name, from the run's record: the
    pose at the SEM orientation (no tilt) the key is in, the milling pose
    first."""
    record = answer.get("record") or {}
    found = {}
    for item in record.get("items") or []:
        poses = item.get("poses") or {}
        names = ["MILLING"] + sorted(set(poses) - {"MILLING"})
        for pose in (poses.get(name) for name in names):
            if pose and "x" in pose and "y" in pose and abs(pose.get("t") or 0) < FLAT:
                found[str(item.get("name"))] = pose
                break
    return found


def score_live_site_selection(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    sites = answer.get("sites")
    if not isinstance(sites, list) or not sites:
        raise InvalidAnswer("no sites list")
    most = int(case.scoring.get("max_sites", 8))
    positions = _positions(answer)
    per_site, good_cells, unsaved = [], set(), 0
    right = called = costly = 0
    for site in sites[:most]:
        if not isinstance(site, dict):
            raise InvalidAnswer(f"a site that is not an object: {site!r}")
        name = str(site.get("item") or "")
        verdict = str(site.get("verdict") or "").strip().lower()
        if verdict not in VERDICTS:
            raise InvalidAnswer(f"{name}: verdict {verdict!r} is not good or bad")
        at = positions.get(name)
        if at is None:
            unsaved += 1
            per_site.append(
                {
                    "item": name,
                    "verdict": verdict,
                    "truth": "not saved at the SEM orientation",
                }
            )
            continue
        truth = site_truth(key, float(at["x"]), float(at["y"]))
        entry = {"item": name, "verdict": verdict, **truth}
        if truth["truth"] == "good":
            good_cells.add(truth.get("cell"))
        if truth["truth"] in VERDICTS:
            called += 1
            right += verdict == truth["truth"]
            if verdict == "good" and truth["truth"] == "bad":
                costly += 1
                entry["costly"] = True
        per_site.append(entry)
    counted = len(sites[:most])
    fraction = len(good_cells) / counted if counted else 0.0
    wanted = float(case.scoring.get("min_good_fraction", 0.5))
    accuracy = right / called if called else 0.0
    wanted_accuracy = float(case.scoring.get("min_verdict_accuracy", 0.75))
    details = {
        "sites": counted,
        "good_sites": len(good_cells),
        "good_fraction": round(fraction, 3),
        "min_good_fraction": wanted,
        "verdict_accuracy": round(accuracy, 3),
        "min_verdict_accuracy": wanted_accuracy,
        "verdicts_scored": called,
        "not_saved": unsaved,
        "costly": costly,
        "per_site": per_site,
    }
    hit = (
        good_cells and fraction >= wanted and accuracy >= wanted_accuracy and not costly
    )
    return ("hit" if hit else "miss"), details


SQUARE_COLOURS = {
    "good": KEY,
    "cell_near_bar": "#e6a700",
    "no_cells": "#999999",
    "damaged": "#d62728",
    "thick": "#1f77b4",
}


def draw_live_site_selection(case: Case, runs: List[Run]):
    """The keyed squares and cells in stage coordinates (mm), and each
    repeat's sites with their verdicts."""
    from matplotlib.patches import Circle, Rectangle

    key = case.key
    fig, ax = _figure(12, 8)
    frame = key["frame"]
    side = (frame["pitch"] - frame["bar_width"]) * 1e3
    for square in key.get("squares") or []:
        colour = SQUARE_COLOURS.get(square["label"], "#999999")
        ax.add_patch(
            Rectangle(
                (square["x"] * 1e3 - side / 2, square["y"] * 1e3 - side / 2),
                side,
                side,
                fill=False,
                color=colour,
                lw=1.0 if square.get("certain", True) else 0.5,
                ls="-" if square.get("certain", True) else ":",
            )
        )
    labels = {tuple(q["index"]): q["label"] for q in key.get("squares") or []}
    for cell in key.get("cells") or []:
        good = (
            cell["on_film"]
            and cell["clear"]
            and labels.get(tuple(cell["square"])) not in BAD_SQUARES
        )
        ax.add_patch(
            Circle(
                (cell["x"] * 1e3, cell["y"] * 1e3),
                cell["radius_um"] * 1e-3,
                color=KEY if good else "#bbbbbb",
                alpha=0.5,
            )
        )
    for n, (_, answer) in enumerate(runs):
        positions = _positions(answer or {})
        for site in (answer or {}).get("sites") or []:
            at = positions.get(str(site.get("item")))
            if at is None:
                continue
            marker = "o" if str(site.get("verdict")) == "good" else "x"
            ax.plot(
                at["x"] * 1e3,
                at["y"] * 1e3,
                marker,
                color=_colour(n),
                ms=8,
                mew=2,
                mfc="none",
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("stage x (mm)")
    ax.set_ylabel("stage y (mm)")
    _legend(ax, runs, "good site (key); o said good, x said bad")
    from matplotlib.lines import Line2D

    squares = [
        Line2D([], [], color=colour, marker="s", ls="", mfc="none", label=label)
        for label, colour in SQUARE_COLOURS.items()
    ]
    fig.legend(handles=squares, loc="lower center", ncol=len(squares), fontsize=8)
    return fig


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    problems = []
    if not isinstance(key.get("frame"), dict):
        problems.append("key has no frame")
    if not key.get("squares"):
        problems.append("key has no squares")
    return problems


def summary(d: Dict[str, Any]) -> str:
    if "sites" not in d:
        return ""
    text = (
        f"{d['good_sites']} of {d['sites']} sites good, verdicts "
        f"{d['verdict_accuracy']:.0%} right"
    )
    if d.get("costly"):
        text += f", {d['costly']} bad called good"
    return text


def key_text(key: Dict[str, Any]) -> str:
    good = sum(1 for c in key.get("cells") or [] if c["on_film"] and c["clear"])
    return f"{len(key.get('squares') or [])} squares, {good} good cells"


TASK = EvalTask(
    name="live_site_selection",
    title="Find, visit and judge sites for lamellae on the simulator",
    skill="screen-grids",
    answer_shape={
        "sites": [{"item": "Site-01", "verdict": "good|bad", "reason": "..."}]
    },
    answer_notes=(
        "Each site is an item you saved in the experiment (item_add), by its "
        "name; it is scored where it was saved."
    ),
    score=score_live_site_selection,
    check_key=check_key,
    summary=summary,
    draw=draw_live_site_selection,
    key_text=key_text,
    sims=(Sim("live-site-selection", "fibsem.ai.evals.live.sim_sites"),),
)
