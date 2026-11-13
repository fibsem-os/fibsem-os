"""Live site selection: on the simulator, take an overview of the loaded
grid, save the likely sites, go to each and say whether it is good for a
lamella. Scored where the sites really are, from the sample the simulator
drew (``fibsem.ai.evals.live.sim_sites``).

The sites' positions are the experiment's saved items, which the runner
puts in the answer as ``record`` after the run: what the agent saved, not
what it says it saved.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import KEY, _colour, _figure, _legend
from fibsem.geometry.saved_images import stage_projector
from fibsem.structures.settings import FibsemStagePosition

VERDICTS = ("good", "bad")
# How far a site may be from a cell body's edge and still be on it (m): a
# lamella is milled through the cell, and a site is saved at its centre.
ON_CELL_SLACK = 1e-6
BAD_SQUARES = ("damaged", "thick")
FLAT = math.radians(1.0)  # a pose this near no tilt is at the SEM orientation
# A site is ready to mill when the SEM's and FIB's view centres meet within
# this (m) at its saved pose, and its SEM image is sharp: the beam's spread
# from being out of focus under this many pixels
COINCIDENT = 1e-6
SHARP_PX = 2.0
NEAR = 5e-6  # m, an image this near a site's x, y is of it
SITE_HFW = 300e-6  # m, wider than this is not a look at one site


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


def prepared(
    key: Dict[str, Any], at: Dict[str, float], answer: Dict[str, Any]
) -> Dict[str, Any]:
    """How ready a site was left for milling, from the run's record: how far
    the beams' view centres are apart at its saved pose (um), and whether
    the run took an SEM image of it and how blurred the last one is (px).
    A key without the optics says nothing."""
    optics = key.get("optics")
    if not optics:
        return {}
    out: Dict[str, Any] = {}
    z = float(at.get("z") or 0.0)
    off = optics["coincidence_per_z"] * abs(z - optics["coincident_z"])
    out["coincidence_um"] = round(off * 1e6, 2)
    out["coincident"] = off <= COINCIDENT
    images = [
        image
        for image in (answer.get("record") or {}).get("images") or []
        if image.get("beam") == "ELECTRON"
        and (image.get("hfw") or 1.0) <= SITE_HFW
        and image.get("stage_position")
        and math.hypot(
            image["stage_position"]["x"] - at["x"],
            image["stage_position"]["y"] - at["y"],
        )
        <= NEAR
    ]
    out["imaged"] = bool(images)
    if images:
        last = max(images, key=lambda i: i.get("time") or 0)
        z_image = float(last["stage_position"].get("z") or 0.0)
        focus = optics["sem_focus_wd"] + optics["sem_focus_per_z"] * z_image
        spread = optics["sem_angle"] * abs(float(last["working_distance"]) - focus)
        out["blur_px"] = round(spread / float(last["pixel_size"]), 2)
        out["sharp"] = out["blur_px"] <= SHARP_PX
        out["image"] = last["path"]
    return out


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
        entry.update(prepared(key, at, answer))
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
    record = answer.get("record") or {}
    scored = [p for p in per_site if "coincident" in p]
    said_good = [p for p in scored if p["verdict"] == "good"]
    # what a site called good must be left with to count as ready to mill:
    # coincident, and (once the simulator's autofocus can find focus) sharp
    need_sharp = bool(case.scoring.get("require_sharp", False))
    ready = [
        p for p in said_good if p["coincident"] and (p.get("sharp") or not need_sharp)
    ]
    steps = {
        "overview": bool(record.get("overviews")),
        "sites_saved": len(positions),
        "sites_imaged": sum(1 for p in scored if p.get("imaged")),
        "verdicts": counted,
    }
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
        "steps": steps,
        "per_site": per_site,
    }
    if scored:
        details.update(
            coincident=sum(1 for p in scored if p["coincident"]),
            sharp=sum(1 for p in scored if p.get("sharp")),
            ready=len(ready),
            said_good=len(said_good),
        )
    # a site called good is one to mill: it must be left ready for it
    unready = len(said_good) - len(ready)
    if not case.scoring.get("require_ready", True):
        unready = 0
    hit = (
        good_cells
        and fraction >= wanted
        and accuracy >= wanted_accuracy
        and not costly
        and not unready
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
    """The keyed squares and cells over each repeat's own overview, with its
    sites and their verdicts; a map in stage coordinates (mm) where no
    overview is to hand."""
    shown = [
        (n, answer, path)
        for n, (_, answer) in enumerate(runs)
        if (path := _overview(answer)) is not None
    ]
    if shown:
        return _draw_overlays(case, shown)
    return _draw_map(case, runs)


def _overview(answer: Any) -> Any:
    """The overview a run saved, as a path, if it is still there."""
    record = (answer or {}).get("record") or {}
    experiment, names = record.get("experiment"), record.get("overviews") or []
    if not experiment or not names:
        return None
    path = Path(experiment) / names[0]
    return path if path.is_file() else None


def _draw_overlays(case: Case, shown: List[Tuple[int, Dict[str, Any], Path]]):
    """One panel per repeat: its overview, the key's squares (by label) and
    good cells projected onto it, and its sites (o said good, x said bad;
    cyan truly good, pink truly bad, yellow unsure)."""
    import tifffile
    from matplotlib.patches import Polygon

    key = case.key
    frame = key["frame"]
    half = (frame["pitch"] - frame["bar_width"]) / 2
    u, v = np.asarray(frame["u"], float), np.asarray(frame["v"], float)
    corners = [half * (a * u + b * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    labels = {tuple(q["index"]): q["label"] for q in key.get("squares") or []}
    truth_colours = {"good": "#00e5ff", "bad": "#ff2d55"}
    fig, axes = _figure(12, 8 * len(shown), nrows=len(shown))
    for ax, (n, answer, path) in zip(np.atleast_1d(axes), shown):
        image = tifffile.imread(str(path))
        image = image if image.ndim == 2 else image[..., 0]
        block, project = stage_projector(str(path))
        at = block["stage_position"]
        at = at if isinstance(at, dict) else at.to_dict()

        def px(xy, at=at, project=project):
            point = project(
                FibsemStagePosition(x=xy[0], y=xy[1], z=at["z"], r=at["r"], t=at["t"])
            )
            return (point.x, point.y)

        ax.imshow(image, cmap="gray")
        for square in key.get("squares") or []:
            centre = np.array([square["x"], square["y"]])
            certain = square.get("certain", True)
            ax.add_patch(
                Polygon(
                    [px(centre + c) for c in corners],
                    closed=True,
                    fill=False,
                    color=SQUARE_COLOURS.get(square["label"], "#999999"),
                    lw=1.4 if certain else 0.8,
                    ls="-" if certain else ":",
                )
            )
        for cell in key.get("cells") or []:
            if (
                cell["on_film"]
                and cell["clear"]
                and labels.get(tuple(cell["square"])) not in BAD_SQUARES
            ):
                ax.plot(*px((cell["x"], cell["y"])), "o", ms=3, color=KEY)
        positions = _positions(answer)
        for site in answer.get("sites") or []:
            name = str(site.get("item"))
            where = positions.get(name)
            if where is None:
                continue
            truth = site_truth(key, where["x"], where["y"])["truth"]
            colour = truth_colours.get(truth, "#ffd60a")
            x, y = px((where["x"], where["y"]))
            good = str(site.get("verdict")) == "good"
            ax.plot(
                x, y, "o" if good else "x", ms=12, mew=2.5, mfc="none", color=colour
            )
            ax.annotate(
                name,
                (x, y),
                xytext=(8, -8),
                textcoords="offset points",
                color=colour,
                fontsize=8,
                weight="bold",
            )
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis("off")
        ax.set_title(
            f"r{n + 1}: {path.name}. Sites: o said good, x said bad; "
            "cyan truly good, pink truly bad, yellow unsure",
            fontsize=9,
        )
    _square_legend(fig)
    return fig


def _square_legend(fig) -> None:
    from matplotlib.lines import Line2D

    squares = [
        Line2D([], [], color=colour, marker="s", ls="", mfc="none", label=label)
        for label, colour in SQUARE_COLOURS.items()
    ]
    squares.append(Line2D([], [], color=KEY, marker="o", ls="", label="good cell"))
    fig.legend(handles=squares, loc="lower center", ncol=len(squares), fontsize=8)


def _draw_map(case: Case, runs: List[Run]):
    """The keyed squares and cells in stage coordinates (mm), and each
    repeat's sites with their verdicts."""
    from matplotlib.patches import Circle, Polygon

    key = case.key
    fig, ax = _figure(12, 8)
    frame = key["frame"]
    half = (frame["pitch"] - frame["bar_width"]) * 1e3 / 2
    # the squares lie along the grid's own axes, not the stage's
    u, v = np.asarray(frame["u"], float), np.asarray(frame["v"], float)
    corners = [half * (a * u + b * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for square in key.get("squares") or []:
        colour = SQUARE_COLOURS.get(square["label"], "#999999")
        centre = np.array([square["x"], square["y"]]) * 1e3
        ax.add_patch(
            Polygon(
                [centre + c for c in corners],
                closed=True,
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
    _square_legend(fig)
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
    if "said_good" in d:
        text += f", {d['ready']} of {d['said_good']} called good left ready to mill"
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
    sims=(
        Sim("live-site-selection", "fibsem.ai.evals.live.sim_sites"),
        Sim("live-site-selection-fib", "fibsem.ai.evals.live.sim_sites_fib"),
    ),
)
