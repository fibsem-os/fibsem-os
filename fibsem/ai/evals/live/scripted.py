"""A scripted agent for live cases: fixed steps and simple image rules, no
model, so a live case runs end to end without spending anything. It is a
baseline, not a good agent: its score is what a naive one gets, and a run of
it checks the case, the runner and the scoring.

Run as the runner runs fibsem-agent, with the same flags (those it has no
use for are ignored)::

    python -m fibsem.ai.evals.live.scripted --files <work> \
        [--task live_site_selection] [--session-id .. --goal .. ..]

It finds its run's App Server by ``FIBSEM_DISCOVERY_FILE`` and its task by
``FIBSEM_EVAL_TASK`` (or ``--task``), and writes its answer to
``<work>/answer.json``.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.evals.live.client import LiveClient, LiveError
from fibsem.ai.evals.live.runs import TASK_ENV
from fibsem.server import discovery

log = logging.getLogger(__name__)

NAME = "fibsem-scripted-eval/1"
ANSWER = "answer.json"

# --- site selection -----------------------------------------------------------

MOST_SITES = 8
SITE_HFW = 80e-6  # m, the image at each site
CLEARANCE_UM = 4.0  # film all round a cell, past its edge
CELL_AREA_UM2 = (40.0, 1200.0)  # a bright blob this big is a cell's core
BODY = 2.0  # a cell's body, from its bright core's radius
TORN_UM2 = 150.0  # this much black near a cell is torn film, not specks


def _levels(image: np.ndarray) -> Tuple[float, float]:
    """The film's and the bars' grey levels: the two most common greys of
    a smoothed image (film darker than the bars)."""
    hist, edges = np.histogram(image, bins=64, range=(0, 255))
    centres = (edges[:-1] + edges[1:]) / 2
    first = int(np.argmax(hist))
    masked = hist.copy()
    masked[max(0, first - 6) : first + 7] = 0
    second = int(np.argmax(masked))
    a, b = sorted((float(centres[first]), float(centres[second])))
    return a, b


def find_cells(
    image: np.ndarray, pixel_size: float, step: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Bright blobs the size of a cell, each with the share of the ring
    around it that is film: 1 is a cell clear of every bar. *step* is how
    much brighter a bar is than the film, for an image that may show no bar
    (a site's): the film is then the image's median grey."""
    from scipy import ndimage as ndi

    smooth = ndi.gaussian_filter(image.astype(np.float32), 1.0e-6 / pixel_size)
    if step is None:
        film, bar = _levels(smooth)
    else:
        film = float(np.median(smooth))
        bar = film + step
    cells_mask = smooth > bar + 0.5 * (bar - film)
    film_mask = smooth < (film + bar) / 2
    torn = smooth < film - 0.6 * (bar - film)
    labels, n = ndi.label(cells_mask)
    px_um2 = (pixel_size * 1e6) ** 2
    found = []
    for index, box in enumerate(ndi.find_objects(labels), start=1):
        blob = labels[box] == index
        area = float(blob.sum()) * px_um2
        if not CELL_AREA_UM2[0] <= area <= CELL_AREA_UM2[1]:
            continue
        ys, xs = np.nonzero(blob)
        cy, cx = float(ys.mean() + box[0].start), float(xs.mean() + box[1].start)
        # the bright core is about half the body: the body's edge fades
        radius = BODY * np.sqrt(area / np.pi) / (pixel_size * 1e6)
        reach = radius + CLEARANCE_UM * 1e-6 / pixel_size
        y0, y1 = int(max(0, cy - reach)), int(min(image.shape[0], cy + reach + 1))
        x0, x1 = int(max(0, cx - reach)), int(min(image.shape[1], cx + reach + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(yy - cy, xx - cx)
        ring = (d > radius) & (d <= reach)
        if not ring.any():
            continue
        film_share = float(film_mask[y0:y1, x0:x1][ring].mean())
        wide = 3 * reach
        ty0, ty1 = int(max(0, cy - wide)), int(min(image.shape[0], cy + wide + 1))
        tx0, tx1 = int(max(0, cx - wide)), int(min(image.shape[1], cx + wide + 1))
        torn_near = bool(torn[ty0:ty1, tx0:tx1].sum() * px_um2 > TORN_UM2)
        found.append(
            {
                "x_px": round(cx, 1),
                "y_px": round(cy, 1),
                "area_um2": round(area, 1),
                "film_share": round(film_share, 3),
                "torn_near": torn_near,
            }
        )
    return found


def assess_site(image: np.ndarray, pixel_size: float, step: float) -> Tuple[str, str]:
    """good or bad for a lamella from the site's image, and why: a cell at
    the centre with film all round it and no tear in view. *step* is the
    bars' grey over the film's, from the overview."""
    cells = find_cells(image, pixel_size, step)
    h, w = image.shape
    centre = [
        c
        for c in cells
        if np.hypot(c["x_px"] - w / 2, c["y_px"] - h / 2) * pixel_size < 6e-6
    ]
    if not centre:
        return "bad", "no cell at the centre of the site's image"
    cell = max(centre, key=lambda c: c["area_um2"])
    if cell["torn_near"]:
        return "bad", "torn film next to the cell"
    if cell["film_share"] < 0.9:
        return "bad", (
            f"the cell is near a grid bar: {cell['film_share']:.0%} of the "
            "ring round it is film"
        )
    return "good", (
        f"a {cell['area_um2']:.0f} um2 cell on intact film, clear of the bars"
    )


def _image_path(answer: Any) -> str:
    """The saved file of an image command's answer."""
    image = (answer or {}).get("image", answer) or {}
    return image.get("path") or (image.get("summary") or {}).get("path")


def _read(path: str) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    return image if image.ndim == 2 else image[..., 0]


def site_selection(client: LiveClient, work: Path) -> Dict[str, Any]:
    # the overview at the experiment's top: a path is inside the experiment
    overview = client.call(
        "beam_acquire_overview", settings={}, name="overview", path="."
    )
    summary = overview["image"]["summary"]
    path, pixel_size = summary["path"], float(summary["pixel_size"])
    image = _read(path)
    film, bar = _levels(image)
    cells = [
        c
        for c in find_cells(image, pixel_size)
        if c["film_share"] >= 0.9 and not c["torn_near"]
    ]
    cells.sort(key=lambda c: (-c["film_share"], -c["area_um2"]))
    sites = []
    for cell in cells:
        if len(sites) == MOST_SITES:
            break
        at = client.call(
            "experiment_get_image_stage_position",
            path=Path(path).name,
            x=cell["x_px"],
            y=cell["y_px"],
        )["stage_position"]
        try:
            client.call(
                "stage_move_absolute",
                x=at["x"],
                y=at["y"],
                z=at["z"],
                r=at["r"],
                t=at["t"],
            )
        except LiveError as error:
            # out of the stage's reach: the next cell
            log.info("Skipping a cell the stage can't reach: %s", error)
            continue
        name = f"Site-{len(sites) + 1:02d}"
        client.call("item_add", name=name, stage_position=at)
        shot = client.call(
            "beam_acquire_image",
            beam_type="ELECTRON",
            image_settings={"hfw": SITE_HFW, "resolution": [1536, 1024]},
            path="sites",
            filename=name,
        )
        shot_path = _image_path(shot)
        shot_image = _read(shot_path)
        verdict, reason = assess_site(
            shot_image, SITE_HFW / shot_image.shape[1], bar - film
        )
        sites.append({"item": name, "verdict": verdict, "reason": reason})
    return {"sites": sites}


AGENTS: Dict[str, Callable[[LiveClient, Path], Dict[str, Any]]] = {
    "live_site_selection": site_selection,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task", default=os.environ.get(TASK_ENV), choices=sorted(AGENTS)
    )
    parser.add_argument("--files", "--code", dest="work", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO)
    found_at = Path(os.environ[discovery.DISCOVERY_FILE_ENV])
    with LiveClient.as_agent(found_at, NAME) as client:
        answer = AGENTS[args.task](client, args.work)
    (args.work / ANSWER).write_text(json.dumps(answer, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
