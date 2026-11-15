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
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.evals.live.client import LiveClient, LiveError
from fibsem.ai.evals.live.runs import CASE_ENV, TASK_ENV
from fibsem.server import discovery

log = logging.getLogger(__name__)

NAME = "fibsem-scripted-eval/1"
ANSWER = "answer.json"

# --- site selection -----------------------------------------------------------

MOST_SITES = 8
SITE_HFW = 80e-6  # m, the image at each site
FOCUS_HFW = 80e-6  # m, the autofocus's field at each site
# scored by the image's variance: on a noisy SEM image tenengrad (the
# default) scores lowest at focus
FOCUS_METHOD = "variance"
COINCIDENCE_TRIES = 3
FIB_TILE_HFW = 300e-6  # m, a FIB overview's tiles at the SEM orientation
RECENTRE_M = 20e-6  # a cell this near a site's centre is the one meant
SEM_LOOK_HFW = 400e-6  # m, the SEM look that sets its greys
# the share of the ring round a cell that must be film for it to be clear:
# the FIB's film and bars are nearer in grey, so its ring reads less clean
CLEAR = {"ELECTRON": 0.9, "ION": 0.8}
COINCIDENT_M = 0.3e-6  # a check this near the centre is coincident
CLEARANCE_UM = 4.0  # film all round a cell, past its edge
CELL_AREA_UM2 = (40.0, 1200.0)  # a bright blob this big is a cell's core
BODY = 2.0  # a cell's body, from its bright core's radius
TORN_UM2 = 150.0  # this much black near a cell is torn film, not specks


def _levels(image: np.ndarray) -> Tuple[float, float]:
    """The film's and the bars' grey levels: the two most common greys of
    a smoothed image (film darker than the bars)."""
    # torn film is black: not a level of its own
    hist, edges = np.histogram(image[image > 40], bins=64, range=(0, 255))
    centres = (edges[:-1] + edges[1:]) / 2
    first = int(np.argmax(hist))
    masked = hist.copy()
    masked[max(0, first - 6) : first + 7] = 0
    second = int(np.argmax(masked))
    a, b = sorted((float(centres[first]), float(centres[second])))
    return a, b


def _film(smooth: np.ndarray, step: float) -> float:
    """The film's grey in an image that may or may not show a bar: the
    darker of its two levels when they are a bar's step apart, else its one
    most common grey (a site's image can be mostly bar, so not the median)."""
    hist, edges = np.histogram(smooth[smooth > 40], bins=64, range=(0, 255))
    centres = (edges[:-1] + edges[1:]) / 2
    film, bar = _levels(smooth)
    if 0.6 * step <= bar - film <= 1.5 * step:
        return film
    return float(centres[int(np.argmax(hist))])


def find_cells(
    image: np.ndarray, pixel_size: float, step: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Bright blobs the size of a cell, each with the share of the ring
    around it that is film: 1 is a cell clear of every bar. *step* is how
    much brighter a bar is than the film, for an image that may show no bar
    (a site's): the film is then found from *step*."""
    from scipy import ndimage as ndi

    smooth = ndi.gaussian_filter(image.astype(np.float32), 1.0e-6 / pixel_size)
    if step is None:
        film, bar = _levels(smooth)
    else:
        film = _film(smooth, step)
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


def _sim_run() -> Dict[str, Any]:
    """The case's ``sim_run`` block (never its key), when the runner names
    the case."""
    import yaml

    folder = os.environ.get(CASE_ENV)
    if not folder:
        return {}
    data = yaml.safe_load((Path(folder) / "case.yaml").read_text()) or {}
    return data.get("sim_run") or {}


def _image_path(answer: Any) -> str:
    """The saved file of an image command's answer."""
    image = (answer or {}).get("image", answer) or {}
    return image.get("path") or (image.get("summary") or {}).get("path")


def _read(path: str) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    return image if image.ndim == 2 else image[..., 0]


def make_coincident(client: LiveClient, tries: int = COINCIDENCE_TRIES) -> float:
    """Bring the FIB onto the point at the SEM's centre: check, then move
    the stage along the vertical by what the check measured, until it is
    within COINCIDENT_M. The last offset measured (m)."""
    dy = float("inf")
    for _ in range(tries):
        check = client.call("alignment_check_coincidence", hfw=SITE_HFW)
        found = (check.get("summary") or {}).get("metadata") or check
        dy = float(found["dy"])
        log.info(
            "coincidence: dy %.2f um, dz %.2f um, score %.3f, agrees %s",
            dy * 1e6,
            float(found.get("dz", 0)) * 1e6,
            float(found.get("score", 0)),
            found.get("agrees"),
        )
        if abs(dy) <= COINCIDENT_M:
            break
        client.call("stage_vertical_move", dy=dy, beam_type="ION")
    return dy


def _stretch(client: LiveClient, name: str, shape) -> float:
    """How much further the stage moves per pixel down *name* than across
    it: 1 face on, more for a view that foreshortens y (the FIB's)."""
    h, w = shape
    at = [
        client.call("experiment_get_image_stage_position", path=name, x=x, y=y)[
            "stage_position"
        ]
        for x, y in ((0, 0), (w - 1, 0), (0, h - 1))
    ]
    across = math.hypot(at[1]["x"] - at[0]["x"], at[1]["y"] - at[0]["y"]) / (w - 1)
    down = math.hypot(at[2]["x"] - at[0]["x"], at[2]["y"] - at[0]["y"]) / (h - 1)
    return down / across


def _centre_on_cell(client: LiveClient, name: str, step: float) -> None:
    """Look with the SEM and move onto the cell nearest the centre, if one
    is within reach of it; the stage stays put otherwise."""
    look = client.call(
        "beam_acquire_image",
        beam_type="ELECTRON",
        image_settings={"hfw": SITE_HFW, "resolution": [1536, 1024]},
        path="sites",
        filename=f"{name}-look",
    )
    path = _image_path(look)
    image = _read(path)
    pixel_size = SITE_HFW / image.shape[1]
    h, w = image.shape
    near = [
        (np.hypot(c["x_px"] - w / 2, c["y_px"] - h / 2) * pixel_size, c)
        for c in find_cells(image, pixel_size, step)
    ]
    near = [(d, c) for d, c in near if d < RECENTRE_M]
    if not near:
        return
    off, cell = min(near, key=lambda pair: pair[0])
    if off < 1e-6:
        return
    at = client.call(
        "experiment_get_image_stage_position",
        path=f"sites/{Path(path).name}",
        x=cell["x_px"],
        y=cell["y_px"],
    )["stage_position"]
    z = client.call("stage_get_position")["value"]["z"]
    client.call("stage_move_absolute", x=at["x"], y=at["y"], z=z, r=at["r"], t=at["t"])
    log.info("%s: centred on the cell, %.1f um over", name, off * 1e6)


def site_selection(client: LiveClient, work: Path) -> Dict[str, Any]:
    beam = _sim_run().get("overview_beam", "ELECTRON")
    settings: Dict[str, Any] = {}
    if beam == "ION":
        # the FIB foreshortens y: smaller tiles, so the rows stay in reach
        settings = {"image_settings": {"beam_type": "ION", "hfw": FIB_TILE_HFW}}
    # the overview at the experiment's top: a path is inside the experiment
    overview = client.call(
        "beam_acquire_overview", settings=settings, name="overview", path="."
    )
    summary = overview["image"]["summary"]
    path, pixel_size = summary["path"], float(summary["pixel_size"])
    image = _read(path)
    # face on: the image stretched so a pixel is as tall as it is wide
    stretch = _stretch(client, Path(path).name, image.shape)
    if abs(stretch - 1) > 0.01:
        from scipy import ndimage as ndi

        image = ndi.zoom(image, (stretch, 1), order=1)
    if beam == "ION":
        # the site images are the SEM's: its greys from a wide SEM look
        wide = client.call(
            "beam_acquire_image",
            beam_type="ELECTRON",
            image_settings={"hfw": SEM_LOOK_HFW, "resolution": [1536, 1024]},
            path="sites",
            filename="grid",
        )
        film, bar = _levels(_read(_image_path(wide)))
    else:
        film, bar = _levels(image)
    clear = CLEAR[beam]
    cells = [
        c
        for c in find_cells(image, pixel_size)
        if c["film_share"] >= clear and not c["torn_near"]
    ]
    for c in cells:
        c["y_px"] = round(c["y_px"] / stretch, 1)
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
            # z as it is: the last site's coincident height holds on a flat grid
            z = client.call("stage_get_position")["value"]["z"]
            client.call(
                "stage_move_absolute", x=at["x"], y=at["y"], z=z, r=at["r"], t=at["t"]
            )
        except LiveError as error:
            # out of the stage's reach: the next cell
            log.info("Skipping a cell the stage can't reach: %s", error)
            continue
        name = f"Site-{len(sites) + 1:02d}"
        # where the overview put the cell is rough (the FIB sees a cell's top
        # displaced by its height): centre on it with the SEM first
        _centre_on_cell(client, name, bar - film)
        # ready to mill: in focus, and the FIB looking where the SEM does
        focus = client.call(
            "beam_auto_focus",
            beam_type="ELECTRON",
            hfw=FOCUS_HFW,
            settings={"method": FOCUS_METHOD},
        )
        log.info(
            "%s focus: %.1f um from %.1f um",
            name,
            focus["working_distance"] * 1e6,
            focus["initial_working_distance"] * 1e6,
        )
        make_coincident(client)
        # saved where the stage is now, coincident
        client.call("item_add", name=name)
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
