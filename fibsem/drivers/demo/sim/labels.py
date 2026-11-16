"""Labelled images from the simulator's 3D sample.

The simulator knows what every pixel of an SEM or FIB image shows, because it
drew the image from objects: the same trace that shades a view says which
object each pixel's beam meets. `view` turns that into a class for every pixel
and a number for every cell; `acquire` takes an image and its labels together;
and run as a program, this writes a small dataset::

    python -m fibsem.drivers.demo.sim.labels --out labels --seeds 1 2 3

which leaves `images/`, `labels/` (class ids), `instances/` (cell numbers),
`overlays/` and a COCO `annotations.json` in the folder.

Labels come from the objects, not from the image: something too faint to make
out (a crack under thick ice) is labelled all the same.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo import sim_solid, sim_view
from fibsem.geometry.projection import BeamStageProjection
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    ImageSettings,
)

UM = 1e6
# every class: its id (the value in a label image), name and overlay colour
CLASSES: Tuple[Tuple[int, str, Tuple[int, int, int]], ...] = (
    (0, "background", (0, 0, 0)),  # nothing there, or past the grid
    (1, "ice", (70, 90, 120)),  # the sheet of ice over film
    (2, "grid bar", (230, 160, 40)),
    (3, "cell", (60, 200, 90)),  # from above, under its ice
    (4, "film hole", (120, 160, 220)),  # ice across a hole in the film
    (5, "broken hole or rip", (200, 40, 40)),
    (6, "crack", (255, 90, 200)),
    (7, "contamination", (250, 240, 80)),
    (8, "ice plate", (150, 240, 250)),
    (9, "fiducial", (255, 255, 255)),
    (10, "milled trench", (110, 30, 140)),  # the way through a pattern cut
    (11, "cut face", (190, 110, 240)),  # ice, film or bar a pattern exposed
    (12, "cell cut face", (20, 140, 60)),  # a cell a pattern cut through
    (13, "coat", (255, 130, 50)),  # the GIS or sputter coat, seen edge-on
)
ID = {name: number for number, name, _ in CLASSES}
PALETTE = np.zeros((256, 3), dtype=np.uint8)
for _number, _name, _colour in CLASSES:
    PALETTE[_number] = _colour
# what gets a box of its own in the annotations, beside each cell
BOXED = ("contamination", "ice plate", "broken hole or rip", "fiducial")
MIN_AREA = 60  # px; a smaller patch gets no box
IN_CELL = (sim_solid.CELL, sim_solid.NUCLEUS, sim_solid.DROPLET, sim_solid.VESICLE)


@dataclass
class Labels:
    """What an image shows: `classes` is the class id of each pixel (uint8,
    see CLASSES) and `cells` the number of the cell at each pixel (uint16,
    0 where there is none). Both are the image's shape."""

    classes: np.ndarray
    cells: np.ndarray

    def boxes(self) -> List[dict]:
        """A box for each cell and each patch of a BOXED class:
        {"class", "box": [x0, y0, x1, y1], "area", and a cell's "cell"}."""
        out = []
        for number, where in enumerate(ndi.find_objects(self.cells), start=1):
            if where is None:
                continue
            area = int((self.cells[where] == number).sum())
            if area >= MIN_AREA:
                out.append(
                    {"class": "cell", "cell": number, "box": _box(where), "area": area}
                )
        for name in BOXED:
            patches, _ = ndi.label(
                ndi.binary_closing(self.classes == ID[name], iterations=2)
            )
            for number, where in enumerate(ndi.find_objects(patches), start=1):
                area = int((patches[where] == number).sum())
                if area >= MIN_AREA:
                    out.append({"class": name, "box": _box(where), "area": area})
        return out

    def overlay(self, image: np.ndarray) -> np.ndarray:
        """The image with its classes tinted over it and each cell outlined
        (uint8, height x width x 3)."""
        grey = np.stack([image] * 3, axis=-1).astype(np.float32)
        out = (0.55 * grey + 0.45 * PALETTE[self.classes]).astype(np.uint8)
        cells = self.cells.astype(np.int32)
        out[ndi.morphological_gradient(cells, size=3) > 0] = (255, 255, 255)
        return out


def _box(where) -> List[int]:
    return [where[1].start, where[0].start, where[1].stop - 1, where[0].stop - 1]


def classify(scene, hit: dict) -> Labels:
    """The labels of a view from its trace (`sim_solid.trace`), over the
    trace's own columns and rows."""
    material, kind = hit["material"], hit["kind"]
    x = np.broadcast_to(hit["x"], material.shape) / UM
    y = np.asarray(hit["y"]) / UM
    bars, _, rips, rim, beyond = scene.film_masks(x, y)
    broken = scene.broken_hole_mask(x, y)
    crack = scene.crack_depth(x, y)
    solid = material != sim_solid.VOID
    top, cut = kind == 1, kind >= 2
    metal = bars | rim
    cell_id = hit["cell_id"]
    over_cell = (cell_id >= 0) & (hit["cell_top"] > sim_solid.FILM_TOP + 0.3)

    classes = np.zeros(material.shape, dtype=np.uint8)
    classes[solid] = ID["ice"]
    classes[solid & top & (hit["hole"] > 0.3)] = ID["film hole"]
    classes[solid & top & (crack > 0.3) & ~metal] = ID["crack"]
    classes[solid & ((material == sim_solid.BAR) | (top & metal))] = ID["grid bar"]
    classes[solid & top & (hit["plate"] > 0.5)] = ID["ice plate"]
    seen_cell = solid & top & over_cell
    classes[seen_cell] = ID["cell"]
    particle = material == sim_solid.PARTICLE
    classes[particle] = ID["contamination"]
    classes[particle & hit["bead"]] = ID["fiducial"]
    classes[solid & cut] = ID["cut face"]
    inside = solid & cut & np.isin(material, IN_CELL)
    classes[inside] = ID["cell cut face"]
    classes[solid & cut & (material == sim_solid.COAT)] = ID["coat"]
    classes[solid & cut & particle] = ID["contamination"]
    void = ~solid & ~beyond
    classes[void & (rips | broken)] = ID["broken hole or rip"]
    if scene.milled:
        # a way through that is not a rip or a broken hole was milled
        classes[void & ~(rips | broken)] = ID["milled trench"]
    cells = np.where((seen_cell | inside) & (cell_id >= 0), cell_id + 1, 0)
    cells = cells.astype(np.uint16)
    if hit["flip"]:
        classes, cells = classes[::-1], cells[::-1]
    return Labels(classes, cells)


def view(
    scene,
    beam_type: BeamType,
    stage_position,
    hfw: float,
    resolution: Tuple[int, int],
    projection: BeamStageProjection,
    beam_shift: Tuple[float, float] = (0.0, 0.0),
    beam_current: Optional[float] = None,
) -> Labels:
    """The labels of the image `scene.render` draws with the same arguments."""
    if scene.reference_position is None:
        scene.anchor(stage_position)
    scene.place_grids(projection)
    frame = scene.view_frame(
        beam_type, stage_position, projection, beam_shift, beam_current
    )
    pixel_size = hfw / int(resolution[0])
    u, w, to_view = sim_view.content(frame, pixel_size, resolution)
    found = classify(scene, sim_solid.trace(scene, frame, u, w, pixel_size * UM))
    return Labels(
        to_view(found.classes, 0, 0).astype(np.uint8),
        to_view(found.cells, 0, 0).astype(np.uint16),
    )


def of(microscope, beam_type: BeamType, hfw: float, resolution) -> Labels:
    """The labels of the image a Demo microscope would take now."""
    connection = microscope._connection
    scene = connection.sample_scene
    if scene is None:
        raise ValueError("labels need the sample scene: set sim.sample.enabled")
    shift = microscope.get_beam_shift(beam_type)
    return view(
        scene,
        beam_type,
        connection.scene_stage_position(),
        hfw,
        resolution,
        BeamStageProjection.from_microscope(microscope, beam_type=beam_type),
        beam_shift=(float(shift.x), float(shift.y)),
        beam_current=float(microscope.get_beam_current(beam_type)),
    )


def acquire(microscope, settings: ImageSettings) -> Tuple[FibsemImage, Labels]:
    """An image from a Demo microscope and its labels. The labels are read
    first: a FIB frame wears the sample a little, and they are of the sample
    the frame saw."""
    found = of(microscope, settings.beam_type, settings.hfw, settings.resolution)
    return microscope.acquire_image(image_settings=settings), found


# ----------------------------------------------------------------- a dataset
VIEWS = ("overview", "tilted", "lamella")
MILLING_ANGLE = 15.0  # deg


def _session(seed: int, config_path: Optional[str], sample: Optional[dict]):
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=config_path)
    configured = (microscope.system.sim or {}).get("sample") or {}
    microscope.system.sim["sample"] = {
        **configured,
        "enabled": True,
        "seed": int(seed),
        # both beams on the same point, so a lamella is in each one's view
        "coincidence_offset": 0.0,
        **(sample or {}),
    }
    microscope._connection.setup_sample_scene()
    return microscope


def _shots(kind: str, microscope) -> List[Tuple[BeamType, float]]:
    """Set the microscope up for a kind of view; the (beam, hfw) to take."""
    if kind == "overview":
        return [(BeamType.ELECTRON, 300e-6), (BeamType.ELECTRON, 120e-6)]
    microscope.move_to_milling_angle(MILLING_ANGLE)
    if kind == "tilted":
        return [(BeamType.ION, 150e-6), (BeamType.ELECTRON, 150e-6)]
    # a lamella: a trench either side of the centre
    for y in (4.5e-6, -4.5e-6):
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=12e-6, height=6e-6, depth=10e-6, centre_x=0.0, centre_y=y
            )
        )
    microscope.run_milling()
    return [(BeamType.ION, 50e-6), (BeamType.ELECTRON, 50e-6)]


def _rle(mask: np.ndarray) -> dict:
    """A mask as COCO's uncompressed run-length encoding."""
    flat = mask.ravel(order="F").astype(np.int8)
    edges = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], edges, [flat.size]])).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": runs}


def _annotations(found: Labels, image_id: int, first_id: int) -> List[dict]:
    out = []
    for item in found.boxes():
        x0, y0, x1, y1 = item["box"]
        where = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        mask = np.zeros(found.classes.shape, dtype=bool)
        if item["class"] == "cell":
            mask[where] = found.cells[where] == item["cell"]
        else:
            mask[where] = found.classes[where] == ID[item["class"]]
        out.append(
            {
                "id": first_id + len(out),
                "image_id": image_id,
                "category_id": ID[item["class"]],
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "area": int(mask.sum()),
                "iscrowd": 0,
                "segmentation": _rle(mask),
            }
        )
    return out


def generate(
    out: str,
    seeds: Sequence[int],
    views: Sequence[str] = VIEWS,
    resolution: Tuple[int, int] = (1536, 1024),
    config_path: Optional[str] = None,
    sample: Optional[dict] = None,
    overlays: bool = True,
) -> dict:
    """Write a labelled dataset to the folder `out`: for each seed and each
    kind of view, the images with their class and cell labels, and one COCO
    `annotations.json` over them all (returned). `sample` is extra
    `sim: sample:` keys for every scene."""
    unknown = set(views) - set(VIEWS)
    if unknown:
        raise ValueError(f"unknown views {sorted(unknown)}: choose from {VIEWS}")
    root = Path(out)
    folders = ["images", "labels", "instances"] + (["overlays"] if overlays else [])
    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)
    coco = {
        "info": {"description": "fibsemOS simulator labels"},
        "categories": [{"id": n, "name": name} for n, name, _ in CLASSES if n],
        "images": [],
        "annotations": [],
    }
    for seed in seeds:
        for kind in views:
            microscope = _session(seed, config_path, sample)
            try:
                for beam_type, hfw in _shots(kind, microscope):
                    settings = ImageSettings(
                        resolution=tuple(resolution),
                        hfw=hfw,
                        dwell_time=1e-6,
                        save=False,
                        beam_type=beam_type,
                    )
                    image, found = acquire(microscope, settings)
                    beam = "sem" if beam_type is BeamType.ELECTRON else "fib"
                    name = f"seed{int(seed):04d}-{kind}-{beam}-{round(hfw * UM)}um"
                    _save(root, name, image.data, found, overlays)
                    image_id = len(coco["images"]) + 1
                    coco["images"].append(
                        {
                            "id": image_id,
                            "file_name": f"images/{name}.png",
                            "labels": f"labels/{name}.png",
                            "instances": f"instances/{name}.png",
                            "width": int(image.data.shape[1]),
                            "height": int(image.data.shape[0]),
                            "seed": int(seed),
                            "view": kind,
                            "beam": beam,
                            "hfw": hfw,
                        }
                    )
                    coco["annotations"].extend(
                        _annotations(found, image_id, len(coco["annotations"]) + 1)
                    )
            finally:
                microscope.disconnect()
    with open(root / "annotations.json", "w") as f:
        json.dump(coco, f)
    return coco


def _save(root: Path, name: str, image: np.ndarray, found: Labels, overlays: bool):
    Image.fromarray(image).save(root / "images" / f"{name}.png")
    classes = Image.fromarray(found.classes, mode="P")
    classes.putpalette(PALETTE.ravel().tolist())
    classes.save(root / "labels" / f"{name}.png")
    Image.fromarray(found.cells).save(root / "instances" / f"{name}.png")
    if overlays:
        over = Image.fromarray(found.overlay(image))
        draw = ImageDraw.Draw(over)
        for item in found.boxes():
            colour = tuple(int(v) for v in PALETTE[ID[item["class"]]])
            draw.rectangle(item["box"], outline=colour, width=2)
        over.save(root / "overlays" / f"{name}.jpg", quality=88)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write labelled SEM and FIB images from the simulator's sample."
    )
    parser.add_argument("--out", required=True, help="the folder to write to")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1], help="one scene per seed"
    )
    parser.add_argument(
        "--views", nargs="+", default=list(VIEWS), choices=VIEWS, help="kinds of view"
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[1536, 1024], metavar=("W", "H")
    )
    parser.add_argument(
        "--config",
        default=os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"),
        help="the microscope configuration the scenes come from",
    )
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args(argv)
    coco = generate(
        args.out,
        args.seeds,
        args.views,
        tuple(args.resolution),
        args.config,
        overlays=not args.no_overlays,
    )
    print(
        f"{len(coco['images'])} images, {len(coco['annotations'])} annotations "
        f"in {args.out}"
    )


if __name__ == "__main__":
    main()
