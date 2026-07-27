"""Export AutoLamella experiment data to the automated lamella targeting ML format.

The training signal in an AutoLamella experiment is the point of interest an operator
picked for each lamella, on the FIB image they picked it in. That FIB image is the
final reference image of the *Select Milling Position* task -- the same kind of image
the targeting ML pipeline runs on -- so the pair (image, point) is directly usable as
training data.

Where those images are missing, a later task's final reference set is used instead
(see :data:`SOURCE_TASK_TYPES`); every one of them images at the milling position, so
the point is equally valid. Which task each sample came from is recorded, because the
later ones show a progressively more milled sample.

Layout written per run, by default into the experiment's own directory so a copied
experiment carries its training data with it::

    <experiment>/targeting-export/
        images/0001.tif      FIB reference image (FibsemImage, metadata preserved)
        points/0001.json     SOURCE OF TRUTH -- exact coordinates + provenance
        labels/0001.tif      disc at the point, rasterised from points/
        manifest.json        index over the samples, plus experiment provenance

One sample is one lamella. ``points/`` is authoritative and ``labels/`` is derived
from it, so the raster should never be hand-edited -- to change the label geometry,
re-export at a different ``radius_m`` (or rasterise the sidecar yourself). The
``images/`` + ``labels/`` pairing is what
:func:`~fibsem.applications.autolamella.tools.upload_to_hf.create_dataset` consumes.

No stage reprojection is involved: the reference image is acquired at the milling
position, so the POI -- already a milling-coordinate offset in metres from the image
centre -- converts straight to pixels via the image's own pixel size.

Failed lamellae are exported and tagged rather than dropped: a target a human picked
that then failed is a useful hard negative, not data to discard.

See FIB-305.
"""
import argparse
import glob
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.conversions import microscope_image_to_image_coordinates
from fibsem.structures import FibsemImage

# Tasks whose final FIB reference image can serve as the training input, in order of
# preference. All of them image at the milling position, so the POI is valid in any of
# them; they are ordered earliest-first in the workflow, because each later task has
# modified the sample further from the unmilled state the targeting model predicts on.
# `MILL_ROUGH` in particular shows trenches already cut -- usable, but a different
# visual distribution, which is why the chosen task is recorded per sample.
#
# A task's `task_name` is user-configurable per protocol, so filenames are derived from
# each lamella's own config rather than hardcoded.
SOURCE_TASK_TYPES = (
    "SELECT_MILLING_POSITION",
    "SPOT_BURN_FIDUCIAL",
    "MILL_FIDUCIAL",
    "MILL_ROUGH",
)

# `_acquire_set_of_channels` writes one image per field of view, suffixed res_01,
# res_02, ... sorted largest FOV to smallest -- so the highest suffix is the most
# zoomed. That is the image the task itself treats as its result (it takes images[-1]
# to display), and the one the targeting pipeline is equivalent to.
FINAL_FIB_IMAGE_GLOB = "ref_{task_name}_final_res_*_ib.tif"
_RES_SUFFIX = re.compile(r"_res_(\d+)_ib\.tif$")

# radius of the disc stamped into labels/, in metres. Specified as a physical size
# (not pixels) so a label means the same thing across images acquired at different
# fields of view.
DEFAULT_DISC_RADIUS_M = 2.0e-6

IMAGES_DIR = "images"
LABELS_DIR = "labels"
POINTS_DIR = "points"
MANIFEST_NAME = "manifest.json"

# exports land beside the data they came from, so a copied experiment directory
# carries its training data with it.
DEFAULT_EXPORT_DIRNAME = "targeting-export"

LABEL_VALUE = 1
LABEL_DTYPE = np.uint8


@dataclass
class ExportedSample:
    """One lamella: its FIB reference image and the point selected in it."""

    stem: str
    source_image: str
    source_task: str  # which SOURCE_TASK_TYPES entry the image came from
    shape: Tuple[int, int]  # (height, width)
    pixelsize: float
    hfw: Optional[float]
    # the point, and where it came from
    petname: str
    lamella_id: str
    pixel_x: float  # sub-pixel, deliberately not rounded
    pixel_y: float
    poi: Dict[str, float]  # original milling coordinates, metres
    stage_position: Dict[str, Optional[float]]
    milling_angle: Optional[float]
    defect: str
    completed_tasks: List[str] = field(default_factory=list)


@dataclass
class ExportSummary:
    """What an export run produced."""

    output_path: str
    samples: List[ExportedSample] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # human-readable reasons

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def task_names_for_type(lamella: Lamella, task_type: str) -> List[str]:
    """Names of the lamella's task(s) of the given type.

    A workflow may run a task more than once under different names, so this returns
    every match. Both the config's own ``task_name`` and the dict key are considered,
    since the key is what the protocol is written under.
    """
    names: List[str] = []
    for key, config in lamella.task_config.items():
        if getattr(config, "task_type", None) != task_type:
            continue
        for candidate in (getattr(config, "task_name", "") or "", key):
            if candidate and candidate not in names:
                names.append(candidate)
    return names


def _most_zoomed(paths: Sequence[str]) -> str:
    """Of a final reference set, the highest res_NN -- the smallest field of view."""

    def resolution_index(path: str) -> int:
        match = _RES_SUFFIX.search(os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(paths, key=resolution_index)


def find_final_fib_image(lamella: Lamella) -> Optional[Tuple[str, str]]:
    """The best available final FIB reference image, as ``(path, task_type)``.

    Walks :data:`SOURCE_TASK_TYPES` in order and takes the first task that actually
    produced images, so a lamella whose Select Milling Position images are missing can
    still contribute via a later task. Returns None if none of them did.
    """
    for task_type in SOURCE_TASK_TYPES:
        candidates: List[str] = []
        for task_name in task_names_for_type(lamella, task_type):
            pattern = FINAL_FIB_IMAGE_GLOB.format(task_name=task_name)
            candidates.extend(glob.glob(os.path.join(str(lamella.path), pattern)))
        if candidates:
            return _most_zoomed(candidates), task_type

    return None


def poi_to_pixels(lamella: Lamella, image: FibsemImage):
    """Convert the lamella's POI to pixel coordinates in ``image``.

    ``Lamella.poi`` is a milling-pattern coordinate: metres from the image centre with
    the y-axis pointing *up*. The reference image is acquired at the milling position,
    so this is a direct conversion -- no stage reprojection -- and it is independent of
    the field of view the POI was originally picked at, because it is held in metres.
    """
    if image.metadata is None or image.metadata.pixel_size is None:
        raise ValueError("reference image has no pixel size metadata")
    return microscope_image_to_image_coordinates(
        lamella.poi, image.data.shape[:2], image.metadata.pixel_size.x
    )


def collect_sample(lamella: Lamella, stem: str) -> Optional[ExportedSample]:
    """Build the sample for one lamella, or None if it has no usable image.

    Raises ValueError if the image exists but lacks the metadata needed to place the
    point -- a real problem worth surfacing, not a silent skip.
    """
    found = find_final_fib_image(lamella)
    if found is None:
        return None
    filename, source_task = found

    image = FibsemImage.load(filename)
    point = poi_to_pixels(lamella, image)
    pose = lamella.milling_pose

    hfw = None
    if image.metadata.image_settings is not None:
        hfw = image.metadata.image_settings.hfw

    return ExportedSample(
        stem=stem,
        source_image=filename,
        source_task=source_task,
        shape=tuple(image.data.shape[:2]),
        pixelsize=image.metadata.pixel_size.x,
        hfw=hfw,
        petname=lamella.name,
        lamella_id=lamella._id,
        pixel_x=float(point.x),
        pixel_y=float(point.y),
        poi=lamella.poi.to_dict(),
        stage_position=(
            pose.stage_position.to_dict()
            if pose is not None and pose.stage_position is not None
            else {}
        ),
        milling_angle=lamella.milling_angle,
        defect=lamella.defect.state.name,
        completed_tasks=list(lamella.completed_tasks),
    )


def rasterise_point(
    pixel_x: float,
    pixel_y: float,
    shape: Tuple[int, int],
    pixelsize: float,
    radius_m: float = DEFAULT_DISC_RADIUS_M,
) -> np.ndarray:
    """A label image with a filled disc at the point."""
    label = np.zeros(shape, dtype=LABEL_DTYPE)
    radius_px = radius_m / pixelsize
    if radius_px < 0.5:
        logging.warning(
            f"disc radius {radius_m:.2e} m is under half a pixel at {pixelsize:.2e} "
            f"m/px; the label will be empty or single-pixel"
        )

    height, width = shape
    r = int(np.ceil(radius_px))
    x0, x1 = max(0, int(np.floor(pixel_x)) - r), min(width, int(np.ceil(pixel_x)) + r + 1)
    y0, y1 = max(0, int(np.floor(pixel_y)) - r), min(height, int(np.ceil(pixel_y)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return label

    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - pixel_x) ** 2 + (ys - pixel_y) ** 2 <= radius_px**2
    label[y0:y1, x0:x1][mask] = LABEL_VALUE
    return label


def _sample_to_dict(sample: ExportedSample, experiment: Experiment) -> dict:
    """The contents of ``points/<stem>.json``."""
    return {
        "image": f"{IMAGES_DIR}/{sample.stem}.tif",
        "experiment": {
            "name": experiment.name,
            "id": experiment._id,
            "path": str(experiment.path),
            "user": experiment.user,
            "project": experiment.project,
            "organisation": experiment.organisation,
        },
        **asdict(sample),
        # asdict() flattens the tuple to a bare list; name the axes instead, so the
        # sidecar is unambiguous about height-vs-width order.
        "shape": {"height": sample.shape[0], "width": sample.shape[1]},
    }


def default_output_path(experiment: Experiment) -> str:
    """Where an experiment's export goes unless told otherwise."""
    return os.path.join(str(experiment.path), DEFAULT_EXPORT_DIRNAME)


def export_experiment(
    experiment: Experiment,
    output_path: Optional[str] = None,
    radius_m: float = DEFAULT_DISC_RADIUS_M,
    start_index: int = 1,
    write_manifest_file: bool = True,
) -> ExportSummary:
    """Export one experiment -- one sample per lamella.

    ``output_path`` defaults to ``<experiment>/targeting-export``.

    ``start_index`` continues the ``NNNN`` numbering, so several experiments can be
    written into one output directory. Batch callers pass ``write_manifest_file=False``
    and write one combined manifest at the end.
    """
    if output_path is None:
        output_path = default_output_path(experiment)
    summary = ExportSummary(output_path=output_path)

    for directory in (IMAGES_DIR, LABELS_DIR, POINTS_DIR):
        os.makedirs(os.path.join(output_path, directory), exist_ok=True)

    index = start_index
    for lamella in experiment.positions:
        stem = f"{index:04d}"
        try:
            sample = collect_sample(lamella, stem)
        except Exception as e:
            reason = f"{lamella.name}: failed to build sample ({e}), skipped"
            logging.warning(reason)
            summary.skipped.append(reason)
            continue

        if sample is None:
            reason = (
                f"{lamella.name}: no final FIB reference image from any of "
                f"{', '.join(SOURCE_TASK_TYPES)}, skipped"
            )
            logging.warning(reason)
            summary.skipped.append(reason)
            continue

        image = FibsemImage.load(sample.source_image)
        image.save(os.path.join(output_path, IMAGES_DIR, stem))
        with open(os.path.join(output_path, POINTS_DIR, f"{stem}.json"), "w") as f:
            json.dump(_sample_to_dict(sample, experiment), f, indent=2)
        tifffile.imwrite(
            os.path.join(output_path, LABELS_DIR, f"{stem}.tif"),
            rasterise_point(
                sample.pixel_x, sample.pixel_y, sample.shape, sample.pixelsize, radius_m
            ),
        )

        summary.samples.append(sample)
        logging.info(
            f"exported {stem}: {lamella.name} from "
            f"{os.path.basename(sample.source_image)}"
        )
        index += 1

    if write_manifest_file:
        write_manifest(output_path, [summary], radius_m=radius_m)

    return summary


def write_manifest(
    output_path: str,
    summaries: Sequence[ExportSummary],
    radius_m: float = DEFAULT_DISC_RADIUS_M,
) -> str:
    """Write ``manifest.json`` indexing everything an export run produced."""
    samples = [s for summary in summaries for s in summary.samples]
    manifest = {
        "format": "autolamella-targeting-points-v1",
        "source_task_preference": list(SOURCE_TASK_TYPES),
        "disc_radius_m": radius_m,
        "n_samples": len(samples),
        # how many samples came from each task, so a consumer can see at a glance how
        # much of the set is post-milling rather than pristine
        "n_samples_by_source_task": {
            task_type: sum(1 for s in samples if s.source_task == task_type)
            for task_type in SOURCE_TASK_TYPES
            if any(s.source_task == task_type for s in samples)
        },
        "samples": [
            {
                "stem": s.stem,
                "image": f"{IMAGES_DIR}/{s.stem}.tif",
                "points": f"{POINTS_DIR}/{s.stem}.json",
                "label": f"{LABELS_DIR}/{s.stem}.tif",
                "source_image": s.source_image,
                "source_task": s.source_task,
                "petname": s.petname,
                "hfw": s.hfw,
                "defect": s.defect,
            }
            for s in samples
        ],
        "skipped": [reason for summary in summaries for reason in summary.skipped],
    }

    path = os.path.join(output_path, MANIFEST_NAME)
    os.makedirs(output_path, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def export_experiments(
    experiment_paths: Sequence[str],
    output_path: Optional[str] = None,
    radius_m: float = DEFAULT_DISC_RADIUS_M,
) -> ExportSummary:
    """Export several experiments.

    With ``output_path`` set, everything lands in that one directory, numbered
    continuously, under a single combined manifest. With it left as None, each
    experiment exports to its own ``<experiment>/targeting-export`` -- separately
    numbered, each with its own manifest.
    """
    combined = ExportSummary(output_path=output_path or "")

    index = 1
    for experiment_path in experiment_paths:
        try:
            experiment = Experiment.load(
                os.path.join(experiment_path, "experiment.yaml")
            )
        except Exception as e:
            reason = f"{experiment_path}: failed to load experiment ({e}), skipped"
            logging.warning(reason)
            combined.skipped.append(reason)
            continue

        summary = export_experiment(
            experiment,
            output_path,  # None -> this experiment's own targeting-export
            radius_m=radius_m,
            start_index=index if output_path else 1,
            # combined runs share one manifest, written below; per-experiment runs
            # each write their own.
            write_manifest_file=output_path is None,
        )
        combined.samples.extend(summary.samples)
        combined.skipped.extend(summary.skipped)
        if output_path:
            index += len(summary.samples)

    if output_path:
        write_manifest(output_path, [combined], radius_m=radius_m)
    return combined


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export AutoLamella experiments to the targeting ML format."
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="experiment directories (each containing experiment.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "output directory for a single combined export. Omit to write each "
            f"experiment to its own <experiment>/{DEFAULT_EXPORT_DIRNAME}"
        ),
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=DEFAULT_DISC_RADIUS_M,
        help=f"label disc radius in metres (default: {DEFAULT_DISC_RADIUS_M:.1e})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = export_experiments(args.experiments, args.output, radius_m=args.radius)

    destination = summary.output_path or f"each <experiment>/{DEFAULT_EXPORT_DIRNAME}"
    print(f"{summary.n_samples} samples -> {destination}")
    for reason in summary.skipped:
        print(f"  skipped: {reason}")

    return 0 if summary.n_samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
