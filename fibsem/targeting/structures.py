from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.patches as mpatches
import numpy as np
import tifffile as tff
from matplotlib.figure import Figure

from fibsem import utils
from fibsem.conversions import microscope_image_to_image_coordinates
from fibsem.segmentation.config import convert_color_names_to_rgb
from fibsem.structures import FibsemImage, FibsemStagePosition, Point
from fibsem.targeting.config import GridScreeningParameters
from fibsem.util import timestamps

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope

# ---------------------------------------------------------------------------
# Rejection reason constants
# ---------------------------------------------------------------------------

REJECT_TOO_SMALL = "too_small"
REJECT_TOO_LARGE = "too_large"
REJECT_EDGE = "too_close_to_edge"
REJECT_GRID_OVERLAP = "grid_overlap"
REJECT_PATTERN_OVERLAP = "pattern_overlap"
REJECT_USER = "user_rejected"
REJECT_SUPERSEDED = "superseded"  # not the best candidate in its grid square

# ---------------------------------------------------------------------------
# Color / class maps
# ---------------------------------------------------------------------------

TARGET_CLASS_MAP = {
    0: "background",
    1: "Cell",
    2: "Grid Bar",
    3: "Beads",
    4: "Contamination",
    5: "Crack",
}
TARGET_COLOR_MAP = {
    0: "black",
    1: "red",
    2: "green",
    3: "cyan",
    4: "orange",
    5: "magenta",
}
TARGET_COLOR_RGB_MAP = convert_color_names_to_rgb(TARGET_COLOR_MAP.values())


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LamellaTarget:
    name: str
    stage_position: FibsemStagePosition
    poi: Point
    score: float = 1.0
    rejection_reason: Optional[str] = None  # None → accepted
    enabled: bool = True  # user-toggle for downstream milling
    pixel_coordinate: Optional[Point] = (
        None  # image-space position (px); set by pipeline
    )
    sub_scores: dict = field(
        default_factory=dict
    )  # per-component scores; not persisted
    alignment_area: Optional[Point] = (
        None  # microscope-image coords of fiducial region on grid bar
    )

    @property
    def is_accepted(self) -> bool:
        return self.rejection_reason is None and self.enabled

    def to_dict(self):
        return {
            "name": self.name,
            "stage_position": self.stage_position.to_dict(),
            "poi": self.poi.to_dict(),
            "score": self.score,
            "rejection_reason": self.rejection_reason,
            "enabled": self.enabled,
            "pixel_coordinate": self.pixel_coordinate.to_dict()
            if self.pixel_coordinate is not None
            else None,
            "sub_scores": self.sub_scores,
            "alignment_area": self.alignment_area.to_dict()
            if self.alignment_area is not None
            else None,
        }

    @staticmethod
    def from_dict(d: dict) -> "LamellaTarget":
        px = d.get("pixel_coordinate")
        aa = d.get("alignment_area")
        return LamellaTarget(
            name=d["name"],
            stage_position=FibsemStagePosition.from_dict(d["stage_position"]),
            poi=Point.from_dict(d["poi"]),
            score=d.get("score", 1.0),
            rejection_reason=d.get("rejection_reason", None),
            enabled=d.get("enabled", True),
            pixel_coordinate=Point.from_dict(px) if px is not None else None,
            sub_scores=d.get("sub_scores", {}),
            alignment_area=Point.from_dict(aa) if aa is not None else None,
        )


@dataclass
class CandidateTarget:
    """Lightweight intermediate used within the detection pipeline.

    Holds pixel-space and microscope-space coordinates alongside a rejection
    reason and score before a final ``LamellaTarget`` (with name/stage_position)
    is constructed.
    """

    pixel_coordinate: Point  # image-space (x=col, y=row in px)
    point: Optional[Point] = None  # microscope-image coordinates (filled by pipeline)
    score: float = 1.0
    rejection_reason: Optional[str] = None
    region_mask: Optional[np.ndarray] = (
        None  # binary mask of this blob, same shape as image
    )
    sub_scores: dict = field(
        default_factory=dict
    )  # per-component scores from score_candidate_targets

    @property
    def is_accepted(self) -> bool:
        return self.rejection_reason is None


@dataclass
class SegmentationTargetResult:
    name: str
    image: FibsemImage
    seg_mask: np.ndarray
    grid_mask: np.ndarray
    candidates: list[LamellaTarget] = field(
        default_factory=list
    )  # all candidates: accepted, rejected, and user-added

    @property
    def targets(self) -> list[LamellaTarget]:
        """Enabled candidates — used for downstream milling."""
        return [t for t in self.candidates if t.enabled]

    @property
    def rejected_targets(self) -> list[LamellaTarget]:
        """Candidates rejected during the detection pipeline."""
        return [t for t in self.candidates if not t.is_accepted]

    def figure(
        self,
        show_seg_mask: bool = True,
        show_grid_mask: bool = True,
        show_names: bool = True,
        show_rejected: bool = False,
        show_grid_centres: bool = False,
        show_scalebar: bool = True,
        selected_target: Optional["LamellaTarget"] = None,
        bg: str = "#262930",
        selected_colour: str = "lime",
        default_colour: str = "cyan",
        rejected_colour: str = "white",
        fig: Optional["Figure"] = None,
        seg_alpha: float = 0.5,
        grid_alpha: float = 0.2,
        intensity_min: float = 0.0,
        intensity_max: float = 100.0,
    ) -> Any:

        if fig is None:
            fig = Figure(facecolor=bg)
        else:
            fig.clf()
            fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)

        dtype = self.image.data.dtype
        dtype_max = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
        vmin = intensity_min / 100.0 * dtype_max
        vmax = intensity_max / 100.0 * dtype_max
        ax.imshow(self.image.data, cmap="gray", vmin=vmin, vmax=vmax)

        # fibsem.segmentation.utils imports torch at module scope; keep the
        # library importable without the [ml] extra.
        from fibsem.segmentation.utils import decode_segmap_v2

        handles = []
        if show_seg_mask and self.seg_mask is not None:
            ax.imshow(
                decode_segmap_v2(self.seg_mask, colormap=TARGET_COLOR_RGB_MAP),
                alpha=seg_alpha,
            )
            for class_idx in np.unique(self.seg_mask):
                if class_idx == 0:
                    continue
                colour = TARGET_COLOR_MAP.get(class_idx, "white")
                label = TARGET_CLASS_MAP.get(class_idx, f"class_{class_idx}")
                handles.append(mpatches.Patch(color=colour, label=label))

        if show_grid_mask and self.grid_mask is not None:
            ax.imshow(
                decode_segmap_v2(
                    self.grid_mask.astype(int),
                    colormap=[(0, 0, 0), (255, 255, 0)],
                ),
                alpha=grid_alpha,
            )
            handles.append(mpatches.Patch(color="yellow", label="Grid Bar Mask"))

        image_shape = self.image.data.shape
        pixelsize = self.image.metadata.pixel_size.x

        for lam in self.candidates:
            if not lam.enabled:
                if not show_rejected:
                    continue
                marker, ms, mew, alpha = "x", 12, 1.5, 0.6
                colour = rejected_colour
                name = f"{lam.name} ({lam.rejection_reason})"
            else:
                marker, ms, mew, alpha = "+", 20, 2, 1.0
                is_selected = (
                    selected_target is not None and lam.name == selected_target.name
                )
                colour = selected_colour if is_selected else default_colour
                name = lam.name

            px = microscope_image_to_image_coordinates(lam.poi, image_shape, pixelsize)
            ax.plot(px.x, px.y, marker, color=colour, ms=ms, mew=mew, alpha=alpha)
            if show_names:
                ax.text(
                    px.x + 10,
                    px.y - 10,
                    name,
                    color=colour,
                    fontsize=8,
                    va="bottom",
                    alpha=alpha,
                    clip_on=True,
                )

            if lam.enabled and lam.alignment_area is not None:
                aa_px = microscope_image_to_image_coordinates(
                    lam.alignment_area, image_shape, pixelsize
                )
                ax.plot(
                    aa_px.x, aa_px.y, "+", color="orange", ms=12, mew=1.5, alpha=0.9
                )
                if show_names:
                    ax.text(
                        aa_px.x + 10,
                        aa_px.y - 10,
                        f"{lam.name} (align)",
                        color="orange",
                        fontsize=8,
                        va="bottom",
                        alpha=0.9,
                        clip_on=True,
                    )

        if show_grid_centres and self.grid_mask is not None:
            from fibsem.targeting.scoring import _find_grid_hole_centers

            for r, c in _find_grid_hole_centers(self.grid_mask):
                ax.plot(
                    c,
                    r,
                    "o",
                    color="yellow",
                    ms=8,
                    mew=1.5,
                    alpha=0.7,
                    fillstyle="none",
                )

        # centre crosshair
        cy, cx = image_shape[0] / 2, image_shape[1] / 2
        ax.plot(cx, cy, "+", color="yellow", ms=16, mew=1.5, alpha=0.8)

        # legend
        if handles:
            ax.legend(
                handles=handles,
                facecolor="#1e2124",
                edgecolor="#3a3d42",
                labelcolor="white",
                fontsize=8,
                loc="upper right",
            )

        if show_scalebar:
            try:
                from matplotlib_scalebar.scalebar import ScaleBar

                ax.add_artist(
                    ScaleBar(
                        dx=pixelsize,
                        color="white",
                        box_color="black",
                        box_alpha=0.4,
                        location="lower right",
                    )
                )
            except ImportError:
                pass

        # lock to image bounds so text near edges doesn't expand the axes
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)  # inverted y for image coords
        ax.axis("off")
        try:
            fig.tight_layout(pad=0)
        except ValueError:
            pass  # zero-size figure during widget init
        return fig


@dataclass
class GridScreeningRun:
    """Complete record of a grid screening run: input parameters + output results."""

    params: GridScreeningParameters = field(default_factory=GridScreeningParameters)
    images: list[FibsemImage] = field(
        default_factory=list
    )  # list[FibsemImage] — acquired tiles (not persisted)
    results: list[SegmentationTargetResult] = field(
        default_factory=list
    )  # list[SegmentationTargetResult]
    name: str = ""
    created_at: Optional[datetime] = field(default_factory=timestamps.now)
    path: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": timestamps.to_iso(self.created_at),
            "path": self.path,
            "params": self.params.to_dict(),
            "results": [
                {
                    "name": r.name,
                    "candidates": [t.to_dict() for t in r.candidates],
                }
                for r in self.results
            ],
        }

    @staticmethod
    def from_dict(d: dict, results: list) -> "GridScreeningRun":
        """Reconstruct from a dict + pre-loaded SegmentationTargetResult list."""
        return GridScreeningRun(
            params=GridScreeningParameters.from_dict(d.get("params", {})),
            results=results,
            name=d.get("name", ""),
            created_at=timestamps.to_datetime(d.get("created_at")),
            path=d.get("path", ""),
        )

    def save(self, path: str) -> None:
        """Save run to disk.

        Directory structure::

            path/
              run.yaml       ← params, result manifest (names + targets), metadata
              <tile>/
                image.tif
                seg_mask.tif
                grid_mask.tif
        """
        self.path = path
        os.makedirs(path, exist_ok=True)

        for result in self.results:
            subdir = os.path.join(path, result.name)
            os.makedirs(subdir, exist_ok=True)
            if result.image is not None:
                result.image.save(os.path.join(subdir, "image.tif"))
            if result.seg_mask is not None:
                tff.imwrite(os.path.join(subdir, "seg_mask.tif"), result.seg_mask)
            if result.grid_mask is not None:
                tff.imwrite(os.path.join(subdir, "grid_mask.tif"), result.grid_mask)

        utils.save_yaml(os.path.join(path, "run.yaml"), self.to_dict())

    @staticmethod
    def load(path: str) -> "GridScreeningRun":
        """Load a run. Falls back to old ``results.yaml`` format if ``run.yaml`` is absent."""

        run_path = os.path.join(path, "run.yaml")
        if not os.path.isfile(run_path):
            raise FileNotFoundError(f"No run.yaml or results.yaml found in {path}")

        # load manifest
        d = utils.load_yaml(run_path)
        manifest = d.get("results", [])

        results = []
        for entry in manifest:
            name = entry["name"]
            subdir = os.path.join(path, name)
            image_path = os.path.join(subdir, "image.tif")
            seg_path = os.path.join(subdir, "seg_mask.tif")
            grid_path = os.path.join(subdir, "grid_mask.tif")
            image = FibsemImage.load(image_path) if os.path.isfile(image_path) else None
            seg = tff.imread(seg_path) if os.path.isfile(seg_path) else None
            grid = tff.imread(grid_path) if os.path.isfile(grid_path) else None
            # backward-compat: old files had "targets" + optional "all_candidates"
            if "candidates" in entry:
                candidates = [LamellaTarget.from_dict(t) for t in entry["candidates"]]
            else:
                candidates = [
                    LamellaTarget.from_dict(t) for t in entry.get("targets", [])
                ]
            results.append(
                SegmentationTargetResult(
                    name=name,
                    image=image,
                    seg_mask=seg,
                    grid_mask=grid,
                    candidates=candidates,
                )
            )

        run = GridScreeningRun.from_dict(d, results)
        run.path = path
        return run

    @property
    def targets(self) -> list[LamellaTarget]:
        """Flat list of all LamellaTarget objects across all results."""
        return [t for result in self.results for t in result.targets]

    def add_to_experiment(
        self, microscope: "FibsemMicroscope", experiment: "Experiment"
    ) -> int:
        """Add all targets from this run to an experiment as lamellae.

        Returns the number of lamellae added.
        """
        for i, target in enumerate(self.targets, 1):
            microscope_state = microscope.get_microscope_state()
            if target.stage_position is not None:
                microscope_state.stage_position = copy.deepcopy(target.stage_position)

            experiment.add_new_lamella(
                microscope_state=microscope_state,
                task_config=experiment.task_protocol.task_config,
                name=f"{i:02d}-{target.name}",
            )
            lamella = experiment.positions[-1]
            lamella.poi = target.poi
            lamella.sync_tasks_to_poi()

        return len(self.targets)

    def to_ml_format(self, output_path: Optional[str] = None) -> None:
        """Export results to ML labelling format (images/ + labels/ directories).

        Caller should load the run first: run = GridScreeningRun.load(src_path)
        Skips any result whose image or seg_mask was not loaded.
        If output_path is None, defaults to self.path/ml-output.
        """
        import tifffile

        if output_path is None:
            if not self.path:
                raise ValueError(
                    "output_path must be provided when run.path is not set."
                )
            output_path = os.path.join(self.path, "ml-output")

        images_dir = os.path.join(output_path, "images")
        labels_dir = os.path.join(output_path, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        idx = 1
        for result in self.results:
            if result.image is None or result.seg_mask is None:
                continue
            stem = f"{idx:04d}"
            result.image.save(os.path.join(images_dir, stem))
            tifffile.imwrite(os.path.join(labels_dir, f"{stem}.tif"), result.seg_mask)
            idx += 1
