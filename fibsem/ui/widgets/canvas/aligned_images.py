"""Images aligned by hand over a real-space canvas: placement and bookkeeping (FIB-1030).

A fluorescence overview laid over the FIB/SEM Overview tab starts where its own
metadata says and is corrected by hand. Two things have to be kept apart for that to
survive a change of view:

* **The geometry's part.** The map from the image's own plane to the view's plane is
  the composition of two projections the repo already has -- image plane to stage
  (`FMStageProjection.from_plane`) and stage to view (`StageProjection.to_plane`).
  Sampled at three points it is a linear map plus an offset, and decomposed it is a
  mirror, a turn, the view's squash and a scale. None of it is the user's to drag.
* **The user's part.** An offset in metres along the sample surface, a turn in degrees
  and a scale, applied on top. The same four numbers in every view, so a correction
  made in SEM @ SEM draws in the right place at FIB @ SEM, and what a record stores.

`AlignedImages` holds one `ImageOverlay` per image on a canvas, re-places them all for
a frame on request, and turns what the overlays emit (a canvas centre, a canvas
rotation) back into the user's part.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.correlation.similarity import SimilarityFit, fit_similarity
from fibsem.fm.preview import composite_projection
from fibsem.imaging.reduce import downsample
from fibsem.projection import FMStageProjection
from fibsem.ui.widgets.canvas.overlays.image_overlay import ImageOverlay

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.structures import FibsemStagePosition
    from fibsem.ui.widgets.canvas.real_space_canvas import FibsemRealSpaceCanvas
    from fibsem.ui.widgets.canvas.stage_frame import StageFrame

logger = logging.getLogger(__name__)

# The composite shown on the canvas is capped here: a 3160 px Arctis overview is
# 30 MB as RGB and is blitted on every drag frame, and at overview scale the
# difference is invisible.
DISPLAY_MAX_PX = 2048
_PROBE = 1e-4  # metres: the step the linear map is sampled with


def image_map(
    fm_projection: FMStageProjection,
    fm_base: "FibsemStagePosition",
    projection,
    origin: "FibsemStagePosition",
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """The image's own plane onto the view's plane, as (A 2x2, offset), in metres.

    `A` takes an offset in the image's plane to an offset in the view's plane;
    `offset` is where the image's centre falls in the view, relative to its origin.
    """

    def go(dx: float, dy: float) -> np.ndarray:
        return np.array(
            projection.to_plane(fm_projection.from_plane(dx, dy, fm_base), origin),
            dtype=float,
        )

    o = go(0.0, 0.0)
    a = np.column_stack(
        [(go(_PROBE, 0.0) - o) / _PROBE, (go(0.0, _PROBE) - o) / _PROBE]
    )
    return a, (float(o[0]), float(o[1]))


def decompose(a: np.ndarray) -> Tuple[bool, float, float, float]:
    """A 2x2 as (mirror, rotation, squash, scale) -- what `set_placement` takes.

    The model is `A = diag(1, squash) . R(rotation) . diag(-1 if mirror else 1, 1) .
    scale`: a mirror of the image's own x, a turn clockwise on a y-down canvas, the
    view's squash along its y, and a uniform scale. A view's foreshortening is a
    squash along the view's y only (FIB-615), which is what makes the top row of `A`
    the unsquashed one and lets the turn be read off it.
    """
    mirror = bool(np.linalg.det(a) < 0)
    m = a @ np.diag([-1.0 if mirror else 1.0, 1.0])
    scale = math.hypot(m[0, 0], m[0, 1])
    if scale == 0:
        return mirror, 0.0, 1.0, 1.0
    rotation = math.degrees(math.atan2(-m[0, 1], m[0, 0]))
    c, s = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    squash = (m[1, 0] * s + m[1, 1] * c) / scale
    return mirror, rotation, (squash if squash else 1.0), scale


@dataclass
class AlignedImage:
    """One image on the canvas: what it is, how it projects, how it was corrected."""

    key: str
    label: str
    rgb: np.ndarray  # the display composite, (H, W, 3)
    shape: Tuple[int, int]  # the image's own (height, width), pixels
    pixel_size: float  # metres
    projection: FMStageProjection
    base: "FibsemStagePosition"
    overlay: ImageOverlay
    path: Optional[str] = None
    # The user's part, on the sample: metres along the surface, degrees, a factor.
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    # The geometry's part for the view last placed in, so an emission from the
    # overlay (canvas units, the whole map applied) can be taken back apart.
    anchor: Tuple[float, float] = (0.0, 0.0)  # canvas coords of the metadata centre
    base_rotation: float = 0.0
    squash: float = 1.0
    per_metre: float = 1.0  # canvas units per metre, at the last placement
    channels: List[str] = field(default_factory=list)
    # The id of the record a host keeps this placement under, once it has one.
    record_id: Optional[str] = None
    # How the placement was last fitted from point pairs, for the record: the pairs
    # (image pixel x, y, reference canvas x, y), the residual per pair and the RMS,
    # in canvas units. Empty for a placement made by hand.
    fit: Dict[str, object] = field(default_factory=dict)

    @property
    def placement(self) -> Tuple[float, float, float, float]:
        return self.dx, self.dy, self.rotation, self.scale


class AlignedImages(QObject):
    """The images aligned over one canvas, and where the user has put them."""

    # The user changed an image's placement: a drag ended, or a reset. Not emitted
    # by `set_placement`, so a host restoring a record cannot be made to write it
    # back.
    placement_changed = pyqtSignal(str)  # key

    def __init__(self, canvas: "FibsemRealSpaceCanvas", parent=None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._images: Dict[str, AlignedImage] = {}
        self._count = 0
        self._frame: Optional["StageFrame"] = None

    # ── the set ───────────────────────────────────────────────────────────

    def keys(self) -> List[str]:
        return list(self._images)

    def get(self, key: str) -> Optional[AlignedImage]:
        return self._images.get(key)

    def add(
        self,
        image: "FluorescenceImage",
        label: Optional[str] = None,
        path: Optional[str] = None,
        rgb: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Place a fluorescence image from its own metadata. None if it cannot be.

        `rgb` lets a caller hand in a composite it already holds; otherwise the
        channels are max-projected and tinted here. Either way the display copy is
        capped at `DISPLAY_MAX_PX`.
        """
        projection = FMStageProjection.from_image(image)
        base = getattr(image.metadata, "stage_position", None)
        pixel_size = getattr(image.metadata, "pixel_size_x", None)
        if projection is None or base is None or not pixel_size:
            missing = [
                name
                for name, value in (
                    ("camera geometry", projection),
                    ("stage position", base),
                    ("pixel size", pixel_size),
                )
                if not value
            ]
            logger.warning(
                f"{label or path or 'A fluorescence image'} cannot be placed: "
                f"no {' or '.join(missing)} in its metadata."
            )
            return None
        if rgb is None:
            rgb = composite_projection(image)
        if max(rgb.shape[:2]) > DISPLAY_MAX_PX:
            rgb = downsample(rgb, max_px=DISPLAY_MAX_PX)
        self._count += 1
        key = f"aligned-{self._count}"
        overlay = ImageOverlay()
        self._canvas.add_overlay(overlay)
        overlay.moved.connect(lambda x, y, key=key: self._on_moved(key, x, y))
        overlay.rotated.connect(lambda r, key=key: self._on_rotated(key, r))
        overlay.drag_finished.connect(lambda key=key: self.placement_changed.emit(key))
        record = AlignedImage(
            key=key,
            label=label or (os_basename(path) if path else key),
            rgb=np.ascontiguousarray(rgb),
            shape=tuple(np.asarray(image.data).shape[-2:]),
            pixel_size=float(pixel_size),
            projection=projection,
            base=base,
            overlay=overlay,
            path=path,
            channels=[c.name for c in getattr(image.metadata, "channels", [])],
        )
        self._images[key] = record
        overlay.set_visible(True)
        self._place(record)
        return key

    def remove(self, key: str) -> bool:
        record = self._images.pop(key, None)
        if record is None:
            return False
        self._canvas.remove_overlay(record.overlay)
        return True

    def clear(self) -> None:
        for key in list(self._images):
            self.remove(key)

    # ── placement ─────────────────────────────────────────────────────────

    def refresh(self, frame: Optional["StageFrame"]) -> None:
        """Re-place every image for *frame* -- the view changed, or the canvas scale."""
        self._frame = frame
        for record in self._images.values():
            self._place(record)

    def set_placement(
        self, key: str, dx: float, dy: float, rotation: float, scale: float = 1.0
    ) -> None:
        """The user's part, in the units :attr:`AlignedImage.placement` reports."""
        record = self._images.get(key)
        if record is None:
            return
        record.dx, record.dy = float(dx), float(dy)
        record.rotation, record.scale = float(rotation), float(scale) or 1.0
        self._place(record)

    def reset(self, key: str) -> None:
        self.set_placement(key, 0.0, 0.0, 0.0, 1.0)
        record = self._images.get(key)
        if record is not None:
            record.fit = {}
        self.placement_changed.emit(key)

    def set_opacity(self, key: str, opacity: float) -> None:
        record = self._images.get(key)
        if record is not None:
            record.overlay.set_opacity(opacity)

    def set_visible(self, key: str, visible: bool) -> None:
        record = self._images.get(key)
        if record is not None:
            record.overlay.set_visible(visible)

    def _place(self, record: AlignedImage) -> None:
        frame = self._frame
        if frame is None:
            record.overlay.set_visible(False)
            return
        try:
            a, offset = image_map(
                record.projection, record.base, frame.projection, frame.origin
            )
            mirror, base_rotation, squash, base_scale = decompose(a)
            anchor = self._canvas.metres_to_canvas(*offset)
            per_metre = frame.length(1.0)
        except Exception as e:  # noqa: BLE001 - a frame that cannot place it
            logger.debug(f"Could not place {record.label}: {e}")
            record.overlay.set_visible(False)
            return
        record.anchor = (float(anchor[0]), float(anchor[1]))
        record.base_rotation = float(base_rotation)
        record.squash = float(squash)
        record.per_metre = float(per_metre)
        height, width = record.shape
        size = record.pixel_size * base_scale * record.scale
        record.overlay.set_image(
            record.rgb,
            frame.length(width * size),
            frame.length(height * size),
            centre=(
                anchor[0] + record.dx * per_metre,
                anchor[1] + record.dy * per_metre * squash,
            ),
            rotation=base_rotation + record.rotation,
            squash=squash,
            mirror=mirror,
        )
        record.overlay.set_visible(True)

    # ── placed from point pairs ───────────────────────────────────────────

    def pixel_to_canvas(self, key: str, x: float, y: float) -> Tuple[float, float]:
        """Where a pixel of the image's display composite falls on the canvas now.

        Through the overlay's own map, so the answer is the placement as drawn --
        the geometry's part and the user's part together.
        """
        record = self._images[key]
        height, width = record.rgb.shape[:2]
        fw, fh = record.overlay.footprint
        u = ((x + 0.5) / width - 0.5) * fw
        v = ((y + 0.5) / height - 0.5) * fh
        return record.overlay.to_canvas(u, v)

    def fit_to_points(
        self,
        key: str,
        image_pixels,
        reference_points,
        fix_scale: bool = False,
    ) -> SimilarityFit:
        """Place the image so its *image_pixels* land on *reference_points*.

        *image_pixels* are (x, y) in the display composite; *reference_points* are
        canvas coordinates, where the same features are in the picture underneath. The
        fit is a similarity on the *sample* -- both sets are unsquashed by the view's
        foreshortening first -- so a fit made in a tilted view asks for the same
        correction as one made looking straight down. What comes out replaces the
        user's part: the image's centre goes where the fit sends it, its turn and
        scale take the fit's on top of what they were. The pairs and residuals are
        kept on the record. Announces, as a drag does.
        """
        record = self._images[key]
        pixels = np.asarray(image_pixels, dtype=float).reshape(-1, 2)
        targets = np.asarray(reference_points, dtype=float).reshape(-1, 2)
        placed = np.array([self.pixel_to_canvas(key, x, y) for x, y in pixels])
        squash = record.squash or 1.0
        ay = record.anchor[1]

        def unsquash(points):
            out = points.copy()
            out[:, 1] = ay + (out[:, 1] - ay) / squash
            return out

        fit = fit_similarity(
            unsquash(placed), unsquash(targets), fix_scale=fix_scale, scale=1.0
        )
        # The composed map is still a similarity, and the placement is about the
        # image's centre: send the centre through the fit, add the turn and scale.
        centre = np.array(record.overlay.centre, dtype=float)
        new_centre = fit.apply(unsquash(centre.reshape(1, 2)))[0]
        record.dx = float((new_centre[0] - record.anchor[0]) / record.per_metre)
        record.dy = float((new_centre[1] - ay) / record.per_metre)
        record.rotation = float(record.rotation + fit.rotation)
        record.scale = float(record.scale * fit.scale)
        record.fit = {
            "pairs": [
                [float(px), float(py), float(rx), float(ry)]
                for (px, py), (rx, ry) in zip(pixels, targets)
            ],
            "residuals": [float(r) for r in fit.residuals],
            "rms": float(fit.rms),
            "scale": float(fit.scale),
            "fix_scale": bool(fix_scale),
        }
        self._place(record)
        logger.info(
            f"Fitted {record.label} from {len(pixels)} pairs: RMS {fit.rms:.2f} canvas px,"
            f" turn {fit.rotation:+.2f} deg, scale x{fit.scale:.4f}"
            f"{' (locked)' if fix_scale else ''}; placement now dx={record.dx:.3e} m"
            f" dy={record.dy:.3e} m rotation={record.rotation:.2f} deg"
            f" scale={record.scale:.4f}"
        )
        self.placement_changed.emit(key)
        return fit

    # ── what the overlays emit, taken back apart ───────────────────────────

    def _on_moved(self, key: str, cx: float, cy: float) -> None:
        record = self._images.get(key)
        if record is None or not record.per_metre:
            return
        # Plain floats: a frame's numbers arrive as numpy scalars, and a record
        # holding one cannot be written to the experiment file.
        record.dx = float((cx - record.anchor[0]) / record.per_metre)
        record.dy = float(
            (cy - record.anchor[1]) / record.per_metre / (record.squash or 1.0)
        )
        record.fit = {}
        self._place(record)

    def _on_rotated(self, key: str, rotation: float) -> None:
        record = self._images.get(key)
        if record is None:
            return
        record.rotation = float(rotation - record.base_rotation)
        record.fit = {}
        self._place(record)


def os_basename(path: str) -> str:
    import os

    return os.path.basename(path)
