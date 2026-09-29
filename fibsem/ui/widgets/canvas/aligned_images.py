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
rotation, a factor on the size) back into the user's part.

How an image looks is kept apart from where it is. Each channel's max projection is
held at display size as an `FMLayer`, so a colour, a hidden channel or a contrast
change re-blends what is in hand without touching the file or the placement. And by
default an image is drawn *signal only*: brightness becomes coverage (`to_rgba`), so
where a fluorescence map holds nothing the overview beneath shows through rather than
a dimmed black rectangle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from fibsem.correlation.similarity import SimilarityFit, fit_similarity
from fibsem.fm.composite import FMLayer, composite_fm_layers, to_rgba
from fibsem.fm.preview import projection_layers
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
    rgb: np.ndarray  # the display composite, (H, W, 3), uint8
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
    # Mirrored left to right about the image's own axis, on top of whatever mirror
    # the geometry supplies: for an image whose camera the file cannot describe.
    # One axis is enough -- a top-to-bottom flip is this and a half turn.
    mirrored: bool = False
    # The geometry's part for the view last placed in, so an emission from the
    # overlay (canvas units, the whole map applied) can be taken back apart.
    anchor: Tuple[float, float] = (0.0, 0.0)  # canvas coords of the metadata centre
    base_rotation: float = 0.0
    squash: float = 1.0
    per_metre: float = 1.0  # canvas units per metre, at the last placement
    # Each channel's max projection at display size, and how it is shown: what the
    # composite in `rgb` is blended from, and what a colour edit changes.
    layers: List[FMLayer] = field(default_factory=list)
    # Brightness as coverage: dark is clear, so the overview shows through.
    signal_only: bool = True
    # The id of the record a host keeps this placement under, once it has one.
    record_id: Optional[str] = None
    # How the placement was last fitted from point pairs, for the record: the pairs
    # (image pixel x, y, reference canvas x, y), the residual per pair and the RMS,
    # in canvas units. Empty for a placement made by hand.
    fit: Dict[str, object] = field(default_factory=dict)
    # What is drawn: `rgb` with an alpha channel, made once per composite rather
    # than on every drag frame. None until asked for.
    drawn: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def placement(self) -> Tuple[float, float, float, float]:
        return self.dx, self.dy, self.rotation, self.scale

    @property
    def channels(self) -> List[str]:
        return [layer.name for layer in self.layers]


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
        # Each image's map into the view, with the frame it was made for: see
        # `_map_for`. Keyed like `_images`.
        self._maps: Dict[str, tuple] = {}

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
    ) -> Optional[str]:
        """Place a fluorescence image from its own metadata. None if it cannot be.

        Each channel is max-projected over z and reduced to `DISPLAY_MAX_PX` once,
        here; the composite is blended from those, in each channel's own colour.
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
        layers = projection_layers(image)
        for layer in layers:
            if max(layer.data.shape[:2]) > DISPLAY_MAX_PX:
                layer.data = downsample(layer.data, max_px=DISPLAY_MAX_PX)
        rgb = composite_fm_layers(layers)
        if rgb is None:
            logger.warning(
                f"{label or path or 'A fluorescence image'} has no channels to show."
            )
            return None
        self._count += 1
        key = f"aligned-{self._count}"
        overlay = ImageOverlay()
        self._canvas.add_overlay(overlay)
        overlay.moved.connect(lambda x, y, key=key: self._on_moved(key, x, y))
        overlay.rotated.connect(lambda r, key=key: self._on_rotated(key, r))
        overlay.scaled.connect(lambda f, key=key: self._on_scaled(key, f))
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
            layers=layers,
        )
        self._images[key] = record
        overlay.set_visible(True)
        self._place(record)
        return key

    def remove(self, key: str) -> bool:
        record = self._images.pop(key, None)
        self._maps.pop(key, None)
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
        self,
        key: str,
        dx: float,
        dy: float,
        rotation: float,
        scale: float = 1.0,
        mirrored: Optional[bool] = None,
    ) -> None:
        """The user's part, in the units :attr:`AlignedImage.placement` reports.
        *mirrored* None leaves the mirror as it is."""
        record = self._images.get(key)
        if record is None:
            return
        record.dx, record.dy = float(dx), float(dy)
        record.rotation, record.scale = float(rotation), float(scale) or 1.0
        if mirrored is not None:
            record.mirrored = bool(mirrored)
        self._place(record)

    def set_mirrored(self, key: str, on: bool) -> None:
        """Mirror an image left to right about its own axis, or undo it. The user's
        doing, so announced; a fit made the other way round no longer describes it."""
        record = self._images.get(key)
        if record is None or record.mirrored == bool(on):
            return
        record.mirrored = bool(on)
        record.fit = {}
        self._place(record)
        self.placement_changed.emit(key)

    def reset(self, key: str) -> None:
        self.set_placement(key, 0.0, 0.0, 0.0, 1.0, mirrored=False)
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

    # ── how it looks ──────────────────────────────────────────────────────

    def recomposite(self, key: str) -> None:
        """Re-blend an image from its layers -- a channel was recoloured, hidden or
        re-contrasted -- and redraw it where it is."""
        record = self._images.get(key)
        if record is None:
            return
        rgb = composite_fm_layers(record.layers, shape=record.rgb.shape[:2])
        record.rgb = np.ascontiguousarray(rgb)
        record.drawn = None
        self._place(record)

    def set_signal_only(self, key: str, on: bool) -> None:
        """Draw dark as clear (the overview shows through), or the whole frame."""
        record = self._images.get(key)
        if record is None or record.signal_only == bool(on):
            return
        record.signal_only = bool(on)
        record.drawn = None
        self._place(record)

    def display_state(self, key: str) -> Dict[str, object]:
        """How an image is shown, as plain values a record can keep."""
        record = self._images[key]
        return {
            "opacity": float(record.overlay.opacity),
            "signal_only": bool(record.signal_only),
            "channels": [
                {
                    "name": str(layer.name),
                    "color": str(layer.color),
                    "visible": bool(layer.visible),
                    "opacity": float(layer.opacity),
                    "gamma": float(layer.gamma),
                    "autocontrast": bool(layer.autocontrast),
                    "clim": None
                    if layer.autocontrast or layer.clim is None
                    else [float(layer.clim[0]), float(layer.clim[1])],
                }
                for layer in record.layers
            ],
        }

    def set_display_state(self, key: str, state: Dict[str, object]) -> None:
        """Show an image the way *state* says -- what :meth:`display_state` gave.

        Channels are matched by name, else by position when the counts agree; a
        channel the state does not mention keeps its own colour. Does not announce.
        """
        record = self._images.get(key)
        if record is None or not state:
            return
        if state.get("opacity") is not None:
            record.overlay.set_opacity(float(state["opacity"]))
        if state.get("signal_only") is not None:
            record.signal_only = bool(state["signal_only"])
        saved = list(state.get("channels") or [])
        by_name = {str(c.get("name")): c for c in saved if isinstance(c, dict)}
        for index, layer in enumerate(record.layers):
            entry = by_name.get(layer.name)
            if entry is None and len(saved) == len(record.layers):
                entry = saved[index] if isinstance(saved[index], dict) else None
            if entry is None:
                continue
            layer.color = str(entry.get("color") or layer.color)
            layer.visible = bool(entry.get("visible", True))
            layer.opacity = float(entry.get("opacity", 1.0))
            layer.gamma = float(entry.get("gamma", 1.0))
            clim = entry.get("clim")
            layer.autocontrast = bool(entry.get("autocontrast", True)) or not clim
            layer.manual = not layer.autocontrast
            layer.clim = (
                None if layer.autocontrast else (float(clim[0]), float(clim[1]))
            )
        self.recomposite(key)

    def _drawn(self, record: AlignedImage) -> np.ndarray:
        if record.drawn is None:
            if record.signal_only:
                record.drawn = to_rgba(record.rgb)
            else:
                opaque = np.full(record.rgb.shape[:2], 255, dtype=np.uint8)
                record.drawn = np.dstack([record.rgb, opaque])
        return record.drawn

    def _place(self, record: AlignedImage) -> None:
        frame = self._frame
        if frame is None:
            record.overlay.set_visible(False)
            return
        try:
            a, offset = self._map_for(record, frame)
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
            self._drawn(record),
            frame.length(width * size),
            frame.length(height * size),
            centre=(
                anchor[0] + record.dx * per_metre,
                anchor[1] + record.dy * per_metre * squash,
            ),
            rotation=base_rotation + record.rotation,
            squash=squash,
            mirror=mirror != record.mirrored,
        )
        record.overlay.set_visible(True)

    def _map_for(self, record: AlignedImage, frame: "StageFrame"):
        """The image's map into *frame*'s view, made once per view.

        It depends only on the image's own projection and the view's -- not on the
        user's placement or the canvas scale -- but a drag redraws through it on
        every mouse move and a stage move rebuilds an equal frame, and each making
        runs three stage transforms. Kept until the view's origin or projection
        changes, compared by value, so an equal frame rebuilt still hits.
        """
        cached = self._maps.get(record.key)
        if cached is not None:
            made_for, a, offset = cached
            if (
                made_for.origin == frame.origin
                and made_for.projection == frame.projection
            ):
                return a, offset
        a, offset = image_map(
            record.projection, record.base, frame.projection, frame.origin
        )
        self._maps[record.key] = (frame, a, offset)
        return a, offset

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

    def _on_scaled(self, key: str, factor: float) -> None:
        # The scale multiplies the pixel size where the footprint is worked out, so
        # this is the pixel size corrected -- kept apart from the file's own, which
        # Reset goes back to. About the centre: the offset is untouched.
        record = self._images.get(key)
        if record is None or not factor > 0:
            return
        record.scale = float(record.scale * factor)
        record.fit = {}
        self._place(record)

    # ── an image the file cannot place ─────────────────────────────────────

    def base_at(
        self,
        projection: FMStageProjection,
        pose: "FibsemStagePosition",
        target: Tuple[float, float],
    ) -> "FibsemStagePosition":
        """Where an image taken at *pose* must have been centred for its centre to
        fall on canvas point *target* in the current view.

        For an imported image, whose file says nothing about where it was taken: it
        starts where the user is looking, and upright -- of *pose*'s rotation and
        the one opposite, whichever shows it the way the file does. The map from
        stage to view is affine over a grid, so a step of Newton's method lands the
        centre; a second takes out rounding.
        """
        import copy

        frame = self._frame
        if frame is None:
            raise ValueError("there is no view to place the image in yet")
        want = np.array(self._canvas.canvas_to_metres(*target), dtype=float)

        def turn_at(r: float) -> float:
            candidate = copy.deepcopy(frame.origin)
            candidate.r, candidate.t = r, pose.t
            a, _ = image_map(projection, candidate, frame.projection, frame.origin)
            return abs(decompose(a)[1])

        base = copy.deepcopy(frame.origin)
        base.r = min((pose.r, pose.r + math.pi), key=turn_at)
        base.t = pose.t

        def centre_of(position) -> np.ndarray:
            return np.array(
                image_map(projection, position, frame.projection, frame.origin)[1]
            )

        for _ in range(2):
            here = centre_of(base)
            columns = []
            for axis in ("x", "y"):
                probe = copy.deepcopy(base)
                setattr(probe, axis, getattr(probe, axis) + _PROBE)
                columns.append((centre_of(probe) - here) / _PROBE)
            step = np.linalg.solve(np.column_stack(columns), want - here)
            base.x = float(base.x + step[0])
            base.y = float(base.y + step[1])
        return base


def os_basename(path: str) -> str:
    import os

    return os.path.basename(path)
