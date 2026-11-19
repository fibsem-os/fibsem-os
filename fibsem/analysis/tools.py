"""The image tools over one root folder: a saved image's metadata, a point on
it as a stage position and back, a drawing of it, two side by side, and
measurements on it. Every path resolves inside the root (`fibsem.analysis.root`),
and every picture is saved beside its image as ``drawn/<name>.png``.

A tool that can't answer raises :class:`ImageRefused`, with an error type the
server answers as its HTTP status."""

import os
from typing import Any, Callable, Dict, List, Optional

from fibsem.analysis import measure
from fibsem.analysis.answers import (
    ImageDrawn,
    ImageMeasured,
    ImagePairDrawn,
    PixelPoint,
    SavedImageMetadata,
    SavedImagePixels,
    SavedImagePosition,
)
from fibsem.analysis.images import load_planes
from fibsem.analysis.root import FileRefusal, resolve
from fibsem.fm.preview import is_fluorescence_image
from fibsem.geometry.saved_images import image_point_to_stage_position, stage_projector
from fibsem.imaging.cells import from_crop
from fibsem.imaging.marks import (
    draw,
    draw_pair,
    drawn_path,
    image_stem,
    plane_shown,
    save_drawn,
    side_by_side,
)
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import FibsemStagePosition
from fibsem.workflows.evidence import EvidenceRefused, checked_marks


class ImageRefused(Exception):
    """Why an image tool did nothing: an error type and a message."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class ImageTools:
    """The image tools over the folder *root* answers: a callable, so the
    root can follow what is open (None: nothing is)."""

    def __init__(self, root: Callable[[], Optional[str]]):
        self._root = root

    def image(self, path: str) -> str:
        """The real path of the file *path* inside the root."""
        try:
            target = resolve(self._root(), path)
        except FileRefusal as refusal:
            raise ImageRefused(refusal.error_type, refusal.message) from None
        if not os.path.isfile(target):
            raise ImageRefused("not_found", f"{path} is a folder.")
        return target

    def metadata(self, path: str) -> SavedImageMetadata:
        """A saved beam image's full metadata, as its file records it."""
        image = FibsemImage.load(self.image(path))
        if image.metadata is None:
            raise ImageRefused("not_found", f"{path} carries no beam image metadata.")
        return SavedImageMetadata(path, image.metadata)

    def pixel_to_stage(
        self, path: str, px=None, fraction=None, crop=None
    ) -> SavedImagePosition:
        """The stage position under a point of a saved image, beam or
        fluorescence, through the projection the overview canvases use, from
        the image's own metadata: as the image was taken, wherever the stage is
        now. ``px`` is [x, y] in source pixels (origin top-left, +y down);
        ``fraction`` is [x, y] from 0 to 1. With *crop* (a cell's name or a
        box), both are read off that crop's picture (`fibsem.imaging.cells`)."""
        target = self.image(path)
        try:
            if crop is not None:
                px, fraction = _uncropped_point(target, crop, px, fraction)
            position = image_point_to_stage_position(target, px=px, fraction=fraction)
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        return SavedImagePosition(path, position)

    def stage_to_pixel(
        self, path: str, stage_positions: List[Dict[str, Any]]
    ) -> SavedImagePixels:
        """Where each stage position falls on a saved image, in source pixels,
        with the image's scale block. A position the image cannot place (no
        geometry for it) answers its reason instead."""
        target = self.image(path)
        try:
            positions = [FibsemStagePosition.from_dict(p) for p in stage_positions]
        except (TypeError, KeyError, AttributeError) as exc:
            raise ImageRefused("invalid", f"Not a stage position: {exc}") from None
        block, project = stage_projector(target)
        points = []
        for position in positions:
            try:
                point = project(position)
            except Exception as exc:  # noqa: BLE001 - per-position answer
                points.append(PixelPoint(reason=str(exc)))
                continue
            x, y = float(point.x), float(point.y)
            inside = 0 <= x <= block["width"] and 0 <= y <= block["height"]
            points.append(PixelPoint(px=[x, y], inside=inside))
        return SavedImagePixels(path, block, points)

    def draw_image(
        self,
        path: str,
        marks: Any = None,
        crosshair: bool = False,
        metadata_bar: bool = True,
        channels: Optional[List[str]] = None,
        plane: Optional[int] = None,
        looks: Optional[Dict[str, Dict[str, Any]]] = None,
        box: Optional[List[float]] = None,
    ) -> ImageDrawn:
        """A saved image (beam, or a fluorescence stack's *plane* or max
        projection) with *marks* (the one marks vocabulary, its pixels) drawn
        on it, as evidence draws them, the centre *crosshair*, the scale bar
        and the metadata bar. Only the region *box* (x, y, width, height in its
        pixels) when given; each layer drawn with its look from *looks* (by
        name: limits, gamma, opacity, colour, as the viewer's layers panel sets
        them). Saved as ``drawn/<image>.png`` beside the image
        (``<image>-plane-<n>.png`` for a plane, ``-box-<x>-<y>-<w>-<h>`` for a
        region; replaced when drawn again), a path to cite as evidence.
        Answers each layer's look and stats."""
        target = self.image(path)
        checked = _checked_marks(marks, target)
        try:
            drawn = draw(
                target,
                checked,
                crosshair=crosshair,
                metadata_bar=metadata_bar,
                channels=channels,
                plane=plane,
                looks=looks,
                box=box,
            )
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        saved = drawn_path(target, _drawn_stem(target, checked, plane, box))
        return ImageDrawn(
            path,
            saved,
            checked,
            save_drawn(drawn.rgb, saved, target, checked, plane),
            drawn.layers,
        )

    def draw_pair(
        self,
        before: str,
        after: str,
        marks: Any = None,
        metadata_bar: bool = True,
        channels: Optional[List[str]] = None,
        before_plane: Optional[int] = None,
        after_plane: Optional[int] = None,
        looks: Optional[Dict[str, Dict[str, Any]]] = None,
        box: Optional[List[float]] = None,
        share_looks: bool = True,
    ) -> ImagePairDrawn:
        """Two saved images side by side at one scale, each with its own bars,
        *marks* (in *before*'s pixels) drawn on both: pre and post mill, before
        and after an alignment, SEM next to FIB, two planes of one
        fluorescence stack (*before_plane*, *after_plane*). *box* (in
        *before*'s pixels) draws that region of each. Both take *looks*; with
        *share_looks*, *after* is drawn with *before*'s limits wherever *looks*
        leaves one on auto. Saved as ``drawn/<before>-<after>.png`` beside
        *before*."""
        first, second = self.image(before), self.image(after)
        checked = _checked_marks(marks, first)
        try:
            left, right = draw_pair(
                first,
                second,
                checked,
                metadata_bar=metadata_bar,
                channels=channels,
                before_plane=before_plane,
                after_plane=after_plane,
                looks=looks,
                box=box,
                share_looks=share_looks,
            )
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        saved = drawn_path(
            first,
            f"{_drawn_stem(first, checked, before_plane, box)}"
            f"-{_drawn_stem(second, checked, after_plane, box)}",
        )
        return ImagePairDrawn(
            before,
            after,
            saved,
            checked,
            save_drawn(
                side_by_side(left.rgb, right.rgb), saved, first, checked, before_plane
            ),
            left.layers,
            right.layers,
        )

    def _measured(self, path: str, run, figure=None, name: str = "") -> ImageMeasured:
        """*run* (and its *figure*, saved as ``drawn/<image>-<name>.png``) on
        the saved image *path*."""
        target = self.image(path)
        try:
            planes = load_planes(target)
            result = run(planes)
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        if figure is None:
            return ImageMeasured(path, name, result)
        saved = drawn_path(target, f"{image_stem(target)}-{name}")
        return ImageMeasured(
            path, name, result, saved, save_drawn(figure(planes, result), saved)
        )

    def measure_intensity(
        self,
        path: str,
        points: List[List[float]],
        radius_px: float = 3.0,
        channel: Optional[str] = None,
        reference: Optional[int] = None,
        planes: Optional[List[int]] = None,
    ) -> ImageMeasured:
        """The signal at *points* of a saved image, per plane, with a figure
        (`fibsem.analysis.measure.measure_intensity`)."""
        return self._measured(
            path,
            lambda image: measure.measure_intensity(
                image, points, radius_px, channel, reference, planes
            ),
            measure.intensity_figure,
            "intensity",
        )

    def line_profile(
        self,
        path: str,
        start: List[float],
        end: List[float],
        width_px: float = 1.0,
        channel: Optional[str] = None,
        plane: Optional[int] = None,
        dip: bool = False,
    ) -> ImageMeasured:
        """A saved image's values along a line, and the FWHM of its peak (or
        *dip*), with a figure (`fibsem.analysis.measure.line_profile`)."""
        line = "-".join(str(round(v)) for v in (*start, *end))
        return self._measured(
            path,
            lambda image: measure.line_profile(
                image, start, end, width_px, channel, plane, dip
            ),
            measure.profile_figure,
            f"profile-{line}",
        )

    def image_quality(
        self, path: str, channel: Optional[str] = None, plane: Optional[int] = None
    ) -> ImageMeasured:
        """A saved image's clipping, histogram, focus and, for a stack, each
        plane's sharpness (`fibsem.analysis.measure.image_quality`)."""
        return self._measured(
            path,
            lambda image: measure.image_quality(image, channel, plane),
            name="quality",
        )


def _uncropped_point(target: str, crop: Any, px, fraction):
    """``(px, None)``: the image's pixels under a point read off *crop*'s
    picture, *px* in its pixels or *fraction* of it. Raises ValueError."""
    scale, x0, y0, box = from_crop(target, crop)
    if (px is None) == (fraction is None):
        raise ValueError("Give exactly one of px ([x, y] source pixels) or fraction.")
    if px is not None:
        return [px[0] * scale + x0, px[1] * scale + y0], None
    return [x0 + fraction[0] * box[2], y0 + fraction[1] * box[3]], None


def _drawn_stem(
    path: str, marks: Any, plane: Optional[int], box: Optional[List[float]] = None
) -> str:
    """The name a picture of *path* is saved by: its stem, with the plane of a
    stack it shows (`plane_shown`) and the region *box*, so each is kept."""
    shown, _ = plane_shown(marks, plane, is_fluorescence_image(path))
    stem = image_stem(path)
    if shown is not None:
        stem = f"{stem}-plane-{shown + 1}"
    if box is not None:
        stem += "-box-" + "-".join(str(round(v)) for v in box)
    return stem


def _checked_marks(marks: Any, image_path: str):
    """*marks* checked and placed on the stage."""
    try:
        return checked_marks(marks, image_path)
    except EvidenceRefused as refusal:
        raise ImageRefused(refusal.error_type, refusal.message) from None
