"""The image tools over one root folder: a saved image's metadata, a point on
it as a stage position and back, a drawing of it, two side by side, and
measurements on it. Every path resolves inside the root (`fibsem.analysis.root`),
and every picture is saved beside its image as ``drawn/<name>.png``.

A tool that can't answer raises :class:`ImageRefused`, with an error type the
server answers as its HTTP status."""

import base64
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from fibsem.alignment import (
    AlignmentMethod,
    _calculate_shift,
    compare_alignment_methods,
)
from fibsem.analysis import coincidence, lattice, measure
from fibsem.analysis.answers import (
    ImageDrawn,
    ImageMeasured,
    ImagePairDrawn,
    ImageShift,
    PixelPoint,
    SavedImageMetadata,
    SavedImagePixels,
    SavedImagePosition,
)
from fibsem.analysis.images import load_planes
from fibsem.analysis.root import FileRefusal, resolve
from fibsem.conversions import image_to_microscope_image_coordinates
from fibsem.fm.preview import is_fluorescence_image
from fibsem.geometry.saved_images import (
    image_point_to_stage_position,
    image_points_to_stage,
    stage_projector,
)
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
from fibsem.imaging.thickness import DEFAULT_WINDOW, measure_lamella_band
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import BeamType, FibsemStagePosition, Point
from fibsem.workflows.evidence import EvidenceRefused, checked_marks

logger = logging.getLogger(__name__)


class ImageRefused(Exception):
    """Why an image tool did nothing: an error type and a message."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


# fibsem's own image files: a FIB-SEM image, or a fluorescence stack.
FIBSEM_FILES = (".tif", ".tiff")


def _fibsem_file(target: str, path: str, what: str) -> None:
    """Refuse plainly a file that isn't one of fibsem's own images: another
    instrument's image, or a picture, has none of the metadata *what* reads."""
    if not target.lower().endswith(FIBSEM_FILES):
        raise ImageRefused(
            "invalid",
            f"{path}: {what} reads a FIB-SEM or fluorescence image's own metadata, "
            "and this file isn't one (another instrument's image, or a picture).",
        )


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
        target = self.image(path)
        _fibsem_file(target, path, "the image's metadata")
        image = FibsemImage.load(target)
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
        _fibsem_file(target, path, "the stage position under a point")
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
        _fibsem_file(target, path, "where a stage position falls")
        try:
            positions = [FibsemStagePosition.from_dict(p) for p in stage_positions]
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            # A missing coordinate_system is a pydantic ValueError: refused
            # as the caller's, never a server error.
            raise ImageRefused(
                "invalid",
                "Not a stage position: give each as x, y, z, r, t and "
                f"coordinate_system (RAW or SPECIMEN): {exc}",
            ) from None
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

    def measure_shift(
        self,
        reference: str,
        image: str,
        method: str = "cross-correlation",
        compare: bool = False,
        agreement_px: float = 2.0,
    ) -> ImageShift:
        """The shift of the saved beam image *image* from *reference*, as an
        alignment to that reference measures it: metres and pixels, the score
        (higher is better) and whether it cleared the minimum. With *compare*,
        by every method too, how far apart they are in pixels and whether they
        agree within *agreement_px*: the check an alignment validates with.
        Reads the files and moves nothing."""
        try:
            chosen = AlignmentMethod(method)
        except ValueError:
            names = ", ".join(m.value for m in AlignmentMethod)
            raise ImageRefused("invalid", f"method is one of {names}") from None
        first, second = (
            _beam_image(self.image(reference), reference),
            _beam_image(self.image(image), image),
        )
        if first.data.shape != second.data.shape:
            raise ImageRefused(
                "invalid",
                f"{reference} is {first.data.shape} and {image} is "
                f"{second.data.shape}: a shift is measured between images of one size",
            )
        shift = _calculate_shift(first, second, chosen).to_dict()
        comparison = None
        if compare:
            comparison = compare_alignment_methods(
                first, second, agreement_threshold_px=agreement_px
            ).to_dict()
        return ImageShift(reference, image, shift, comparison)

    def coincidence_offset(
        self,
        sem_path: str,
        sem_point: Optional[List[float]],
        fib_path: str,
        fib_point: Optional[List[float]],
        sem_crop: Any = None,
        fib_crop: Any = None,
        pairs: Optional[List[Dict[str, Any]]] = None,
        tolerance_um: Optional[float] = None,
    ) -> ImageMeasured:
        """The vertical move that brings features picked in both the saved
        SEM and FIB images to coincidence
        (`fibsem.analysis.coincidence.coincidence_offset`): one at
        *sem_point* and *fib_point*, or *pairs* (``{sem, fib, label?}``).
        With a picture of every pair, labelled, and where each lands in the
        FIB, saved as ``drawn/<sem>-<fib>-coincidence.png`` beside the SEM
        image. Points read off a crop's picture name that crop (*sem_crop*,
        *fib_crop*: a cell's name or a box), as `pixel_to_stage` takes it."""
        if pairs is None:
            if sem_point is None or fib_point is None:
                raise ImageRefused("invalid", "Give sem_point and fib_point, or pairs.")
            pairs = [{"sem": sem_point, "fib": fib_point}]
        elif sem_point is not None or fib_point is not None:
            raise ImageRefused(
                "invalid", "Give sem_point and fib_point, or pairs, not both."
            )
        sem_target, fib_target = self.image(sem_path), self.image(fib_path)
        sem, fib = _beam_image(sem_target, sem_path), _beam_image(fib_target, fib_path)
        try:
            pairs = [
                {
                    **pair,
                    "sem": _uncropped_point(sem_target, sem_crop, pair["sem"], None)[0]
                    if sem_crop is not None
                    else pair["sem"],
                    "fib": _uncropped_point(fib_target, fib_crop, pair["fib"], None)[0]
                    if fib_crop is not None
                    else pair["fib"],
                }
                for pair in pairs
            ]
            result = coincidence.coincidence_offset(
                sem,
                fib,
                pairs,
                None if tolerance_um is None else float(tolerance_um) * 1e-6,
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ImageRefused("invalid", f"{type(exc).__name__}: {exc}") from None
        sem_marks, fib_marks = [], []
        for pair, (x, y) in ((p, p["sem_px"]) for p in result["pairs"]):
            sem_marks.append({"x": x, "y": y, "label": pair["label"]})
            fx, fy = pair["fib_px"]
            fib_marks.append({"x": fx, "y": fy, "label": pair["label"]})
            if "after_fib_px" in pair:
                ax, ay = pair["after_fib_px"]
                fib_marks.append(
                    {
                        "x": ax,
                        "y": ay,
                        "shape": "target",
                        "label": f"{pair['label']} after",
                    }
                )
        try:
            left = draw(
                sem_target,
                _checked_marks({"points": sem_marks}, sem_target),
                crosshair=True,
            )
            right = draw(
                fib_target,
                _checked_marks({"points": fib_marks}, fib_target),
                crosshair=True,
            )
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        saved = drawn_path(
            sem_target,
            f"{image_stem(sem_target)}-{image_stem(fib_target)}-coincidence",
        )
        return ImageMeasured(
            sem_path,
            "coincidence",
            {"sem_path": sem_path, "fib_path": fib_path, **result},
            saved,
            save_drawn(side_by_side(left.rgb, right.rgb), saved),
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
        points: Optional[int] = None,
    ) -> ImageMeasured:
        """A saved image's values along a line, and the FWHM of its peak (or
        *dip*), with a figure (`fibsem.analysis.measure.line_profile`). The
        answer leaves the per-pixel samples out; *points* adds the profile
        resampled to that many (`fibsem.analysis.measure.profile_summary`)."""
        line = "-".join(str(round(v)) for v in (*start, *end))
        answer = self._measured(
            path,
            lambda image: measure.line_profile(
                image, start, end, width_px, channel, plane, dip
            ),
            measure.profile_figure,
            f"profile-{line}",
        )
        try:
            answer.result = measure.profile_summary(answer.result, points)
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        return answer

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

    def histogram(
        self,
        path: str,
        channel: Optional[str] = None,
        plane: Optional[int] = None,
        bins: int = 64,
        box: Optional[List[float]] = None,
    ) -> ImageMeasured:
        """The counts of a saved image's values, its percentiles and what is
        clipped, with a figure (`fibsem.analysis.measure.histogram`)."""
        return self._measured(
            path,
            lambda image: measure.histogram(image, channel, plane, bins, box),
            measure.histogram_figure,
            _measure_name("histogram", plane, box),
        )

    def power_spectrum(
        self,
        path: str,
        channel: Optional[str] = None,
        plane: Optional[int] = None,
        box: Optional[List[float]] = None,
        points: Optional[int] = None,
    ) -> ImageMeasured:
        """A saved image's radially averaged power spectrum, the finest
        period with signal and its anisotropy, with a figure
        (`fibsem.analysis.measure.power_spectrum`). The answer leaves the
        radial average out; *points* adds it resampled to that many
        (`fibsem.analysis.measure.spectrum_summary`)."""
        answer = self._measured(
            path,
            lambda image: measure.power_spectrum(image, channel, plane, box),
            measure.spectrum_figure,
            _measure_name("spectrum", plane, box),
        )
        try:
            answer.result = measure.spectrum_summary(answer.result, points)
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        return answer

    def find_grid_squares(
        self,
        path: str,
        points: Optional[List[List[float]]] = None,
        limit: Optional[int] = None,
        box: Optional[List[float]] = None,
        pitch_um: Optional[float] = None,
    ) -> ImageMeasured:
        """The grid-bar lattice on a saved image (fitted in *box*, preferring
        *pitch_um* when given), its squares and *points* checked against the
        bars (`fibsem.analysis.lattice`), each with the stage position under
        it when the image says where it was taken; the figure saved as
        ``drawn/<image>-grid.png``."""
        target = self.image(path)
        try:
            planes = load_planes(target)
            fit = lattice.fit_lattice(planes, box, pitch_um)
            result = lattice.grid_answer(fit, planes.pixel_size, points, limit)
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        measured = result.get("candidate") or result
        placed = [
            p
            for p in measured.get("squares", []) + measured.get("points", [])
            if p.get("bar_distance_px", 0) is not None
        ]
        try:
            stages = image_points_to_stage(target, [p["px"] for p in placed])
        except Exception:  # noqa: BLE001 - a picture or a file with no geometry
            stages = [None] * len(placed)
        for p, stage in zip(placed, stages):
            p["stage_position"] = None if stage is None else stage.to_dict()
        saved = drawn_path(target, f"{image_stem(target)}-grid")
        plot = save_drawn(lattice.grid_figure(planes, fit, result), saved)
        return ImageMeasured(path, "grid", result, saved, plot)

    def measure_thickness(
        self,
        path: str,
        width_um: Optional[float] = None,
        height_um: Optional[float] = None,
        centre: Optional[List[float]] = None,
        *,
        poi: Optional[Point] = None,
        window: Optional[tuple] = None,
        window_from: str = "default",
    ) -> Dict[str, Any]:
        """A lamella's thickness read off the saved FIB image *path*: the
        bright band between the trenches, its width at half its peak column by
        column (`fibsem.imaging.thickness.measure_lamella_band`), in a window
        *width_um* along the lamella by *height_um* across it.

        The window is centred at *centre* ([x, y] in the image's pixels), else
        at *poi* (a point in the image's own frame, as an item keeps it), else
        at the image's centre. *window* (metres) and *window_from* are the
        size an item's trench patterns give, used where *width_um* and
        *height_um* are not given; with neither, DEFAULT_WINDOW. Nothing
        moves; the result is saved in a ``thickness`` folder beside the image,
        replaced when the same image is measured again."""
        target = self.image(path)
        if is_fluorescence_image(target):
            raise ImageRefused(
                "invalid",
                f"{path!r} is a fluorescence image; the band is read off a FIB image.",
            )
        _fibsem_file(target, path, "the thickness")
        try:
            image = FibsemImage.load(target)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is a data fact
            raise ImageRefused("invalid", f"{path!r}: {exc}") from None
        settings = getattr(image.metadata, "image_settings", None)
        if getattr(settings, "beam_type", None) is BeamType.ELECTRON:
            raise ImageRefused(
                "invalid",
                f"{path!r} is an SEM image; the band is read off a FIB image.",
            )
        default = window or DEFAULT_WINDOW
        if window is None:
            window_from = "default"
        size = (
            default[0] if width_um is None else width_um * 1e-6,
            default[1] if height_um is None else height_um * 1e-6,
        )
        if width_um is not None or height_um is not None:
            window_from = "given"
        if centre is not None:
            pixel_size = getattr(image.metadata, "pixel_size", None)
            if pixel_size is None:
                raise ImageRefused("invalid", f"{path!r} carries no pixel size.")
            poi, centre_from = (
                image_to_microscope_image_coordinates(
                    Point(*centre), image.data, pixel_size.x, subpixel_precision=True
                ),
                "given",
            )
        elif poi is not None:
            centre_from = "poi"
        else:
            poi, centre_from = Point(0, 0), "image_centre"
        try:
            result = measure_lamella_band(image, poi, size)
        except ValueError as exc:
            raise ImageRefused("invalid", str(exc)) from None
        # Kept beside the image, as an autofocus run keeps its own: one folder
        # per image, replaced when the same image is measured again.
        stem = os.path.splitext(os.path.basename(target))[0]
        try:
            saved_to = str(
                result.save(os.path.join(os.path.dirname(target), "thickness"), stem)
            )
        except OSError:
            logger.exception(f"Could not save the thickness result for {path!r}")
            saved_to = None
        # A dict as `Thickness` lays it out: to_dict carries more than the
        # model names (contrast, other_bands, ...), and routes answer as sent.
        return {
            "image": path,
            "poi": {"x": poi.x, "y": poi.y},
            "centre_from": centre_from,
            "window_um": [round(size[0] * 1e6, 3), round(size[1] * 1e6, 3)],
            "window_from": window_from,
            "saved_to": saved_to,
            **result.to_dict(),
            "plot": {"image_b64_jpeg": base64.b64encode(result.plot_jpeg()).decode()},
        }


def _measure_name(name: str, plane: Optional[int], box: Optional[List[float]]) -> str:
    """A measurement's figure name: *name*, then the plane and box it was on."""
    if plane is not None:
        name += f"-plane-{plane}"
    if box is not None:
        name += "-box-" + "-".join(str(round(v)) for v in box)
    return name


def _beam_image(target: str, path: str) -> FibsemImage:
    """The beam image at *target* (*path* as given), or the refusal."""
    if is_fluorescence_image(target):
        raise ImageRefused(
            "invalid", f"{path} is a fluorescence stack; a shift is between beam images"
        )
    _fibsem_file(target, path, "the shift")
    image = FibsemImage.load(target)
    if image.metadata is None or np.asarray(image.data).ndim != 2:
        raise ImageRefused(
            "invalid", f"{path} is not a saved beam image with its pixel size"
        )
    return image


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
