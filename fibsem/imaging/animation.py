"""An animation of what a workflow did: the images its tasks saved, one frame each.

Built on the single-image export (``fibsem.imaging.export``): each frame is an
``ExportImage`` drawn by ``render_export`` -- scalebar, metadata bar -- with the task's
name on a plate over it. Qt-free for the same reasons that module is: the dialog's
preview and the saved file come from the same functions, and a script or a report can
make one without the dialog.

Which images belong to which step is the caller's business. For AutoLamella that is
``task_outputs.lamella_task_steps``, which reads what each task run recorded rather
than guessing from filenames. This module only chooses among a step's images -- by beam
and by magnification -- and draws them.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from fibsem.imaging.drawing import _get_font
from fibsem.imaging.export import (
    BAR_COLOR,
    PLATE_ALPHA,
    PLATE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    ExportImage,
    adjust_contrast,
    auto_contrast_limits,
    default_options,
    load_export_image,
    render_export,
)

BEAMS = ("SEM", "FIB", "both")
MAGNIFICATIONS = ("high", "low")

# The format follows the file's extension. GIF is the default: it plays everywhere a
# figure goes (slides, chat, browsers, email), and 256 grey levels lose nothing for
# SEM/FIB. WebP keeps full colour and is far smaller, and Pillow writes it with nothing
# new installed -- but PowerPoint does not reliably play it, so it is the second.
ANIMATION_FORMATS = {".gif": "GIF", ".webp": "WEBP"}

# Lossy, at a quality where the noise in an SEM/FIB image is what is lost first.
_WEBP_QUALITY = 90


def webp_supported() -> bool:
    """Whether this Pillow can write animated WebP. Every current wheel can; a build
    without libwebp, or a Pillow before 11 without its animation support, cannot."""
    from PIL import features

    if not features.check("webp"):
        return False
    # Pillow 11 folded animation into "webp"; before that it was its own feature.
    if "webp_anim" in features.get_supported_features():
        return bool(features.check_feature("webp_anim"))
    return True


# Values on the bar of an animation frame: fewer than a single export's, because the
# frame is small and the values barely change from task to task.
_BAR_FIELDS = ("detector", "objective", "hfw", "voltage", "current")
# The bar's second row: the experiment, which is the same on every frame and too long
# for the plate, and the date. The lamella is on the plate instead, not both.
_PROVENANCE = ("experiment", "date")


@dataclass
class AnimationStep:
    """One step of the story: a title and the image files it may be drawn from."""

    title: str
    paths: List[str]


@dataclass
class AnimationFrame:
    """A step with its images read, once, so changing an option only re-draws."""

    title: str
    images: List[ExportImage]

    def pick(self, kind: str, magnification: str = "high") -> Optional[ExportImage]:
        """The step's ``kind`` image ("SEM" or "FIB") at the narrowest field of view
        ("high") or the widest ("low"), judged by the field width each image records
        -- not by its filename or its place in the list."""
        candidates = [i for i in self.images if i.kind == kind and i.pixel_size]
        if not candidates:
            return None

        def hfw(image: ExportImage) -> float:
            return float(image.pixel_size) * image.rgb.shape[1]

        return (
            min(candidates, key=hfw)
            if magnification == "high"
            else max(candidates, key=hfw)
        )


@dataclass
class AnimationOptions:
    beam: str = "FIB"  # one of BEAMS
    magnification: str = "high"  # one of MAGNIFICATIONS
    title: bool = True  # the task's name on a plate
    lamella: bool = True  # the lamella's name under it
    step_counter: bool = True  # "2 / 4" beside that
    scalebar: bool = True
    bar: bool = True  # acquisition values
    experiment: bool = True  # experiment and date, on the bar
    # Saved images vary in brightness from task to task; without this the
    # animation flickers, and flicker reads as change in the sample.
    auto_contrast: bool = True
    frame_ms: int = 1200
    hold_ms: int = 2500  # the last frame; 0 to hold it no longer than the others
    # Of each image: SEM and FIB side by side is twice this, so neither shrinks to
    # half a frame's worth of unreadable bar text.
    width: int = 768
    skipped: Sequence[int] = field(default_factory=tuple)  # frame indices left out


def load_frames(steps: Sequence[AnimationStep]) -> List[AnimationFrame]:
    """Read each step's images. A file that cannot be read is left out, and a step
    left with none is dropped -- one bad file should not lose the whole animation."""
    frames: List[AnimationFrame] = []
    for step in steps:
        images = []
        for path in step.paths:
            try:
                images.append(load_export_image(path))
            except Exception:
                logging.warning("Animation: could not read %s", path, exc_info=True)
        if images:
            frames.append(AnimationFrame(title=step.title, images=images))
    return frames


def _kinds(options: AnimationOptions) -> List[str]:
    return ["SEM", "FIB"] if options.beam == "both" else [options.beam]


def _draws(frame: AnimationFrame, options: AnimationOptions) -> bool:
    """Whether the frame has every image the options ask for."""
    return all(frame.pick(k, options.magnification) for k in _kinds(options))


def included(frames: Sequence[AnimationFrame], options: AnimationOptions) -> List[int]:
    """Indices of the frames that will be drawn: not skipped, and with the images the
    chosen beam needs. A step without a FIB image is left out of a FIB animation,
    rather than shown as a gap."""
    skipped = set(options.skipped)
    return [
        i
        for i, frame in enumerate(frames)
        if i not in skipped and _draws(frame, options)
    ]


def _panel(image: ExportImage, options: AnimationOptions, width: int) -> np.ndarray:
    o = default_options(image)
    keys = image.field_keys()
    o.fields = [k for k in _BAR_FIELDS if k in keys] if options.bar else []
    o.provenance = (
        [k for k in _PROVENANCE if k in image.provenance_keys()]
        if options.experiment
        else []
    )
    o.scalebar = options.scalebar
    if options.auto_contrast and image.adjustable:
        image = _stretched(image)
    rgb = render_export(image, o)
    if rgb.shape[1] != width:
        height = round(rgb.shape[0] * width / rgb.shape[1])
        rgb = np.array(Image.fromarray(rgb).resize((width, height), Image.LANCZOS))
    return rgb


def _stretched(image: ExportImage) -> ExportImage:
    """The image with its own 1st-99th percentile stretched to full range.

    Done here rather than through the export's contrast options: there, limits of
    0 and 1 mean "as acquired", which is what auto contrast comes to on a noisy image
    (its percentiles sit at the ends of its own range) -- and as-acquired frames
    saved at different brightness are the flicker this is meant to remove.
    """
    lo, hi = auto_contrast_limits(image)
    out = adjust_contrast(image.gray, lo, hi)
    gray = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return replace(image, rgb=np.stack([gray] * 3, axis=2), gray=None)


def _pad_to(rgb: np.ndarray, height: int) -> np.ndarray:
    """Extend with the bar's colour: frames whose bars wrapped differently must still
    be one size, as a GIF's frames are."""
    if rgb.shape[0] >= height:
        return rgb
    pad = np.empty((height - rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
    pad[:] = BAR_COLOR
    return np.concatenate([rgb, pad], axis=0)


def _draw_title(
    rgb: np.ndarray, title: str, subtitle: Optional[str], w: int
) -> np.ndarray:
    """The task's name over which lamella and how far along ("02-pro-moose · 2 / 4"),
    on a plate top-left: the plate the scalebar and legend use, so a frame pasted
    alone still says what it shows. Sized from ``w``, one image's width."""
    big = _get_font(max(13, round(w * 0.022)), bold=True)
    small = _get_font(max(11, round(w * 0.015)))
    margin = max(8, round(w * 0.012))
    pad = max(5, big.size // 2)
    lines = [(title, big, TEXT_STRONG_COLOR)]
    if subtitle:
        lines.append((subtitle, small, TEXT_MUTED_COLOR))
    text_w = max(math.ceil(font.getlength(text)) for text, font, _ in lines)
    line_h = [round(font.size * 1.25) for _, font, _ in lines]

    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = (margin, margin, margin + text_w + 2 * pad, margin + sum(line_h) + 2 * pad)
    draw.rectangle(box, fill=(*PLATE_COLOR, int(PLATE_ALPHA * 255)))
    y = margin + pad
    for (text, font, color), h in zip(lines, line_h):
        draw.text((margin + pad, y + h // 2), text, font=font, fill=color, anchor="lm")
        y += h
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def render_frame(
    frame: AnimationFrame,
    options: AnimationOptions,
    step: int = 1,
    total: int = 1,
    name: Optional[str] = None,
) -> np.ndarray:
    """One frame: one beam ``options.width`` wide, or SEM and FIB side by side.

    ``name`` is the lamella's, for the plate; without one, the name its images
    recorded (FIB-466), if they did.
    """
    kinds = _kinds(options)
    gap = 6 if len(kinds) > 1 else 0
    width = options.width
    panels = []
    images = []
    for kind in kinds:
        image = frame.pick(kind, options.magnification)
        if image is None:
            raise ValueError(f"{frame.title} has no {kind} image")
        images.append(image)
        panels.append(_panel(image, options, width))
    height = max(p.shape[0] for p in panels)
    panels = [_pad_to(p, height) for p in panels]
    if gap:
        divider = np.empty((height, gap, 3), dtype=np.uint8)
        divider[:] = BAR_COLOR
        rgb = np.concatenate([panels[0], divider, panels[1]], axis=1)
    else:
        rgb = panels[0]
    if options.title:
        if name is None:
            recorded = {f.key: f.value for f in images[0].provenance}
            name = recorded.get("item")
        parts = [
            name if options.lamella else None,
            f"{step} / {total}" if options.step_counter else None,
        ]
        subtitle = " · ".join(p for p in parts if p) or None
        rgb = _draw_title(rgb, frame.title, subtitle, width)
    return rgb


def render_animation(
    frames: Sequence[AnimationFrame],
    options: AnimationOptions,
    name: Optional[str] = None,
) -> List[np.ndarray]:
    """Every included frame, all one size. ``name``: the lamella's, for the plate."""
    indices = included(frames, options)
    rendered = [
        render_frame(frames[i], options, step=n + 1, total=len(indices), name=name)
        for n, i in enumerate(indices)
    ]
    if not rendered:
        return []
    height = max(r.shape[0] for r in rendered)
    return [_pad_to(r, height) for r in rendered]


def frame_durations(count: int, options: AnimationOptions) -> List[int]:
    """Milliseconds per frame: the last one held, so a looping GIF reads as an ending
    rather than a blur back to the start."""
    if count == 0:
        return []
    last = max(options.hold_ms, options.frame_ms)
    return [options.frame_ms] * (count - 1) + [last]


def save_animation(
    frames: Sequence[np.ndarray], path: str, options: AnimationOptions
) -> str:
    """Write rendered frames to ``path``; the format follows its extension."""
    path = os.fspath(path)
    extension = os.path.splitext(path)[1].lower()
    if extension not in ANIMATION_FORMATS:
        supported = ", ".join(sorted(ANIMATION_FORMATS))
        raise ValueError(f"Can't write {extension or 'that'}; use {supported}")
    if not frames:
        raise ValueError("There are no frames to save")
    fmt = ANIMATION_FORMATS[extension]
    if fmt == "WEBP" and not webp_supported():
        raise ValueError("This installation can't write animated WebP; use .gif")
    if fmt == "GIF":
        # Each frame gets its own 256-colour palette. Frames are greyscale but for
        # the bar's faint blue, so this costs nothing visible.
        images = [
            Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=256)
            for f in frames
        ]
        extra = {}
    else:
        images = [Image.fromarray(f) for f in frames]
        extra = {"quality": _WEBP_QUALITY, "method": 4}
    images[0].save(
        path,
        format=fmt,
        save_all=True,
        append_images=images[1:],
        duration=frame_durations(len(images), options),
        loop=0,
        **extra,
    )
    return path


def default_animation_name(title: str, directory: str, extension: str = ".gif") -> str:
    """`<title>_workflow.gif` (or the given extension) in ``directory``."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title).strip("_")
    return os.path.join(directory, f"{safe or 'workflow'}_workflow{extension}")
