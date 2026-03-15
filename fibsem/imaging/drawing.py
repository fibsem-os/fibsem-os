"""Reusable image overlay drawing functions (scalebar, crosshair).

All functions operate on numpy arrays and return modified copies.
No Qt, napari, or matplotlib dependencies. Uses PIL for drawing.
"""

from __future__ import annotations

import functools
import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fibsem.constants import MICRON_SYMBOL

# Round "nice" numbers for scalebar: 1, 2, 5, 10, 20, 50, ...
_NICE_NUMBERS = [1, 2, 5]


@functools.lru_cache(maxsize=8)
def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, falling back to PIL default. Cached."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _pick_scalebar_length_m(fov_m: float, target_ratio: float = 0.2) -> float:
    """Pick a round scalebar length that is approximately target_ratio of the FOV.

    Returns the length in metres.
    """
    target_m = fov_m * target_ratio
    if target_m <= 0:
        return 0.0

    exponent = math.floor(math.log10(target_m))
    base = 10 ** exponent

    best = _NICE_NUMBERS[0] * base
    best_diff = abs(target_m - best)
    for n in _NICE_NUMBERS[1:]:
        candidate = n * base
        diff = abs(target_m - candidate)
        if diff < best_diff:
            best = candidate
            best_diff = diff

    return best


def _format_length(length_m: float) -> str:
    """Format a length in metres as a string with SI prefix, no decimals.

    Examples: "10 μm", "200 nm", "1 mm"
    """
    if length_m == 0:
        return "0 m"

    # (upper_bound_exclusive, multiplier, unit)
    # Each range covers values strictly below its upper bound.
    si_ranges = [
        (1e-9, 1e12, "pm"),
        (1e-6, 1e9, "nm"),
        (1e-3, 1e6, f"{MICRON_SYMBOL}"), # default font on windows doesnt support greek characters
        (1.0, 1e3, "mm"),
        (1e3, 1.0, "m"),
        (1e6, 1e-3, "km"),
    ]

    for upper_bound, multiplier, unit in si_ranges:
        if length_m < upper_bound:
            val = length_m * multiplier
            return f"{val:.0f} {unit}"

    return f"{length_m:.0f} m"


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert grayscale to RGB if needed."""
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=2)
    return arr


def draw_scalebar(
    arr: np.ndarray,
    pixel_size_x: float,
    location: str = "lower right",
    bar_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.5,
    margin: int = 10,
    bar_height: int = 6,
    font_scale: float = 0.4,
) -> np.ndarray:
    """Draw a scalebar overlay on an image array.

    Args:
        arr: Image array (grayscale or RGB, uint8).
        pixel_size_x: Size of one pixel in metres.
        location: Scalebar position ("lower right", "lower left").
        bar_color: RGB color of the scale bar.
        text_color: RGB color of the label text.
        bg_color: RGB color of the background rectangle.
        bg_alpha: Opacity of the background rectangle (0-1).
        margin: Pixel margin from image edges.
        bar_height: Height of the scale bar in pixels.
        font_scale: Font scale factor (0.4 ≈ 12px).

    Returns:
        Modified copy of the array with scalebar drawn.
    """
    rgb = _ensure_rgb(arr)
    h, w = rgb.shape[:2]

    fov_m = pixel_size_x * w
    bar_length_m = _pick_scalebar_length_m(fov_m)
    if bar_length_m <= 0:
        return rgb.copy()

    bar_length_px = int(bar_length_m / pixel_size_x)
    bar_length_px = min(bar_length_px, w - 2 * margin)

    label = _format_length(bar_length_m)

    # Font and text measurement
    font_size = max(10, int(font_scale * 30))
    font = _get_font(font_size)
    bbox = font.getbbox(label)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Background rectangle dimensions
    pad = 6
    bg_w = max(bar_length_px, tw) + 2 * pad
    bg_h = bar_height + th + 3 * pad

    # Position
    if "left" in location:
        bg_x = margin
    else:
        bg_x = w - margin - bg_w
    bg_y = h - margin - bg_h

    # Single RGBA overlay for all scalebar elements
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Semi-transparent background
    draw.rectangle(
        (bg_x, bg_y, bg_x + bg_w, bg_y + bg_h),
        fill=(*bg_color, int(bg_alpha * 255)),
    )

    # Bar (fully opaque)
    bar_x = bg_x + (bg_w - bar_length_px) // 2
    bar_y_top = bg_y + pad
    draw.rectangle(
        (bar_x, bar_y_top, bar_x + bar_length_px, bar_y_top + bar_height),
        fill=(*bar_color, 255),
    )

    # Label centred below bar (fully opaque)
    text_x = bg_x + (bg_w - tw) // 2
    text_y = bar_y_top + bar_height + pad
    draw.text((text_x, text_y), label, font=font, fill=(*text_color, 255))

    result = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(result)


def draw_crosshair(
    arr: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 0),
    alpha: float = 0.3,
    size_ratio: float = 0.05,
    thickness: int = 1,
) -> np.ndarray:
    """Draw a crosshair at the centre of the image.

    Args:
        arr: Image array (grayscale or RGB, uint8).
        color: RGB color of the crosshair lines.
        alpha: Opacity of the crosshair (0-1).
        size_ratio: Length of each arm as a fraction of image width.
        thickness: Line thickness in pixels.

    Returns:
        Modified copy of the array with crosshair drawn.
    """
    rgb = _ensure_rgb(arr)
    h, w = rgb.shape[:2]

    cx, cy = w // 2, h // 2
    arm = int(w * size_ratio)

    # Draw lines on RGBA overlay for alpha blending
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(alpha * 255)
    line_color = (*color, a)
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=line_color, width=thickness)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=line_color, width=thickness)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(result)


def draw_image_overlays(
    arr: np.ndarray,
    pixel_size_x: float,
    show_scalebar: bool = True,
    show_crosshair: bool = True,
    **kwargs,
) -> np.ndarray:
    """Draw scalebar and/or crosshair overlays on an image.

    Args:
        arr: Image array (grayscale or RGB, uint8).
        pixel_size_x: Size of one pixel in metres.
        show_scalebar: Whether to draw a scalebar.
        show_crosshair: Whether to draw a centre crosshair.
        **kwargs: Passed to draw_scalebar / draw_crosshair.

    Returns:
        Modified copy of the array with overlays drawn.
    """
    out = arr
    if show_crosshair:
        crosshair_kwargs = {
            k: kwargs[k] for k in ("color", "alpha", "size_ratio", "thickness")
            if k in kwargs
        }
        out = draw_crosshair(out, **crosshair_kwargs)
    if show_scalebar:
        scalebar_kwargs = {
            k: kwargs[k] for k in (
                "location", "bar_color", "text_color", "bg_color",
                "bg_alpha", "margin", "bar_height", "font_scale",
            ) if k in kwargs
        }
        out = draw_scalebar(out, pixel_size_x, **scalebar_kwargs)
    return out
