"""Preview renditions of microscope images for agent and browser clients.

Full-resolution 16-bit TIFF is the right transport for a drop-in client; an
agent inspecting a milling result needs a small 8-bit JPEG it can actually
look at. Harvested from the feat-mcp-server prototype, which proved that a
768 px JPEG is enough for Claude to assess milling quality.

`PIL.Resampling.LANCZOS` requires Pillow >= 9.1; Pillow arrives transitively
with the core matplotlib dependency, so it is imported lazily and reported
clearly if absent.
"""

import base64
import io
from typing import Optional, Tuple

import numpy as np


def preview_jpeg_bytes(
    data: "np.ndarray", max_width: int = 768, quality: int = 70
) -> Tuple[bytes, int, int]:
    """Min-max normalize to uint8, downscale to max_width, encode JPEG.

    Returns (jpeg_bytes, width, height) of the encoded rendition.
    """
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required for image previews (pip install pillow)."
        ) from e

    if data.dtype != np.uint8:
        d_min, d_max = float(data.min()), float(data.max())
        if d_max > d_min:
            data = ((data.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(
                np.uint8
            )
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    # No explicit mode: fromarray infers L for 2-D uint8 and RGB for 3-D, and
    # the mode parameter is deprecated in Pillow 11.3 / removed in Pillow 13 —
    # under filterwarnings=error the deprecation made every preview fail.
    pil = PILImage.fromarray(data)
    if pil.width > max_width:
        scale = max_width / pil.width
        pil = pil.resize(
            (max_width, max(1, int(pil.height * scale))), PILImage.Resampling.LANCZOS
        )
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), pil.width, pil.height


def preview_payload(image, max_width: int = 768) -> dict:
    """Build the JSON payload for a preview endpoint from a FibsemImage."""
    jpeg, width, height = preview_jpeg_bytes(image.data, max_width=max_width)
    payload = {
        "image_b64_jpeg": base64.b64encode(jpeg).decode(),
        "width": width,
        "height": height,
        "full_width": int(image.data.shape[1]),
        "full_height": int(image.data.shape[0]),
        "beam_type": None,
        "hfw": None,
        # metres per SOURCE pixel (full_width, not the downscaled preview):
        # every image payload states its own scale so an agent never maps
        # coordinates using an assumed field of view.
        "pixelsize": None,
    }
    md = image.metadata
    if md is not None and md.image_settings is not None:
        payload["hfw"] = _maybe_float(md.image_settings.hfw)
        beam_type = md.image_settings.beam_type
        payload["beam_type"] = beam_type.name if beam_type is not None else None
    pixel_size = getattr(md, "pixel_size", None)
    if pixel_size is not None:
        payload["pixelsize"] = {
            "x": _maybe_float(pixel_size.x),
            "y": _maybe_float(pixel_size.y),
        }
    return payload


def _maybe_float(value) -> Optional[float]:
    return float(value) if value is not None else None
