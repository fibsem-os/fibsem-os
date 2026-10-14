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


TIFF_MEDIA_TYPE = "image/tiff"


def tiff_bytes(image) -> bytes:
    """A ``FibsemImage`` as TIFF with its metadata, as ``FibsemImage.load`` reads it."""
    import tifffile as tff

    buf = io.BytesIO()
    metadata = image.metadata.to_dict() if image.metadata is not None else None
    tff.imwrite(buf, image.data, metadata=metadata)
    return buf.getvalue()


# --- The answer an agent gets for a command that returns an image -------------


def _viewable(data: "np.ndarray") -> "np.ndarray":
    """A 2-D frame of *data* to preview: a stack (z or channels) as its maximum
    projection; an RGB frame as it is."""
    data = np.asarray(data)
    while data.ndim > 2 and not (data.ndim == 3 and data.shape[-1] in (3, 4)):
        data = data.max(axis=0)
    return data


def _array_preview(data: "np.ndarray", max_width: int = 768) -> dict:
    jpeg, width, height = preview_jpeg_bytes(_viewable(data), max_width=max_width)
    return {
        "image_b64_jpeg": base64.b64encode(jpeg).decode(),
        "width": width,
        "height": height,
    }


def image_answer(result) -> dict:
    """What a command that returned an image answers an agent: a summary, the
    saved path (None when the command saved nothing), and a small JPEG preview
    with its scale. The same shape for a beam's image, a camera frame, an array
    and a z-stack, so an agent reads them all one way.

    The full data is for programs, which ask for it (the TIFF or npy answer);
    the agent works from the preview and, when it needs the file, the path.
    """
    from fibsem.devices.wire import Frame
    from fibsem.structures import FibsemImage

    if is_image_list(result):  # reference images: each in its place, None kept
        return [None if item is None else image_answer(item) for item in result]
    if isinstance(result, FibsemImage):
        preview = preview_payload(result)
        summary = {
            "kind": "image",
            "shape": list(result.data.shape),
            "dtype": str(result.data.dtype),
            "beam_type": preview["beam_type"],
            "hfw": preview["hfw"],
            "pixelsize": preview["pixelsize"],
        }
        md = result.metadata
        acquired = getattr(md, "acquisition_datetime", None)
        if acquired:
            summary["acquired_at"] = str(acquired)
        return {"summary": summary, "path": result.filepath, "preview": preview}
    if (
        isinstance(result, list)
        and result
        and all(isinstance(item, Frame) for item in result)
    ):
        frames = [np.asarray(frame.data) for frame in result]
        shapes = {frame.shape for frame in frames}
        data = np.stack(frames) if len(shapes) == 1 else frames[0]
        return {
            "summary": {
                "kind": "frames",
                "count": len(result),
                "shape": list(frames[0].shape),
                "dtype": str(frames[0].dtype),
                "metadata": [frame.metadata for frame in result],
                "preview_of": "maximum projection"
                if len(shapes) == 1
                else "the first frame",
            },
            "path": None,
            "preview": _array_preview(data),
        }
    metadata = None
    if isinstance(result, Frame):
        metadata = result.metadata
        result = result.data
    data = np.asarray(result)
    summary = {"kind": "array", "shape": list(data.shape), "dtype": str(data.dtype)}
    if metadata is not None:
        summary.update(kind="frame", metadata=metadata)
    if data.ndim > 2 and _viewable(data).ndim != data.ndim:
        summary["preview_of"] = "maximum projection"
    return {"summary": summary, "path": None, "preview": _array_preview(data)}


def is_image_result(result) -> bool:
    """Whether a command's *result* is an image an agent should get as
    :func:`image_answer`: a FibsemImage, a frame, frames, or an array of 2 or
    more dimensions."""
    from fibsem.devices.wire import Frame
    from fibsem.structures import FibsemImage

    if isinstance(result, (FibsemImage, Frame)) or is_image_list(result):
        return True
    if isinstance(result, list) and result:
        return all(isinstance(item, Frame) for item in result)
    return isinstance(result, np.ndarray) and result.ndim >= 2


def is_image_list(result) -> bool:
    """Whether *result* is a list of FibsemImages (or lists of them), None in
    places, with at least one image: what the reference image commands return.
    No program format carries it, so it always answers as :func:`image_answer`."""
    from fibsem.structures import FibsemImage

    if not isinstance(result, (list, tuple)) or not result:
        return False
    found = False
    for item in result:
        if item is None:
            continue
        if isinstance(item, FibsemImage) or is_image_list(item):
            found = True
        else:
            return False
    return found
