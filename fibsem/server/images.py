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
import math
from typing import Any, Dict, Optional, Sequence, Tuple

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
    return array_preview_payload(image.data, image.metadata, max_width=max_width)


def array_preview_payload(
    data: "np.ndarray",
    metadata=None,
    max_width: int = 768,
    pixel_size: Optional[float] = None,
    full_shape: Optional[Sequence[int]] = None,
) -> dict:
    """:func:`preview_payload` for *data* (2-D, or RGB) and the beam image
    *metadata* it came with, if any: a region of an image, or one that is not a
    beam image (*pixel_size* in metres then gives its scale). *full_shape* is the
    source image's when *data* is a smaller copy of it."""
    jpeg, width, height = preview_jpeg_bytes(data, max_width=max_width)
    full_shape = data.shape if full_shape is None else full_shape
    payload = {
        "image_b64_jpeg": base64.b64encode(jpeg).decode(),
        "width": width,
        "height": height,
        "full_width": int(full_shape[1]),
        "full_height": int(full_shape[0]),
        "beam_type": None,
        "hfw": None,
        # metres per SOURCE pixel (full_width, not the downscaled preview):
        # every image payload states its own scale so an agent never maps
        # coordinates using an assumed field of view.
        "pixelsize": None,
    }
    md = metadata
    if md is not None and md.image_settings is not None:
        payload["hfw"] = _maybe_float(md.image_settings.hfw)
        beam_type = md.image_settings.beam_type
        payload["beam_type"] = beam_type.name if beam_type is not None else None
    md_pixel_size = getattr(md, "pixel_size", None)
    if md_pixel_size is not None:
        payload["pixelsize"] = {
            "x": _maybe_float(md_pixel_size.x),
            "y": _maybe_float(md_pixel_size.y),
        }
    elif pixel_size is not None:
        payload["pixelsize"] = {"x": float(pixel_size), "y": float(pixel_size)}
    size_x = (payload["pixelsize"] or {}).get("x")
    if size_x is not None and width:
        # metres per PREVIEW pixel: what a measurement on the preview uses.
        payload["preview_pixelsize"] = size_x * payload["full_width"] / width
    return payload


def _degrees(radians) -> Optional[float]:
    return round(math.degrees(float(radians)), 4) if radians is not None else None


def image_metadata(
    image, path: Optional[str] = None, shape: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    """What an agent needs to know about a beam image, in about 500 bytes: the
    same block on every image answer, saved outputs included.

    Where and how it was taken (stage position, scan rotation, beam, detector)
    so the agent can tell an image is stale after a move, and its scale.
    ``pixel_size`` is measured, so it is the one to trust; ``hfw`` is what was
    asked for (FIB-482). Angles are in degrees, named so in the keys; lengths
    in metres. The rest of the file's metadata is ``experiment_get_image_metadata``'s.
    *shape* is the image's when ``image.data`` is a smaller copy of it.
    """
    data = image.data
    block: Dict[str, Any] = {
        "beam_type": None,
        "acquired_at": None,
        "shape": list(data.shape if shape is None else shape),
        "dtype": str(data.dtype),
        "hfw": None,
        "pixel_size": None,
        "stage_position": None,
        "scan_rotation_deg": None,
        "reduced_area": None,
        "beam": None,
        "detector": None,
        "item": None,
        "task": None,
        "path": path if path is not None else getattr(image, "filepath", None),
    }
    md = image.metadata
    if md is None:
        return block
    settings = md.image_settings
    beam_type = getattr(settings, "beam_type", None)
    if beam_type is not None:
        block["beam_type"] = beam_type.name
    if settings is not None:
        block["hfw"] = _maybe_float(settings.hfw)
        if settings.reduced_area is not None:
            block["reduced_area"] = settings.reduced_area.to_dict()
    if md.acquisition_datetime is not None:
        block["acquired_at"] = md.acquisition_datetime.isoformat()
    if md.pixel_size is not None:
        block["pixel_size"] = _maybe_float(md.pixel_size.x)
    state = md.microscope_state
    if state is not None:
        position = state.stage_position
        if position is not None:
            block["stage_position"] = {
                "x": _maybe_float(position.x),
                "y": _maybe_float(position.y),
                "z": _maybe_float(position.z),
                "r_deg": _degrees(position.r),
                "t_deg": _degrees(position.t),
                "coordinate_system": position.coordinate_system,
            }
        # The state records the imaging beam and its detector; the other is empty.
        ion = beam_type is not None and beam_type.name == "ION"
        beam = state.ion_beam if ion else state.electron_beam
        detector = state.ion_detector if ion else state.electron_detector
        if beam is not None:
            block["scan_rotation_deg"] = _degrees(beam.scan_rotation)
            block["beam"] = {
                "voltage": _maybe_float(beam.voltage),
                "current": _maybe_float(beam.beam_current),
                "working_distance": _maybe_float(beam.working_distance),
                "dwell_time": _maybe_float(beam.dwell_time),
            }
        if detector is not None:
            block["detector"] = {"type": detector.type, "mode": detector.mode}
    experiment = md.experiment
    if experiment is not None:
        block["item"] = experiment.item_name
        block["task"] = experiment.task_name
    return block


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
        summary = {"kind": "image", **image_metadata(result)}
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
