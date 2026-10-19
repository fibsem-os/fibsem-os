"""An image as an agent-sized preview payload, for a proposal's reference image.

Accepts a FibsemImage or a bare array (a fluorescence composite). Everything
returned is plain data.
"""

import base64
from typing import Any, Dict, Optional

__all__ = ["image_payload"]


def _preview_b64(data) -> Optional[Dict[str, Any]]:
    """An ndarray as the standard agent-sized JPEG payload, or None."""
    if data is None:
        return None
    try:
        from fibsem.server.images import preview_jpeg_bytes

        jpeg, width, height = preview_jpeg_bytes(data)
        return {
            "image_b64_jpeg": base64.b64encode(jpeg).decode(),
            "width": width,
            "height": height,
        }
    except Exception:
        return None


def _acquired_at(image) -> Optional[str]:
    try:
        return image.metadata.acquisition_date.isoformat()
    except Exception:
        return None


def image_payload(image) -> Optional[Dict[str, Any]]:
    """A preview plus the scale facts needed to map coordinates onto it.

    Accepts a FibsemImage or a bare ndarray. For a FibsemImage this is the
    acquisition endpoints' own payload (``width``/``height`` of the downscaled
    rendition, ``full_width``/``full_height`` of the source, ``hfw``,
    ``beam_type``, ``pixelsize`` in metres per *source* pixel). Bare arrays
    carry no metadata, so they report only the preview and source dimensions.
    The mirrored-overlay incident (live FOV ≠ acquisition FOV) is why every
    image carries its own scale rather than letting the client assume one;
    failures return None.
    """
    if image is None:
        return None
    if hasattr(image, "metadata"):
        try:
            from fibsem.server.images import preview_payload

            payload = preview_payload(image)
        except Exception:
            return None
        # When it was taken, so an agent can tell a stale frame (FIB-1399).
        payload["acquired_at"] = _acquired_at(image)
        return payload
    entry = _preview_b64(image)
    if entry is None:
        return None
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        entry["full_height"] = int(shape[0])
        entry["full_width"] = int(shape[1])
    return entry
