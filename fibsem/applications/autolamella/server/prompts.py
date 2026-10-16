"""Serializing a pending supervision question for a remote agent.

The requests already obey FIB-826's rule — "a responder must be able to answer
from the request alone" — so serializing one *is* handing the agent everything
it needs: the question, the options, and the context images, rendered through
the same preview pipeline the acquisition endpoints use (agent-sized JPEG,
never raw arrays).

Everything returned is plain data. Unknown request types degrade to their type
name and JSON-able fields rather than failing: an agent that sees a question it
cannot fully parse should still know one is standing.
"""

import base64
from typing import Any, Dict, Optional

from fibsem.applications.autolamella.server.events import to_plain

__all__ = ["serialize_request"]


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
        # A preview failure must not hide the question itself.
        return None


def _acquired_at(image) -> Optional[str]:
    try:
        return image.metadata.acquisition_date.isoformat()
    except Exception:
        return None


def _preview_payload(image) -> Optional[Dict[str, Any]]:
    """A preview plus the scale facts needed to map coordinates onto it.

    Accepts a FibsemImage or a bare ndarray. For a FibsemImage this is the
    acquisition endpoints' own payload (``width``/``height`` of the downscaled
    rendition, ``full_width``/``full_height`` of the source, ``hfw``,
    ``beam_type``, ``pixelsize`` in metres per *source* pixel). Bare arrays
    carry no metadata, so they report only the preview and source dimensions.
    The mirrored-overlay incident (live FOV ≠ acquisition FOV) is why every
    image carries its own scale rather than letting the client assume one;
    failures return None rather than hiding the question.
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


def serialize_request(request) -> Dict[str, Any]:
    """A pending question as plain data: its type, what it says
    (``Request.describe``), and, when an answer may carry an edit, the shape of
    one as ``value``. An ``image`` becomes the standard preview payload; a
    question with no ``describe`` (not a ``Request``) still names its type."""
    payload: Dict[str, Any] = {"type": type(request).__name__}
    describe = getattr(request, "describe", None)
    if describe is None:
        return payload
    try:
        described = describe()
    except Exception:
        # A question that cannot describe itself must still be seen standing.
        return payload
    for key, value in described.items():
        if key == "image":
            payload[key] = _preview_payload(value)
        else:
            payload[key] = to_plain(value)
    shape = getattr(request, "value_shape", None)
    if shape is not None:
        payload["value"] = shape
    if getattr(request, "requires_value", False):
        payload["value_required"] = True
    return payload
