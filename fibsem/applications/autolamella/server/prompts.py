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


def _image_data(image) -> Optional[Any]:
    return getattr(image, "data", None) if image is not None else None


def serialize_request(request) -> Dict[str, Any]:
    from fibsem.applications.autolamella.workflows.interaction import (
        Confirm,
        ConfirmDetection,
        EditAlignmentArea,
        PickPOI,
        RunMillingTask,
    )

    payload: Dict[str, Any] = {"type": type(request).__name__}

    if isinstance(request, Confirm):
        payload.update(
            {
                "message": request.message,
                "positive": request.positive,
                "negative": request.negative,
            }
        )
        return payload

    if isinstance(request, ConfirmDetection):
        detection = request.detection
        payload.update(
            {
                "message": "Confirm the detected features (answering yes accepts "
                "the positions as currently shown).",
                "features": [
                    {"name": f.name, "px": to_plain(f.px.to_dict())}
                    for f in getattr(detection, "features", [])
                ],
                "pixelsize": to_plain(getattr(detection, "pixelsize", None)),
                "image": _preview_b64(getattr(detection, "image", None)),
            }
        )
        return payload

    if isinstance(request, EditAlignmentArea):
        payload.update(
            {
                "message": "Confirm the alignment area (answering yes accepts "
                "the area as currently shown).",
                "initial": to_plain(request.initial.to_dict())
                if request.initial is not None
                else None,
            }
        )
        return payload

    if isinstance(request, PickPOI):
        payload.update(
            {
                "message": "Pick the point of interest on the image.",
                "initial": to_plain(request.initial.to_dict())
                if request.initial is not None
                else None,
                "image": _preview_b64(_image_data(request.image)),
            }
        )
        return payload

    if isinstance(request, RunMillingTask):
        config = request.config
        payload.update(
            {
                "message": "Confirm and run this milling task (answering yes "
                "runs the mill with the configuration as currently shown).",
                "enabled": request.enabled,
                "task_name": to_plain(getattr(config, "name", None)),
                "num_stages": len(getattr(config, "stages", []) or []),
            }
        )
        return payload

    # Unknown/newer request types: name + whatever serializes cleanly.
    try:
        import dataclasses

        if dataclasses.is_dataclass(request):
            for f in dataclasses.fields(request):
                value = getattr(request, f.name)
                if hasattr(value, "data") or type(value).__name__ == "ndarray":
                    continue  # images only travel as previews, never raw
                payload[f.name] = to_plain(value)
    except Exception:
        pass
    return payload
