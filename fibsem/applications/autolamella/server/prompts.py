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
from typing import Any, Dict, List, Optional

from fibsem.applications.autolamella.server.events import to_plain

__all__ = [
    "apply_stage_points",
    "milling_preview",
    "milling_stages",
    "serialize_request",
]


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

            return preview_payload(image)
        except Exception:
            return None
    entry = _preview_b64(image)
    if entry is None:
        return None
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        entry["full_height"] = int(shape[0])
        entry["full_width"] = int(shape[1])
    return entry


def milling_stages(config) -> List[Dict[str, Any]]:
    """Each stage of a milling config as plain geometry, in the config's order.

    The index is the one an answer's ``stages`` edits name. Disabled stages are
    listed too (``enabled: false``): they are in the editor, they just don't
    run, and leaving them out would shift every index after them.
    """
    stages = []
    for index, stage in enumerate(getattr(config, "stages", None) or []):
        entry: Dict[str, Any] = {
            "index": index,
            "name": stage.name,
            "enabled": bool(stage.enabled),
        }
        try:
            entry["pattern"] = to_plain(stage.pattern.to_dict())
        except Exception:
            entry["pattern"] = None
        entry["milling_current"] = to_plain(stage.milling.milling_current)
        entry["preset"] = to_plain(stage.milling.preset)
        stages.append(entry)
    return stages


def milling_preview(image, config) -> Optional[Dict[str, Any]]:
    """The FIB frame with the config's enabled patterns drawn on it, as the
    milling widget shows them, in the standard preview payload; or None.

    Drawn with ``draw_milling_patterns`` on an Agg canvas, never pyplot: this
    runs on a server thread, and pyplot's Qt backend would build a window
    there. The axes fill the figure so the preview stays the frame, and the
    scale facts (``full_width``, ``hfw``, ``pixelsize``) are the frame's own.
    """
    if image is None or getattr(image, "metadata", None) is None or config is None:
        return None
    try:
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from fibsem.milling.patterning.plotting import draw_milling_patterns
        from fibsem.server.images import preview_jpeg_bytes, preview_payload
        from fibsem.theme.plots import DARK, plot_theme

        height, width = image.data.shape[:2]
        with plot_theme(DARK):
            fig = Figure(figsize=(width / 100, height / 100), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, 1))
            draw_milling_patterns(
                image,
                [stage for stage in config.stages if stage.enabled],
                title="",
                show_current=True,
                show_depth=True,
                ax=ax,
            )
            ax.set_axis_off()
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()

        # The frame's own scale facts, with the drawn rendition in its place.
        entry = preview_payload(image)
        jpeg, entry["width"], entry["height"] = preview_jpeg_bytes(rgb)
        entry["image_b64_jpeg"] = base64.b64encode(jpeg).decode()
        return entry
    except Exception:
        # A preview failure must not hide the question itself.
        return None


def apply_stage_points(config, points: Dict[int, Any], image):
    """Move the named stages' patterns to ``points``; return the moved copy.

    ``points`` maps a stage index to a :class:`Point` in microscope image
    coordinates. Each pattern moves by the difference to its own point, the
    way the milling widget's Move moves it (a line's ends move with it), and
    must stay inside ``image``, the frame the widget draws on -- the same
    check the widget's own Move makes. Raises ValueError naming the stage
    otherwise; ``config`` itself is never changed.
    """
    from copy import deepcopy

    from fibsem.milling.patterning.patterns2 import LinePattern
    from fibsem.milling.patterning.shapes import is_pattern_placement_valid

    moved = deepcopy(config)
    stages = list(getattr(moved, "stages", None) or [])
    for index, point in points.items():
        if not 0 <= index < len(stages):
            raise ValueError(
                f"stage {index} does not exist: this task has stages "
                f"0..{len(stages) - 1}."
            )
        pattern = stages[index].pattern
        diff = point - pattern.point
        # As the widget's _apply_diff_to_pattern, which lives with Qt.
        pattern.point = pattern.point + diff
        if isinstance(pattern, LinePattern):
            pattern.start_x += diff.x
            pattern.start_y += diff.y
            pattern.end_x += diff.x
            pattern.end_y += diff.y
        if image is None or getattr(image, "metadata", None) is None:
            raise ValueError(
                "there is no FIB frame to check the patterns against; "
                "the patterns cannot be moved."
            )
        if not is_pattern_placement_valid(pattern, image):
            raise ValueError(
                f"stage {index} ({stages[index].name!r}) at ({point.x:.3e}, "
                f"{point.y:.3e}) m would be outside the FIB image."
            )
    return moved


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
                "coordinates": "feature px are source-image pixels (origin "
                "top-left, +y down); pixelsize is metres per source pixel",
                "image": _preview_payload(getattr(detection, "image", None)),
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
                "coordinates": "areas are fractions of the frame (origin "
                "top-left), so they need no rescaling against the preview",
                # The frame is display state the request doesn't carry;
                # AutoLamellaAppContext.pending_prompt fills it from the FIB display
                # cache — the same source /app/images serves.
                "image": None,
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
                "coordinates": "points are microscope image coordinates: "
                "metres, origin at the image centre, +x right, +y UP",
                "image": _preview_payload(request.image),
            }
        )
        return payload

    if isinstance(request, RunMillingTask):
        config = request.config
        payload.update(
            {
                "message": "Confirm and run this milling task (answering yes "
                "runs the mill with the patterns as currently shown).",
                "enabled": request.enabled,
                "task_name": to_plain(getattr(config, "name", None)),
                "num_stages": len(getattr(config, "stages", []) or []),
                "stages": milling_stages(config),
                "coordinates": "pattern points are microscope image "
                "coordinates: metres, origin at the image centre, +x right, "
                "+y UP",
                # The patterns are drawn on the FIB frame the milling widget
                # shows, which is display state the request doesn't carry:
                # AutoLamellaAppContext.pending_prompt draws it there.
                "image": None,
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
