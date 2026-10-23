"""An image as an agent-sized preview payload, for a proposal's reference image,
and what the bar under an image says of it.

Accepts a FibsemImage or a bare array (a fluorescence composite). Everything
returned is plain data.
"""

import base64
from typing import Any, Dict, Optional

__all__ = ["image_payload", "image_info", "fm_info", "fields_info"]


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


def image_info(image, plane: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """`fields_info` of a FibsemImage or FluorescenceImage; None for a bare array.
    *plane* is the FM plane shown (0-based), None the max projection."""
    if getattr(image, "metadata", None) is None:
        return None
    from fibsem.fm.structures import FluorescenceImage
    from fibsem.imaging.export import image_fields

    if isinstance(image, FluorescenceImage):
        return fm_info(image.metadata, image.data.shape, plane)
    size = image.metadata.pixel_size
    return fields_info(
        image_fields(image), size.x if size is not None else None, image.data.shape[1]
    )


def fm_info(metadata, shape, plane: Optional[int] = None) -> Dict[str, Any]:
    """`fields_info` of a fluorescence stack from its metadata and (C, Z, Y, X)
    *shape*, so a file needn't be read whole; Z names *plane* when one is shown."""
    from fibsem.imaging.export import fluorescence_fields, stack_planes, z_value

    stack = stack_planes(metadata, shape)
    z = z_value(*stack, plane) if stack is not None and plane is not None else None
    return fields_info(
        fluorescence_fields(metadata, shape), metadata.pixel_size_x, shape[-1], z
    )


def fields_info(
    info, pixel_size: Optional[float], width: int, z: Optional[str] = None
) -> Dict[str, Any]:
    """What the bar under an image says, as the Qt view bar says it: *info*'s
    (an `ImageFields`) kind, its unlabelled header fields (detector, objective),
    its labelled fields and the time, as ``display.info_bar_fields`` picks them
    for its kind; *z* replaces the Z field's value. ``scalebar`` is ``{length,
    label, fraction}``: a round length in metres, about a fifth of an image *width*
    pixels of *pixel_size* metres, and the part of the width it spans; None when
    the size isn't known.
    """
    from fibsem.imaging.drawing import _format_length, _pick_scalebar_length_m
    from fibsem.imaging.export import (
        BAR_DEFAULT_FIELDS,
        bar_selection,
        field_title,
        stored_bar_keys,
    )

    keys = stored_bar_keys(info.kind) or BAR_DEFAULT_FIELDS
    header, labelled, time = bar_selection(info, keys)

    def entry(item) -> Dict[str, str]:
        value = z if item.key == "z" and z is not None else item.value
        return {
            "key": item.key,
            "label": item.label,
            "title": field_title(item),
            "value": value,
        }

    scalebar = None
    length = _pick_scalebar_length_m(pixel_size * width) if pixel_size else 0.0
    if length > 0:
        scalebar = {
            "length": length,
            "label": _format_length(length),
            "fraction": length / (pixel_size * width),
        }
    return {
        "kind": info.kind,
        "header": [entry(f) for f in header],
        "fields": [entry(f) for f in labelled],
        "time": time,
        "scalebar": scalebar,
    }
