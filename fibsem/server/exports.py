"""Exports for people, as the Qt dialogs make them: an image with its scalebar,
legend and metadata bar (`fibsem.imaging.export`), and a lamella's tasks as an
animation (`fibsem.imaging.animation`). Each answers the file's bytes and the
name the dialog would save it under.

Options are the dialogs' (`ExportOptions`, `AnimationOptions`): a client sends
the ones it changes, the rest are the dialog's defaults. A refusal is a
:class:`~fibsem.server.files.FileRefusal`, as the files routes refuse.
"""

import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fibsem.server.files import FileRefusal, resolve

IMAGE_FORMATS = {"png": ".png", "tiff": ".tif"}
ANIMATION_FORMATS = {"gif": ".gif", "webp": ".webp"}
MEDIA_TYPES = {
    ".png": "image/png",
    ".tif": "image/tiff",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _invalid(message: str) -> FileRefusal:
    return FileRefusal(422, "invalid", message)


def _decoded(cls, defaults, options: Optional[Dict[str, Any]]):
    """*defaults* with *options* over them, read as *cls*, or the 422 naming
    what didn't read."""
    from fibsem.codec import decode

    if options is None:
        return defaults
    if not isinstance(options, dict):
        raise _invalid("options is an object of the export's options.")
    try:
        return decode(cls, {**asdict(defaults), **options}, strict=True)
    except ValueError as exc:
        raise _invalid(f"options did not read: {exc}") from exc


def _one_of(name: str, value: Any, allowed: Sequence[Any]) -> None:
    if value not in allowed:
        raise _invalid(f"{name} is one of {', '.join(map(str, allowed))}, not {value}.")


def _file(rgb_or_frames, name: str, save) -> bytes:
    """What *save* writes to *name*, as bytes: the dialogs' writers take a path."""
    with tempfile.TemporaryDirectory() as folder:
        path = save(rgb_or_frames, os.path.join(folder, name))
        with open(path, "rb") as f:
            return f.read()


# --- one image -----------------------------------------------------------------


def _export_image(root: Optional[str], path: str):
    """The image at *path* in the experiment, read for export."""
    from fibsem.imaging.export import load_export_image

    target = resolve(root, path)
    if os.path.isdir(target) or not target.lower().endswith((".tif", ".tiff")):
        raise _invalid(
            f"{path} is not a beam image (.tif) or a fluorescence stack (.ome.tiff)."
        )
    try:
        return load_export_image(target)
    except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
        raise FileRefusal(
            422, "unreadable", f"Could not read {path} as an image: {exc}"
        ) from exc


def _field_rows(fields) -> List[Dict[str, str]]:
    from fibsem.imaging.export import field_title

    return [
        {
            "key": f.key,
            "name": f.name,
            "label": f.label,
            "value": f.value,
            "title": field_title(f),
        }
        for f in fields
    ]


def image_form(root: Optional[str], path: str) -> Dict[str, Any]:
    """What the export dialog offers for the image at *path*: its options as a
    new export starts, every value its bar can show, and its channels."""
    from fibsem.imaging.export import (
        DEFAULT_PROVENANCE,
        SCALEBAR_LOCATIONS,
        auto_contrast_limits,
        default_options,
    )

    image = _export_image(root, path)
    provenance = image.provenance_keys()
    return {
        "path": path,
        "kind": image.kind,
        "width": int(image.rgb.shape[1]),
        "height": int(image.rgb.shape[0]),
        "pixel_size": image.pixel_size,
        # contrast and gamma apply to a greyscale image only
        "adjustable": image.adjustable,
        "auto_contrast": list(auto_contrast_limits(image)),
        "options": asdict(default_options(image)),
        # what the provenance row ticks when it is turned on
        "default_provenance": [k for k in DEFAULT_PROVENANCE if k in provenance],
        "fields": _field_rows(image.fields),
        "provenance": _field_rows(image.provenance),
        "channels": [
            {"name": c.name, "color": "#{:02x}{:02x}{:02x}".format(*c.color)}
            for c in image.channels
        ],
        "scalebar_locations": list(SCALEBAR_LOCATIONS),
        "formats": list(IMAGE_FORMATS),
    }


def image_file(
    root: Optional[str],
    path: str,
    options: Optional[Dict[str, Any]] = None,
    format: str = "png",
) -> Tuple[bytes, str]:
    """The image at *path* exported under *options*: its bytes and its name,
    ``<image>_export.png`` as the dialog names it."""
    from fibsem.imaging.export import (
        SCALEBAR_LOCATIONS,
        ExportOptions,
        default_export_name,
        default_options,
        render_export,
        save_export,
    )

    _one_of("format", format, list(IMAGE_FORMATS))
    image = _export_image(root, path)
    chosen = _decoded(ExportOptions, default_options(image), options)
    _one_of("scalebar_location", chosen.scalebar_location, SCALEBAR_LOCATIONS)
    _one_of("scale", chosen.scale, (1, 2))
    if not 0.0 <= chosen.contrast_min < chosen.contrast_max <= 1.0:
        raise _invalid("contrast_min and contrast_max are 0 to 1, min below max.")
    if chosen.gamma <= 0:
        raise _invalid("gamma is above 0.")
    name = os.path.basename(default_export_name(image, IMAGE_FORMATS[format]))
    return _file(render_export(image, chosen), name, save_export), name


# --- an item's tasks, animated -------------------------------------------------


def _relative(root: str, path: str) -> str:
    return os.path.relpath(path, os.path.realpath(root)).replace(os.sep, "/")


def animation_frames(root: str, item_name: str, steps) -> Dict[str, Any]:
    """The frames *steps* make, a row per frame a page can tick (by its index,
    ``options.skipped``), with the beams each can draw."""
    from fibsem.imaging.animation import (
        BEAMS,
        MAGNIFICATIONS,
        AnimationOptions,
        load_frames,
        webp_supported,
    )

    frames = load_frames(steps)
    return {
        "item_name": item_name,
        "frames": [
            {
                "index": i,
                "title": frame.title,
                "beams": [k for k in ("SEM", "FIB") if frame.pick(k)]
                + (["FM"] if frame.fluorescence_paths else []),
                "paths": [_relative(root, image.path) for image in frame.images],
                "fluorescence": [_relative(root, p) for p in frame.fluorescence_paths],
            }
            for i, frame in enumerate(frames)
        ],
        "options": asdict(AnimationOptions()),
        "beams": list(BEAMS),
        "magnifications": list(MAGNIFICATIONS),
        "formats": [f for f in ANIMATION_FORMATS if f != "webp" or webp_supported()],
    }


def animation_file(
    item_name: str,
    steps,
    options: Optional[Dict[str, Any]] = None,
    format: str = "gif",
) -> Tuple[bytes, str]:
    """*steps* animated under *options*: the bytes and the name the dialog
    saves it under, ``<item>_workflow.gif``."""
    from fibsem.imaging.animation import (
        BEAMS,
        MAGNIFICATIONS,
        AnimationOptions,
        default_animation_name,
        load_frames,
        render_animation,
        save_animation,
        webp_supported,
    )

    _one_of("format", format, list(ANIMATION_FORMATS))
    if format == "webp" and not webp_supported():
        raise _invalid("This installation can't write animated WebP; use gif.")
    chosen = _decoded(AnimationOptions, AnimationOptions(), options)
    _one_of("beam", chosen.beam, BEAMS)
    _one_of("magnification", chosen.magnification, MAGNIFICATIONS)
    if chosen.frame_ms <= 0 or chosen.hold_ms < 0 or not 64 <= chosen.width <= 4096:
        raise _invalid("frame_ms is above 0, hold_ms 0 or more, width 64 to 4096.")
    rendered = render_animation(load_frames(steps), chosen, name=item_name)
    if not rendered:
        raise FileRefusal(
            404,
            "not_found",
            f"No frame of {item_name} has a {chosen.beam} image to animate.",
        )
    name = os.path.basename(
        default_animation_name(item_name, "", ANIMATION_FORMATS[format])
    )
    return (
        _file(rendered, name, lambda frames, p: save_animation(frames, p, chosen)),
        name,
    )
