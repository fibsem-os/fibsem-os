"""The open experiment's files, read for a client that cannot touch the disk.

One folder listed at a time; a text file (yaml, json, log, md, csv) as text; an
image as a preview with its metadata, the shape every image answer has
(:func:`fibsem.server.images.image_answer`), or one region of it at native
resolution. Previews and metadata only: no file's raw bytes (Patrick,
2026-10-14).

Every path resolves inside the experiment folder, symlinks followed, so ``..``,
an absolute path elsewhere and a link out of the folder are refused alike
(:class:`FileRefusal`, 403 ``not_allowed``).
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

TEXT_SUFFIXES = (".yaml", ".yml", ".json", ".jsonl", ".log", ".md", ".csv", ".txt")
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
# A text answer's cap: a session log runs to megabytes, and app_get_log reads it
# by window.
TEXT_LIMIT = 512 * 1024
# The widest preview served: an image enlarged to fill a screen.
MAX_PREVIEW_WIDTH = 3072


class FileRefusal(Exception):
    """Why a file is not served: an HTTP status, an error type and a message."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message

    @property
    def detail(self) -> Dict[str, str]:
        return {"error_type": self.error_type, "message": self.message}


def resolve(root: Optional[str], path: str) -> str:
    """*path* (relative to *root*, or absolute) as a real path inside *root*,
    or the :class:`FileRefusal` saying why not."""
    if root is None:
        raise FileRefusal(
            409, "no_experiment", "No experiment is open: load or create one first."
        )
    root = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root, path or ""))
    if os.path.commonpath([root, target]) != root:
        raise FileRefusal(
            403,
            "not_allowed",
            f"Only files inside the open experiment ({root}), not {path}.",
        )
    if not os.path.exists(target):
        raise FileRefusal(404, "not_found", f"No file or folder at {path}.")
    return target


def kind(path: str) -> str:
    """folder, text, image or other: how :func:`read` answers *path*."""
    if os.path.isdir(path):
        return "folder"
    name = path.lower()
    if name.endswith(TEXT_SUFFIXES):
        return "text"
    if name.endswith(IMAGE_SUFFIXES):
        return "image"
    return "other"


def _images_under(folder: str) -> int:
    """How many images *folder* holds, at any depth (links not followed)."""
    return sum(
        name.lower().endswith(IMAGE_SUFFIXES)
        for _, _, names in os.walk(folder)
        for name in names
    )


def _entry(root: str, path: str) -> Dict[str, Any]:
    stat = os.stat(path)
    entry_kind = kind(path)
    entry = {
        "name": os.path.basename(path),
        "path": os.path.relpath(path, root).replace(os.sep, "/"),
        "kind": entry_kind,
        "size": None if entry_kind == "folder" else stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if entry_kind == "folder":
        # where the images are, so a reader goes down the right folders
        entry["images"] = _images_under(path)
    return entry


def list_folder(root: Optional[str], path: str = "") -> Dict[str, Any]:
    """One folder of the experiment: its folders first, then its files, by name."""
    target = resolve(root, path)
    if not os.path.isdir(target):
        raise FileRefusal(
            422, "invalid", f"{path} is a file: read it, or list its folder."
        )
    real_root = os.path.realpath(root)
    names = sorted(os.listdir(target))
    paths = [os.path.join(target, name) for name in names]
    paths = [p for p in paths if os.path.isdir(p)] + [
        p for p in paths if not os.path.isdir(p)
    ]
    return {
        "root": real_root,
        "dir": os.path.relpath(target, real_root).replace(os.sep, "/"),
        "entries": [_entry(real_root, p) for p in paths],
    }


def load_image(path: str) -> Tuple[np.ndarray, Any, Optional[float]]:
    """A saved image as something viewable: its data (2-D, or RGB), the
    FibsemImage it is when it is a beam image (else None), and its pixel size in
    metres when known.

    Fluorescence stacks (``.ome.tiff``) composite through the FM canvas's own
    projection, so each channel keeps its colour; TIFFs read as FibsemImages;
    anything else (the PNG thumbnails grid tasks write beside their overviews)
    through PIL. Stacks are max-projected; the alpha channel is dropped.
    """
    from fibsem.fm.preview import is_fluorescence_image, load_projection
    from fibsem.structures import FibsemImage

    image = None
    pixel_size = None
    if is_fluorescence_image(path):
        data, pixel_size = load_projection(path)
    elif path.lower().endswith((".tif", ".tiff")):
        try:
            image = FibsemImage.load(path)
        except Exception:  # noqa: BLE001 - not a beam image: read it plain
            image = None
        if image is not None and image.metadata is not None:
            data = image.data
            if image.metadata.pixel_size is not None:
                pixel_size = image.metadata.pixel_size.x
        else:
            import tifffile

            image = None
            data = np.squeeze(np.asarray(tifffile.imread(path)))
    else:
        from PIL import Image

        with Image.open(path) as im:
            data = np.asarray(im.convert("RGB"))
    # z-stacks / channel stacks project on leading axes; a trailing RGB(A) axis
    # is colour, not a stack, and stays.
    while data.ndim > 2 and data.shape[-1] not in (3, 4):
        data = data.max(axis=0)
    if data.ndim == 3 and data.shape[-1] == 4:
        data = data[..., :3]
    return data, image, pixel_size


def jpeg_preview(path: str, filename: str, max_width: int = 768) -> Dict[str, Any]:
    """One saved image as downscaled JPEG bytes with its metadata block, or the
    reason it is not: the outputs routes' answer, served as image/jpeg."""
    from fibsem.server.images import image_metadata, preview_jpeg_bytes

    try:
        data, image, _ = load_image(path)
        jpeg, width, height = preview_jpeg_bytes(data, max_width=max_width)
        metadata = image_metadata(image, path) if image is not None else None
        if metadata is not None and metadata["pixel_size"] and width:
            # metres per PREVIEW pixel, as image answers carry it.
            metadata["preview_pixelsize"] = (
                metadata["pixel_size"] * data.shape[1] / width
            )
    except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
        return {
            "available": True,
            "jpeg": None,
            "error": f"Could not render {filename!r}: {exc}",
        }
    return {
        "available": True,
        "jpeg": jpeg,
        "width": width,
        "height": height,
        "filename": filename,
        "metadata": metadata,
    }


def _region(
    shape: Sequence[int],
    box: Optional[Sequence[float]],
    fraction: Optional[Sequence[float]],
) -> Tuple[int, int, int, int]:
    """The (x, y, width, height) source-pixel region a crop names: *box* in
    source pixels, or *fraction* as fractions of the image, 0 to 1."""
    if (box is None) == (fraction is None):
        raise FileRefusal(422, "invalid", "Give one region: box or fraction, not both.")
    values = list(box if box is not None else fraction)
    if len(values) != 4:
        raise FileRefusal(422, "invalid", "A region is x, y, width, height.")
    full_h, full_w = int(shape[0]), int(shape[1])
    if fraction is not None:
        if not all(0.0 <= v <= 1.0 for v in values):
            raise FileRefusal(
                422, "invalid", "fraction is left, top, width, height, each 0 to 1."
            )
        values = [
            values[0] * full_w,
            values[1] * full_h,
            values[2] * full_w,
            values[3] * full_h,
        ]
    x0 = max(0, int(round(values[0])))
    y0 = max(0, int(round(values[1])))
    x1 = min(full_w, int(round(values[0] + values[2])))
    y1 = min(full_h, int(round(values[1] + values[3])))
    if x1 <= x0 or y1 <= y0:
        raise FileRefusal(
            422,
            "invalid",
            f"The region misses the image ({full_w} x {full_h} source pixels).",
        )
    return x0, y0, x1 - x0, y1 - y0


def _image_answer(
    target: str,
    box: Optional[Sequence[float]],
    fraction: Optional[Sequence[float]],
    max_width: int,
) -> Dict[str, Any]:
    from fibsem.server.images import array_preview_payload, image_metadata

    data, image, pixel_size = load_image(target)
    source_shape = list(data.shape)
    region = None
    if box is not None or fraction is not None:
        x, y, width, height = _region(data.shape, box, fraction)
        data = data[y : y + height, x : x + width]
        region = {"x": x, "y": y, "width": width, "height": height}
    metadata = image.metadata if image is not None else None
    preview = array_preview_payload(
        data, metadata, max_width=max_width, pixel_size=pixel_size
    )
    if image is not None:
        summary = {"kind": "image", **image_metadata(image, target)}
    else:
        summary = {
            "kind": "image",
            "shape": source_shape,
            "dtype": str(data.dtype),
            "pixel_size": pixel_size,
            "path": target,
        }
    if region is not None:
        # The region in source pixels; its field width, not the image's.
        preview["region"] = region
        if pixel_size is not None:
            preview["hfw"] = pixel_size * region["width"]
    return {"summary": summary, "path": target, "preview": preview}


def preview_width(width: Optional[int]) -> int:
    """A preview's widest, as a client asks for it: 768 when it doesn't say,
    else between 64 (a thumbnail) and `MAX_PREVIEW_WIDTH`."""
    if width is None:
        return 768
    return max(64, min(int(width), MAX_PREVIEW_WIDTH))


def read(
    root: Optional[str],
    path: str,
    box: Optional[Sequence[float]] = None,
    fraction: Optional[Sequence[float]] = None,
    max_width: int = 768,
) -> Dict[str, Any]:
    """One file of the experiment, as its kind is read: a folder as its listing,
    text as text, an image as a preview and its metadata (or the region *box* or
    *fraction* names, at native resolution up to *max_width*), anything else as
    its name and size only."""
    target = resolve(root, path)
    file_kind = kind(target)
    if file_kind == "folder":
        return list_folder(root, path)
    cropping = box is not None or fraction is not None
    if cropping and file_kind != "image":
        raise FileRefusal(422, "invalid", f"Only an image crops; {path} is not one.")
    entry = _entry(os.path.realpath(root), target)
    if file_kind == "text":
        with open(target, "rb") as f:
            raw = f.read(TEXT_LIMIT + 1)
        return {
            **entry,
            "text": raw[:TEXT_LIMIT].decode("utf-8", errors="replace"),
            "truncated": len(raw) > TEXT_LIMIT,
        }
    if file_kind == "image":
        try:
            return {**entry, **_image_answer(target, box, fraction, max_width)}
        except FileRefusal:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            raise FileRefusal(
                422, "unreadable", f"Could not read {path} as an image: {exc}"
            ) from exc
    return entry
