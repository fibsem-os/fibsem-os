"""The open experiment's files, read for a client that cannot touch the disk.

One folder listed at a time; a text file (yaml, json, log, md, csv) as text; an
image as a preview with its metadata, the shape every image answer has
(:func:`fibsem.server.images.image_answer`), or one region of it at native
resolution. A file's raw bytes too (:func:`raw_file`), for an agent whose code
runs on another machine (Patrick, 2026-10-21; previews only before).

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
        raise FileRefusal(
            404,
            "not_found",
            f"No file or folder at {path} in {os.path.basename(root)}. Paths are "
            "inside the experiment; for another experiment's files, pass "
            "experiment.",
        )
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


def load_image(
    path: str,
    z: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
    max_px: Optional[int] = None,
) -> Tuple[np.ndarray, Any, Optional[float], Optional[Dict[str, Any]], Tuple[int, ...]]:
    """A saved image as something viewable: its data (2-D, or RGB), the
    FibsemImage it is when it is a beam image (else None), its pixel size in
    metres when known, a fluorescence stack's legend facts (else None), and
    the full image's shape. With *max_px*, the data may be a smaller copy the
    file holds, still that size on its long side; the pixel size and the shape
    stay the full image's.

    Fluorescence stacks (``.ome.tiff``) composite through the FM canvas's own
    projection, so each channel keeps its colour: plane *z* (0 the first) in
    *channels* when given, else the max projection of every channel. TIFFs read
    as FibsemImages; anything else (the PNG thumbnails grid tasks write beside
    their overviews) through PIL. Stacks are max-projected; the alpha channel is
    dropped.
    """
    from fibsem.fm.preview import is_fluorescence_image, load_stack_preview
    from fibsem.structures import FibsemImage

    image = None
    pixel_size = None
    fm = None
    full = None
    picked = z is not None or bool(channels)
    if is_fluorescence_image(path):
        try:
            data, pixel_size, fm, full = load_stack_preview(
                path, z, channels or None, max_px
            )
        except ValueError as exc:
            if picked:  # a plane or channel the stack doesn't have
                raise FileRefusal(422, "invalid", str(exc)) from exc
            raise
    elif picked:
        raise FileRefusal(
            422,
            "invalid",
            "Only a fluorescence stack (.ome.tiff) has planes and channels to pick.",
        )
    elif path.lower().endswith((".tif", ".tiff")):
        try:
            image, full = _beam_image(path, max_px)
        except Exception:  # noqa: BLE001 - not a beam image: read it plain
            image = None
        if image is not None and image.metadata is not None:
            data = image.data
            if image.metadata.pixel_size is not None:
                pixel_size = image.metadata.pixel_size.x
        else:
            import tifffile

            image, full = None, None
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
    if full is None:
        full = data.shape[:2]
    return data, image, pixel_size, fm, tuple(full[-2:]) + tuple(data.shape[2:])


def _beam_image(path: str, max_px: Optional[int]) -> Tuple[Any, Tuple[int, ...]]:
    """A beam image's FibsemImage, its data from the smallest copy the file holds
    that is still *max_px* on its long side, and its full shape."""
    import tifffile

    from fibsem.image_format import read_level, read_metadata_dict
    from fibsem.structures import FibsemImage, FibsemImageMetadata

    if max_px is None:
        image = FibsemImage.load(path)
        return image, image.data.shape
    with tifffile.TiffFile(path) as tif:
        data, full = read_level(tif, max_px)
        raw = read_metadata_dict(tif)
    metadata = FibsemImageMetadata.from_dict(raw) if raw is not None else None
    image = FibsemImage(data=data, metadata=metadata)
    image.filepath = path
    return image, full


def jpeg_preview(path: str, filename: str, max_width: int = 768) -> Dict[str, Any]:
    """One saved image as downscaled JPEG bytes with its metadata block, or the
    reason it is not: the outputs routes' answer, served as image/jpeg."""
    from fibsem.server.images import image_metadata, preview_jpeg_bytes

    try:
        data, image, pixel_size, fm, full = load_image(path, max_px=max_width)
        jpeg, width, height = preview_jpeg_bytes(data, max_width=max_width)
        metadata = image_metadata(image, path, full) if image is not None else None
        if fm is not None:
            metadata = {"fm": fm, "pixel_size": pixel_size}
        if metadata is not None and metadata["pixel_size"] and width:
            # metres per PREVIEW pixel, as image answers carry it.
            metadata["preview_pixelsize"] = metadata["pixel_size"] * full[1] / width
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
    z: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    from fibsem.server.images import array_preview_payload, image_metadata

    cropping = box is not None or fraction is not None
    # a crop is in source pixels, so it reads the full image
    data, image, pixel_size, fm, full = load_image(
        target, z, channels, None if cropping else max_width
    )
    source_shape = list(full)
    region = None
    if cropping:
        x, y, width, height = _region(data.shape, box, fraction)
        data = data[y : y + height, x : x + width]
        region = {"x": x, "y": y, "width": width, "height": height}
    metadata = image.metadata if image is not None else None
    preview = array_preview_payload(
        data,
        metadata,
        max_width=max_width,
        pixel_size=pixel_size,
        full_shape=None if cropping else full,
    )
    if image is not None:
        summary = {"kind": "image", **image_metadata(image, target, full)}
    else:
        summary = {
            "kind": "image",
            "shape": source_shape,
            "dtype": str(data.dtype),
            "pixel_size": pixel_size,
            "path": target,
        }
        if fm is not None:
            summary["fm"] = fm
            summary["preview_of"] = {
                "z": z,  # None: the max projection
                "channels": list(channels) if channels else None,  # None: all
            }
    if region is not None:
        # The region in source pixels; its field width, not the image's.
        preview["region"] = region
        if pixel_size is not None:
            preview["hfw"] = pixel_size * region["width"]
    return {
        "summary": summary,
        "path": target,
        "preview": preview,
        "info": _bar_info(target, image, fm, z),
    }


def _bar_info(path: str, image, fm, z: Optional[int]) -> Optional[Dict[str, Any]]:
    """What the bar under the image says (`fibsem.server.image_payloads`); None
    for a file that records nothing."""
    from fibsem.server.image_payloads import fm_info, image_info

    if image is not None:
        return image_info(image)
    if fm is None:
        return None
    from fibsem.fm.stack_file import StackFile

    with StackFile(path) as stack:
        return fm_info(stack.metadata, stack.shape, z)


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
    z: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """One file of the experiment, as its kind is read: a folder as its listing,
    text as text, an image as a preview and its metadata (or the region *box* or
    *fraction* names, at native resolution up to *max_width*), anything else as
    its name and size only. A fluorescence stack shows plane *z* in *channels*
    when given (:func:`load_image`)."""
    target = resolve(root, path)
    file_kind = kind(target)
    if file_kind == "folder":
        return list_folder(root, path)
    cropping = box is not None or fraction is not None
    if (cropping or z is not None or channels) and file_kind != "image":
        raise FileRefusal(
            422, "invalid", f"Only an image crops or shows a plane; {path} is not one."
        )
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
            return {
                **entry,
                **_image_answer(target, box, fraction, max_width, z, channels),
            }
        except FileRefusal:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            raise FileRefusal(
                422, "unreadable", f"Could not read {path} as an image: {exc}"
            ) from exc
    return entry


def read_etag(root: Optional[str], path: str, *query: Any) -> Optional[str]:
    """An ETag for :func:`read`'s answer for a file: its size and modification time,
    with *query* (what was asked of it), so a browser keeps the answer until the
    file changes. None for a folder, whose listing changes with what it holds."""
    import hashlib

    target = resolve(root, path)
    if os.path.isdir(target):
        return None
    stat = os.stat(target)
    key = repr((os.path.realpath(target), stat.st_size, stat.st_mtime_ns, query))
    return '"' + hashlib.sha1(key.encode()).hexdigest() + '"'


def raw_file(root: Optional[str], path: str) -> Tuple[str, str]:
    """One file of the experiment to send as it is: its real path and its path
    relative to the experiment, or the :class:`FileRefusal` saying why not."""
    target = resolve(root, path)
    if os.path.isdir(target):
        raise FileRefusal(
            422, "invalid", f"{path} is a folder: fetch its files one at a time."
        )
    relative = os.path.relpath(target, os.path.realpath(root)).replace(os.sep, "/")
    return target, relative
