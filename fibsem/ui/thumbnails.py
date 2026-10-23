"""Image loading for thumbnails, and a label that opens its image."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QLabel
from skimage.transform import resize

from fibsem.fm.preview import composite_projection, is_fluorescence_image
from fibsem.fm.structures import FluorescenceImage
from fibsem.imaging.export import ImageFields, image_fields
from fibsem.structures import FibsemImage

# Thumbnails, two to a line in the panel's width, up to the size the tab has
# always shown them at: big enough to judge a lamella from without opening it.
# Narrower panels shrink them, down to the minimum. TARGET_WIDTH is also the
# loader's default.
TARGET_WIDTH = 512
PLACEHOLDER_HEIGHT = 384  # 3:4 of the width, the box a 3:2 beam image fits


def load_and_resize(
    filepath: str, target_width: int = TARGET_WIDTH
) -> Tuple[np.ndarray, float, ImageFields]:
    """Load an image and resize to target width, preserving aspect ratio.

    Handles both a plain .tif and a fluorescence z-stack, which becomes an RGB
    channel composite. resize() preserves a trailing channel axis on its own, so
    both shapes go through the same path below.

    Returns:
        Tuple of (resized array, pixel_size_x in metres adjusted for resize, the
        image's fields, as its bar would show them). The fields come from the file
        already open for the pixels, so a caption costs no second read.
    """
    if is_fluorescence_image(filepath):
        stack = FluorescenceImage.load(filepath)
        pixel_size_x = stack.metadata.pixel_size_x
        info = image_fields(stack)
        try:
            data = composite_projection(stack)
        except ValueError as e:
            raise ValueError(f"{e}: {filepath}") from e
        finally:
            # A real METEOR stack is ~530 MB in memory where its projection is
            # ~25 MB, and the loader reads several in a row: keep only the latter.
            del stack
    else:
        img = FibsemImage.load(filepath)
        data = img.data
        if data.ndim == 3 and data.shape[2] in (3, 4):
            data = data[..., :3].mean(axis=2).astype(data.dtype)
        pixel_size_x = img.metadata.pixel_size.x
        info = image_fields(img)
    h, w = data.shape[:2]
    # Fit inside the tile box rather than filling its width. Beam images are 3:2 and
    # are width-limited, but a fluorescence stack is square: scaling it to the full
    # width makes it half again as tall as its neighbours, and taller than the
    # placeholder it replaces, so the row jumps when it finishes loading.
    max_height = target_width * PLACEHOLDER_HEIGHT / TARGET_WIDTH
    scale = min(target_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Average whole blocks of pixels down to near the size first: resize's
    # anti-aliasing filter over a full 6144 x 4096 frame took seconds an image.
    # The few edge pixels that do not fill a block are left off.
    factor = int(1 / scale)
    covered = w
    if factor >= 2:
        bh, bw = h // factor, w // factor
        covered = bw * factor
        data = (
            data[: bh * factor, :covered]
            .reshape(bh, factor, bw, factor, *data.shape[2:])
            .mean(axis=(1, 3))
        )
    resized = resize(data, (new_h, new_w), preserve_range=True).astype(np.uint8)
    return resized, pixel_size_x * covered / w / scale, info


def file_stamp(path: str) -> Optional[Tuple[int, int]]:
    """What identifies this file's *contents*: its mtime and size.

    Size as well as mtime: a filesystem with one-second mtime granularity would
    otherwise go on serving the old picture for the rest of the second in which a
    workflow rewrote it. None when there is no file to read.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


_THUMBNAIL_CACHE: "OrderedDict[tuple, QImage]" = OrderedDict()
# Enough for one arrangement of a large experiment's cards, with room for the other to
# linger after a mode switch. The bound is what stops a long run growing this without
# limit: every thumbnail a workflow rewrites lands under a new key. Cozy entries are
# the big ones at ~200 KB, so the ceiling is ~40 MB.
_THUMBNAIL_CACHE_MAX = 200


def cached_thumbnail(
    path: str, w: int, h: int, decode: Callable[[], QImage]
) -> Optional[QImage]:
    """The image `decode` reads for `path`, scaled to fill a `w` x `h` label, decoded
    at most once per version of the file.

    A strip of cards is rebuilt on every refresh, so without this a 100-lamella
    experiment re-opened and re-scaled 100 PNGs on the GUI thread each time -- 2.8 s
    of frozen UI per added lamella.

    Cached as a QImage rather than a QPixmap on purpose. It is just as quick to hand
    to a label, because `QPixmap.fromImage` on an already-scaled image costs nothing,
    and a QImage holds no platform resources -- so the cache outliving the
    QApplication at shutdown is inert rather than a teardown crash.

    None, uncached, when `decode` gives a null image.
    """
    key = (path, file_stamp(path), w, h)
    image = _THUMBNAIL_CACHE.get(key)
    if image is not None:
        _THUMBNAIL_CACHE.move_to_end(key)
        return image
    image = decode()
    if image.isNull():
        return None
    image = image.scaled(
        w,
        h,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    _THUMBNAIL_CACHE[key] = image
    if len(_THUMBNAIL_CACHE) > _THUMBNAIL_CACHE_MAX:
        _THUMBNAIL_CACHE.popitem(last=False)
    return image


class ClickableLabel(QLabel):
    """QLabel that emits clicked(filepath) when left-clicked."""

    clicked = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None) -> None:
        super().__init__(parent)
        self._filepath = filepath
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._filepath)
        super().mousePressEvent(event)
