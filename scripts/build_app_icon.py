"""Rebuild fibsem/ui/icons/autolamella.ico from the icon's SVG drawings.

Run after changing autolamella.svg or autolamella-16.svg:

    python scripts/build_app_icon.py

Windows shortcuts need an .ico; the app itself renders the SVGs directly (see
fibsem.ui.icon.autolamella_icon), so this file is the only raster copy. Each
size is stored as a PNG entry, which Windows has read since Vista.
"""

import sys

from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
from PyQt5.QtGui import QGuiApplication

from fibsem.ui.icon import ICONS_DIR, autolamella_icon

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)  # 256 is the largest .ico allows


def png_bytes(icon, size: int) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    icon.pixmap(size, size).save(buffer, "PNG")
    return bytes(data)


def ico_bytes(images: dict) -> bytes:
    header = b"\0\0\1\0" + len(images).to_bytes(2, "little")
    offset = len(header) + 16 * len(images)
    entries, payload = b"", b""
    for size, png in images.items():
        edge = size if size < 256 else 0  # 0 means 256 in the directory entry
        entries += bytes([edge, edge, 0, 0])
        entries += (1).to_bytes(2, "little") + (32).to_bytes(2, "little")
        entries += len(png).to_bytes(4, "little")
        entries += (offset + len(payload)).to_bytes(4, "little")
        payload += png
    return header + entries + payload


def main() -> None:
    app = QGuiApplication(sys.argv[:1])  # noqa: F841 — must outlive the pixmaps
    icon = autolamella_icon()
    images = {size: png_bytes(icon, size) for size in ICO_SIZES}
    path = f"{ICONS_DIR}/autolamella.ico"
    with open(path, "wb") as f:
        f.write(ico_bytes(images))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
