"""Build every icon variant from the hand-drawn SVGs.

    python scripts/build_icons.py

The drawings are the only files edited by hand: AutoLamella's and fibsemOS's
in fibsem/ui/icons/ (the apps ship them), the others in docs/brand/icons/. Every
shape carries a role class -- tile, dark, light, detail or accent -- and each
variant below is a colour per role, so a change to a drawing reaches every
variant.

Writes:
- fibsem/ui/icons/<app>.ico for each app whose drawing is shipped there, the
  only generated files the repo keeps: Windows shortcuts need them at runtime
  (fibsem.tools.desktop_shortcut).
- docs/brand/icons/build/ (gitignored): per icon and variant an SVG and PNGs,
  an .ico for the colour variant, and contact-sheet.png with all of them.

Variants:
- colour: the app icon. Blue tile, except where ICON_COLOURS says otherwise
  (fibsemOS sits on the app's dark surface, apart from the apps it runs).
- black: a tiled monochrome for slides or print where blue clashes.
- glyph, glyph-white: one ink and no tile, for toolbars, menus and single-colour
  material. A shape classed ``glyph-hide`` is left out of the glyph (AutoLamella's
  lamella, which would merge with its trenches; the gap between them reads as it).
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
from PyQt5.QtSvg import QSvgRenderer

REPO = Path(__file__).resolve().parent.parent
APP_ICONS = REPO / "fibsem" / "ui" / "icons"
BRAND = REPO / "docs" / "brand" / "icons"
BUILD = BRAND / "build"

# name: (drawing, drawing used at 16 px and for the glyph, or None for the same)
ICONS = {
    "autolamella": (APP_ICONS / "autolamella.svg", APP_ICONS / "autolamella-16.svg"),
    "liftout": (BRAND / "liftout.svg", None),
    "volume": (BRAND / "volume.svg", None),
    "fibsemos": (APP_ICONS / "fibsemos.svg", APP_ICONS / "fibsemos-16.svg"),
}

# Per-icon changes to the colour variant.
ICON_COLOURS = {"fibsemos": {"tile": "#262930", "dark": "#5A626C"}}

# Colour per role; None for the tile means no tile. A (colour, opacity) pair
# draws that role at part strength.
VARIANTS = {
    "colour": {
        "tile": "#007ACC",
        "dark": "#0B3A63",
        "light": "#FFFFFF",
        "detail": "#9FB3C8",
        "accent": "#007ACC",
    },
    "black": {
        "tile": "#2B2B2B",
        "dark": "#0A0A0A",
        "light": "#FFFFFF",
        "detail": "#8C8C8C",
        "accent": "#8C8C8C",
    },
    "glyph": {
        "tile": None,
        "dark": "#000000",
        "light": "#000000",
        "detail": ("#000000", 0.45),
        "accent": ("#000000", 0.45),
    },
    "glyph-white": {
        "tile": None,
        "dark": "#FFFFFF",
        "light": "#FFFFFF",
        "detail": ("#FFFFFF", 0.45),
        "accent": ("#FFFFFF", 0.45),
    },
}

PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)  # 256 is the largest .ico allows
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def recolour(path: Path, colours: dict) -> bytes:
    """The drawing at ``path`` with each role in its variant colour."""
    root = ET.parse(path).getroot()
    glyph = colours["tile"] is None
    for parent in list(root.iter()):
        for shape in list(parent):
            classes = shape.get("class", "").split()
            role = next((c for c in classes if c in colours), None)
            if role is None:
                continue
            colour = colours[role]
            if colour is None or (glyph and "glyph-hide" in classes):
                parent.remove(shape)
                continue
            colour, opacity = colour if isinstance(colour, tuple) else (colour, 1.0)
            paint = "stroke" if shape.get("stroke") else "fill"
            shape.set(paint, colour)
            if opacity < 1:
                shape.set(f"{paint}-opacity", str(opacity))
    return ET.tostring(root)


def render(svg: bytes, size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg)).render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


def png_bytes(image: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data)


def ico_bytes(pngs: dict) -> bytes:
    """A Windows .ico holding one PNG per size (readable since Vista)."""
    header = b"\0\0\1\0" + len(pngs).to_bytes(2, "little")
    offset = len(header) + 16 * len(pngs)
    entries, payload = b"", b""
    for size, png in pngs.items():
        edge = size if size < 256 else 0  # 0 means 256 in the directory entry
        entries += bytes([edge, edge, 0, 0])
        entries += (1).to_bytes(2, "little") + (32).to_bytes(2, "little")
        entries += len(png).to_bytes(4, "little")
        entries += (offset + len(payload)).to_bytes(4, "little")
        payload += png
    return header + entries + payload


def build_icon(name: str, variant: str) -> dict:
    """Write one icon in one variant; return its renders by size."""
    main, small = ICONS[name]
    small = small or main
    colours = dict(VARIANTS[variant])
    if variant == "colour":
        colours.update(ICON_COLOURS.get(name, {}))
    glyph = colours["tile"] is None
    # A glyph draws from the small drawing only when that hides a shape for it.
    uses_small = glyph and "glyph-hide" in small.read_text()
    svg = recolour(small if uses_small else main, colours)
    svg_small = recolour(small, colours)

    out = BUILD / variant
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.svg").write_bytes(svg)
    renders = {s: render(svg_small if s == 16 else svg, s) for s in PNG_SIZES}
    for size, image in renders.items():
        image.save(str(out / f"{name}-{size}.png"))
    if variant == "colour":
        ico = ico_bytes({s: png_bytes(renders[s]) for s in ICO_SIZES})
        (out / f"{name}.ico").write_bytes(ico)
        if main.parent == APP_ICONS:  # a shipped app: its shortcut needs the .ico
            (APP_ICONS / f"{name}.ico").write_bytes(ico)
    return renders


def contact_sheet(renders: dict) -> QImage:
    """Every icon and variant at 128, 32 and 16 px, on dark and on light."""
    sizes, cell, row, left, top = (128, 32, 16), 230, 150, 110, 40
    width, height = left + cell * len(ICONS), top + row * len(VARIANTS)
    sheet = QImage(width, 2 * height, QImage.Format.Format_ARGB32)
    painter = QPainter(sheet)
    font = QFont()
    font.setPointSize(12)
    painter.setFont(font)
    for band, (background, ink) in enumerate(
        [("#1e1e1e", "#d6d6d6"), ("#f5f5f5", "#333333")]
    ):
        y0 = band * height
        painter.fillRect(0, y0, width, height, QColor(background))
        painter.setPen(QColor(ink))
        for column, name in enumerate(ICONS):
            painter.drawText(left + column * cell, y0 + 26, name)
        for r, variant in enumerate(VARIANTS):
            y = y0 + top + r * row
            painter.drawText(14, y + 70, variant)
            for column, name in enumerate(ICONS):
                x = left + column * cell
                for size in sizes:
                    painter.drawImage(x, y + 128 - size, renders[name, variant][size])
                    x += size + 14
    painter.end()
    return sheet


def main() -> None:
    app = QGuiApplication(sys.argv[:1])  # noqa: F841 — must outlive the renders
    renders = {(n, v): build_icon(n, v) for n in ICONS for v in VARIANTS}
    contact_sheet(renders).save(str(BUILD / "contact-sheet.png"))
    print(f"wrote the shipped apps' .ico files in {APP_ICONS}, and {BUILD}")


if __name__ == "__main__":
    main()
