"""Labels on a matplotlib picture, placed where they read: one size rule for
every picture, each label in a see-through box beside its mark, clear of the
picture's edges, its other labels and marks, and what is drawn over it later
(a scale bar, a legend).

The picture is a figure whose one axes fills it, at 1 display pixel per PNG
pixel; places are in the PNG's pixels from its top left, as ``(x0, y0, x1,
y1)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from fibsem.imaging.export import PLATE_COLOR

Box = Tuple[float, float, float, float]

# A label's text height, as a fraction of the picture's width, never under
# MIN_SIZE pixels.
SIZE_FRACTION = 0.024
MIN_SIZE = 12.0
# The box behind a label: the scale bar plate's fill, more see-through.
BOX_ALPHA = 0.55
# The box's padding, as a fraction of the text height.
_PAD = 0.3


def label_size(width: float) -> float:
    """The text height, in pixels, of a label on a picture *width* wide."""
    return max(MIN_SIZE, width * SIZE_FRACTION)


def _overlap(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0 and height > 0 else 0.0


def _inside(box: Box, width: float, height: float) -> bool:
    return box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height


def _clamped(box: Box, width: float, height: float) -> Box:
    w, h = box[2] - box[0], box[3] - box[1]
    x = min(max(box[0], 0.0), max(0.0, width - w))
    y = min(max(box[1], 0.0), max(0.0, height - h))
    return (x, y, x + w, y + h)


class LabelPlacer:
    """Places labels on *axes* (its data in source pixels, *scale* PNG pixels
    to one) so they don't cover each other, the marks, or *blocked* (places
    drawn over the picture later)."""

    def __init__(self, axes, scale: float, blocked: Sequence[Box] = ()) -> None:
        figure = axes.figure
        self.axes = axes
        self.scale = scale
        self.width, self.height = figure.bbox.width, figure.bbox.height
        self.size = label_size(self.width)
        self._renderer = figure.canvas.get_renderer()
        self._pt = 72 / figure.dpi
        self._taken: List[Box] = list(blocked)

    def keep_clear(self, box: Box) -> None:
        """A mark's place (PNG pixels) no label may cover."""
        self._taken.append(box)

    def keep_clear_source(
        self, x0: float, y0: float, x1: float, y1: float, margin: float = 0.0
    ) -> None:
        """A mark's place in the source's pixels, *margin* PNG pixels wider
        all round (a line's width)."""
        s, m = self.scale, margin
        self.keep_clear(
            (
                min(x0, x1) * s - m,
                min(y0, y1) * s - m,
                max(x0, x1) * s + m,
                max(y0, y1) * s + m,
            )
        )

    def _measure(self, text: str) -> Tuple[float, float]:
        probe = self.axes.text(0, 0, text, fontsize=self.size * self._pt)
        extent = probe.get_window_extent(self._renderer)
        probe.remove()
        pad = self.size * _PAD
        return extent.width + 2 * pad, self.size * 1.2 + 2 * pad

    def _spots(self, anchor: Box, w: float, h: float, gap: float):
        x0, y0, x1, y1 = anchor
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (
            (x1 + gap, cy - h / 2),
            (x0 - gap - w, cy - h / 2),
            (cx - w / 2, y1 + gap),
            (cx - w / 2, y0 - gap - h),
            (x1 + gap, y0 - gap - h),
            (x1 + gap, y1 + gap),
            (x0 - gap - w, y0 - gap - h),
            (x0 - gap - w, y1 + gap),
        )

    def place(self, anchor: Box, text: str, colour) -> Optional[Box]:
        """*text* in a box beside *anchor* (the mark's place, PNG pixels): the
        first spot right, left, below or above it that is inside the picture
        and clear; further off with a leader line when none near is; else the
        clamped spot that covers least. Answers the box drawn; None for a
        mark off the picture, which is left unlabelled."""
        if not text or not _overlap(anchor, (0.0, 0.0, self.width, self.height)):
            return None  # nothing to label, or a mark off the picture
        w, h = self._measure(text)
        near = self.size * 0.4
        tries = []
        for gap, leader in ((near, False), (self.size * 2.5, True)):
            for x, y in self._spots(anchor, w, h, gap):
                tries.append(((x, y, x + w, y + h), leader))
        chosen = None
        for box, leader in tries:
            if _inside(box, self.width, self.height) and not any(
                _overlap(box, taken) for taken in self._taken
            ):
                chosen = (box, leader)
                break
        if chosen is None:
            box, leader = min(
                (
                    (_clamped(box, self.width, self.height), leader)
                    for box, leader in tries
                ),
                key=lambda t: sum(_overlap(t[0], taken) for taken in self._taken),
            )
            chosen = (box, True)
        box, leader = chosen
        self._draw(box, anchor, text, colour, leader)
        self._taken.append(box)
        return box

    def put(self, x: float, y: float, text: str, colour) -> Box:
        """*text* in a box with its top-left corner at (*x*, *y*) (PNG pixels,
        kept inside the picture): a label whose place is fixed, as a cell's
        name in its corner. Later labels keep clear of it."""
        w, h = self._measure(text)
        box = _clamped((x, y, x + w, y + h), self.width, self.height)
        self._draw(box, box, text, colour, False)
        self._taken.append(box)
        return box

    def _draw(self, box: Box, anchor: Box, text: str, colour, leader: bool) -> None:
        from matplotlib.colors import to_rgba

        s, pad = self.scale, self.size * _PAD
        if leader:
            ax, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
            bx = min(max(ax, box[0]), box[2])
            by = min(max(ay, box[1]), box[3])
            self.axes.plot(
                [ax / s, bx / s],
                [ay / s, by / s],
                color=colour,
                linewidth=max(1.0, self.size / 14) * self._pt,
                alpha=0.8,
            )
        plate = tuple(c / 255 for c in PLATE_COLOR)
        self.axes.text(
            (box[0] + pad) / s,
            (box[1] + box[3]) / 2 / s,
            text,
            ha="left",
            va="center",
            color=colour,
            fontsize=self.size * self._pt,
            bbox=dict(
                boxstyle=f"round,pad={_PAD},rounding_size={_PAD}",
                facecolor=to_rgba(plate, BOX_ALPHA),
                edgecolor=colour,
                linewidth=1.0 * self._pt * max(1.0, self.size / 16),
            ),
        )
