"""How the handoff map is laid out on the page.

Split from `handoff_map`, which decides *what* the document says. This decides what it
looks like, and it has its own styles rather than borrowing `PDFReportGenerator`'s.

That reuse is what the first version did, and it showed: a 24pt centred title that broke
the experiment name mid-token and left page one 85% empty, a table with a black gridline
around every cell, and images dropped into a fixed 4x3 inch box whatever shape they were.
None of it is wrong for the report it was built for; all of it was wrong for a page
someone reads once, at an instrument, to decide where to point a microscope.

What this aims at instead:

* **A masthead, not a title page.** Name and provenance on one band with a rule under,
  and the map immediately beneath. Nothing here is worth a page of its own -- the reader
  wants the picture.
* **Rules, not boxes.** Horizontal hairlines only. A grid of black cells is louder than
  the numbers in it, and the numbers are the point.
* **Numbers that line up.** Right-aligned and tabular, so a column can be read down.
* **An absence that reads as an absence.** "not milled" rather than a bare dash, which
  looks like a value that failed to render.

**The palette is the application's**, not one invented for print. Every colour below is a
token from `fibsem/ui/tokens.py`, unchanged where it already works on white and darkened
by lightness alone where it does not -- so the hue is still the app's. Contrast ratios
against white are recorded beside each, because that is the thing that decides them.

They are literals rather than imports of the tokens they name for one specific reason:
`fibsem/ui/__init__` eagerly imports the whole Qt widget stack, so importing anything
from that package -- `tokens` included, despite `tokens` itself being Qt-free -- makes
the importing module unimportable wherever PyQt5 is absent. This module is reachable
from a workflow hook and from CI, both of which are such places. Keep these in step with
tokens.py by hand.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# CANVAS_BG. 16.2:1 on white -- the palette's darkest neutral is already the right ink
# for paper, so the document's body colour is literally the application's canvas.
INK = colors.HexColor("#1e2124")
# TEXT_MUTED_COLOR darkened to 7.1:1, for text that is secondary but still read.
INK_SOFT = colors.HexColor("#53595d")
# TEXT_MUTED_COLOR unchanged, 3.3:1 -- fine for labels and captions, not for body.
INK_FAINT = colors.HexColor("#868e93")
# BORDER_COLOR lightened to a hairline weight; the same colour at full strength is a
# 10:1 line, which on paper reads as a box rather than a rule.
RULE = colors.HexColor("#dcdee4")
ROW_TINT = colors.HexColor("#f8f9fa")
# PRIMARY_ACCENT, 5.3:1 on white and unchanged. The quad-view selection blue works on
# paper as-is.
ACCENT = colors.HexColor("#3a6ea5")
# ERROR_COLOR nudged 4.7 -> 5.4:1, and WARN_COLOR moved 2.3 -> 4.5:1, which is the one
# that genuinely fails on white. Lightness only; hue and saturation are the tokens'.
DEFECT_FAILED = colors.HexColor("#c23434")
DEFECT_REWORK = colors.HexColor("#9e6d17")

PAGE_MARGIN = 0.62 * inch
CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN


def _styles() -> Dict[str, ParagraphStyle]:
    """The text roles this document has. Deliberately few."""
    return {
        # Left-aligned and sized so a full experiment name fits one line at A4 width.
        # The centred 24pt it replaces broke `AutoLamella-2026-07-29-18-00-DEV` across
        # two lines mid-token, which is unreadable and looks like a bug.
        "name": ParagraphStyle(
            "name",
            fontName="Helvetica-Bold",
            fontSize=14.5,
            leading=17.5,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "summary": ParagraphStyle(
            "summary",
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            textColor=INK_SOFT,
            alignment=TA_LEFT,
        ),
        # Right of the masthead, so the two are read as a pair rather than as a heading
        # followed by small print.
        "provenance": ParagraphStyle(
            "provenance",
            fontName="Helvetica",
            fontSize=7,
            leading=9.6,
            textColor=INK_FAINT,
            alignment=TA_RIGHT,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=8.6,
            leading=12,
            textColor=INK_SOFT,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.8,
            textColor=INK_FAINT,
        ),
        "lamella": ParagraphStyle(
            "lamella",
            fontName="Helvetica-Bold",
            fontSize=9.6,
            leading=12,
            textColor=INK,
        ),
        "shotlabel": ParagraphStyle(
            "shotlabel",
            fontName="Helvetica-Bold",
            fontSize=6.4,
            leading=8.5,
            textColor=INK_FAINT,
        ),
    }


def _figure_png(fig) -> Tuple[io.BytesIO, float]:
    """Render *fig* to a PNG buffer and return it with its aspect (height / width)."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    w_in, h_in = fig.get_size_inches()
    aspect = float(h_in) / float(w_in) if w_in else 0.75

    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf, aspect


class HandoffDocument:
    """Builds the handoff PDF, one band at a time."""

    def __init__(self, output_filename: str):
        self.output_filename = output_filename
        self.doc = SimpleDocTemplate(
            output_filename,
            pagesize=A4,
            leftMargin=PAGE_MARGIN,
            rightMargin=PAGE_MARGIN,
            topMargin=PAGE_MARGIN,
            bottomMargin=PAGE_MARGIN,
            title=os.path.splitext(os.path.basename(output_filename))[0],
        )
        self.s = _styles()
        self.story: List[Any] = []

    # ── bands ────────────────────────────────────────────────────────────

    def masthead(
        self, name: str, summary: str = "", provenance: str = "", right_note: str = ""
    ) -> None:
        """Name and contents left, provenance right, one rule under.

        A two-column table rather than stacked paragraphs, so the provenance sits
        *beside* the title instead of under it -- which is what keeps this a band a few
        centimetres tall rather than the top third of a page.
        """
        left: List[Any] = [Paragraph(name, self.s["name"])]
        if summary:
            left.append(Paragraph(summary, self.s["summary"]))

        right_text = provenance.replace("  |  ", "<br/>") if provenance else right_note
        row = Table(
            [[left, Paragraph(right_text, self.s["provenance"])]],
            colWidths=[CONTENT_WIDTH * 0.62, CONTENT_WIDTH * 0.38],
        )
        row.setStyle(TableStyle(_flush(valign="TOP")))
        self.story.append(row)
        self.story.append(Spacer(1, 5))
        self.story.append(
            HRFlowable(
                width="100%", thickness=1, color=INK, spaceBefore=0, spaceAfter=9
            )
        )

    def note(self, text: str) -> None:
        if not text:
            return
        self.story.append(Paragraph(text, self.s["body"]))
        self.story.append(Spacer(1, 7))

    def heading(self, text: str) -> None:
        if not text:
            return
        self.story.append(Paragraph(text, self.s["heading"]))

    def caption(self, text: str) -> None:
        if not text:
            return
        self.story.append(Spacer(1, 4))
        self.story.append(Paragraph(text, self.s["caption"]))

    def figure(
        self, fig, max_width: float = CONTENT_WIDTH, max_height: float = 4.9 * inch
    ) -> None:
        """Place a matplotlib figure at its own aspect, fitted to the space.

        Fitted rather than forced into a box: the fixed 6.5x4 inch box it replaces
        letterboxed a wide overview and spent the difference on white space -- the same
        defect the on-screen export had.
        """
        buf, aspect = _figure_png(fig)
        width = max_width
        height = width * aspect
        if height > max_height:
            height = max_height
            width = height / aspect if aspect else max_width
        self.story.append(Image(buf, width=width, height=height))

    def table(
        self,
        columns: Sequence[str],
        rows: Sequence[Dict[str, str]],
        numeric: Sequence[str] = (),
    ) -> None:
        """A table of rules, not boxes."""
        if not rows:
            self.story.append(Paragraph("Nothing to list.", self.s["body"]))
            return

        th = ParagraphStyle(
            "th",
            fontName="Helvetica-Bold",
            fontSize=6.6,
            leading=8.6,
            textColor=INK_FAINT,
        )
        th_r = ParagraphStyle("thr", parent=th, alignment=TA_RIGHT)
        td = ParagraphStyle(
            "td", fontName="Helvetica", fontSize=7.4, leading=9.8, textColor=INK
        )
        td_r = ParagraphStyle("tdr", parent=td, alignment=TA_RIGHT)
        td_faint = ParagraphStyle("tdf", parent=td, textColor=INK_FAINT)
        td_faint_r = ParagraphStyle("tdfr", parent=td_r, textColor=INK_FAINT)

        numeric = set(numeric)
        data = [[Paragraph(c.upper(), th_r if c in numeric else th) for c in columns]]
        for r in rows:
            cells = []
            for c in columns:
                value = str(r.get(c, ""))
                # An unknown reads faint, so a column of real numbers is not broken up
                # by full-strength text saying there is no number.
                faint = value in _ABSENT
                if c in numeric:
                    style = td_faint_r if faint else td_r
                else:
                    style = td_faint if faint else td
                cells.append(Paragraph(value, style))
            data.append(cells)

        table = Table(data, repeatRows=1, colWidths=self._column_widths(columns, rows))
        style = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
            ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_TINT]),
        ]
        # Colour the defect cell, not the row. A full red band reads as "do not use",
        # which is a stronger claim than the record makes: a flagged lamella is still a
        # lamella, and the reader decides.
        if "Defect" in columns:
            col = list(columns).index("Defect")
            for i, r in enumerate(rows, start=1):
                value = str(r.get("Defect", ""))
                if value == "failed":
                    style.append(("TEXTCOLOR", (col, i), (col, i), DEFECT_FAILED))
                elif value == "rework":
                    style.append(("TEXTCOLOR", (col, i), (col, i), DEFECT_REWORK))
        table.setStyle(TableStyle(style))
        self.story.append(table)

    @staticmethod
    def _column_widths(columns: Sequence[str], rows: Sequence[Dict[str, str]]):
        """Room for each column's widest entry, then share out what is left.

        Without this reportlab divides the width evenly, so a column of "15.0 deg" gets
        as much room as one holding a task name, and the task names wrap.
        """
        weights = []
        for c in columns:
            longest = max([len(c)] + [len(str(r.get(c, ""))) for r in rows])
            weights.append(max(4, min(longest, 26)))
        total = float(sum(weights)) or 1.0
        return [CONTENT_WIDTH * w / total for w in weights]

    def lamella_card(
        self,
        name: str,
        facts: Sequence[Tuple[str, str]],
        note: str,
        shots: Sequence[Tuple[str, Optional[Any], str]],
        first: bool = False,
    ) -> None:
        """One lamella: a fact line, then its images across the page.

        Images across the width rather than beside the facts, because there are up to
        three of them at different fields of view -- the fluorescence stack is not the
        ion image at another zoom, so each carries its own scalebar and its own label.

        A slot whose image was never acquired is kept and says so. Closing up would make
        a two-image card look like a short one, and "not acquired" would be
        indistinguishable from "did not render".

        Held together so a card never splits across a page break: half a lamella at the
        foot of a page is worse than a gap.
        """
        block: List[Any] = []
        if not first:
            block.append(Spacer(1, 9))
            block.append(
                HRFlowable(
                    width="100%", thickness=0.4, color=RULE, spaceBefore=0, spaceAfter=9
                )
            )

        block.append(self._fact_line(name, facts))
        if note:
            block.append(Spacer(1, 3))
            block.append(Paragraph(note, self.s["caption"]))
        block.append(Spacer(1, 6))

        if shots:
            block.append(self._shot_row(shots))

        self.story.append(KeepTogether(block))

    def _fact_line(self, name: str, facts: Sequence[Tuple[str, str]]) -> Any:
        """The name, then every fact on one line, so the pictures get the width."""
        parts = []
        for key, value in facts:
            colour = INK
            if key == "Defect" and value == "failed":
                colour = DEFECT_FAILED
            elif key == "Defect" and value == "rework":
                colour = DEFECT_REWORK
            elif value in _ABSENT:
                colour = INK_FAINT
            # `hexval()` returns "0xrrggbb"; reportlab's inline colour attribute wants
            # "#rrggbb", and rejects the bare digits with a ValueError that names the
            # colour rather than the tag, which is a slow thing to read.
            parts.append(
                f'<font color="{_hex(INK_FAINT)}" size="6.6">{key.upper()}</font> '
                f'<font color="{_hex(colour)}"><b>{value}</b></font>'
            )
        line = ParagraphStyle(
            "fl", fontName="Helvetica", fontSize=6.9, leading=9.6, textColor=INK
        )
        row = Table(
            [
                [
                    Paragraph(name, self.s["lamella"]),
                    Paragraph("&nbsp;&nbsp;".join(parts), line),
                ]
            ],
            colWidths=[1.15 * inch, CONTENT_WIDTH - 1.15 * inch],
        )
        row.setStyle(TableStyle(_flush(valign="BOTTOM")))
        return row

    def _shot_row(self, shots: Sequence[Tuple[str, Optional[Any], str]]) -> Any:
        """Up to three images across the page, each labelled underneath.

        Every picture is fitted into a box of the *same height*, so the labels sit on one
        line. Without that the tallest image -- the fluorescence projection, which is
        square where the beam images are 3:2 -- hangs below the others and drags its
        label with it, and the row reads as three things that were not meant to be
        compared. The box takes the shallowest aspect in the row, so nothing grows the
        row taller than the images that were already there; a taller image is scaled to
        fit and centred, giving up width rather than pushing the row down.
        """
        gap = 9
        n = len(shots)
        width = (CONTENT_WIDTH - gap * (n - 1)) / n

        rendered = []
        aspects = []
        for label, fig, detail in shots:
            if fig is None:
                rendered.append((label, None, detail))
            else:
                buf, aspect = _figure_png(fig)
                rendered.append((label, (buf, aspect), detail))
                aspects.append(aspect)

        box_h = width * (min(aspects) if aspects else 0.62)

        cells = []
        for label, payload, detail in rendered:
            if payload is None:
                picture: Any = _missing_box(
                    detail or "Not acquired", width, box_h, self.s["caption"]
                )
            else:
                buf, aspect = payload
                height = min(width * aspect, box_h)
                image_width = height / aspect if aspect else width
                picture = Table(
                    [[Image(buf, width=image_width, height=height)]],
                    colWidths=[width],
                    rowHeights=[box_h],
                )
                picture.setStyle(
                    TableStyle(
                        _flush(valign="TOP") + [("ALIGN", (0, 0), (-1, -1), "CENTER")]
                    )
                )
            cells.append(
                [picture, Spacer(1, 2), Paragraph(label.upper(), self.s["shotlabel"])]
            )

        row = Table(
            [cells], colWidths=[width + (gap if i < n - 1 else 0) for i in range(n)]
        )
        row.setStyle(TableStyle(_flush(valign="TOP")))
        return row

    def page_break(self) -> None:
        self.story.append(PageBreak())

    def build(self) -> str:
        self.doc.build(self.story)
        return self.output_filename


# The strings `handoff_map` uses for a value it does not have. Listed here so the table
# and the fact line can render them faintly rather than at full strength -- an unknown
# should not draw the eye the way a measurement does.
_ABSENT = {"-", "not milled", "none", "not acquired", ""}


def _missing_box(text: str, width: float, height: float, style) -> Any:
    """A ruled placeholder where an image would be.

    A kept, labelled gap rather than a closed-up row: otherwise a two-image card looks
    like a short one, and "never acquired" is indistinguishable from "did not render".
    """
    box = Table([[Paragraph(text, style)]], colWidths=[width], rowHeights=[height])
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("BACKGROUND", (0, 0), (-1, -1), ROW_TINT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return box


def _hex(colour) -> str:
    """A reportlab Color as the "#rrggbb" its inline markup expects."""
    return "#" + colour.hexval()[2:]


def _flush(valign: str = "TOP") -> List[tuple]:
    """A table style with no padding anywhere -- for tables used purely as layout."""
    return [
        ("VALIGN", (0, 0), (-1, -1), valign),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]


def bare_image_figure(image, width_in: float = 3.4):
    """A beam image and a scalebar, nothing else, at the image's own aspect."""
    import matplotlib.pyplot as plt

    from fibsem.imaging.tiling.plotting import figsize_for_image

    fig, ax = plt.subplots(
        figsize=figsize_for_image(image.data.shape, width_in=width_in)
    )
    ax.imshow(image.data, cmap="gray")
    ax.axis("off")
    _add_scalebar(ax, getattr(getattr(image.metadata, "pixel_size", None), "x", None))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def fluorescence_figure(image, width_in: float = 3.4):
    """A fluorescence stack as the application would show it: max projection, blended.

    Reuses `composite_fm_layers`, which is what the FM canvas composites with, so the
    picture on the card is the one the operator saw on screen -- channel colours and
    all -- rather than a grey projection that happens to come from the same file.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from fibsem.fm.composite import FMLayer, composite_fm_layers

    projection = np.asarray(image.max_intensity_projection(return_2d=True))
    if projection.ndim == 2:
        projection = projection[np.newaxis, ...]

    channels = getattr(image.metadata, "channels", None) or []
    layers = []
    for i in range(projection.shape[0]):
        channel = channels[i] if i < len(channels) else None
        layers.append(
            FMLayer(
                name=getattr(channel, "name", f"channel {i}"),
                data=projection[i],
                color=getattr(channel, "color", None) or "gray",
            )
        )
    rgb = composite_fm_layers(layers)
    if rgb is None:
        raise ValueError("the fluorescence stack composited to nothing")

    height, width = rgb.shape[0], rgb.shape[1]
    aspect = height / width if width else 1.0
    fig, ax = plt.subplots(figsize=(width_in, width_in * aspect))
    ax.imshow(rgb)
    ax.axis("off")
    _add_scalebar(ax, getattr(image.metadata, "pixel_size_x", None))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def _add_scalebar(ax, pixel_size: Optional[float]) -> None:
    """A scalebar per image, because the images on a card are at different scales.

    A fluorescence stack and an ion image of the same lamella differ in pixel size by
    more than an order of magnitude, so one shared bar would be wrong on at least one
    of them.
    """
    if not pixel_size:
        return
    try:
        from matplotlib_scalebar.scalebar import ScaleBar

        ax.add_artist(
            ScaleBar(
                dx=pixel_size,
                color="black",
                box_color="white",
                box_alpha=0.6,
                location="lower right",
                font_properties={"size": 6},
            )
        )
    except Exception as e:
        logger.debug(f"Could not add a scalebar: {e}")
