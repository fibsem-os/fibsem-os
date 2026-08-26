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
  and the map immediately beneath. Every page after the first opens with the same band in
  miniature -- its own name, a page number, the same rule -- so a sheet that gets
  separated from the others still says what it is.
* **Rules, not boxes.** Horizontal hairlines only, and the table's first column flush
  with the margin. A grid of black cells is louder than the numbers in it, and the
  numbers are the point.
* **Numbers that line up, and headings that do not wrap.** Columns are measured in
  points, from the strings that will actually be set, so `THICKNESS` never breaks across
  two lines to leave a stray `S`.
* **Pictures on a matte.** Every image sits on the near-black the application's canvas
  uses. On the map that is load-bearing -- the corners between two composited overviews
  are ground that was never imaged, and on white they read as a hole in the page. On a
  card it is only an edge, so it hugs the picture rather than filling the column.
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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# CANVAS_BG. 16.2:1 on white -- the palette's darkest neutral is already the right ink
# for paper, so the document's body colour is literally the application's canvas.
INK = colors.HexColor("#1e2124")
# TEXT_MUTED_COLOR darkened to 7.0:1, for text that is secondary but still read.
INK_SOFT = colors.HexColor("#55595f")
# TEXT_MUTED_COLOR unchanged, 3.3:1 -- fine for labels and captions, not for body.
INK_FAINT = colors.HexColor("#868e93")
# BORDER_COLOR lightened to a hairline weight; the same colour at full strength is a
# 10:1 line, which on paper reads as a box rather than a rule.
RULE = colors.HexColor("#dce1e7")
ROW_TINT = colors.HexColor("#f4f6f8")
# The dashed edge of a slot whose image was never acquired: heavier than RULE, because a
# dash at hairline weight disappears, and the empty slot has to be legible as a choice.
DASH = colors.HexColor("#c9ced4")
# IMAGE_CANVAS_BG. What sits behind a picture on screen, and behind every picture here.
MATTE = colors.HexColor("#0d0d0d")
# PRIMARY_ACCENT, 5.3:1 on white and unchanged. The quad-view selection blue works on
# paper as-is.
ACCENT = colors.HexColor("#3a6ea5")
# ERROR_COLOR nudged 4.7 -> 5.4:1, and WARN_COLOR moved 2.3 -> 4.5:1, which is the one
# that genuinely fails on white. Lightness only; hue and saturation are the tokens'.
DEFECT_FAILED = colors.HexColor("#c23434")
DEFECT_REWORK = colors.HexColor("#9e6d17")

PAGE_MARGIN = 0.62 * inch
CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN

# Where the page number is stamped: right-aligned at the margin, on the first baseline
# under it, which is where a continuation page's own title sits. The two line up because
# both are measured from the same edge, not because either was nudged to match.
_STAMP_SIZE = 7.2
_STAMP_BASELINE = A4[1] - PAGE_MARGIN - 7.6


def _styles() -> Dict[str, ParagraphStyle]:
    """The text roles this document has. Deliberately few.

    Sizes are the mockup's, read as points. The one that matters is `fact` against
    `caption`: the facts are the larger of the two, because a reader flipping to a card
    is looking for a number, not for the note someone left about it.
    """
    return {
        # Left-aligned and sized so a full experiment name fits one line at A4 width.
        # The centred 24pt it replaces broke `AutoLamella-2026-07-29-18-00-DEV` across
        # two lines mid-token, which is unreadable and looks like a bug.
        "name": ParagraphStyle(
            "name",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=3,
        ),
        # The same band in miniature, opening every page after the first.
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13.5,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "summary": ParagraphStyle(
            "summary",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=INK_SOFT,
            alignment=TA_LEFT,
        ),
        # Right of the masthead, so the two are read as a pair rather than as a heading
        # followed by small print.
        "provenance": ParagraphStyle(
            "provenance",
            fontName="Helvetica",
            fontSize=_STAMP_SIZE,
            leading=11,
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
            fontSize=9,
            leading=12.5,
            textColor=INK_SOFT,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=INK_FAINT,
        ),
        "legend": ParagraphStyle(
            "legend",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK_SOFT,
        ),
        "lamella": ParagraphStyle(
            "lamella",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12.5,
            textColor=INK,
        ),
        # The facts under a lamella's name. Sentence case, not the uppercase the table
        # headers and the picture labels use: those label a column or a frame, where the
        # reader is scanning for a position. These sit inside a sentence-shaped line and
        # are read.
        "fact": ParagraphStyle(
            "fact",
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=INK_SOFT,
        ),
        "shotlabel": ParagraphStyle(
            "shotlabel",
            fontName="Helvetica-Bold",
            fontSize=6.6,
            leading=8.6,
            textColor=INK_SOFT,
        ),
        "shotdetail": ParagraphStyle(
            "shotdetail",
            fontName="Helvetica",
            fontSize=6.6,
            leading=8.6,
            textColor=INK_FAINT,
            alignment=TA_RIGHT,
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


class _NumberedCanvas(pdfcanvas.Canvas):
    """Stamps `page N of M` in the top-right corner of every page after the first.

    Two passes, because M is not known while the pages are being laid out: each page's
    drawing operations are held, and the numbers go on once the count is final. Page one
    is skipped -- the provenance block already occupies that corner, and a sheet holding
    the masthead does not need to be told which sheet it is.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pages: List[dict] = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for number, state in enumerate(self._pages, start=1):
            self.__dict__.update(state)
            if number > 1:
                self.setFont("Helvetica", _STAMP_SIZE)
                self.setFillColor(INK_FAINT)
                self.drawRightString(
                    A4[0] - PAGE_MARGIN,
                    _STAMP_BASELINE,
                    f"page {number} of {total}",
                )
            super().showPage()
        super().save()


class _Doc(BaseDocTemplate):
    """A page whose frame has no padding of its own.

    `SimpleDocTemplate` builds its frame with reportlab's default 6pt padding, so the
    usable width is 12pt narrower than the margins claim. The two kinds of flowable here
    then disagree about where the page starts: a `Table` given the full measure is wider
    than the frame and gets centred, landing on the margin, while a `Paragraph` is laid
    inside the padding, 6pt in. That is why the section headings used to sit indented
    from the tables beneath them. Zeroing the padding makes `CONTENT_WIDTH` true and puts
    everything on the same left edge -- the one the page number is measured from too.
    """

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="page",
        )
        self.addPageTemplates([PageTemplate(id="page", frames=[frame])])


class HandoffDocument:
    """Builds the handoff PDF, one band at a time."""

    def __init__(self, output_filename: str):
        self.output_filename = output_filename
        self.doc = _Doc(
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
        self._rule()

    def section(self, title: str) -> None:
        """The masthead in miniature, opening a continuation page.

        Nothing goes in the right-hand corner: `_NumberedCanvas` stamps `page N of M`
        there once the page count is known, which is not while this is being written.
        """
        self.story.append(Paragraph(title, self.s["section"]))
        self._rule()

    def band(self, title: str, new_page: bool = True) -> None:
        """Open a band of the document, on its own page or below what came before.

        One call site for a decision that is easy to get wrong in two directions: a
        section that starts a page needs the miniature masthead, and a section that
        follows something on the page it is already on must *not* have one -- a second
        full-width rule directly under the masthead's reads as a printing fault.
        """
        if new_page:
            self.page_break()
            self.section(title)
        else:
            self.heading(title)

    def _rule(self) -> None:
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

    def legend(self, entries: Sequence[Tuple[Optional[Any], str]]) -> None:
        """A row of swatch-and-label pairs under a figure.

        What the marker colours mean, said once on the page they appear on, rather than
        left for the reader to infer from a red cross. An entry with no colour is set
        faint and carries no swatch -- that is where the view the map was drawn in goes,
        which is a caption rather than a key.
        """
        entries = [e for e in entries if e[1]]
        if not entries:
            return

        cells: List[Any] = []
        widths: List[float] = []
        for colour, label in entries:
            style = self.s["legend"]
            if colour is None:
                style = ParagraphStyle("legendfaint", parent=style, textColor=INK_FAINT)
                cells.append(Paragraph(label, style))
            else:
                pair = Table(
                    [[_dot(colour), Paragraph(label, style)]],
                    colWidths=[10, None],
                )
                pair.setStyle(TableStyle(_flush(valign="MIDDLE")))
                cells.append(pair)
            widths.append(
                pdfmetrics.stringWidth(label, "Helvetica", 8)
                + (10 if colour is not None else 0)
                + 14
            )

        # The last column takes the slack, so the row fills the measure and a trailing
        # caption sits against the right margin rather than in the middle of the page.
        slack = CONTENT_WIDTH - sum(widths)
        if slack > 0:
            widths[-1] += slack

        self.story.append(Spacer(1, 6))
        row = Table([cells], colWidths=widths)
        row.setStyle(TableStyle(_flush(valign="MIDDLE")))
        self.story.append(row)

    def figure(
        self,
        fig,
        max_width: float = CONTENT_WIDTH,
        max_height: float = 4.9 * inch,
        matte: bool = True,
    ) -> None:
        """Place a matplotlib figure at its own aspect, fitted to the space.

        Fitted rather than forced into a box: the fixed 6.5x4 inch box it replaces
        letterboxed a wide overview and spent the difference on white space -- the same
        defect the on-screen export had.

        On a matte by default. A composited map covers the ground its overviews covered
        and no more, so the corners between them are genuinely unimaged; on white they
        read as a hole in the page, and on the canvas colour they read as what they are.
        """
        buf, aspect = _figure_png(fig)
        width = max_width
        height = width * aspect
        if height > max_height:
            height = max_height
            width = height / aspect if aspect else max_width
        image = Image(buf, width=width, height=height)
        self.story.append(_matted(image, width, height) if matte else image)

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
            fontSize=_TH_SIZE,
            leading=9,
            textColor=INK_FAINT,
        )
        th_r = ParagraphStyle("thr", parent=th, alignment=TA_RIGHT)
        td = ParagraphStyle(
            "td", fontName="Helvetica", fontSize=_TD_SIZE, leading=10, textColor=INK
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
            # Flush with the margin on the left and the right, so the table's edges are
            # the page's edges and the heading above it does not appear indented.
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), _CELL_PAD),
            ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
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
        """Measured in points, from the strings that will actually be set.

        The version this replaces counted characters, which under-reads a header: set in
        bold capitals, `THICKNESS` is far wider than nine characters of the lower-case
        body text it was compared against. The columns came out too narrow and the
        headers wrapped mid-word, leaving a stray `S` on its own line -- and `15.0 deg`
        broke in half in the cell below.

        So every column is given its header's width as a floor and is never squeezed
        below it. What is left over goes to the columns whose *content* is wider than
        their header -- the prose ones, notes and task names -- in proportion to how much
        more they asked for. Prose is what wraps, which is what wrapping is for.
        """
        floors, wants = [], []
        for c in columns:
            header = pdfmetrics.stringWidth(c.upper(), "Helvetica-Bold", _TH_SIZE)
            widest = max(
                [
                    pdfmetrics.stringWidth(str(r.get(c, "")), "Helvetica", _TD_SIZE)
                    for r in rows
                ]
                or [0.0]
            )
            floors.append(header + _CELL_PAD)
            wants.append(max(header, widest) + _CELL_PAD)

        total_floor, total_want = sum(floors), sum(wants)
        if total_want <= CONTENT_WIDTH:
            # Room to spare: hand it to the prose columns rather than padding every
            # column equally, which would leave a lake of white beside `15.0 deg`.
            excess = [w - f for w, f in zip(wants, floors)]
            spare = CONTENT_WIDTH - total_want
            share = sum(excess) or 1.0
            return [w + spare * e / share for w, e in zip(wants, excess)]
        if total_floor >= CONTENT_WIDTH:
            # More headers than the page is wide. Nothing can be honoured, so fall back
            # to proportional and let it wrap rather than overflowing the margin.
            return [CONTENT_WIDTH * f / (total_floor or 1.0) for f in floors]
        excess = [w - f for w, f in zip(wants, floors)]
        keep = (CONTENT_WIDTH - total_floor) / (sum(excess) or 1.0)
        return [f + e * keep for f, e in zip(floors, excess)]

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
            block.append(Spacer(1, 13))
            block.append(
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=RULE,
                    spaceBefore=0,
                    spaceAfter=13,
                )
            )

        block.append(self._fact_line(name, facts))
        if note:
            block.append(Spacer(1, 3))
            block.append(Paragraph(note, self.s["caption"]))
        block.append(Spacer(1, 7))

        if shots:
            block.append(self._shot_row(shots))

        self.story.append(KeepTogether(block))

    def _fact_line(self, name: str, facts: Sequence[Tuple[str, str]]) -> Any:
        """The name, then every fact on one line, so the pictures get the width.

        Sized to the name rather than given a fixed column: a fixed column has to be wide
        enough for the longest name any experiment might produce, which leaves a gap
        after every name shorter than that, and the facts start somewhere that has
        nothing to do with where the name ended.
        """
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
            parts.append(f'{key} <font color="{_hex(colour)}"><b>{value}</b></font>')

        gap = 12
        name_width = (
            pdfmetrics.stringWidth(name, "Helvetica-Bold", self.s["lamella"].fontSize)
            + gap
        )
        name_width = min(name_width, CONTENT_WIDTH * 0.45)
        row = Table(
            [
                [
                    Paragraph(name, self.s["lamella"]),
                    Paragraph("&nbsp;&nbsp;&nbsp;".join(parts), self.s["fact"]),
                ]
            ],
            colWidths=[name_width, CONTENT_WIDTH - name_width],
        )
        # Baselines, not boxes: the facts read as a continuation of the name rather than
        # as a caption that happens to be beside it.
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
        fit and centred, giving up width rather than pushing the row down. Its matte hugs
        it rather than filling the column: on a card the matte is an edge, not ground --
        every picture fills its own frame -- and a column-wide one turns a square
        fluorescence projection into two black pillars that read as part of the design.

        The label is a pair: what the picture is, left; how wide a field it covers,
        right. The second is not decoration -- these three images differ in field of view
        by more than an order of magnitude, and the reader is about to look for the same
        object in a third instrument.
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
                caption = "-"
            else:
                buf, aspect = payload
                height = min(width * aspect, box_h)
                image_width = height / aspect if aspect else width
                picture = Table(
                    [
                        [
                            _matted(
                                Image(buf, width=image_width, height=height),
                                image_width,
                                height,
                            )
                        ]
                    ],
                    colWidths=[width],
                    rowHeights=[box_h],
                )
                picture.setStyle(
                    TableStyle(
                        _flush(valign="TOP") + [("ALIGN", (0, 0), (-1, -1), "CENTER")]
                    )
                )
                caption = detail
            label_row = Table(
                [
                    [
                        Paragraph(label.upper(), self.s["shotlabel"]),
                        Paragraph(caption, self.s["shotdetail"]),
                    ]
                ],
                colWidths=[width * 0.62, width * 0.38],
            )
            label_row.setStyle(TableStyle(_flush(valign="TOP")))
            cells.append([picture, Spacer(1, 3), label_row])

        row = Table(
            [cells], colWidths=[width + (gap if i < n - 1 else 0) for i in range(n)]
        )
        row.setStyle(TableStyle(_flush(valign="TOP")))
        return row

    def page_break(self) -> None:
        self.story.append(PageBreak())

    def build(self) -> str:
        self.doc.build(self.story, canvasmaker=_NumberedCanvas)
        return self.output_filename


# The strings `handoff_map` uses for a value it does not have. Listed here so the table
# and the fact line can render them faintly rather than at full strength -- an unknown
# should not draw the eye the way a measurement does.
_ABSENT = {"-", "not milled", "none", "not acquired", ""}

# The table's type sizes and the space kept between one column's text and the next.
# Named because `_column_widths` measures with them, and a size that drifted out of step
# with the measurement is exactly how the headers came to wrap.
_TH_SIZE = 7.0
_TD_SIZE = 7.6
_CELL_PAD = 8.0


def _dot(colour) -> Any:
    """A small filled circle, for a legend swatch.

    Drawn rather than set as a glyph: the obvious character for this, U+25CF, is not in
    the WinAnsi encoding reportlab's built-in Helvetica uses, and comes out as a black
    box or as nothing at all depending on the reader.

    Outlined, because the fill is the *marker's* colour rather than a colour chosen for
    paper -- and the markers are drawn to be seen against a dark image, so the default
    cyan is about 1.3:1 on white. The ring is what makes the swatch a swatch there. The
    alternative, recolouring the dot to something legible, would print a key that does
    not match the map it is a key to.
    """
    from reportlab.graphics.shapes import Circle, Drawing

    d = Drawing(7, 7)
    d.add(
        Circle(
            3.5,
            2.5,
            3.0,
            fillColor=colour,
            strokeColor=INK_FAINT,
            strokeWidth=0.5,
        )
    )
    return d


def _matted(flowable: Any, width: float, height: float) -> Any:
    """A picture on the near-black the application's canvas uses.

    Both for what it says and for what it hides: the ground between two composited
    overviews was never imaged, and on white that reads as a hole in the page.
    """
    box = Table([[flowable]], colWidths=[width], rowHeights=[height])
    box.setStyle(
        TableStyle(
            _flush(valign="MIDDLE")
            + [
                ("BACKGROUND", (0, 0), (-1, -1), MATTE),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return box


def _missing_box(text: str, width: float, height: float, style) -> Any:
    """A ruled placeholder where an image would be.

    A kept, labelled gap rather than a closed-up row: otherwise a two-image card looks
    like a short one, and "never acquired" is indistinguishable from "did not render".

    Dashed and pale rather than matted like a real picture, so it never reads at a glance
    as an image that came out black.
    """
    box = Table([[Paragraph(text, style)]], colWidths=[width], rowHeights=[height])
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, DASH, None, (2, 2)),
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

        # The canvas's own scalebar, so a picture on a card is the picture on screen.
        from fibsem.imaging.tiling.plotting import _SCALEBAR

        ax.add_artist(ScaleBar(dx=pixel_size, location="lower right", **_SCALEBAR))
    except Exception as e:
        logger.debug(f"Could not add a scalebar: {e}")
