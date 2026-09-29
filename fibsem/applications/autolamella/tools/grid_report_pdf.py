"""The grid screening report as a PDF (FIB-1057).

Lays out what `grid_report.collect_grid_report` read: a cover that is the
operator's verdicts, one row per grid with its latest overview; then a section
per grid, one row per overview in history order, the facts small on the left and
the image as large as the page allows on the right; then an appendix of how each
task ran. A grid that did not load, or a run that recorded nothing, keeps its row
and says why, so a missing grid reads as missing rather than absent.

Sits beside `reporting.py` (the lamella report) and shares its PDF library.
reportlab is imported inside `generate_grid_report`, not at the top, so the image
rendering here can be tested where the `reporting` extra is not installed.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Experiment,
    Verdict,
)
from fibsem.applications.autolamella.tools.grid_report import (
    GridReport,
    GridSection,
    OverviewEntry,
    collect_grid_report,
)
from fibsem.fm.preview import is_fluorescence_image, load_projection
from fibsem.structures import FibsemImage

logger = logging.getLogger(__name__)

REPORT_FILENAME = "grid-screening-report.pdf"

# The long edge of an overview as printed. The stored thumbnail (512 px) shows grid
# squares but not ice or cells; this is enough to judge a grid on an A4 page.
PRINT_MAX_EDGE = 1600
COVER_MAX_EDGE = 400

_MARK_COLOUR = (255, 210, 31)
_SCALE_BAR_CHOICES_UM = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

_STATUS_TEXT = {
    AutoLamellaTaskStatus.Completed: "Completed",
    AutoLamellaTaskStatus.Failed: "Failed",
    AutoLamellaTaskStatus.Cancelled: "Cancelled",
    AutoLamellaTaskStatus.Skipped: "Skipped",
    AutoLamellaTaskStatus.InProgress: "In progress",
    AutoLamellaTaskStatus.NotStarted: "Not started",
}
_VERDICT_TEXT = {
    Verdict.UNASSESSED: "Unassessed",
    Verdict.GOOD: "Good",
    Verdict.REWORK: "Rework",
    Verdict.FAILED: "Poor",
}


# ---------------------------------------------------------------------------
# Rendering one overview for the page
# ---------------------------------------------------------------------------


def _to_uint8(data: np.ndarray) -> np.ndarray:
    """Grey data as 8-bit for display: a 1-99 percentile stretch unless it is
    already 8-bit."""
    data = np.asarray(data)
    if data.dtype == np.uint8:
        return data
    lo, hi = np.percentile(data, (1, 99))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip((data - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _load_pixels(path: str) -> np.ndarray:
    """The image at *path* as something PIL can draw on: (H, W) grey or (H, W, 3)."""
    if is_fluorescence_image(path):
        rgb, _ = load_projection(path)
        return np.asarray(rgb, dtype=np.uint8)
    return _to_uint8(FibsemImage.load(path).data)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1 has no size argument
        return ImageFont.load_default()


def scale_bar_length(pixel_size: float, width_px: int) -> Tuple[float, str]:
    """A round bar length in metres, about a fifth of the image wide, and its label."""
    target = width_px * pixel_size * 0.2
    best = min(_SCALE_BAR_CHOICES_UM, key=lambda um: abs(um * 1e-6 - target))
    # "um", not "µm": the label is drawn into the image with Pillow's built-in
    # font, which has no micro sign. The captions, set by reportlab, use µ.
    label = f"{best / 1000:g} mm" if best >= 1000 else f"{best} um"
    return best * 1e-6, label


def render_overview(entry: OverviewEntry, max_edge: int = PRINT_MAX_EDGE) -> bytes:
    """The entry's image as PNG bytes, no longer than *max_edge* on its long side,
    with a scale bar and its lamellae boxed and named.

    Downsampled from the full overview rather than taken from the card thumbnail,
    so ice and cells are visible on the page. Marks are in the full image's pixels
    and scale with it.
    """
    if entry.path is None:
        raise ValueError(f"{entry.task_name} recorded no image to render.")
    pixels = _load_pixels(entry.path)
    image = Image.fromarray(pixels).convert("RGB")
    full_w = image.width
    image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    scale = image.width / full_w
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(image.width / 500))
    font = _font(max(10, round(image.width / 60)))

    for mark in entry.marks:
        x, y = mark.x * scale, mark.y * scale
        w, h = mark.width * scale, mark.height * scale
        box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        draw.rectangle(box, outline=(0, 0, 0), width=stroke + 1)
        draw.rectangle(box, outline=_MARK_COLOUR, width=stroke)
        draw.text(
            (box[0], box[3] + 2),
            mark.name,
            fill=_MARK_COLOUR,
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    if entry.pixel_size:
        printed_pixel = entry.pixel_size / scale
        length, label = scale_bar_length(printed_pixel, image.width)
        bar = length / printed_pixel
        margin = max(8, round(image.width / 60))
        x0, y0 = margin, image.height - margin
        draw.rectangle((x0 - 1, y0 - stroke - 2, x0 + bar + 1, y0 + 1), fill=(0, 0, 0))
        draw.rectangle((x0, y0 - stroke - 1, x0 + bar, y0), fill=(255, 255, 255))
        draw.text(
            (x0, y0 - stroke - 4),
            label,
            fill=(255, 255, 255),
            font=font,
            anchor="ls",
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _when(stamp: Optional[float], fmt: str = "%H:%M") -> str:
    return datetime.fromtimestamp(stamp).strftime(fmt) if stamp else ""


def _fov_text(fov: Optional[Tuple[float, float]]) -> str:
    if fov is None:
        return ""
    w, h = fov
    if max(w, h) >= 1e-3:
        return f"{w * 1e3:.2f} × {h * 1e3:.2f} mm"
    return f"{w * 1e6:.0f} × {h * 1e6:.0f} µm"


def _tiles_text(tiles: Optional[Tuple[int, int]]) -> str:
    return f"{tiles[0]}×{tiles[1]}" if tiles else ""


def _cover_image(section: GridSection) -> Optional[OverviewEntry]:
    """The overview a cover row shows: the latest SEM with an image, else the
    latest of any modality."""
    with_image = [o for o in section.overviews if o.path is not None]
    for entry in reversed(with_image):
        if entry.modality == "SEM":
            return entry
    return with_image[-1] if with_image else None


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def generate_grid_report(
    experiment: Experiment,
    output_path: Optional[str] = None,
    inventory: Optional[Sequence] = None,
    compress: bool = True,
) -> str:
    """Write the grid screening report and return its path.

    `output_path` defaults to `grid-screening-report.pdf` under the experiment
    folder. `inventory` is the stage's rows if one is connected, for the slot
    column. `compress` is reportlab's page compression; off, the text is greppable.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
    )
    from reportlab.platypus import (
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    report = collect_grid_report(experiment, inventory=inventory)
    if output_path is None:
        output_path = os.path.join(str(experiment.path), REPORT_FILENAME)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"Grid screening report: {report.experiment_name}",
        pageCompression=1 if compress else 0,
    )
    width = A4[0] - 30 * mm

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "t",
            parent=base["Title"],
            fontSize=20,
            leading=24,
            alignment=0,
            spaceAfter=4,
        ),
        "eyebrow": ParagraphStyle(
            "e",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#0f6f7c"),
        ),
        "h": ParagraphStyle(
            "h",
            parent=base["Heading2"],
            fontSize=14,
            leading=17,
            spaceBefore=0,
            spaceAfter=2,
        ),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=12.5),
        "muted": ParagraphStyle(
            "m",
            parent=base["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#5b6670"),
        ),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=7.5, leading=9.5),
        "th": ParagraphStyle(
            "th",
            parent=base["Normal"],
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#5b6670"),
        ),
        "info_b": ParagraphStyle(
            "ib",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12,
            fontName="Helvetica-Bold",
        ),
    }
    rule = colors.HexColor("#cfd5da")
    flat = TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 1), (-1, -1), 0.5, rule),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )

    def p(text: str, style: str = "body") -> Paragraph:
        return Paragraph(_escape(text), styles[style])

    def picture(
        entry: OverviewEntry, max_edge: int, box_width: float
    ) -> Optional[RLImage]:
        try:
            png = render_overview(entry, max_edge=max_edge)
        except Exception as e:  # noqa: BLE001 - one unreadable image must not lose the report
            logger.warning(f"Could not render {entry.path}: {e}")
            return None
        with Image.open(io.BytesIO(png)) as im:
            w, h = im.size
        shown_w = box_width if w >= h else box_width * w / h
        return RLImage(io.BytesIO(png), width=shown_w, height=shown_w * h / w)

    story: list = []

    # -- cover -----------------------------------------------------------------
    story.append(p("AutoLamella · Grid screening report", "eyebrow"))
    story.append(p(report.experiment_name, "title"))
    screened = report.screened
    meta = [
        ("Generated", _when(report.generated_at, "%d %b %Y %H:%M")),
        (
            "Screened",
            f"{_when(screened[0], '%d %b %Y %H:%M')} to {_when(screened[1])}"
            if screened
            else "nothing ran",
        ),
        ("Microscope", report.microscope or "not recorded"),
        ("Grid protocol", " → ".join(report.protocol) or "none"),
        ("Experiment folder", report.experiment_path),
    ]
    n_loaded = sum(1 for s in report.sections if s.loaded)
    n_failed = sum(1 for s in report.sections if s.loaded is False)
    n_rec = sum(1 for s in report.sections if s.recommended)
    meta.append(
        (
            "Result",
            f"{len(report.sections)} grids · {n_loaded} screened · {n_failed} did not load · {n_rec} recommended",
        )
    )
    meta_table = Table(
        [[p(k, "muted"), p(v, "small")] for k, v in meta],
        colWidths=[28 * mm, width - 28 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        ),
    )
    story += [meta_table, Spacer(1, 8)]

    thumb_w = 42 * mm
    rows = [
        [p(h, "th") for h in ("", "Grid", "Quality", "Operator's note", "Lamellae", "")]
    ]
    for index, section in enumerate(report.sections):
        cover = _cover_image(section)
        cell = picture(cover, COVER_MAX_EDGE, thumb_w) if cover else None
        if cell is None:
            reason = "did not load" if section.loaded is False else "no overview"
            cell = p(reason, "muted")
        verdict = _VERDICT_TEXT.get(
            section.quality.verdict, section.quality.verdict.name
        )
        by = f" · {section.quality.author}" if section.quality.author else ""
        sub = " · ".join(x for x in [section.slot or "", f"p. {index + 2}"] if x)
        note = section.description or section.quality.reason
        if not note and section.load is not None and section.loaded is False:
            note = f"Load failed: {section.load.status_message}"
        rows.append(
            [
                cell,
                Paragraph(
                    f"<b>{_escape(section.name)}</b><br/><font size=7 color='#5b6670'>{_escape(sub)}</font>",
                    styles["body"],
                ),
                p(verdict + by, "small"),
                p(note, "small"),
                p(str(len(section.lamellae)), "small"),
                p("Recommended" if section.recommended else "", "small"),
            ]
        )
    story.append(
        Table(
            rows,
            colWidths=[
                thumb_w + 4,
                30 * mm,
                24 * mm,
                width - thumb_w - 4 - 30 * mm - 24 * mm - 16 * mm - 24 * mm,
                16 * mm,
                24 * mm,
            ],
            style=flat,
            repeatRows=1,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        p(
            "Quality and the note are what the operator set on the grid's card. Recommended means the verdict is Good. How each task ran is in the appendix.",
            "muted",
        )
    )

    # -- one section per grid --------------------------------------------------
    info_w = 42 * mm
    image_w = width - info_w - 8
    for index, section in enumerate(report.sections):
        story.append(PageBreak())
        story.append(p(f"Grid {index + 1} of {len(report.sections)}", "eyebrow"))
        verdict = _VERDICT_TEXT.get(
            section.quality.verdict, section.quality.verdict.name
        )
        badges = verdict + (" · Recommended" if section.recommended else "")
        story.append(
            Paragraph(
                f"{_escape(section.name)} <font size=9 color='#0f6f7c'>{_escape(badges)}</font>",
                styles["h"],
            )
        )
        if section.description:
            story.append(p(section.description))
        facts = []
        if section.slot:
            facts.append(f"Slot {section.slot}")
        if section.load is not None:
            facts.append(
                f"Load {_STATUS_TEXT.get(section.load.status, section.load.status.name).lower()} {_when(section.load.when)} · {section.load.status_message}"
            )
        facts.append(
            "Lamellae " + (", ".join(section.lamellae) if section.lamellae else "none")
        )
        story.append(p("   ".join(facts), "muted"))
        story.append(Spacer(1, 4))

        if not section.overviews:
            why = (
                section.load.status_message
                if section.load
                else "No task ran on this grid."
            )
            story.append(p("No overview was taken. " + why, "body"))
            continue

        for entry in section.overviews:
            status = _STATUS_TEXT.get(entry.status, entry.status.name)
            lines = [
                f"<b>{_escape(entry.task_name)}</b>",
                _escape(entry.modality + (f" · {entry.pose}" if entry.pose else "")),
            ]
            details = [
                ("When", _when(entry.when)),
                ("Tiles", _tiles_text(entry.tiles)),
                ("FOV", _fov_text(entry.fov)),
                ("Status", status),
            ]
            if entry.channels:
                details.append(("Channels", ", ".join(entry.channels)))
            # A lamella can project to a point off the image: it is placed, but
            # not in this field. Say which, rather than listing it as marked.
            in_view, out_of_view = [], []
            for mark in entry.marks:
                h, w = entry.shape or (0, 0)
                (
                    in_view if 0 <= mark.x < w and 0 <= mark.y < h else out_of_view
                ).append(mark.name)
            if in_view:
                details.append(("Marked", ", ".join(in_view)))
            if out_of_view:
                details.append(("Out of view", ", ".join(out_of_view)))
            if entry.unmarked and entry.path is not None:
                details.append(("Not placed", ", ".join(entry.unmarked)))
            if entry.path is not None:
                details.append(("File", os.path.basename(entry.path)))
            for key, value in details:
                if value:
                    lines.append(
                        f"<font color='#5b6670'>{_escape(key)}</font> {_escape(value)}"
                    )
            info = Paragraph("<br/>".join(lines), styles["small"])

            right = (
                picture(entry, PRINT_MAX_EDGE, image_w)
                if entry.path is not None
                else None
            )
            if right is None:
                reason = entry.status_message or (
                    "No image was recorded for this run."
                    if entry.status is AutoLamellaTaskStatus.Completed
                    else ""
                )
                right = Table(
                    [[p(f"{status}", "info_b")], [p(reason, "small")]],
                    colWidths=[image_w],
                    style=TableStyle(
                        [
                            ("BOX", (0, 0), (-1, -1), 0.75, rule),
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, -1),
                                colors.HexColor("#f1f3f5"),
                            ),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ]
                    ),
                )
            row = Table(
                [[info, right]],
                colWidths=[info_w, image_w + 8],
                style=TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.5, rule),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            )
            story.append(KeepTogether(row))

    # -- appendix: how each task ran -----------------------------------------------
    story.append(PageBreak())
    story.append(p("Appendix", "eyebrow"))
    story.append(p("How each task ran", "h"))
    header = [p("Grid", "th"), p("Load", "th")] + [
        p(name, "th") for name in report.protocol
    ]
    rows = [header]
    for section in report.sections:
        load = (
            _STATUS_TEXT.get(section.load.status, section.load.status.name)
            if section.load
            else "not tried"
        )
        cells = [p(section.name, "small"), p(load, "small")]
        for name in report.protocol:
            outcome = report.outcomes[section.name][name]
            text = (
                "not run"
                if outcome.status is None
                else _STATUS_TEXT.get(outcome.status, outcome.status.name)
            )
            if outcome.runs > 1:
                text += f" ×{outcome.runs}"
            cells.append(p(text, "small"))
        rows.append(cells)
    story.append(Table(rows, colWidths=[width / len(header)] * len(header), style=flat))

    doc.build(story)
    logger.info(f"Grid screening report written to {output_path}")
    return output_path
