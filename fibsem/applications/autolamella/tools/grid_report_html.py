"""The grid screening report as an HTML page.

Lays out what `grid_report.collect_grid_report` read: a cover that is the
operator's verdicts, one row per grid with its latest overview; then a section
per grid, one row per overview in history order, the facts small on the left and
the image as large as the page allows on the right; then an appendix of how each
task ran. A grid that did not load, or a run that recorded nothing, keeps its row
and says why, so a missing grid reads as missing rather than absent.

One file in the lamella report's style (`report_v2.html_page`), the overviews
embedded as JPEGs, so it can be passed on as it is. It prints to A4, one grid to a
page, and a browser on the machine prints it to a PDF beside it (`pdf_export`).
"""

from __future__ import annotations

import base64
import html
import io
import logging
import os
from datetime import tzinfo
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fibsem.acting import Actor
from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.tools.grid_report import (
    GridReport,
    GridSection,
    OverviewEntry,
    collect_grid_report,
)
from fibsem.applications.autolamella.tools.report_v2 import REPORT_DIRNAME, html_page
from fibsem.fm.preview import is_fluorescence_image, load_projection
from fibsem.structures import FibsemImage
from fibsem.util.timestamps import to_datetime, zone_label
from fibsem.workflows.quality import Verdict
from fibsem.workflows.status import TaskStatus

logger = logging.getLogger(__name__)

REPORT_FILENAME = "grid-screening-report.html"

# The long edge of an overview as printed. The stored thumbnail (512 px) shows grid
# squares but not ice or cells; this is enough to judge a grid on an A4 page.
PRINT_MAX_EDGE = 1600
COVER_MAX_EDGE = 400

_MARK_COLOUR = (255, 210, 31)
_SCALE_BAR_CHOICES_UM = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

_STATUS_TEXT = {
    TaskStatus.Completed: "Completed",
    TaskStatus.Failed: "Failed",
    TaskStatus.Cancelled: "Cancelled",
    TaskStatus.Skipped: "Skipped",
    TaskStatus.InProgress: "In progress",
    TaskStatus.NotStarted: "Not started",
}
# The app's words for a verdict, so the page says what the Grids tab said. Kept as
# literals: the app's table lives in a Qt module this report must not import.
_VERDICT_TEXT = {
    Verdict.UNASSESSED: "Not assessed",
    Verdict.GOOD: "Good",
    Verdict.REWORK: "Rework",
    Verdict.FAILED: "Failed",
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
    # font, which has no micro sign. The page's captions use µ.
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


def _when(stamp, fmt: str = "%H:%M", zone: Optional[tzinfo] = None) -> str:
    """A time in the zone the cover names: the instrument's when the
    experiment recorded it, else this machine's. A task's time is aware,
    the report's own is POSIX; both are converted, so every time on the page is on
    the one clock the cover states."""
    when = to_datetime(stamp) if stamp else None
    if when is None:
        return ""
    if when.tzinfo is not None:
        when = when.astimezone(zone)
    return when.strftime(fmt)


def _clock(zone: Optional[tzinfo]) -> str:
    if zone is None:
        return "this computer's clock"
    return f"the instrument's clock, {zone_label(zone)}"


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


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _picture(entry: OverviewEntry, max_edge: int) -> Optional[str]:
    """The entry's overview as an ``<img>``, or None when it can't be read."""
    try:
        png = render_overview(entry, max_edge=max_edge)
        with Image.open(io.BytesIO(png)) as image:
            out = io.BytesIO()
            image.convert("RGB").save(out, format="JPEG", quality=85)
    except Exception as e:  # noqa: BLE001 - one unreadable image must not lose the report
        logger.warning(f"Could not render {entry.path}: {e}")
        return None
    data = base64.b64encode(out.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{data}" alt="{_e(entry.task_name)}">'


def _table(head: Sequence[str], rows: List[List[str]]) -> str:
    """A list table; the cells are HTML already."""
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<table class="list"><tr>{th}</tr>{body}</table>'


def _verdict(section: GridSection) -> str:
    return _VERDICT_TEXT.get(section.quality.verdict, section.quality.verdict.name)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def generate_grid_report(
    experiment: Experiment,
    output_path: Optional[str] = None,
    inventory: Optional[Sequence] = None,
) -> str:
    """Write the grid screening report and return its path.

    `output_path` defaults to `reporting/grid-screening-report.html` under the
    experiment folder. `inventory` is the stage's rows if one is connected, for
    the slot column.
    """
    report = collect_grid_report(experiment, inventory=inventory)
    session = getattr(experiment, "session", None)
    zone = session.zone if session is not None else None
    if output_path is None:
        output_path = os.path.join(
            str(experiment.path), REPORT_DIRNAME, REPORT_FILENAME
        )
    body = [_cover(report, zone)]
    for index, section in enumerate(report.sections):
        body.append(_grid(section, index, len(report.sections), zone))
    body.append(_appendix(report))
    page = html_page(
        f"{report.experiment_name} · Grid screening report", "\n".join(body), _STYLE
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(page, encoding="utf-8")
    logger.info(f"Grid screening report written to {output_path}")
    return output_path


def write_grid_report(
    experiment: Experiment, inventory: Optional[Sequence] = None
) -> Path:
    """Write the page under the experiment folder, then print it to a PDF beside
    it where a browser on the machine can; the page's path. Without a browser
    there is the page and a log line saying why there is no PDF."""
    from fibsem.applications.autolamella.tools import pdf_export

    path = Path(generate_grid_report(experiment, inventory=inventory))
    try:
        pdf = pdf_export.html_to_pdf(path)
    except pdf_export.PdfExportError as e:
        logger.info(f"No PDF of the grid screening report: {e}.")
    else:
        logger.info(f"Grid screening report printed to {pdf}")
    return path


def _cover(report: GridReport, zone: Optional[tzinfo]) -> str:
    screened = report.screened
    n_loaded = sum(1 for s in report.sections if s.loaded)
    n_failed = sum(1 for s in report.sections if s.loaded is False)
    n_rec = sum(1 for s in report.sections if s.recommended)
    meta = [
        ("Generated", _when(report.generated_at, "%d %b %Y %H:%M", zone)),
        (
            "Screened",
            f"{_when(screened[0], '%d %b %Y %H:%M', zone)} to "
            f"{_when(screened[1], zone=zone)}"
            if screened
            else "nothing ran",
        ),
        ("Times", _clock(zone)),
        ("Microscope", report.microscope or "not recorded"),
        ("Grid protocol", " → ".join(report.protocol) or "none"),
        ("Experiment folder", report.experiment_path),
        (
            "Result",
            f"{len(report.sections)} grids · {n_loaded} screened · "
            f"{n_failed} did not load · {n_rec} recommended",
        ),
    ]
    parts = [
        '<header><div><div class="eyebrow">AutoLamella · Grid screening report</div>'
        f"<h1>{_e(report.experiment_name)}</h1></div>"
        '<button class="noprint" onclick="window.print()">Print</button></header>',
        '<table class="meta">'
        + "".join(
            f'<tr><th>{_e(k)}</th><td class="small">{_e(v)}</td></tr>' for k, v in meta
        )
        + "</table>",
    ]
    # Written at the end of a run, before anyone has looked: a record of the
    # overviews, not yet the handout for choosing a grid. Said, so it is not
    # read as "nothing was any good".
    if report.sections and all(
        s.quality.verdict is Verdict.UNASSESSED for s in report.sections
    ):
        parts.append(
            '<p class="muted">No grid has a verdict yet, so none is recommended. '
            "Set each grid's quality on the Grids tab, then write the report again "
            "to choose a grid.</p>"
        )
    rows = []
    for index, section in enumerate(report.sections):
        cover = _cover_image(section)
        picture = _picture(cover, COVER_MAX_EDGE) if cover else None
        if picture is None:
            reason = "did not load" if section.loaded is False else "no overview"
            picture = f'<span class="muted">{reason}</span>'
        # The person, not the stored "operator:<name>" form a verdict is signed with.
        actor = section.quality.actor
        by = f" · {Actor.parse(actor).label}" if actor else ""
        note = section.description or section.quality.reason
        if not note and section.load is not None and section.loaded is False:
            note = f"Load failed: {section.load.status_message}"
        name = f'<a href="#grid-{index + 1}"><b>{_e(section.name)}</b></a>'
        if section.slot:
            name += f'<br><span class="muted">{_e(section.slot)}</span>'
        rows.append(
            [
                f'<div class="thumb">{picture}</div>',
                name,
                _e(_verdict(section) + by),
                _e(note or ""),
                _e(len(section.lamellae)),
                "Recommended" if section.recommended else "",
            ]
        )
    parts.append(
        _table(("", "Grid", "Quality", "Operator's note", "Lamellae", ""), rows)
    )
    parts.append(
        '<p class="muted small">Quality and the note are what the operator set on '
        "the grid's card. Recommended means the verdict is Good. How each task ran "
        "is in the appendix.</p>"
    )
    return "<section>" + "".join(parts) + "</section>"


def _grid(section: GridSection, index: int, count: int, zone: Optional[tzinfo]) -> str:
    badges = _verdict(section) + (" · Recommended" if section.recommended else "")
    parts = [
        f'<div class="eyebrow">Grid {index + 1} of {count}</div>',
        f'<h2>{_e(section.name)} <span class="badge">{_e(badges)}</span></h2>',
    ]
    if section.description:
        parts.append(f"<p>{_e(section.description)}</p>")
    facts = []
    if section.slot:
        facts.append(f"Slot {section.slot}")
    if section.load is not None:
        status = _STATUS_TEXT.get(section.load.status, section.load.status.name)
        facts.append(
            f"Load {status.lower()} {_when(section.load.when, zone=zone)} · "
            f"{section.load.status_message}"
        )
    facts.append(
        "Lamellae " + (", ".join(section.lamellae) if section.lamellae else "none")
    )
    parts.append(f'<p class="muted small">{_e("   ".join(facts))}</p>')
    if not section.overviews:
        why = (
            section.load.status_message if section.load else "No task ran on this grid."
        )
        parts.append(f"<p>No overview was taken. {_e(why)}</p>")
    for entry in section.overviews:
        parts.append(_overview(entry, zone))
    return (
        f'<section class="grid" id="grid-{index + 1}">' + "".join(parts) + "</section>"
    )


def _overview(entry: OverviewEntry, zone: Optional[tzinfo]) -> str:
    status = _STATUS_TEXT.get(entry.status, entry.status.name)
    details = [
        ("When", _when(entry.when, zone=zone)),
        ("Tiles", _tiles_text(entry.tiles)),
        ("FOV", _fov_text(entry.fov)),
        ("Status", status),
    ]
    if entry.channels:
        details.append(("Channels", ", ".join(entry.channels)))
    # A run that finished with something to say: an overview that skipped
    # tiles past the stage's reach has blanks there, and this says why.
    if entry.status is TaskStatus.Completed and (
        entry.status_message not in ("", "Finished")
    ):
        details.append(("Run", entry.status_message))
    # A lamella can project to a point off the image: it is placed, but
    # not in this field. Say which, rather than listing it as marked.
    in_view, out_of_view = [], []
    for mark in entry.marks:
        h, w = entry.shape or (0, 0)
        (in_view if 0 <= mark.x < w and 0 <= mark.y < h else out_of_view).append(
            mark.name
        )
    if in_view:
        details.append(("Marked", ", ".join(in_view)))
    if out_of_view:
        details.append(("Out of view", ", ".join(out_of_view)))
    if entry.unmarked and entry.path is not None:
        details.append(("Not placed", ", ".join(entry.unmarked)))
    if entry.path is not None:
        details.append(("File", os.path.basename(entry.path)))
    modality = entry.modality + (f" · {entry.pose}" if entry.pose else "")
    info = [f"<b>{_e(entry.task_name)}</b>", _e(modality)] + [
        f'<span class="muted">{_e(key)}</span> {_e(value)}'
        for key, value in details
        if value
    ]
    picture = _picture(entry, PRINT_MAX_EDGE) if entry.path is not None else None
    if picture is None:
        reason = entry.status_message or (
            "No image was recorded for this run."
            if entry.status is TaskStatus.Completed
            else ""
        )
        picture = f'<div class="no-image"><b>{_e(status)}</b><br>{_e(reason)}</div>'
    return (
        f'<div class="overview"><div class="small">{"<br>".join(info)}</div>'
        f"<div>{picture}</div></div>"
    )


def _appendix(report: GridReport) -> str:
    rows = []
    for section in report.sections:
        load = (
            _STATUS_TEXT.get(section.load.status, section.load.status.name)
            if section.load
            else "not tried"
        )
        cells = [_e(section.name), _e(load)]
        for name in report.protocol:
            outcome = report.outcomes[section.name][name]
            text = (
                "not run"
                if outcome.status is None
                else _STATUS_TEXT.get(outcome.status, outcome.status.name)
            )
            if outcome.runs > 1:
                text += f" ×{outcome.runs}"
            cells.append(_e(text))
        rows.append(cells)
    return (
        '<section class="grid"><div class="eyebrow">Appendix</div>'
        "<h2>How each task ran</h2>"
        + _table(["Grid", "Load", *report.protocol], rows)
        + "</section>"
    )


_STYLE = """
.eyebrow { font-size: 12px; color: var(--muted); }
.meta { border-collapse: collapse; margin: 12px 0 20px; }
.meta th { text-align: left; font-weight: 500; color: var(--muted);
  padding: 1px 16px 1px 0; vertical-align: top; white-space: nowrap; }
.list td { vertical-align: top; }
.thumb img { width: 160px; display: block; border-radius: 4px; background: #000; }
.list a { white-space: nowrap; }
.badge { font-size: 13px; font-weight: 500; color: var(--muted); }
.overview { display: grid; grid-template-columns: 180px 1fr; gap: 12px;
  padding: 10px 0; border-bottom: 1px solid var(--line); break-inside: avoid; }
.overview img { max-width: 100%; display: block; border-radius: 4px;
  background: #000; }
.no-image { border: 1px solid var(--line); background: var(--soft);
  border-radius: 4px; padding: 8px 10px; font-size: 12px; }
@media print {
  section.grid { break-before: page; break-inside: auto; }
  .thumb img { width: 120px; }
  .thumb img, .overview img { print-color-adjust: exact;
    -webkit-print-color-adjust: exact; }
}
"""
