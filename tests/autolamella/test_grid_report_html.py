"""The grid screening report on the page (FIB-1057): the overviews as drawn, and
the HTML page they are embedded in. Printing it to a PDF is `pdf_export`'s,
tested in `test_pdf_export.py`; here it is a stand-in.
"""

import io
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fibsem.applications.autolamella.tools import grid_report_html, pdf_export
from fibsem.applications.autolamella.tools.grid_report_html import (
    PRINT_MAX_EDGE,
    REPORT_FILENAME,
    _clock,
    _when,
    generate_grid_report,
    render_overview,
    scale_bar_length,
    write_grid_report,
)
from fibsem.util.timestamps import from_posix
from fibsem.workflows.quality import Verdict

from .test_grid_report import report, screened  # noqa: F401 - fixtures

_YELLOW = np.array([255, 210, 31])


def _png(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


class TestRenderOverview:
    def test_downsampled_from_the_full_image_keeping_its_shape(self, report):
        sem = report.sections[0].overviews[0]
        png = _png(render_overview(sem, max_edge=200))
        h, w = png.shape[:2]
        assert max(h, w) == 200
        assert w / h == pytest.approx(sem.shape[1] / sem.shape[0], abs=0.02)

    def test_never_upscales(self, report):
        sem = report.sections[0].overviews[0]
        png = _png(render_overview(sem, max_edge=PRINT_MAX_EDGE))
        assert max(png.shape[:2]) <= max(sem.shape)

    def test_lamellae_are_boxed_where_their_mark_is(self, report):
        sem = report.sections[0].overviews[0]
        png = _png(render_overview(sem, max_edge=sem.shape[1]))  # 1:1
        mark = next(m for m in sem.marks if m.name == "lamella-01")
        # the box's left edge, at the mark's height: yellow
        x = round(mark.x - mark.width / 2)
        y = round(mark.y)
        patch = png[y - 2 : y + 3, x - 2 : x + 3].reshape(-1, 3)
        assert any(np.array_equal(px, _YELLOW) for px in patch)
        # far from any mark: not yellow
        assert not any(
            np.array_equal(px, _YELLOW) for px in png[2:8, 2:8].reshape(-1, 3)
        )

    def test_a_scale_bar_sits_bottom_left(self, report):
        sem = report.sections[0].overviews[0]
        png = _png(render_overview(sem, max_edge=400))
        h, w = png.shape[:2]
        corner = png[int(h * 0.85) :, : int(w * 0.4)].reshape(-1, 3)
        assert any(np.array_equal(px, (255, 255, 255)) for px in corner)

    def test_the_fm_composite_renders_in_colour(self, report):
        fm = report.sections[0].overviews[2]
        png = _png(render_overview(fm, max_edge=300))
        assert png.ndim == 3 and png.shape[2] == 3

    def test_a_row_without_an_image_refuses(self, report):
        failed = report.sections[1].overviews[1]
        with pytest.raises(ValueError, match="recorded no image"):
            render_overview(failed)


class TestScaleBar:
    def test_a_round_length_about_a_fifth_of_the_width(self):
        # 1.85 mm across 1600 px: a fifth is 370 um, nearest round is 500 um
        length, label = scale_bar_length(1.85e-3 / 1600, 1600)
        assert (length, label) == (500e-6, "500 um")
        length, label = scale_bar_length(10e-3 / 1000, 1000)
        assert (length, label) == (2e-3, "2 mm")


@pytest.fixture(scope="module")
def page(screened, tmp_path_factory):
    path = generate_grid_report(
        screened, output_path=str(tmp_path_factory.mktemp("page") / "report.html")
    )
    return Path(path).read_text(encoding="utf-8")


def _shown(page: str) -> str:
    """The text the page shows, without its tags or its embedded images.

    For asserting a word is *absent*: the images' data is long, and a short word
    turns up in it by chance.
    """
    return re.sub(r"<[^>]+>", " ", re.sub(r"<img[^>]*>", "", page))


class TestPage:
    def test_writes_under_the_experiment_by_default(self, screened):
        path = generate_grid_report(screened)
        assert Path(path) == Path(screened.path) / "reporting" / REPORT_FILENAME
        assert Path(path).read_text(encoding="utf-8").startswith("<!doctype html>")

    def test_a_cover_a_section_per_grid_and_an_appendix(self, page):
        # three grids and the appendix, each printed from a new page
        assert page.count('<section class="grid"') == 4
        assert "How each task ran" in page

    def test_the_overviews_are_in_the_page(self, page):
        """One file to pass on: every image is embedded, none is a link."""
        sources = re.findall(r'<img src="([^"]*)"', page)
        assert len(sources) >= 5  # the cover's and each section's
        assert all(s.startswith("data:image/jpeg;base64,") for s in sources)

    def test_the_cover_reads_as_the_verdicts(self, page):
        for text in (
            "Grid screening report",
            "grid-aspen",
            "grid-birch",
            "grid-cedar",
            "Recommended",
            "Even ice, cells on the east half.",
            "could not be gripped",
        ):
            assert text in page, text

    def test_the_verdicts_read_as_the_app_says_them(self, page):
        """The Grids tab says Failed and Not assessed; the page said Poor and
        Unassessed."""
        text = _shown(page)
        assert "Failed" in text and "Not assessed" in text
        assert "Poor" not in text and "Unassessed" not in text

    def test_a_verdict_names_the_person_not_the_stored_form(self, page):
        """A verdict is signed "operator:<name>"; the page shows the name."""
        text = _shown(page)
        assert "Good · operator" in text
        assert "operator:operator" not in text

    def test_a_failed_run_says_why(self, page):
        assert "Load it first" in page
        assert "Failed" in page

    def test_the_fm_row_names_its_channels(self, page):
        assert "GFP, mCherry" in page

    def test_with_verdicts_the_cover_says_nothing_of_their_absence(self, page):
        assert "No grid has a verdict yet" not in _shown(page)

    def test_before_any_verdict_the_cover_says_so(self, screened, tmp_path):
        """Written at the end of a run, before anyone has judged a grid: a
        record of the overviews, not yet the handout for choosing one."""
        verdicts = [g.quality.verdict for g in screened.grids]
        try:
            for grid in screened.grids:
                grid.quality.verdict = Verdict.UNASSESSED
            path = generate_grid_report(screened, output_path=str(tmp_path / "r.html"))
        finally:
            for grid, verdict in zip(screened.grids, verdicts):
                grid.quality.verdict = verdict
        text = _shown(Path(path).read_text(encoding="utf-8"))
        assert "No grid has a verdict yet" in text
        assert "0 recommended" in text

    def test_every_run_has_a_row(self, page):
        assert page.count("overview_sem") >= 3  # protocol line, aspen row, birch row


class TestPrinting:
    def test_the_pdf_is_printed_beside_the_page(self, screened, monkeypatch):
        printed = []

        def print_to_pdf(path):
            printed.append(Path(path))
            return Path(path).with_suffix(".pdf")

        monkeypatch.setattr(pdf_export, "html_to_pdf", print_to_pdf)
        path = write_grid_report(screened)
        assert printed == [path]
        assert path.suffix == ".html" and path.exists()

    def test_without_a_browser_there_is_still_the_page(
        self, screened, monkeypatch, caplog
    ):
        def no_browser(path):
            raise pdf_export.PdfExportError("no Edge, Chrome or Chromium was found")

        monkeypatch.setattr(pdf_export, "html_to_pdf", no_browser)
        with caplog.at_level("INFO", logger=grid_report_html.logger.name):
            path = write_grid_report(screened)
        assert path.exists()
        assert "No PDF of the grid screening report" in caplog.text


class TestTimes:
    """A task's POSIX time reads on the instrument's clock once the experiment has
    recorded its zone, and on this computer's until then; the cover says which
    (FIB-1196)."""

    STAMP = 1789355531.0  # 2026-09-14 03:12:11 UTC
    MDT = timezone(timedelta(hours=-6), "MDT")

    @pytest.fixture(params=["UTC", "Australia/Sydney", "America/Los_Angeles"])
    def viewer_zone(self, request, monkeypatch):
        if not hasattr(time, "tzset"):
            pytest.skip("time.tzset is POSIX-only")
        monkeypatch.setenv("TZ", request.param)
        time.tzset()
        yield request.param
        monkeypatch.undo()
        time.tzset()

    def test_the_instruments_clock_from_any_zone(self, viewer_zone):
        assert _when(self.STAMP, "%d %b %Y %H:%M", self.MDT) == "13 Sep 2026 21:12"

    def test_a_task_time_reads_on_the_instruments_clock_from_any_zone(
        self, viewer_zone
    ):
        """A task's time is aware since FIB-1197: one loaded from an older float
        carries this machine's offset, one written at the instrument carries its
        own. Both land on the clock the cover names."""
        loaded = from_posix(self.STAMP)
        written = datetime.fromisoformat("2026-09-13T21:12:11-06:00")
        for when in (loaded, written):
            assert _when(when, "%d %b %Y %H:%M", self.MDT) == "13 Sep 2026 21:12"

    def test_this_computers_clock_without_a_recorded_zone(self, viewer_zone):
        expected = time.strftime("%H:%M", time.localtime(self.STAMP))
        assert _when(self.STAMP) == expected

    def test_the_cover_names_the_clock(self, page):
        assert _clock(self.MDT) == "the instrument's clock, MDT (UTC-06:00)"
        # the fixture's session is a Demo one, which records no zone
        assert "this computer" in page
