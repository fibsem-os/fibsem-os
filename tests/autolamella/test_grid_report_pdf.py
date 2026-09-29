"""The grid screening report on the page (FIB-1057).

The image rendering is checked wherever Pillow is, which is everywhere. The PDF
itself needs the `reporting` extra (reportlab), which CI does not install, so
those tests skip there and run locally.
"""

import io
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fibsem.applications.autolamella.tools.grid_report_pdf import (
    PRINT_MAX_EDGE,
    REPORT_FILENAME,
    generate_grid_report,
    render_overview,
    scale_bar_length,
)

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
def pdf(screened, tmp_path_factory):
    pytest.importorskip("reportlab")
    path = generate_grid_report(
        screened,
        output_path=str(tmp_path_factory.mktemp("pdf") / "report.pdf"),
        compress=False,
    )
    return Path(path).read_bytes()


def _pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf))


class TestPDF:
    def test_writes_under_the_experiment_by_default(self, screened):
        pytest.importorskip("reportlab")
        path = generate_grid_report(screened)
        assert path.endswith(REPORT_FILENAME)
        assert path.startswith(str(screened.path))
        assert Path(path).read_bytes()[:4] == b"%PDF"

    def test_a_cover_a_page_per_grid_and_an_appendix(self, pdf):
        # cover, three grids (one of them over two pages), appendix
        assert _pages(pdf) >= 5

    def test_the_cover_reads_as_the_verdicts(self, pdf):
        for text in (
            b"Grid screening report",
            b"grid-aspen",
            b"grid-birch",
            b"grid-cedar",
            b"Recommended",
            b"Even ice, cells on the east half.",
            b"could not be gripped",
        ):
            assert text in pdf, text

    def test_a_failed_run_says_why(self, pdf):
        assert b"Load it first" in pdf
        assert b"Failed" in pdf

    def test_the_fm_row_names_its_channels(self, pdf):
        assert b"GFP, mCherry" in pdf

    def test_every_run_has_a_row(self, pdf):
        assert pdf.count(b"overview_sem") >= 3  # protocol line, aspen row, birch row
