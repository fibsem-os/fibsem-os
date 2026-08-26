"""Where things sit on the handoff page.

Its sibling, `test_handoff_map`, is about what the document is allowed to *say*. This is
about whether the page is readable at all, and every test here stands for something that
was wrong in output I looked at rather than something that seemed like it might be:

- `THICKNESS` broke across two lines and left a stray `S` under the header, and `15.0
  deg` split in half in the cell below it, because the columns were sized by counting
  characters -- which under-reads a word set in bold capitals against lower-case body
  text;
- every section heading sat 6pt to the right of the table beneath it, because
  `SimpleDocTemplate`'s frame carries 6pt of padding: a full-measure table is wider than
  that frame and gets centred back onto the margin, while a paragraph is laid inside the
  padding;
- the facts under a lamella's name were set in the same tiny uppercase as the table
  headers, which made a card look like a fragment of the table rather than a caption to
  a picture.

The left-edge tests check the invariant rather than the rendered coordinates -- a PDF
reader is not a dependency of this suite. The rendered positions were confirmed by hand:
with the frame padding zeroed every band starts at 44.64pt and ends at 550.64pt, which
is the margin, on both sides.
"""

import pytest

reportlab = pytest.importorskip("reportlab")

from reportlab.lib import colors  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.platypus import HRFlowable, PageBreak  # noqa: E402

from fibsem.applications.autolamella.structures import DefectType, Lamella  # noqa: E402
from fibsem.applications.autolamella.tools import handoff_document as hd  # noqa: E402
from fibsem.applications.autolamella.tools.handoff_map import (  # noqa: E402
    HandoffOptions,
    _map_legend,
    field_width,
)

COLUMNS = (
    "Lamella",
    "Last task",
    "Finished",
    "Angle",
    "Thickness",
    "Width",
    "Defect",
    "Note",
)
NUMERIC = ("Finished", "Angle", "Thickness", "Width")


def _rows(n: int = 2, note: str = "mitochondria cluster, good ice"):
    return [
        {
            "Lamella": f"0{i + 1}-fancy-mite",
            "Last task": "Acquire Fluorescence Image",
            "Finished": "2026-07-30 11:58",
            "Angle": "15.0 deg",
            "Thickness": "4000 nm",
            "Width": "10.0 um",
            "Defect": "-",
            "Note": note,
        }
        for i in range(n)
    ]


class TestColumnWidths:
    """Measured in points, from the strings that will actually be set."""

    def test_every_header_fits_on_one_line(self):
        """The defect this replaces: `THICKNESS` wrapping to leave a stray `S`."""
        rows = _rows()
        widths = hd.HandoffDocument._column_widths(COLUMNS, rows)
        for column, width in zip(COLUMNS, widths):
            needed = pdfmetrics.stringWidth(
                column.upper(), "Helvetica-Bold", hd._TH_SIZE
            )
            assert needed <= width, f"{column!r} header does not fit its column"

    def test_a_short_value_is_not_split(self):
        """`15.0 deg` came out as `15.0` over `deg`, which reads as two numbers."""
        rows = _rows()
        widths = dict(zip(COLUMNS, hd.HandoffDocument._column_widths(COLUMNS, rows)))
        for column in ("Angle", "Thickness", "Width", "Finished"):
            value = rows[0][column]
            needed = pdfmetrics.stringWidth(value, "Helvetica", hd._TD_SIZE)
            assert needed <= widths[column], f"{value!r} does not fit {column!r}"

    def test_the_columns_fill_the_measure_exactly(self):
        widths = hd.HandoffDocument._column_widths(COLUMNS, _rows())
        assert sum(widths) == pytest.approx(hd.CONTENT_WIDTH)

    def test_prose_gives_way_and_numbers_do_not(self):
        """A long note must cost the note column, not the measurements beside it."""
        short = dict(zip(COLUMNS, hd.HandoffDocument._column_widths(COLUMNS, _rows())))
        long = dict(
            zip(
                COLUMNS,
                hd.HandoffDocument._column_widths(
                    COLUMNS, _rows(note="stopped for time " * 12)
                ),
            )
        )
        assert long["Note"] > short["Note"]
        for column in ("Angle", "Thickness", "Width"):
            needed = pdfmetrics.stringWidth(
                column.upper(), "Helvetica-Bold", hd._TH_SIZE
            )
            assert long[column] >= needed

    def test_more_headers_than_the_page_is_wide_still_fits_the_page(self):
        """Nothing can be honoured here, but nothing may run past the margin either."""
        columns = tuple(f"A very long column heading {i}" for i in range(12))
        widths = hd.HandoffDocument._column_widths(
            columns, [dict.fromkeys(columns, "x")]
        )
        assert sum(widths) == pytest.approx(hd.CONTENT_WIDTH)


class TestOneLeftEdge:
    """Every band starts at the margin, because the frame adds nothing of its own."""

    def test_the_frame_has_no_padding(self, tmp_path):
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        frame = document.doc.pageTemplates[0].frames[0]
        assert (frame._leftPadding, frame._rightPadding) == (0, 0)
        assert (frame._topPadding, frame._bottomPadding) == (0, 0)

    def test_the_measure_is_the_frame(self, tmp_path):
        """`CONTENT_WIDTH` is what every table is built to. If the frame is narrower,
        those tables are centred back onto the margin while the paragraphs are not, and
        the two disagree by exactly the padding."""
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        assert document.doc.width == pytest.approx(hd.CONTENT_WIDTH)

    def test_a_full_measure_table_is_not_wider_than_the_frame(self, tmp_path):
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        document.table(COLUMNS, _rows(), numeric=NUMERIC)
        table = document.story[-1]
        width, _ = table.wrap(document.doc.width, document.doc.height)
        assert width <= document.doc.width + 1e-6


class TestBands:
    """A section that starts a page announces itself; one that does not, must not."""

    def test_a_band_on_a_new_page_opens_with_the_miniature_masthead(self, tmp_path):
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        document.band("Lamellae", new_page=True)
        assert isinstance(document.story[0], PageBreak)
        assert document.story[1].style.fontSize == document.s["section"].fontSize
        assert any(isinstance(f, HRFlowable) for f in document.story)

    def test_a_band_that_follows_on_breaks_nothing_and_rules_nothing(self, tmp_path):
        """Under the masthead its rule is already there, and a second one directly
        beneath reads as a printing fault rather than as a section."""
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        document.masthead("Experiment")
        before = len(document.story)
        document.band("Lamellae", new_page=False)
        added = document.story[before:]
        assert not any(isinstance(f, PageBreak) for f in added)
        assert not any(isinstance(f, HRFlowable) for f in added)


class TestTheFactLine:
    """The facts under a name are a caption, not a row of the table."""

    def _text(self, tmp_path, facts):
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        row = document._fact_line("01-fancy-mite", facts)
        return row._cellvalues[0][1].text

    def test_the_labels_are_sentence_case(self, tmp_path):
        text = self._text(tmp_path, [("Last task", "Polishing")])
        assert "Last task" in text
        assert "LAST TASK" not in text

    def test_the_value_is_the_emphasis(self, tmp_path):
        text = self._text(tmp_path, [("Thickness", "180 nm")])
        assert "<b>180 nm</b>" in text

    def test_a_flagged_lamella_is_coloured(self, tmp_path):
        text = self._text(tmp_path, [("Defect", "failed")])
        assert hd._hex(hd.DEFECT_FAILED) in text

    def test_an_unknown_is_faint_rather_than_absent(self, tmp_path):
        text = self._text(tmp_path, [("Thickness", "not milled")])
        assert "not milled" in text
        assert hd._hex(hd.INK_FAINT) in text

    def test_the_facts_are_larger_than_the_note_beneath_them(self, tmp_path):
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        assert document.s["fact"].fontSize > document.s["caption"].fontSize


class TestThePageNumber:
    def test_the_first_page_is_not_stamped_and_the_others_are(
        self, tmp_path, monkeypatch
    ):
        stamped = []

        class Spy(hd._NumberedCanvas):
            def drawRightString(self, x, y, text, *args, **kwargs):
                stamped.append(text)
                return super().drawRightString(x, y, text, *args, **kwargs)

        monkeypatch.setattr(hd, "_NumberedCanvas", Spy)
        document = hd.HandoffDocument(str(tmp_path / "x.pdf"))
        document.masthead("Experiment")
        document.band("Lamellae")
        document.table(COLUMNS, _rows(), numeric=NUMERIC)
        document.band("Lamella detail")
        document.note("something")
        document.build()

        assert stamped == ["page 2 of 3", "page 3 of 3"]


class TestTheLegend:
    """A key to the markers, in the markers' own colours."""

    def _lamella(self, tmp_path, state: DefectType, number: int = 1) -> Lamella:
        lam = Lamella(
            path=tmp_path / f"lam{number}", number=number, petname=f"0{number}-x"
        )
        if state is not DefectType.NONE:
            lam.defect.set_defect("because", state)
        return lam

    def test_only_the_states_on_the_grid_appear(self, tmp_path):
        lamellae = [self._lamella(tmp_path, DefectType.NONE)]
        labels = [
            label
            for _, label in _map_legend(("ION", 0, -23), lamellae, HandoffOptions())
        ]
        assert "no defect flagged" in labels
        assert "rework" not in labels
        assert "failed" not in labels

    def test_a_flagged_state_brings_its_own_line(self, tmp_path):
        lamellae = [
            self._lamella(tmp_path, DefectType.NONE, 1),
            self._lamella(tmp_path, DefectType.FAILURE, 2),
        ]
        labels = [
            label
            for _, label in _map_legend(("ION", 0, -23), lamellae, HandoffOptions())
        ]
        assert "failed" in labels
        assert "rework" not in labels

    def test_the_swatch_is_the_marker_colour(self, tmp_path):
        """Not a colour picked to look better on paper: a key that does not match the
        map is worse than no key."""
        from fibsem.imaging.tiled import DEFECT_FAILURE_COLOUR

        lamellae = [self._lamella(tmp_path, DefectType.FAILURE, 2)]
        options = HandoffOptions(marker_color="cyan")
        entries = dict(
            (label, colour)
            for colour, label in _map_legend(("ION", 0, -23), lamellae, options)
        )
        assert entries["failed"] == colors.toColor(DEFECT_FAILURE_COLOUR)

    def test_the_view_label_carries_no_swatch(self, tmp_path):
        """It is a caption, not a key entry."""
        lamellae = [self._lamella(tmp_path, DefectType.NONE)]
        colour, label = _map_legend(("ION", 0, -23), lamellae, HandoffOptions())[-1]
        assert colour is None
        assert "Ion beam" in label


class TestTheFieldWidth:
    """What the label under a picture says about its scale."""

    def test_it_prefers_the_recorded_field_width(self):
        from fibsem.structures import FibsemImage

        image = FibsemImage.generate_blank_image(resolution=(600, 400), hfw=100e-6)
        assert field_width(image) == "fov 100 um"

    def test_it_falls_back_to_the_pixel_size(self):
        class Metadata:
            image_settings = None
            pixel_size_x = 0.25e-6

        class Image:
            metadata = Metadata()
            data = __import__("numpy").zeros((400, 600))

        assert field_width(Image()) == "fov 150 um"

    def test_an_image_that_cannot_say_says_nothing(self):
        class Image:
            metadata = None
            data = None

        assert field_width(Image()) == ""
