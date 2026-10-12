"""The PDF report draws its figures in the light plot theme (FIB-1292).

The report's plotting functions are shared with on-screen views that are dark,
so the theme is the report's to set, around each section. These tests replace
the plotting functions with stubs that record the matplotlib defaults in force
when the report calls them, and check the figure reaches the PDF before the
theme is left.
"""

from types import SimpleNamespace

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

# The report is built with reportlab, an optional extra the build jobs do not install.
pytest.importorskip("reportlab")

import fibsem.applications.autolamella.tools.reporting as reporting  # noqa: E402
from fibsem.theme.plots import LIGHT  # noqa: E402


class _PDF:
    def __init__(self):
        self.saved_in = []

    def add_page_break(self):
        pass

    def add_heading(self, *a, **k):
        pass

    def add_paragraph(self, *a, **k):
        pass

    def add_dataframe(self, *a, **k):
        pass

    def add_mpl_figure(self, fig, **k):
        # the background a savefig here would get
        self.saved_in.append(matplotlib.rcParams["savefig.facecolor"])


@pytest.fixture
def drawn_in(monkeypatch):
    seen = []

    def stub(*a, **k):
        seen.append(matplotlib.rcParams["axes.facecolor"])
        return Figure()

    monkeypatch.setattr(reporting, "plot_lamella_task_workflow_summary", stub)
    monkeypatch.setattr(
        reporting, "plot_task_milling_summary", lambda *a, **k: [stub()]
    )
    monkeypatch.setattr(reporting, "generate_final_overview_image", stub)
    return seen


def test_a_lamellas_figures_are_drawn_and_saved_light(drawn_in):
    pdf = _PDF()
    empty = pd.DataFrame({"Lamella Name": []})
    reporting._add_lamella_section(pdf, SimpleNamespace(name="lam"), empty, empty)

    assert drawn_in == [LIGHT.background, LIGHT.background]
    assert pdf.saved_in == [LIGHT.background, LIGHT.background]


def test_the_overview_is_drawn_and_saved_light(drawn_in, monkeypatch, tmp_path):
    monkeypatch.setattr(reporting, "find_overviews", lambda path: [tmp_path / "ov.tif"])
    monkeypatch.setattr(reporting.FibsemImage, "load", lambda *a, **k: object())
    pdf = _PDF()
    reporting._add_overview_section(pdf, SimpleNamespace(path=tmp_path))

    assert drawn_in == [LIGHT.background]
    assert pdf.saved_in == [LIGHT.background]


def test_the_theme_ends_with_the_section(drawn_in):
    before = matplotlib.rcParams["axes.facecolor"]
    empty = pd.DataFrame({"Lamella Name": []})
    reporting._add_lamella_section(_PDF(), SimpleNamespace(name="lam"), empty, empty)
    assert matplotlib.rcParams["axes.facecolor"] == before
