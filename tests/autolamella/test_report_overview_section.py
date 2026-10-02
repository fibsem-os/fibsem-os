"""The PDF report's overview page survives an overview it cannot read.

The section drew every overview inside one ``try``. An unreadable file -- an
interrupted copy, say -- that sorted first lost every overview after it, and the page
kept its "Overview (Positions)" heading with nothing under it. The only trace was a
warning in the log. Reproduced on a copy of a real experiment: two overviews on disk,
no figures in the report.

Needs the `reporting` extra (reportlab), which CI does not install, so this runs
locally and skips there -- as the grid report's PDF tests do.
"""

import os

import numpy as np
import pytest

pytest.importorskip("reportlab")

from reportlab.platypus import Image, Paragraph  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.poses import build_lamella_poses  # noqa: E402
from fibsem.applications.autolamella.structures import Experiment, Lamella  # noqa: E402
from fibsem.applications.autolamella.tools.reporting import (  # noqa: E402
    PDFReportGenerator,
    _add_overview_section,
)
from fibsem.structures import BeamType, FibsemImage, ImageSettings  # noqa: E402


@pytest.fixture(scope="module")
def microscope():
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "microscope-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


@pytest.fixture
def experiment(microscope, tmp_path):
    """One lamella, and one good overview with the metadata to place it."""
    experiment = Experiment(path=tmp_path, name="report-overview-test")
    os.makedirs(str(experiment.path), exist_ok=True)

    state = microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
    image = FibsemImage.generate_blank_image(resolution=(96, 32), hfw=900e-6)
    image.data = np.zeros((32, 96), dtype=np.uint8)
    image.metadata.image_settings = ImageSettings(
        hfw=900e-6, beam_type=BeamType.ELECTRON
    )
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    image.save(os.path.join(str(experiment.path), "overview-image-good.tif"))

    lamella = Lamella(
        petname="01-a", path=os.path.join(str(experiment.path), "01-a"), number=1
    )
    lamella.milling_pose = build_lamella_poses(microscope, state.stage_position).milling
    experiment.positions.append(lamella)
    return experiment


def _unreadable_overview_first(experiment) -> str:
    good = os.path.join(str(experiment.path), "overview-image-good.tif")
    bad = os.path.join(str(experiment.path), "overview-image-bad.tif")
    with open(bad, "wb") as f:
        f.write(b"not a tiff")
    older = os.path.getmtime(good) - 3600
    os.utime(bad, (older, older))  # oldest, so it is drawn first
    return bad


def _section(experiment, tmp_path):
    pdf = PDFReportGenerator(str(tmp_path / "report.pdf"))
    _add_overview_section(pdf, experiment)
    images = [f for f in pdf.story if isinstance(f, Image)]
    text = " ".join(f.text for f in pdf.story if isinstance(f, Paragraph))
    return pdf, images, text


def test_one_unreadable_overview_does_not_cost_the_others(experiment, tmp_path):
    _unreadable_overview_first(experiment)

    _, images, _ = _section(experiment, tmp_path)

    assert len(images) == 1, "the readable overview was not drawn"


def test_the_page_says_which_overview_it_could_not_draw(experiment, tmp_path):
    bad = _unreadable_overview_first(experiment)

    _, _, text = _section(experiment, tmp_path)

    assert f"Could not draw the overview {os.path.basename(bad)}" in text


def test_the_report_still_builds(experiment, tmp_path):
    """The note is ordinary page content: the PDF is written with it in."""
    _unreadable_overview_first(experiment)

    pdf, _, _ = _section(experiment, tmp_path)
    pdf.generate()

    assert os.path.getsize(tmp_path / "report.pdf") > 0


def test_an_experiment_without_overviews_gets_no_page(experiment, tmp_path):
    os.remove(os.path.join(str(experiment.path), "overview-image-good.tif"))

    pdf, images, text = _section(experiment, tmp_path)

    assert images == [] and "Overview" not in text
