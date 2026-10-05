"""Saving from the Generate Overview Image dialog writes the overview's shape.

The Save button used to write the preview's own figure. A FigureCanvas resizes its
figure to the widget, so the saved page took the dialog's shape: a 3:1 overview saved
from a 3:2 dialog filled about half the page height (55% on a real experiment), with a
blank band between the title and the image. These drive the dialog's own Save handler,
since that is the path a user takes.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_image_export.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PyQt5.QtWidgets import QApplication, QFileDialog  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.poses import build_lamella_poses  # noqa: E402
from fibsem.applications.autolamella.structures import Experiment, Lamella  # noqa: E402
from fibsem.applications.autolamella.ui import (  # noqa: E402
    autolamella_overview_image_widget as overview_dialog,
)
from fibsem.structures import BeamType, FibsemImage, ImageSettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

_WIDE = (256, 768)  # (height, width), 3:1


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
def widget(microscope, tmp_path):
    """The dialog's widget, opened on a real experiment with one lamella on a 3:1
    overview, found the way the dialog finds it: by filename in the experiment folder."""
    experiment = Experiment(path=tmp_path, name="overview-export-test")
    os.makedirs(str(experiment.path), exist_ok=True)

    image = FibsemImage.generate_blank_image(
        resolution=(_WIDE[1], _WIDE[0]), hfw=900e-6
    )
    image.data = np.full(_WIDE, 90, dtype=np.uint8)
    image.metadata.image_settings = ImageSettings(
        hfw=900e-6, beam_type=BeamType.ELECTRON
    )
    state = microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    image.save(os.path.join(str(experiment.path), "overview-image.tif"))

    lamella = Lamella(
        petname="01-test", path=os.path.join(str(experiment.path), "01-test"), number=1
    )
    lamella.milling_pose = build_lamella_poses(microscope, state.stage_position).milling
    experiment.positions.append(lamella)

    dialog = overview_dialog.create_overview_image_widget(experiment)
    dialog.resize(1200, 800)  # 3:2, so the dialog's shape and the overview's differ
    dialog.show()
    QApplication.processEvents()
    w = dialog.findChild(overview_dialog.OverviewImageWidget)
    assert w.overview_image is not None, "the dialog did not find the overview"
    w._on_preview_clicked()
    QApplication.processEvents()
    yield w
    dialog.close()


def _save(widget, tmp_path, monkeypatch) -> np.ndarray:
    out = str(tmp_path / "exported.png")
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (out, "png"))
    )
    widget._on_save_clicked()
    assert os.path.exists(out), "nothing was saved"
    return np.asarray(Image.open(out).convert("L"))


def _image_fraction_of_height(page: np.ndarray) -> float:
    """How much of the page's height the overview occupies (its rows are not white)."""
    rows = (page < 245).mean(axis=1) > 0.5
    return rows.sum() / page.shape[0]


def test_the_exported_page_is_the_overview_s_shape_not_the_dialog_s(
    widget, tmp_path, monkeypatch
):
    """Measured on main before the fix: 55% on a real 3:1 experiment."""
    page = _save(widget, tmp_path, monkeypatch)
    assert _image_fraction_of_height(page) > 0.7


def test_a_zoomed_preview_exports_the_zoomed_region(widget, tmp_path, monkeypatch):
    """The export is a fresh figure, so the view has to be carried across to it."""
    widget.title_textbox.setText("")  # a title adds height; measure the image alone
    ax = widget.current_figure.axes[0]
    ax.set_xlim(0, _WIDE[0])  # a square region of the 3:1 overview
    ax.set_ylim(_WIDE[0], 0)

    page = _save(widget, tmp_path, monkeypatch)
    height, width = page.shape
    assert width / height == pytest.approx(1.0, abs=0.15)


def test_saving_leaves_the_preview_as_it_was(widget, tmp_path, monkeypatch):
    """The title is drawn black for the page without recolouring the preview's."""
    widget.title_textbox.setText("Grid A")
    widget._on_preview_clicked()
    preview = widget.current_figure
    colour = preview._suptitle.get_color()

    _save(widget, tmp_path, monkeypatch)

    assert widget.current_figure is preview
    assert preview._suptitle.get_color() == colour
