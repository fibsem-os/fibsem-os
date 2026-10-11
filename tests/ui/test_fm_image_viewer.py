"""The standalone FM Image Viewer — loading, the filmstrip, and what it displays.

This widget used to host a napari viewer and push one layer per channel, then a list
beside an ``FMCanvasWidget``. It is now the image viewer (FIB-1189) with an Open…
button: loaded files join its filmstrip, and because ``set_fm_image`` resets the channel
set on every call, *which* image is displayed is what these tests pin.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_fm_image_viewer.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there. Without this the
# module-level imports below turn a skip into a collection error.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.ui.fm.widgets.fm_image_viewer_widget import FMImageViewerWidget

_app = QApplication.instance() or QApplication(sys.argv)


def _channel(name: str, color: str) -> FluorescenceChannelMetadata:
    return FluorescenceChannelMetadata(
        name=name,
        excitation_wavelength=488.0,
        emission_wavelength=509.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=100.0,
        color=color,
    )


def _image(
    filepath: str | None = "/experiments/run/fm-stack-01.ome.tiff",
    channels=(("DAPI", "cyan"), ("GFP", "green")),
    z: int = 4,
    shape=(64, 96),
) -> FluorescenceImage:
    h, w = shape
    data = (np.random.default_rng(0).random((len(channels), z, h, w)) * 4095).astype(
        np.uint16
    )
    md = FluorescenceImageMetadata(
        acquisition_date="2026-08-12T15:00:00",
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=500e-9,
        resolution=(w, h),
        channels=[_channel(n, c) for n, c in channels],
        z_positions=[i * 500e-9 for i in range(z)],
    )
    return FluorescenceImage(data=data, metadata=md, filepath=filepath)


def test_the_viewer_holds_no_napari_viewer():
    """The whole point of the swap: display is the canvas, not a napari Viewer.

    Also guards the constructor signature — every caller stopped passing ``viewer``, and a
    reintroduced parameter would silently accept one again.
    """
    widget = FMImageViewerWidget()
    assert not hasattr(widget, "viewer"), "the viewer attribute came back"
    with pytest.raises(TypeError):
        FMImageViewerWidget(viewer=object())  # type: ignore[call-arg]


def test_the_viewer_carries_the_dark_theme_itself():
    """It opens parentless, and a Qt stylesheet only cascades to *children* — so it
    inherits nothing from the main window and must set the theme on itself.

    This shipped wrong once. Every standalone window in the app does this (the
    coincidence viewer, the task config editor, FibsemUI), but a rendering harness that
    styles the QApplication hides the omission completely: the widget looks correct in
    isolation and comes up light-on-dark inside the app.
    """
    widget = FMImageViewerWidget()
    assert widget.styleSheet(), (
        "no stylesheet — the sidebar renders light against the canvas"
    )


def test_the_load_dialog_carries_the_dark_theme_too():
    """A QDialog is its own top-level window: it does *not* pick up the stylesheet its
    parent carries, even though it is a child in the object tree (verified — the child's
    ``styleSheet()`` comes back empty).

    The dialog used to be styled by napari, which sets a stylesheet on the QApplication.
    Nothing does that now, so it has to carry the theme itself.
    """
    from fibsem.ui.fm.widgets.load_image_dialog import LoadImageDialog

    parent = FMImageViewerWidget()
    dialog = LoadImageDialog(parent)
    assert dialog.styleSheet(), "the load dialog renders light against a dark app"


def _filmstrip_texts(widget) -> list:
    return [b.text() for b in widget.image_viewer._film_buttons]


def test_it_is_the_image_viewer_with_open():
    widget = FMImageViewerWidget()
    viewer = widget.image_viewer
    assert widget.pushButton_load_image.text() == "Open…"
    header = viewer._header
    assert header.indexOf(widget.pushButton_load_image) < header.indexOf(
        viewer.export_button
    ), "Open… sits before Export"
    assert widget.canvas is viewer.fm_widget


def test_loading_appends_to_the_filmstrip_and_shows_the_new_image():
    widget = FMImageViewerWidget()
    first = _image(filepath="/a/first.ome.tiff")
    second = _image(filepath="/b/second.ome.tiff")

    widget.add_image(first)
    widget.add_image(second)

    assert _filmstrip_texts(widget) == ["first.ome.tiff", "second.ome.tiff"]
    assert widget.image_viewer.index == 1, "the newest is shown"
    assert widget.image_viewer.image is second
    assert widget.image_viewer.title_label.text() == "second.ome.tiff"
    assert widget.image_viewer.position_label.text() == "2 of 2"


def test_stepping_back_switches_which_image_is_displayed():
    """The filmstrip is the only thing choosing what the canvas shows, so a step that
    does not reach ``set_fm_image`` leaves the viewer showing the wrong file."""
    widget = FMImageViewerWidget()
    widget.add_image(_image(filepath="/a/first.ome.tiff", channels=(("DAPI", "cyan"),)))
    widget.add_image(
        _image(
            filepath="/b/second.ome.tiff", channels=(("GFP", "green"), ("RFP", "red"))
        )
    )

    assert [layer.name for layer in widget.canvas.layers] == ["GFP", "RFP"]

    widget.image_viewer.step(-1)
    assert [layer.name for layer in widget.canvas.layers] == ["DAPI"], (
        "stepping to an earlier image did not re-composite the canvas"
    )
    assert widget.image_viewer.title_label.text() == "first.ome.tiff"


def test_display_image_goes_to_an_image_already_held():
    widget = FMImageViewerWidget()
    first, second = (
        _image(filepath="/a/first.ome.tiff"),
        _image(filepath="/b/b.ome.tiff"),
    )
    widget.add_image(first)
    widget.add_image(second)
    widget.display_image(first)
    assert widget.image_viewer.index == 0
    assert len(widget.images) == 2, "showing a held image must not add it again"


def test_an_image_with_no_filepath_still_gets_a_name():
    """Images handed over in memory have no filepath; an empty tile is unusable."""
    widget = FMImageViewerWidget()
    image = _image(filepath=None)
    image.metadata.description = "acquired stack"

    widget.add_image(image)
    assert _filmstrip_texts(widget) == ["acquired stack"]

    bare = _image(filepath=None)
    bare.metadata.description = None
    widget.add_image(bare)
    assert _filmstrip_texts(widget)[1].strip(), "tile label was blank"


def test_the_bar_says_what_the_image_is():
    """The bar replaced the sidebar's summary: channels' objective, HFW, Z and when."""
    widget = FMImageViewerWidget()
    widget.resize(1600, 700)
    widget.show()
    widget.add_image(_image())  # 96 px wide at 100 nm, 4 planes at 500 nm
    bar = widget.image_viewer.fm_bar
    bar.resize(2000, bar.height())
    shown = {f.key: f.value for f in bar.visible_fields()}
    assert shown["hfw"] == "9.6 µm"
    assert shown["z"] == "MIP · 4 × 500 nm"
    assert bar.time_label.text(), "the acquisition time is shown"
    widget.close()


def test_a_sparsely_described_image_still_shows():
    """Files loaded from disk vary in what metadata they carry; a missing field must
    not take the display down."""
    widget = FMImageViewerWidget()
    image = _image()
    image.metadata.acquisition_date = None
    image.metadata.pixel_size_x = None

    widget.add_image(image)  # must not raise
    assert widget.image_viewer.image is image
    assert [layer.name for layer in widget.canvas.layers] == ["DAPI", "GFP"]
