"""The import dialog: confirm how a file is read, see it as it will look (FIB-1030).

The preview is the FM canvas itself, so these check that what the form says and what
the canvas shows stay one thing: the axis roles, the channels and their colours, the
mirror. And that the image built on Import is what the form said.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import tifffile

pytest.importorskip("PyQt5")

from fibsem.fm.reader import read_source
from fibsem.ui.widgets.fm_import_dialog import MANY_CHANNELS, ImportImageDialog

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def _destroy_widgets(destroy_widgets_after_test):
    """The FM canvas's channel panel is a top-level window."""


def planes(nc, nz, h=16, w=18):
    data = np.zeros((nc, nz, h, w), dtype=np.uint16)
    for c in range(nc):
        for z in range(nz):
            data[c, z] = (10 * c + z + 1) * 97 + np.arange(h * w).reshape(h, w) % 7
    return data


@pytest.fixture
def hyperstack(tmp_path):
    """An ImageJ stack, 3 channels by 8 z-slices stored z outside, 0.325 um."""
    path = str(tmp_path / "stack.ij.tiff")
    tifffile.imwrite(
        path,
        planes(3, 8).transpose(1, 0, 2, 3),
        imagej=True,
        resolution=(1 / 0.325, 1 / 0.325),
        metadata={"axes": "ZCYX", "unit": "micron"},
    )
    return path


@pytest.fixture
def screenshot(tmp_path):
    rgb = np.zeros((16, 18, 3), dtype=np.uint8)
    rgb[:8, :, 0] = 200  # red top half
    path = str(tmp_path / "shot.tif")
    tifffile.imwrite(path, rgb, photometric="rgb")
    return path


def _open(path):
    dialog = ImportImageDialog(read_source(path))
    dialog.show()
    return dialog


class TestTheFormStartsFromTheFile:
    def test_with_its_axes_pixel_size_and_channels(self, hyperstack):
        dialog = _open(hyperstack)
        assert dialog.roles == "ZCYX"
        assert dialog.pixel_size == pytest.approx(0.325e-6)
        assert "From the file" in dialog.label_pixel_size.text()
        assert len(dialog.channel_names) == 3
        assert dialog.btn_import.isEnabled()

    def test_an_unknown_pixel_size_is_a_guess_and_says_so(self, screenshot):
        dialog = _open(screenshot)
        assert dialog.pixel_size == pytest.approx(1e-6)
        assert "guess" in dialog.label_pixel_size.text()

    def test_an_rgb_file_is_red_green_and_blue(self, screenshot):
        dialog = _open(screenshot)
        assert dialog.roles == "YXC"
        assert dialog.channel_colors == ["red", "green", "blue"]
        assert not dialog.btn_swap.isEnabled()  # no z to swap with


class TestThePreviewIsTheImage:
    def test_the_canvas_shows_each_channel_in_its_colour(self, hyperstack):
        dialog = _open(hyperstack)
        layers = dialog.preview.layers
        assert [layer.name for layer in layers] == dialog.channel_names
        assert [layer.color for layer in layers] == dialog.channel_colors
        # Channels, not z-slices: each layer is one channel's projection.
        np.testing.assert_array_equal(layers[1].data, planes(3, 8)[1].max(axis=0))

    def test_a_colour_picked_in_the_form_reaches_the_canvas(self, hyperstack):
        dialog = _open(hyperstack)
        dialog._channel_rows[0][1].setCurrentText("yellow")
        assert dialog.preview.layers[0].color == "yellow"

    def test_a_colour_picked_on_the_canvas_reaches_the_form(self, hyperstack):
        """The canvas's own channel controls recolour too; the form follows, so
        what is saved is what is shown."""
        dialog = _open(hyperstack)
        panel = dialog.preview._panel
        panel.set_layers(dialog.preview.layers)
        panel._select_row(2)
        panel.colormap.setCurrentText("red")
        assert dialog.channel_colors[2] == "red"

    def test_a_mirror_mirrors_the_preview(self, screenshot):
        dialog = _open(screenshot)
        before = dialog.preview.layers[0].data.copy()
        dialog.check_flip.setChecked(True)
        np.testing.assert_array_equal(dialog.preview.layers[0].data, before[:, ::-1])


class TestCorrectingTheAxes:
    def test_swapping_makes_z_the_channels_and_warns(self, hyperstack):
        dialog = _open(hyperstack)
        dialog.btn_swap.click()
        assert dialog.roles == "CZYX"
        assert len(dialog.channel_names) == 8 > MANY_CHANNELS
        assert "swap channel and z" in dialog.label_axes.text()
        assert len(dialog.preview.layers) == 8
        dialog.btn_swap.click()
        assert dialog.roles == "ZCYX"
        assert dialog.label_axes.text() == ""
        assert len(dialog.preview.layers) == 3

    def test_roles_that_cannot_be_used_stop_the_import_and_say_why(self, hyperstack):
        dialog = _open(hyperstack)
        dialog.combo_roles[2].setCurrentIndex(3)  # Y -> X: two X, no Y
        assert not dialog.btn_import.isEnabled()
        assert "Y" in dialog.label_axes.text()
        dialog.combo_roles[2].setCurrentIndex(2)  # back to Y
        assert dialog.btn_import.isEnabled()

    def test_a_new_channel_axis_drops_the_old_channel_names(self, hyperstack):
        dialog = _open(hyperstack)
        dialog._channel_rows[0][0].setText("DAPI")
        dialog.btn_swap.click()
        assert "DAPI" not in dialog.channel_names


class TestImportBuildsWhatTheFormSays:
    def test_axes_pixel_size_channels_and_mirror(self, hyperstack):
        dialog = _open(hyperstack)
        dialog.spin_pixel_size.setValue(0.5)
        dialog._channel_rows[1][0].setText("GFP")
        dialog._channel_rows[1][1].setCurrentText("green")
        dialog.check_flip.setChecked(True)
        base = object()

        image = dialog.build(geometry="geometry", stage_position=base)

        np.testing.assert_array_equal(image.data, planes(3, 8)[..., ::-1])
        md = image.metadata
        assert md.pixel_size_x == pytest.approx(0.5e-6)
        assert (md.channels[1].name, md.channels[1].color) == ("GFP", "green")
        assert md.geometry == "geometry" and md.stage_position is base

    def test_an_emptied_name_falls_back_to_the_suggested_one(self, hyperstack):
        dialog = _open(hyperstack)
        dialog._channel_rows[0][0].setText("   ")
        assert dialog.channel_names[0] == "Channel-01"

    def test_closing_takes_the_canvas_channel_panel_with_it(self, hyperstack):
        dialog = _open(hyperstack)
        dialog.preview._btn_layers.click()
        assert dialog.preview._panel.isVisible()
        dialog.close()
        assert not dialog.preview._panel.isVisible()
