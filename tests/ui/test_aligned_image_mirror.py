"""Mirroring an image laid over the Overview tab (FIB-1030).

An image from another microscope may have been recorded mirrored, and no turn lines a
mirror image up. The Mirror button flips it left to right about its own axis -- one axis
is enough, since a top-to-bottom flip is that and a half turn -- as part of the user's
placement: it keeps the image where it is, turned as it was, and is saved with the grid.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

import fibsem.config as fibsem_config
from fibsem import utils
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, ImageSettings
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

SQUARE = ("SEM", BeamType.ELECTRON)
FORESHORTENED = ("MILLING", BeamType.ION)
HEIGHT, WIDTH = 40, 60


@pytest.fixture(scope="module")
def microscope():
    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


@pytest.fixture(autouse=True)
def _destroy_widgets(destroy_widgets_after_test):
    """See tests/ui/test_overview_widget.py: the widget leaves top-levels behind."""


def _beam_image(scope, orientation, beam_type):
    pose = scope.get_orientation(orientation)
    hfw = 128 * 2e-6
    image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=hfw)
    state = scope.get_microscope_state(beam_type=beam_type)
    state.stage_position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam_type)
    image.metadata.microscope_state = state
    image.metadata.system_info = scope.system.info
    image.metadata.hardware_geometry = scope.hardware_geometry()
    return image


def _fm_image(scope):
    pose = scope.get_orientation("FM")
    data = (np.random.default_rng(1).random((1, 1, HEIGHT, WIDTH)) * 4000).astype(
        np.uint16
    )
    image = FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-09-24T10:00:00",
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            stage_position=FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t),
            channels=[
                FluorescenceChannelMetadata(
                    name="GFP",
                    excitation_wavelength=488.0,
                    power=0.5,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                )
            ],
        ),
    )
    image.metadata.geometry = scope.fm_image_geometry()
    return image


def _widget(scope, view=SQUARE):
    w = FibsemOverviewWidget(scope)
    w.resize(900, 700)
    w.set_image(_beam_image(scope, *view))
    key = w.add_aligned_image(_fm_image(scope), label="fm")
    return w, key


PIXELS = [(3.0, 4.0), (50.0, 7.0), (20.0, 33.0), (57.0, 36.0)]


def _where(w, key, pixels=PIXELS):
    return np.array([w.aligned_images.pixel_to_canvas(key, x, y) for x, y in pixels])


class TestMirroring:
    @pytest.mark.parametrize("view", [SQUARE, FORESHORTENED])
    def test_flips_the_image_about_its_own_axis_where_it_stands(self, microscope, view):
        """Mirrored, pixel (x, y) lands where (W-1-x, y) did: left to right about the
        image's own axis, whatever the turn and whatever the view's squash."""
        w, key = _widget(microscope, view)
        images = w.aligned_images
        images.set_placement(key, 5e-6, -3e-6, 25.0)
        record = images.get(key)
        centre, rotation = record.overlay.centre, record.overlay.rotation
        opposite = [(WIDTH - 1 - x, y) for x, y in PIXELS]
        before = _where(w, key, opposite)

        images.set_mirrored(key, True)

        np.testing.assert_allclose(_where(w, key), before, atol=1e-6)
        assert record.overlay.centre == pytest.approx(centre)
        assert record.overlay.rotation == pytest.approx(rotation)
        assert record.placement == (5e-6, -3e-6, 25.0, 1.0)
        w.close()

    def test_twice_is_as_it_was(self, microscope):
        w, key = _widget(microscope)
        before = _where(w, key)
        w.aligned_images.set_mirrored(key, True)
        w.aligned_images.set_mirrored(key, False)
        np.testing.assert_allclose(_where(w, key), before, atol=1e-9)
        w.close()

    def test_is_announced_and_forgets_a_fit_made_the_other_way_round(self, microscope):
        w, key = _widget(microscope)
        heard = []
        w.aligned_images.placement_changed.connect(heard.append)
        w.aligned_images.get(key).fit = {"rms": 1.0}
        w.aligned_images.set_mirrored(key, True)
        assert heard == [key]
        assert w.aligned_images.get(key).fit == {}
        w.aligned_images.set_mirrored(key, True)  # no change, no announcement
        assert heard == [key]
        w.close()

    def test_reset_undoes_it(self, microscope):
        w, key = _widget(microscope)
        w.aligned_images.set_mirrored(key, True)
        w.aligned_images.reset(key)
        assert not w.aligned_images.get(key).mirrored
        w.close()

    def test_a_mirrored_image_can_then_be_fitted_exactly(self, microscope):
        """The point: a turn cannot line up a mirror image, and after Mirror the fit
        can. Targets are where the pixels would be mirrored, turned and moved."""
        w, key = _widget(microscope)
        images = w.aligned_images
        images.set_placement(key, 4e-6, 2e-6, 30.0, mirrored=True)
        targets = _where(w, key)
        images.set_placement(key, 0.0, 0.0, 0.0, mirrored=False)

        unmirrored = images.fit_to_points(key, PIXELS, targets)
        images.set_placement(key, 0.0, 0.0, 0.0)
        images.set_mirrored(key, True)
        mirrored = images.fit_to_points(key, PIXELS, targets)

        assert unmirrored.rms > 5.0
        assert mirrored.rms == pytest.approx(0.0, abs=1e-6)
        w.close()


class TestTheButton:
    def test_mirrors_the_selected_image_and_says_so(self, microscope):
        w, key = _widget(microscope)
        panel = w.aligned_image_panel
        panel.btn_mirror.click()
        assert w.aligned_images.get(key).mirrored
        assert "mirrored" in panel.label_placement.text()
        panel.btn_mirror.click()
        assert not w.aligned_images.get(key).mirrored
        assert "mirrored" not in panel.label_placement.text()
        w.close()

    def test_shows_the_selected_images_state(self, microscope):
        w, first = _widget(microscope)
        second = w.add_aligned_image(_fm_image(microscope), label="second")
        panel = w.aligned_image_panel
        w.aligned_images.set_mirrored(first, True)
        panel.combo.setCurrentIndex(panel.combo.findData(first))
        assert panel.btn_mirror.isChecked()
        panel.combo.setCurrentIndex(panel.combo.findData(second))
        assert not panel.btn_mirror.isChecked()
        w.close()

    def test_reset_unchecks_it(self, microscope):
        w, key = _widget(microscope)
        panel = w.aligned_image_panel
        panel.btn_mirror.click()
        panel.btn_reset.click()
        assert not panel.btn_mirror.isChecked()
        w.close()

    def test_a_restored_mirror_is_shown_and_not_announced(self, microscope):
        w, key = _widget(microscope)
        heard = []
        w.image_placement_changed.connect(heard.append)
        w.set_aligned_image_placement(key, 1e-6, 0.0, 0.0, mirrored=True)
        assert w.aligned_images.get(key).mirrored
        assert w.aligned_image_panel.btn_mirror.isChecked()
        assert heard == []
        w.close()


class TestTheFitDialogNoticesAMirror:
    """The hint, and the points kept for the next go."""

    @staticmethod
    def _reference_pixels(w, key, canvas_points):
        """Canvas points as pixels of the overview the fit dialog shows."""
        canvas_key, tile = w._reference_tile_for_fit(key)
        (cx, cy), (width_m, height_m) = w._extents[canvas_key]
        height, width = tile.grey.shape[:2]
        out = []
        for X, Y in canvas_points:
            mx, my = w.canvas.canvas_to_metres(X, Y)
            out.append(
                (
                    (mx - (cx - width_m / 2)) / width_m * width - 0.5,
                    (my - (cy - height_m / 2)) / height_m * height - 0.5,
                )
            )
        return out

    def test_says_so_keeps_the_points_and_goes_quiet_once_mirrored(
        self, microscope, monkeypatch
    ):
        from PyQt5.QtWidgets import QDialog

        from fibsem.ui.widgets.image_fit_dialog import ImageFitDialog

        w, key = _widget(microscope)
        images = w.aligned_images
        # Where the features would be were the image mirrored, turned and moved.
        images.set_placement(key, 3e-6, 1e-6, 20.0, mirrored=True)
        targets = self._reference_pixels(w, key, _where(w, key))
        images.set_placement(key, 0.0, 0.0, 0.0, mirrored=False)

        seen = []

        def exec_(dialog):
            if not seen:  # the first time only: the user clicks the pairs
                for (px, py), (rx, ry) in zip(PIXELS, targets):
                    dialog.add_pair((px, py), (rx, ry))
            seen.append((len(dialog.pairs()), dialog.label_hint.text()))
            return QDialog.Rejected

        monkeypatch.setattr(ImageFitDialog, "exec_", exec_)

        w._fit_aligned_image(key)
        assert "mirrored" in seen[0][1]

        w.aligned_image_panel.btn_mirror.click()
        w._fit_aligned_image(key)
        assert seen[1] == (len(PIXELS), "")  # the points kept, and no hint now
        w.close()

    def test_the_points_are_forgotten_with_the_image(self, microscope, monkeypatch):
        from PyQt5.QtWidgets import QDialog

        from fibsem.ui.widgets.image_fit_dialog import ImageFitDialog

        w, key = _widget(microscope)

        def exec_(dialog):
            dialog.add_pair((1.0, 1.0), (2.0, 2.0))
            return QDialog.Rejected

        monkeypatch.setattr(ImageFitDialog, "exec_", exec_)
        w._fit_aligned_image(key)
        assert key in w._fit_pairs
        w.remove_aligned_image(key)
        assert key not in w._fit_pairs
        w.close()
