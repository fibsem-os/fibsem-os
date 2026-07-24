"""Tests for FM z-stack interpolation (FIB-253).

Pure-array + FluorescenceImage-level; no Qt. Uses small synthetic volumes so it
runs fast and deterministically, with the numbers chosen to mirror the real
METEOR case (500 nm z step, 130 nm XY -> isotropic).
"""
import numpy as np
import pytest

from fibsem.correlation.util import (
    interpolate_fm_volume,
    interpolate_z_stack,
    multi_channel_interpolation,
)


def _fake_fm(nc=2, nz=21, ny=8, nx=8, z_step=500e-9, xy=130e-9, z_positions=True):
    """A minimal FluorescenceImage stand-in with the fields interpolation reads."""
    from types import SimpleNamespace

    # a ramp along z so interpolation has structure to preserve
    data = np.empty((nc, nz, ny, nx), dtype=np.float32)
    for z in range(nz):
        data[:, z] = float(z)
    positions = None
    if z_positions:
        # channel-major flat ramp, mirroring the real files (len = nc * nz)
        positions = list(np.arange(nc * nz, dtype=float) * z_step)
    meta = SimpleNamespace(
        pixel_size_x=xy,
        pixel_size_y=xy,
        pixel_size_z=z_step,
        z_positions=positions,
    )
    return SimpleNamespace(data=data, metadata=meta)


# ---------------------------------------------------------------------------
# Array-level
# ---------------------------------------------------------------------------


def test_interpolate_z_stack_only_scales_z():
    img = np.random.rand(10, 6, 6).astype(np.float32)
    out = interpolate_z_stack(img, pixelsize_in=500e-9, pixelsize_out=250e-9)
    assert out.shape[1:] == img.shape[1:]  # XY untouched
    assert out.shape[0] == 20  # z doubled (500 -> 250)


def test_multi_channel_progress_callback_is_ui_agnostic():
    """The algorithm reports progress through a plain callable, not a Qt object."""
    img = np.random.rand(3, 5, 4, 4).astype(np.float32)
    calls = []
    multi_channel_interpolation(
        img, 500e-9, 250e-9, progress_callback=lambda done, total: calls.append((done, total))
    )
    # one call before the loop, one after each of the 3 channels
    assert calls == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_multi_channel_runs_without_a_callback():
    img = np.random.rand(2, 5, 4, 4).astype(np.float32)
    out = multi_channel_interpolation(img, 500e-9, 250e-9)  # no callback
    assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# Volume-level: interpolate_fm_volume
# ---------------------------------------------------------------------------


def test_interpolate_fm_volume_makes_it_isotropic():
    fm = _fake_fm(nc=2, nz=21, z_step=500e-9, xy=130e-9)
    out = interpolate_fm_volume(fm, target_z_size_m=130e-9)

    assert out.data.shape[0] == 2  # channels preserved
    assert out.data.shape[2:] == fm.data.shape[2:]  # XY preserved
    # 21 * (500/130) = 80.8 -> scipy rounds to 81 slices
    assert out.data.shape[1] == 81


def test_pixel_size_z_matches_the_actual_slice_ratio_not_the_nominal():
    """scipy rounds the slice count, so pixel_size_z must come from the achieved
    ratio (old_nz/new_nz) — otherwise metadata and data drift apart."""
    fm = _fake_fm(nz=21, z_step=500e-9, xy=130e-9)
    out = interpolate_fm_volume(fm, target_z_size_m=130e-9)

    new_nz = out.data.shape[1]  # 81
    expected = 500e-9 * 21 / new_nz  # ~129.6 nm, NOT the nominal 130
    assert out.metadata.pixel_size_z == pytest.approx(expected)
    assert out.metadata.pixel_size_z != pytest.approx(130e-9)  # the nominal is wrong


def test_physical_depth_is_preserved_under_the_matched_rescale():
    """A coordinate at slice k, rescaled by new_nz/old_nz, keeps its physical
    depth (z_index * pixel_size_z) — the property the whole feature rests on."""
    fm = _fake_fm(nz=21, z_step=500e-9, xy=130e-9)
    old_nz = fm.data.shape[1]
    out = interpolate_fm_volume(fm, target_z_size_m=130e-9)
    new_nz = out.data.shape[1]

    scale = new_nz / old_nz
    for k in (0, 7, 15, 20):
        depth_before = k * fm.metadata.pixel_size_z
        depth_after = (k * scale) * out.metadata.pixel_size_z
        assert depth_after == pytest.approx(depth_before)


def test_z_positions_length_scales_with_slice_count():
    fm = _fake_fm(nc=3, nz=21, z_step=500e-9)  # 63 positions, like the real file
    assert len(fm.metadata.z_positions) == 63
    out = interpolate_fm_volume(fm, target_z_size_m=130e-9)
    new_nz = out.data.shape[1]  # 81
    # resampled proportionally: 63 * 81/21 = 243
    assert len(out.metadata.z_positions) == round(63 * new_nz / 21)


def test_does_not_mutate_the_source_image():
    fm = _fake_fm(nz=21)
    before_shape = fm.data.shape
    before_z = fm.metadata.pixel_size_z
    interpolate_fm_volume(fm, target_z_size_m=130e-9)
    assert fm.data.shape == before_shape
    assert fm.metadata.pixel_size_z == before_z  # metadata deep-copied, not edited


def test_single_plane_volume_is_rejected():
    fm = _fake_fm(nz=1, z_step=None, z_positions=False)
    with pytest.raises(ValueError, match="single plane"):
        interpolate_fm_volume(fm, target_z_size_m=130e-9)


def test_non_positive_target_is_rejected():
    fm = _fake_fm(nz=10)
    with pytest.raises(ValueError, match="positive"):
        interpolate_fm_volume(fm, target_z_size_m=0.0)


def test_returns_a_fluorescence_image():
    from fibsem.fm.structures import FluorescenceImage

    fm = _fake_fm(nz=10)
    out = interpolate_fm_volume(fm, target_z_size_m=130e-9)
    assert isinstance(out, FluorescenceImage)


# ---------------------------------------------------------------------------
# Dialog (headless Qt)
# ---------------------------------------------------------------------------

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _real_fm(crop=64):
    """A concrete FluorescenceImage the dialog/widget can consume, or skip.

    Cropped in XY by default: interpolation is z-only, so a small frame keeps the
    slice counts, pixel sizes, and the physical-depth invariant identical while
    the 2048x2048 -> would-be 2 GB / ~36 s full run stays out of the test suite.
    """
    import copy

    from fibsem.fm.structures import FluorescenceImage

    path = os.path.join(
        os.getcwd(), "tmp", "BeforeMilling_G1_-Feature-2-Active-001.ome.tiff"
    )
    if not os.path.exists(path):
        pytest.skip("tmp/ FM volume not present")
    fm = FluorescenceImage.load(path)
    if crop:
        fm = FluorescenceImage(
            data=fm.data[:, :, :crop, :crop].copy(),
            metadata=copy.deepcopy(fm.metadata),
        )
    return fm


def test_dialog_defaults_to_isotropic(qapp):
    from fibsem.correlation.ui.widgets.fm_interpolate_dialog import InterpolateZDialog

    fm = _real_fm()
    dlg = InterpolateZDialog(fm)
    assert dlg._chk_iso.isChecked()  # isotropic on by default
    assert not dlg._spin_target.isEnabled()  # driven by the checkbox
    assert dlg.target_z_size_m() == pytest.approx(fm.metadata.pixel_size_x)
    assert dlg.method() == "linear"
    assert "→ 81 slices" in dlg._preview.text()


def test_dialog_unchecking_isotropic_frees_the_spinbox(qapp):
    from fibsem.correlation.ui.widgets.fm_interpolate_dialog import InterpolateZDialog

    fm = _real_fm()
    dlg = InterpolateZDialog(fm)
    dlg._chk_iso.setChecked(False)
    assert dlg._spin_target.isEnabled()
    dlg._spin_target.setValue(250.0)
    assert dlg.target_z_size_m() == pytest.approx(250e-9)


def test_enter_does_not_run_the_interpolation(qapp):
    """Pressing Enter while editing must not fire the (tens-of-seconds) run — only
    a real click on Interpolate accepts. Escape still cancels."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QDialog

    from fibsem.correlation.ui.widgets.fm_interpolate_dialog import InterpolateZDialog

    dlg = InterpolateZDialog(_real_fm())
    dlg.show()
    QTest.keyClick(dlg, Qt.Key_Return)
    QTest.keyClick(dlg, Qt.Key_Enter)
    assert dlg.result() != QDialog.Accepted  # Enter did not accept
    assert not dlg.isHidden()  # still open

    QTest.keyClick(dlg, Qt.Key_Escape)
    assert dlg.isHidden()  # Escape still cancels

    dlg2 = InterpolateZDialog(_real_fm())
    dlg2.accept()  # the click path still accepts
    assert dlg2.result() == QDialog.Accepted


# ---------------------------------------------------------------------------
# Widget wiring
# ---------------------------------------------------------------------------


def _widget(qapp):
    import fibsem.correlation.ui.widgets.refractive_index_widget as riw

    riw._ensure_lut = lambda: None  # never hit the network for the LUT
    from fibsem.correlation.ui.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def test_interpolate_button_lives_in_images_tab_and_needs_a_zstack(qapp):
    w = _widget(qapp)
    btn = w._images_tab._btn_interpolate  # under the FM load controls, not the canvas
    assert btn.isEnabled() is False  # no image yet
    w.set_fm_image(_real_fm())
    assert btn.isEnabled() is True  # 21-slice stack


def test_interpolating_shows_embedded_progress_and_locks_the_button(qapp):
    """Progress is a non-modal bar in the Images tab; the button locks while it
    runs, and neither blocks the rest of the GUI."""
    w = _widget(qapp)
    w.set_fm_image(_real_fm())
    tab = w._images_tab
    # isHidden(), not isVisible(): the latter is False for any child of an unshown
    # top-level window, so it can't distinguish our explicit hide.
    assert tab._interp_progress.isHidden() is True  # hidden until running

    tab.set_interpolating(True, n_channels=3)
    assert tab._interp_progress.isHidden() is False
    assert tab._btn_interpolate.isEnabled() is False  # locked during the run
    tab.set_interpolation_progress(2, 3)
    assert tab._interp_progress.value() == 2

    tab.set_interpolating(False)
    assert tab._interp_progress.isHidden() is True
    assert tab._btn_interpolate.isEnabled() is True  # re-armed


def test_adopt_interpolated_volume_preserves_physical_depth(qapp):
    """The end-to-end widget behaviour: after adopting the resampled volume, every
    FM point keeps its physical depth and pixel_size_z propagates to the tab."""
    from fibsem.correlation.structures import PointType
    from fibsem.correlation.util import interpolate_fm_volume

    w = _widget(qapp)
    fm = _real_fm()
    w.set_fm_image(fm)
    old_nz = fm.data.shape[1]
    old_z_step = fm.metadata.pixel_size_z

    w._on_canvas_add_requested(20.0, 20.0, PointType.FM)
    w._coords_tab.fm_list.coordinates[0].point.z = 15.0
    w._on_canvas_add_requested(30.0, 30.0, PointType.POI)
    w._coords_tab.poi_list.coordinates[0].point.z = 11.0
    assert w._fm_point_count() == 2

    depth_before = 11.0 * old_z_step

    new_image = interpolate_fm_volume(fm, fm.metadata.pixel_size_x, "linear")
    w._adopt_interpolated_volume(new_image, old_nz)

    new_z = w._coords_tab.poi_list.coordinates[0].point.z
    depth_after = new_z * new_image.metadata.pixel_size_z
    assert depth_after == pytest.approx(depth_before)  # matched pair holds
    assert new_z != pytest.approx(11.0)  # it actually moved
    assert w._fm_pixel_size_z() == pytest.approx(new_image.metadata.pixel_size_z)


def test_rescale_only_touches_fm_side_points(qapp):
    from fibsem.correlation.structures import PointType

    w = _widget(qapp)
    w.set_fm_image(_real_fm())
    w._on_canvas_add_requested(10.0, 10.0, PointType.FIB)  # FIB side
    w._coords_tab.fib_list.coordinates[0].point.z = 5.0
    w._on_canvas_add_requested(20.0, 20.0, PointType.FM)  # FM side
    w._coords_tab.fm_list.coordinates[0].point.z = 5.0

    w._rescale_fm_z(2.0)
    assert w._coords_tab.fib_list.coordinates[0].point.z == 5.0  # FIB untouched
    assert w._coords_tab.fm_list.coordinates[0].point.z == 10.0  # FM scaled
