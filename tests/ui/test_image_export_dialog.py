"""The export dialog: its settings reach the renderer, and its colours match the app's."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from fibsem.imaging import export as ex
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemExperimentRef,
    FibsemImage,
    MicroscopeState,
)
from fibsem.ui import tokens


def _sem_image() -> FibsemImage:
    image = FibsemImage.generate_blank_image(resolution=(768, 512), hfw=150e-6)
    image.metadata.microscope_state = MicroscopeState(
        electron_beam=BeamSettings(
            BeamType.ELECTRON,
            voltage=2e3,
            beam_current=50e-12,
            working_distance=4e-3,
        ),
        electron_detector=FibsemDetectorSettings(type="ETD", mode="SE"),
    )
    image.metadata.experiment = FibsemExperimentRef(
        name="exp", item_name="lamella-03", task_name="Polishing"
    )
    return image


@pytest.fixture
def dialog(qapp):
    from fibsem.ui.widgets.image_export_dialog import ImageExportDialog

    d = ImageExportDialog(ex.from_fibsem_image(_sem_image(), "/data/ref_image.tif"))
    yield d
    d.deleteLater()


def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def test_the_renderers_colours_are_the_palettes():
    """The renderer copies these rather than importing fibsem.ui, which needs Qt."""
    assert _hex(ex.BAR_COLOR) == tokens.PANEL_COLOR.lower()
    assert _hex(ex.BORDER_COLOR) == tokens.BORDER_COLOR.lower()
    assert _hex(ex.TEXT_COLOR) == tokens.TEXT_COLOR.lower()
    assert _hex(ex.TEXT_STRONG_COLOR) == tokens.TEXT_STRONG_COLOR.lower()
    assert _hex(ex.TEXT_MUTED_COLOR) == tokens.TEXT_MUTED_COLOR.lower()
    assert _hex(ex.CROSSHAIR_COLOR) == tokens.CURRENT_POSITION_COLOUR.lower()


def test_starts_from_the_defaults(dialog):
    assert dialog.options.fields == [
        "detector",
        "hfw",
        "pixel_size",
        "voltage",
        "current",
    ]
    assert not dialog.options.provenance
    assert not dialog.options.crosshair


def test_at_five_values_the_rest_cannot_be_added(dialog):
    # The defaults are four values, plus the detector, which does not count.
    working_distance = dialog.field_checkboxes["working_distance"]
    dwell_time = dialog.field_checkboxes["dwell_time"]
    assert working_distance.isEnabled() and dwell_time.isEnabled()

    working_distance.setChecked(True)
    assert "working_distance" in dialog.options.fields
    assert not dwell_time.isEnabled()
    assert dialog.field_checkboxes["detector"].isEnabled()

    dialog.field_checkboxes["voltage"].setChecked(False)
    assert dwell_time.isEnabled()


def test_provenance_remembers_its_fields_while_off(dialog):
    assert not dialog.provenance_checkboxes["item"].isEnabled()

    dialog.checkbox_provenance.setChecked(True)
    assert dialog.options.provenance == ["item", "task", "date"]

    dialog.provenance_checkboxes["date"].setChecked(False)
    dialog.checkbox_provenance.setChecked(False)
    assert dialog.options.provenance == []
    dialog.checkbox_provenance.setChecked(True)
    assert dialog.options.provenance == ["item", "task"]


def test_settings_reach_the_render(dialog):
    before = dialog.rendered()
    dialog.checkbox_crosshair.setChecked(True)
    dialog.segment_scale.group.button(1).click()
    after = dialog.rendered()
    assert after.shape[1] == 2 * before.shape[1]
    assert dialog.label_size.text().startswith(f"{after.shape[1]} × {after.shape[0]}")


def test_save_writes_the_rendered_image(dialog, tmp_path):
    from PIL import Image

    dialog.segment_format.group.button(1).click()  # TIFF
    written = dialog.save(str(tmp_path / "figure"))
    assert written.endswith(".tif")
    assert np.array_equal(np.array(Image.open(written)), dialog.rendered())


def test_open_image_export_starts_in_the_given_directory(qapp, tmp_path, monkeypatch):
    from fibsem.ui.widgets import image_export_dialog as module

    path = _sem_image().save(str(tmp_path / "ref_image.tif"))
    asked = {}

    def pick(parent, caption, directory, filters):
        asked["directory"] = directory
        return path, filters

    shown = []
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", pick)
    monkeypatch.setattr(
        module.ImageExportDialog, "exec_", lambda self: shown.append(self) or 0
    )

    module.open_image_export(None, str(tmp_path))

    assert asked["directory"] == str(tmp_path)
    assert len(shown) == 1
    assert shown[0].image.path == path
    shown[0].deleteLater()


def test_open_image_export_cancelled(qapp, monkeypatch):
    from fibsem.ui.widgets import image_export_dialog as module

    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    monkeypatch.setattr(
        module.ImageExportDialog,
        "exec_",
        lambda self: pytest.fail("no dialog without a file"),
    )
    module.open_image_export(None, "")


def test_contrast_is_the_canvas_contrast(qapp):
    """The renderer copies ContrastGammaControl.apply; the two must give one image."""
    from fibsem.ui.widgets.canvas.contrast_gamma_control import ContrastGammaControl

    norm = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    control = ContrastGammaControl()
    for lo, hi, gamma in [
        (0.0, 1.0, 1.0),
        (0.2, 0.7, 1.0),
        (0.1, 0.9, 0.5),
        (0.3, 0.6, 2.4),
    ]:
        control.sld_min.setValue(lo)
        control.sld_max.setValue(hi)
        control.sld_gamma.setValue(gamma)
        assert np.allclose(
            ex.adjust_contrast(norm, lo, hi, gamma), control.apply(norm), atol=1e-6
        )
    control.deleteLater()


def test_detector_abbreviations_are_the_state_panels():
    from fibsem.ui.widgets.microscope_state_widget import _detector

    for mode, short in ex.DETECTOR_MODE_ABBREVIATIONS.items():
        assert (
            _detector(FibsemDetectorSettings(type="ETD", mode=mode)) == f"ETD / {short}"
        )


def test_auto_and_reset_contrast(dialog):
    assert dialog.options.contrast_is_default()
    dialog.auto_contrast()
    lo, hi = ex.auto_contrast_limits(dialog.image)
    assert dialog.options.contrast_min == pytest.approx(lo, abs=0.01)
    assert dialog.options.contrast_max == pytest.approx(hi, abs=0.01)
    dialog.reset_contrast()
    assert dialog.options.contrast_is_default()


def test_a_slider_drag_waits_before_re_rendering(dialog):
    dialog.slider_gamma.slider.setValue(2.0)
    assert dialog._refresh_timer.isActive()
    assert dialog.options.gamma == 1.0  # not yet
    dialog._refresh_timer.timeout.emit()
    assert dialog.options.gamma == pytest.approx(2.0)


def test_crossed_limits_keep_a_sliver_of_range(dialog):
    dialog.slider_min.set_value(0.6)
    dialog.slider_max.set_value(0.4)
    dialog._on_changed()
    assert dialog.options.contrast_min < dialog.options.contrast_max


def test_no_contrast_controls_for_a_colour_image(qapp):
    from fibsem.ui.widgets.image_export_dialog import ImageExportDialog

    colour = ex.ExportImage(
        rgb=np.zeros((64, 64, 3), dtype=np.uint8), pixel_size=1e-7, kind="FM"
    )
    d = ImageExportDialog(colour)
    assert not hasattr(d, "slider_min")
    d.deleteLater()


def test_channel_checkboxes_hide_channels(qapp):
    from fibsem.ui.widgets.image_export_dialog import ImageExportDialog

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    channels = [
        ex.ExportChannel("DAPI", (0, 255, 255)),
        ex.ExportChannel("GFP", (0, 255, 0)),
    ]
    d = ImageExportDialog(
        ex.ExportImage(rgb=rgb, pixel_size=1e-7, kind="FM", channels=channels)
    )
    assert [cb.text() for cb in d.channel_checkboxes] == ["DAPI", "GFP"]
    assert not d.options.hidden_channels
    d.channel_checkboxes[1].setChecked(False)
    assert d.options.hidden_channels == [1]
    d.deleteLater()
