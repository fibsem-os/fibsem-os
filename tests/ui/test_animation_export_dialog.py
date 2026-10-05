"""The GIF export dialog: its settings reach the frames it plays and saves."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PIL import Image

from fibsem.imaging.animation import AnimationStep, load_frames
from fibsem.structures import BeamType, FibsemImage, ImageSettings, MicroscopeState


def _save(path: str, beam: BeamType, hfw: float, value: int) -> str:
    image = FibsemImage.generate_blank_image(resolution=(300, 200), hfw=hfw)
    noise = np.random.default_rng(value).integers(0, 40, size=image.data.shape)
    image.data[:] = (value + noise).astype(np.uint8)
    image.metadata.image_settings = ImageSettings(
        resolution=(300, 200), hfw=hfw, beam_type=beam
    )
    image.metadata.microscope_state = MicroscopeState()
    return image.save(path)


@pytest.fixture
def frames(tmp_path):
    """Three tasks with SEM and FIB at two field widths; the last has no SEM image."""
    steps = []
    for i, task in enumerate(("Setup", "Mill", "Polish")):
        paths = []
        for hfw in (150e-6, 100e-6):
            paths.append(
                _save(str(tmp_path / f"{task}_{hfw}_ib.tif"), BeamType.ION, hfw, 60 + i)
            )
            if task != "Polish":
                paths.append(
                    _save(
                        str(tmp_path / f"{task}_{hfw}_eb.tif"),
                        BeamType.ELECTRON,
                        hfw,
                        90 + i,
                    )
                )
        steps.append(AnimationStep(task, paths))
    return load_frames(steps)


@pytest.fixture
def dialog(qapp, frames, tmp_path):
    from fibsem.ui.widgets.animation_export_dialog import AnimationExportDialog

    d = AnimationExportDialog(frames, "lam-01", str(tmp_path))
    yield d
    d.done(0)
    d.deleteLater()


def test_starts_with_every_task_playing(dialog):
    assert len(dialog._rendered) == 3
    assert dialog.label_summary.text().startswith("768 × ")
    assert "3 frames" in dialog.label_summary.text()
    assert dialog.button_save.isEnabled()


def test_unticking_a_task_leaves_it_out(dialog):
    dialog.tiles[1].checkbox.setChecked(False)
    assert dialog.options.skipped == [1]
    assert len(dialog._rendered) == 2
    assert "2 frames" in dialog.label_summary.text()


def test_a_task_without_the_beam_is_left_out_of_the_strip(dialog):
    dialog.segment_beam.group.button(0).click()  # SEM
    assert dialog.tiles[2].isHidden()  # Polish has no SEM image
    assert not dialog.tiles[0].isHidden()
    assert len(dialog._rendered) == 2

    dialog.segment_beam.group.button(1).click()  # FIB: every task has one
    assert not dialog.tiles[2].isHidden()


def test_both_beams_doubles_the_width(dialog):
    dialog.segment_beam.group.button(3).click()  # Both
    assert dialog._rendered[0].shape[1] == 2 * 768 + 6
    assert len(dialog._rendered) == 2  # Polish has no SEM image


def test_clicking_a_tile_jumps_to_it_and_pauses(dialog):
    dialog.show_frame_of(2)
    assert dialog._current == 2
    assert not dialog._playing
    assert dialog.label_position.text() == "3 / 3 · Polish"


def test_nothing_left_disables_save(dialog):
    for tile in dialog.tiles:
        tile.checkbox.setChecked(False)
    assert not dialog.button_save.isEnabled()
    assert dialog.label_summary.text().startswith("No frames")


def test_save_writes_the_gif_it_plays(dialog, tmp_path):
    dialog.segment_width.group.button(0).click()  # 768
    written = dialog.save(str(tmp_path / "out"))
    assert written.endswith(".gif")
    with Image.open(written) as gif:
        assert gif.n_frames == len(dialog._rendered)
        assert gif.size == (dialog._rendered[0].shape[1], dialog._rendered[0].shape[0])


def test_webp_is_offered_and_names_the_file(dialog, tmp_path):
    from fibsem.imaging.animation import webp_supported

    webp = dialog.segment_format.group.button(1)
    assert webp.isEnabled() == webp_supported()
    if not webp_supported():
        pytest.skip("this Pillow has no animated WebP")
    webp.click()
    assert dialog.label_summary.text().endswith("WebP")
    written = dialog.save(str(tmp_path / "out"))
    assert written.endswith(".webp")
    with Image.open(written) as image:
        assert image.format == "WEBP"


def test_the_lamella_and_counter_need_the_title(dialog):
    dialog.checkbox_title.setChecked(False)
    assert not dialog.checkbox_lamella.isEnabled()
    assert not dialog.checkbox_counter.isEnabled()


def test_fm_is_unavailable_without_a_stack(dialog):
    fm = dialog.segment_beam.group.button(2)
    assert fm.text() == "FM"
    assert not fm.isEnabled()
    assert dialog.label_colour_hint.isHidden()


def test_fm_plays_only_the_stack_and_hints_at_webp(qapp, frames, tmp_path):
    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )
    from fibsem.imaging.animation import AnimationFrame
    from fibsem.ui.widgets.animation_export_dialog import AnimationExportDialog

    data = np.zeros((1, 2, 64, 64), dtype=np.uint16)
    data[0, :, 10:30, 10:30] = 30000
    channel = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color="cyan",
    )
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-10-02T14:48:00",
        pixel_size_x=1e-6,
        pixel_size_y=1e-6,
        channels=[channel],
        z_positions=[0.0, 0.5e-6],
    )
    stack = FluorescenceImage(data=data, metadata=metadata).save(
        str(tmp_path / "stack.ome.tiff")
    )
    fm = AnimationFrame(title="Acquire FM", images=[], fluorescence_paths=[stack])
    d = AnimationExportDialog([*frames[:1], fm, *frames[1:]], "lam-01", str(tmp_path))
    try:
        fm_button = d.segment_beam.group.button(2)
        assert fm_button.isEnabled()
        assert d.tiles[1].isHidden()  # FIB: the FM task has no FIB image
        assert len(d._rendered) == 3
        assert fm._fluorescence is None, "the stack is read only for FM"
        assert not d.channel_checkboxes, "listing channels would read the stack"

        fm_button.click()
        assert len(d._rendered) == 1
        assert not d.tiles[1].isHidden()
        assert all(t.isHidden() for i, t in enumerate(d.tiles) if i != 1)
        assert not d.label_colour_hint.isHidden()
        assert d.segment_magnification.isHidden()
        assert d.checkbox_auto_contrast.isHidden()

        assert list(d.channel_checkboxes) == ["DAPI"]
        assert not d.channels_box.isHidden()
        d.channel_checkboxes["DAPI"].setChecked(False)
        assert d.options.hidden_channels == ["DAPI"]

        d.segment_format.group.button(1).click()  # WebP keeps the colours
        assert d.label_colour_hint.isHidden()

        d.segment_beam.group.button(1).click()  # FIB: channels do not apply
        assert d.channels_box.isHidden()
    finally:
        d.done(0)
        d.deleteLater()
