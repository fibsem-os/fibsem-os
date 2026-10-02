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


def test_a_task_without_the_beam_is_greyed_with_the_reason(dialog):
    dialog.segment_beam.group.button(0).click()  # SEM
    polish = dialog.tiles[2]
    assert not polish.checkbox.isEnabled()
    assert polish.toolTip() == "No SEM image"
    assert len(dialog._rendered) == 2


def test_both_beams_doubles_the_width(dialog):
    dialog.segment_beam.group.button(2).click()
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
