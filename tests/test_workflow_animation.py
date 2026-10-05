"""A workflow animation: which images make its frames, and what it writes.

Qt-free, like the modules under test, so this runs on every CI job.
"""

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image

from fibsem.applications.autolamella.structures import AutoLamellaTaskState, Lamella
from fibsem.applications.autolamella.task_outputs import (
    final_images_by_task,
    images_by_task,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.imaging.animation import (
    AnimationFrame,
    AnimationOptions,
    AnimationStep,
    default_animation_name,
    frame_durations,
    included,
    load_frames,
    render_animation,
    render_frame,
    save_animation,
    webp_supported,
)
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemExperimentRef,
    FibsemImage,
    ImageSettings,
    MicroscopeState,
)


def _image(beam: BeamType, hfw: float, value: int = 100) -> FibsemImage:
    image = FibsemImage.generate_blank_image(resolution=(300, 200), hfw=hfw)
    # Noise about a level, as a real image has: what auto contrast stretches.
    noise = np.random.default_rng(value).integers(0, 40, size=image.data.shape)
    image.data[:] = (value + noise).astype(np.uint8)
    image.metadata.image_settings = ImageSettings(
        resolution=(300, 200), hfw=hfw, beam_type=beam
    )
    image.metadata.microscope_state = MicroscopeState(
        electron_beam=BeamSettings(BeamType.ELECTRON, voltage=2e3),
        ion_beam=BeamSettings(BeamType.ION, voltage=30e3),
    )
    image.metadata.experiment = FibsemExperimentRef(name="exp", item_name="lam-01")
    return image


def _lamella(tmp_path: Path) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=1, petname="lam-01")
    os.makedirs(lamella.path, exist_ok=True)
    return lamella


def _run(
    lamella: Lamella, task: str, value: int = 100, sem: bool = True
) -> AutoLamellaTaskState:
    """A completed run of ``task`` that saved its final set, as a task does: SEM and
    FIB at a wide (150 µm) and a narrow (100 µm) field width."""
    outputs = {"final_sem": [], "final_fib": []}
    for res, hfw in (("01", 150e-6), ("02", 100e-6)):
        for beam, suffix, role in (
            (BeamType.ELECTRON, "eb", "final_sem"),
            (BeamType.ION, "ib", "final_fib"),
        ):
            if beam is BeamType.ELECTRON and not sem:
                continue
            name = f"ref_{task}_final_res_{res}_{suffix}.tif"
            _image(beam, hfw, value).save(os.path.join(lamella.path, name))
            outputs[role].append(name)
    run = AutoLamellaTaskState(name=task, outputs=outputs)
    lamella.task_history.append(run)
    return run


def _frames(tmp_path: Path, tasks=("Setup", "Mill", "Polish")):
    lamella = _lamella(tmp_path)
    for i, task in enumerate(tasks):
        _run(lamella, task, value=60 + 40 * i)
    steps = [AnimationStep(n, p) for n, p in final_images_by_task(lamella)]
    return load_frames(steps)


# -- which images ------------------------------------------------------------


def test_a_step_per_task_in_the_order_they_ran(tmp_path):
    lamella = _lamella(tmp_path)
    for task in ("Setup", "Mill", "Polish"):
        _run(lamella, task)
    steps = final_images_by_task(lamella)
    assert [name for name, _ in steps] == ["Setup", "Mill", "Polish"]
    assert all(len(paths) == 4 for _, paths in steps)


def test_a_run_that_saved_nothing_adds_no_step(tmp_path):
    lamella = _lamella(tmp_path)
    _run(lamella, "Setup")
    lamella.task_history.append(AutoLamellaTaskState(name="Acquire FM"))  # cancelled
    _run(lamella, "Mill")
    assert [name for name, _ in final_images_by_task(lamella)] == ["Setup", "Mill"]


def test_a_task_run_again_is_one_step_where_it_last_ran(tmp_path):
    """Reference images are rewritten to the same filename on every run, so an
    earlier run's files are the later run's; one frame, at the later place."""
    lamella = _lamella(tmp_path)
    _run(lamella, "Mill")
    _run(lamella, "Setup")
    _run(lamella, "Mill")
    assert [name for name, _ in final_images_by_task(lamella)] == ["Setup", "Mill"]


def test_magnification_is_chosen_by_field_width_not_filename(tmp_path):
    frame = _frames(tmp_path, tasks=("Setup",))[0]
    high, low = frame.pick("FIB", "high"), frame.pick("FIB", "low")

    def hfw(image):
        return image.pixel_size * image.rgb.shape[1]

    assert hfw(high) == pytest.approx(100e-6)
    assert hfw(low) == pytest.approx(150e-6)
    assert frame.pick("SEM", "high").kind == "SEM"


def test_an_unreadable_file_is_left_out_not_fatal(tmp_path):
    good = str(tmp_path / "good.tif")
    _image(BeamType.ION, 100e-6).save(good)
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff")
    frames = load_frames(
        [AnimationStep("A", [good, str(bad)]), AnimationStep("B", [str(bad)])]
    )
    assert [f.title for f in frames] == ["A"]
    assert len(frames[0].images) == 1


# -- drawing -----------------------------------------------------------------


def test_every_frame_is_one_size_and_the_chosen_width(tmp_path):
    frames = _frames(tmp_path)
    rendered = render_animation(frames, AnimationOptions(width=400))
    assert len(rendered) == 3
    assert {r.shape for r in rendered} == {rendered[0].shape}
    assert rendered[0].shape[1] == 400


def test_both_beams_side_by_side_are_each_the_chosen_width(tmp_path):
    frames = _frames(tmp_path)
    rendered = render_animation(frames, AnimationOptions(beam="both", width=400))
    assert rendered[0].shape[1] == 2 * 400 + 6


def test_a_step_without_the_beam_is_left_out_not_a_gap(tmp_path):
    lamella = _lamella(tmp_path)
    _run(lamella, "Setup")
    _run(lamella, "Mill", sem=False)  # FIB only
    frames = load_frames(
        [AnimationStep(n, p) for n, p in final_images_by_task(lamella)]
    )
    assert included(frames, AnimationOptions(beam="FIB")) == [0, 1]
    assert included(frames, AnimationOptions(beam="SEM")) == [0]
    assert included(frames, AnimationOptions(beam="both")) == [0]


def test_skipped_frames_are_left_out_and_the_counter_follows(tmp_path):
    frames = _frames(tmp_path)
    options = AnimationOptions(skipped=[1])
    assert included(frames, options) == [0, 2]
    assert len(render_animation(frames, options)) == 2


def test_the_lamella_is_named_on_the_plate_and_the_experiment_on_the_bar(tmp_path):
    frame: AnimationFrame = _frames(tmp_path, tasks=("Setup",))[0]
    named = render_frame(frame, AnimationOptions(), name="a-much-longer-lamella-name")
    unnamed = render_frame(frame, AnimationOptions(lamella=False))
    # The plate grows to fit the name; the rest of the image is the same.
    assert not np.array_equal(named[:60, :250], unnamed[:60, :250])
    assert np.array_equal(named[-80:-40, -60:], unnamed[-80:-40, -60:])

    with_experiment = render_frame(frame, AnimationOptions(experiment=True))
    without = render_frame(frame, AnimationOptions(experiment=False))
    assert with_experiment.shape[0] > without.shape[0]  # the bar's second row


def test_the_plate_falls_back_to_the_name_the_images_recorded(tmp_path):
    """Without a name passed in, the item the workflow stamped (here "lam-01")."""
    frame: AnimationFrame = _frames(tmp_path, tasks=("Setup",))[0]
    recorded = render_frame(frame, AnimationOptions())
    passed = render_frame(frame, AnimationOptions(), name="lam-01")
    assert np.array_equal(recorded, passed)


def test_the_title_plate_is_drawn_top_left(tmp_path):
    frame: AnimationFrame = _frames(tmp_path, tasks=("Setup",))[0]
    with_title = render_frame(frame, AnimationOptions(title=True))
    without = render_frame(frame, AnimationOptions(title=False))
    assert not np.array_equal(with_title[:30, :120], without[:30, :120])
    assert np.array_equal(with_title[-30:, -60:], without[-30:, -60:])


def test_auto_contrast_evens_out_brightness_between_frames(tmp_path):
    """Saved images differ in brightness; auto contrast is what stops the flicker."""
    frames = _frames(tmp_path)  # backgrounds at 60, 100 and 140
    plain = AnimationOptions(
        auto_contrast=False, title=False, bar=False, scalebar=False
    )
    auto = AnimationOptions(auto_contrast=True, title=False, bar=False, scalebar=False)

    def spread(options):
        means = [r.mean() for r in render_animation(frames, options)]
        return max(means) - min(means)

    assert spread(auto) < spread(plain)


# -- writing -----------------------------------------------------------------


def test_the_last_frame_is_held(tmp_path):
    options = AnimationOptions(frame_ms=1200, hold_ms=2500)
    assert frame_durations(3, options) == [1200, 1200, 2500]
    assert frame_durations(3, AnimationOptions(frame_ms=600, hold_ms=0)) == [
        600,
        600,
        600,
    ]
    assert frame_durations(0, options) == []


def test_save_writes_a_looping_gif_with_every_frame(tmp_path):
    frames = _frames(tmp_path)
    options = AnimationOptions(width=300)
    rendered = render_animation(frames, options)
    path = save_animation(rendered, str(tmp_path / "workflow.gif"), options)

    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 3
        assert gif.info.get("loop") == 0
        assert gif.size == (300, rendered[0].shape[0])


@pytest.mark.skipif(not webp_supported(), reason="this Pillow has no animated WebP")
def test_save_writes_an_animated_webp_in_full_colour(tmp_path):
    frames = _frames(tmp_path)
    options = AnimationOptions(width=300)
    rendered = render_animation(frames, options)
    path = save_animation(rendered, str(tmp_path / "workflow.webp"), options)

    with Image.open(path) as webp:
        assert webp.format == "WEBP"
        assert webp.n_frames == 3
        assert webp.mode in ("RGB", "RGBA")


@pytest.mark.parametrize("name", ["workflow.mp4", "workflow.png", "workflow"])
def test_other_formats_say_which_are_supported(tmp_path, name):
    frames = render_animation(_frames(tmp_path), AnimationOptions(width=200))
    with pytest.raises(ValueError, match=r"\.gif, \.webp"):
        save_animation(frames, str(tmp_path / name), AnimationOptions())


def test_nothing_to_save_says_so(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        save_animation([], str(tmp_path / "x.gif"), AnimationOptions())


@pytest.mark.parametrize(
    "title, expected",
    [("02-pro-moose", "02-pro-moose_workflow.gif"), ("a/b c", "a_b_c_workflow.gif")],
)
def test_default_name(title, expected, tmp_path):
    folder: Optional[str] = str(tmp_path)
    assert default_animation_name(title, folder) == os.path.join(folder, expected)


# -- fluorescence frames -----------------------------------------------------


# Where each channel's signal sits, whatever its index in the stack.
# 256 px: big enough that the channel legend, at its smallest readable size, stays in
# its corner rather than covering the image.
_SIGNAL = {"DAPI": (40, 120), "GFP": (140, 220)}


def _fm_run(lamella: Lamella, task: str = "Acquire FM", order=("DAPI", "GFP")) -> str:
    """A run of a fluorescence task that saved one z-stack, as the task records it.
    ``order`` is the channels' order in the stack."""
    data = np.zeros((len(order), 3, 256, 256), dtype=np.uint16)
    for index, name in enumerate(order):
        lo, hi = _SIGNAL[name]
        data[index, :, lo:hi, lo:hi] = 30000
    colours = {"DAPI": "cyan", "GFP": "green"}
    channels = [
        FluorescenceChannelMetadata(
            name=name,
            excitation_wavelength=488.0,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
            color=color,
        )
        for name, color in ((n, colours[n]) for n in order)
    ]
    image = FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-10-02T14:48:00",
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            channels=channels,
            z_positions=[0.0, 0.5e-6, 1.0e-6],
        ),
    )
    name = f"{lamella.petname}-{task.replace(' ', '-')}-zstack.ome.tiff"
    image.save(os.path.join(lamella.path, name))
    lamella.task_history.append(
        AutoLamellaTaskState(name=task, outputs={"fluorescence": [name]})
    )
    return name


def _fm_frames(tmp_path):
    lamella = _lamella(tmp_path)
    _run(lamella, "Setup")
    _fm_run(lamella)
    _run(lamella, "Mill")
    steps = [AnimationStep(n, f, s) for n, f, s in images_by_task(lamella)]
    return load_frames(steps)


def test_a_fluorescence_task_is_a_step_where_it_ran(tmp_path):
    lamella = _lamella(tmp_path)
    _run(lamella, "Setup")
    stack = _fm_run(lamella)
    _run(lamella, "Mill")
    steps = images_by_task(lamella)
    assert [name for name, _, _ in steps] == ["Setup", "Acquire FM", "Mill"]
    assert [os.path.basename(p) for p in steps[1][2]] == [stack]
    # The beam-only reader is unchanged: no step for a task with no final images.
    assert [n for n, _ in final_images_by_task(lamella)] == ["Setup", "Mill"]


def test_the_stack_is_not_read_until_fm_is_chosen(tmp_path):
    frames = _fm_frames(tmp_path)
    fm = frames[1]
    assert fm._fluorescence is None, "read at load, though the beam is FIB"

    assert included(frames, AnimationOptions(beam="FIB")) == [0, 2]
    assert fm._fluorescence is None

    assert included(frames, AnimationOptions(beam="FM")) == [1]
    assert fm.pick("FM").kind == "FM"


def test_fm_shows_only_the_tasks_with_a_stack(tmp_path):
    lamella = _lamella(tmp_path)
    _run(lamella, "Setup")
    _fm_run(lamella, "Acquire FM")
    _run(lamella, "Mill")
    _fm_run(lamella, "Acquire FM again")
    frames = load_frames(
        [AnimationStep(n, f, s) for n, f, s in images_by_task(lamella)]
    )
    rendered = render_animation(frames, AnimationOptions(beam="FM", width=300))
    assert included(frames, AnimationOptions(beam="FM")) == [1, 3]
    assert len(rendered) == 2
    assert {r.shape for r in rendered} == {rendered[0].shape}
    assert rendered[0].shape[1] == 300


def test_an_unreadable_stack_is_left_out_not_fatal(tmp_path):
    bad = tmp_path / "bad.ome.tiff"
    bad.write_bytes(b"not a tiff")
    good = str(tmp_path / "good.tif")
    _image(BeamType.ION, 100e-6).save(good)
    frames = load_frames(
        [AnimationStep("A", [good]), AnimationStep("FM", [], [str(bad)])]
    )
    assert included(frames, AnimationOptions(beam="FM")) == []
    assert render_animation(frames, AnimationOptions(beam="FM")) == []
    assert included(frames, AnimationOptions(beam="FIB")) == [0]


def test_a_hidden_channel_leaves_every_stack_by_name(tmp_path):
    """Stacks need not list their channels in one order; hiding goes by name."""
    lamella = _lamella(tmp_path)
    _fm_run(lamella, "First", order=("DAPI", "GFP"))
    _fm_run(lamella, "Second", order=("GFP", "DAPI"))
    frames = load_frames(
        [AnimationStep(n, f, s) for n, f, s in images_by_task(lamella)]
    )
    # Drawn at the stacks' own 256 px: DAPI's square is at 40-120, GFP's at 140-220.
    plain = AnimationOptions(
        beam="FM", width=256, title=False, bar=False, scalebar=False
    )
    no_dapi = replace(plain, hidden_channels=["DAPI"])

    shown = render_animation(frames, plain)
    hidden = render_animation(frames, no_dapi)
    assert len(shown) == len(hidden) == 2
    for with_dapi, without in zip(shown, hidden):
        assert with_dapi[70:90, 70:90].any()
        assert not without[70:90, 70:90].any()
        assert without[170:190, 170:190].any()
