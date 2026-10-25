"""The figures alignment and autofocus save wear the shared dark theme (FIB-1289).

Built from synthetic runs and read back off the saved PNG, so what is checked is
the file the History tab shows, not the in-memory figure.
"""

import numpy as np
import pytest
from matplotlib.colors import to_hex

from fibsem.alignment import (
    AlignmentIteration,
    AlignmentMethod,
    AlignmentResult,
    AlignmentSubsystem,
)
from fibsem.autofunctions.autofocus import AutoFocusIteration, AutoFocusResult
from fibsem.autofunctions.plotting import plot_autofocus_result
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import Point
from fibsem.theme.plots import DARK


def _corner(path):
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB"))[0, 0]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _image():
    image = FibsemImage.generate_blank_image(resolution=(64, 48))
    image.data[:] = np.random.default_rng(0).integers(0, 255, image.data.shape)
    return image


@pytest.fixture
def alignment_run():
    steps = [
        AlignmentIteration(shift=Point(1e-6, 0), score=0.9, image=_image()),
        AlignmentIteration(shift=Point(0, 1e-7), score=0.4, image=_image()),
    ]
    return AlignmentResult(
        name="lam - Rough Milling",
        reference_image=_image(),
        subsystem=AlignmentSubsystem.BEAM_SHIFT,
        method=AlignmentMethod.CROSS_CORRELATION,
        results=steps,
        final_image=_image(),
    )


def test_a_saved_alignment_figure_is_dark(alignment_run, tmp_path):
    run_dir = alignment_run.save(str(tmp_path))
    assert _corner(f"{run_dir}/figure.png") == DARK.background


def test_alignment_scores_take_the_themes_meaning_colours(alignment_run):
    fig = alignment_run.plot(save=False)
    colours = {to_hex(t.get_color()) for ax in fig.axes for t in ax.texts}
    assert {DARK.good, DARK.bad} <= colours, "0.9 is good, 0.4 is bad"
    assert "#00ff00" not in colours, "no lime left over"


def test_a_saved_autofocus_figure_is_dark(tmp_path):
    iterations = [
        AutoFocusIteration(
            working_distance=0.004 + i * 1e-5,
            focus_score=float(10 - abs(i - 3)),
            pass_index=0,
            image=np.random.default_rng(i).integers(0, 255, (48, 64)).astype(np.uint8),
        )
        for i in range(7)
    ]
    result = AutoFocusResult(
        image=iterations[3].image,
        working_distance=iterations[3].working_distance,
        initial_working_distance=0.004,
        focus_score=10.0,
        iterations=iterations,
        method="tenengrad",
    )
    plot_autofocus_result(result, save_path=str(tmp_path / "plot.png"))
    assert _corner(tmp_path / "plot.png") == DARK.background
