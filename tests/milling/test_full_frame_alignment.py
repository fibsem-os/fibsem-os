"""Full-frame alignment at low currents (``MillingAlignment.full_frame_below``).

A stage whose milling current reads back at or below the setting aligns on the
whole frame instead of the alignment area; above it, or with the setting off
(0), it aligns on the area as before.
"""

import pytest

from fibsem import alignment
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.core import setup_milling
from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig
from fibsem.structures.settings import MillingAlignment
from tests.fixtures.microscopes import demo_microscope


def test_off_by_default_and_for_protocols_predating_it():
    assert MillingAlignment().full_frame_below == 0
    assert MillingAlignment.from_dict({}).full_frame_below == 0


def test_round_trips():
    settings = MillingAlignment(full_frame_below=80e-12)
    assert MillingAlignment.from_dict(settings.to_dict()).full_frame_below == 80e-12


microscope = demo_microscope(scope="function")


@pytest.fixture
def aligned_on(monkeypatch):
    """The reference image each alignment at the milling current was given."""
    refs = []

    def fake_alignment(microscope, ref_image, **kwargs):
        refs.append(ref_image)

    monkeypatch.setattr(alignment, "multi_step_alignment_v2", fake_alignment)
    return refs


def _stage(tmp_path, current, full_frame_below):
    stage = FibsemMillingStage(name="polish")
    stage.milling.milling_current = current
    stage.alignment.full_frame_below = full_frame_below
    stage.output_dir = str(tmp_path)
    return stage


def _is_full_frame(image):
    return image.metadata.image_settings.reduced_area is None


@pytest.mark.parametrize(
    "current, full_frame_below, full_frame",
    [
        (60e-12, 80e-12, True),  # below the setting
        (200e-12, 80e-12, False),  # above it
        (60e-12, 0.0, False),  # off
    ],
)
def test_the_stage_aligns_on_the_reference_its_current_calls_for(
    tmp_path, microscope, aligned_on, current, full_frame_below, full_frame
):
    stage = _stage(tmp_path, current, full_frame_below)

    setup_milling(microscope, stage)

    assert len(aligned_on) == 1
    assert _is_full_frame(aligned_on[0]) is full_frame


def test_the_read_back_current_decides_not_the_configured_one(
    tmp_path, microscope, aligned_on, monkeypatch
):
    """A Tescan sets its ion current by preset and ignores milling_current."""
    stage = _stage(tmp_path, 60e-12, 80e-12)
    monkeypatch.setattr(microscope, "get_beam_current", lambda beam_type: 300e-12)

    setup_milling(microscope, stage)

    assert not _is_full_frame(aligned_on[0])


def test_the_task_takes_the_full_frame_reference_only_when_it_is_on(
    tmp_path, microscope
):
    for full_frame_below, expected in ((80e-12, True), (0.0, False)):
        config = FibsemMillingTaskConfig.from_stages(
            stages=[FibsemMillingStage(name="polish")], name="t"
        )
        config.alignment.full_frame_below = full_frame_below
        task = FibsemMillingTask(microscope, config, output_dir=str(tmp_path))
        task._acquire_reference_image()

        assert not _is_full_frame(task.reference_image)
        assert (task.reference_image_full_frame is not None) is expected
        if expected:
            assert _is_full_frame(task.reference_image_full_frame)


def test_the_alignment_widget_edits_it_in_pA(qapp):
    pytest.importorskip("superqt")  # the widgets need [ui]
    from fibsem.ui.widgets.milling_alignment_widget import (
        FibsemMillingAlignmentWidget,
    )

    widget = FibsemMillingAlignmentWidget()
    assert widget.full_frame_spinbox.text() == "Off"

    widget.update_from_settings(MillingAlignment(full_frame_below=80e-12))
    assert widget.full_frame_spinbox.value() == pytest.approx(80)

    widget.full_frame_spinbox.setValue(60)
    assert widget.get_settings().full_frame_below == pytest.approx(60e-12)
