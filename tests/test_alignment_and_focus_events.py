"""Alignment, coincidence and autofocus results on the experiment's record (FIB-1042).

Each test runs the real procedure on Demo, with a real ``EventRecorder`` writing a
real ``events.jsonl``, and reads back what it wrote.
"""

import os
import threading
from datetime import datetime

import pytest

from fibsem import acquire, utils
from fibsem.alignment import AlignmentSubsystem, multi_step_alignment_v2
from fibsem.alignment.coincidence import check_coincidence
from fibsem.autofunctions.autofocus import (
    AutoFocusSettings,
    FocusSweepPass,
    run_auto_focus,
)
from fibsem.cancellation import OperationCancelledError
from fibsem.structures import BeamType, ImageSettings
from fibsem.workflows.event_recording import (
    EVENTS_FILENAME,
    EventRecorder,
    read_events,
)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def recorded(microscope, tmp_path):
    """The events recorded while the test body runs, read back from disk."""
    recorder = EventRecorder(microscope, experiment_path=tmp_path)

    def read(kind):
        recorder.close()
        path = tmp_path / EVENTS_FILENAME
        if not path.exists():  # nothing recorded at all
            return []
        return [r["payload"] for r in read_events(path) if r["kind"] == kind]

    yield read
    recorder.close()


def test_an_alignment_is_one_event_with_every_step(microscope, recorded, tmp_path):
    settings = ImageSettings(
        resolution=(256, 256), hfw=80e-6, beam_type=BeamType.ION, save=False
    )
    ref_image = acquire.acquire_image(microscope, settings)

    run = multi_step_alignment_v2(
        microscope,
        ref_image,
        steps=2,
        subsystem=AlignmentSubsystem.BEAM_SHIFT,
        validate=False,
        acquire_final_image=False,
        path=str(tmp_path / "alignment"),
    )

    (alignment,) = recorded("alignment")
    assert alignment["name"] == run.name
    assert alignment["beam_type"] == "ION"
    assert alignment["subsystem"] == AlignmentSubsystem.BEAM_SHIFT.value
    assert alignment["method"] == run.method.value
    assert alignment["steps"] == 2
    assert alignment["aborted"] is False
    assert [r["shift"] for r in alignment["results"]] == [
        {"x": pytest.approx(r.shift.x), "y": pytest.approx(r.shift.y), "name": None}
        for r in run.results
    ]
    assert alignment["validation"] is None
    assert alignment["check"] == run.check()
    assert alignment["path"] == str(tmp_path / "alignment" / run.name)
    assert os.path.exists(os.path.join(alignment["path"], "data.json"))
    assert alignment["status"] == "completed"
    assert datetime.fromisoformat(alignment["started_at"]).tzinfo is not None


def _reference(microscope):
    settings = ImageSettings(
        resolution=(256, 256), hfw=80e-6, beam_type=BeamType.ION, save=False
    )
    return acquire.acquire_image(microscope, settings)


def test_an_alignment_stopped_before_its_first_step_is_recorded_cancelled(
    microscope, recorded
):
    """It used to return without a record, so the stop left no trace."""
    stop = threading.Event()
    stop.set()

    run = multi_step_alignment_v2(microscope, _reference(microscope), stop_event=stop)

    (alignment,) = recorded("alignment")
    assert run.results == []
    assert alignment["status"] == "cancelled"
    assert alignment["results"] == [] and alignment["path"] is None


def test_a_stop_mid_alignment_takes_no_more_images(
    microscope, recorded, monkeypatch, tmp_path
):
    """A stop during a step ends the run there: no later step and no final
    image to check it, so a stop means no more beam (FIB-1430)."""
    import fibsem.alignment as alignment_module

    stop = threading.Event()
    step = alignment_module.beam_shift_alignment_v2
    acquire_from = alignment_module._acquire_from_reference_image
    images = []

    def stopped_during(**kwargs):
        result = step(**kwargs)
        stop.set()  # the operator pressed Stop while this step ran
        return result

    def counted(**kwargs):
        images.append(kwargs)
        return acquire_from(**kwargs)

    monkeypatch.setattr(alignment_module, "beam_shift_alignment_v2", stopped_during)
    monkeypatch.setattr(alignment_module, "_acquire_from_reference_image", counted)

    run = multi_step_alignment_v2(
        microscope,
        _reference(microscope),
        steps=3,
        stop_event=stop,
        path=str(tmp_path / "alignment"),
    )

    assert len(run.results) == 1
    assert len(images) == 1  # the stopped step's own; no final image after it
    assert run.final_image is None and run.validation is None
    (alignment,) = recorded("alignment")
    assert alignment["status"] == "cancelled"


def test_an_alignment_that_raises_is_recorded_failed_and_still_raises(
    microscope, recorded, monkeypatch
):
    import fibsem.alignment as alignment_module

    def fail(**kwargs):
        raise RuntimeError("no signal")

    monkeypatch.setattr(alignment_module, "beam_shift_alignment_v2", fail)
    with pytest.raises(RuntimeError, match="no signal"):
        multi_step_alignment_v2(microscope, _reference(microscope))

    (alignment,) = recorded("alignment")
    assert alignment["status"] == "failed"
    assert alignment["error"] == "RuntimeError: no signal"
    assert alignment["beam_type"] == "ION"


def test_a_coincidence_check_records_what_it_measured(microscope, recorded):
    measurement = check_coincidence(microscope, prior=(1e-6, -2e-6))

    (measured,) = recorded("coincidence_measured")
    assert measured["dx"] == pytest.approx(measurement.dx)
    assert measured["dy"] == pytest.approx(measurement.dy)
    assert measured["dz"] == pytest.approx(measurement.dz)
    assert measured["is_reliable"] is measurement.is_reliable
    assert measured["refusal_reason"] == measurement.refusal_reason
    assert measured["prior"] == [1e-6, -2e-6]
    assert measured["hfw"] == pytest.approx(
        measurement.sem_image.metadata.image_settings.hfw
    )


def _focus_settings():
    return AutoFocusSettings(
        passes=[FocusSweepPass(search_range=200e-6, step_size=50e-6)],
    )


def test_an_autofocus_records_the_working_distance_it_left(microscope, recorded):
    result = run_auto_focus(
        microscope, BeamType.ELECTRON, hfw=100e-6, settings=_focus_settings()
    )

    (focus,) = recorded("autofocus")
    assert focus["beam_type"] == "ELECTRON"
    assert focus["working_distance"] == pytest.approx(result.working_distance)
    assert focus["working_distance"] == pytest.approx(
        microscope.get_working_distance(BeamType.ELECTRON)
    )
    assert focus["initial_working_distance"] == pytest.approx(
        result.initial_working_distance
    )
    assert focus["focus_score"] == pytest.approx(result.focus_score)
    assert focus["check"] == result.check()
    assert focus["steps"] == len(result.iterations)
    assert focus["hfw"] == pytest.approx(100e-6)
    assert focus["status"] == "completed"
    assert focus["path"] is None, "nothing saved without a path"


def test_an_autofocus_given_a_path_saves_its_run_and_says_where(
    microscope, recorded, tmp_path
):
    run_auto_focus(
        microscope,
        BeamType.ELECTRON,
        settings=_focus_settings(),
        path=str(tmp_path / "autofunctions"),
        name="focus",
    )

    (focus,) = recorded("autofocus")
    assert focus["path"] == str(tmp_path / "autofunctions" / "focus")
    assert os.path.exists(os.path.join(focus["path"], "data.json"))


def test_a_cancelled_autofocus_records_the_stop_but_no_focus(microscope, recorded):
    """Recorded cancelled, and still no working distance: the sweep put the
    starting one back, so there is no focus to record (FIB-1042)."""
    stop = threading.Event()
    stop.set()
    with pytest.raises(OperationCancelledError):
        run_auto_focus(
            microscope, BeamType.ELECTRON, settings=_focus_settings(), stop_event=stop
        )

    (focus,) = recorded("autofocus")
    assert focus["status"] == "cancelled"
    assert "working_distance" not in focus and "error" not in focus


def test_an_autofocus_that_cannot_set_the_working_distance_is_recorded_skipped(
    microscope, recorded, monkeypatch
):
    monkeypatch.setattr(microscope, "is_working_distance_settable", lambda beam: False)

    assert run_auto_focus(microscope, BeamType.ION, settings=_focus_settings()) is None

    (focus,) = recorded("autofocus")
    assert focus["status"] == "skipped"
    assert "ION" in focus["reason"]


def test_an_autofocus_that_raises_is_recorded_failed(microscope, recorded, monkeypatch):
    import fibsem.autofunctions.autofocus as autofocus_module

    def fail(*args, **kwargs):
        raise RuntimeError("stage lost")

    monkeypatch.setattr(autofocus_module, "_run_sweep", fail)
    with pytest.raises(RuntimeError, match="stage lost"):
        run_auto_focus(microscope, BeamType.ELECTRON, settings=_focus_settings())

    (focus,) = recorded("autofocus")
    assert focus["status"] == "failed"
    assert focus["error"] == "RuntimeError: stage lost"


def _focus_result(start, best, wds):
    from fibsem.autofunctions.autofocus import AutoFocusIteration, AutoFocusResult

    iterations = [
        AutoFocusIteration(
            working_distance=wd,
            focus_score=10.0 if wd == best else 5.0,
            pass_index=0,
            image=None,
        )
        for wd in wds
    ]
    return AutoFocusResult(
        image=None,
        working_distance=best,
        initial_working_distance=start,
        focus_score=10.0,
        iterations=iterations,
    )


def test_an_autofocus_that_jumps_or_ends_at_the_sweep_edge_is_doubtful():
    """2026-10-19: Setup's focus picked a bad working distance on dark sites and
    the run carried on. The check says so; it stops nothing."""
    wds = [15.9e-3, 15.95e-3, 16.0e-3, 16.05e-3, 16.1e-3]
    near = _focus_result(16.0e-3, 16.05e-3, wds).check()
    assert near == {"verdict": "passed", "reasons": [], "peak_ratio": 2.0}

    far = _focus_result(15.8e-3, 16.05e-3, wds).check()
    assert far["verdict"] == "doubtful"
    assert far["reasons"] == ["the working distance moved +0.250 mm, more than 0.1 mm"]

    edge = _focus_result(16.0e-3, 16.1e-3, wds).check()
    assert edge["reasons"] == ["the best focus is at the edge of the sweep"]


def test_an_alignment_is_doubtful_when_its_methods_disagree_or_a_step_fails():
    from types import SimpleNamespace

    from fibsem.alignment import AlignmentResult

    def run(validation, *success):
        steps = [SimpleNamespace(success=ok) for ok in success]
        return AlignmentResult(
            name="a",
            reference_image=None,
            subsystem=AlignmentSubsystem.BEAM_SHIFT,
            method=None,
            results=steps,
            validation=validation,
        ).check()

    agree = SimpleNamespace(agreement=True, max_disagreement_px=0.4)
    disagree = SimpleNamespace(agreement=False, max_disagreement_px=4.2)
    assert run(agree, True) == {"verdict": "passed", "reasons": []}
    assert run(None, True) == {"verdict": "unchecked", "reasons": []}
    assert run(disagree, True) == {
        "verdict": "doubtful",
        "reasons": ["the methods disagree by 4.2 px"],
    }
    assert run(None, True, False)["reasons"] == [
        "step 2 scored under the minimum response, so it did not move"
    ]
