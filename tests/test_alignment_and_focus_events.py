"""Alignment, coincidence and autofocus results on the experiment's record (FIB-1042).

Each test runs the real procedure on Demo, with a real ``EventRecorder`` writing a
real ``events.jsonl``, and reads back what it wrote.
"""

import os
import threading

import pytest

from fibsem import acquire, utils
from fibsem.alignment import AlignmentSubsystem, multi_step_alignment_v2
from fibsem.alignment.coincidence import check_coincidence
from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    EventRecorder,
    read_events,
)
from fibsem.autofunctions.autofocus import (
    AutoFocusSettings,
    FocusSweepPass,
    run_auto_focus,
)
from fibsem.cancellation import OperationCancelledError
from fibsem.structures import BeamType, ImageSettings


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
        {"x": pytest.approx(r.shift.x), "y": pytest.approx(r.shift.y)}
        for r in run.results
    ]
    assert alignment["validation"] is None
    assert alignment["path"] == str(tmp_path / "alignment" / run.name)
    assert os.path.exists(os.path.join(alignment["path"], "data.json"))


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
    assert focus["steps"] == len(result.iterations)
    assert focus["hfw"] == pytest.approx(100e-6)


def test_a_cancelled_autofocus_records_no_focus(microscope, recorded):
    stop = threading.Event()
    stop.set()
    with pytest.raises(OperationCancelledError):
        run_auto_focus(
            microscope, BeamType.ELECTRON, settings=_focus_settings(), stop_event=stop
        )

    assert recorded("autofocus") == []
