"""Fluorescence acquisitions on the experiment's record (FIB-1033).

Each test runs the real acquisition code on the simulated FM (this package's
conftest switches to the Arctis simulator configuration), with a real
``EventRecorder`` writing a real ``events.jsonl``, and reads back what it wrote.
"""

import os

import pytest

from fibsem import utils
from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    EventRecorder,
    read_events,
)
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.calibration import run_coarse_fine_autofocus
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    FocusMethod,
    OverviewParameters,
    ZParameters,
)

CHANNEL = ChannelSettings(
    name="GFP",
    excitation_wavelength=488,
    emission_wavelength=509,
    power=0.2,
    exposure_time=0.1,
)


@pytest.fixture
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope.stage_is_compustage = True
    microscope.move_to_microscope("FM")
    return microscope


def _recorded(tmp_path, kind):
    path = tmp_path / EVENTS_FILENAME
    return [r for r in read_events(path) if r["kind"] == kind] if path.exists() else []


def test_a_z_stack_records_the_file_it_was_saved_to(microscope, tmp_path):
    filename = str(tmp_path / "01-lamella-zstack.ome.tiff")
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        image = acquire_image(
            microscope.fm,
            CHANNEL,
            zparams=ZParameters(zmin=-2e-6, zmax=2e-6, zstep=1e-6),
            filename=filename,
        )
    finally:
        recorder.close()
    (event,) = _recorded(tmp_path, "fm_image_acquired")
    payload = event["payload"]
    assert payload["path"] == image.filepath == filename
    assert os.path.exists(payload["path"])
    assert [c["name"] for c in payload["channels"]] == ["GFP"]
    assert payload["channels"][0]["exposure_time"] == pytest.approx(0.1)
    assert len(payload["z_positions"]) == image.metadata.get_z_count()
    assert payload["acquired_at"] == image.metadata.acquisition_date
    assert payload["overview"] is None


def test_an_image_that_was_not_saved_is_recorded_without_a_path(microscope, tmp_path):
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        acquire_image(microscope.fm, CHANNEL)
    finally:
        recorder.close()
    (event,) = _recorded(tmp_path, "fm_image_acquired")
    assert event["payload"]["path"] is None
    assert event["payload"]["z_positions"] is None


def test_an_overview_is_one_event_not_one_per_tile(microscope, tmp_path):
    from fibsem.applications.autolamella.workflows.tasks.grid.fluorescence import (
        acquire_fluorescence_overview,
    )

    parameters = OverviewParameters(
        rows=2, cols=2, overlap=0.1, use_zstack=False, autofocus_mode=AutoFocusMode.NONE
    )
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        _, path = acquire_fluorescence_overview(
            microscope,
            [CHANNEL],
            parameters,
            centre=microscope.get_stage_position(),
            directory=tmp_path / "overviews",
        )
    finally:
        recorder.close()
    (event,) = _recorded(tmp_path, "fm_image_acquired")  # not the four tiles
    assert event["payload"]["path"] == path and os.path.exists(path)
    assert event["payload"]["overview"] == {"rows": 2, "cols": 2, "overlap": 0.1}


def test_fm_autofocus_records_where_it_moved_the_objective(microscope, tmp_path):
    settings = AutoFocusSettings.from_coarse_fine(
        coarse_range=4e-6,
        coarse_step=2e-6,
        fine_range=2e-6,
        fine_step=1e-6,
        method=FocusMethod.LAPLACIAN,
    )
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        result = run_coarse_fine_autofocus(microscope.fm, settings, CHANNEL)
    finally:
        recorder.close()
    (event,) = _recorded(tmp_path, "fm_autofocus")
    assert event["payload"]["position"] == pytest.approx(result.working_distance)
    assert event["payload"]["initial_position"] == pytest.approx(
        result.initial_working_distance
    )
    assert event["payload"]["passes"] == event["payload"]["completed_passes"] == 2


def test_the_objective_records_its_insert_and_retract_not_every_move(
    microscope, tmp_path
):
    objective = microscope.fm.objective
    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        objective.retract()
        objective.insert()
        objective.move_relative(1e-6)  # a focus step: still inserted
        objective.move_relative(-1e-6)
        objective.retract()
    finally:
        recorder.close()
    states = [
        r["payload"]["state"] for r in _recorded(tmp_path, "objective_state_changed")
    ]
    assert states == ["Retracted", "Inserted", "Retracted"]


def test_report_v2_reads_each_acquisition_once(microscope, tmp_path):
    """A z-stack and a stitched overview, read back as report v2's fm table
    (FIB-1036): one row each, with what was acquired and for how long."""
    from fibsem.applications.autolamella.tools.event_tables import read_event_tables
    from fibsem.applications.autolamella.workflows.tasks.grid.fluorescence import (
        acquire_fluorescence_overview,
    )

    recorder = EventRecorder(microscope, experiment_path=tmp_path)
    try:
        stack = acquire_image(
            microscope.fm,
            CHANNEL,
            zparams=ZParameters(zmin=-2e-6, zmax=2e-6, zstep=1e-6),
            filename=str(tmp_path / "01-lamella-zstack.ome.tiff"),
        )
        _, overview = acquire_fluorescence_overview(
            microscope,
            [CHANNEL],
            OverviewParameters(
                rows=2,
                cols=2,
                overlap=0.1,
                use_zstack=False,
                autofocus_mode=AutoFocusMode.NONE,
            ),
            centre=microscope.get_stage_position(),
            directory=tmp_path / "overviews",
        )
    finally:
        recorder.close()

    rows = read_event_tables(tmp_path).fm.to_dict("records")

    first, second = rows
    assert (first["planes"], first["path"]) == (
        stack.metadata.get_z_count(),
        stack.filepath,
    )
    assert not isinstance(first["overview"], str)  # None, or NaN on pandas 3
    assert (second["planes"], second["overview"], second["path"]) == (
        1,
        "2×2",
        overview,
    )
    assert all(r["channels"] == ["GFP"] for r in rows)
    assert all(r["start"] <= r["end"] and r["duration"] >= 0 for r in rows)
