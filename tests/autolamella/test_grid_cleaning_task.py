"""The cryo cleaning grid task, end to end on the simulator.

It moves to the grid's slot at the requested orientation, scans the ion beam over
the field at the cleaning current for the time asked, stops between frames when
asked, always puts the ion current back, and records an ion reference image of the
cleaned field with a thumbnail by role.
"""

import threading
import time

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.task_outputs import (
    grid_outputs,
    latest_grid_output,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    GRID_TASK_REGISTRY,
    CryoCleaningGridTask,
    CryoCleaningGridTaskConfig,
    run_grid_task,
)
from fibsem.applications.autolamella.workflows.tasks.grid.cleaning import cryo_clean
from fibsem.cancellation import OperationCancelledError
from fibsem.microscopes._stage import SampleGrid, SlotCalibration, _create_sample_stage
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition

IMAGING_CURRENT = 1e-9


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    slot = microscope._stage.holder.slots["Slot-01"]
    slot.position = FibsemStagePosition(
        name="Slot-01", x=-4e-3, y=1e-3, z=4e-3, r=0.0, t=0.61
    )
    slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-02T11:24:09", "test")
    slot.loaded_grid = SampleGrid(name="grid-aspen")
    microscope.set_beam_current(IMAGING_CURRENT, BeamType.ION)
    return microscope


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    # No time: one frame is the least a clean scans.
    exp.grid_protocol.add(
        CryoCleaningGridTaskConfig(task_name="cryo_clean", duration=0)
    )
    exp.add_grid(GridRecord(name="grid-aspen"))
    return exp


class _StopAfter:
    """A stop event that reads as set from its `n`th check on."""

    def __init__(self, n: int):
        self.n = n
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks >= self.n


class TestConfig:
    def test_registered(self):
        assert GRID_TASK_REGISTRY["CRYO_CLEANING_GRID"] is CryoCleaningGridTask

    def test_round_trips_through_protocol_yaml(self, experiment, tmp_path):
        config = experiment.grid_protocol.task_config["cryo_clean"]
        config.orientation = "FIB"
        config.current = 5e-9
        config.field_of_view = 600e-6
        experiment.save(save_protocol=True)
        loaded = Experiment.load(tmp_path / "exp" / "experiment.yaml")
        again = loaded.grid_protocol.task_config["cryo_clean"]
        assert isinstance(again, CryoCleaningGridTaskConfig)
        assert again.orientation == "FIB"
        assert again.current == 5e-9 and again.field_of_view == 600e-6
        assert again.duration == 0 and again.acquire_reference is True


class TestOperation:
    def test_scans_at_the_current_and_restores_it(self, microscope):
        seen = []
        acquire = microscope.acquire_image

        def spy(*args, **kwargs):
            seen.append(microscope.get_beam_current(BeamType.ION))
            return acquire(*args, **kwargs)

        microscope.acquire_image = spy
        frames = cryo_clean(microscope, 15e-9, 900e-6, 0)
        assert frames == 1
        assert seen == [pytest.approx(15e-9)]
        assert microscope.get_beam_current(BeamType.ION) == pytest.approx(
            IMAGING_CURRENT
        )

    def test_scans_until_the_time_is_up(self, microscope):
        start = time.monotonic()
        frames = cryo_clean(microscope, 15e-9, 900e-6, 0.5)
        assert time.monotonic() - start >= 0.5
        assert frames >= 1

    def test_a_stop_ends_it_between_frames_and_restores_the_current(self, microscope):
        stop = _StopAfter(2)
        with pytest.raises(OperationCancelledError):
            cryo_clean(microscope, 15e-9, 900e-6, 60, stop_event=stop)
        assert stop.checks == 2  # one frame, then the stop was seen
        assert microscope.get_beam_current(BeamType.ION) == pytest.approx(
            IMAGING_CURRENT
        )

    def test_a_failed_frame_restores_the_current(self, microscope):
        def fail(*args, **kwargs):
            raise RuntimeError("scan failed")

        microscope.acquire_image = fail
        with pytest.raises(RuntimeError, match="scan failed"):
            cryo_clean(microscope, 15e-9, 900e-6, 0)
        assert microscope.get_beam_current(BeamType.ION) == pytest.approx(
            IMAGING_CURRENT
        )


class TestRun:
    def test_records_an_ion_reference_and_a_thumbnail(
        self, microscope, experiment, tmp_path
    ):
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "cryo_clean", experiment, grid)

        entry = grid.task_history[-1]
        assert entry.status is AutoLamellaTaskStatus.Completed
        assert set(entry.outputs) == {"cleaning_fib", "cleaning_fib_thumbnail"}
        (reference,) = grid_outputs(experiment, grid, "cleaning_fib")
        assert reference.startswith(
            str(tmp_path / "exp" / "grids" / "grid-aspen" / "cryo_clean" / "cleaned-")
        )
        image = FibsemImage.load(reference)
        assert image.metadata.image_settings.beam_type is BeamType.ION
        assert image.metadata.image_settings.hfw == pytest.approx(900e-6)
        thumbnail = latest_grid_output(experiment, grid, "cleaning_fib_thumbnail")
        assert thumbnail.endswith("-thumbnail.png")
        # the reference is taken at the imaging current, not the cleaning one
        assert microscope.get_beam_current(BeamType.ION) == pytest.approx(
            IMAGING_CURRENT
        )

    def test_moves_to_the_slot_at_the_orientation(self, microscope, experiment):
        grid = experiment.get_grid_by_name("grid-aspen")
        run_grid_task(microscope, "cryo_clean", experiment, grid)
        centre = microscope.get_target_position(
            microscope._stage.holder.slots["Slot-01"].position, "SEM"
        )
        after = microscope.get_stage_position()
        assert after.x == pytest.approx(centre.x, abs=1e-6)
        assert after.y == pytest.approx(centre.y, abs=1e-6)

    def test_without_a_reference_records_nothing(self, microscope, experiment):
        experiment.grid_protocol.task_config["cryo_clean"].acquire_reference = False
        grid = experiment.get_grid_by_name("grid-aspen")
        task = run_grid_task(microscope, "cryo_clean", experiment, grid)
        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
        assert grid.task_history[-1].outputs == {}
        assert task.result_images == {}

    def test_refuses_a_grid_that_is_not_in_a_slot(self, microscope, experiment):
        grid = experiment.add_grid(GridRecord(name="grid-nowhere"))
        with pytest.raises(RuntimeError, match="not in a holder slot"):
            run_grid_task(microscope, "cryo_clean", experiment, grid)
        assert grid.task_state.status is AutoLamellaTaskStatus.Failed

    def test_a_stop_is_a_cancellation(self, microscope, experiment):
        class Manager:
            abort_token = threading.Event()
            hook_manager = None

            @property
            def should_abort(self):
                return self.abort_token.is_set()

            def hook_run_context(self):
                return {}

        manager = Manager()
        manager.abort_token.set()
        grid = experiment.get_grid_by_name("grid-aspen")
        with pytest.raises(Exception):
            run_grid_task(
                microscope, "cryo_clean", experiment, grid, task_manager=manager
            )
        assert grid.task_state.status is AutoLamellaTaskStatus.Cancelled
        assert microscope.get_beam_current(BeamType.ION) == pytest.approx(
            IMAGING_CURRENT
        )
