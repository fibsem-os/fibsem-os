"""The grid task lifecycle, its config round trip, the protocol, and recorded outputs.

A grid task is a thin wrapper: state and history on the GridRecord, outputs by
role relative to the grid's directory, hooks and the stop token. The operation
inside `_run` is the test's own, so nothing here touches a runner.
"""

import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Type

import pytest
import yaml

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    GridTaskProtocol,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    GRID_TASK_REGISTRY,
    GridTask,
    GridTaskConfig,
    load_grid_task_configs,
    register_grid_task,
    run_grid_task,
)

# ---------------------------------------------------------------------------
# A test task: writes one file, can be told to fail or to block on the token
# ---------------------------------------------------------------------------


class Flavour(Enum):
    PLAIN = "plain"
    SPICY = "spicy"


@dataclass
class Tile:
    rows: int = 1
    cols: int = 1

    def to_dict(self):
        return {"rows": self.rows, "cols": self.cols}

    @classmethod
    def from_dict(cls, data):
        return cls(rows=int(data.get("rows", 1)), cols=int(data.get("cols", 1)))


@dataclass
class EchoConfig(GridTaskConfig):
    task_type: ClassVar[str] = "ECHO_GRID"
    display_name: ClassVar[str] = "Echo"
    label: str = "hello"
    repeats: int = 1
    flavour: Flavour = Flavour.PLAIN
    tile: Tile = field(default_factory=Tile)
    fail: bool = False
    wait_for_abort: bool = False


@register_grid_task
class EchoTask(GridTask):
    config_cls: ClassVar[Type[GridTaskConfig]] = EchoConfig
    config: EchoConfig

    def _run(self) -> None:
        if self.config.wait_for_abort:
            self._check_for_abort()
        if self.config.fail:
            raise RuntimeError("echo failed on purpose")
        for i in range(self.config.repeats):
            path = self.output_dir / f"{self.config.label}-{i}.txt"
            path.write_text(self.config.flavour.value)
            self.record_output("echo", path)
        self.log_status_message("WROTE", "Wrote files")


class FakeManager:
    """What a task reads off the manager: the token, should_abort, hooks."""

    def __init__(self) -> None:
        self.abort_token = threading.Event()
        self.hook_manager = None

    @property
    def should_abort(self) -> bool:
        return self.abort_token.is_set()

    def hook_run_context(self):
        return {"run_id": "test-run"}


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    return microscope


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(EchoConfig(task_name="echo"))
    return exp


def _grid(experiment, name="grid-aspen") -> GridRecord:
    return experiment.add_grid(GridRecord(name=name))


# ---------------------------------------------------------------------------
# Config round trip
# ---------------------------------------------------------------------------


class TestConfig:
    def test_parameters_are_the_task_specific_fields(self):
        assert EchoConfig().parameters == (
            "label",
            "repeats",
            "flavour",
            "tile",
            "fail",
            "wait_for_abort",
        )

    def test_flat_round_trip_with_enum_and_nested_dataclass(self):
        config = EchoConfig(
            task_name="echo",
            label="x",
            repeats=3,
            flavour=Flavour.SPICY,
            tile=Tile(2, 3),
        )
        data = yaml.safe_load(yaml.safe_dump(config.to_dict()))
        assert data["task_type"] == "ECHO_GRID"
        assert data["flavour"] == "SPICY"
        assert data["tile"] == {"rows": 2, "cols": 3}
        assert "parameters" not in data  # flat, not a sub-dict
        again = EchoConfig.from_dict(data)
        assert again == config

    def test_unknown_field_is_ignored_with_a_warning(self, caplog):
        again = EchoConfig.from_dict(
            {"task_type": "ECHO_GRID", "label": "y", "bogus": 1}
        )
        assert again.label == "y"
        assert "bogus" in caplog.text


# ---------------------------------------------------------------------------
# Protocol on the experiment
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_add_keeps_order_and_names_are_keys(self):
        protocol = GridTaskProtocol()
        protocol.add(EchoConfig(task_name="overview_sem"))
        protocol.add(EchoConfig(task_name="overview_fib"))
        assert protocol.ordered_task_names == ["overview_sem", "overview_fib"]
        protocol.remove("overview_sem")
        assert protocol.ordered_task_names == ["overview_fib"]

    def test_a_nameless_config_is_refused(self):
        with pytest.raises(ValueError):
            GridTaskProtocol().add(EchoConfig())

    def test_is_a_section_of_protocol_yaml(self, experiment, tmp_path):
        experiment.grid_protocol.task_config["echo"].repeats = 4
        experiment.save(save_protocol=True)
        data = yaml.safe_load((tmp_path / "exp" / "protocol.yaml").read_text())
        assert data["grid_tasks"]["tasks"]["echo"]["repeats"] == 4
        assert "grid_protocol" not in experiment.to_dict()  # not experiment state

        loaded = Experiment.load(tmp_path / "exp" / "experiment.yaml")
        assert loaded.grid_protocol is loaded.task_protocol.grid_tasks
        assert loaded.grid_protocol.id == experiment.grid_protocol.id
        assert loaded.grid_protocol.ordered_task_names == ["echo"]
        assert isinstance(loaded.grid_protocol.task_config["echo"], EchoConfig)
        assert loaded.grid_protocol.task_config["echo"].repeats == 4

    def test_a_protocol_written_before_grid_tasks_offers_none(self):
        data = AutoLamellaTaskProtocol().to_dict()
        del data["grid_tasks"]
        assert AutoLamellaTaskProtocol.from_dict(data).grid_tasks.task_config == {}

    def test_requires_a_task_protocol(self, tmp_path):
        exp = Experiment(path=tmp_path, name="exp")
        assert exp.task_protocol is None
        with pytest.raises(ValueError, match="No task protocol"):
            exp.grid_protocol
        exp.task_protocol = AutoLamellaTaskProtocol()
        assert exp.grid_protocol is exp.task_protocol.grid_tasks


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_completed_run_is_on_the_history_with_its_outputs(
        self, microscope, experiment
    ):
        grid = _grid(experiment)
        experiment.grid_protocol.task_config["echo"].repeats = 2
        task = run_grid_task(microscope, "echo", experiment, grid)

        assert grid.task_state.status is AutoLamellaTaskStatus.Completed
        assert len(grid.task_history) == 1
        entry = grid.task_history[0]
        assert entry.name == "echo" and entry.task_type == "ECHO_GRID"
        assert entry.task_id == task.task_id
        assert entry.end_timestamp is not None
        # relative to the grid's directory, under the task's own folder
        assert entry.outputs == {"echo": ["echo/hello-0.txt", "echo/hello-1.txt"]}
        assert (experiment.grid_path(grid) / "echo" / "hello-1.txt").exists()
        assert grid.has_completed_task("echo")

    def test_output_layout_is_grids_name_task(self, microscope, experiment, tmp_path):
        grid = _grid(experiment, "grid-birch")
        run_grid_task(microscope, "echo", experiment, grid)
        assert (
            tmp_path / "exp" / "grids" / "grid-birch" / "echo" / "hello-0.txt"
        ).exists()

    def test_the_same_output_recorded_twice_is_one_file(self, microscope, experiment):
        grid = _grid(experiment)
        task = EchoTask(microscope, EchoConfig(task_name="echo"), grid, experiment)
        task.pre_task()
        path = task.output_dir / "same.txt"
        path.write_text("x")
        task.record_output("echo", path)
        task.record_output("echo", str(path))
        assert grid.task_state.outputs["echo"] == ["echo/same.txt"]

    def test_task_state_is_reset_per_run_not_replaced(self, microscope, experiment):
        grid = _grid(experiment)
        state = grid.task_state
        run_grid_task(microscope, "echo", experiment, grid)
        run_grid_task(microscope, "echo", experiment, grid)
        assert grid.task_state is state
        assert len(grid.task_history) == 2
        assert state.outputs == {"echo": ["echo/hello-0.txt"]}

    def test_failure_is_recorded_then_raised(self, microscope, experiment):
        grid = _grid(experiment)
        experiment.grid_protocol.task_config["echo"].fail = True
        with pytest.raises(RuntimeError, match="on purpose"):
            run_grid_task(microscope, "echo", experiment, grid)
        assert grid.task_state.status is AutoLamellaTaskStatus.Failed
        assert grid.is_failure
        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Failed
        assert "on purpose" in grid.task_history[-1].status_message

    def test_a_stop_is_a_cancellation_not_a_failure(self, microscope, experiment):
        grid = _grid(experiment)
        experiment.grid_protocol.task_config["echo"].wait_for_abort = True
        manager = FakeManager()
        manager.abort_token.set()
        with pytest.raises(InterruptedError):
            run_grid_task(microscope, "echo", experiment, grid, task_manager=manager)
        assert grid.task_state.status is AutoLamellaTaskStatus.Cancelled
        assert not grid.is_failure

    def test_run_reads_a_copy_of_the_saved_config(self, microscope, experiment):
        grid = _grid(experiment)
        task = run_grid_task(microscope, "echo", experiment, grid)
        task.config.label = "mutated"
        assert experiment.grid_protocol.task_config["echo"].label == "hello"

    def test_unknown_task_name_is_refused(self, microscope, experiment):
        with pytest.raises(KeyError, match="not in the grid protocol"):
            run_grid_task(microscope, "nope", experiment, _grid(experiment))

    def test_slot_is_resolved_live_from_the_holder(self, microscope, experiment):
        from fibsem.microscopes._stage import SampleGrid

        grid = _grid(experiment, "grid-cedar")
        task = EchoTask(microscope, EchoConfig(task_name="echo"), grid, experiment)
        assert task.slot is None
        microscope._stage.assign_grid(
            "Slot-01", SampleGrid(name="grid-cedar"), persist=False
        )
        assert task.slot.name == "Slot-01"

    def test_registry_knows_the_task(self):
        assert GRID_TASK_REGISTRY["ECHO_GRID"] is EchoTask


# ---------------------------------------------------------------------------
# Reading outputs back
# ---------------------------------------------------------------------------


class TestReadingOutputs:
    def test_grid_outputs_resolve_existing_files_only(self, microscope, experiment):
        from fibsem.applications.autolamella.task_outputs import (
            grid_outputs,
            latest_grid_output,
        )

        grid = _grid(experiment)
        experiment.grid_protocol.task_config["echo"].repeats = 2
        run_grid_task(microscope, "echo", experiment, grid)
        run_grid_task(microscope, "echo", experiment, grid)  # same files again
        paths = grid_outputs(experiment, grid, "echo")
        assert [os.path.basename(p) for p in paths] == ["hello-0.txt", "hello-1.txt"]
        assert grid_outputs(experiment, grid, "nothing") == []
        assert latest_grid_output(experiment, grid, "echo").endswith("hello-1.txt")

        (experiment.grid_path(grid) / "echo" / "hello-1.txt").unlink()
        assert latest_grid_output(experiment, grid, "echo").endswith("hello-0.txt")

    def test_grid_outputs_can_be_narrowed_to_one_task(self, microscope, experiment):
        from fibsem.applications.autolamella.task_outputs import grid_outputs

        experiment.grid_protocol.add(EchoConfig(task_name="echo2", label="other"))
        grid = _grid(experiment)
        run_grid_task(microscope, "echo", experiment, grid)
        run_grid_task(microscope, "echo2", experiment, grid)
        assert len(grid_outputs(experiment, grid, "echo")) == 2
        only = grid_outputs(experiment, grid, "echo", task_name="echo2")
        assert [os.path.basename(p) for p in only] == ["other-0.txt"]
