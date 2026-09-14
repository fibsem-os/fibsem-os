"""A lamella run is refused, by name, while a linked lamella's grid is off the
stage (FIB-71). See `workflows/grid_gate.py`."""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.workflows.grid_gate import (
    lamellae_off_the_stage,
    run_refusal,
)
from fibsem.structures import MicroscopeState


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(tmp_path, arctis):
    exp = Experiment(path=tmp_path, name="gate")
    (tmp_path / "gate").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.sync_grids_from_inventory(arctis._stage)
    return exp


def _lamella(experiment, name, grid_name):
    grid = experiment.get_grid_by_name(grid_name) if grid_name else None
    experiment.add_new_lamella(
        microscope_state=MicroscopeState(),
        task_config=experiment.task_protocol.task_config,
        name=name,
        grid_id=grid.id if grid else None,
    )
    return experiment.positions[-1]


def test_nothing_loaded_refuses_every_linked_lamella_and_no_unlinked_one(
    experiment, arctis
):
    stage = arctis._stage
    on_two = _lamella(experiment, "on-two", "Grid-02")
    on_three = _lamella(experiment, "on-three", "Grid-03")
    also_two = _lamella(experiment, "also-two", "Grid-02")
    free = _lamella(experiment, "free", None)
    assert stage.loaded_grids == []
    assert lamellae_off_the_stage(
        experiment, stage, [on_two, on_three, also_two, free]
    ) == {
        "Grid-02": ["on-two", "also-two"],
        "Grid-03": ["on-three"],
    }
    assert lamellae_off_the_stage(experiment, stage, [free]) == {}


def test_loading_the_grid_clears_its_lamellae(experiment, arctis):
    stage = arctis._stage
    on_two = _lamella(experiment, "on-two", "Grid-02")
    on_three = _lamella(experiment, "on-three", "Grid-03")
    stage.ensure_loaded("Grid-02")
    assert lamellae_off_the_stage(experiment, stage, [on_two, on_three]) == {
        "Grid-03": ["on-three"]
    }
    assert run_refusal(experiment, stage, [on_two]) == ""


def test_the_refusal_names_the_lamellae_and_the_grid(experiment, arctis):
    stage = arctis._stage
    on_two = _lamella(experiment, "on-two", "Grid-02")
    also_two = _lamella(experiment, "also-two", "Grid-02")
    assert run_refusal(experiment, stage, [on_two]) == (
        "Cannot run: on-two is on Grid-02, which is not on the stage. "
        "Load Grid-02 from the Grids tab first, or leave that lamella out."
    )
    assert run_refusal(experiment, stage, [on_two, also_two]) == (
        "Cannot run: on-two, also-two are on Grid-02, which is not on the stage. "
        "Load Grid-02 from the Grids tab first, or leave those lamellae out."
    )


def test_a_grid_with_no_record_is_not_refused(experiment, arctis):
    """The rule is about a known grid being elsewhere, not missing bookkeeping."""
    stage = arctis._stage
    on_two = _lamella(experiment, "on-two", "Grid-02")
    experiment.remove_grid("Grid-02")
    assert on_two.grid_id is None
    assert run_refusal(experiment, stage, [on_two]) == ""


def test_the_inventory_is_read_first(experiment, arctis, monkeypatch):
    """The answer is the hardware's: the read happens before the check."""
    stage = arctis._stage
    on_two = _lamella(experiment, "on-two", "Grid-02")
    calls = []
    original = stage.get_inventory
    monkeypatch.setattr(
        stage, "get_inventory", lambda: (calls.append(1), original())[1]
    )
    run_refusal(experiment, stage, [on_two])
    assert calls == [1]
