"""Which experiment produced an image is a property of the run, not of the GUI.

Registration used to happen only in AutoLamellaUI, so images acquired from a script
or a headless run carried the microscope's default ``FibsemExperimentRef()`` -- id
``None`` -- and could not be associated with an experiment afterwards. See FIB-449.
"""

import os
from pathlib import Path

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.microscope import FibsemMicroscope


@pytest.fixture
def microscope() -> FibsemMicroscope:
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-experiment")
    exp.task_protocol = AutoLamellaTaskProtocol()
    # Constructing an Experiment deliberately does not write to disk (FIB-420);
    # Experiment.create() is what makes the directory in the app.
    os.makedirs(exp.path, exist_ok=True)
    return exp


def test_a_fresh_microscope_records_no_experiment(
    microscope: FibsemMicroscope,
) -> None:
    """The gap this closes: without registration there is nothing to associate."""
    assert microscope.experiment.id is None


def test_register_metadata_stamps_the_experiment_onto_the_microscope(
    microscope: FibsemMicroscope, experiment: Experiment
) -> None:
    experiment.register_metadata(microscope)

    # The UUID is the join key; the name is the display string (FIB-446).
    assert microscope.experiment.id == experiment.id
    assert microscope.experiment.name == "test-experiment"
    assert microscope.experiment.application == "autolamella"


def test_creating_a_task_manager_registers_without_a_gui(
    microscope: FibsemMicroscope, experiment: Experiment
) -> None:
    """A headless run goes through TaskManager, never through AutoLamellaUI."""
    TaskManager(microscope=microscope, experiment=experiment, parent_ui=None)

    assert microscope.experiment.id == experiment.id
    assert microscope.experiment.name == "test-experiment"


def test_loading_an_experiment_does_not_touch_the_microscope(
    microscope: FibsemMicroscope, experiment: Experiment, tmp_path: Path
) -> None:
    """Mirrors configure_logging (FIB-421): reading an experiment must not reach
    into the caller's microscope. The load dialog reads one on every click to
    preview it, and previewing is not adopting."""
    experiment.save()

    Experiment.load(Path(experiment.path) / "experiment.yaml")

    assert microscope.experiment.id is None
