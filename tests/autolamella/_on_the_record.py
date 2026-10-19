"""A run for a task built on its own: an experiment holding its lamella, and
a manager over it.

A task asks its questions on the record (``BaseTask.ask``), which
needs an experiment to record them on. A test that builds a task by hand,
rather than running it (``tests.run_state.run_lamellae``), gives it one with this.
"""

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager


def manager_for(microscope, lamella: Lamella) -> TaskManager:
    """A headless manager whose experiment holds ``lamella``."""
    experiment = Experiment(path=lamella.path.parent, name="on-the-record")
    experiment.positions.append(lamella)
    return TaskManager(microscope=microscope, experiment=experiment)
