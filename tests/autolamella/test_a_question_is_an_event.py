"""A question a task asks mid-run is said on the record (FIB-1050).

``ask`` raises no prompt, so nothing in the event stream told a watcher the
run had stopped to be told something. The manager's hold on the question
records ``question_asked`` when it is raised and ``question_released`` when
it is let go, through the microscope's ``record_event``, which is what the
event stream taps. A microscope without one records nothing.
"""

from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from tests.autolamella.test_task_manager_status import (
    NoMicroscope,
    RecordingUI,
    make_experiment,
)


class RecordingMicroscope(NoMicroscope):
    def __init__(self):
        super().__init__()
        self.recorded = []

    def record_event(self, kind, payload):
        self.recorded.append((kind, dict(payload)))


def test_the_hold_on_a_question_is_two_events(tmp_path):
    microscope = RecordingMicroscope()
    manager = TaskManager(
        microscope=microscope,
        experiment=make_experiment(tmp_path, lamella_names=["L1"]),
        parent_ui=RecordingUI(),
    )

    with manager.holding_a_question("L1", "Trench"):
        assert microscope.recorded == [
            ("question_asked", {"item_name": "L1", "task_name": "Trench"})
        ]
    assert microscope.recorded[-1] == (
        "question_released",
        {"item_name": "L1", "task_name": "Trench"},
    )


def test_a_microscope_with_no_record_records_nothing(tmp_path):
    manager = TaskManager(
        microscope=NoMicroscope(),
        experiment=make_experiment(tmp_path, lamella_names=["L1"]),
        parent_ui=RecordingUI(),
    )
    with manager.holding_a_question("L1", "Trench"):
        pass
