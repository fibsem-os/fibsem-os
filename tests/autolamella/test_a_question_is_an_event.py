"""A question a task asks mid-run is said on the record (FIB-1050).

``ask`` raises no prompt, so nothing in the event stream told a watcher the
run had stopped to be told something. The manager's hold on the question
records ``proposal_held`` when it is raised and ``proposal_released`` when
it is let go, through the microscope's ``record_event``, which is what the
event stream taps. A microscope without one records nothing.
"""

from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from tests.autolamella.test_task_manager_status import (
    NoMicroscope,
    _listened,
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
    manager = _listened(
        TaskManager(
            microscope=microscope,
            experiment=make_experiment(tmp_path, lamella_names=["L1"]),
        )
    )

    with manager.holding_a_proposal("L1", "Trench"):
        assert microscope.recorded == [
            ("proposal_held", {"item_name": "L1", "task_name": "Trench"})
        ]
    assert microscope.recorded[-1] == (
        "proposal_released",
        {"item_name": "L1", "task_name": "Trench"},
    )


def test_a_microscope_with_no_record_records_nothing(tmp_path):
    manager = _listened(
        TaskManager(
            microscope=NoMicroscope(),
            experiment=make_experiment(tmp_path, lamella_names=["L1"]),
        )
    )
    with manager.holding_a_proposal("L1", "Trench"):
        pass


def test_who_can_answer_changing_is_on_the_record(tmp_path):
    """While a question waits, the window opening or the agent going quiet is
    said once, as ``proposal_waiting``, so a watcher can tell why it waits."""
    from fibsem.acting import OPERATOR

    microscope = RecordingMicroscope()
    who = []
    manager = _listened(
        TaskManager(
            microscope=microscope,
            experiment=make_experiment(tmp_path, lamella_names=["L1"]),
            deciders=lambda: tuple(who),
        )
    )

    with manager.holding_a_proposal("L1", "Trench") as refresh:
        refresh()
        assert [kind for kind, _ in microscope.recorded] == ["proposal_held"]
        who.append(OPERATOR)
        refresh()
        refresh()
    assert microscope.recorded[1] == (
        "proposal_waiting",
        {"item_name": "L1", "task_name": "Trench", "deciders": ["operator"]},
    )
    assert [kind for kind, _ in microscope.recorded][2:] == ["proposal_released"]
