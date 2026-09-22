"""What a task needs from a person, said before the run.

Derived from what the task type declares (``questions``, ``sessions``,
``proposer``) and the attention the protocol gives it; never stored.
"""

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
)
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.attendance import (
    KIND_LABELS,
    attendance,
    attendance_for,
    run_attendance,
)
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTask,
    MillUndercutTaskConfig,
)

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"
UNDERCUT = "Mill Undercut"
REF = "Acquire Reference Image"


def test_a_supervised_task_with_a_session_needs_you_there():
    att = attendance(
        MillUndercutTask,
        MillUndercutTaskConfig(task_name=UNDERCUT),
        Attention.supervised,
        waiters=["Polishing"],
    )
    assert att.needs_a_person_present
    assert att.present == ("detection", "milling")
    assert att.line == (
        "Needs you at the microscope while it runs: detection and milling. "
        "Polishing waits for your decision on its result."
    )


def test_a_task_that_asks_nothing_runs_on_its_own_and_its_value_waits():
    att = attendance(
        MillRoughTask,
        MillRoughTaskConfig(task_name=ROUGH),
        Attention.supervised,
        waiters=[],
    )
    assert att.present == ("milling",), "the mill is a session"
    auto = attendance(
        AcquireReferenceImageTask,
        AcquireReferenceImageConfig(task_name=REF),
        Attention.supervised,
    )
    assert auto.present == ("the position",)
    assert auto.line == (
        "Needs you at the microscope while it runs: the position. "
        "Nothing waits on its result; it is listed to check."
    )


def test_setups_settings_turn_its_questions_off():
    on = SelectMillingPositionTaskConfig(
        task_name=SETUP, auto_milling_alignment=True, select_poi=True
    )
    off = SelectMillingPositionTaskConfig(
        task_name=SETUP, auto_milling_alignment=False, select_poi=False
    )
    with_all = attendance(
        SelectMillingPositionTask, on, Attention.supervised, waiters=[ROUGH]
    )
    assert with_all.present == (
        "the position",
        "detection",
        "the point of interest",
        "the alignment area",
    )
    fewer = attendance(
        SelectMillingPositionTask, off, Attention.supervised, waiters=[ROUGH]
    )
    assert fewer.present == ("the position", "the alignment area")
    assert fewer.line.endswith(
        f"{ROUGH} waits for your decision on the point of interest."
    )


def test_automated_nobody_is_asked_and_the_value_is_open_until_used():
    att = attendance(
        SelectMillingPositionTask,
        SelectMillingPositionTaskConfig(task_name=SETUP),
        Attention.automated,
        waiters=[ROUGH],
    )
    assert not att.needs_a_person_present
    assert att.line == (
        f"Nobody is asked. The point of interest is open to correct until {ROUGH} starts."
    )
    alone = attendance(
        SelectMillingPositionTask,
        SelectMillingPositionTaskConfig(task_name=SETUP),
        Attention.automated,
    )
    assert alone.line == "Nobody is asked."


def test_with_the_review_workflow_off_nothing_waits_afterwards():
    att = attendance(
        MillRoughTask,
        MillRoughTaskConfig(task_name=ROUGH),
        Attention.supervised,
        waiters=["Polishing"],
        review_on=False,
    )
    assert att.line == "Needs you at the microscope while it runs: milling."


def _protocol() -> AutoLamellaTaskProtocol:
    protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SETUP, required=True, attention=Attention.supervised
                ),
                AutoLamellaTaskDescription(
                    name=ROUGH,
                    required=True,
                    attention=Attention.automated,
                    requires=[SETUP],
                ),
                AutoLamellaTaskDescription(name="Ghost", required=True),
            ]
        )
    )
    protocol.task_config = EventedDict(
        {
            SETUP: SelectMillingPositionTaskConfig(
                task_name=SETUP, auto_milling_alignment=False, select_poi=True
            ),
            ROUGH: MillRoughTaskConfig(task_name=ROUGH),
        }
    )
    return protocol


def test_the_protocol_names_the_type_the_attention_and_who_waits():
    protocol = _protocol()
    setup = attendance_for(protocol, SETUP)
    assert setup.attention is Attention.supervised
    assert setup.waiters == (ROUGH,)
    assert setup.present == (
        "the position",
        "the point of interest",
        "the alignment area",
    )
    rough = attendance_for(protocol, ROUGH)
    assert rough.attention is Attention.automated and rough.waiters == ()
    assert attendance_for(protocol, "Ghost") is None, "no config, nothing said"


def test_the_run_summary_names_who_needs_you():
    protocol = _protocol()
    assert run_attendance(protocol, [SETUP, ROUGH]) == (
        f"This run needs you present for {SETUP}. Everything else runs on its own."
    )
    assert run_attendance(protocol, [ROUGH]) == "This run needs nobody present."
    protocol.workflow_config.tasks[1].attention = Attention.supervised
    assert run_attendance(protocol, [SETUP, ROUGH]).startswith(
        "This run needs you present for every task:"
    )


def test_every_task_type_declares_tuples_the_words_are_known_for():
    for name, cls in get_tasks().items():
        assert isinstance(cls.questions, tuple) and isinstance(cls.sessions, tuple), (
            name
        )
        for kind in cls.questions:
            assert kind in KIND_LABELS, f"{name}: no words for {kind!r}"
        for session in cls.sessions:
            assert isinstance(session, str) and session, name
