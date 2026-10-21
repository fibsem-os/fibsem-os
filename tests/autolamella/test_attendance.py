"""What a task needs from a person, said before the run.

Derived from what the task type declares (``asks``) and the attention the
protocol gives it; never stored.
"""

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
)
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.attendance import (
    KIND_LABELS,
    attendance,
    attendance_for,
    run_attendance,
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
from fibsem.workflows.attention import Attention
from fibsem.workflows.site_tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from tests.protocols import planned, task_protocol

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"
UNDERCUT = "Mill Undercut"
REF = "Acquire Reference Image"


def test_a_supervised_task_with_a_session_needs_you_there():
    att = attendance(
        MillUndercutTask,
        MillUndercutTaskConfig(task_name=UNDERCUT),
        Attention.supervised,
    )
    assert att.needs_a_person_present
    assert att.present == ("detection", "milling")
    assert att.line == (
        "Needs you at the microscope while it runs: detection and milling."
    )


def test_a_task_that_asks_only_its_mill_needs_you_for_that():
    att = attendance(
        MillRoughTask,
        MillRoughTaskConfig(task_name=ROUGH),
        Attention.supervised,
    )
    assert att.present == ("milling",), "the mill is a session"
    auto = attendance(
        AcquireReferenceImageTask,
        AcquireReferenceImageConfig(task_name=REF),
        Attention.supervised,
    )
    assert auto.present == ("the position",)
    assert auto.line == ("Needs you at the microscope while it runs: the position.")


def test_setups_settings_turn_its_questions_off():
    on = SelectMillingPositionTaskConfig(
        task_name=SETUP, auto_milling_alignment=True, select_poi=True
    )
    off = SelectMillingPositionTaskConfig(
        task_name=SETUP, auto_milling_alignment=False, select_poi=False
    )
    with_all = attendance(SelectMillingPositionTask, on, Attention.supervised)
    assert with_all.present == (
        "the position",
        "detection",
        "the point of interest",
        "the alignment area",
    )
    fewer = attendance(SelectMillingPositionTask, off, Attention.supervised)
    assert fewer.present == ("the position", "the alignment area")
    assert fewer.line == (
        "Needs you at the microscope while it runs: the position and the "
        "alignment area."
    ), "nothing waits on it once it ends"


def test_automated_nobody_is_asked():
    att = attendance(
        SelectMillingPositionTask,
        SelectMillingPositionTaskConfig(task_name=SETUP),
        Attention.automated,
    )
    assert not att.needs_a_person_present
    assert att.line == "Nobody is asked."
    alone = attendance(
        SelectMillingPositionTask,
        SelectMillingPositionTaskConfig(task_name=SETUP),
        Attention.automated,
    )
    assert alone.line == "Nobody is asked."


def _protocol() -> AutoLamellaTaskProtocol:
    protocol = task_protocol(
        planned(
            name=SETUP,
            attention=Attention.supervised,
            config=SelectMillingPositionTaskConfig(
                auto_milling_alignment=False, select_poi=True
            ),
        ),
        planned(
            name=ROUGH,
            attention=Attention.automated,
            requires=[SETUP],
            config=MillRoughTaskConfig(),
        ),
    )
    # A task the plan lists with no config.
    protocol.workflow_config.add("Ghost")
    return protocol


def test_the_protocol_names_the_type_and_the_attention():
    protocol = _protocol()
    setup = attendance_for(protocol, SETUP)
    assert setup.attention is Attention.supervised
    assert setup.present == (
        "the position",
        "the point of interest",
        "the alignment area",
    )
    rough = attendance_for(protocol, ROUGH)
    assert rough.attention is Attention.automated
    assert attendance_for(protocol, "Ghost") is None, "no config, nothing said"


def test_the_run_summary_names_who_needs_you():
    protocol = _protocol()
    assert run_attendance(protocol, [SETUP, ROUGH]).summary == (
        f"This run needs you present for {SETUP}. Everything else runs on its own."
    )
    assert run_attendance(protocol, [ROUGH]).summary == "This run needs nobody present."
    protocol.task_config[ROUGH].attention = Attention.supervised
    assert run_attendance(protocol, [SETUP, ROUGH]).summary.startswith(
        "This run needs you present for every task:"
    )


def test_every_task_type_declares_tuples_the_words_are_known_for():
    for name, cls in get_tasks().items():
        assert isinstance(cls.asks, tuple), name
        for kind in cls.asks:
            assert kind in KIND_LABELS, f"{name}: no words for {kind!r}"
