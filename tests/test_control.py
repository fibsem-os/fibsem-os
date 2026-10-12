"""The control authority (FIB-1270): one action at a time, refused not queued,
lent during a question."""

import threading
import time

import pytest

from fibsem.control import (
    AGENT,
    OPERATOR,
    SCRIPT,
    WORKFLOW,
    Control,
    MicroscopeBusy,
    lending,
)


def test_a_second_take_is_refused_naming_who_is_in_control():
    control = Control()
    with control.take(WORKFLOW, "a lamellae run"):
        with pytest.raises(MicroscopeBusy) as refused:
            with control.take(AGENT, "POST /move_stage_relative"):
                pass
    assert refused.value.in_control.by == WORKFLOW
    assert "workflow" in str(refused.value)
    assert control.current is None


def test_release_frees_control_even_when_the_action_fails():
    control = Control()
    with pytest.raises(ValueError):
        with control.take(SCRIPT, "tilt.py"):
            raise ValueError("boom")
    with control.take(AGENT, "next"):
        assert control.current.by == AGENT


def test_changed_emits_on_take_and_release():
    control = Control()
    seen = []
    control.changed.connect(seen.append)
    with control.take(SCRIPT, "tilt.py"):
        pass
    assert [s.by if s else None for s in seen] == [SCRIPT, None]


def test_operator_actions_are_timed_and_nobody_elses_are():
    control = Control()
    with control.take(AGENT, "move"):
        pass
    assert control.last_operator_action is None
    before = time.time()
    with control.take(OPERATOR, "acquire"):
        pass
    assert control.last_operator_action >= before


def test_lent_control_lets_one_other_action_in_at_a_time():
    control = Control()
    with control.take(WORKFLOW, "run"), control.lend():
        with control.take(AGENT, "move"):
            assert control.current.by == AGENT
            with pytest.raises(MicroscopeBusy) as refused:
                with control.take(OPERATOR, "acquire"):
                    pass
            assert refused.value.in_control.by == AGENT
        assert control.current.by == WORKFLOW
    # The question ended: lending ends with it.
    with control.take(WORKFLOW, "run"):
        pass


def test_after_the_question_nothing_borrows():
    control = Control()
    with control.take(WORKFLOW, "run"):
        with control.lend():
            pass
        with pytest.raises(MicroscopeBusy):
            with control.take(AGENT, "move"):
                pass


def test_lend_waits_for_the_borrowers_action_to_finish():
    control = Control()
    borrowed = threading.Event()
    order = []

    def borrower():
        with control.take(AGENT, "move"):
            borrowed.set()
            time.sleep(0.2)
            order.append("borrower released")

    with control.take(WORKFLOW, "run"):
        thread = threading.Thread(target=borrower)
        with control.lend():
            thread.start()
            assert borrowed.wait(5)
        order.append("question over")
        assert control.current.by == WORKFLOW
    thread.join(5)
    assert order == ["borrower released", "question over"]


def test_lending_lends_what_this_thread_took_and_nothing_else():
    control = Control()
    with lending():  # holds nothing: lends nothing
        pass
    with control.take(WORKFLOW, "run"):
        with lending():
            with control.take(AGENT, "move"):
                pass

    # Another thread's control is not this thread's to lend.
    taken = threading.Event()
    done = threading.Event()

    def other():
        with control.take(SCRIPT, "tilt.py"):
            taken.set()
            done.wait(5)

    thread = threading.Thread(target=other)
    thread.start()
    assert taken.wait(5)
    try:
        with lending():
            with pytest.raises(MicroscopeBusy):
                with control.take(AGENT, "move"):
                    pass
    finally:
        done.set()
        thread.join(5)


def test_release_from_another_thread_is_allowed():
    # A server dependency enters on one pool thread and exits on another.
    control = Control()
    taking = control.take(AGENT, "move")
    taking.__enter__()
    thread = threading.Thread(target=taking.__exit__, args=(None, None, None))
    thread.start()
    thread.join(5)
    assert control.current is None
    with control.take(WORKFLOW, "run"):
        pass


def test_every_microscope_has_its_own_control():
    from fibsem import utils

    first, _ = utils.setup_session(manufacturer="Demo")
    second, _ = utils.setup_session(manufacturer="Demo")
    assert isinstance(first.control, Control)
    assert first.control is first.control
    assert first.control is not second.control
