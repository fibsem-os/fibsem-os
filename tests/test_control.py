"""The control authority (FIB-1270): one action at a time, refused not queued,
lent during a question."""

import threading
import time

import pytest

from fibsem.acting import AGENT, OPERATOR, SCRIPT, WORKFLOW
from fibsem.control import (
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


def test_last_is_the_most_recent_take_and_outlives_release():
    control = Control()
    assert control.last is None
    with control.take(AGENT, "POST /start_milling"):
        pass
    assert control.current is None
    assert control.last.by == AGENT
    with control.take(WORKFLOW, "run"), control.lend():
        with control.take(OPERATOR, "acquire"):
            assert control.last.by == OPERATOR
    with pytest.raises(MicroscopeBusy):
        with control.take(WORKFLOW, "run"):
            with control.take(AGENT, "move"):
                pass
    assert control.last.by == WORKFLOW  # a refused take is not a take


def _opened(control, by, action):
    """A long command's control, as the App Server keeps it: taken, then let
    its starter act beside it."""
    taking = control.take(by, action)
    entry = taking.__enter__()
    control.let_act_beside(entry)
    return taking


def test_the_starter_of_a_long_command_acts_beside_it_one_action_at_a_time():
    control = Control()
    mill = _opened(control, AGENT, "milling run_stage")
    with control.take(AGENT, "FIB · Take an image") as image:
        assert control.last is image
        assert control.current.action == "milling run_stage"
        with pytest.raises(MicroscopeBusy) as refused:
            with control.take(AGENT, "Stage · Move"):
                pass
        assert refused.value.in_control.action == "FIB · Take an image"
    for other in (OPERATOR, WORKFLOW):
        with pytest.raises(MicroscopeBusy):
            with control.take(other, "acquire"):
                pass
    mill.__exit__(None, None, None)
    assert control.current is None
    # A command not opened is acted beside by nobody.
    with control.take(AGENT, "move"):
        with pytest.raises(MicroscopeBusy):
            with control.take(AGENT, "image"):
                pass


def test_a_long_command_ends_after_what_acts_beside_it():
    control = Control()
    mill = _opened(control, AGENT, "milling run_stage")
    beside = threading.Event()
    order = []

    def image():
        with control.take(AGENT, "image"):
            beside.set()
            time.sleep(0.2)
            order.append("image done")

    thread = threading.Thread(target=image)
    thread.start()
    assert beside.wait(5)
    mill.__exit__(None, None, None)
    order.append("mill released")
    thread.join(5)
    assert order == ["image done", "mill released"]
    assert control.current is None


def test_a_run_lending_control_waits_for_the_agent_s_mill():
    """The run's question ends while the agent's mill goes on: the run waits
    for the mill, not only for the request that started it."""
    control = Control()
    released = threading.Event()
    order = []

    with control.take(WORKFLOW, "run"):
        with control.lend():
            mill = _opened(control, AGENT, "milling run_stage")
            with control.take(AGENT, "image"):  # beside the mill, not refused
                pass

            def end_mill():
                time.sleep(0.2)
                order.append("mill ended")
                mill.__exit__(None, None, None)
                released.set()

            threading.Thread(target=end_mill).start()
        order.append("run carries on")
        assert control.current.by == WORKFLOW
    assert released.wait(5)
    assert order == ["mill ended", "run carries on"]


def test_takes_that_join_share_control_and_the_holder_waits_for_them():
    control = Control()
    with control.take(OPERATOR, "mill", join=True) as mill:
        with control.take(OPERATOR, "acquire", join=True) as joined:
            assert joined is mill and control.current is mill
        with pytest.raises(MicroscopeBusy):
            with control.take(AGENT, "move"):
                pass
    assert control.current is None


def test_a_take_that_joins_never_joins_one_that_did_not():
    control = Control()
    # The remote operator's lease, or a script: not the window's clicks.
    with control.take(OPERATOR, "remote run"):
        with pytest.raises(MicroscopeBusy):
            with control.take(OPERATOR, "acquire", join=True):
                pass


def test_the_holder_ends_after_what_joined_it():
    control = Control()
    holder = control.take(OPERATOR, "mill", join=True)
    holder.__enter__()
    joined = control.take(OPERATOR, "acquire", join=True)
    joined.__enter__()
    released = threading.Event()

    def release():
        holder.__exit__(None, None, None)
        released.set()

    threading.Thread(target=release, daemon=True).start()
    assert not released.wait(0.2)
    joined.__exit__(None, None, None)
    assert released.wait(5)
    assert control.current is None
