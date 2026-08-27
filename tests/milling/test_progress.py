"""The typed contract carried by ``milling_progress_signal`` (FIB-797).

Nothing emits or reads a ``MillingProgress`` yet -- the producers still build nested
dicts and the consumers still take them apart. What is worth pinning before either moves
is the shape, and specifically the decisions that are cheap to quietly undo later: that
a stage outcome and a task outcome are different things, that the offset lives in one
place, and that a plugin's empty string is not a request for a blank label.

``milling_progress_signal`` has had **no** test coverage of any kind until now.
"""

import dataclasses

import pytest

from fibsem.milling.progress import (
    _DEFAULT_MESSAGES,
    MillingProgress,
    MillingStatus,
)
from fibsem.structures import ACTIVE_MILLING_STATES, MillingState


class TestTheStatusVocabulary:
    def test_a_stage_finishing_is_not_the_task_finishing(self):
        """The vocabulary this replaces had `state: "start"` (emitted per stage, N
        times) sitting beside `state: "finished"` (emitted once, from the task's
        `finally`). They read like a matched pair and were not one, so a consumer
        bracketing them was wrong N-1 times."""
        assert MillingStatus.STAGE_FINISHED is not MillingStatus.TASK_FINISHED
        assert MillingStatus.STAGE_STARTED is not MillingStatus.TASK_STARTED

    def test_only_the_task_outcomes_are_terminal(self):
        """`STAGE_FINISHED` is not the end of anything. A consumer that hides its
        progress bar on it hides it after the first stage of an N-stage task."""
        terminal = {s for s in MillingStatus if s.is_terminal}
        assert terminal == {
            MillingStatus.TASK_FINISHED,
            MillingStatus.TASK_CANCELLED,
            MillingStatus.TASK_FAILED,
        }
        assert not MillingStatus.STAGE_FINISHED.is_terminal

    def test_a_cancelled_task_is_not_a_failed_one(self):
        """A cancel is someone getting what they asked for. Today both land in the same
        `finally` and report as *finished*; keeping them apart in the vocabulary is what
        lets a consumer paint one neutrally and the other red."""
        assert MillingStatus.TASK_CANCELLED is not MillingStatus.TASK_FAILED
        assert MillingStatus.TASK_CANCELLED is not MillingStatus.TASK_FINISHED

    def test_a_member_compares_equal_to_its_own_value(self):
        """The `str` mixin, so these read as themselves in a log line."""
        assert MillingStatus.STAGE_UPDATE == "stage-update"

    def test_every_status_has_a_default_message(self):
        """`display_message` indexes `_DEFAULT_MESSAGES` directly. A status added
        without one is a `KeyError` raised inside a queued Qt slot, which on PyQt5 is
        `qFatal` -- the process aborts with nothing in the logfile (FIB-329)."""
        assert set(_DEFAULT_MESSAGES) == set(MillingStatus)


class TestDisplayStage:
    def test_the_offset_is_applied_once_here(self):
        """`current_stage` is 0-based on the wire, like every other index in the
        codebase. The `+ 1` was written out by hand at seven call sites across three
        widgets before this property existed."""
        first = MillingProgress(MillingStatus.STAGE_STARTED, current_stage=0)
        fourth = MillingProgress(MillingStatus.STAGE_STARTED, current_stage=3)
        assert first.display_stage == 1
        assert fourth.display_stage == 4

    def test_a_report_with_no_stage_index_has_no_display_stage(self):
        """Rather than defaulting to 0 and rendering "Stage 1" for a task-level report
        that is about no stage at all."""
        assert MillingProgress(MillingStatus.TASK_FINISHED).display_stage is None


class TestDisplayMessage:
    def test_a_producer_that_supplies_words_gets_them_back(self):
        """The point of keeping `message` at all: milling strategies are plugin-loadable
        and users asked to customise the text a strategy shows while it runs."""
        words = "Polishing at 30 pA"
        report = MillingProgress(MillingStatus.STAGE_UPDATE, message=words)
        assert report.display_message == words

    def test_an_empty_message_falls_back_to_the_default(self):
        """Falsy, not `is not None`. A plugin returning `""` is a bug, not a request for
        a blank label -- FIB-401 hit this exactly, where an empty channel rendered
        "Acquiring  (1/1)..."."""
        report = MillingProgress(MillingStatus.STAGE_UPDATE, message="")
        assert report.display_message == _DEFAULT_MESSAGES[MillingStatus.STAGE_UPDATE]

    def test_the_default_says_which_stage_when_the_report_names_one(self):
        """One home for the wording. The three consumers defaulted three different ways
        for the same report -- "Preparing Milling Conditions...", "Preparing...",
        "Milling..." -- which is the drift a single table removes."""
        report = MillingProgress(MillingStatus.STAGE_STARTED, stage_name="Rough Mill")
        assert report.display_message == "Preparing: Rough Mill"

    def test_the_default_says_which_task_when_the_report_names_one(self):
        report = MillingProgress(MillingStatus.TASK_FINISHED, task_name="Trench")
        assert report.display_message == "Finished milling task: Trench"

    def test_an_outcome_is_not_qualified_by_a_stage_name(self):
        """A label reading "Milling cancelled: Rough Mill" invites the reading that one
        *stage* was cancelled. The outcomes are task-level; that is the whole reason the
        vocabulary has two scales."""
        report = MillingProgress(MillingStatus.TASK_CANCELLED, stage_name="Rough Mill")
        assert report.display_message == "Milling cancelled"

    def test_a_report_naming_nothing_still_renders(self):
        """Every status, with no fields at all. A label reading "Preparing: None" is
        worse than one reading "Preparing..."."""
        for status in MillingStatus:
            message = MillingProgress(status).display_message
            assert message
            assert "None" not in message

    def test_a_qualified_label_does_not_keep_its_ellipsis(self):
        """It reads as a pause rather than as "and then the name" once something
        follows it."""
        report = MillingProgress(MillingStatus.STAGE_UPDATE, stage_name="Polish")
        assert report.display_message == "Milling: Polish"


class TestTheRecord:
    def test_status_is_the_only_required_field(self):
        """Which is also what keeps this constructible on the 3.8 CI jobs, where
        `dataclass(kw_only=...)` does not exist and required fields must come first."""
        report = MillingProgress(MillingStatus.TASK_STARTED)
        assert report.status is MillingStatus.TASK_STARTED
        assert report.message is None
        assert report.error is None

    def test_a_report_is_frozen(self):
        report = MillingProgress(MillingStatus.STAGE_UPDATE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.remaining_time = 1.0  # type: ignore[misc]

    def test_two_reports_carrying_the_same_thing_are_equal(self):
        """Equality is left generated, unlike `TiledProgress` -- nothing here holds a
        numpy array whose `__eq__` returns an array. That is what makes these usable in
        `assert emitted == [...]`, which every consumer test in this stack relies on."""
        a = MillingProgress(MillingStatus.STAGE_UPDATE, remaining_time=12.0)
        b = MillingProgress(MillingStatus.STAGE_UPDATE, remaining_time=12.0)
        assert a == b
        assert len({a, b}) == 1

    def test_a_report_can_carry_the_instrument_state(self):
        """Carried on the report so a consumer does not have to ask -- and asking is not
        free: on ThermoFisher `get_milling_state()` sets the active view as a side
        effect."""
        report = MillingProgress(
            MillingStatus.STAGE_UPDATE, milling_state=MillingState.RUNNING
        )
        assert report.milling_state is MillingState.RUNNING


class TestMillingStateUnknown:
    def test_unknown_is_a_real_member(self):
        """Rather than the magic string `"UNKNOWN"` the coincidence strategy sends
        today, which every other producer sends as an enum."""
        assert MillingState.UNKNOWN is not MillingState.ERROR
        assert MillingState.UNKNOWN is not MillingState.IDLE

    def test_adding_unknown_did_not_renumber_the_existing_states(self):
        """Purely additive. These values reach the server client over the wire."""
        assert MillingState.IDLE.value == 0
        assert MillingState.RUNNING.value == 1
        assert MillingState.STOPPING.value == 2
        assert MillingState.PAUSED.value == 3
        assert MillingState.ERROR.value == 4

    def test_unknown_does_not_keep_a_milling_wait_spinning(self):
        """`ACTIVE_MILLING_STATES` is a loop condition. Both classifications have a
        failure mode -- excluded, a running mill that reported `UNKNOWN` exits the loop
        early; included, a stopped mill spins forever. Exiting early is bounded and
        recoverable; spinning is not."""
        assert MillingState.UNKNOWN not in ACTIVE_MILLING_STATES
