"""The typed contract carried by ``milling_progress_signal`` (FIB-797).

What is pinned here is the shape, and specifically the decisions that are cheap to
quietly undo later: that a stage outcome and a task outcome are different things, that
the offset lives in one place, and that a plugin's empty string is not a request for a
blank label.

``milling_progress_signal`` had **no** test coverage of any kind before FIB-797.

`from_payload` is covered here too. Every in-tree producer now emits a typed report,
so it decodes nothing of ours -- it is kept as the totality guard at a queued-Qt-slot
boundary, because a plugin-loaded strategy is a producer this repo never sees.
"""

import dataclasses

import pytest

from fibsem.milling.progress import (
    _DEFAULT_MESSAGES,
    MillingMessageTracker,
    MillingProgress,
    MillingProgressStatus,
)
from fibsem.structures import ACTIVE_MILLING_STATES, MillingState


class TestTheStatusVocabulary:
    def test_a_stage_finishing_is_not_the_task_finishing(self):
        """The vocabulary this replaces had `state: "start"` (emitted per stage, N
        times) sitting beside `state: "finished"` (emitted once, from the task's
        `finally`). They read like a matched pair and were not one, so a consumer
        bracketing them was wrong N-1 times."""
        assert (
            MillingProgressStatus.STAGE_FINISHED
            is not MillingProgressStatus.TASK_FINISHED
        )
        assert (
            MillingProgressStatus.STAGE_STARTED
            is not MillingProgressStatus.TASK_STARTED
        )

    def test_only_the_task_outcomes_are_terminal(self):
        """`STAGE_FINISHED` is not the end of anything. A consumer that hides its
        progress bar on it hides it after the first stage of an N-stage task."""
        terminal = {s for s in MillingProgressStatus if s.is_terminal}
        assert terminal == {
            MillingProgressStatus.TASK_FINISHED,
            MillingProgressStatus.TASK_CANCELLED,
            MillingProgressStatus.TASK_FAILED,
        }
        assert not MillingProgressStatus.STAGE_FINISHED.is_terminal

    def test_a_cancelled_task_is_not_a_failed_one(self):
        """A cancel is someone getting what they asked for. Today both land in the same
        `finally` and report as *finished*; keeping them apart in the vocabulary is what
        lets a consumer paint one neutrally and the other red."""
        assert (
            MillingProgressStatus.TASK_CANCELLED
            is not MillingProgressStatus.TASK_FAILED
        )
        assert (
            MillingProgressStatus.TASK_CANCELLED
            is not MillingProgressStatus.TASK_FINISHED
        )

    def test_a_member_compares_equal_to_its_own_value(self):
        """The `str` mixin, so these read as themselves in a log line."""
        assert MillingProgressStatus.STAGE_UPDATE == "stage-update"

    def test_every_status_has_a_default_message(self):
        """`display_message` indexes `_DEFAULT_MESSAGES` directly. A status added
        without one is a `KeyError` raised inside a queued Qt slot, which on PyQt5 is
        `qFatal` -- the process aborts with nothing in the logfile (FIB-329)."""
        assert set(_DEFAULT_MESSAGES) == set(MillingProgressStatus)


class TestDisplayStage:
    def test_the_offset_is_applied_once_here(self):
        """`current_stage` is 0-based on the wire, like every other index in the
        codebase. The `+ 1` was written out by hand at seven call sites across three
        widgets before this property existed."""
        first = MillingProgress(MillingProgressStatus.STAGE_STARTED, current_stage=0)
        fourth = MillingProgress(MillingProgressStatus.STAGE_STARTED, current_stage=3)
        assert first.display_stage == 1
        assert fourth.display_stage == 4

    def test_a_report_with_no_stage_index_has_no_display_stage(self):
        """Rather than defaulting to 0 and rendering "Stage 1" for a task-level report
        that is about no stage at all."""
        assert (
            MillingProgress(MillingProgressStatus.TASK_FINISHED).display_stage is None
        )


class TestDisplayMessage:
    def test_a_producer_that_supplies_words_gets_them_back(self):
        """The point of keeping `message` at all: milling strategies are plugin-loadable
        and users asked to customise the text a strategy shows while it runs."""
        words = "Polishing at 30 pA"
        report = MillingProgress(MillingProgressStatus.STAGE_UPDATE, message=words)
        assert report.display_message == words

    def test_an_empty_message_falls_back_to_the_default(self):
        """Falsy, not `is not None`. A plugin returning `""` is a bug, not a request for
        a blank label -- FIB-401 hit this exactly, where an empty channel rendered
        "Acquiring  (1/1)..."."""
        report = MillingProgress(MillingProgressStatus.STAGE_UPDATE, message="")
        assert (
            report.display_message
            == _DEFAULT_MESSAGES[MillingProgressStatus.STAGE_UPDATE]
        )

    def test_the_default_says_which_stage_when_the_report_names_one(self):
        """One home for the wording. The three consumers defaulted three different ways
        for the same report -- "Preparing Milling Conditions...", "Preparing...",
        "Milling..." -- which is the drift a single table removes."""
        report = MillingProgress(
            MillingProgressStatus.STAGE_STARTED, stage_name="Rough Mill"
        )
        assert report.display_message == "Preparing: Rough Mill"

    def test_the_default_says_which_task_when_the_report_names_one(self):
        report = MillingProgress(
            MillingProgressStatus.TASK_FINISHED, task_name="Trench"
        )
        assert report.display_message == "Finished milling task: Trench"

    def test_an_outcome_is_not_qualified_by_a_stage_name(self):
        """A label reading "Milling cancelled: Rough Mill" invites the reading that one
        *stage* was cancelled. The outcomes are task-level; that is the whole reason the
        vocabulary has two scales."""
        report = MillingProgress(
            MillingProgressStatus.TASK_CANCELLED, stage_name="Rough Mill"
        )
        assert report.display_message == "Milling cancelled"

    def test_a_report_naming_nothing_still_renders(self):
        """Every status, with no fields at all. A label reading "Preparing: None" is
        worse than one reading "Preparing..."."""
        for status in MillingProgressStatus:
            message = MillingProgress(status).display_message
            assert message
            assert "None" not in message

    def test_a_qualified_label_does_not_keep_its_ellipsis(self):
        """It reads as a pause rather than as "and then the name" once something
        follows it."""
        report = MillingProgress(
            MillingProgressStatus.STAGE_UPDATE, stage_name="Polish"
        )
        assert report.display_message == "Milling: Polish"


class TestTheRecord:
    def test_status_is_the_only_required_field(self):
        """Which is also what keeps this constructible on the 3.8 CI jobs, where
        `dataclass(kw_only=...)` does not exist and required fields must come first."""
        report = MillingProgress(MillingProgressStatus.TASK_STARTED)
        assert report.status is MillingProgressStatus.TASK_STARTED
        assert report.message is None
        assert report.error is None

    def test_a_report_is_frozen(self):
        report = MillingProgress(MillingProgressStatus.STAGE_UPDATE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.remaining_time = 1.0  # type: ignore[misc]

    def test_two_reports_carrying_the_same_thing_are_equal(self):
        """Equality is left generated, unlike `TiledProgress` -- nothing here holds a
        numpy array whose `__eq__` returns an array. That is what makes these usable in
        `assert emitted == [...]`, which every consumer test in this stack relies on."""
        a = MillingProgress(MillingProgressStatus.STAGE_UPDATE, remaining_time=12.0)
        b = MillingProgress(MillingProgressStatus.STAGE_UPDATE, remaining_time=12.0)
        assert a == b
        assert len({a, b}) == 1

    def test_a_report_can_carry_the_instrument_state(self):
        """Carried on the report so a consumer does not have to ask -- and asking is not
        free: on ThermoFisher `get_milling_state()` sets the active view as a side
        effect."""
        report = MillingProgress(
            MillingProgressStatus.STAGE_UPDATE, milling_state=MillingState.RUNNING
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


class TestTheTotalityGuard:
    """`from_payload` decodes whatever actually arrived, including the older nested dict.

    Not a migration shim -- every in-tree producer emits a `MillingProgress`, so this is
    a no-op for all of them. It stands because the producer set is **open**: milling
    strategies are plugin-loadable and drive their own execution loops, so one built
    against the older contract still emits a dict. `psygnal` does not validate
    emissions, so that dict reaches the slot unchanged and the first attribute access on
    it raises -- inside a queued Qt slot, which on PyQt5 is `qFatal` (FIB-329).
    """

    def test_a_stage_start_decodes(self):
        report = MillingProgress.from_payload(
            {
                "msg": "Preparing: Rough Mill",
                "progress": {
                    "state": "start",
                    "current_stage": 0,
                    "total_stages": 3,
                    "task_id": "abc",
                    "task_name": "Trench",
                    "stage_name": "Rough Mill",
                    "start_time": 100.0,
                },
            }
        )
        assert report.status is MillingProgressStatus.STAGE_STARTED
        assert report.display_stage == 1
        assert report.total_stages == 3
        assert report.stage_name == "Rough Mill"
        assert report.message == "Preparing: Rough Mill"

    def test_an_update_decodes(self):
        report = MillingProgress.from_payload(
            {
                "progress": {
                    "state": "update",
                    "start_time": 100.0,
                    "milling_state": MillingState.RUNNING,
                    "estimated_time": 60.0,
                    "remaining_time": 12.0,
                }
            }
        )
        assert report.status is MillingProgressStatus.STAGE_UPDATE
        assert report.remaining_time == 12.0
        assert report.milling_state is MillingState.RUNNING

    def test_the_task_terminal_decodes(self):
        report = MillingProgress.from_payload(
            {
                "msg": "Finished Milling Task: Trench...",
                "progress": {
                    "state": "finished",
                    "task_id": "abc",
                    "task_name": "Trench",
                },
            }
        )
        assert report.status is MillingProgressStatus.TASK_FINISHED
        assert report.status.is_terminal

    def test_a_delegating_strategys_shape_decodes_to_something(self):
        """The shape with no `state` at all: a strategy that emits its label, then hands
        the loop to a backend via `microscope.run_milling()`.

        Under the old vocabulary this matched no branch in any of the three consumers,
        so the strategy's own words -- the customisable text users asked for -- rendered
        nowhere. Decoding it as an update is what finally puts them on the screen. Such
        a strategy can be plugin-loaded, so this shape outlives the in-tree producers.
        """
        report = MillingProgress.from_payload(
            {
                "msg": "Running Rough Mill...",
                "progress": {
                    "started": True,
                    "start_time": 100.0,
                    "estimated_time": 60.0,
                    "name": "Rough Mill",
                },
            }
        )
        assert report.status is MillingProgressStatus.STAGE_UPDATE
        assert report.display_message == "Running Rough Mill..."
        # `name` is the older spelling of `stage_name`.
        assert report.stage_name == "Rough Mill"

    def test_a_string_milling_state_becomes_the_enum(self):
        """Some producers send `"UNKNOWN"` where others send a `MillingState`, and that
        is deliberate: calling the getter would steal the active view from a
        fluorescence acquisition the strategy may be running."""
        report = MillingProgress.from_payload(
            {"progress": {"state": "update", "milling_state": "UNKNOWN"}}
        )
        assert report.milling_state is MillingState.UNKNOWN

    def test_a_typed_report_passes_straight_through(self):
        """The in-tree path, which is every producer in this repo. Identity, not a
        rebuild -- the guard must cost nothing on the path that is always taken."""
        original = MillingProgress(MillingProgressStatus.TASK_CANCELLED, error=None)
        assert MillingProgress.from_payload(original) is original

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "not a dict",
            42,
            {},
            {"progress": None},
            {"progress": "not a dict"},
            {"msg": 42, "progress": {"state": "start"}},
            {"progress": {"state": "start", "current_stage": "one"}},
            {"progress": {"state": "update", "remaining_time": object()}},
            {"progress": {"state": "update", "milling_state": "NOT_A_STATE"}},
            {"progress": {"state": "update", "milling_state": 3}},
        ],
    )
    def test_nothing_makes_it_raise(self, payload):
        """Total by construction, which is the entire reason it is kept. This runs
        inside a queued Qt slot, and on PyQt5 an exception escaping one of those is
        `qFatal`: the process aborts with nothing written to the logfile (FIB-329). A
        bar that does not move is recoverable; a dead application is not."""
        report = MillingProgress.from_payload(payload)
        assert isinstance(report, MillingProgress)
        assert report.display_message

    def test_a_boolean_is_not_a_count(self):
        """`bool` is an `int` subclass, so an unguarded `isinstance` check turns the
        older `"started": True` into stage 1."""
        report = MillingProgress.from_payload(
            {"progress": {"state": "start", "current_stage": True}}
        )
        assert report.current_stage is None


class TestTheStickyMessage:
    """The rule that lets a *delegating* strategy -- one that calls
    `microscope.run_milling()` and hands the loop to a backend -- set its label once and
    have it survive every messageless tick that follows."""

    def test_a_message_survives_the_ticks_that_follow_it(self):
        tracker = MillingMessageTracker()
        tracker.label(
            MillingProgress(MillingProgressStatus.STAGE_UPDATE, message="Polishing")
        )
        plain = MillingProgress(MillingProgressStatus.STAGE_UPDATE, remaining_time=4.0)
        assert tracker.label(plain) == "Polishing"

    def test_a_new_stage_drops_the_previous_stages_words(self):
        """Only `STAGE_UPDATE` inherits. Every other status is its own moment: a stage
        that has just started is not still "Polishing" the one before it."""
        tracker = MillingMessageTracker()
        tracker.label(
            MillingProgress(MillingProgressStatus.STAGE_UPDATE, message="Polishing")
        )
        started = MillingProgress(
            MillingProgressStatus.STAGE_STARTED, stage_name="Rough Mill"
        )
        assert tracker.label(started) == "Preparing: Rough Mill"

    def test_a_finished_task_is_not_captioned_with_what_it_was_doing(self):
        tracker = MillingMessageTracker()
        tracker.label(
            MillingProgress(MillingProgressStatus.STAGE_UPDATE, message="Polishing")
        )
        assert (
            tracker.label(MillingProgress(MillingProgressStatus.TASK_FINISHED))
            == "Finished milling task"
        )

    def test_an_empty_message_does_not_blank_the_label(self):
        """Falsy, not `is not None`: a plugin returning `""` is a bug, not a request for
        a blank label."""
        tracker = MillingMessageTracker()
        tracker.label(
            MillingProgress(MillingProgressStatus.STAGE_UPDATE, message="Polishing")
        )
        blank = MillingProgress(MillingProgressStatus.STAGE_UPDATE, message="")
        assert tracker.label(blank) == "Polishing"

    def test_a_tracker_that_has_seen_nothing_still_renders(self):
        tracker = MillingMessageTracker()
        assert (
            tracker.label(MillingProgress(MillingProgressStatus.STAGE_UPDATE))
            == "Milling..."
        )
