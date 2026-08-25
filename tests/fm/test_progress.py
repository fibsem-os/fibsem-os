"""The typed contract carried by ``acquisition_progress_signal`` (FIB-401).

Nothing emits or reads a ``FluorescenceAcquisitionProgress`` yet -- the producers still
build dicts and the consumers still read them. What is worth pinning before either moves
is the shape, and specifically the two decisions that are easy to quietly undo later:
that the three acquiring states are distinct, and that a z count can arrive under two of
them.
"""

import dataclasses

import pytest

from fibsem.fm.progress import (
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)


class TestTheStatusVocabulary:
    def test_the_three_acquiring_states_are_distinct(self):
        """They differ in what is being counted, which is the only thing consumers use
        them for: channels, z planes, or focus steps."""
        acquiring = {
            FluorescenceAcquisitionStatus.ACQUIRING_CHANNELS,
            FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
            FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
        }
        assert len(acquiring) == 3
        assert FluorescenceAcquisitionStatus.FINISHED not in acquiring

    def test_a_focus_sweep_is_not_a_z_stack(self):
        """A sweep steps the objective through a search range; a stack acquires planes.
        Labelling sweep positions "Z-level" names something that is not running, which
        is the distinction this pair exists to keep."""
        assert (
            FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS
            is not FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK
        )

    def test_autofocus_has_exactly_one_spelling(self):
        """It used to have two live at once -- a `state: "autofocus"` at sweep start and
        an `operation: "autofocus"` per step -- and briefly a third, `"autofocusing"`,
        that no consumer decoded. A single closed enum makes that unrepresentable."""
        autofocus = [s for s in FluorescenceAcquisitionStatus if "autofocus" in s.value]
        assert autofocus == [FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS]

    def test_a_member_compares_equal_to_its_own_value(self):
        assert FluorescenceAcquisitionStatus.FINISHED == "finished"


class TestTheReportShape:
    def test_status_is_the_only_thing_every_report_carries(self):
        """Everything else is absent on some report, so everything else has a default.

        Also what keeps this constructible on Python 3.8: `kw_only` does not exist
        there, so exactly one required field, and it has to be first.
        """
        report = FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.FINISHED
        )

        assert report.status is FluorescenceAcquisitionStatus.FINISHED
        assert report.channel is None
        assert report.zlevel is None and report.total_zlevels is None

        required = [
            f.name
            for f in dataclasses.fields(FluorescenceAcquisitionProgress)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        assert required == ["status"]

    def test_a_report_cannot_be_written_to(self):
        report = FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.FINISHED
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            report.status = FluorescenceAcquisitionStatus.ACQUIRING_CHANNELS

    def test_two_equal_reports_compare_equal(self):
        """Equality is left generated here, unlike `TiledProgress`, which had to set
        `eq=False` because a preview image compares its numpy data elementwise and
        raises. Nothing on this signal carries an array, so `==` is usable -- which is
        what lets a producer test assert on a whole list of emitted reports.
        """

        def report():
            return FluorescenceAcquisitionProgress(
                status=FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
                channel="DAPI",
                zlevel=3,
                total_zlevels=7,
            )

        assert report() == report()
        assert len({report(), report()}) == 1
        assert report() != FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.FINISHED
        )


class TestWhatTheConsumersActuallyBranchOn:
    """The reason this is one flat record rather than a type per acquisition routine.

    Both consumers ask *is there a z count?* before anything else, and the answer is yes
    for two different routines. A type per routine would force
    `isinstance(e, (ZStackProgress, AutofocusProgress))` at that branch -- a multi-class
    tuple whose failure mode is silent, because an omitted class is a bar that stops
    moving rather than an error.
    """

    def test_a_z_count_arrives_under_two_different_statuses(self):
        stack = FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
            channel="DAPI",
            zlevel=3,
            total_zlevels=7,
        )
        sweep = FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS,
            channel="DAPI",
            zlevel=3,
            total_zlevels=7,
            pass_index=2,
            total_passes=2,
        )

        assert stack.status is not sweep.status
        for report in (stack, sweep):
            assert report.zlevel and report.total_zlevels, (
                "both routines count z; a type per routine would split this branch"
            )

    def test_a_channel_acquisition_carries_no_z_count(self):
        """Which is what sends it down the other branch, to count channels instead."""
        report = FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.ACQUIRING_CHANNELS,
            channel="GFP",
            channel_index=2,
            total_channels=3,
        )

        assert not (report.zlevel and report.total_zlevels)
        assert (report.channel_index, report.total_channels) == (2, 3)

    def test_counts_are_one_based_on_the_wire(self):
        """Counts of work done, not indices into anything -- nothing subscripts by them.
        So the number a reader sees is the number the producer sends, and unlike the
        tile indices on `TiledProgress` there is no offset to apply anywhere.
        """
        first = FluorescenceAcquisitionProgress(
            status=FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
            zlevel=1,
            total_zlevels=7,
        )

        assert first.zlevel == 1, "the first plane reads as 1 of 7, never 0 of 7"
        assert f"{first.zlevel}/{first.total_zlevels}" == "1/7"
