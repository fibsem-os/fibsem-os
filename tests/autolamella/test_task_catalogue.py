"""Tests for the Qt-free half of the Add Task dialog.

These matter more than the usual widget test: CI installs fibsem without the
``[ui]`` extra, so nothing importing PyQt5 runs here at all. This file is the
only coverage the dialog's logic gets.
"""

import pytest

from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.catalogue import (
    TAG_ORDER,
    UNTAGGED,
    available_task_types,
    has_advanced_tasks,
    task_info,
    task_source,
    task_tags,
    tasks_by_tag,
    unique_task_name,
)


class TestAvailableTaskTypes:
    def test_every_offered_key_is_its_own_task_type(self):
        """The invariant the alias filter exists to guarantee.

        A caller builds `get_tasks()[key].config_cls()` and expects back a
        config whose `task_type` is `key`. Anywhere that fails, the picker is
        offering a row that cannot do what its label says.
        """
        for task_type, task_cls in available_task_types().items():
            assert task_cls.config_cls.task_type == task_type

    def test_setup_lamella_alias_is_not_offered(self):
        """SETUP_LAMELLA is a load-time shim, not a task a user can choose.

        It is registered so an old protocol still loads. It points at
        MillFiducialTask, whose config reports MILL_FIDUCIAL regardless -- so
        offering it puts Mill Fiducial in the picker twice and the second row
        silently produces the first row's config.

        If this alias is ever removed from BUILTIN_TASKS this test goes green
        for a different reason, which is fine; the guard below is the one that
        keeps the filter honest in general.
        """
        assert "SETUP_LAMELLA" not in available_task_types()

    def test_is_a_subset_of_the_registry(self):
        assert set(available_task_types()) <= set(get_tasks())

    def test_keeps_the_real_task_behind_the_alias(self):
        """Filtering aliases must not cost us the task they alias."""
        assert "MILL_FIDUCIAL" in available_task_types()

    def test_offers_the_builtin_tasks(self):
        offered = available_task_types()
        for task_type in (
            "MILL_TRENCH",
            "MILL_UNDERCUT",
            "MILL_ROUGH",
            "MILL_POLISHING",
            "ACQUIRE_REFERENCE_IMAGE",
            "ACQUIRE_FLUORESCENCE_IMAGE",
            "SELECT_FLUORESCENCE_POSITION",
            "SELECT_MILLING_POSITION",
            "SPOT_BURN_FIDUCIAL",
            "BASIC_MILLING",
        ):
            assert task_type in offered


class TestUniqueTaskName:
    def test_free_name_is_returned_unchanged(self):
        assert unique_task_name("Trench Milling", {}) == "Trench Milling"

    def test_taken_name_gets_suffix_two(self):
        """The unsuffixed name is the first, so the second is 2, not 1."""
        assert (
            unique_task_name("Trench Milling", {"Trench Milling": 1})
            == "Trench Milling 2"
        )

    def test_skips_suffixes_already_in_use(self):
        existing = {"Trench Milling": 1, "Trench Milling 2": 1, "Trench Milling 3": 1}
        assert unique_task_name("Trench Milling", existing) == "Trench Milling 4"

    def test_gap_in_the_sequence_is_filled(self):
        """Numbering looks for the first free slot, not the highest plus one."""
        existing = {"Trench Milling": 1, "Trench Milling 3": 1}
        assert unique_task_name("Trench Milling", existing) == "Trench Milling 2"

    def test_surrounding_whitespace_is_stripped(self):
        assert unique_task_name("  Polishing  ", {}) == "Polishing"

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_base_yields_blank(self, blank):
        """An empty name is the caller's to reject; do not invent '2' for it."""
        assert unique_task_name(blank, {"x": 1}) == ""

    def test_a_similar_but_different_name_does_not_collide(self):
        assert unique_task_name("Mill", {"Mill Fiducial": 1}) == "Mill"


class TestTaskSource:
    def test_builtin_task_reports_builtin(self):
        from fibsem.plugins.report import ExtensionSource

        source = task_source("MILL_TRENCH")
        assert source is not None
        assert source.source is ExtensionSource.BUILTIN

    def test_unknown_task_type_is_none(self):
        assert task_source("NO_SUCH_TASK_TYPE") is None

    def test_every_offered_task_has_a_source(self):
        """The picker shows provenance for each row, so each row needs one.

        A None here means the registries and fibsem.plugins.report disagree
        about what is installed -- which would also make the Plugins dialog
        wrong, so it is worth failing loudly.
        """
        for task_type in available_task_types():
            assert task_source(task_type) is not None, task_type


class TestTags:
    def test_every_builtin_declares_a_purpose(self):
        """A built-in with no tag would land in Other beside stray plugins."""
        for task_type, task_cls in available_task_types().items():
            if task_source(task_type).source.name != "BUILTIN":
                continue
            assert task_tags(task_cls), task_type

    def test_tags_are_strings(self):
        for task_cls in available_task_types().values():
            for tag in task_tags(task_cls):
                assert isinstance(tag, str) and tag

    def test_a_task_can_declare_several(self):
        """The reason this is a tuple and not one category."""
        spot_burn = available_task_types()["SPOT_BURN_FIDUCIAL"]
        assert task_tags(spot_burn) == ("Correlation", "Alignment")


class TestTasksByTag:
    def test_every_offered_task_appears_exactly_once(self):
        """Filed under its primary only. Listing it per tag would duplicate rows.

        Asked with advanced included, since the default listing deliberately
        withholds those.
        """
        offered = set(available_task_types())
        filed = [
            t for tasks in tasks_by_tag(include_advanced=True).values() for t in tasks
        ]
        assert sorted(filed) == sorted(offered)
        assert len(filed) == len(set(filed))

    def test_known_tags_come_in_declared_order(self):
        groups = [g for g in tasks_by_tag() if g in TAG_ORDER]
        assert groups == [t for t in TAG_ORDER if t in groups]

    def test_the_on_grid_steps_share_one_heading(self):
        on_grid = tasks_by_tag()["On-grid milling"]
        for task_type in (
            "SELECT_MILLING_POSITION",
            "MILL_FIDUCIAL",
            "MILL_ROUGH",
            "MILL_POLISHING",
        ):
            assert task_type in on_grid

    def test_the_waffle_steps_share_one_heading(self):
        waffle = tasks_by_tag()["Waffle milling"]
        assert set(waffle) == {"MILL_TRENCH", "MILL_UNDERCUT"}

    def test_untagged_sorts_last_of_all(self):
        grouped = tasks_by_tag()
        if UNTAGGED not in grouped:
            pytest.skip("every installed task declares a tag")
        assert list(grouped)[-1] == UNTAGGED

    def test_the_two_position_tasks_are_tagged_alike(self):
        """Siblings by name and by behaviour: go somewhere, confirm, record.

        They sit under different headings, so this tag is the only thing that
        says they are the same kind of task. Tagging one and not the other is
        how a vocabulary starts to rot.
        """
        offered = available_task_types()
        for task_type in ("SELECT_MILLING_POSITION", "SELECT_FLUORESCENCE_POSITION"):
            assert "Positioning" in task_tags(offered[task_type])

    def test_no_cross_cutting_tag_has_a_single_holder(self):
        """A tag nothing is filed under earns its place by spanning headings.

        Deliberately not asserted of primary tags: a heading with one task under
        it is thin but legitimate -- "Imaging" and "Custom" are both genuine
        categories that happen to hold one task today. A *secondary* tag with
        one holder is different: it becomes a filter that narrows to a single
        row, which the search box already does better.
        """
        offered = available_task_types()
        primaries = {task_tags(c)[0] for c in offered.values() if task_tags(c)}
        counts = {}
        for task_cls in offered.values():
            for tag in task_tags(task_cls):
                counts[tag] = counts.get(tag, 0) + 1
        singles = sorted(
            tag for tag, n in counts.items() if n < 2 and tag not in primaries
        )
        assert not singles, f"cross-cutting tags carried by one task: {singles}"

    def test_a_cross_cutting_tag_is_never_a_heading(self):
        """Alignment is a purpose four tasks share and none is filed under.

        That is what a tuple of tags buys over one category: the tag is
        filterable without inventing a heading nothing would sit in.
        """
        grouped = tasks_by_tag(include_advanced=True)
        assert "Alignment" not in grouped
        holders = {
            t for t, c in available_task_types().items() if "Alignment" in task_tags(c)
        }
        assert len(holders) > 1
        for task_type in holders:
            primary = task_tags(available_task_types()[task_type])[0]
            assert task_type in grouped[primary]


class TestTaskInfo:
    def test_a_task_declaring_nothing_still_has_defaults(self):
        """A plugin that sets no info must not crash the picker."""

        class Bare:
            class config_cls:
                pass

        info = task_info(Bare)
        assert info.tags == ()
        assert info.description == ""
        assert info.advanced is False

    def test_a_bad_level_is_rejected_at_construction(self):
        """A typo would sort silently to the end and never be hidden."""
        from fibsem.applications.autolamella.structures import TaskInfo

        with pytest.raises(ValueError):
            TaskInfo(level="beginner")

    def test_level_is_one_axis_not_two_flags(self):
        """`recommended` and `advanced` are ends of the same scale.

        Two booleans would admit `beginner and advanced`, which means nothing.
        """
        from fibsem.applications.autolamella.structures import TaskInfo

        assert TaskInfo(level="recommended").recommended
        assert not TaskInfo(level="recommended").advanced
        assert TaskInfo(level="advanced").advanced
        assert not TaskInfo(level="advanced").recommended

    def test_experimental_is_independent_of_level(self):
        """Stability is a different question from how often it is the right answer."""
        from fibsem.applications.autolamella.structures import TaskInfo

        info = TaskInfo(level="recommended", experimental=True)
        assert info.recommended and info.experimental

    def test_misspelled_keyword_is_an_error(self):
        """The whole reason this is a structure and not a ClassVar apiece."""
        from fibsem.applications.autolamella.structures import TaskInfo

        with pytest.raises(TypeError):
            TaskInfo(descripton="typo")  # noqa: F821

    def test_info_is_frozen(self):
        from fibsem.applications.autolamella.structures import TaskInfo

        with pytest.raises(Exception):
            TaskInfo().tags = ("mutated",)


class TestAdvanced:
    def test_advanced_tasks_are_hidden_by_default(self):
        offered = tasks_by_tag()
        for tasks in offered.values():
            for task_cls in tasks.values():
                assert not task_info(task_cls).advanced

    def test_including_them_adds_rather_than_replaces(self):
        without = {t for tasks in tasks_by_tag().values() for t in tasks}
        with_adv = {t for tasks in tasks_by_tag(True).values() for t in tasks}
        assert without < with_adv

    def test_basic_milling_is_advanced(self):
        """It runs whatever stages it is handed -- rarely the right answer."""
        assert task_info(available_task_types()["BASIC_MILLING"]).advanced

    def test_has_advanced_tasks_agrees_with_the_listing(self):
        without = sum(len(v) for v in tasks_by_tag().values())
        with_adv = sum(len(v) for v in tasks_by_tag(True).values())
        assert has_advanced_tasks() == (with_adv > without)


class TestOrderWithinAHeading:
    def test_the_on_grid_sequence_is_in_working_order(self):
        """Alphabetical put Polishing and Rough before the steps that precede them."""
        assert list(tasks_by_tag()["On-grid milling"]) == [
            "SELECT_MILLING_POSITION",
            "MILL_FIDUCIAL",
            "MILL_ROUGH",
            "MILL_POLISHING",
        ]

    def test_the_waffle_sequence_is_in_working_order(self):
        assert list(tasks_by_tag()["Waffle milling"]) == [
            "MILL_TRENCH",
            "MILL_UNDERCUT",
        ]

    def test_advanced_sort_last_within_their_heading(self):
        custom = list(tasks_by_tag(include_advanced=True)["Custom"])
        assert custom[-1] == "BASIC_MILLING"

    def test_unordered_tasks_fall_to_the_end(self):
        """The default order is high, so a task that says nothing does not jump."""
        from fibsem.applications.autolamella.structures import TaskInfo

        # Advanced tasks are excluded: their position comes from the flag, not
        # from order, so Basic Milling sensibly leaves order at the default.
        assert TaskInfo().order > max(
            task_info(c).order
            for c in available_task_types().values()
            if task_info(c).tags and not task_info(c).advanced
        )


class TestDeprecation:
    def test_a_deprecated_task_is_not_offered(self):
        """It still loads, so existing protocols keep working -- it is just not
        something to add a new one of."""
        from fibsem.applications.autolamella.structures import TaskInfo

        config_cls = get_tasks()["MILL_TRENCH"].config_cls
        original = config_cls.info
        try:
            config_cls.info = TaskInfo(
                tags=("On-grid milling",), deprecated="Superseded by Rough Milling."
            )
            assert "MILL_TRENCH" not in available_task_types()
            assert "MILL_TRENCH" in get_tasks(), "it must still load"
        finally:
            config_cls.info = original
        assert "MILL_TRENCH" in available_task_types(), "fixture leaked"

    def test_nothing_shipped_is_deprecated(self):
        for task_cls in available_task_types().values():
            assert not task_info(task_cls).is_deprecated

    def test_deprecation_carries_a_reason(self):
        """A bare flag strands whoever already has it in a protocol."""
        from fibsem.applications.autolamella.structures import TaskInfo

        assert TaskInfo(deprecated="Use X.").deprecated == "Use X."
        assert not TaskInfo().is_deprecated


class TestDescriptions:
    def test_every_builtin_describes_itself(self):
        """The description slot is the picker's whole reason for a detail pane."""
        for task_type, task_cls in available_task_types().items():
            if task_source(task_type).source.name != "BUILTIN":
                continue
            assert task_info(task_cls).description, task_type

    def test_descriptions_are_sentences_not_labels(self):
        for task_cls in available_task_types().values():
            description = task_info(task_cls).description
            if not description:
                continue
            assert description[0].isupper(), description
            assert description.rstrip().endswith("."), description
            assert len(description) > 40, description

    def test_descriptions_fit_a_table_cell(self):
        """The whole description is rendered, not a lead sentence.

        The table used to show only the text up to the first ". ", which
        silently dropped clauses like "Aligns before milling." Now that the cell
        shows all of it, length is what keeps rows comparable.
        """
        for task_cls in available_task_types().values():
            description = task_info(task_cls).description
            if not description:
                continue
            assert len(description) <= 160, description


class TestHardwareField:
    def test_is_not_called_requires(self):
        """`AutoLamellaTaskDescription.requires` already means prerequisite tasks.

        Two kinds of prerequisite under one name in one domain is the confusion
        this whole structure exists to stop.
        """
        from fibsem.applications.autolamella.structures import TaskInfo

        assert not hasattr(TaskInfo(), "requires")
        assert TaskInfo().requires_hardware == ()

    def test_nothing_declares_it_yet(self):
        """Groundwork only -- deliberately unread until the checks land."""
        for task_cls in available_task_types().values():
            assert task_info(task_cls).requires_hardware == ()
