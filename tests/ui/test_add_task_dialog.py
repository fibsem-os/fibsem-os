"""Headless tests for the Add Task table.

CI installs fibsem without the [ui] extra, so this file does not run there --
the Qt-free half is covered by tests/autolamella/test_task_catalogue.py. What is
left here is the wiring only reachable through the widget.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_add_task_dialog.py
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.ui.add_task_dialog import AddTaskDialog
from fibsem.applications.autolamella.workflows.tasks.catalogue import (
    available_task_types,
    task_info,
    task_source,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(app):
    """A dialog over a protocol that already holds three tasks.

    `show()` matters twice over: isVisible() is False for every child of a top
    level that was never shown, and the row heights are only measured once the
    table has real column widths.
    """
    existing = {"Trench Milling": 1, "Rough Milling": 1, "Rough Milling 2": 1}
    d = AddTaskDialog(existing, lamella_count=4)
    d.show()
    app.processEvents()
    yield d
    d.close()
    d.deleteLater()


def _visible_tasks(dialog):
    return [
        t
        for row, t in sorted(dialog._rows.items())
        if not dialog.table.isRowHidden(row)
    ]


def _visible_headings(dialog):
    return [
        dialog.table.item(row, 0).text()
        for row in range(dialog.table.rowCount())
        if row not in dialog._rows and not dialog.table.isRowHidden(row)
    ]


class TestTable:
    def test_lists_every_offered_task_once(self, dialog):
        dialog.advanced_checkbox.setChecked(True)
        tasks = _visible_tasks(dialog)
        assert sorted(tasks) == sorted(available_task_types())
        assert len(tasks) == len(set(tasks))

    def test_mill_fiducial_appears_exactly_once(self, dialog):
        """The SETUP_LAMELLA alias used to put this row in the table twice."""
        assert _visible_tasks(dialog).count("MILL_FIDUCIAL") == 1

    def test_groups_under_purpose_headings(self, dialog):
        assert "ON-GRID MILLING" in _visible_headings(dialog)

    def test_each_sequence_is_listed_in_working_order(self, dialog):
        tasks = _visible_tasks(dialog)
        for sequence in (
            [
                "SELECT_MILLING_POSITION",
                "MILL_FIDUCIAL",
                "MILL_ROUGH",
                "MILL_POLISHING",
            ],
            ["MILL_TRENCH", "MILL_UNDERCUT"],
        ):
            assert [t for t in tasks if t in sequence] == sequence

    def test_opens_on_a_task_not_a_heading(self, dialog):
        assert dialog.selected_task_type is not None

    def test_no_description_is_clipped(self, dialog):
        """Row heights are measured, not computed; this is what that protects.

        Every arithmetic version of the sizing was short by a line, and a
        clipped description gives no sign that anything is missing.
        """
        for row in dialog._rows:
            if dialog.table.isRowHidden(row):
                continue
            label = dialog.table.cellWidget(row, 1).property("_label")
            assert label.height() >= label.heightForWidth(label.width())

    def test_rows_resize_when_the_dialog_narrows(self, dialog, app):
        """The description column stretches, so a resize re-wraps every row."""
        dialog.resize(1200, 620)
        app.processEvents()
        wide = [dialog.table.rowHeight(r) for r in sorted(dialog._rows)]
        dialog.resize(840, 620)
        app.processEvents()
        narrow = [dialog.table.rowHeight(r) for r in sorted(dialog._rows)]
        assert sum(narrow) > sum(wide), "narrowing must make some row taller"


class TestFiltering:
    def test_search_matches_display_name(self, dialog):
        dialog.search_field.setText("undercut")
        assert _visible_tasks(dialog) == ["MILL_UNDERCUT"]

    def test_search_matches_what_a_task_does(self, dialog):
        """Spot Burn does not say "fluorescence" in its name, only in its summary.

        Matching the description is the point: someone searching for the job
        rather than the label should still find it.
        """
        dialog.search_field.setText("fluor")
        found = _visible_tasks(dialog)
        assert "SPOT_BURN_FIDUCIAL" in found
        assert "ACQUIRE_FLUORESCENCE_IMAGE" in found
        assert "MILL_TRENCH" not in found

    def test_search_matches_the_identifier(self, dialog):
        """Someone holding a traceback has MILL_TRENCH, not 'Trench Milling'."""
        dialog.search_field.setText("MILL_TRENCH")
        assert _visible_tasks(dialog) == ["MILL_TRENCH"]

    def test_tag_filter_lists_every_declared_tag(self, dialog):
        items = [
            dialog.tag_filter.itemText(i) for i in range(dialog.tag_filter.count())
        ]
        assert items[0] == "All purposes"
        assert {
            "On-grid milling",
            "Waffle milling",
            "Correlation",
            "Imaging",
            "Alignment",
            "Fluorescence",
        } <= set(items)

    def test_tag_filter_matches_secondary_tags_too(self, dialog):
        """Spot Burn files under Correlation but is also an Alignment aid."""
        dialog.tag_filter.setCurrentText("Alignment")
        assert "SPOT_BURN_FIDUCIAL" in _visible_tasks(dialog)
        assert "MILL_TRENCH" not in _visible_tasks(dialog)

    def test_a_filtered_task_keeps_its_own_heading(self, dialog):
        """It files under its primary even when matched on a secondary."""
        dialog.tag_filter.setCurrentText("Alignment")
        assert "CORRELATION" in _visible_headings(dialog)

    def test_headings_with_nothing_under_them_are_hidden(self, dialog):
        dialog.tag_filter.setCurrentText("Fluorescence")
        assert "ON-GRID MILLING" not in _visible_headings(dialog)

    def test_clearing_the_filter_restores_everything(self, dialog):
        before = _visible_tasks(dialog)
        dialog.tag_filter.setCurrentText("Alignment")
        dialog.tag_filter.setCurrentText("All purposes")
        assert _visible_tasks(dialog) == before

    def test_search_and_tag_filter_compose(self, dialog):
        dialog.tag_filter.setCurrentText("Correlation")
        dialog.search_field.setText("spot")
        assert _visible_tasks(dialog) == ["SPOT_BURN_FIDUCIAL"]


class TestAdvanced:
    def test_hidden_until_asked_for(self, dialog):
        assert "BASIC_MILLING" not in _visible_tasks(dialog)
        dialog.advanced_checkbox.setChecked(True)
        assert "BASIC_MILLING" in _visible_tasks(dialog)

    def test_sorts_last_under_its_heading(self, dialog):
        dialog.advanced_checkbox.setChecked(True)
        tasks = _visible_tasks(dialog)
        assert tasks.index("BASIC_MILLING") > tasks.index("MILL_POLISHING")

    def test_toggling_keeps_the_selection(self, dialog):
        dialog.select("MILL_ROUGH")
        dialog.advanced_checkbox.setChecked(True)
        assert dialog.selected_task_type == "MILL_ROUGH"

    def test_toggling_off_recovers_a_lost_selection(self, dialog):
        dialog.advanced_checkbox.setChecked(True)
        dialog.select("BASIC_MILLING")
        dialog.advanced_checkbox.setChecked(False)
        assert dialog.selected_task_type not in (None, "BASIC_MILLING")

    def test_toggle_hides_itself_when_it_would_do_nothing(self, dialog):
        from fibsem.applications.autolamella.workflows.tasks.catalogue import (
            has_advanced_tasks,
        )

        assert dialog.advanced_checkbox.isHidden() != has_advanced_tasks()


class TestSecondaryTagColumn:
    def test_shows_only_the_tags_the_heading_does_not(self, dialog):
        """The heading states the primary; repeating it in a column says nothing."""
        from PyQt5.QtWidgets import QLabel

        row = next(r for r, t in dialog._rows.items() if t == "SPOT_BURN_FIDUCIAL")
        chips = [
            lb.text() for lb in dialog.table.cellWidget(row, 2).findChildren(QLabel)
        ]
        assert chips == ["Alignment"], "Correlation is the heading, not a chip"

    def test_is_empty_for_a_task_with_one_purpose(self, dialog):
        from PyQt5.QtWidgets import QLabel

        row = next(r for r, t in dialog._rows.items() if t == "MILL_TRENCH")
        assert not dialog.table.cellWidget(row, 2).findChildren(QLabel)


class TestTooltip:
    def test_carries_what_the_table_has_no_room_for(self, dialog):
        row = next(r for r, t in dialog._rows.items() if t == "MILL_TRENCH")
        tip = dialog.table.item(row, 0).toolTip()
        assert "MILL_TRENCH" in tip, "the identifier lives here now, not in the row"
        assert "Built-in" in tip
        assert "Waffle milling" in tip
        assert "pedestal" in tip, "the full description, not just the summary"

    def test_every_cell_in_a_row_carries_it(self, dialog):
        """Hovering anywhere on the row should explain the row."""
        row = next(iter(dialog._rows))
        tips = {
            dialog.table.item(row, c).toolTip()
            for c in range(dialog.table.columnCount())
        }
        assert len(tips) == 1 and tips != {""}


class TestNaming:
    def test_default_name_avoids_a_collision(self, dialog):
        dialog.select("MILL_TRENCH")
        assert dialog.task_name == "Trench Milling 2"

    def test_default_name_skips_taken_suffixes(self, dialog):
        dialog.select("MILL_ROUGH")
        assert dialog.task_name == "Rough Milling 3"

    def test_free_name_is_left_alone(self, dialog):
        dialog.select("MILL_POLISHING")
        assert dialog.task_name == "Polishing"

    def test_changing_selection_replaces_the_name(self, dialog):
        dialog.select("MILL_POLISHING")
        dialog.name_field.setText("something custom")
        dialog.select("MILL_TRENCH")
        assert dialog.task_name == "Trench Milling 2"

    def test_typed_name_survives_within_one_selection(self, dialog):
        """Re-validating must not overwrite what the user is typing."""
        dialog.select("MILL_POLISHING")
        dialog.name_field.setText("Final Polish")
        assert dialog.task_name == "Final Polish"


class TestValidation:
    def test_add_enabled_for_a_fresh_name(self, dialog):
        dialog.select("MILL_POLISHING")
        assert dialog.add_button.isEnabled()

    def test_duplicate_disables_add_and_suggests_one(self, dialog):
        dialog.select("MILL_POLISHING")
        dialog.name_field.setText("Trench Milling")
        assert not dialog.add_button.isEnabled()
        assert "Trench Milling 2" in dialog.error_label.text()
        assert not dialog.error_label.isHidden()

    def test_the_destination_survives_an_error(self, dialog):
        """It describes the button, not the input.

        Watching it vanish while you fix a typo is how a static line becomes
        wallpaper -- and how nobody reads it when it matters.
        """
        dialog.select("MILL_POLISHING")
        before = dialog.status_label.text()
        dialog.name_field.setText("Trench Milling")
        assert dialog.status_label.text() == before

    def test_a_valid_name_clears_the_error(self, dialog):
        dialog.select("MILL_POLISHING")
        dialog.name_field.setText("Trench Milling")
        dialog.name_field.setText("Something Free")
        assert dialog.error_label.isHidden()
        assert dialog.add_button.isEnabled()

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_disables_add(self, dialog, blank):
        dialog.select("MILL_POLISHING")
        dialog.name_field.setText(blank)
        assert not dialog.add_button.isEnabled()

    def test_accept_refused_while_invalid(self, dialog):
        dialog.select("MILL_POLISHING")
        dialog.name_field.setText("Trench Milling")
        dialog.accept()
        assert dialog.result() != AddTaskDialog.Accepted

    def test_states_where_the_task_lands(self, dialog):
        dialog.select("MILL_POLISHING")
        assert "all 4 existing lamellae" in dialog.status_label.text()

    def test_get_task_info_returns_type_and_name(self, dialog):
        dialog.select("MILL_POLISHING")
        dialog.name_field.setText("Final Polish")
        assert dialog.get_task_info() == ("MILL_POLISHING", "Final Polish")


class TestExistingCount:
    def test_counts_what_the_protocol_already_holds(self, app):
        """Also what explains a default name of "Trench Milling 3"."""
        from PyQt5.QtWidgets import QLabel

        from fibsem.applications.autolamella.workflows.tasks import get_tasks

        existing = {}
        for name, task_type in [
            ("Trench Milling", "MILL_TRENCH"),
            ("Trench Milling 2", "MILL_TRENCH"),
            ("Rough Milling", "MILL_ROUGH"),
        ]:
            config = get_tasks()[task_type].config_cls()
            config.task_name = name
            existing[name] = config

        d = AddTaskDialog(existing, lamella_count=1)
        d.show()
        app.processEvents()
        try:

            def cell_text(task_type):
                row = next(r for r, t in d._rows.items() if t == task_type)
                return " ".join(
                    lb.text() for lb in d.table.cellWidget(row, 0).findChildren(QLabel)
                )

            assert "2 in this protocol" in cell_text("MILL_TRENCH")
            assert "1 in this protocol" in cell_text("MILL_ROUGH")
            assert "in this protocol" not in cell_text("MILL_POLISHING")
            d.select("MILL_TRENCH")
            assert d.task_name == "Trench Milling 3"
        finally:
            d.close()
            d.deleteLater()


class TestPlugins:
    def test_a_plugin_task_is_labelled_in_its_tooltip(self, dialog):
        """Only reachable with a plugin installed; CI has none."""
        plugins = [
            t
            for t in available_task_types()
            if (src := task_source(t)) is not None and src.source.name == "PLUGIN"
        ]
        if not plugins:
            pytest.skip("no plugin tasks installed")
        dialog.advanced_checkbox.setChecked(True)
        row = next(r for r, t in dialog._rows.items() if t == plugins[0])
        assert "Plugin" in dialog.table.item(row, 0).toolTip()

    def test_a_task_declaring_no_tags_still_appears(self, dialog):
        untagged = [
            t for t, c in available_task_types().items() if not task_info(c).tags
        ]
        if not untagged:
            pytest.skip("every installed task declares a tag")
        dialog.advanced_checkbox.setChecked(True)
        assert untagged[0] in _visible_tasks(dialog)
