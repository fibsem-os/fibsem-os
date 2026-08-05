"""Tests for the Plugins dialog.

Built from synthetic groups rather than whatever happens to be installed: the
rows worth testing are the ones a normal environment does not have -- a plugin
that failed to load, one a built-in shadowed, and a script-loaded task that
nothing produces until FIB-339 lands.
"""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QLabel  # noqa: E402

from fibsem.plugins.report import (  # noqa: E402
    Extension,
    ExtensionGroup,
    ExtensionSource,
    environment_line,
    render_report,
)
from fibsem.ui.widgets.plugins_dialog import PluginsDialog  # noqa: E402

PLUGIN = Extension(
    group="fibsem.patterns",
    name="WaffleTrench",
    source=ExtensionSource.PLUGIN,
    target="lab_patterns.trench.WaffleTrenchPattern",
    distribution="lab-patterns",
    version="0.3.1",
)
BUILTIN = Extension(
    group="fibsem.patterns",
    name="Rectangle",
    source=ExtensionSource.BUILTIN,
    target="fibsem.milling.patterning.patterns2.RectanglePattern",
)
SHADOWED = Extension(
    group="fibsem.strategies",
    name="Overtilt",
    source=ExtensionSource.PLUGIN,
    target="lab_patterns.strategy.LabOvertilt",
    distribution="lab-patterns",
    version="0.3.1",
    problem="name taken by a built-in - the built-in is used, this is inactive",
)
SCRIPT = Extension(
    group="fibsem.tasks",
    name="MY_POLISH",
    source=ExtensionSource.SCRIPT,
    target="custom_polish.MyPolishTask",
    path="~/.autolamella/scripts/custom_polish.py",
)
FAILED = Extension(
    group="fibsem.tasks",
    name=None,
    source=ExtensionSource.FAILED,
    target="lab_patterns.tasks:BadTask",
    distribution="lab-patterns",
    version="0.3.1",
    problem="not a subclass of AutoLamellaTask - nothing was registered",
)

GROUPS = (
    ExtensionGroup("fibsem.patterns", "Milling patterns", (PLUGIN, BUILTIN)),
    ExtensionGroup("fibsem.strategies", "Milling strategies", (SHADOWED,)),
    ExtensionGroup("fibsem.tasks", "AutoLamella tasks", (SCRIPT, FAILED)),
)


@pytest.fixture
def dialog(qapp):
    widget = PluginsDialog(groups=GROUPS)
    yield widget
    widget.deleteLater()


def _cell_text(dialog, row: int, column: int) -> str:
    """Everything a cell widget renders, joined -- the name and detail lines."""
    widget = dialog.table.cellWidget(row, column)
    if widget is None:
        item = dialog.table.item(row, column)
        return item.text() if item else ""
    return "\n".join(label.text() for label in widget.findChildren(QLabel))


def _row_of(dialog, extension: Extension) -> int:
    return dialog.rows.index(extension)


def _tooltip_of(dialog, extension: Extension) -> str:
    return dialog.table.cellWidget(_row_of(dialog, extension), 0).toolTip()


def _chip_colours(dialog, extension: Extension) -> list:
    """The colour each chip in the Source cell is painted, left to right."""
    import re

    cell = dialog.table.cellWidget(_row_of(dialog, extension), 1)
    return [
        re.search(r"color: (#[0-9a-fA-F]{6});", label.styleSheet()).group(1)
        for label in cell.findChildren(QLabel)
    ]


def test_every_extension_gets_a_row_in_group_order(dialog):
    # The built-in is counted but hidden, so four of the five.
    assert dialog.table.rowCount() == 4
    assert [e.group for e in dialog.rows] == [
        "fibsem.patterns",
        "fibsem.strategies",
        "fibsem.tasks",
        "fibsem.tasks",
    ]


def test_the_entry_point_group_column_shows_the_literal_string(dialog):
    """The exact string, not just "Milling patterns".

    A group mistyped in a plugin's pyproject registers nothing and logs nothing,
    so comparing the real string against their own file is the only way a user
    can find that mistake.
    """
    groups = {_cell_text(dialog, row, 2) for row in range(dialog.table.rowCount())}
    assert groups == {"fibsem.patterns", "fibsem.strategies", "fibsem.tasks"}


def test_failed_row_leads_with_what_was_declared_and_why_it_failed(dialog):
    """No name registered, so the declared target takes the name line."""
    text = _cell_text(dialog, _row_of(dialog, FAILED), 0)

    assert "lab_patterns.tasks:BadTask" in text
    assert "Not a subclass of AutoLamellaTask - nothing was registered" in text
    assert "Failed" in _cell_text(dialog, _row_of(dialog, FAILED), 1)
    # Which package declared it is unrecoverable elsewhere when several plugins
    # are installed, so it has to be reachable -- just not at the reason's expense.
    assert "lab-patterns 0.3.1" in _tooltip_of(dialog, FAILED)


def test_shadowed_row_reads_as_unavailable_rather_than_broken(dialog):
    """The consequence, not the mechanism: it loaded, it just is not in use.

    "shadowed" describes fibsem's merge order, which is not the user's problem.
    """
    row = _row_of(dialog, SHADOWED)
    source = _cell_text(dialog, row, 1)

    assert "Unavailable" in source
    # Still a plugin: it is installed and it loaded, it just is not the one used.
    assert "Plugin" in source
    assert "built-in" in _tooltip_of(dialog, SHADOWED)


def test_colour_marks_state_not_where_something_came_from(dialog):
    """A plugin and a script are both in use, so they get the same colour.

    Separate colours would imply a distinction that does not exist -- the chip
    label already says which is which. Four states, four colours: in use,
    installed-but-unused, unremarkable, broken.
    """
    from fibsem.ui.widgets.plugins_dialog import (
        _SOURCE_STYLE,
        _TEXT_MUTED,
        _WARN,
        _reason_colour,
    )

    _, plugin_colour = _SOURCE_STYLE[ExtensionSource.PLUGIN]
    _, script_colour = _SOURCE_STYLE[ExtensionSource.SCRIPT]
    _, failed_colour = _SOURCE_STYLE[ExtensionSource.FAILED]
    _, builtin_colour = _SOURCE_STYLE[ExtensionSource.BUILTIN]

    assert plugin_colour == script_colour
    assert len({plugin_colour, _WARN, builtin_colour, failed_colour}) == 4
    assert builtin_colour == _TEXT_MUTED

    # The reason wears the same colour as the chip that raised it.
    assert _reason_colour(FAILED) == failed_colour
    assert _reason_colour(SHADOWED) == _WARN
    assert _reason_colour(PLUGIN) == _TEXT_MUTED


def test_an_unavailable_row_does_not_wear_the_in_use_colour(dialog):
    """It is installed, but it is not the copy being used.

    Leaving the Source chip blue would say "in use" next to a chip saying it is
    not, and grey would file one of the two rows this dialog exists to surface
    alongside the built-ins nobody came for.
    """
    from fibsem.ui.widgets.plugins_dialog import _SOURCE_STYLE, _TEXT_MUTED, _WARN

    _, in_use = _SOURCE_STYLE[ExtensionSource.PLUGIN]
    chips = _chip_colours(dialog, SHADOWED)

    assert in_use not in chips
    assert chips == [_TEXT_MUTED, _WARN]

    # ...while an active plugin still does.
    assert _chip_colours(dialog, PLUGIN) == [in_use]


def test_an_inactive_row_explains_itself_without_being_selected(dialog):
    """The reason is on the row, not behind a click or a hover.

    A shadowed plugin used to explain itself only in a detail panel below the
    table, which meant the Unavailable chip was unexplained until you selected
    it -- the same shape of silence this dialog exists to end.
    """
    text = _cell_text(dialog, _row_of(dialog, SHADOWED), 0)
    assert "Name taken by a built-in" in text

    # The class path it displaced is still reachable, just not at the reason's
    # expense.
    assert "lab_patterns.strategy.LabOvertilt" in _tooltip_of(dialog, SHADOWED)


def test_a_healthy_row_shows_what_it_resolved_to(dialog):
    text = _cell_text(dialog, _row_of(dialog, PLUGIN), 0)
    assert "lab_patterns.trench.WaffleTrenchPattern" in text
    assert "lab-patterns 0.3.1" in text


def test_open_folder_is_enabled_only_for_something_with_a_file(dialog):
    """The action is meaningless for a packaged plugin -- there is no file."""
    dialog.table.selectRow(_row_of(dialog, SCRIPT))
    assert dialog.open_folder_button.isEnabled()

    dialog.table.selectRow(_row_of(dialog, PLUGIN))
    assert not dialog.open_folder_button.isEnabled()


def test_open_folder_is_absent_when_nothing_has_a_file(qapp):
    """Which is every environment until the scripts folder lands (FIB-339).

    A button that can never be enabled reads as broken, not as inapplicable.
    """
    packaged_only = (
        ExtensionGroup("fibsem.patterns", "Milling patterns", (PLUGIN,)),
    )
    widget = PluginsDialog(groups=packaged_only)
    try:
        assert not widget.open_folder_button.isVisible()
    finally:
        widget.deleteLater()

    with_a_script = packaged_only + (
        ExtensionGroup("fibsem.tasks", "AutoLamella tasks", (SCRIPT,)),
    )
    widget = PluginsDialog(groups=with_a_script)
    try:
        assert not widget.open_folder_button.isHidden()
    finally:
        widget.deleteLater()


def test_show_builtins_reveals_them_and_drops_the_hidden_count(dialog):
    assert "1 built-in hidden" in dialog.meta_label.text()

    dialog.builtins_checkbox.setChecked(True)
    assert dialog.table.rowCount() == 5
    assert BUILTIN in dialog.rows
    assert "built-in hidden" not in dialog.meta_label.text()

    dialog.builtins_checkbox.setChecked(False)
    assert dialog.table.rowCount() == 4


def test_the_selection_survives_a_rebuild(dialog):
    """Toggling built-ins must not bounce the selection back to the top."""
    dialog.table.selectRow(_row_of(dialog, FAILED))
    dialog.builtins_checkbox.setChecked(True)

    assert dialog.selected_extension() is FAILED


def test_copy_report_emits_exactly_what_the_cli_prints(dialog):
    """One renderer for both surfaces.

    If these drift, a report a user pastes into a bug tracker stops matching
    what support is looking at on screen -- worse than having only one of them.
    """
    assert dialog.report_text() == render_report(GROUPS, show_builtins=False)

    dialog.builtins_checkbox.setChecked(True)
    assert dialog.report_text() == render_report(GROUPS, show_builtins=True)


def test_copy_confirms_then_recovers_so_it_can_be_copied_again(dialog):
    """An icon button gives no other sign it did anything."""
    dialog.copy_report()
    assert dialog.copy_button.toolTip() == "Copied"

    # Toggling changes what a copy would produce, so the button has to come back.
    dialog.builtins_checkbox.setChecked(True)
    assert "Copy this listing" in dialog.copy_button.toolTip()


def test_meta_line_counts_everything_installed(dialog):
    # The shadowed plugin counts as installed: it is on disk and it loaded, it
    # just is not the one in use. Saying "1 plugin" would contradict the row
    # sitting in the table below.
    assert dialog.meta_label.text().startswith("2 plugins, 1 script, 1 failed to load")
    # Per-group counts are the tooltip, so the line stays one line.
    assert "fibsem.strategies: 1 plugin" in dialog.meta_label.toolTip()
    # "which install is this?" is the first question about any pasted report,
    # and it used to live on a footer that no longer exists.
    assert environment_line() in dialog.meta_label.toolTip()


def test_a_group_with_nothing_registered_leaves_an_empty_table(qapp):
    widget = PluginsDialog(groups=(ExtensionGroup("fibsem.patterns", "Milling patterns", ()),))
    try:
        assert widget.table.rowCount() == 0
        assert widget.selected_extension() is None
        assert not widget.open_folder_button.isEnabled()
        assert "no plugins installed" in widget.meta_label.text()
    finally:
        widget.deleteLater()


def test_no_widget_stylesheet_swallows_its_own_tooltip(dialog):
    """A selector-less rule on a widget also styles the tooltip shown over it.

    It sits nearer than the application stylesheet, so it wins: `background:
    transparent` on a table cell left its tooltip as floating text with no panel
    behind it, and a muted `color` would make tooltip text unreadable. Every rule
    written without a selector therefore has to restate the QToolTip block.

    Checked as an invariant rather than by rendering, because the offscreen
    platform never realises a tooltip window to look at.
    """
    from PyQt5.QtWidgets import QWidget

    offenders = []
    for widget in [dialog] + dialog.findChildren(QWidget):
        css = widget.styleSheet()
        if not css:
            continue
        # A rule that names a type or an id cannot match QToolTip.
        scoped = css.lstrip().startswith(("Q", "#", "*"))
        if not scoped and "QToolTip" not in css:
            offenders.append((type(widget).__name__, widget.toolTip(), css[:60]))

    assert not offenders, offenders


def test_the_tooltip_rule_matches_the_application_theme(dialog):
    """Restated locally, so it has to stay in step with the real one."""
    from fibsem.ui import stylesheets
    from fibsem.ui.widgets.plugins_dialog import _BORDER, _PANEL, _TEXT

    app_rule = stylesheets.NAPARI_STYLE.split("QToolTip {")[1].split("}")[0]
    for value in (_PANEL, _TEXT, _BORDER):
        assert value in app_rule, f"{value} no longer matches the app tooltip style"
