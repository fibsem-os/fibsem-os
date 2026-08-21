"""The plugin listing's text rendering.

``fibsem-cli plugins`` output and the Plugins panel's Copy report button are the
same string, and it gets pasted into bug reports -- so it has to stay readable
and stable rather than merely non-empty.

These build :class:`Extension` rows directly instead of installing plugins.
That is deliberate: it covers the rows this environment cannot produce (a
script-loaded task, which nothing loads until FIB-339 lands) and the pathological
widths that would otherwise need a package named to order. The other half of the
contract -- that real entry points actually produce these rows -- lives in
tests/test_plugin_entry_points.py against an installed fixture.
"""

import importlib
import os

import pytest

from fibsem.plugins import report
from fibsem.plugins.report import (
    Extension,
    ExtensionGroup,
    ExtensionSource,
    environment_line,
    group_counts_phrase,
    home_relative,
    render_report,
    summarise,
)


def _extension(**kwargs) -> Extension:
    defaults = dict(
        group="fibsem.tasks",
        name="MY_TASK",
        source=ExtensionSource.PLUGIN,
        target="lab_patterns.tasks.MyTask",
        distribution="lab-patterns",
        version="0.3.1",
    )
    defaults.update(kwargs)
    return Extension(**defaults)


def _group(
    *extensions: Extension, group="fibsem.tasks", label="AutoLamella tasks"
) -> ExtensionGroup:
    return ExtensionGroup(group=group, label=label, extensions=tuple(extensions))


def _row_for(text: str, name: str) -> str:
    return next(line for line in text.splitlines() if line.strip().startswith(name))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_columns_do_not_collide_when_a_value_fills_its_column():
    """A name or distribution as wide as its column must not run into the next.

    Fixed column widths are the obvious implementation and the wrong one: task
    types run to nearly thirty characters, and "built-in" is exactly as long as
    a naive source column, which silently produced `built-infibsem.milling...`.
    """
    long_name = "SELECT_FLUORESCENCE_POSITION_AND_THEN_SOME"
    text = render_report(
        [
            _group(
                _extension(name=long_name),
                _extension(
                    name="SHORT",
                    source=ExtensionSource.BUILTIN,
                    distribution=None,
                    version=None,
                    target="fibsem.tasks.Short",
                ),
            )
        ],
        show_builtins=True,
    )

    # Every row splits into exactly the columns it should: nothing has been
    # glued to its neighbour by a value that filled its column exactly.
    assert _row_for(text, long_name).split() == [
        long_name,
        "plugin",
        "lab-patterns",
        "0.3.1",
        "lab_patterns.tasks.MyTask",
    ]
    assert _row_for(text, "SHORT").split() == [
        "SHORT",
        "built-in",
        "fibsem.tasks.Short",
    ]


def test_builtin_targets_line_up_with_plugin_targets():
    """Built-ins have no distribution, but still hold the column."""
    text = render_report(
        [
            _group(
                _extension(name="PLUGGED"),
                _extension(
                    name="NATIVE",
                    source=ExtensionSource.BUILTIN,
                    distribution=None,
                    version=None,
                    target="fibsem.tasks.Native",
                ),
            )
        ],
        show_builtins=True,
    )
    plugged, native = _row_for(text, "PLUGGED"), _row_for(text, "NATIVE")
    assert plugged.index("lab_patterns.tasks.MyTask") == native.index(
        "fibsem.tasks.Native"
    )


def test_no_distribution_column_when_nothing_is_installed():
    """With only built-ins, do not indent every row past an empty column.

    Also the one arrangement where "built-in" is the widest source value, so a
    source column sized to anything but the content puts it flush against the
    class path -- `built-infibsem.milling...`, which is what shipped first.
    """
    text = render_report(
        [
            _group(
                _extension(
                    name="NATIVE",
                    source=ExtensionSource.BUILTIN,
                    distribution=None,
                    version=None,
                    target="fibsem.tasks.Native",
                )
            )
        ],
        show_builtins=True,
    )
    row = _row_for(text, "NATIVE")
    assert row.split() == ["NATIVE", "built-in", "fibsem.tasks.Native"]
    assert row.rstrip().endswith("fibsem.tasks.Native")


def test_rendering_is_deterministic():
    """Two runs on one machine must be diffable; entry_points() is not ordered."""
    groups = [_group(_extension(name="B"), _extension(name="A"))]
    assert render_report(groups) == render_report(groups)


# ---------------------------------------------------------------------------
# Rows that the registries cannot express
# ---------------------------------------------------------------------------


def test_failed_row_shows_the_declared_target_and_the_reason():
    """A failed entry point has no name, so the declared value carries the row."""
    text = render_report(
        [
            _group(
                _extension(
                    name=None,
                    source=ExtensionSource.FAILED,
                    target="lab_patterns.tasks:BadTask",
                    problem="not a subclass of AutoLamellaTask - nothing was registered",
                )
            )
        ]
    )

    assert "lab_patterns.tasks:BadTask" in text
    assert "-  " in text  # the empty name column
    # Problems are indented under their entry and marked, so `| grep '!'` works.
    problem = next(line for line in text.splitlines() if "!" in line)
    assert problem.lstrip().startswith("! not a subclass of AutoLamellaTask")
    assert problem.startswith("  ")


def test_shadowed_row_is_listed_with_its_reason():
    text = render_report(
        [
            _group(
                _extension(
                    name="Rectangle",
                    problem="name taken by a built-in - the built-in is used, this is inactive",
                )
            )
        ]
    )
    assert "Rectangle" in text
    assert "! name taken by a built-in" in text


def test_script_row_shows_its_path_and_keeps_the_class():
    """The path takes the origin column, so the class moves to the line below.

    Nothing produces script rows until the scripts folder lands (FIB-339); this
    is what it has to emit for the listing to be complete.
    """
    text = render_report(
        [
            _group(
                _extension(
                    name="MY_POLISH",
                    source=ExtensionSource.SCRIPT,
                    target="custom_polish.MyPolishTask",
                    distribution=None,
                    version=None,
                    path="~/.autolamella/scripts/custom_polish.py",
                )
            )
        ]
    )
    lines = text.splitlines()
    row = lines.index(_row_for(text, "MY_POLISH"))
    assert "~/.autolamella/scripts/custom_polish.py" in lines[row]
    assert lines[row + 1].strip() == "custom_polish.MyPolishTask"


def test_empty_group_says_so_rather_than_rendering_nothing():
    text = render_report([_group()])
    assert "fibsem.tasks  (nothing registered)" in text


# ---------------------------------------------------------------------------
# A registry that will not import
# ---------------------------------------------------------------------------


def _boom():
    raise ImportError("No module named 'psygnal'")


def test_a_group_that_cannot_be_inspected_becomes_one_row_saying_so():
    group = report._collect_group(
        "fibsem.tasks",
        "AutoLamella tasks",
        "fibsem.applications.autolamella.workflows.tasks",
        _boom,
    )

    (row,) = group.extensions
    assert row.source is ExtensionSource.FAILED
    # The module that could not be read, so the reader knows what to try importing.
    assert row.target == "fibsem.applications.autolamella.workflows.tasks"
    assert "No module named 'psygnal'" in row.problem
    # ...and that this is not one plugin failing but the whole group being unknown.
    assert "whole group could not be inspected" in row.problem


def test_one_unreadable_group_does_not_take_the_others_down(monkeypatch):
    """The point of the isolation: a broken task registry still leaves a report.

    This listing is what a user is asked for when their install misbehaves. If it
    needs all three registries intact to print anything, it cannot do the one job
    it exists for.
    """
    monkeypatch.setattr(
        report,
        "_GROUPS",
        tuple(
            (group, label, module, _boom if group == "fibsem.tasks" else collect)
            for group, label, module, collect in report._GROUPS
        ),
    )

    groups = {g.group: g for g in report.collect_extensions()}

    assert set(groups) == {"fibsem.patterns", "fibsem.strategies", "fibsem.tasks"}
    assert [e.source for e in groups["fibsem.tasks"].extensions] == [
        ExtensionSource.FAILED
    ]
    # The milling groups are not merely present, they are populated: every
    # install has built-in patterns and strategies.
    for intact in ("fibsem.patterns", "fibsem.strategies"):
        assert any(
            e.source is ExtensionSource.BUILTIN for e in groups[intact].extensions
        )


def test_an_unreadable_group_still_renders(monkeypatch):
    """Rendering must not assume a row has a name or a distribution."""
    group = report._collect_group(
        "fibsem.tasks", "AutoLamella tasks", "some.module", _boom
    )

    text = render_report([group])

    assert "fibsem.tasks" in text
    assert "some.module" in text
    assert "! ImportError: No module named 'psygnal'" in text


def test_the_group_table_matches_the_registries():
    """_GROUPS repeats the group strings, because it is used when the import fails.

    Repeating them is the only option -- the constant lives in the module that may
    not be importable -- so the duplication gets pinned here instead.
    """
    from fibsem.applications.autolamella.workflows.tasks import TASK_ENTRY_POINT_GROUP
    from fibsem.milling.patterning import PATTERN_ENTRY_POINT_GROUP
    from fibsem.milling.strategy import STRATEGY_ENTRY_POINT_GROUP

    assert [group for group, _, _, _ in report._GROUPS] == [
        PATTERN_ENTRY_POINT_GROUP,
        STRATEGY_ENTRY_POINT_GROUP,
        TASK_ENTRY_POINT_GROUP,
    ]
    # The module named in the fallback row has to be the one actually imported,
    # or the row sends the reader somewhere irrelevant.
    for _, _, module, _ in report._GROUPS:
        importlib.import_module(module)


# ---------------------------------------------------------------------------
# Counts and trailer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sources, expected",
    [
        ([ExtensionSource.PLUGIN], "1 plugin"),
        ([ExtensionSource.PLUGIN, ExtensionSource.PLUGIN], "2 plugins"),
        ([ExtensionSource.PLUGIN, ExtensionSource.FAILED], "1 plugin, 1 failed"),
        ([ExtensionSource.BUILTIN, ExtensionSource.BUILTIN], "2 built-in"),
        ([], "nothing registered"),
    ],
)
def test_group_counts_phrase(sources, expected):
    group = _group(*[_extension(source=s, name=f"N{i}") for i, s in enumerate(sources)])
    assert group_counts_phrase(group) == expected


def test_summary_counts_across_groups_and_reports_hidden_builtins():
    groups = [
        _group(_extension(), group="fibsem.patterns"),
        _group(
            _extension(source=ExtensionSource.FAILED, name=None),
            _extension(source=ExtensionSource.BUILTIN, name="X"),
            _extension(source=ExtensionSource.BUILTIN, name="Y"),
        ),
    ]
    summary, hidden = summarise(groups)
    assert summary == "1 plugin, 1 failed to load"
    assert hidden == 2

    assert "2 built-in hidden (use --all)." in render_report(groups)
    # ...and not once they are on screen.
    assert "built-in hidden" not in render_report(groups, show_builtins=True)


def test_summary_says_so_when_nothing_is_installed():
    summary, _ = summarise([_group(_extension(source=ExtensionSource.BUILTIN))])
    assert summary == "no plugins installed"


def test_report_ends_with_the_environment_it_describes():
    """Pasted into a bug report, "which install is this?" is the next question."""
    text = render_report([_group(_extension())])
    assert text.splitlines()[-1] == environment_line()
    assert environment_line().startswith("fibsem ")
    assert "Python" in environment_line()


def test_report_is_ascii_so_it_survives_a_windows_console():
    text = render_report(
        [_group(_extension(problem="name taken by a built-in - this is inactive"))]
    )
    text.encode("ascii")  # raises if a dash or bullet crept in


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_home_relative_strips_the_username():
    """Script paths and the env path both carry the user's name otherwise."""
    home = os.path.expanduser("~")
    inside = os.path.join(home, "scripts", "x.py")

    assert home_relative(inside) == "~" + inside[len(home) :]
    assert not home_relative(inside).startswith(home)
    # Anything outside the home directory is left exactly as it is.
    assert home_relative("/opt/env/fibsem") == "/opt/env/fibsem"


def test_environment_line_does_not_leak_the_home_path():
    assert os.path.expanduser("~") not in environment_line()
