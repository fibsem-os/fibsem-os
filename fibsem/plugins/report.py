"""What is registered, where it came from, and why something is missing.

Assembles the three plugin groups into one listing and renders it as text.
Both surfaces read from here -- ``fibsem-cli plugins`` and the Plugins panel --
so a report a user pastes into a bug tracker says the same thing as the dialog
support is looking at.

The interesting rows are the ones :func:`~fibsem.milling.patterning.get_patterns`
and friends cannot return:

* **failed** -- an entry point that did not load. It is absent from every
  registry, so without the load records from ``fibsem.plugins.loader`` there is
  nothing to list and the user's only evidence is a log line that scrolled past
  at startup.
* **shadowed** -- a plugin whose name a built-in already claims. It loads, it
  registers, and then ``{**plugins, **registered, **builtins}`` overwrites it.
  The merged dict cannot express that it was ever there.

Unlike ``loader``, this module imports all three registries, so it must only be
imported well after startup -- from the CLI or the UI, never from a registry.

Script-loaded tasks (FIB-339) are modelled here but not yet produced: nothing
loads a scripts folder today. When that lands it only has to emit
:class:`Extension` rows with ``source=ExtensionSource.SCRIPT`` and a ``path``;
the renderers and the panel already handle them.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Type

from fibsem.plugins.loader import PluginRecord

__all__ = [
    "ExtensionSource",
    "Extension",
    "ExtensionGroup",
    "collect_extensions",
    "render_report",
    "group_counts_phrase",
    "environment_line",
    "summarise",
]


class ExtensionSource(Enum):
    """Where a registered extension came from."""

    BUILTIN = "built-in"
    REGISTERED = "registered"
    PLUGIN = "plugin"
    SCRIPT = "script"
    FAILED = "failed"


# Successes before failures, and built-ins last since they are hidden by
# default. Also the order rows appear in, so the listing is stable between runs
# -- entry_points() iterates in filesystem order, which differs by machine, and
# an unstable listing is useless for comparing two pasted reports.
_SOURCE_ORDER = {
    ExtensionSource.PLUGIN: 0,
    ExtensionSource.SCRIPT: 1,
    ExtensionSource.REGISTERED: 2,
    ExtensionSource.FAILED: 3,
    ExtensionSource.BUILTIN: 4,
}


@dataclass(frozen=True)
class Extension:
    """One row of the listing."""

    group: str
    """Entry point group, e.g. ``"fibsem.patterns"``."""

    name: Optional[str]
    """The name it registered under. ``None`` when nothing registered."""

    source: ExtensionSource

    target: str
    """What the row points at.

    For anything that loaded, the resolved ``module.QualName``. For a failure,
    the *declared* ``module:Attr`` exactly as written in the plugin's
    pyproject -- a listing that only shows what resolved cannot answer "why
    isn't mine here?", so a failed row has to show what was asked for.
    """

    distribution: Optional[str] = None
    version: Optional[str] = None

    path: Optional[str] = None
    """Home-relative path to the source file, for script-loaded extensions."""

    problem: Optional[str] = None
    """Why this row is inactive: a load failure, or what shadowed it."""

    @property
    def is_problem(self) -> bool:
        return self.problem is not None

    @property
    def shadowed(self) -> bool:
        """Loaded and registered, but another registration won the name."""
        return self.problem is not None and self.source is not ExtensionSource.FAILED

    @property
    def origin(self) -> str:
        """``"lab-patterns 0.3.1"``, a script path, or ``""``."""
        if self.path is not None:
            return self.path
        if self.distribution is None:
            return ""
        if self.version is None:
            return self.distribution
        return f"{self.distribution} {self.version}"


@dataclass(frozen=True)
class ExtensionGroup:
    """One entry point group and everything registered under it."""

    group: str
    """The literal entry point group string, e.g. ``"fibsem.patterns"``.

    Shown verbatim rather than only the friendly label: a group name mistyped
    in a plugin's pyproject registers nothing and produces no error anywhere,
    so the exact string is the whole diagnostic. The user diffs it against
    their own file.
    """

    label: str
    """Human name, e.g. ``"Milling patterns"``."""

    extensions: Tuple[Extension, ...]

    def counts(self) -> "Dict[ExtensionSource, int]":
        counts: Dict[ExtensionSource, int] = {}
        for extension in self.extensions:
            counts[extension.source] = counts.get(extension.source, 0) + 1
        return counts

    def visible(self, show_builtins: bool) -> Tuple[Extension, ...]:
        if show_builtins:
            return self.extensions
        return tuple(e for e in self.extensions if e.source is not ExtensionSource.BUILTIN)


def home_relative(path: str) -> str:
    """``/Users/someone/.autolamella/x`` -> ``~/.autolamella/x``.

    This listing is meant to be pasted into a bug report, so the paths in it
    should not carry the user's name. Purely cosmetic -- nothing parses these
    back.
    """
    try:
        home = os.path.expanduser("~")
    except Exception:  # pragma: no cover - expanduser is not supposed to fail
        return path
    if home and path.startswith(home):
        return "~" + path[len(home):]
    return path


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _shadow_reason(name: str, builtins: Dict[str, type], registered: Dict[str, type]) -> str:
    if name in builtins:
        return "name taken by a built-in - the built-in is used, this is inactive"
    if name in registered:
        return "name taken by a runtime registration - this is inactive"
    return "name taken by another plugin - this is inactive"


def _build_group(
    group: str,
    label: str,
    records: Iterable[PluginRecord],
    builtins: Dict[str, Type],
    registered: Dict[str, Type],
    active: Dict[str, Type],
) -> ExtensionGroup:
    """Assemble one group's rows from its load records and its registries.

    ``active`` is what the registry actually hands out, and it is the arbiter:
    a plugin is shadowed precisely when the name it claimed now maps to a
    different class.
    """
    extensions: List[Extension] = []

    for record in records:
        if not record.loaded:
            extensions.append(
                Extension(
                    group=group,
                    name=None,
                    source=ExtensionSource.FAILED,
                    target=record.value,
                    distribution=record.distribution,
                    version=record.version,
                    # Spell out the consequence. "ModuleNotFoundError" on its
                    # own does not tell a user whether their plugin is
                    # half-loaded or absent.
                    problem=f"{record.error} - nothing was registered",
                )
            )
            continue

        assert record.name is not None and record.cls is not None
        won = active.get(record.name) is record.cls
        extensions.append(
            Extension(
                group=group,
                name=record.name,
                source=ExtensionSource.PLUGIN,
                target=_qualified(record.cls),
                distribution=record.distribution,
                version=record.version,
                problem=None if won else _shadow_reason(record.name, builtins, registered),
            )
        )

    for name, cls in registered.items():
        if name in builtins:
            # Registered at runtime, then shadowed by the built-in of the same
            # name. Same story as a shadowed plugin.
            problem = "name taken by a built-in - the built-in is used, this is inactive"
        else:
            problem = None
        extensions.append(
            Extension(
                group=group,
                name=name,
                source=ExtensionSource.REGISTERED,
                target=_qualified(cls),
                problem=problem,
            )
        )

    for name, cls in builtins.items():
        extensions.append(
            Extension(
                group=group,
                name=name,
                source=ExtensionSource.BUILTIN,
                target=_qualified(cls),
            )
        )

    extensions.sort(key=lambda e: (_SOURCE_ORDER[e.source], e.name or e.target))
    return ExtensionGroup(group=group, label=label, extensions=tuple(extensions))


def collect_extensions() -> Tuple[ExtensionGroup, ...]:
    """Every registered extension across all three groups, with provenance."""
    from fibsem.applications.autolamella.workflows.tasks import (
        BUILTIN_TASKS,
        REGISTERED_TASKS,
        TASK_ENTRY_POINT_GROUP,
        get_task_plugin_records,
        get_tasks,
    )
    from fibsem.milling.patterning import (
        BUILTIN_PATTERNS,
        PATTERN_ENTRY_POINT_GROUP,
        REGISTERED_PATTERNS,
        get_pattern_plugin_records,
        get_patterns,
    )
    from fibsem.milling.strategy import (
        BUILTIN_STRATEGIES,
        REGISTERED_STRATEGIES,
        STRATEGY_ENTRY_POINT_GROUP,
        get_strategies,
        get_strategy_plugin_records,
    )

    return (
        _build_group(
            group=PATTERN_ENTRY_POINT_GROUP,
            label="Milling patterns",
            records=get_pattern_plugin_records(),
            builtins=BUILTIN_PATTERNS,
            registered=REGISTERED_PATTERNS,
            active=get_patterns(),
        ),
        _build_group(
            group=STRATEGY_ENTRY_POINT_GROUP,
            label="Milling strategies",
            records=get_strategy_plugin_records(),
            builtins=BUILTIN_STRATEGIES,
            registered=REGISTERED_STRATEGIES,
            active=get_strategies(),
        ),
        _build_group(
            group=TASK_ENTRY_POINT_GROUP,
            label="AutoLamella tasks",
            records=get_task_plugin_records(),
            builtins=BUILTIN_TASKS,
            registered=REGISTERED_TASKS,
            active=get_tasks(),
        ),
    )


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def group_counts_phrase(group: ExtensionGroup) -> str:
    """``"1 plugin, 1 failed, 11 built-in"`` for a group header.

    Shared with the Plugins panel rather than reimplemented there: two headers
    disagreeing about how many plugins are installed would undermine the point
    of both.
    """
    counts = group.counts()
    parts = []
    for source in (ExtensionSource.PLUGIN, ExtensionSource.SCRIPT, ExtensionSource.REGISTERED):
        if counts.get(source):
            parts.append(_plural(counts[source], source.value))
    if counts.get(ExtensionSource.FAILED):
        parts.append(f"{counts[ExtensionSource.FAILED]} failed")
    if counts.get(ExtensionSource.BUILTIN):
        parts.append(f"{counts[ExtensionSource.BUILTIN]} built-in")
    return ", ".join(parts) if parts else "nothing registered"


def summarise(groups: Iterable[ExtensionGroup]) -> Tuple[str, int]:
    """``("2 plugins, 1 script, 1 failed to load", 20)``.

    The count is of built-ins, which are hidden by default in both surfaces.
    """
    totals: Dict[ExtensionSource, int] = {}
    for group in groups:
        for source, count in group.counts().items():
            totals[source] = totals.get(source, 0) + count

    parts = []
    for source in (ExtensionSource.PLUGIN, ExtensionSource.SCRIPT, ExtensionSource.REGISTERED):
        if totals.get(source):
            parts.append(_plural(totals[source], source.value))
    if totals.get(ExtensionSource.FAILED):
        parts.append(f"{totals[ExtensionSource.FAILED]} failed to load")

    summary = ", ".join(parts) if parts else "no plugins installed"
    return summary, totals.get(ExtensionSource.BUILTIN, 0)


def environment_line() -> str:
    """``"fibsem 0.5.2 - Python 3.11.9 - ~/miniconda3/envs/fibsem"``.

    Pasted into a bug report, "which install is this?" is otherwise always the
    next question -- a microscope PC routinely has several environments and the
    plugin is only installed into one of them.
    """
    try:
        import fibsem

        version = getattr(fibsem, "__version__", "unknown")
    except Exception:  # pragma: no cover - fibsem is obviously importable here
        version = "unknown"
    return (
        f"fibsem {version} - Python {platform.python_version()} - "
        f"{home_relative(sys.prefix)}"
    )


_MIN_NAME_WIDTH = 17
_MIN_ORIGIN_WIDTH = 20
_INDENT = "  "


def render_report(groups: Iterable[ExtensionGroup], show_builtins: bool = False) -> str:
    """The plugin listing as text.

    This is what ``fibsem-cli plugins`` prints and what the panel's Copy report
    button puts on the clipboard -- one renderer, so a pasted report and the
    dialog never disagree.

    Problems get a ``!`` continuation line indented under their entry: greppable
    (``| grep '!'``), and it keeps the happy path at one line per plugin so a
    long list stays scannable.

    ASCII only. This gets pasted into terminals, issue trackers and Windows
    consoles, where an em dash is a mojibake risk for no benefit.
    """
    groups = list(groups)
    rows_by_group = [(g, g.visible(show_builtins)) for g in groups]
    visible = [e for _, rows in rows_by_group for e in rows]

    # Size the columns to the content rather than fixing them: task types run to
    # nearly 30 characters while pattern names are short, and a name that
    # overflows a fixed column runs into the one after it. Every width is
    # derived purely from the rows being printed, so the same input always
    # renders identically -- two pasted reports stay diffable. The +1 is the
    # gutter, and the reason a value exactly as wide as its column does not
    # collide with the next.
    name_width = max([_MIN_NAME_WIDTH] + [len(e.name or "-") for e in visible]) + 1
    source_width = max([len(e.source.value) for e in visible] or [0]) + 1
    origins = [len(e.origin) for e in visible if e.path is None and e.origin]
    # With no plugins installed at all, every row is a built-in with no
    # distribution to show. Collapse the column rather than indenting the whole
    # listing past an empty one.
    origin_width = max([_MIN_ORIGIN_WIDTH] + origins) + 1 if origins else 0

    lines: List[str] = []
    for group, rows in rows_by_group:
        lines.append(f"{group.group}  ({group_counts_phrase(group)})")
        if not rows:
            lines.append(f"{_INDENT}-")
        for extension in rows:
            # A failed entry point registered nothing, so it has no name to
            # show; "-" holds the column while the declared target carries the
            # information.
            name = extension.name or "-"
            lines.append(
                _INDENT
                + name.ljust(name_width)
                + extension.source.value.ljust(source_width)
                + _origin_and_target(extension, origin_width)
            )
            if extension.path is not None:
                # The path took the origin column, so the class goes below it
                # rather than being dropped.
                lines.append(_INDENT + " " * (name_width + source_width) + extension.target)
            if extension.problem:
                lines.append(_INDENT + " " * name_width + "! " + extension.problem)
        lines.append("")

    summary, hidden = summarise(groups)
    if hidden and not show_builtins:
        summary += f". {hidden} built-in hidden (use --all)."
    else:
        summary += "."
    lines.append(summary)
    lines.append(environment_line())
    return "\n".join(lines)


def _origin_and_target(extension: Extension, origin_width: int) -> str:
    if extension.path is not None:
        return extension.path
    # Built-ins have no distribution, but still pad the column so their targets
    # line up with the plugins above them.
    return extension.origin.ljust(origin_width) + extension.target
