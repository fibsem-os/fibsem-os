"""What the Add Task dialog needs to know about a task, without a Qt import.

The dialog that adds a task to a protocol has to answer three questions before
the user commits to anything: which task types are actually on offer, where
each one came from, and what to call the new one. None of those are drawing
problems, so none of them live in the widget.

Deliberately not here: anything that has to build a config to find out, such as
the default milling stages or an estimated duration. Those come from
``__post_init__``, so reading them means constructing and discarding an evented
dataclass per row -- and a duration derived from defaults describes milling
parameters the user has not chosen yet, which is worse than showing nothing. If
the detail pane wants that, it should be declared on the config, not derived.

Keeping them here is not tidiness. This repository's CI installs fibsem without
the ``[ui]`` extra, so nothing that imports PyQt5 is covered by a test run --
logic left inside a dialog is logic nobody checks. Everything below is import-
clean of Qt and tested directly.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

from fibsem.applications.autolamella.workflows.tasks import (
    TASK_ENTRY_POINT_GROUP,
    get_tasks,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.workflows.tasks.tasks import AutoLamellaTask
    from fibsem.plugins.report import Extension


def task_info(task_cls) -> "TaskInfo":
    """The task type's picker metadata, defaulted if it declares none."""
    from fibsem.applications.autolamella.structures import TaskInfo

    return getattr(task_cls.config_cls, "info", None) or TaskInfo()


def available_task_types() -> "Dict[str, Type[AutoLamellaTask]]":
    """The task types worth offering a user, as ``{task_type: class}``.

    Two things are filtered out, for the same underlying reason -- offering them
    would put a row in the picker that cannot do what its label implies.

    **Aliases.** ``get_tasks()`` is the run-time registry, so it also carries
    backwards-compatibility keys kept alive only so an old protocol still loads.
    ``"SETUP_LAMELLA"`` is one, pointing at the same class as ``"MILL_FIDUCIAL"``.
    Choosing it produces a ``MILL_FIDUCIAL`` config regardless, because
    ``task_type`` is a ClassVar on the config rather than something the registry
    key decides. An alias is recognised by shape, not by name: a registry key
    that disagrees with its class's own ``config_cls.task_type`` is another
    spelling of a task type, not a task type. Future aliases drop out too,
    without this module being told about them.

    **Deprecated tasks.** A task that says why it is on its way out is still
    loaded, so protocols holding one keep working -- it is simply not offered
    for new ones. That is the declared way to retire a task; the alias rule
    above is only for registry keys, which have no config to declare anything.
    """
    return {
        task_type: task_cls
        for task_type, task_cls in get_tasks().items()
        if task_type == task_cls.config_cls.task_type
        and not task_info(task_cls).is_deprecated
    }


def unique_task_name(base: str, existing: Dict[str, object]) -> str:
    """``base``, or the first ``base N`` that is free.

    The old dialog defaulted the name to the task's display name and left it
    there, which meant adding a second Trench Milling was guaranteed to land on
    a name collision -- every time, with a warning but no suggestion. Numbering
    from the start makes the common case correct instead of merely diagnosed.

    Counting starts at 2 because the unsuffixed name is the first one: the pair
    reads ``Trench Milling`` / ``Trench Milling 2``, not ``... 1`` / ``... 2``.
    """
    base = base.strip()
    if not base:
        return ""
    if base not in existing:
        return base
    n = 2
    while f"{base} {n}" in existing:
        n += 1
    return f"{base} {n}"


def task_source(task_type: str) -> "Optional[Extension]":
    """Where ``task_type`` came from -- built-in, plugin, or runtime-registered.

    Reuses the collector behind both the Plugins dialog and ``fibsem-cli
    plugins`` rather than re-deriving provenance, so the Add Task dialog cannot
    end up disagreeing with the screen a user checks when a plugin looks
    missing.

    ``None`` when the type is not in the task group at all, which for a type
    that came out of :func:`available_task_types` means the registries and the
    report disagree -- a caller should render nothing rather than guess.
    """
    from fibsem.plugins.report import collect_extensions

    for group in collect_extensions():
        if group.group != TASK_ENTRY_POINT_GROUP:
            continue
        for extension in group.extensions:
            if extension.name == task_type:
                return extension
    return None


# The order headings appear in the picker: purpose-ordered, widest use first,
# and deliberately not alphabetical -- which split the four tasks of a standard
# lamella prep across the whole list. A tag no task declares simply never
# appears, so this can name more than is installed.
TAG_ORDER = (
    "On-grid milling",
    "Waffle milling",
    "Correlation",
    "Imaging",
    # Cross-cutting: nothing is filed under these, so they never appear as a
    # heading. They are still declared here so the filter orders them
    # sensibly, and so a task that does adopt one as its primary lands in a
    # predictable place rather than sorting to the end.
    "Alignment",
    "Fluorescence",
    "Positioning",
    # Last: the escape hatch, behind "Show advanced" anyway.
    "Custom",
)

UNTAGGED = "Other"


def task_tags(task_cls) -> Tuple[str, ...]:
    """Every purpose the task declares, in declaration order.

    The first is its primary -- the heading it files under. The rest still match
    a search, so a task serving two purposes is findable under both without
    being listed twice.
    """
    return tuple(task_info(task_cls).tags)


def tasks_by_tag(
    include_advanced: bool = False,
) -> "OrderedDict[str, Dict[str, Type[AutoLamellaTask]]]":
    """Everything on offer, filed under its primary tag, in display order.

    A task appears once, under its first tag. Listing it under each of its tags
    would duplicate rows, which reads as a registry bug rather than as a task
    with two uses -- the other tags reach the detail pane and the search instead.

    `include_advanced` brings in the primitives, which are hidden by default:
    they are rarely the right answer and they push the common ones off screen.
    """
    by_tag: Dict[str, Dict[str, Type[AutoLamellaTask]]] = {}
    for task_type, task_cls in available_task_types().items():
        info = task_info(task_cls)
        if info.advanced and not include_advanced:
            continue
        primary = info.tags[0] if info.tags else UNTAGGED
        by_tag.setdefault(primary, {})[task_type] = task_cls

    def tag_rank(tag: str):
        if tag in TAG_ORDER:
            return (0, TAG_ORDER.index(tag), "")
        # a tag a plugin invented, then the untagged bucket, last of all
        return (1, 0, tag) if tag != UNTAGGED else (2, 0, tag)

    def task_rank(item):
        task_type, task_cls = item
        info = task_info(task_cls)
        # advanced last within its heading, then declared order, then name --
        # so a tie is stable rather than dependent on registry insertion order.
        return (info.advanced, info.order, task_cls.config_cls.display_name)

    ordered: "OrderedDict[str, Dict[str, Type[AutoLamellaTask]]]" = OrderedDict()
    for tag in sorted(by_tag, key=tag_rank):
        ordered[tag] = OrderedDict(sorted(by_tag[tag].items(), key=task_rank))
    return ordered


def has_advanced_tasks() -> bool:
    """Whether hiding the advanced ones is actually hiding anything."""
    return any(task_info(cls).advanced for cls in available_task_types().values())
