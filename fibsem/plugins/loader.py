"""Load fibsem's entry point groups, and remember what happened.

fibsem exposes three plugin groups -- ``fibsem.patterns``,
``fibsem.strategies`` and ``fibsem.tasks`` -- each loaded by what used to be
three near-identical private functions. They returned ``{name: class}`` and
wrote every failure to a log line, which meant the two questions a user
actually asks could not be answered:

* *"my plugin isn't in the list"* -- a plugin that failed to import leaves no
  trace in the returned dict at all.
* *"why is my plugin being ignored?"* -- a plugin whose name collides with a
  built-in loads fine, registers fine, and is then overwritten when
  ``get_patterns()`` merges built-ins last. Nothing downstream can tell it ever
  existed.

So loading records what it did. ``load_entry_point_group`` returns one
:class:`PluginRecord` per declared entry point -- successes, failures and
shadowed-later-on alike -- and the registries derive their dict from those.
Recording at load time rather than re-walking the entry points separately is
the point: a second walk can disagree with what actually registered, and a
disagreement is precisely the fault this data exists to diagnose.

``fibsem.plugins.report`` turns these records into the plugin listing shown by
``fibsem-cli plugins`` and the Plugins panel.

**This module must not import anything from fibsem.** ``milling/base.py``
imports ``fibsem.milling.patterning``, whose ``__init__`` ends with a
module-level ``MILLING_PATTERNS = get_patterns()`` -- so this code runs while
``fibsem.milling.base`` is only half initialised, and any import reaching back
into it raises. The base class and the name-extraction callable are therefore
passed in as arguments rather than imported here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

__all__ = [
    "PluginRecord",
    "load_entry_point_group",
    "plugin_classes",
    "clear_cache",
]


@dataclass(frozen=True)
class PluginRecord:
    """One declared entry point, and what became of it.

    A record exists whether or not the plugin loaded, so a failure is
    reportable rather than only loggable. ``cls`` and ``name`` are ``None``
    exactly when ``error`` is set.
    """

    group: str
    """The entry point group, e.g. ``"fibsem.patterns"``."""

    entry_point: str
    """The entry point's own name, as written in the plugin's pyproject."""

    value: str
    """The declared target, e.g. ``"lab_patterns.trench:WaffleTrenchPattern"``.

    Kept even when loading failed -- a user needs to see what they *asked* for
    to work out why it is missing.
    """

    distribution: Optional[str] = None
    """The providing distribution's name, if it could be determined."""

    version: Optional[str] = None
    """The providing distribution's version, if it could be determined."""

    cls: Optional[type] = None
    """The loaded class, or ``None`` if it failed."""

    name: Optional[str] = None
    """The name it registered under, or ``None`` if it failed."""

    error: Optional[str] = None
    """Why it failed, phrased for a user, or ``None`` on success."""

    @property
    def loaded(self) -> bool:
        return self.error is None


# One load per group per process, mirroring the @cache the registries used to
# carry. Keyed on the group alone: a group has exactly one base class, and
# loading it twice under different rules would be a bug, not a feature.
_CACHE: Dict[str, Tuple[PluginRecord, ...]] = {}


def _entry_points(group: str) -> Iterator[Any]:
    """Iterate the entry points in ``group``.

    Guard on the version, not on ImportError: ``importlib.metadata`` exists
    from 3.8, so a try/except import succeeds on 3.8/3.9 and then fails at the
    call -- the ``group=`` filter only arrived in 3.10.

    Note that the objects returned are not necessarily
    ``importlib.metadata.EntryPoint``. The ``importlib_metadata`` backport
    installs its own finder on ``sys.meta_path``, so on a Python where the
    backport is also present the stdlib call hands back *backport* entry points
    and distributions. Both expose the attributes used here, but nothing should
    assume the concrete class.
    """
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    return iter(entry_points(group=group))


def _distribution_of(entry_point: Any) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort ``(name, version)`` of the package providing an entry point.

    Which install a plugin came from is the question that matters on a
    microscope PC with several environments, but it is metadata, not the
    plugin: never let reading it break loading.
    """
    try:
        dist = entry_point.dist
        if dist is None:
            return None, None
        return dist.name, dist.version
    except Exception:  # pragma: no cover - defensive, varies by metadata impl
        return None, None


def load_entry_point_group(
    group: str,
    base_cls: type,
    name_of: Callable[[type], str],
    kind: str,
) -> Tuple[PluginRecord, ...]:
    """Load every entry point in ``group``, returning a record for each.

    Args:
        group: The entry point group, e.g. ``"fibsem.patterns"``.
        base_cls: The class every plugin in this group must subclass.
        name_of: Extracts the name a loaded class registers under. Patterns and
            strategies use ``cls.name``; tasks use ``cls.config_cls.task_type``.
        kind: Singular noun for log messages, e.g. ``"pattern"``.

    Returns:
        One record per declared entry point, in discovery order. Never raises:
        a plugin that fails to load is reported, not propagated, because a
        third-party package must not be able to stop fibsem from starting.
    """
    cached = _CACHE.get(group)
    if cached is not None:
        return cached

    records = []
    for entry_point in _entry_points(group):
        distribution, version = _distribution_of(entry_point)
        record = PluginRecord(
            group=group,
            entry_point=entry_point.name,
            value=entry_point.value,
            distribution=distribution,
            version=version,
        )

        try:
            obj = entry_point.load()
        except Exception as exc:
            logging.error(
                "Unexpected error raised while attempting to import %s from '%s'",
                kind,
                entry_point.value,
                exc_info=True,
            )
            records.append(replace(record, error=f"{type(exc).__name__}: {exc}"))
            continue

        # issubclass() raises rather than returning False when handed a
        # non-class, which an entry point pointing at a function or a constant
        # does. Check first, so the reason names the real problem.
        if not isinstance(obj, type):
            reason = f"{type(obj).__name__} is not a class"
            logging.warning("Invalid %s plugin found: '%s' %s", kind, entry_point.value, reason)
            records.append(replace(record, error=reason))
            continue

        if not issubclass(obj, base_cls):
            reason = f"not a subclass of {base_cls.__name__}"
            logging.warning("Invalid %s plugin found: '%s' is %s", kind, entry_point.value, reason)
            records.append(replace(record, error=reason))
            continue

        try:
            name = name_of(obj)
        except Exception as exc:
            logging.warning(
                "Invalid %s plugin found: could not read the name of '%s': %s",
                kind,
                entry_point.value,
                exc,
            )
            records.append(replace(record, error=f"could not read its name: {exc}"))
            continue

        logging.info("Loaded %s plugin '%s'", kind, name)
        records.append(replace(record, cls=obj, name=name))

    loaded = tuple(records)
    _CACHE[group] = loaded
    return loaded


def plugin_classes(records: Tuple[PluginRecord, ...]) -> Dict[str, Type[Any]]:
    """The ``{name: class}`` mapping the registries merge.

    Built in record order, so when two entry points claim the same name the
    last one wins -- the behaviour these registries have always had. The
    displaced record is still in ``records``, which is how the listing can
    report a collision the mapping cannot express.
    """
    return {r.name: r.cls for r in records if r.loaded}  # type: ignore[misc]


def clear_cache() -> None:
    """Forget every loaded group, so the next call re-walks the entry points.

    For tests. Calling this in a running application will make the registries
    disagree with the module-level snapshots taken at import time
    (``MILLING_PATTERNS``, ``TASK_REGISTRY``), which is worse than a stale list.
    """
    _CACHE.clear()
