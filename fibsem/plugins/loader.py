"""Load fibsem's entry point groups, and remember what happened.

fibsem's class plugin groups -- ``fibsem.patterns``,
``fibsem.strategies`` and ``fibsem.tasks`` -- were each loaded by what used to be
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

A group whose plugins are not classes passes ``resolve`` instead: it turns what
the entry point loaded into the name and object to register, and raises
:class:`PluginRejected` to refuse it. ``fibsem.drivers`` does, because its
entry points are functions returning a ``DriverEntry``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

__all__ = [
    "PluginRecord",
    "PluginRegistry",
    "PluginRejected",
    "load_entry_point_group",
    "subclass_of",
    "plugin_classes",
    "clear_cache",
]


@dataclass(frozen=True)
class PluginRecord:
    """One declared entry point, and what became of it.

    A record exists whether or not the plugin loaded, so a failure is
    reportable rather than only loggable. ``obj`` and ``name`` are ``None``
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

    obj: Any = None
    """What it registered (a class, or a driver's ``DriverEntry``), or ``None``
    if it failed."""

    name: Optional[str] = None
    """The name it registered under, or ``None`` if it failed."""

    error: Optional[str] = None
    """Why it failed, phrased for a user, or ``None`` on success."""

    @property
    def loaded(self) -> bool:
        return self.error is None

    @property
    def cls(self) -> Any:
        """``obj``, under the name the class groups read it by."""
        return self.obj


class PluginRejected(Exception):
    """Raised by a ``resolve`` callable to refuse a plugin; the message says why."""


# One load per group per process, mirroring the @cache the registries used to
# carry. Keyed on the group alone: a group has exactly one base class, and
# loading it twice under different rules would be a bug, not a feature.
_CACHE: Dict[str, Tuple[PluginRecord, ...]] = {}


def _entry_points(group: str) -> Iterator[Any]:
    """Iterate the entry points in ``group``.

    Note that the objects returned are not necessarily
    ``importlib.metadata.EntryPoint``. The ``importlib_metadata`` backport
    installs its own finder on ``sys.meta_path``, so on a Python where the
    backport is also present the stdlib call hands back *backport* entry points
    and distributions. Both expose the attributes used here, but nothing should
    assume the concrete class.
    """
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


Resolver = Callable[[Any], Tuple[str, Any]]
"""Turns what an entry point loaded into ``(name, object to register)``."""


def subclass_of(base_cls: type, name_of: Callable[[type], str]) -> Resolver:
    """The resolver for a group of classes: each must subclass ``base_cls``, and
    registers under ``name_of(cls)``."""

    def resolve(obj: Any) -> Tuple[str, Any]:
        # issubclass() raises rather than returning False when handed a
        # non-class, which an entry point pointing at a function or a constant
        # does. Check first, so the reason names the real problem.
        if not isinstance(obj, type):
            raise PluginRejected(f"{type(obj).__name__} is not a class")
        if not issubclass(obj, base_cls):
            raise PluginRejected(f"not a subclass of {base_cls.__name__}")
        try:
            name = name_of(obj)
        except Exception as exc:
            raise PluginRejected(f"could not read its name: {exc}") from exc
        return name, obj

    return resolve


def load_entry_point_group(
    group: str,
    base_cls: Optional[type] = None,
    name_of: Optional[Callable[[type], str]] = None,
    kind: str = "plugin",
    resolve: Optional[Resolver] = None,
) -> Tuple[PluginRecord, ...]:
    """Load every entry point in ``group``, returning a record for each.

    Args:
        group: The entry point group, e.g. ``"fibsem.patterns"``.
        base_cls: The class every plugin in this group must subclass.
        name_of: Extracts the name a loaded class registers under. Patterns and
            strategies use ``cls.name``; tasks use ``cls.config_cls.task_type``.
        kind: Singular noun for log messages, e.g. ``"pattern"``.
        resolve: In place of ``base_cls`` and ``name_of``, for a group whose
            plugins are not classes: see :data:`Resolver`.

    Returns:
        One record per declared entry point, in discovery order. Never raises:
        a plugin that fails to load is reported, not propagated, because a
        third-party package must not be able to stop fibsem from starting.
    """
    cached = _CACHE.get(group)
    if cached is not None:
        return cached
    if resolve is None:
        if base_cls is None or name_of is None:
            raise TypeError("pass either resolve, or base_cls and name_of")
        resolve = subclass_of(base_cls, name_of)

    # Empty while loading, so a plugin that asks for its own group as it loads
    # gets nothing rather than starting a second walk.
    _CACHE[group] = ()
    try:
        loaded = tuple(_load(group, kind, resolve))
    except BaseException:
        del _CACHE[group]
        raise
    _CACHE[group] = loaded
    return loaded


def _load(group: str, kind: str, resolve: Resolver) -> List[PluginRecord]:
    records: List[PluginRecord] = []
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

        try:
            name, registered = resolve(obj)
        except PluginRejected as exc:
            logging.warning(
                "Invalid %s plugin found: '%s': %s", kind, entry_point.value, exc
            )
            records.append(replace(record, error=str(exc)))
            continue
        except Exception as exc:
            logging.error(
                "Unexpected error raised while loading %s from '%s'",
                kind,
                entry_point.value,
                exc_info=True,
            )
            records.append(replace(record, error=f"{type(exc).__name__}: {exc}"))
            continue

        logging.info("Loaded %s plugin '%s'", kind, name)
        records.append(replace(record, obj=registered, name=name))

    return records


def plugin_classes(records: Tuple[PluginRecord, ...]) -> Dict[str, Type[Any]]:
    """The ``{name: class}`` mapping the registries merge (``{name: object}``
    for a group with its own ``resolve``).

    Built in record order, so when two entry points claim the same name the
    last one wins -- the behaviour these registries have always had. The
    displaced record is still in ``records``, which is how the listing can
    report a collision the mapping cannot express.
    """
    return {r.name: r.obj for r in records if r.loaded}  # type: ignore[misc]


def clear_cache() -> None:
    """Forget every loaded group, so the next call re-walks the entry points.

    For tests. Calling this in a running application will make the registries
    disagree with the module-level snapshots taken at import time
    (``MILLING_PATTERNS``, ``TASK_REGISTRY``), which is worse than a stale list.
    """
    _CACHE.clear()


T = TypeVar("T")


def qualified_name(obj: Any) -> str:
    """``module.QualName``, how a listing names a registered class."""
    return f"{obj.__module__}.{obj.__qualname__}"


class PluginRegistry(Generic[T]):
    """One plugin group: its built-ins, what is registered at runtime, and its
    entry points.

    When two claim one name, the built-in wins, then the runtime registration,
    then the plugin (the later plugin, between two). ``fibsem.plugins.report``
    lists any registry from what it holds.
    """

    def __init__(
        self,
        group: str,
        kind: str,
        resolve: Resolver,
        builtins: Optional[Mapping[str, T]] = None,
        describe: Callable[[T], str] = qualified_name,
    ) -> None:
        self.group = group
        """The entry point group, e.g. ``"fibsem.patterns"``."""
        self.kind = kind
        """Singular noun for log messages, e.g. ``"pattern"``."""
        self.resolve = resolve
        self.builtins: Mapping[str, T] = {} if builtins is None else builtins
        self.registered: Dict[str, T] = {}
        self.describe = describe
        """What a listing shows for a registered object."""

    def register(self, name: str, obj: T) -> None:
        """Register *obj* under *name* at runtime, replacing an earlier one."""
        self.registered[name] = obj
        logging.info("Registered %s '%s'", self.kind, name)

    def plugin_records(self) -> Tuple[PluginRecord, ...]:
        """Every entry point in the group and what became of it, read once.

        Includes the plugins that failed and the ones a built-in or a runtime
        registration shadows, neither of which :meth:`all` returns.
        """
        return load_entry_point_group(self.group, kind=self.kind, resolve=self.resolve)

    def plugins(self) -> Dict[str, T]:
        """The plugins that loaded, by name."""
        return plugin_classes(self.plugin_records())

    def all(self) -> Dict[str, T]:
        """Everything registered, by name, with clashes settled."""
        return {**self.plugins(), **self.registered, **self.builtins}

    def get(self, name: str) -> Optional[T]:
        """What *name* is registered as, or ``None``. Looks at the plugins only
        when neither a built-in nor a runtime registration has the name."""
        if name in self.builtins:
            return self.builtins[name]
        if name in self.registered:
            return self.registered[name]
        return self.plugins().get(name)

    def describe_builtin(self, name: str) -> str:
        """What a listing shows for the built-in *name*."""
        return self.describe(self.builtins[name])
