"""The shared entry point loader.

Covers the paths an installed fixture cannot reach without a package built to
order for each one: an entry point pointing at something that is not a class,
a class whose name cannot be read, and the load-once caching the registries
depend on.

The counterpart against a real installed plugin is in
tests/test_plugin_entry_points.py -- these fakes prove the branching, that one
proves it is wired to real entry point metadata.
"""

import pytest

from fibsem.plugins import loader
from fibsem.plugins.loader import load_entry_point_group, plugin_classes


class Base:
    name = "base"


class Good(Base):
    name = "Good"


class Unrelated:
    name = "Unrelated"


class NoConfig(Base):
    """Shaped like a task class that forgot its ``config_cls``.

    The real name extraction for tasks is ``cls.config_cls.task_type``, two
    attribute hops that a plugin author can get wrong while still subclassing
    the right base -- so reading the name is its own failure mode.
    """


class FakeEntryPoint:
    """Stands in for an importlib EntryPoint.

    Deliberately duck-typed rather than a real one: the loader must not depend
    on the concrete class, since importlib_metadata's finder makes even a stdlib
    ``entry_points()`` call hand back backport objects.
    """

    def __init__(self, name, value, target, dist="lab-thing", version="1.2.3"):
        self.name = name
        self.value = value
        self._target = target
        self.dist = type("Dist", (), {"name": dist, "version": version})()

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


@pytest.fixture
def fake_group(monkeypatch):
    """Install a set of entry points under a group, with a clean cache."""

    def install(*entry_points, group="test.group", name_of=lambda cls: cls.name):
        loader.clear_cache()
        monkeypatch.setattr(loader, "_entry_points", lambda g: iter(entry_points))
        return load_entry_point_group(
            group=group, base_cls=Base, name_of=name_of, kind="thing"
        )

    yield install
    loader.clear_cache()


def test_a_loaded_plugin_records_its_class_name_and_provenance(fake_group):
    (record,) = fake_group(FakeEntryPoint("good", "pkg.mod:Good", Good))

    assert record.loaded
    assert record.cls is Good
    assert record.name == "Good"
    assert (record.distribution, record.version) == ("lab-thing", "1.2.3")
    assert record.error is None
    assert plugin_classes((record,)) == {"Good": Good}


def test_an_entry_point_that_is_not_a_class_names_the_real_problem(fake_group):
    """issubclass() raises rather than returning False when handed a non-class.

    Left to the generic handler that surfaces as "issubclass() arg 1 must be a
    class", which describes fibsem's internals rather than the user's mistake.
    """
    (record,) = fake_group(FakeEntryPoint("fn", "pkg.mod:helper", lambda: None))

    assert not record.loaded
    assert "is not a class" in record.error
    assert record.cls is None


def test_a_class_with_the_wrong_base_is_rejected_by_name(fake_group):
    (record,) = fake_group(FakeEntryPoint("nope", "pkg.mod:Unrelated", Unrelated))

    assert record.error == "not a subclass of Base"
    assert record.cls is None
    # It loaded fine; only the type check rejected it. The declared target has
    # to survive so the listing can show what was asked for.
    assert record.value == "pkg.mod:Unrelated"


def test_an_import_failure_keeps_the_exception_type(fake_group):
    boom = ModuleNotFoundError("No module named 'pkg.nope'")
    (record,) = fake_group(FakeEntryPoint("boom", "pkg.nope:Thing", boom))

    assert record.error == "ModuleNotFoundError: No module named 'pkg.nope'"
    assert record.name is None


def test_a_class_whose_name_cannot_be_read_is_a_failure_not_a_crash(fake_group):
    """A third-party package must not be able to stop fibsem from starting."""
    (record,) = fake_group(
        FakeEntryPoint("odd", "pkg.mod:NoConfig", NoConfig),
        name_of=lambda cls: cls.config_cls.task_type,
    )

    assert not record.loaded
    assert "could not read its name" in record.error
    assert "config_cls" in record.error


def test_one_broken_plugin_does_not_take_out_the_others(fake_group):
    records = fake_group(
        FakeEntryPoint("boom", "pkg.nope:Thing", ImportError("nope")),
        FakeEntryPoint("good", "pkg.mod:Good", Good),
    )

    assert [r.loaded for r in records] == [False, True]
    assert plugin_classes(records) == {"Good": Good}


def test_the_last_entry_point_wins_a_name_clash_and_the_loser_is_kept(fake_group):
    """plugin_classes() must match the merge the registries have always done.

    The displaced record stays in the list, which is the only reason the listing
    can report a collision that the mapping cannot express.
    """

    class AlsoGood(Base):
        name = "Good"

    records = fake_group(
        FakeEntryPoint("a", "pkg.mod:Good", Good),
        FakeEntryPoint("b", "pkg.mod:AlsoGood", AlsoGood),
    )

    assert plugin_classes(records) == {"Good": AlsoGood}
    assert len(records) == 2


def test_a_group_loads_once_per_process(fake_group, monkeypatch):
    """The registries rely on this: loading is not free and must not repeat."""
    calls = []

    class Counting(FakeEntryPoint):
        def load(self):
            calls.append(self.name)
            return super().load()

    fake_group(Counting("good", "pkg.mod:Good", Good))
    assert calls == ["good"]

    # A second call must not touch the entry points again, even though the
    # monkeypatched iterator would happily yield them.
    load_entry_point_group(
        group="test.group", base_cls=Base, name_of=lambda cls: cls.name, kind="thing"
    )
    assert calls == ["good"]

    loader.clear_cache()
    load_entry_point_group(
        group="test.group", base_cls=Base, name_of=lambda cls: cls.name, kind="thing"
    )
    assert calls == ["good", "good"]


def test_missing_distribution_metadata_does_not_break_loading(fake_group):
    """Provenance is metadata about the plugin, not the plugin."""
    entry_point = FakeEntryPoint("good", "pkg.mod:Good", Good)
    entry_point.dist = None

    (record,) = fake_group(entry_point)
    assert record.loaded and record.cls is Good
    assert record.distribution is None and record.version is None
