"""The plugin entry point contract.

fibsem exposes three extension points -- ``fibsem.patterns``,
``fibsem.strategies`` and ``fibsem.tasks`` -- and for patterns and strategies a
packaged plugin is the *only* way in. Nothing in the test suite exercised that
path, so a refactor could break every third-party plugin in existence without a
single test failing; the breakage would surface months later on a user's
microscope, as a class that silently fails to appear.

These tests close that gap. ``tests/fixtures/plugin`` is a package that
declares all three groups the way a real plugin does; CI installs it before
running the suite, and the tests assert it resolves through fibsem's own
registries -- not merely that it imports.

The fixture is not installed by default, so a bare local ``pytest`` skips these
rather than failing. To run them:

    pip install --no-deps tests/fixtures/plugin

In CI the workflow sets ``FIBSEM_REQUIRE_PLUGIN_FIXTURE=1``, which turns that
skip into a hard failure. Without it these tests could be disabled by deleting
the install step and CI would stay green -- which is precisely the shape of
silent breakage this file exists to catch, so it must not be possible here.

The counterpart to this file lives in the fibsem-plugin-example repository,
which checks the same contract from the outside, against fibsem main.
"""

import os

import pytest

PATTERN_NAME = "Fixture Pattern"
STRATEGY_NAME = "Fixture Strategy"
TASK_TYPE = "FIXTURE_TASK"
CLASHING_PATTERN_NAME = "Rectangle"  # a built-in name the fixture claims on purpose

def _fixture_installed() -> bool:
    """Detect the fixture without importing it.

    Deliberately not ``pytest.importorskip``: importing the plugin package
    before fibsem's registries are built poisons them. The registry functions
    are ``@cache``d, so the empty result is locked in for the rest of the
    process and every assertion below fails for the wrong reason. Asking the
    entry point metadata is both safer and a more direct test of the contract.

    Guard on the version, not on ImportError: ``importlib.metadata`` exists
    from 3.8, so a try/except import succeeds on 3.8/3.9 and then fails at the
    call with ``entry_points() got an unexpected keyword argument 'group'`` --
    the ``group=`` filter only arrived in 3.10. This is the same check the
    plugin loaders themselves use.
    """
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    return any(ep.name == "fixture_pattern" for ep in entry_points(group="fibsem.patterns"))


_INSTALLED = _fixture_installed()

if not _INSTALLED and os.environ.get("FIBSEM_REQUIRE_PLUGIN_FIXTURE") == "1":
    raise RuntimeError(
        "FIBSEM_REQUIRE_PLUGIN_FIXTURE=1 but the plugin fixture is not installed. "
        "The CI step that runs `pip install --no-deps tests/fixtures/plugin` is "
        "missing or failed. Skipping here would leave the entry point contract "
        "untested with a green build."
    )

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _INSTALLED,
        reason="plugin fixture not installed: pip install --no-deps tests/fixtures/plugin",
    ),
]


def test_pattern_plugin_resolves_through_the_registry():
    from fibsem.milling.patterning import get_pattern, get_pattern_names, get_patterns

    assert PATTERN_NAME in get_patterns(), sorted(get_patterns())
    assert PATTERN_NAME in get_pattern_names()

    pattern = get_pattern(PATTERN_NAME, {"width": 20.0e-6})
    assert type(pattern).__name__ == "FixturePattern"
    assert pattern.width == 20.0e-6
    assert len(pattern.define()) == 1


def test_pattern_plugin_round_trips_through_a_protocol_dict():
    """A pattern that cannot survive to_dict/from_dict works in the UI and
    vanishes when the protocol is reloaded."""
    from fibsem.milling.patterning import get_pattern

    pattern = get_pattern(PATTERN_NAME, {"width": 15.0e-6})
    ddict = pattern.to_dict()

    assert ddict["name"] == PATTERN_NAME
    assert get_pattern(ddict["name"], ddict).width == 15.0e-6


def test_strategy_plugin_resolves_through_the_registry():
    from fibsem.milling.base import get_strategy
    from fibsem.milling.strategy import get_strategies

    assert STRATEGY_NAME in get_strategies(), sorted(get_strategies())

    strategy = get_strategy(STRATEGY_NAME, {"config": {"passes": 3}})
    assert type(strategy).__name__ == "FixtureMillingStrategy"
    assert strategy.config.passes == 3
    assert strategy.to_dict()["name"] == STRATEGY_NAME


def test_unselectable_strategy_is_hidden_from_the_dropdown():
    """``selectable = False`` is how a strategy stays out of the UI while
    remaining resolvable by name from an existing protocol."""
    from fibsem.milling.strategy import get_strategies, get_strategy_names

    assert STRATEGY_NAME in get_strategies()
    assert STRATEGY_NAME not in get_strategy_names()


def test_task_plugin_resolves_through_the_registry():
    from fibsem.applications.autolamella.workflows.tasks import (
        get_task_config,
        get_task_names,
        get_tasks,
    )

    assert TASK_TYPE in get_tasks(), sorted(get_tasks())
    assert TASK_TYPE in get_task_names()
    assert get_task_config(TASK_TYPE).__name__ == "FixtureTaskConfig"


def test_task_plugin_survives_protocol_load():
    """load_task_config() skips task types it does not recognise with only a log
    warning: the task disappears from the protocol and re-saving drops it from
    the yaml. A plugin task must not be one of them."""
    from fibsem.applications.autolamella.workflows.tasks import load_task_config

    loaded = load_task_config(
        {"Fixture": {"task_type": TASK_TYPE, "parameters": {"number": 42}}}
    )

    assert "Fixture" in loaded, dict(loaded)
    assert loaded["Fixture"].number == 42
    assert loaded["Fixture"].task_name == "Fixture"


def test_milling_stage_built_from_yaml_uses_both_plugin_classes():
    """The end-to-end shape: a protocol naming a plugin pattern and a plugin
    strategy produces a milling stage."""
    from fibsem.milling.base import FibsemMillingStage

    stage = FibsemMillingStage.from_dict(
        {
            "name": "Fixture Stage",
            "milling": {},
            "pattern": {"name": PATTERN_NAME, "width": 12.0e-6},
            "strategy": {"name": STRATEGY_NAME, "config": {"passes": 2}},
        }
    )

    assert stage.pattern.name == PATTERN_NAME
    assert stage.strategy.name == STRATEGY_NAME

    ddict = stage.to_dict()
    assert ddict["pattern"]["name"] == PATTERN_NAME
    assert ddict["strategy"]["name"] == STRATEGY_NAME


def test_builtins_win_name_clashes():
    """Documented precedence: builtins > runtime-registered > plugins.

    The fixture registers a pattern under a built-in's name on purpose. Without
    a plugin that actually collides this test asserts nothing -- every built-in
    trivially maps to itself when no plugin competes for the name.
    """
    from fibsem.milling.patterning import BUILTIN_PATTERNS, get_patterns

    patterns = get_patterns()

    # the colliding entry point resolved, so the precedence rule is under load
    assert CLASHING_PATTERN_NAME in BUILTIN_PATTERNS
    assert patterns[CLASHING_PATTERN_NAME] is BUILTIN_PATTERNS[CLASHING_PATTERN_NAME]
    assert patterns[CLASHING_PATTERN_NAME].__module__.startswith("fibsem.")

    # and no other built-in was displaced either
    for name, cls in BUILTIN_PATTERNS.items():
        assert patterns[name] is cls


def test_shadowed_plugin_is_not_silently_reported_as_the_builtin():
    """A shadowed plugin should be absent, not masquerading.

    get_pattern() must hand back the built-in class, so a protocol naming
    "Rectangle" keeps meaning the built-in even with the plugin installed.
    """
    from fibsem.milling.patterning import get_pattern
    from fibsem.milling.patterning.patterns2 import RectanglePattern

    assert isinstance(get_pattern(CLASHING_PATTERN_NAME), RectanglePattern)
