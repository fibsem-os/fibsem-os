"""A minimal third-party plugin, used to test the entry point contract.

This is not documentation -- see the fibsem-plugin-example repository for that.
It exists so ``tests/test_plugin_entry_points.py`` can assert that a plugin
installed the way a user installs one actually resolves through every registry.

Deliberately empty. Each entry point points at its own submodule, because the
three groups are loaded at different points in fibsem's own import sequence and
cannot share a module -- see the note in ``patterns.py``.

Class names avoid a ``Test`` prefix so pytest does not try to collect them.
"""

PATTERN_NAME = "Fixture Pattern"
STRATEGY_NAME = "Fixture Strategy"
TASK_TYPE = "FIXTURE_TASK"
