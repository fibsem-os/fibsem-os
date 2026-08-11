"""AutoLamella's UI package.

Deliberately exports nothing. `AutoLamellaUI` used to be re-exported here, which meant
importing *any* leaf widget in this package ran the whole application's import first --
the package `__init__` runs before the submodule. That closed import loops with
`fibsem.ui`, and the `except ImportError` wrapped around it bound `AutoLamellaUI` to
None, so the failure surfaced as a dead application rather than a traceback.

Import it from its own module instead:

    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
"""
