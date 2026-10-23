"""AutoLamella's UI package.

Deliberately exports nothing: the package `__init__` runs before any submodule, so a
re-export of the window here would run the whole application's import for every leaf
widget, and close import loops with `fibsem.ui`.

Import the window from its own module instead:

    from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
"""
