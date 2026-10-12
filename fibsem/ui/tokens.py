"""The app's colour tokens, as the Qt UI has always imported them.

They live in ``fibsem.theme.tokens``, which imports without Qt; this module
re-exports every public name so ``from fibsem.ui.tokens import X`` keeps
working. Code outside ``fibsem.ui`` imports ``fibsem.theme.tokens``: importing
anything under ``fibsem.ui`` loads the widgets.
"""

from fibsem.theme.tokens import *  # noqa: F401, F403
