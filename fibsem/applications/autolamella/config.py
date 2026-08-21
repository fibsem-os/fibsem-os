"""AutoLamella's short names for the AUTOLAMELLA_* paths in fibsem.config.

Assignments rather than `from fibsem.config import AUTOLAMELLA_X as X`. The two
are equivalent at runtime, but an aliased import means "deliberate re-export" to
a reader and "unused import" to every tool: ruff rates deleting these as a *safe*
fix, which empties this module and takes `import ...autolamella.structures` down
with it. Its isort rule also splits a multi-alias import into one statement per
name, which is how six constants came to occupy eighteen lines. Assignment says
the same thing unambiguously to both audiences.
"""

from fibsem import config as _cfg

BASE_PATH = _cfg.AUTOLAMELLA_BASE_PATH
CONFIG_PATH = _cfg.AUTOLAMELLA_CONFIG_PATH
EXPERIMENT_NAME = _cfg.AUTOLAMELLA_EXPERIMENT_NAME
LOG_PATH = _cfg.AUTOLAMELLA_LOG_PATH
PROTOCOL_PATH = _cfg.AUTOLAMELLA_PROTOCOL_PATH
TASK_PROTOCOL_PATH = _cfg.AUTOLAMELLA_TASK_PROTOCOL_PATH
