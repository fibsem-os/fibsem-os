"""AutoLamella's paths and the default name of a new experiment."""

import os

from fibsem import config as _cfg

BASE_PATH = os.path.dirname(__file__)
CONFIG_PATH = BASE_PATH
# Where a new experiment goes when the preferences name no folder.
LOG_PATH = _cfg.EXPERIMENTS_PATH
PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol", "legacy", "protocol-on-grid.yaml")
TASK_PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol", "task-protocol.yaml")
# The operator's own protocols; the shipped ones stay beside the package code.
USER_PROTOCOL_PATH = os.path.join(_cfg.STATE_PATH, "autolamella", "protocols")
EXPERIMENT_NAME = "AutoLamella"

os.makedirs(USER_PROTOCOL_PATH, exist_ok=True)
