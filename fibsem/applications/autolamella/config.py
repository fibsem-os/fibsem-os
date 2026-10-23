"""AutoLamella's paths and the default name of a new experiment."""

import os

from fibsem import config as _cfg

BASE_PATH = os.path.dirname(__file__)
CONFIG_PATH = BASE_PATH
LOG_PATH = _cfg.DEFAULT_EXPERIMENTS_PATH
PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol", "legacy", "protocol-on-grid.yaml")
TASK_PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol", "task-protocol.yaml")
EXPERIMENT_NAME = "AutoLamella"
