import os
from pathlib import Path

from fibsem.config import DEFAULT_CHECKPOINT

import autolamella

BASE_PATH: Path = os.path.dirname(__file__)
LOG_PATH: Path = os.path.join(BASE_PATH, 'log')
CONFIG_PATH: Path = os.path.join(BASE_PATH)
PROTOCOL_PATH: Path = os.path.join(BASE_PATH, "protocol", "protocol-on-grid.yaml")
DESKTOP_SHORTCUT_PATH= os.path.dirname(autolamella.__path__[0])

os.makedirs(LOG_PATH, exist_ok=True)

EXPERIMENT_NAME = "AutoLamella"

LIFTOUT_JOIN_METHODS = ["None", "Weld"]
LIFTOUT_LANDING_JOIN_METHODS = ["Weld"]


####### FEATURE FLAGS