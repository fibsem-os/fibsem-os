import os
from pathlib import Path

from fibsem.config import DEFAULT_CHECKPOINT

from fibsem.applications import autolamella

BASE_PATH: Path = os.path.dirname(__file__)
LOG_PATH: Path = os.path.join(BASE_PATH, 'log')
CONFIG_PATH: Path = os.path.join(BASE_PATH)
PROTOCOL_PATH: Path = os.path.join(BASE_PATH, "protocol", "legacy", "protocol-on-grid.yaml")
DESKTOP_SHORTCUT_PATH= os.path.dirname(autolamella.__path__[0])
TASK_PROTOCOL_PATH: Path = os.path.join(BASE_PATH, "protocol", "task-protocol.yaml")

os.makedirs(LOG_PATH, exist_ok=True)

EXPERIMENT_NAME = "AutoLamella"

####### FEATURE FLAGS
FEATURE_MINIMAP_PLOT_WIDGET_ENABLED = True
FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED = False
FEATURE_POSE_CONTROLS_ENABLED = False