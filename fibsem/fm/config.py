"""Persistence for the fluorescence (FM) working state.

A single ``fm-configuration.yaml`` holds the live FM config (channels / camera
transform / objective). It is auto-saved on change + on close and auto-loaded +
applied on startup, so FM settings — notably the camera transform — survive
restarts. See docs/design/fm-settings-persistence.md.
"""

import logging
import os
from typing import Optional

from fibsem import config as cfg
from fibsem.fm.structures import FluorescenceConfiguration


def save_fm_working_state(config: FluorescenceConfiguration) -> str:
    """Persist the current FM configuration to the working-state file."""
    os.makedirs(cfg.CONFIG_PATH, exist_ok=True)
    return config.export(cfg.FM_CONFIGURATION_PATH)


def load_fm_working_state() -> Optional[FluorescenceConfiguration]:
    """Load the working-state FM configuration, or None if it doesn't exist."""
    if not os.path.exists(cfg.FM_CONFIGURATION_PATH):
        return None
    try:
        return FluorescenceConfiguration.load(cfg.FM_CONFIGURATION_PATH)
    except Exception as e:
        logging.warning(f"Failed to load FM working state: {e}")
        return None
