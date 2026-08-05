"""Persistence for the fluorescence (FM) configuration.

A single ``fm-configuration.yaml`` holds the live FM config. It is auto-saved on
change + on close and auto-loaded + applied on startup, so FM settings survive
restarts.

``fm-recent-channels.yaml`` holds the recently-used channel settings, recorded
whenever the user starts an acquisition, and offered as quick-select entries
when adding a channel.
"""

import logging
import os
from typing import List, Optional, Union

import yaml

from fibsem import config as cfg
from fibsem.fm.structures import ChannelSettings, FluorescenceConfiguration

MAX_RECENT_CHANNELS = 10


def save_fm_configuration(config: FluorescenceConfiguration) -> str:
    """Persist the current FM configuration to the working-state file."""
    os.makedirs(cfg.CONFIG_PATH, exist_ok=True)
    return config.export(cfg.FM_CONFIGURATION_PATH)


def load_fm_configuration() -> Optional[FluorescenceConfiguration]:
    """Load the working-state FM configuration, or None if it doesn't exist."""
    if not os.path.exists(cfg.FM_CONFIGURATION_PATH):
        return None
    try:
        return FluorescenceConfiguration.load(cfg.FM_CONFIGURATION_PATH)
    except Exception as e:
        logging.warning(f"Failed to load FM working state: {e}")
        return None


def _recent_channel_key(channel: ChannelSettings) -> str:
    """Dedup key: entries matching on it are the same logical channel.

    Keyed on name alone — the quick-select menu identifies entries by name, so
    re-using a name overwrites its stored settings rather than accumulating
    visually-identical duplicates that differ only in the tooltip.
    """
    return channel.name


def load_recent_channels() -> List[ChannelSettings]:
    """Load the recently-used channel settings, most recent first."""
    if not os.path.exists(cfg.FM_RECENT_CHANNELS_PATH):
        return []
    try:
        with open(cfg.FM_RECENT_CHANNELS_PATH, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Failed to load recent FM channels: {e}")
        return []
    if not isinstance(data, list):
        return []
    channels = []
    for entry in data:
        try:
            channels.append(ChannelSettings.from_dict(entry))
        except Exception as e:
            logging.warning(f"Skipping malformed recent FM channel entry {entry}: {e}")
    return channels


def record_recent_channels(
    channels: Union[ChannelSettings, List[ChannelSettings]],
) -> None:
    """Record channel settings as recently used (deduped, most recent first)."""
    if isinstance(channels, ChannelSettings):
        channels = [channels]
    try:
        recents = load_recent_channels()
        previous = [ch.to_dict() for ch in recents]
        # later entries in `channels` end up further down the list, matching
        # the on-screen channel order
        for channel in reversed(channels):
            key = _recent_channel_key(channel)
            recents = [ch for ch in recents if _recent_channel_key(ch) != key]
            recents.insert(0, channel)
        recents = recents[:MAX_RECENT_CHANNELS]
        updated = [ch.to_dict() for ch in recents]
        if updated == previous:
            return
        os.makedirs(cfg.CONFIG_PATH, exist_ok=True)
        with open(cfg.FM_RECENT_CHANNELS_PATH, "w") as f:
            yaml.safe_dump(updated, f, sort_keys=False)
    except Exception as e:
        logging.warning(f"Failed to record recent FM channels: {e}")


def remove_recent_channel(channel: ChannelSettings) -> None:
    """Remove a channel from the recently-used list by its dedup key."""
    try:
        recents = load_recent_channels()
        key = _recent_channel_key(channel)
        remaining = [ch for ch in recents if _recent_channel_key(ch) != key]
        if len(remaining) == len(recents):
            return
        os.makedirs(cfg.CONFIG_PATH, exist_ok=True)
        with open(cfg.FM_RECENT_CHANNELS_PATH, "w") as f:
            yaml.safe_dump([ch.to_dict() for ch in remaining], f, sort_keys=False)
    except Exception as e:
        logging.warning(f"Failed to remove recent FM channel: {e}")
