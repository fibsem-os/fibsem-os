"""The fluorescence (FM) session state: the working configuration and recent channels.

Both live in the instrument's session state (`fibsem.session_state`) under `fm:`:

- ``fm.working`` -- the live FM configuration (channels, z-stack, camera, autofocus).
  Autosaved as it changes and applied at startup, so FM settings survive restarts.
  It is session state because it has no other home: there is no FM block among the
  microscope configuration's defaults.
- ``fm.recent_channels`` -- recently used channel settings, recorded whenever an
  acquisition starts and offered as quick-select entries when adding a channel.

Each function takes the `SessionState` to use, so every caller says which
instrument it means and whether it may write: the application passes a writable
store, a script a read-only one. Each sub-key is imported from the file it used to
live in (``fm-configuration.yaml``, ``fm-recent-channels.yaml``) the first time it
is absent; those files are never deleted.
"""

import logging
import os
from typing import List, Optional, Union

import yaml

from fibsem import config as cfg
from fibsem.fm.structures import ChannelSettings, FluorescenceConfiguration
from fibsem.session_state import SessionState, session_state_for

MAX_RECENT_CHANNELS = 10

WORKING = "fm.working"
RECENT_CHANNELS = "fm.recent_channels"


def fm_session_state(fm, writable: bool = False) -> SessionState:
    """The session state for the instrument an FM belongs to (its ``parent``)."""
    return session_state_for(getattr(fm, "parent", None), writable=writable)


# ---- the working configuration ----------------------------------------------


def _import_working() -> Optional[dict]:
    """The working configuration from the file it lived in before, if any."""
    path = cfg.FM_CONFIGURATION_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    # Parsed once here so a file that cannot become a configuration is not
    # imported as one.
    return FluorescenceConfiguration.from_dict(data).to_dict() if data else None


def save_fm_configuration(
    config: FluorescenceConfiguration, state: SessionState
) -> bool:
    """Save the working FM configuration. Returns whether anything was written."""
    return state.save_section(WORKING, config.to_dict())


def load_fm_configuration(state: SessionState) -> Optional[FluorescenceConfiguration]:
    """The working FM configuration, or None if there is none."""
    data = state.load_section(WORKING, migrate=_import_working)
    if not data:
        return None
    try:
        return FluorescenceConfiguration.from_dict(data)
    except Exception as e:
        logging.warning(f"Failed to load FM working state: {e}")
        return None


# ---- recent channels ---------------------------------------------------------


def _recent_channel_key(channel: ChannelSettings) -> str:
    """Dedup key: entries matching on it are the same logical channel.

    Keyed on name alone — the quick-select menu identifies entries by name, so
    re-using a name overwrites its stored settings rather than accumulating
    visually-identical duplicates that differ only in the tooltip.
    """
    return channel.name


def _import_recent_channels() -> Optional[list]:
    """The recent channels from the file they lived in before, if any."""
    path = cfg.FM_RECENT_CHANNELS_PATH
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else None


def load_recent_channels(state: SessionState) -> List[ChannelSettings]:
    """The recently used channel settings, most recent first."""
    data = state.load_section(RECENT_CHANNELS, migrate=_import_recent_channels)
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
    state: SessionState,
) -> None:
    """Record channel settings as recently used (deduped, most recent first)."""
    if isinstance(channels, ChannelSettings):
        channels = [channels]
    try:
        recents = load_recent_channels(state)
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
        state.save_section(RECENT_CHANNELS, updated)
    except Exception as e:
        logging.warning(f"Failed to record recent FM channels: {e}")


def remove_recent_channel(channel: ChannelSettings, state: SessionState) -> None:
    """Remove a channel from the recently used list by its dedup key."""
    try:
        recents = load_recent_channels(state)
        key = _recent_channel_key(channel)
        remaining = [ch for ch in recents if _recent_channel_key(ch) != key]
        if len(remaining) == len(recents):
            return
        state.save_section(RECENT_CHANNELS, [ch.to_dict() for ch in remaining])
    except Exception as e:
        logging.warning(f"Failed to remove recent FM channel: {e}")
