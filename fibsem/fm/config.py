"""Persistence for fluorescence (FM) configurations.

Two layers (see docs/design/fm-settings-persistence.md):

- **Working state** — a single ``fm-configuration.yaml`` that holds the live FM
  config. Auto-saved on close, auto-loaded + applied on startup. This is what
  makes the camera transform survive restarts without touching named presets.

- **Preset library** — a registry of named ``FluorescenceConfiguration`` files
  (per sample: "GFP-sample", "mCherry-sample", ...), mirroring the microscope
  configuration registry in ``fibsem.config``. Presets are explicit: the user
  saves-as / loads / deletes them.
"""

import logging
import os
import re
from typing import Dict, List, Optional

import yaml

from fibsem import config as cfg
from fibsem.fm.structures import FluorescenceConfiguration

# ----------------------------------------------------------------------------
# Working state (single file, auto-managed)
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Named preset library (registry, explicit save/load)
# ----------------------------------------------------------------------------


def _empty_registry() -> Dict:
    return {"configurations": {}, "default": None}


def _load_registry() -> Dict:
    if not os.path.exists(cfg.FM_CONFIGURATIONS_PATH):
        return _empty_registry()
    try:
        with open(cfg.FM_CONFIGURATIONS_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logging.warning(f"Failed to load FM preset registry: {e}")
        return _empty_registry()
    # tolerate a partially-formed file
    data.setdefault("configurations", {})
    data.setdefault("default", None)
    return data


def _save_registry(registry: Dict) -> None:
    os.makedirs(cfg.CONFIG_PATH, exist_ok=True)
    with open(cfg.FM_CONFIGURATIONS_PATH, "w") as f:
        yaml.safe_dump(registry, f, indent=4, sort_keys=False)


def _slugify(name: str) -> str:
    """Filesystem-safe filename stem for a preset name."""
    slug = re.sub(r"[^\w.-]+", "-", name.strip()).strip("-")
    return slug or "preset"


def list_fm_presets() -> List[str]:
    """Return the names of all saved FM presets."""
    return list(_load_registry()["configurations"].keys())


def get_fm_preset_path(name: str) -> Optional[str]:
    """Return the file path for a named preset, or None if unknown."""
    entry = _load_registry()["configurations"].get(name)
    return entry.get("path") if entry else None


def save_fm_preset(
    name: str, config: FluorescenceConfiguration, set_default: bool = False
) -> str:
    """Save (or overwrite) a named preset and register it. Returns the file path."""
    if not name or not name.strip():
        raise ValueError("Preset name must not be empty.")

    os.makedirs(cfg.FM_CONFIGURATIONS_DIR, exist_ok=True)
    registry = _load_registry()

    # reuse the existing path when overwriting, else derive a fresh one
    existing = registry["configurations"].get(name)
    if existing and existing.get("path"):
        path = existing["path"]
    else:
        path = os.path.join(cfg.FM_CONFIGURATIONS_DIR, f"{_slugify(name)}.yaml")

    config.export(path)

    registry["configurations"][name] = {"path": path}
    if set_default or registry["default"] is None:
        registry["default"] = name
    _save_registry(registry)
    return path


def load_fm_preset(name: str) -> FluorescenceConfiguration:
    """Load a named preset. Raises if the preset is unknown or unreadable."""
    path = get_fm_preset_path(name)
    if path is None:
        raise ValueError(f"FM preset '{name}' does not exist.")
    return FluorescenceConfiguration.load(path)


def remove_fm_preset(name: str, delete_file: bool = True) -> None:
    """Remove a preset from the registry (and optionally its file)."""
    registry = _load_registry()
    entry = registry["configurations"].pop(name, None)
    if entry is None:
        raise ValueError(f"FM preset '{name}' does not exist.")
    if registry.get("default") == name:
        registry["default"] = next(iter(registry["configurations"]), None)
    _save_registry(registry)

    if delete_file and entry.get("path") and os.path.exists(entry["path"]):
        try:
            os.remove(entry["path"])
        except OSError as e:
            logging.warning(f"Could not delete FM preset file {entry['path']}: {e}")


def set_default_fm_preset(name: str) -> None:
    """Mark a preset as the default."""
    registry = _load_registry()
    if name not in registry["configurations"]:
        raise ValueError(f"FM preset '{name}' does not exist.")
    registry["default"] = name
    _save_registry(registry)


def get_default_fm_preset_name() -> Optional[str]:
    """Return the default preset name, or None if the library is empty."""
    return _load_registry().get("default")


def get_default_fm_preset() -> Optional[FluorescenceConfiguration]:
    """Load the default preset, or None if there isn't one."""
    name = get_default_fm_preset_name()
    if name is None:
        return None
    try:
        return load_fm_preset(name)
    except Exception as e:
        logging.warning(f"Failed to load default FM preset '{name}': {e}")
        return None
