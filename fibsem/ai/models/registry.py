"""The registry: the models this install offers, by name.

An entry names a model and says where it is (*ref*), which version it pins, the
settings it runs with, which tasks it is the default for, and whether the agent
may use it. Entries come from fibsem itself, then from ``models.yaml`` in the
fibsem config folder, which adds models and overrides fibsem's by name::

    models:
      grid-bars:
        ref: D:/models/grid-bars
        version: 0.2.0
        config: {checkpoint: checkpoint_best.pth}
        default_for: [segmentation]

The agent names a registered model, never a path: loading a checkpoint can run
the code inside it.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from fibsem.config import CONFIG_PATH

MODELS_PATH = os.path.join(CONFIG_PATH, "models.yaml")

#: A ref of a model that ships inside fibsem: ``builtin://<backend>/<name>``.
BUILTIN = "builtin://"


@dataclass
class ModelEntry:
    """One registered model. *version* pins it ("" takes what *ref* holds);
    *config* is its settings; *card* is a card for a model that has none of its
    own; *agent* False keeps it from the agent."""

    name: str
    ref: str
    version: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    default_for: List[str] = field(default_factory=list)
    agent: bool = True
    card: Optional[Dict[str, Any]] = None

    @property
    def builtin_backend(self) -> Optional[str]:
        """The backend of a built-in model (``builtin://fake/segmentation`` is
        ``fake``), or None for a model elsewhere."""
        if not self.ref.startswith(BUILTIN):
            return None
        return self.ref[len(BUILTIN) :].split("/", 1)[0]


BUILTIN_ENTRIES: Tuple[ModelEntry, ...] = (
    ModelEntry(
        name="fake-segmentation",
        ref="builtin://fake/segmentation",
        default_for=["segmentation"],
    ),
)


class ModelRegistry:
    """The registered models by name: fibsem's, then the file's."""

    def __init__(self, entries: Iterable[ModelEntry] = BUILTIN_ENTRIES):
        self._entries: Dict[str, ModelEntry] = {}
        for entry in entries:
            self.add(entry)

    @classmethod
    def load(cls, path: str = MODELS_PATH) -> "ModelRegistry":
        """fibsem's models, with the ones in *path* added over them. A missing
        file adds none."""
        entries = list(BUILTIN_ENTRIES)
        if os.path.isfile(path):
            entries.extend(read_entries(path))
        return cls(entries)

    def add(self, entry: ModelEntry) -> None:
        """Register *entry*, over one of the same name."""
        self._entries[entry.name] = entry

    def entries(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def resolve(self, name: str) -> Tuple[ModelEntry, str]:
        """The entry for *name* and the version asked for: ``name@0.2`` pins
        ``0.2`` over the entry's own pin. LookupError for a name not registered."""
        name, _, version = name.partition("@")
        entry = self._entries.get(name)
        if entry is None:
            known = ", ".join(sorted(self._entries)) or "none"
            raise LookupError(f"no model '{name}' is registered; registered: {known}")
        return entry, version or entry.version

    def default_for(self, task: str) -> Optional[ModelEntry]:
        """The model a task uses when none is named: the last registered for it,
        so the file's choice wins over fibsem's."""
        found = [entry for entry in self._entries.values() if task in entry.default_for]
        return found[-1] if found else None


def read_entries(path: str) -> List[ModelEntry]:
    """The entries in the registry file at *path*. One that doesn't read is
    logged and left out, so one bad entry doesn't hide the rest."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    entries = []
    for name, value in (data.get("models") or {}).items():
        try:
            entries.append(_entry(name, value))
        except (TypeError, ValueError) as error:
            logging.warning(f"{path}: model '{name}' left out: {error}")
    return entries


def _entry(name: str, value: Mapping[str, Any]) -> ModelEntry:
    if not isinstance(value, Mapping) or not value.get("ref"):
        raise ValueError("an entry needs a ref")
    return ModelEntry(name=name, **{k: v for k, v in value.items() if k != "name"})
