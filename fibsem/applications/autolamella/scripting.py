"""AutoLamella's binding for user scripts.

The discovery and execution machinery is application-agnostic and lives in
``fibsem.scripting``. This module supplies the two things that are specific to
AutoLamella: where the scripts live, and what a script is handed when it runs.

See SCRIPTING.md for the headless equivalent, and FIB-338 for the design.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

from fibsem.scripting import (  # noqa: F401 - re-exported for callers
    DiscoveredScript,
    ScriptResult,
    discover_scripts as _discover_scripts,
    load_script,
    run_script,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope

# Interim location: the final root is decided by FIB-336, and this should become
# <user root>/scripts rather than being chosen separately. `%APPDATA%` was rejected
# (hidden in Explorer, and users must browse here to drop files in); Documents was
# rejected (OneDrive-synced on most institutional machines). A leading dot does not
# set the hidden attribute on Windows, so this is visible there too. See FIB-341.
SCRIPTS_DIR_ENV_VAR = "AUTOLAMELLA_SCRIPTS_DIR"
DEFAULT_SCRIPTS_DIR = Path.home() / ".autolamella" / "scripts"


def get_scripts_directory() -> Path:
    """Resolve the user scripts directory. Does not create it."""
    override = os.environ.get(SCRIPTS_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_SCRIPTS_DIR


def discover_scripts(directory: Optional[Path] = None) -> List[DiscoveredScript]:
    """Load the user's scripts, defaulting to the AutoLamella scripts folder."""
    return _discover_scripts(directory or get_scripts_directory())


@dataclass
class ScriptContext:
    """What a script is handed when it runs.

    Deliberately not the ``AutoLamellaUI`` object. ``AutoLamellaUI.experiment`` is
    *rebound* on create/load rather than mutated, so a script that cached the ui
    would silently operate on a dead experiment; and this stays a documented,
    stable surface while the GUI internals are still moving.

    ``microscope`` is optional so the same file also runs headlessly, with no
    hardware attached.
    """

    experiment: "Experiment"
    log: Callable[[str], None] = logging.info
    microscope: Optional["FibsemMicroscope"] = None

    @property
    def path(self) -> Path:
        """The experiment directory — the default place to write output."""
        return Path(self.experiment.path)

    def save(self) -> None:
        """Persist the experiment. Called by the runner only for ``writes`` scripts."""
        self.experiment.save()
