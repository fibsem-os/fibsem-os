"""The desktop apps fibsemOS can launch, and what the OS should call them.

One record per app holds what the launcher, the window and the desktop shortcut
each need: the name, the Windows taskbar id, the icon, the console command and
the function to run. AutoLamella and the fibsemOS UI are built in; a package
outside this repo adds its own app with an entry point in the ``fibsem.apps``
group naming an ``App``::

    [project.entry-points."fibsem.apps"]
    liftout = "autoliftout.app:APP"

Keep the module that entry point names cheap to import (an ``App`` and
nothing more): the launcher loads it before the splash screen is up.

Pure Python, no Qt: the desktop shortcut code reads this too.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

ENTRY_POINT_GROUP = "fibsem.apps"
ICONS_DIR = Path(str(importlib.resources.files("fibsem"))) / "ui" / "icons"


@dataclass(frozen=True)
class App:
    """A desktop app the launcher can start."""

    name: str
    """What people see: window, menu bar, splash title, shortcut name."""
    app_id: str
    """The Windows taskbar identity, ``Company.Product``."""
    command: str
    """The console command that runs it; desktop shortcuts point at this."""
    target: str
    """``"package.module:function"``, imported once the splash screen is up and
    called as ``function(argv, splash=splash)``."""
    icon: Path
    """The full-colour SVG. A ``<stem>-16.svg`` beside it is the 16 px drawing,
    and a ``<stem>.ico`` beside it is the Windows shortcut icon."""

    @property
    def icon_16(self) -> Path:
        small = self.icon.with_name(f"{self.icon.stem}-16.svg")
        return small if small.is_file() else self.icon

    @property
    def windows_icon(self) -> Optional[Path]:
        ico = self.icon.with_suffix(".ico")
        return ico if ico.is_file() else None


AUTOLAMELLA = App(
    name="AutoLamella",
    app_id="fibsemOS.AutoLamella",
    command="fibsem-autolamella-ui",
    target="fibsem.applications.autolamella.ui.AutoLamellaMainUI:run_ui",
    icon=ICONS_DIR / "autolamella.svg",
)

FIBSEM_UI = App(
    name="fibsemOS",
    app_id="fibsemOS.FibsemUI",
    command="fibsem_ui",
    target="fibsem.ui.FibsemUI:main",
    icon=ICONS_DIR / "fibsemos.svg",
)

BUILTIN_APPS: Dict[str, App] = {"autolamella": AUTOLAMELLA, "fibsem-ui": FIBSEM_UI}


def apps() -> Dict[str, App]:
    """The built-in apps, then any registered under ``fibsem.apps``.

    An entry point that fails to load, or names something other than an
    ``App``, is logged and skipped: one broken plugin must not stop the others
    from launching. A plugin cannot replace a built-in app.
    """
    found = dict(BUILTIN_APPS)
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name in found:
            logging.warning("ignoring app entry point %r: name taken", entry_point.name)
            continue
        try:
            app = entry_point.load()
        except Exception as e:
            logging.warning("could not load app %r: %s", entry_point.name, e)
            continue
        if not isinstance(app, App):
            logging.warning("app entry point %r is not an App", entry_point.name)
            continue
        found[entry_point.name] = app
    return found
