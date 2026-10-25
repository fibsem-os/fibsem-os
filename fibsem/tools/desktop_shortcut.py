"""Create a desktop shortcut that launches one of the apps in fibsem.tools.apps.

Replaces the manual steps in INSTALLATION.md ("Create a desktop shortcut"): the
running application already knows which environment it lives in, so the entry
point is resolved from ``sys.executable`` rather than relying on ``where``/
``which`` in an activated shell.

Pure Python, no Qt -- the UI wires this into a menu action, and tests exercise
it headlessly.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fibsem.tools.apps import AUTOLAMELLA, App


def find_entry_point(name: str = AUTOLAMELLA.command) -> Optional[Path]:
    """Locate the console-script executable for the current environment.

    Looks beside ``sys.executable`` first: a venv on Windows puts entry points
    in ``Scripts/`` next to ``python.exe``, while conda puts ``python.exe`` in
    the environment root with entry points in a ``Scripts/`` subdirectory. On
    POSIX both live in ``bin/`` beside the interpreter. Falls back to PATH,
    which only helps when the launching shell had the environment activated.

    The interpreter's own path comes first, unresolved: in a venv or uv environment
    on POSIX it is a symlink to a shared Python, and resolving it leaves the
    environment.
    """
    exe_name = name + ".exe" if os.name == "nt" else name
    interpreter = Path(sys.executable)
    for interpreter_dir in (interpreter.parent, interpreter.resolve().parent):
        for directory in (interpreter_dir, interpreter_dir / "Scripts"):
            candidate = directory / exe_name
            if candidate.is_file():
                return candidate

    import shutil

    which = shutil.which(name)
    return Path(which) if which else None


def get_desktop_directory() -> Path:
    """Return the user's Desktop directory.

    On Windows the Desktop is queried from the shell (SHGetKnownFolderPath) so
    OneDrive-redirected desktops resolve correctly; ``~/Desktop`` is only the
    fallback. Elsewhere ``~/Desktop`` is used directly.
    """
    if os.name == "nt":
        known = _windows_known_desktop()
        if known is not None:
            return known
    return Path.home() / "Desktop"


def _windows_known_desktop() -> Optional[Path]:
    """SHGetKnownFolderPath(FOLDERID_Desktop), or None if the query fails."""
    try:
        import ctypes
        from ctypes import windll, wintypes  # type: ignore[attr-defined]

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", wintypes.BYTE * 8),
            ]

        # FOLDERID_Desktop {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
        folder_id = GUID(
            0xB4BFCC3A,
            0xDB2C,
            0x424C,
            (wintypes.BYTE * 8)(0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41),
        )
        path_ptr = ctypes.c_wchar_p()
        result = windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(folder_id), 0, None, ctypes.byref(path_ptr)
        )
        if result != 0 or not path_ptr.value:
            return None
        try:
            return Path(path_ptr.value)
        finally:
            windll.ole32.CoTaskMemFree(path_ptr)
    except Exception:
        return None


def shortcut_path(directory: Optional[Path] = None, app: App = AUTOLAMELLA) -> Path:
    """The path ``app``'s shortcut will be written to on this platform.

    ``directory`` is where the shortcut goes; the Desktop when omitted.
    """
    if directory is None:
        directory = get_desktop_directory()
    if os.name == "nt":
        return directory / (app.name + ".lnk")
    if sys.platform == "darwin":
        return directory / (app.name + ".command")
    return directory / (app.name + ".desktop")


def create_desktop_shortcut(
    overwrite: bool = False, directory: Optional[Path] = None, app: App = AUTOLAMELLA
) -> Path:
    """Create a shortcut launching ``app``; return its path.

    Written into ``directory``, or the user's Desktop when omitted. Raises
    FileNotFoundError when the entry point cannot be located (e.g. an
    editable install whose scripts were never generated), FileExistsError when
    a shortcut already exists and ``overwrite`` is False, and OSError /
    subprocess.CalledProcessError when writing fails.
    """
    target = find_entry_point(app.command)
    if target is None:
        raise FileNotFoundError(
            f"Could not locate the '{app.command}' executable in this environment."
        )

    destination = shortcut_path(directory, app)
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        _write_windows_lnk(destination, target, app)
    elif sys.platform == "darwin":
        _write_script(destination, f"#!/bin/bash\nexec '{_posix_quote(target)}'\n")
    else:
        _write_script(
            destination,
            "[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={app.name}\n"
            f"Comment=Launch {app.name}\n"
            f'Exec="{str(target)}"\n'
            f"Icon={app.icon}\n"
            "Terminal=false\n",
        )
    return destination


def _posix_quote(path: Path) -> str:
    """Escape a path for inclusion inside single quotes in a shell script."""
    return str(path).replace("'", "'\\''")


def _write_script(destination: Path, content: str) -> None:
    destination.write_text(content)
    destination.chmod(
        destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )


def _write_windows_lnk(destination: Path, target: Path, app: App) -> None:
    """Create a real .lnk via WScript.Shell so no console window flashes.

    PowerShell single-quoted strings escape a quote by doubling it, and paths
    cannot contain double quotes on Windows, so doubling ``'`` is sufficient.
    """

    def q(value) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    lines = [
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({q(destination)})",
        f"$s.TargetPath = {q(target)}",
        f"$s.WorkingDirectory = {q(Path.home())}",
        f"$s.Description = {q('Launch ' + app.name)}",
    ]
    if app.windows_icon is not None:
        lines.append(f"$s.IconLocation = {q(app.windows_icon)}")
    lines.append("$s.Save()")
    script = "; ".join(lines)
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
