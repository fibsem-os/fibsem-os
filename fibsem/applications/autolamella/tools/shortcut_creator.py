from __future__ import annotations
import shutil

import pyshortcuts


def _check_create_shortcut(shortcut_type: str) -> bool:
    msg = f"Do you wish to create a {shortcut_type} shortcut'? [Y/N]"
    option = str(input(msg)).strip().lower()
    if option in ("y", "yes"):
        return True
    elif option in ("n", "no"):
        return False
    print(f"Invalid option '{option}'")
    return _check_create_shortcut(shortcut_type=shortcut_type)


def _create_shortcut_file(
    desktop: bool = True, start_menu: bool = True, gui: bool = True
) -> None:
    pyshortcuts.make_shortcut(
        shutil.which("fibsem-autolamella-ui"),
        name="AutoLamella",
        desktop=desktop,
        startmenu=start_menu,
    )


def create_shortcuts() -> None:
    _create_shortcut_file(
        desktop=_check_create_shortcut("desktop"),
        start_menu=_check_create_shortcut("start menu"),
    )


if __name__ == "__main__":
    create_shortcuts()
