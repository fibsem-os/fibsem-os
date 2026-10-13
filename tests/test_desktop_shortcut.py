import os
import stat
import sys

import pytest

from fibsem.tools import desktop_shortcut


def test_find_entry_point_resolves_in_this_environment():
    path = desktop_shortcut.find_entry_point()
    assert path is not None
    assert path.is_file()
    assert path.name.startswith(desktop_shortcut.ENTRY_POINT_NAME)


def test_shortcut_path_extension(tmp_path):
    path = desktop_shortcut.shortcut_path(tmp_path)
    assert path.parent == tmp_path
    if os.name == "nt":
        assert path.suffix == ".lnk"
    elif sys.platform == "darwin":
        assert path.suffix == ".command"
    else:
        assert path.suffix == ".desktop"


@pytest.mark.skipif(os.name == "nt", reason="POSIX writes a script; Windows a .lnk")
def test_create_desktop_shortcut_writes_executable_launcher(tmp_path):
    created = desktop_shortcut.create_desktop_shortcut(directory=tmp_path)
    assert created == desktop_shortcut.shortcut_path(tmp_path)
    assert str(desktop_shortcut.find_entry_point()) in created.read_text()
    assert created.stat().st_mode & stat.S_IXUSR


@pytest.mark.skipif(os.name == "nt", reason="POSIX writes a script; Windows a .lnk")
def test_create_desktop_shortcut_refuses_then_overwrites(tmp_path):
    first = desktop_shortcut.create_desktop_shortcut(directory=tmp_path)
    with pytest.raises(FileExistsError):
        desktop_shortcut.create_desktop_shortcut(directory=tmp_path)
    again = desktop_shortcut.create_desktop_shortcut(directory=tmp_path, overwrite=True)
    assert again == first


def test_shortcut_icons_ship_with_the_package():
    ico = desktop_shortcut.WINDOWS_ICON_PATH.read_bytes()
    assert ico[:4] == b"\0\0\1\0"  # an .ico, not a renamed PNG
    count = int.from_bytes(ico[4:6], "little")
    sizes = [ico[6 + 16 * i] or 256 for i in range(count)]
    assert sizes == [16, 24, 32, 48, 64, 128, 256]
    assert desktop_shortcut.LINUX_ICON_PATH.is_file()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux .desktop")
def test_linux_desktop_entry_names_its_icon(tmp_path):
    created = desktop_shortcut.create_desktop_shortcut(directory=tmp_path)
    assert f"Icon={desktop_shortcut.LINUX_ICON_PATH}\n" in created.read_text()
