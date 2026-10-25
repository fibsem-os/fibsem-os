import os
import sys

import pytest

from fibsem.tools import desktop_shortcut


@pytest.mark.skipif(os.name == "nt", reason="a POSIX venv's python is a symlink")
def test_the_entry_point_is_found_beside_a_symlinked_interpreter(tmp_path, monkeypatch):
    shared = tmp_path / "shared" / "bin"
    shared.mkdir(parents=True)
    (shared / "python3").write_text("")
    env_bin = tmp_path / "env" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "python").symlink_to(shared / "python3")
    entry_point = env_bin / "fibsem-autolamella-ui"
    entry_point.write_text("")
    monkeypatch.setattr(sys, "executable", str(env_bin / "python"))
    monkeypatch.setenv("PATH", "")

    assert desktop_shortcut.find_entry_point("fibsem-autolamella-ui") == entry_point
