"""A driver's vendor library served from a folder outside the environment."""

import os
import subprocess
import sys

import pytest

from fibsem import config as cfg
from fibsem import vendor_libraries as vl
from fibsem.drivers.setup import SetupLibrary

ENV = "FIBSEM_TEST_VENDOR_PATH"
LIBRARY = SetupLibrary(
    label="Test vendor library",
    packages=("tvendor_client", "tvendor_extra"),
    env=ENV,
)


def _library_folder(root, version="1.2.3", name="site-packages"):
    folder = root / name
    package = folder / "tvendor_client"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"VERSION = {version!r}\n")
    (package / "child.py").write_text("NAME = 'child'\n")
    (folder / "tvendor_extra.py").write_text("EXTRA = True\n")
    # a package the folder also holds that must never be served from it
    (folder / "tvendor_decoy.py").write_text("raise RuntimeError('served')\n")
    meta = folder / f"tvendor_client-{version}.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: tvendor-client\nVersion: {version}\n"
    )
    return folder


@pytest.fixture
def preferences(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(
        cfg, "USER_PREFERENCES_PATH", str(tmp_path / "user-preferences.yaml")
    )
    monkeypatch.delenv(ENV, raising=False)
    yield
    finder = vl._FINDERS.pop("TestVendor", None)
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)
    for name in list(sys.modules):
        if name.startswith("tvendor_"):
            del sys.modules[name]


def test_a_folder_says_what_it_holds_and_its_version(tmp_path):
    folder = _library_folder(tmp_path)
    found = vl.check(str(folder), LIBRARY)
    assert found.ok
    assert found.found == ("tvendor_client", "tvendor_extra")
    assert found.version == "1.2.3"
    assert not vl.check(str(tmp_path / "missing"), LIBRARY).ok


def test_the_environment_wins_then_the_preference(tmp_path, monkeypatch, preferences):
    preferred = _library_folder(tmp_path / "a")
    from_env = _library_folder(tmp_path / "b")
    vl.save_folder("TestVendor", str(preferred))
    assert vl.resolve("TestVendor", LIBRARY).folder == str(preferred)
    assert vl.resolve("TestVendor", LIBRARY).source == vl.SOURCE_PREFERENCE
    monkeypatch.setenv(ENV, str(from_env))
    assert vl.resolve("TestVendor", LIBRARY).folder == str(from_env)
    assert vl.resolve("TestVendor", LIBRARY).source == vl.SOURCE_ENV


def test_an_empty_folder_clears_the_preference(tmp_path, preferences):
    vl.save_folder("TestVendor", str(tmp_path))
    vl.save_folder("TestVendor", "  ")
    assert "TestVendor" not in cfg.load_user_preferences().install.vendor_libraries


def test_unset_it_takes_the_newest_candidate(tmp_path, preferences):
    _library_folder(tmp_path / "Python310", version="1.9.0")
    newest = _library_folder(tmp_path / "Python311", version="1.10.0")
    (tmp_path / "Python312" / "site-packages").mkdir(parents=True)  # holds nothing
    library = SetupLibrary(
        label=LIBRARY.label,
        packages=LIBRARY.packages,
        candidates=(str(tmp_path / "Python3*" / "site-packages"),),
    )
    found = vl.resolve("TestVendor", library)
    assert (found.folder, found.source) == (str(newest), vl.SOURCE_FOUND)
    assert found.version == "1.10.0"


def test_only_the_named_packages_are_served(tmp_path, preferences):
    folder = _library_folder(tmp_path)
    vl.save_folder("TestVendor", str(folder))
    assert vl.serve("TestVendor", LIBRARY) == str(folder)
    import tvendor_client
    import tvendor_client.child
    import tvendor_extra

    assert tvendor_client.VERSION == "1.2.3"
    assert tvendor_client.child.NAME == "child" and tvendor_extra.EXTRA
    assert str(folder) not in sys.path
    with pytest.raises(ModuleNotFoundError):
        import tvendor_decoy  # noqa: F401


def test_serving_again_replaces_the_finder(tmp_path, preferences):
    vl.serve("TestVendor", LIBRARY, str(_library_folder(tmp_path / "a")))
    vl.serve("TestVendor", LIBRARY, str(_library_folder(tmp_path / "b")))
    ours = [f for f in sys.meta_path if isinstance(f, vl._LibraryFinder)]
    assert [f.folder for f in ours] == [str(tmp_path / "b" / "site-packages")]


def test_nothing_found_says_where_to_set_it(preferences):
    with pytest.raises(ImportError, match=f"Preferences.*{ENV}"):
        vl.serve("TestVendor", LIBRARY)


def test_a_copy_in_this_environment_wins_and_is_not_served(
    tmp_path, monkeypatch, preferences
):
    own = _library_folder(tmp_path / "own")
    vl.save_folder("TestVendor", str(_library_folder(tmp_path / "elsewhere")))
    monkeypatch.syspath_prepend(str(own))
    found = vl.resolve("TestVendor", LIBRARY)
    assert (found.folder, found.source) == (str(own), vl.SOURCE_INSTALLED)
    assert vl.serve("TestVendor", LIBRARY) is None
    assert "TestVendor" not in vl._FINDERS


def test_an_upgrade_in_a_new_folder_is_taken_up(tmp_path, preferences):
    folder = _library_folder(tmp_path / "Python311")
    library = SetupLibrary(
        label=LIBRARY.label,
        packages=LIBRARY.packages,
        candidates=(str(tmp_path / "Python3*" / "site-packages"),),
    )
    assert vl.serve("TestVendor", library) == str(folder)
    assert "TestVendor" not in cfg.load_user_preferences().install.vendor_libraries
    newer = _library_folder(tmp_path / "Python312", version="9.0.0")
    assert vl.serve("TestVendor", library) == str(newer)


def test_the_version_is_the_first_named_packages(tmp_path):
    folder = _library_folder(tmp_path)
    meta = folder / "tvendor_extra-7.0.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: tvendor-extra\nVersion: 7.0\n"
    )
    assert vl.version(str(folder), LIBRARY.packages) == "1.2.3"
    assert vl.version(str(folder), ("tvendor_extra",)) == "7.0"


def _fake_python(directory, prints):
    directory.mkdir(parents=True)
    python = directory / "python"
    python.write_text(f"#!/bin/sh\necho '{prints}'\n")
    python.chmod(0o755)
    return python


@pytest.mark.skipif(os.name == "nt", reason="the fake Python is a shell script")
def test_search_asks_each_python_on_path(tmp_path, monkeypatch):
    old = _library_folder(tmp_path / "a", version="1.0")
    new = _library_folder(tmp_path / "b", version="2.0")
    _fake_python(tmp_path / "bin1", old)
    _fake_python(tmp_path / "bin2", new)
    _fake_python(tmp_path / "bin3", tmp_path / "nothing-here")
    _fake_python(tmp_path / "WindowsApps", new)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            str(tmp_path / d) for d in ("bin1", "bin2", "bin3", "WindowsApps", "bin1")
        ),
    )
    assert vl.pythons_on_path() == [
        str(tmp_path / d / "python") for d in ("bin1", "bin2", "bin3")
    ]
    found = vl.search(LIBRARY)
    assert [(f.folder, f.version) for f in found] == [
        (str(new), "2.0"),
        (str(old), "1.0"),
    ]
    assert all(f.source == vl.SOURCE_FOUND for f in found)


@pytest.mark.skipif(os.name == "nt", reason="the fake Python is a shell script")
def test_a_python_that_hangs_is_given_up_on(tmp_path, monkeypatch):
    directory = tmp_path / "bin"
    directory.mkdir()
    python = directory / "python"
    python.write_text("#!/bin/sh\nsleep 5\n")
    python.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory))
    assert vl.search(LIBRARY, timeout=0.2) == []


def test_the_search_script_finds_a_package_without_importing_it(tmp_path):
    folder = _library_folder(tmp_path)
    (folder / "tvendor_client" / "__init__.py").write_text("raise RuntimeError\n")
    result = subprocess.run(
        [sys.executable, "-c", vl._WHERE_SCRIPT, "tvendor_client"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(folder)},
    )
    assert result.stdout.strip() == str(folder)
