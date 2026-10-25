"""A driver's vendor library served from a folder outside the environment."""

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
