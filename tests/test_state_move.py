"""The first start brings an older release's files out of the package folder."""

import logging
import os
import shutil
import subprocess

import pytest
import yaml

from fibsem import paths, state_move


def _write(path, data=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data if data is not None else {"x": 1}))
    return path


@pytest.fixture
def old_install(tmp_path):
    """A package folder as an older release left it, and an empty home."""
    package = tmp_path / "site-packages" / "fibsem"
    config = package / "config"
    autolamella = package / "applications" / "autolamella"
    _write(config / "microscope-configuration.yaml", {"shipped": True})
    _write(config / "tfs-arctis-configuration.yaml", {"edited": "in place"})
    _write(config / "my-arctis.yaml", {"mine": True})
    _write(config / "saved-positions.yaml", [])
    _write(config / "session" / "my-arctis.yaml", {"saved_positions": []})
    _write(config / "fm" / "fm-configuration-default.yaml")
    _write(config / "fm" / "exported.yaml")
    _write(package / "log" / "data" / "ml" / "a.tif")
    _write(autolamella / "log" / "exp1" / "experiment.yaml", {"name": "exp1"})
    _write(autolamella / "log" / "stray.yaml")
    _write(autolamella / "protocol" / "task-protocol.yaml")
    _write(autolamella / "protocol" / "mine.yaml")
    _write(autolamella / "protocol" / "legacy" / "protocol-on-grid.yaml")
    scripts = tmp_path / "home" / ".autolamella" / "scripts"
    _write(scripts / "hello.py")
    elsewhere = _write(tmp_path / "network" / "site.yaml")
    _write(
        config / "user-configurations.yaml",
        {
            "configurations": {
                "my-arctis": {"path": str(config / "my-arctis.yaml")},
                "arctis": {"path": str(config / "tfs-arctis-configuration.yaml")},
                "site": {"path": str(elsewhere)},
            },
            "default": "my-arctis",
        },
    )
    exp1 = autolamella / "log" / "exp1"
    _write(
        config / "user-preferences.yaml",
        {
            "experiment": {
                "default_experiment_directory": str(autolamella / "log"),
                "default_protocol_path": str(autolamella / "protocol" / "mine.yaml"),
                "last_experiment_path": str(exp1),
                "recent_experiments": [str(exp1 / "experiment.yaml"), str(elsewhere)],
            }
        },
    )
    return package, tmp_path / "home" / ".fibsem", scripts


def _load(path):
    return yaml.safe_load(path.read_text())


def test_settings_are_copied_and_shipped_files_stay(old_install):
    package, root, scripts = old_install
    state_move.move_package_state(package, root, scripts)

    config = root / "config"
    assert _load(config / "my-arctis.yaml") == {"mine": True}
    assert (config / "session" / "my-arctis.yaml").is_file()
    assert (config / "saved-positions.yaml").is_file()
    assert (config / "fm" / "exported.yaml").is_file()
    assert not (config / "fm" / "fm-configuration-default.yaml").exists()
    assert not (config / "microscope-configuration.yaml").exists()
    # Copied, so an older release still finds its own.
    assert (package / "config" / "my-arctis.yaml").is_file()
    assert (package / "config" / "user-preferences.yaml").is_file()


def test_the_list_of_configurations_points_at_the_copies(old_install):
    package, root, scripts = old_install
    state_move.move_package_state(package, root, scripts)

    listed = _load(root / "config" / "user-configurations.yaml")["configurations"]
    assert listed["my-arctis"]["path"] == str(root / "config" / "my-arctis.yaml")
    # A shipped file someone edited where it was is copied out with them.
    arctis = root / "config" / "tfs-arctis-configuration.yaml"
    assert listed["arctis"]["path"] == str(arctis)
    assert _load(arctis) == {"edited": "in place"}
    assert listed["site"]["path"].endswith(os.path.join("network", "site.yaml"))


def test_logs_experiments_protocols_and_scripts_move(old_install):
    package, root, scripts = old_install
    state_move.move_package_state(package, root, scripts)

    autolamella = package / "applications" / "autolamella"
    assert (root / "log" / "data" / "ml" / "a.tif").is_file()
    assert not (package / "log" / "data").exists()
    assert (root / "experiments" / "exp1" / "experiment.yaml").is_file()
    assert not (autolamella / "log" / "exp1").exists()
    assert (autolamella / "log" / "stray.yaml").is_file()
    assert (root / "autolamella" / "protocols" / "mine.yaml").is_file()
    assert (autolamella / "protocol" / "task-protocol.yaml").is_file()
    assert (autolamella / "protocol" / "legacy" / "protocol-on-grid.yaml").is_file()
    assert (root / "autolamella" / "scripts" / "hello.py").is_file()
    assert not scripts.joinpath("hello.py").exists()


def test_preferences_follow_what_moved(old_install):
    package, root, scripts = old_install
    state_move.move_package_state(package, root, scripts)

    experiment = _load(root / "config" / "user-preferences.yaml")["experiment"]
    exp1 = root / "experiments" / "exp1"
    assert experiment["default_experiment_directory"] == ""
    assert experiment["default_protocol_path"] == str(
        root / "autolamella" / "protocols" / "mine.yaml"
    )
    assert experiment["last_experiment_path"] == str(exp1)
    assert experiment["recent_experiments"][0] == str(exp1 / "experiment.yaml")
    assert experiment["recent_experiments"][1].endswith("site.yaml")


def test_it_says_what_it_did(old_install, caplog):
    package, root, scripts = old_install
    with caplog.at_level(logging.WARNING):
        moves = state_move.move_package_state(package, root, scripts)

    assert f"Moved fibsemOS settings from {package} to {root}" in caplog.text
    record = (root / state_move.RECORD_FILE).read_text()
    for line in moves.lines():
        assert line in record

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        state_move.log_pending()
        state_move.log_pending()
    assert caplog.text.count("Moved fibsemOS settings") == 1


def test_it_runs_once(old_install):
    package, root, scripts = old_install
    (root / "config").mkdir(parents=True)

    moves = state_move.move_package_state(package, root, scripts)

    assert not moves.copied and not moves.moved
    assert (package / "log" / "data" / "ml" / "a.tif").is_file()


def test_an_existing_target_is_left_alone(old_install):
    package, root, scripts = old_install
    _write(root / "experiments" / "exp1" / "experiment.yaml", {"name": "already"})

    moves = state_move.move_package_state(package, root, scripts)

    autolamella = package / "applications" / "autolamella"
    assert (autolamella / "log" / "exp1").is_dir()
    assert str(autolamella / "log" / "exp1") in moves.left


def test_the_shipped_lists_match_what_the_package_ships():
    if shutil.which("git") is None:
        pytest.skip("needs git")
    package = paths.PACKAGE_DIR
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "config", "applications/autolamella/protocol"],
            cwd=package,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout")
    if not tracked:
        pytest.skip("not a git checkout")
    top = {t.split("/", 1)[1] for t in tracked if t.startswith("config/")}
    assert {t for t in top if "/" not in t} == set(state_move.SHIPPED_CONFIGURATIONS)
    assert {t[3:] for t in top if t.startswith("fm/")} == set(
        state_move.SHIPPED_FM_FILES
    )
    protocols = {
        t.rsplit("/", 1)[1]
        for t in tracked
        if t.count("/") == 3 and t.endswith((".yaml", ".yml"))
    }
    assert protocols == set(state_move.SHIPPED_PROTOCOLS)
