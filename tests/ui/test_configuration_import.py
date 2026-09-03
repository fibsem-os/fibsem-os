"""Importing a microscope configuration from the connection tab.

Declining the old "add to user configurations?" prompt still put the name in the
combo box, so selecting it raised KeyError out of a currentTextChanged slot. That
is not a catchable error: PyQt5 prints the traceback and then aborts the process,
so the whole app died and the configuration stayed permanently unselectable.

These tests assert the property that keeps the app alive -- every name the combo
box offers resolves to a registered path, and selecting one never raises.
"""

import importlib
import shutil

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate
pytest.importorskip("PyQt5")

import fibsem.config as cfg
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

# fibsem.ui re-exports the class under the submodule's name, so `import
# fibsem.ui.FibsemSystemSetupWidget as ...` binds the class rather than the module.
widget_module = importlib.import_module("fibsem.ui.FibsemSystemSetupWidget")


@pytest.fixture
def configuration_file(tmp_path):
    """A real configuration, so the import path is exercised end to end."""

    def _make(name: str, subdirectory: str = "") -> str:
        directory = tmp_path / subdirectory if subdirectory else tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / name
        shutil.copyfile(cfg.MICROSCOPE_CONFIGURATION_PATH, destination)
        return str(destination)

    return _make


@pytest.fixture
def setup_widget(qapp, tmp_path, monkeypatch):
    """The connection tab, with the user configuration state isolated per test."""
    configurations = {
        "default-configuration": {"path": cfg.MICROSCOPE_CONFIGURATION_PATH}
    }
    monkeypatch.setattr(cfg, "USER_CONFIGURATIONS", configurations)
    monkeypatch.setattr(
        cfg,
        "USER_CONFIGURATIONS_YAML",
        {"configurations": configurations, "default": "default-configuration"},
    )
    monkeypatch.setattr(
        cfg, "USER_CONFIGURATIONS_PATH", str(tmp_path / "user-configurations.yaml")
    )
    monkeypatch.setattr(cfg, "DEFAULT_CONFIGURATION_NAME", "default-configuration")

    toasts = []
    monkeypatch.setattr(
        widget_module.notification_service,
        "show_toast",
        lambda message, notification_type="info": toasts.append(
            (notification_type, message)
        ),
    )

    widget = FibsemSystemSetupWidget()
    widget.toasts = toasts
    return widget


def _import_file(widget, monkeypatch, path: str, set_as_default: bool = False):
    """Drive import_configuration_from_file with the dialogs answered for us."""
    monkeypatch.setattr(
        widget_module, "open_existing_file_dialog", lambda **kwargs: path
    )
    monkeypatch.setattr(
        widget_module, "message_box_ui", lambda **kwargs: set_as_default
    )
    widget.import_configuration_from_file()


def _items(widget) -> list:
    combo = widget.comboBox_configuration
    return [combo.itemText(i) for i in range(combo.count())]


def test_imported_configuration_is_selectable(
    setup_widget, monkeypatch, configuration_file
):
    """The reported crash: import a configuration, then select it."""
    widget = setup_widget
    path = configuration_file("meteor.yaml")

    _import_file(widget, monkeypatch, path)

    assert "meteor" in _items(widget)
    assert cfg.USER_CONFIGURATIONS["meteor"]["path"] == path
    assert widget.comboBox_configuration.currentText() == "meteor"

    # the action that used to abort the process
    for index in range(widget.comboBox_configuration.count()):
        widget.comboBox_configuration.setCurrentIndex(index)

    assert not [t for t in widget.toasts if t[0] == "error"]


def test_every_offered_name_resolves_to_a_path(
    setup_widget, monkeypatch, configuration_file
):
    """The invariant the combo box depends on -- no entry may be unregistered."""
    widget = setup_widget

    _import_file(widget, monkeypatch, configuration_file("meteor.yaml", "a"))
    _import_file(widget, monkeypatch, configuration_file("meteor.yaml", "b"))
    _import_file(widget, monkeypatch, configuration_file("hydra.yml"))

    for name in _items(widget):
        assert cfg.USER_CONFIGURATIONS.get(name, {}).get("path") is not None


def test_reimporting_the_same_file_adds_no_duplicate(
    setup_widget, monkeypatch, configuration_file
):
    widget = setup_widget
    path = configuration_file("meteor.yaml")

    _import_file(widget, monkeypatch, path)
    _import_file(widget, monkeypatch, path)

    items = _items(widget)
    assert items.count("meteor") == 1
    assert len(items) == len(set(items))


def test_configurations_sharing_a_basename_keep_their_own_paths(
    setup_widget, monkeypatch, configuration_file
):
    """Two microscope-configuration.yaml files must not collapse onto one entry."""
    widget = setup_widget
    first = configuration_file("microscope-configuration.yaml", "a")
    second = configuration_file("microscope-configuration.yaml", "b")

    _import_file(widget, monkeypatch, first)
    _import_file(widget, monkeypatch, second)

    assert cfg.USER_CONFIGURATIONS["microscope-configuration"]["path"] == first
    assert cfg.USER_CONFIGURATIONS["microscope-configuration (2)"]["path"] == second
    assert widget.comboBox_configuration.currentText() == "microscope-configuration (2)"


def test_import_loads_the_configuration(setup_widget, monkeypatch, configuration_file):
    widget = setup_widget
    widget.settings = None

    _import_file(widget, monkeypatch, configuration_file("meteor.yaml"))

    assert widget.settings is not None


def test_declining_the_default_prompt_still_leaves_it_selectable(
    setup_widget, monkeypatch, configuration_file
):
    """Answering No to "make this the default?" must not strand the entry."""
    widget = setup_widget

    _import_file(
        widget, monkeypatch, configuration_file("meteor.yaml"), set_as_default=False
    )

    assert cfg.USER_CONFIGURATIONS_YAML["default"] == "default-configuration"
    assert cfg.USER_CONFIGURATIONS.get("meteor", {}).get("path") is not None


def test_cancelling_the_dialog_imports_nothing(setup_widget, monkeypatch):
    widget = setup_widget
    before = _items(widget)

    _import_file(widget, monkeypatch, "")

    assert _items(widget) == before


def test_selecting_an_unregistered_name_reports_instead_of_raising(setup_widget):
    """A stale combo entry -- hand-edited yaml, or a configuration since removed."""
    widget = setup_widget
    widget.comboBox_configuration.addItem("ghost-configuration")

    widget.comboBox_configuration.setCurrentText("ghost-configuration")

    assert ("error", "Configuration ghost-configuration not found.") in widget.toasts


def test_unreadable_configuration_reports_instead_of_raising(setup_widget, tmp_path):
    """The registered file was moved or corrupted after it was added."""
    widget = setup_widget
    cfg.USER_CONFIGURATIONS["missing"] = {"path": str(tmp_path / "gone.yaml")}
    widget.comboBox_configuration.addItem("missing")

    widget.comboBox_configuration.setCurrentText("missing")

    assert [t for t in widget.toasts if t[0] == "error"]
