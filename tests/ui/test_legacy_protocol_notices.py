"""The load-experiment dialog's protocol notices (FIB-663).

A legacy protocol.yaml is now converted transparently on load, so the dialog
can end up showing a protocol that does not match the file on disk. These tests
drive the real dialog against real protocol files and assert the user is told:
a modal on the committed load, and a banner that stays up while the converted
protocol is loaded.

QMessageBox is stubbed rather than shown -- a modal would block the run -- so
the recorder doubles as the assertion that the right notice fired.
"""
import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol, Experiment
from fibsem.applications.autolamella.ui import autolamella_load_experiment_widget as mod
from fibsem.applications.autolamella.ui.autolamella_load_experiment_widget import (
    AutoLamellaLoadExperimentWidget,
)

PROTOCOL_DIR = (
    Path(__file__).parents[2] / "fibsem/applications/autolamella/protocol"
)
LEGACY_PROTOCOL = PROTOCOL_DIR / "legacy/protocol-on-grid.yaml"
TASK_PROTOCOL = PROTOCOL_DIR / "task-protocol.yaml"


@pytest.fixture
def notices(monkeypatch) -> List[Tuple[str, str, str]]:
    """Record (kind, title, text) for every modal the dialog raises."""
    recorded: List[Tuple[str, str, str]] = []

    def record(kind):
        def _stub(parent, title, text, *args, **kwargs):
            recorded.append((kind, title, text))
            return mod.QtWidgets.QMessageBox.Ok
        return staticmethod(_stub)

    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(mod.QtWidgets.QMessageBox, kind, record(kind))
    return recorded


@pytest.fixture
def dialog(qapp):
    widget = AutoLamellaLoadExperimentWidget(parent=None)
    yield widget
    widget.close()


def make_experiment(path: Path, protocol: Path = None) -> Path:
    """Write a real experiment, optionally with a protocol.yaml verbatim."""
    experiment = Experiment.create(path=path, name="legacy-experiment")
    experiment.save()
    if protocol is not None:
        (Path(experiment.path) / "protocol.yaml").write_text(protocol.read_text())
    return Path(experiment.path) / "experiment.yaml"


# ── converted protocol ────────────────────────────────────────────────────────

def test_legacy_experiment_loads_and_warns(dialog, notices, tmp_path):
    """The headline case: the protocol converts, and the user is told."""
    assert dialog._load_experiment_from_path(str(make_experiment(tmp_path, LEGACY_PROTOCOL)))

    assert dialog.experiment.task_protocol is not None
    assert dialog.experiment.task_protocol.converted_from_legacy
    assert dialog.btn_ok.isEnabled()

    assert [n[1] for n in notices] == [mod.LEGACY_PROTOCOL_CONVERTED_TITLE]
    kind, _, text = notices[0]
    assert kind == "information"
    assert "protocol.yaml" in text          # names the file it converted
    assert "until the experiment is saved" in text


def test_converted_protocol_shows_the_banner(dialog, notices, tmp_path):
    dialog._load_experiment_from_path(str(make_experiment(tmp_path, LEGACY_PROTOCOL)))
    assert dialog.label_legacy_protocol_banner.isVisibleTo(dialog)


def test_task_protocol_is_silent_and_unbannered(dialog, notices, tmp_path):
    """A current protocol must load exactly as before -- no notice, no banner."""
    dialog._load_experiment_from_path(str(make_experiment(tmp_path, TASK_PROTOCOL)))

    assert dialog.experiment.task_protocol is not None
    assert not dialog.experiment.task_protocol.converted_from_legacy
    assert notices == []
    assert not dialog.label_legacy_protocol_banner.isVisibleTo(dialog)


def test_preview_click_is_quiet_but_still_banners(dialog, notices, tmp_path):
    """Single-click preview must not pop a modal; the banner still shows state."""
    dialog._load_experiment_from_path(
        str(make_experiment(tmp_path, LEGACY_PROTOCOL)), show_protocol_notices=False
    )

    assert notices == []
    assert dialog.experiment.task_protocol.converted_from_legacy
    assert dialog.label_legacy_protocol_banner.isVisibleTo(dialog)


def test_banner_clears_when_the_display_is_cleared(dialog, notices, tmp_path):
    dialog._load_experiment_from_path(str(make_experiment(tmp_path, LEGACY_PROTOCOL)))
    assert dialog.label_legacy_protocol_banner.isVisibleTo(dialog)

    dialog._clear_display()
    assert not dialog.label_legacy_protocol_banner.isVisibleTo(dialog)


def test_banner_clears_when_a_task_protocol_replaces_a_converted_one(
    dialog, notices, tmp_path
):
    """The banner tracks the loaded protocol, not the dialog's history."""
    dialog._load_experiment_from_path(str(make_experiment(tmp_path / "a", LEGACY_PROTOCOL)))
    assert dialog.label_legacy_protocol_banner.isVisibleTo(dialog)

    dialog._load_experiment_from_path(str(make_experiment(tmp_path / "b", TASK_PROTOCOL)))
    assert not dialog.label_legacy_protocol_banner.isVisibleTo(dialog)


# ── protocol missing or unreadable ────────────────────────────────────────────

def test_missing_protocol_says_not_found(dialog, notices, tmp_path):
    dialog._load_experiment_from_path(str(make_experiment(tmp_path)))

    assert [n[1] for n in notices] == [mod.ERROR_PROTOCOL_NOT_FOUND_TITLE]
    assert not dialog.btn_ok.isEnabled()
    assert dialog.btn_select_protocol.isVisibleTo(dialog)


def test_unreadable_protocol_reports_the_error(dialog, notices, tmp_path):
    """A protocol that exists but cannot be read is not a missing protocol."""
    experiment_yaml = make_experiment(tmp_path)
    (experiment_yaml.parent / "protocol.yaml").write_text("{not: [valid")

    dialog._load_experiment_from_path(str(experiment_yaml))

    assert [n[1] for n in notices] == [mod.ERROR_PROTOCOL_LOAD_FAILED_TITLE]
    assert "protocol.yaml" in notices[0][2]
    assert not dialog.btn_ok.isEnabled()
    assert dialog.btn_select_protocol.isVisibleTo(dialog)


def test_liftout_protocol_reports_the_unsupported_method(dialog, notices, tmp_path):
    """Conversion failures surface with their reason, not as 'not found'."""
    experiment_yaml = make_experiment(
        tmp_path, PROTOCOL_DIR / "legacy/protocol-liftout.yaml"
    )

    dialog._load_experiment_from_path(str(experiment_yaml))

    assert [n[1] for n in notices] == [mod.ERROR_PROTOCOL_LOAD_FAILED_TITLE]
    assert "not supported for conversion" in notices[0][2]


def test_wrong_file_is_named_as_such(dialog, notices, tmp_path):
    """A microscope configuration has a top-level 'milling' key too."""
    experiment_yaml = make_experiment(tmp_path)
    (experiment_yaml.parent / "protocol.yaml").write_text(
        (Path(__file__).parents[2] / "fibsem/config/microscope-configuration.yaml").read_text()
    )

    dialog._load_experiment_from_path(str(experiment_yaml))

    assert [n[1] for n in notices] == [mod.ERROR_PROTOCOL_LOAD_FAILED_TITLE]
    assert "does not look like an AutoLamella protocol" in notices[0][2]


# ── the manual Select Protocol button ─────────────────────────────────────────

def test_select_protocol_announces_a_legacy_file(dialog, notices, tmp_path, monkeypatch):
    """The primary button accepts legacy files, so it must say when it converts."""
    dialog._load_experiment_from_path(str(make_experiment(tmp_path)))
    notices.clear()
    monkeypatch.setattr(mod.fui, "open_existing_file_dialog", lambda **kw: str(LEGACY_PROTOCOL))

    dialog._select_protocol()

    assert dialog.experiment.task_protocol.converted_from_legacy
    assert [n[1] for n in notices] == [mod.LEGACY_PROTOCOL_CONVERTED_TITLE]
    assert dialog.label_legacy_protocol_banner.isVisibleTo(dialog)
    assert dialog.btn_ok.isEnabled()


def test_select_protocol_is_silent_for_a_task_protocol(dialog, notices, tmp_path, monkeypatch):
    dialog._load_experiment_from_path(str(make_experiment(tmp_path)))
    notices.clear()
    monkeypatch.setattr(mod.fui, "open_existing_file_dialog", lambda **kw: str(TASK_PROTOCOL))

    dialog._select_protocol()

    assert notices == []
    assert not dialog.label_legacy_protocol_banner.isVisibleTo(dialog)
    assert dialog.btn_ok.isEnabled()


def test_protocol_load_error_distinguishes_absent_from_broken(tmp_path):
    """The helper behind the two different warnings."""
    assert AutoLamellaLoadExperimentWidget._protocol_load_error(
        str(tmp_path / "nope.yaml")
    ) is None

    broken = tmp_path / "protocol.yaml"
    broken.write_text("{not: [valid")
    assert AutoLamellaLoadExperimentWidget._protocol_load_error(str(broken)) is not None

    assert AutoLamellaLoadExperimentWidget._protocol_load_error(str(TASK_PROTOCOL)) == (
        "unknown error"
    )
