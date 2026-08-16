"""Loading protocols and experiments written before the task-based format (FIB-663).

protocol.yaml is the filename for both the legacy and the task-based protocol
format, so the loader has to sniff which one it has rather than assume the new
one. Before this, a legacy protocol reached the task-protocol constructor and
died on the first legacy key it saw -- 'method',
'alignment_at_milling_current' -- which the UI reported as a corrupt file.
"""

import os
from pathlib import Path

import pytest
import yaml

from fibsem.applications.autolamella.protocol.legacy import (
    get_legacy_protocol_method,
    is_legacy_protocol,
    AutoLamellaMethod,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    AutoLamellaWorkflowOptions,
    Experiment,
)

LEGACY_PROTOCOL_DIR = (
    Path(__file__).parents[2]
    / "fibsem"
    / "applications"
    / "autolamella"
    / "protocol"
    / "legacy"
)
TASK_PROTOCOL_DIR = LEGACY_PROTOCOL_DIR.parent

CONVERTIBLE_LEGACY_PROTOCOLS = [
    "protocol-on-grid.yaml",
    "protocol-trench.yaml",
    "protocol-waffle.yaml",
]
UNSUPPORTED_LEGACY_PROTOCOLS = [
    "protocol-liftout.yaml",
    "protocol-serial-liftout.yaml",
]
TASK_PROTOCOLS = ["task-protocol.yaml", "task-protocol-waffle.yaml"]


# ── format detection ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", CONVERTIBLE_LEGACY_PROTOCOLS + UNSUPPORTED_LEGACY_PROTOCOLS)
def test_legacy_protocols_are_detected(filename: str):
    data = yaml.safe_load((LEGACY_PROTOCOL_DIR / filename).read_text())
    assert is_legacy_protocol(data)


@pytest.mark.parametrize("filename", TASK_PROTOCOLS)
def test_task_protocols_are_not_detected_as_legacy(filename: str):
    data = yaml.safe_load((TASK_PROTOCOL_DIR / filename).read_text())
    assert not is_legacy_protocol(data)


def test_saved_task_protocol_is_not_detected_as_legacy():
    """A round-tripped protocol must not be mistaken for the legacy format."""
    assert not is_legacy_protocol(AutoLamellaTaskProtocol().to_dict())


def test_is_legacy_protocol_ignores_non_dicts():
    assert not is_legacy_protocol(None)
    assert not is_legacy_protocol([])


def test_legacy_method_read_from_options():
    """Older protocols keep the method inside options rather than at top level."""
    data = yaml.safe_load((LEGACY_PROTOCOL_DIR / "protocol-waffle.yaml").read_text())
    assert "method" not in data
    assert get_legacy_protocol_method(data) is AutoLamellaMethod.WAFFLE


def test_legacy_method_defaults_to_on_grid_when_absent():
    assert get_legacy_protocol_method({"milling": {}}) is AutoLamellaMethod.ON_GRID


def test_legacy_method_unknown_returns_none():
    assert get_legacy_protocol_method({"method": "not-a-method"}) is None


# ── loading ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", CONVERTIBLE_LEGACY_PROTOCOLS)
def test_load_converts_legacy_protocol(filename: str):
    """The plain loader accepts a legacy file instead of raising TypeError."""
    protocol = AutoLamellaTaskProtocol.load(LEGACY_PROTOCOL_DIR / filename)

    assert isinstance(protocol, AutoLamellaTaskProtocol)
    assert len(protocol.task_config) > 0
    assert protocol.workflow_config.tasks
    # every workflow entry must have a matching task config
    for task in protocol.workflow_config.tasks:
        assert task.name in protocol.task_config


def test_load_legacy_protocol_matches_explicit_conversion():
    path = LEGACY_PROTOCOL_DIR / "protocol-waffle.yaml"
    auto = AutoLamellaTaskProtocol.load(path).to_dict()
    explicit = AutoLamellaTaskProtocol.load_from_old_protocol(path).to_dict()
    # ids are generated per protocol, everything else must match
    auto.pop("_id"), explicit.pop("_id")
    assert auto == explicit


@pytest.mark.parametrize("filename", UNSUPPORTED_LEGACY_PROTOCOLS)
def test_unsupported_legacy_methods_report_the_method(filename: str):
    """Liftout has no task equivalent -- say so, don't fail in the milling parser."""
    with pytest.raises(ValueError, match="not supported for conversion"):
        AutoLamellaTaskProtocol.load(LEGACY_PROTOCOL_DIR / filename)


@pytest.mark.parametrize("filename", TASK_PROTOCOLS)
def test_task_protocols_still_load(filename: str):
    protocol = AutoLamellaTaskProtocol.load(TASK_PROTOCOL_DIR / filename)
    assert len(protocol.task_config) > 0


def test_load_from_old_protocol_accepts_a_task_protocol():
    """Picking 'Select Legacy Protocol' for a new protocol is not an error."""
    protocol = AutoLamellaTaskProtocol.load_from_old_protocol(
        TASK_PROTOCOL_DIR / "task-protocol.yaml"
    )
    assert len(protocol.task_config) > 0


def test_converted_legacy_protocol_round_trips(tmp_path: Path):
    """A converted protocol saves in the new format and reloads unchanged."""
    protocol = AutoLamellaTaskProtocol.load(LEGACY_PROTOCOL_DIR / "protocol-on-grid.yaml")

    path = tmp_path / "protocol.yaml"
    protocol.save(str(path))

    assert not is_legacy_protocol(yaml.safe_load(path.read_text()))
    assert AutoLamellaTaskProtocol.load(str(path)).to_dict() == protocol.to_dict()


# ── unknown keys ──────────────────────────────────────────────────────────────

def test_workflow_options_ignore_unknown_keys():
    """Options written by another version load; the keys we know still apply."""
    options = AutoLamellaWorkflowOptions.from_dict(
        {
            "turn_beams_off": True,
            "alignment_at_milling_current": False,
            "method": "autolamella-on-grid",
        }
    )
    assert options.turn_beams_off is True


def test_workflow_config_ignores_unknown_keys():
    config = AutoLamellaWorkflowConfig.from_dict(
        {"name": "wf", "tasks": [], "some_removed_field": 3}
    )
    assert config.name == "wf"


def test_task_protocol_from_dict_ignores_unknown_option_keys():
    data = AutoLamellaTaskProtocol().to_dict()
    data["options"]["alignment_at_milling_current"] = True
    assert AutoLamellaTaskProtocol.from_dict(data).options.turn_beams_off is False


# ── experiment loading ────────────────────────────────────────────────────────

def test_experiment_loads_alongside_a_legacy_protocol(tmp_path: Path):
    """An experiment saved with a legacy protocol.yaml opens with a protocol."""
    experiment = Experiment(path=str(tmp_path), name="legacy-experiment")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.save()

    legacy = (LEGACY_PROTOCOL_DIR / "protocol-on-grid.yaml").read_text()
    (Path(experiment.path) / "protocol.yaml").write_text(legacy)

    loaded = Experiment.load(Path(experiment.path) / "experiment.yaml")
    assert loaded.task_protocol is not None
    assert len(loaded.task_protocol.task_config) > 0
