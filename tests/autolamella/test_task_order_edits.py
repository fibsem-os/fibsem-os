"""Task order and requirements: one function each, which the GUI and the App
Server both call (FIB-1446)."""

from dataclasses import dataclass
from typing import ClassVar

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)
from fibsem.workflows.grid.base import GridTaskConfig
from fibsem.workflows.grid.protocol import GridTaskProtocol


def _workflow():
    return AutoLamellaWorkflowConfig(
        tasks=[
            AutoLamellaTaskDescription(name="Trench"),
            AutoLamellaTaskDescription(name="Undercut", requires=["Trench"]),
            AutoLamellaTaskDescription(name="Polish"),
        ]
    )


def test_a_reorder_keeps_requirements_first():
    config = _workflow()

    why = config.reorder(["Undercut", "Trench", "Polish"])

    assert "Undercut" in why and "Trench" in why
    assert config.workflow == ["Trench", "Undercut", "Polish"]
    assert config.reorder(["Trench", "Polish", "Undercut"]) == ""
    assert config.workflow == ["Trench", "Polish", "Undercut"]


def test_a_reorder_names_every_task_once():
    config = _workflow()
    assert config.reorder(["Trench", "Undercut"]) != ""
    assert config.workflow == ["Trench", "Undercut", "Polish"]


def test_a_task_can_only_require_earlier_tasks():
    config = _workflow()

    assert "not Polish" in config.set_requirements("Undercut", ["Polish"])
    assert config.requirements("Undercut") == ["Trench"]
    assert config.set_requirements("Polish", ["Trench"], required=False) == ""
    assert config.requirements("Polish") == ["Trench"]
    assert config.tasks[2].required is False
    assert config.set_requirements("Nope", []) != ""


@dataclass
class _Config(GridTaskConfig):
    task_type: ClassVar[str] = "ORDER_TEST_GRID"


def _grid_protocol():
    protocol = GridTaskProtocol()
    protocol.add(_Config(task_name="overview"))
    protocol.add(_Config(task_name="screen"))
    return protocol


def test_a_grid_task_requires_others_in_run_order():
    protocol = _grid_protocol()
    first, second = protocol.ordered_task_names

    assert protocol.set_requires(second, [first]) == ""
    assert protocol.requirements(second) == [first]
    assert protocol.set_requires(second, [second, "Nope"]) != ""
    assert protocol.requirements(second) == [first]


def test_a_grid_reorder_names_every_task_once():
    protocol = _grid_protocol()
    first, second = protocol.ordered_task_names

    assert protocol.reorder([second]) != ""
    assert protocol.reorder([second, first]) == ""
    assert protocol.ordered_task_names == [second, first]
