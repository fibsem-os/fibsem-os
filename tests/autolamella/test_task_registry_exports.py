"""The task registry's public names exist, and register_task reaches get_tasks()."""

import fibsem.applications.autolamella.workflows.tasks as tasks
from fibsem.applications.autolamella.workflows.tasks import (
    MillRoughTask,
    MillRoughTaskConfig,
)


def test_every_exported_name_exists():
    assert [name for name in tasks.__all__ if not hasattr(tasks, name)] == []


def test_a_registered_task_is_listed_and_named():
    class _Config(MillRoughTaskConfig):
        task_type = "EXPORTS_TEST_TASK"

    class _Task(MillRoughTask):
        config_cls = _Config

    tasks.register_task(_Task)
    try:
        assert tasks.get_tasks()["EXPORTS_TEST_TASK"] is _Task
        assert "EXPORTS_TEST_TASK" in tasks.get_task_names()
        assert tasks.REGISTERED_TASKS["EXPORTS_TEST_TASK"] is _Task
    finally:
        tasks.TASK_PLUGINS.registered.pop("EXPORTS_TEST_TASK", None)


def test_plugin_records_come_from_the_registry():
    assert tasks.get_task_plugin_records() == tasks.TASK_PLUGINS.plugin_records()
