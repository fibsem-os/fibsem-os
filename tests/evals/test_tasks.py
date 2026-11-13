"""Every eval task is one object in one module: each fills every part the
evals need, so a new task can't be half added."""

import inspect

import pytest

from fibsem.ai.evals import tasks
from fibsem.ai.evals.tasks import EvalTask

NAMES = tasks.names()


def test_the_tasks_are_found():
    assert {"site_selection", "polishing_stop", "rough_milling"} <= set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_a_task_fills_its_parts(name):
    task = tasks.get(name)
    assert isinstance(task, EvalTask) and task.name == name
    assert task.title and task.skill
    assert isinstance(task.answer_shape, dict) and task.answer_shape
    for part in ("score", "check_key", "summary", "draw"):
        assert callable(getattr(task, part)), part
    # A summary of an empty score says nothing rather than failing.
    assert task.summary({}) == ""


@pytest.mark.parametrize("name", NAMES)
def test_a_point_task_both_reads_and_sets_its_point(name):
    task = tasks.get(name)
    assert (task.key_point is None) == (task.set_point is None)
    if task.key_point:
        key = {}
        task.set_point(key, 12.5, 7.0)
        assert task.key_point(key) == (12.5, 7.0)


def test_each_sim_generator_is_named_once_and_has_a_writer():
    sims = [sim for task in tasks.all_tasks().values() for sim in task.sims]
    assert len({sim.name for sim in sims}) == len(sims)
    for sim in sims:
        module = __import__(sim.module, fromlist=["write_case"])
        assert "seed" in inspect.signature(module.write_case).parameters
