"""The lamella run's plan: what the Workflow tab's confirmations show, served
at /app/workflow/plan with the same numbers, for a fresh run and for adding to
the running queue."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fibsem.applications.autolamella.workflows.estimate import (  # noqa: E402
    estimate_lamella_run,
)
from fibsem.applications.autolamella.workflows.plan import (  # noqa: E402
    plan_lamella_run,
)
from fibsem.workflows.runner import ITEMS, RunRequest  # noqa: E402
from tests.autolamella.test_select_position_asks import (  # noqa: E402, F401
    SETUP,
    microscope,
)
from tests.server.test_a_run_with_no_window import _held, _ok, agent  # noqa: E402, F401


def test_a_run_s_plan_is_the_confirmation_s_estimate(agent):  # noqa: F811
    experiment = agent.session.experiment
    names = [p.name for p in experiment.positions]

    plan = _ok(agent.post("/app/workflow/plan", json={"task_names": [SETUP]}))

    assert plan["valid"] and not plan["add"]
    assert [(s["item"], s["task"]) for s in plan["steps"]] == [
        (name, SETUP) for name in names
    ]
    direct = estimate_lamella_run(experiment, [(name, SETUP) for name in names])
    assert plan["estimate"]["work_seconds"] == pytest.approx(direct.work_seconds)
    assert [t["task_name"] for t in plan["estimate"]["tasks"]] == [SETUP]
    assert plan["addition"] is None


def test_a_refused_run_s_plan_says_why(agent):  # noqa: F811
    plan = _ok(agent.post("/app/workflow/plan", json={"task_names": ["Nope"]}))

    assert not plan["valid"]
    assert "unknown tasks" in plan["reason"]
    assert SETUP in plan["task_names"]
    assert agent.post("/app/workflow/plan", json={}).status_code == 422


def test_adding_needs_a_running_queue(agent):  # noqa: F811
    plan = _ok(
        agent.post("/app/workflow/plan", json={"task_names": [SETUP], "add": True})
    )
    assert not plan["valid"] and "no workflow is running" in plan["reason"]


def test_an_addition_s_plan_is_its_confirmation_s(agent):  # noqa: F811
    session = agent.session
    _ok(agent.post("/app/workflow/start", json={"task_names": [SETUP]}))
    assert _held(agent) is not None
    name = session.experiment.positions[0].name

    plan = _ok(
        agent.post(
            "/app/workflow/plan",
            json={"task_names": [SETUP], "item_names": [name], "add": True},
        )
    )

    assert plan["valid"] and plan["add"]
    assert plan["steps"] == [{"item": name, "task": SETUP}]
    direct = plan_lamella_run(
        session.runner,
        session.experiment,
        RunRequest(kind=ITEMS, task_names=(SETUP,), names=(name,)),
        add=True,
    )
    assert plan["addition"]["work_seconds"] == pytest.approx(
        direct.addition.work_seconds
    )
    assert plan["estimate"] is None
    assert session.runner.manager.queue.pending == []
