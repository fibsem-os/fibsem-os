"""A live case's part in a run (``fibsem.ai.evals.runner``): setting the
run up as the operator would before the agent starts, and recording what
the run did once it ends, to score."""

import sys
from pathlib import Path
from typing import Any, Dict, List

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.live.client import LiveClient

# The run's record, beside its result: what the experiment holds at the end.
RECORD = "record.json"
# Which case's task a scripted agent runs.
TASK_ENV = "FIBSEM_EVAL_TASK"


def scripted_command() -> List[str]:
    """The scripted agent, run in fibsem-agent's place (``run --scripted``)."""
    return [sys.executable, "-m", "fibsem.ai.evals.live.scripted"]


def set_up(case: Case, found_at: Path, home: Path) -> None:
    """The case's ``sim_run`` set-up, as the operator: the agent's level,
    then each tool call in ``setup`` (a grid loaded, the stage moved)."""
    sim_run = case.data["sim_run"]
    with LiveClient.as_operator(found_at, home) as operator:
        operator.set_level(sim_run.get("level", "watch"))
        for step in sim_run.get("setup") or []:
            operator.call(step["tool"], **(step.get("arguments") or {}))


def record(found_at: Path, home: Path) -> Dict[str, Any]:
    """What the run left in its experiment: each item's name, id and the
    stage position of each of its poses."""
    items = []
    with LiveClient.as_operator(found_at, home) as operator:
        listed = operator.call("item_list") or {}
        for row in listed.get("items") or []:
            name = row.get("item_name")
            detail = operator.call("item_get", item_name=name) or {}
            items.append(
                {
                    "name": name,
                    "id": row.get("item_id"),
                    "poses": detail.get("poses") or {},
                }
            )
    return {"items": items}
