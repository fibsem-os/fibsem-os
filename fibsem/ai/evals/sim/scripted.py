"""A scripted agent for live cases: fixed steps and simple image rules, no
model, so a live case runs end to end without spending anything. It is a
baseline, not a good agent: its score is what a naive one gets, and a run of
it checks the case, the runner and the scoring.

Run as the runner runs fibsem-agent, with the same flags (those it has no
use for are ignored)::

    python -m fibsem.ai.evals.sim.scripted --files <work> \
        [--task live_site_selection] [--session-id .. --goal .. ..]

It finds its run's App Server by ``FIBSEM_DISCOVERY_FILE`` and its task by
``FIBSEM_EVAL_TASK`` (or ``--task``), and writes its answer to
``<work>/answer.json``.
"""

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from fibsem.ai.evals import tasks
from fibsem.ai.evals.sim.live_client import LiveClient
from fibsem.ai.evals.sim.live_runs import CASE_ENV, TASK_ENV
from fibsem.server import discovery

log = logging.getLogger(__name__)

NAME = "fibsem-scripted-eval/1"
ANSWER = "answer.json"

# shared by the tasks' baselines
SITE_HFW = 80e-6  # m, the image at each site
COINCIDENCE_TRIES = 3
COINCIDENT_M = 0.3e-6  # a check this near the centre is coincident


def _sim_run() -> Dict[str, Any]:
    """The case's ``sim_run`` block (never its key), when the runner names
    the case."""
    import yaml

    folder = os.environ.get(CASE_ENV)
    if not folder:
        return {}
    data = yaml.safe_load((Path(folder) / "case.yaml").read_text()) or {}
    return data.get("sim_run") or {}


def _image_path(answer: Any) -> str:
    """The saved file of an image command's answer."""
    image = (answer or {}).get("image", answer) or {}
    return image.get("path") or (image.get("summary") or {}).get("path")


def _read(path: str) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    return image if image.ndim == 2 else image[..., 0]


def make_coincident(client: LiveClient, tries: int = COINCIDENCE_TRIES) -> float:
    """Bring the FIB onto the point at the SEM's centre: check, then move
    the stage along the vertical by what the check measured, until it is
    within COINCIDENT_M. The last offset measured (m)."""
    dy = float("inf")
    for _ in range(tries):
        check = client.call("alignment_check_coincidence", hfw=SITE_HFW)
        found = (check.get("summary") or {}).get("metadata") or check
        dy = float(found["dy"])
        log.info(
            "coincidence: dy %.2f um, dz %.2f um, score %.3f, agrees %s",
            dy * 1e6,
            float(found.get("dz", 0)) * 1e6,
            float(found.get("score", 0)),
            found.get("agrees"),
        )
        if abs(dy) <= COINCIDENT_M:
            break
        client.call("stage_vertical_move", dy=dy, beam_type="ION")
    return dy


def agents() -> Dict[str, Callable[[LiveClient, Path], Dict[str, Any]]]:
    """Each task's scripted baseline, by task name: ``EvalTask.scripted``."""
    found = {}
    for name, task in tasks.all_tasks().items():
        if task.scripted:
            module, _, function = task.scripted.partition(":")
            found[name] = getattr(importlib.import_module(module), function)
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task", default=os.environ.get(TASK_ENV), choices=sorted(agents())
    )
    parser.add_argument("--files", "--code", dest="work", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO)
    found_at = Path(os.environ[discovery.DISCOVERY_FILE_ENV])
    with LiveClient.as_agent(found_at, NAME) as client:
        answer = agents()[args.task](client, args.work)
    (args.work / ANSWER).write_text(json.dumps(answer, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
