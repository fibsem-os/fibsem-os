"""Simulate and print the multi-grid lamella execution schedule.

Builds an in-memory experiment (grids + lamellae, no microscope), runs the
load-aware planner, and prints the schedule as a text table per grid load.

Examples:
    python scripts/simulate_grid_schedule.py
    python scripts/simulate_grid_schedule.py --grids A:3,B:2,C:3 --policy both
    python scripts/simulate_grid_schedule.py --grids A:4,B:4 --capacity 2

Design doc: docs/design/multi-grid-lamella-execution.md
"""

import argparse
import tempfile
from typing import Dict, List, Optional

from fibsem.applications.autolamella.structures import (
    Experiment,
    GridRecord,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.scheduling import (
    GRID_GREEDY,
    TASK_GREEDY,
    Plan,
    build_plan,
)


def build_experiment(grids_spec: Dict[str, int]) -> Experiment:
    """Create an in-memory experiment: one GridRecord per grid, N lamellae each."""
    exp = Experiment.create(path=tempfile.mkdtemp(), name="schedule-sim")
    for gname, n in grids_spec.items():
        rec = GridRecord(name=gname)
        exp.add_grid(rec)
        for i in range(n):
            lam = Lamella(petname=f"{gname}{i + 1}",
                          path=f"{exp.path}/{gname}{i + 1}",
                          number=i + 1, grid_id=rec._id)
            exp.positions.append(lam)
    return exp


def _blocks(plan: Plan) -> List[dict]:
    """Segment the event stream into per-load blocks (a grid + its work)."""
    blocks: List[dict] = []
    cur: Optional[dict] = None
    step = 0
    for e in plan.events:
        if e.kind in ("load", "exchange"):
            cur = {"kind": e.kind, "grid": e.grid, "works": []}
            blocks.append(cur)
        elif e.kind == "work":
            step += 1
            if cur is None or cur["grid"] != e.grid:  # work on a pre-loaded grid
                cur = {"kind": "load", "grid": e.grid, "works": []}
                blocks.append(cur)
            cur["works"].append((step, e.task, e.lamella))
    return blocks


def print_schedule(plan: Plan, task_names: List[str], policy: str) -> None:
    """Print the schedule as a per-grid task x lamella table, step-numbered."""
    order = " -> ".join(str(g) for g in plan.grid_order) or "(none)"
    print(f"\ngrid schedule — {policy}")
    print(f"  {plan.n_work} work · {plan.n_exchanges} exchanges · "
          f"{plan.n_realigns} realigns · {plan.n_skipped} skipped · order {order}")
    for lam, task, reason in plan.skipped:
        print(f"    skipped: {task} on {lam} ({reason})")
    print()

    blocks = _blocks(plan)
    names = {lam for b in blocks for (_, _, lam) in b["works"]}
    cw = max(2, len(str(plan.n_work)), max((len(n) for n in names), default=2)) + 1
    indent = 5

    for b in blocks:
        lams: List[str] = []
        for _, _, lam in b["works"]:
            if lam not in lams:
                lams.append(lam)
        tasks = [t for t in task_names if any(w[1] == t for w in b["works"])]
        step_of = {(t, lam): s for s, t, lam in b["works"]}
        gut = max((len(t) for t in tasks), default=0)

        head = (f"exchange -> grid {b['grid']}" if b["kind"] == "exchange"
                else f"load grid {b['grid']}")
        print(f"  {head}")
        print(f"    realign {b['grid']}")
        print(" " * (indent + gut) + "".join(f"{lam:>{cw}}" for lam in lams))
        for t in tasks:
            cells = "".join(f"{step_of.get((t, lam), ''):>{cw}}" for lam in lams)
            print(" " * indent + f"{t:<{gut}}" + cells)
        print()


def _parse_grids(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in s.split(","):
        name, _, n = part.partition(":")
        out[name.strip()] = int(n) if n else 1
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grids", default="A:3,B:2,C:3", help="e.g. A:3,B:2,C:3")
    p.add_argument("--tasks", default="mill_trench,rough_mill,polish")
    p.add_argument("--policy", default="grid_greedy",
                   choices=["grid_greedy", "task_greedy", "both"])
    p.add_argument("--capacity", type=int, default=1, help="working slots (1=autoloader)")
    p.add_argument("--loaded", default="", help="grid names already loaded, comma-sep")
    args = p.parse_args()

    grids_spec = _parse_grids(args.grids)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    exp = build_experiment(grids_spec)
    loaded_ids = [g._id for g in exp.grids if g.name in
                  {n.strip() for n in args.loaded.split(",") if n.strip()}]

    policies = [GRID_GREEDY, TASK_GREEDY] if args.policy == "both" else [args.policy]
    for pol in policies:
        plan = build_plan(exp, tasks, loaded_ids=loaded_ids,
                          capacity=args.capacity, policy=pol)
        print_schedule(plan, tasks, pol)


if __name__ == "__main__":
    main()
