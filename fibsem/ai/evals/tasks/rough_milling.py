"""Rough milling: whether each step milled through and the lamella held."""

from typing import Any, Dict, List

import numpy as np

from fibsem.ai.evals.cases import Case, pass_names
from fibsem.ai.evals.scoring import InvalidAnswer, yes_no
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import _pass_strip

# What a rough_milling key says about each step.
ROUGH_CHECKS = ("milled_through", "intact")


def score_rough_milling(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    """A hit when every step's two answers match the key. Calling a step
    through, or the lamella intact, when it is not is costly: the run would
    carry on from a lamella that needs redoing or is lost."""
    names = pass_names(case)
    wrong, costly, total = [], 0, 0
    for check in ROUGH_CHECKS:
        said = answer.get(check)
        if not isinstance(said, dict):
            raise InvalidAnswer(f"no {check} map")
        for name in names:
            total += 1
            want = yes_no(key[check][name])
            got = yes_no(said[name]) if name in said else None
            if got != want:
                wrong.append(f"{name} {check}")
                costly += int(got is True and want is False)
    details = {"correct": total - len(wrong), "total": total}
    if wrong:
        details["wrong"] = wrong
    if costly:
        details["costly"] = costly
    return ("miss" if wrong else "hit"), details


def draw_rough_milling(case: Case, runs: List[Run], key=None, part=None):
    """The steps' FIB images, then a yes/no grid per check: the key's row,
    then each repeat's (red text where it differs from the key)."""
    key = key or case.key
    names = pass_names(case)
    import matplotlib.pyplot as plt

    rows = [("key", key)]
    for result, answer in runs:
        answer = (
            answer if part is None else ((answer or {}).get("parts") or {}).get(part)
        )
        rows.append((f"r{result['repeat']}", answer or {}))
    cols = [(check, name) for check in ROUGH_CHECKS for name in names]
    fig = plt.figure(figsize=(max(8, 1.1 * len(cols)), 3 + 0.4 * len(rows)))
    grid = fig.add_gridspec(2, len(names), height_ratios=(1.6, 0.4 * (len(rows) + 1)))
    _pass_strip(case, [fig.add_subplot(grid[0, i]) for i in range(len(names))])
    ax = fig.add_subplot(grid[1, :])
    for r, (label, said) in enumerate(rows):
        for c, (check, name) in enumerate(cols):
            value = (said.get(check) or {}).get(name)
            try:
                value = yes_no(value) if value is not None else None
            except InvalidAnswer:
                value = None
            want = yes_no(key[check][name])
            colour = "#bbbbbb" if value is None else ("#2ca02c" if value else "#444444")
            ax.add_patch(plt.Rectangle((c, r), 0.9, 0.9, color=colour))
            ax.text(
                c + 0.45,
                r + 0.45,
                {True: "yes", False: "no", None: "?"}[value],
                ha="center",
                va="center",
                fontsize=7,
                color="red" if r and value != want else "white",
            )
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_yticks(np.arange(len(rows)) + 0.45, [label for label, _ in rows])
    ax.set_xticks(
        np.arange(len(cols)) + 0.45,
        [f"{name}\n{check.replace('_', ' ')}" for check, name in cols],
        fontsize=6,
    )
    return fig


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    names = set(pass_names(case))
    return [
        f"key {check} does not name every step"
        for check in ROUGH_CHECKS
        if set(key.get(check) or {}) != names
    ]


def summary(d: Dict[str, Any]) -> str:
    return f"{d['correct']} of {d['total']} step checks right" if "total" in d else ""


TASK = EvalTask(
    name="rough_milling",
    title="Say if each rough milling step went through and the lamella held",
    skill="autolamella",
    answer_shape={
        "milled_through": {"<step>": "yes | no"},
        "intact": {"<step>": "yes | no"},
        "reason": "...",
    },
    score=score_rough_milling,
    check_key=check_key,
    summary=summary,
    draw=draw_rough_milling,
    sims=(Sim("rough", "fibsem.ai.evals.sim_rough"),),
)
