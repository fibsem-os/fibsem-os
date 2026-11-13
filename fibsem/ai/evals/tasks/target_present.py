"""Target present: is the FM target still in the lamella, pass by pass."""

from typing import Any, Dict, List

import numpy as np

from fibsem.ai.evals.cases import Case, pass_names
from fibsem.ai.evals.scoring import InvalidAnswer, yes_no
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.visual import _pass_strip


def score_target_present(
    case: Case, answer: Dict[str, Any], key: Dict[str, Any]
) -> tuple:
    present = answer.get("present")
    if not isinstance(present, dict):
        raise InvalidAnswer("no present map")
    wrong, missed_loss, unanswered = [], [], []
    for name, expected in key.items():
        if name not in present:
            unanswered.append(name)
            continue
        said = yes_no(present[name])
        if said != yes_no(expected):
            wrong.append(name)
            if said is True and yes_no(expected) is False:
                missed_loss.append(name)
    details = {
        "correct": len(key) - len(wrong) - len(unanswered),
        "total": len(key),
        "wrong": wrong,
        "unanswered": unanswered,
        "missed_loss": missed_loss,
        "costly": len(missed_loss),
    }
    return ("hit" if not wrong and not unanswered else "miss"), details


def draw_target_present(case: Case, runs: List[Run], key=None, part=None):
    key = key or case.key
    names = pass_names(case)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(max(8, 1.1 * len(names)), 2.5 + 0.4 * len(runs)))
    grid = fig.add_gridspec(2, len(names), height_ratios=(1.4, 0.4 * (len(runs) + 1)))
    _pass_strip(case, [fig.add_subplot(grid[0, i]) for i in range(len(names))])
    ax = fig.add_subplot(grid[1, :])
    rows = [("key", key)]
    for result, answer in runs:
        answer = (
            answer if part is None else ((answer or {}).get("parts") or {}).get(part)
        )
        rows.append((f"r{result['repeat']}", (answer or {}).get("present") or {}))
    for r, (label, said) in enumerate(rows):
        for c, name in enumerate(names):
            if name not in said:
                continue
            try:
                value = yes_no(said[name])
            except InvalidAnswer:
                value = None
            want = yes_no(key[name]) if name in key else None
            colour = "#bbbbbb" if value is None else ("#2ca02c" if value else "#444444")
            ax.add_patch(plt.Rectangle((c, r), 0.9, 0.9, color=colour))
            text = {True: "yes", False: "no", None: "?"}[value]
            ax.text(
                c + 0.45,
                r + 0.45,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="red" if r and want is not None and value != want else "white",
            )
    ax.set_xlim(0, len(names))
    ax.set_ylim(len(rows), 0)
    ax.set_yticks(np.arange(len(rows)) + 0.45, [label for label, _ in rows])
    ax.set_xticks([])
    ax.set_title("target present (red text: differs from the key)", fontsize=9)
    return fig


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    unknown = set(key) - set(pass_names(case))
    return [f"key names unknown passes {sorted(unknown)}"] if unknown else []


def summary(d: Dict[str, Any]) -> str:
    return f"{d['correct']} of {d['total']} passes right" if "total" in d else ""


TASK = EvalTask(
    name="target_present",
    title="Say if the target is still in the lamella, pass by pass",
    skill="autolamella-coincidence-milling",
    answer_shape={"present": {"<pass>": "yes | no"}, "reason": "..."},
    score=score_target_present,
    check_key=check_key,
    summary=summary,
    draw=draw_target_present,
    sims=(Sim("coincidence", "fibsem.ai.evals.sim_coincidence"),),
)
