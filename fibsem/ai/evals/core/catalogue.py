"""The case catalogue: every task a case can be cut from a real run for, as
the segmenter reads it. It is made from the task registry, so a new task
with ``from_run`` is in it without being listed anywhere."""

from typing import Any, Dict, List

from fibsem.ai.evals import tasks

FORMAT = 0
# What a decision no task scores yet is answered with, so a segment with no
# task still says what a case for it would score.
ANSWER_SHAPES = {
    "choice": "one of a few named options (a milling current, a pattern)",
    "yes_no": "yes or no, with a reason",
    "point": "a point on an image (x, y px)",
    "number": "a number with a unit (a thickness, a depth)",
    "text": "free text a person judges",
}


def entries() -> List[Dict[str, Any]]:
    """One entry per task with ``from_run``, by name."""
    found = []
    for name in tasks.names():
        task = tasks.get(name)
        if task is None or task.from_run is None:
            continue
        entry: Dict[str, Any] = {
            "task": name,
            "title": task.title,
            "skill": task.skill,
            "decision": task.from_run.decision,
            "inputs": dict(task.from_run.inputs),
            "answer_shape": task.answer_shape,
            "can_cut": task.from_run.cut is not None,
        }
        if task.answer_notes:
            entry["answer_notes"] = task.answer_notes
        found.append(entry)
    return found


def catalogue() -> Dict[str, Any]:
    return {
        "format": FORMAT,
        "tasks": entries(),
        "no_task_yet": {
            "note": "a decision worth scoring that no task above covers: give "
            "it a short name, the question, and one of these answer shapes",
            "answer_shapes": dict(ANSWER_SHAPES),
        },
    }
