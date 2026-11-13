"""What an eval task is: everything the evals need to know about one task, in
one place. Each module in this package defines one, as ``TASK``."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# (result, answer) of one repeat.
Run = Tuple[Dict[str, Any], Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class Sim:
    """A simulator generator for the task: ``make-sim --task <name>`` runs
    ``write_case(seed, out, overrides)`` from *module*."""

    name: str
    module: str


@dataclass(frozen=True)
class EvalTask:
    name: str
    # The review page's one-line description.
    title: str
    # The skill the task is decided by.
    skill: str
    # What the agent is shown as the answer's shape, and what its fields mean
    # where the shape leaves it open.
    answer_shape: Dict[str, Any]
    # score(case, answer, key) -> (result, details); raises InvalidAnswer.
    score: Callable[..., tuple]
    # check_key(case, key) -> what is missing or wrong in the key.
    check_key: Callable[..., List[str]]
    # summary(details) -> what one repeat did, in a few words.
    summary: Callable[[Dict[str, Any]], str]
    # draw(case, runs[, key=, part=]) -> a matplotlib figure.
    draw: Callable[..., Any]
    answer_notes: str = ""
    # key_text(key) -> the key in a few words.
    key_text: Optional[Callable[[Dict[str, Any]], str]] = None
    # disagreements(case, runs) -> what the repeats said that the key doesn't
    # accept; a task with them can be settled on the review page.
    disagreements: Optional[Callable[..., List[Dict[str, Any]]]] = None
    # pixel_size_um(case) -> the answer image's pixel size.
    pixel_size_um: Optional[Callable[..., Optional[float]]] = None
    # key_point(key) / set_point(key, x, y): the key's target, for a task
    # answered with one point.
    key_point: Optional[Callable[..., Optional[Tuple[float, float]]]] = None
    set_point: Optional[Callable[..., None]] = None
    sims: Tuple[Sim, ...] = field(default_factory=tuple)
