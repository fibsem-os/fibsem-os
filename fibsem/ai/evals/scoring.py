"""Scoring an agent's answer against a case's key, in code.

One scorer per task. Each returns a Score: a result (hit, miss, early, late
or invalid) and the numbers behind it. A mistake that would cost a lamella or
a site (a bad site picked, a target milled away) is a miss like any other;
``details["costly"]`` counts them so a report can call them out.

Each task's scorer and answer shape live in ``fibsem.ai.evals.tasks``.
A case with parts is answered as ``{"parts": {"a": {...}, "b": {...}}}``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fibsem.ai.evals import tasks
from fibsem.ai.evals.cases import (
    Case,
)

RESULTS = ("hit", "miss", "early", "late", "invalid")


@dataclass
class Score:
    case: str
    task: str
    result: str
    part: Optional[str] = None
    headline: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def costly(self) -> int:
        return int(self.details.get("costly") or 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "part": self.part,
            "task": self.task,
            "result": self.result,
            "headline": self.headline,
            "details": self.details,
        }


class InvalidAnswer(ValueError):
    pass


def yes_no(value: Any) -> Optional[bool]:
    """yes/no answers as True/False; unsure as None."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "y"):
        return True
    if text in ("no", "false", "n"):
        return False
    if text in ("unsure", "maybe", "unknown"):
        return None
    raise InvalidAnswer(f"not yes, no or unsure: {value!r}")


def _point(answer: Dict[str, Any]) -> tuple:
    try:
        return float(answer["x_px"]), float(answer["y_px"])
    except (KeyError, TypeError, ValueError) as e:
        raise InvalidAnswer(f"no x_px, y_px: {e}") from e


def _score_one(case, task, answer, key, part=None, headline=True) -> Score:
    try:
        result, details = tasks.get(task).score(case, answer or {}, key)
    except InvalidAnswer as e:
        result, details = "invalid", {"error": str(e)}
    return Score(
        case=case.id,
        task=task,
        result=result,
        part=part,
        headline=headline,
        details=details,
    )


def score_case(case: Case, answer: Dict[str, Any]) -> List[Score]:
    """One Score per part, or one for a case without parts."""
    if not case.parts:
        return [_score_one(case, case.task, answer, case.key)]
    answers = answer.get("parts") or {}
    scores = []
    for name, part in case.parts.items():
        key = part.get("key") if isinstance(part.get("key"), dict) else case.key
        scores.append(
            _score_one(
                case,
                part["task"],
                answers.get(name),
                key,
                part=name,
                headline=bool(part.get("headline", True)),
            )
        )
    return scores


def score_live(case: Case) -> Optional[List[Score]]:
    """The scores of what the real run decided (the case's ``live`` block),
    for the parts it decided; None when the case records no live decision."""
    live = case.data.get("live")
    if not isinstance(live, dict) or not isinstance(live.get("answer"), dict):
        return None
    scores = score_case(case, live["answer"])
    decided = live.get("parts")
    if decided:
        scores = [s for s in scores if s.part in decided]
    return scores
