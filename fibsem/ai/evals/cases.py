"""Loading and checking eval cases (``case.yaml``, format 0)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

from fibsem.ai.evals import tasks

FORMAT = 0
FILE = "case.yaml"
# Where a case's images come from: a real session, or the simulator. Sim
# cases are scored apart and never count towards the real scores.
TIERS = ("real", "sim")
# What a square is, in a square_quality case. A key may accept more than one.
SQUARE_LABELS = (
    "good",
    "no_cells",
    "damaged",
    "thick",
    "cell_near_bar",
    "cell_too_big",
)


# A square_quality square is a good or bad site, with reasons beside it (a
# good site can have some): the old single label is still read.
SITES = ("good", "bad")
SQUARE_REASONS = tuple(label for label in SQUARE_LABELS if label != "good")


def _word(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def square_call(value: Any) -> Dict[str, Any]:
    """A square's key entry or answer as ``{"sites": [...], "reasons": [...]}``:
    ``sites`` holds every site call that is right (a key may accept both).
    ``{"site", "reasons"}`` is read as it is; an old label, or list of
    labels, maps good to a good site and anything else to a bad site with
    that label as its reason."""
    if isinstance(value, dict) and "site" in value:
        site = value["site"]
        sites = [_word(s) for s in (site if isinstance(site, list) else [site])]
        reasons = value.get("reasons") or []
        reasons = [
            _word(r) for r in (reasons if isinstance(reasons, list) else [reasons])
        ]
        return {"sites": sites, "reasons": reasons}
    if isinstance(value, dict):
        value = value.get("label")
    labels = [_word(v) for v in (value if isinstance(value, list) else [value])]
    sites = sorted({"good" if label == "good" else "bad" for label in labels})
    return {"sites": sites, "reasons": [label for label in labels if label != "good"]}


@dataclass
class Case:
    path: Path
    data: Dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data.get("id") or self.path.name)

    @property
    def task(self) -> str:
        return str(self.data.get("task", ""))

    @property
    def key(self) -> Dict[str, Any]:
        return self.data.get("key") or {}

    @property
    def scoring(self) -> Dict[str, Any]:
        return self.data.get("scoring") or {}

    @property
    def parts(self) -> Dict[str, Dict[str, Any]]:
        return self.data.get("parts") or {}

    @property
    def tier(self) -> str:
        return str(self.data.get("tier") or "real")

    @property
    def status(self) -> str:
        """keyed, parked or unkeyed."""
        status = str(self.key.get("status", "")).lower()
        if status.startswith("parked"):
            return "parked"
        if status.startswith("todo") or not self.key:
            return "unkeyed"
        if self.task == "site_selection" and not self.key.get("good"):
            return "unkeyed"
        return "keyed"


@dataclass
class CaseCheck:
    case: Case
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def load_case(path: Path) -> Case:
    path = Path(path)
    folder = path.parent if path.name == FILE else path
    with open(folder / FILE) as f:
        data = yaml.safe_load(f) or {}
    return Case(path=folder, data=data)


def find_cases(root: Path) -> Iterator[Case]:
    """Every case under ``root``, in path order."""
    for path in sorted(Path(root).rglob(FILE)):
        yield load_case(path)


def _input_paths(value: Any) -> Iterator[str]:
    """Strings in a case that name an input file."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _input_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _input_paths(item)
    elif isinstance(value, str) and value.startswith(("experiment/", "../")):
        yield value


def input_paths(case: Case) -> List[str]:
    """The input files a case names, relative to its folder: its own under
    experiment/, or another case's (``../``)."""
    keys = ("inputs", "stack", "passes", "views", "inputs_extra")
    return list(_input_paths({k: case.data.get(k) for k in keys}))


def check_case(case: Case) -> CaseCheck:
    """What is missing or wrong in a case. Parked cases are only read."""
    check = CaseCheck(case)
    data = case.data
    for name in ("id", "task", "brief"):
        if not data.get(name):
            check.problems.append(f"no {name}")
    if data.get("format") != FORMAT:
        check.problems.append(f"format {data.get('format')!r}, expected {FORMAT}")
    if tasks.get(case.task) is None:
        check.problems.append(f"unknown task {case.task!r}")
    if case.tier not in TIERS:
        check.problems.append(f"unknown tier {case.tier!r}")
    for name, part in case.parts.items():
        if tasks.get(part.get("task")) is None:
            check.problems.append(f"part {name}: unknown task {part.get('task')!r}")
    playbook = data.get("playbook")
    if playbook is not None and not isinstance(playbook, str):
        check.problems.append("playbook is the name of one playbook")
    if case.status == "parked":
        return check
    for rel in input_paths(case):
        if not (case.path / rel).exists():
            check.problems.append(f"missing input {rel}")
    if case.status == "keyed":
        check.problems.extend(_key_problems(case))
    return check


def _key_problems(case: Case) -> List[str]:
    """What is missing or wrong in the case's key, and in each part's own."""
    task = tasks.get(case.task)
    problems = list(task.check_key(case, case.key)) if task else []
    for name, part in case.parts.items():
        if isinstance(part.get("key"), dict) and tasks.get(part.get("task")):
            check = tasks.get(part["task"]).check_key(case, part["key"])
            problems += [f"part {name}: {problem}" for problem in check]
    return problems


def bad_mask(key: Dict[str, Any]) -> Optional[str]:
    """A setup placement key's mask of costly placements (``unsafe_mask`` in
    keys written before the rename)."""
    return key.get("bad_mask") or key.get("unsafe_mask")


def stop_passes(key: Dict[str, Any]) -> List[str]:
    """The passes after which stopping is right: ``stop_after`` names one, or
    a list of them when any in that span will do."""
    stop = key.get("stop_after")
    if stop is None:
        return []
    return [str(s) for s in stop] if isinstance(stop, list) else [str(stop)]


def pass_names(case: Case) -> List[str]:
    return [p.get("name") for p in case.data.get("passes") or []]


def pixel_size_um(case: Case) -> Optional[float]:
    task = tasks.get(case.task)
    return task.pixel_size_um(case) if task and task.pixel_size_um else None
