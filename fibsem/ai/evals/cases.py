"""Loading and checking eval cases (``case.yaml``, format 0)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

FORMAT = 0
FILE = "case.yaml"
TASKS = (
    "site_selection",
    "fm_target",
    "correlation",
    "polishing_stop",
    "target_present",
    "square_finding",
    "square_quality",
)
# What a square is, in a square_quality case. A key may accept more than one.
SQUARE_LABELS = (
    "good",
    "no_cells",
    "damaged",
    "thick",
    "cell_near_bar",
    "cell_too_big",
)


def accepted_labels(value: Any) -> List[str]:
    """A square's key label, or its list of labels that are all right."""
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


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
    if case.task not in TASKS:
        check.problems.append(f"unknown task {case.task!r}")
    for name, part in case.parts.items():
        if part.get("task") not in TASKS:
            check.problems.append(f"part {name}: unknown task {part.get('task')!r}")
    if case.status == "parked":
        return check
    for rel in input_paths(case):
        if not (case.path / rel).exists():
            check.problems.append(f"missing input {rel}")
    if case.status == "keyed":
        check.problems.extend(_key_problems(case))
    return check


def _key_problems(case: Case) -> List[str]:
    key, problems = case.key, []
    if case.task == "fm_target":
        target = key.get("target") or {}
        if "usable" not in key:
            problems.append("key has no usable")
        if key.get("usable") is True and not {"x_px", "y_px"} <= set(target):
            problems.append("key target has no x_px, y_px")
        if not pixel_size_um(case):
            problems.append("no stack pixel_size_um")
    elif case.task == "correlation":
        if not {"x", "y"} <= set(key.get("target_fib_px") or {}):
            problems.append("key has no target_fib_px x, y")
        if not pixel_size_um(case):
            problems.append("no inputs fib_pixel_size_um")
    elif case.task == "polishing_stop":
        names = pass_names(case)
        stops = stop_passes(key)
        if not stops or any(stop not in names for stop in stops):
            problems.append(f"key stop_after {key.get('stop_after')!r} is not a pass")
    elif case.task == "site_selection":
        for square in key.get("good", []) + key.get("bad", []):
            if not {"x_px", "y_px"} <= set(square):
                problems.append(f"key square without x_px, y_px: {square}")
    elif case.task == "square_finding":
        squares = key.get("squares") or []
        if not squares:
            problems.append("key has no squares")
        for square in squares:
            if not {"x_px", "y_px"} <= set(square):
                problems.append(f"key square without x_px, y_px: {square}")
        for name in ("pitch_px", "angle_deg"):
            if name not in key:
                problems.append(f"key has no {name}")
    elif case.task == "square_quality":
        shown = {str(s.get("square")) for s in case.data.get("squares") or []}
        labels = key.get("labels") or {}
        if not labels:
            problems.append("key has no labels")
        for name, label in labels.items():
            for one in accepted_labels(label):
                if one not in SQUARE_LABELS:
                    problems.append(f"square {name}: unknown label {one!r}")
            if str(name) not in shown:
                problems.append(f"square {name} is keyed but not in squares")
    for name, part in case.parts.items():
        if part.get("task") == "target_present":
            unknown = set(part.get("key") or {}) - set(pass_names(case))
            if unknown:
                problems.append(
                    f"part {name}: key names unknown passes {sorted(unknown)}"
                )
    return problems


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
    if case.task == "fm_target":
        return (case.data.get("stack") or {}).get("pixel_size_um")
    if case.task == "correlation":
        return (case.data.get("inputs") or {}).get("fib_pixel_size_um")
    return None
