"""The hardware check: does fibsemOS drive this instrument's devices correctly?

It runs one level of steps per device and records a ``Finding`` for each, which a
``Report`` collects. It reports and never fixes: a failed step is recorded and the
run goes on, and every setting a step changes is put back before the next step.

Levels, each including the one before:

- ``READ``: read every parameter and check it against its own limits and choices.
  Nothing changes.
- ``SETTINGS``: set each changeable setting to another valid value, read it back, put
  it back. FM settings are for the next exposure (``fibsem.devices.fm``), so on every
  driver this level turns no light on and moves nothing.
- ``ACQUIRE``: take one frame with the settings as they are. The light comes on, so a
  sample under the objective is exposed.

No level moves the objective or the stage.

The FM's steps are in :mod:`fibsem.fm.hardware_check`. Only the FM is checked so far.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from fibsem.structures import RangeLimit
from fibsem.util.timestamps import format_time, now


class Level(str, Enum):
    READ = "read"
    SETTINGS = "settings"
    ACQUIRE = "acquire"

    def includes(self, other: "Level") -> bool:
        order = list(Level)
        return order.index(self) >= order.index(other)


class Result(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOTE = "note"  # nothing wrong, but worth knowing: a parameter this driver lacks


@dataclass
class Finding:
    """One step: what was asked, what was read back, and how it went."""

    device: str
    step: str
    result: Result
    asked: Any = None
    read: Any = None
    detail: str = ""
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "step": self.step,
            "result": self.result.value,
            "asked": _plain(self.asked),
            "read": _plain(self.read),
            "detail": self.detail,
            "seconds": round(self.seconds, 4),
        }


@dataclass
class Report:
    level: Level
    instrument: str = ""
    started_at: datetime = field(default_factory=now)
    finished_at: Optional[datetime] = None
    findings: List[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> Finding:
        self.findings.append(finding)
        return finding

    def count(self, result: Result) -> int:
        return sum(f.result is result for f in self.findings)

    @property
    def passed(self) -> bool:
        return self.count(Result.FAIL) == 0

    @contextmanager
    def step(self, device: str, step: str, asked: Any = None) -> Iterator[Finding]:
        """Time one step and record it. The step fills in ``read`` and its result;
        an exception fails it, with the exception as the detail, and the run goes on."""
        finding = Finding(device, step, Result.PASS, asked=asked)
        start = time.perf_counter()
        try:
            yield finding
        except Exception as e:
            finding.result = Result.FAIL
            finding.detail = f"{type(e).__name__}: {e}"
        finally:
            finding.seconds = time.perf_counter() - start
            self.add(finding)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "instrument": self.instrument,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "passed": self.passed,
            "counts": {r.value: self.count(r) for r in Result},
            "findings": [f.to_dict() for f in self.findings],
        }

    def format(self) -> str:
        """The report as text, one line per step, for a terminal or a support email."""
        lines = [
            f"Hardware check · {self.level.value} · {self.instrument} · "
            f"{format_time(self.started_at)}",
            "",
        ]
        width = max((len(f.step) for f in self.findings), default=0)
        device = None
        for f in self.findings:
            if f.device != device:
                if device is not None:
                    lines.append("")
                lines.append(f.device)
                device = f.device
            values = _arrow(f.asked, f.read)
            line = f"  {f.result.value.upper():4}  {f.step:<{width}}  {values}"
            if f.detail:
                line += f"  ({f.detail})" if values else f"  {f.detail}"
            lines.append(line.rstrip())
        took = ""
        if self.finished_at is not None:
            took = f" in {(self.finished_at - self.started_at).total_seconds():.2f} s"
        lines += [
            "",
            f"{self.count(Result.PASS)} passed, {self.count(Result.FAIL)} failed, "
            f"{self.count(Result.NOTE)} notes{took}",
        ]
        return "\n".join(lines)


def run_hardware_check(
    microscope: Any,
    level: Level = Level.READ,
    on_frame: Optional[Callable[[Any], None]] = None,
) -> Report:
    """Check *microscope*'s devices at *level*. *on_frame* receives the image the
    ``ACQUIRE`` level takes, to save or show it."""
    from fibsem.fm.hardware_check import check_fm

    report = Report(level=Level(level), instrument=_instrument(microscope))
    fm = getattr(microscope, "fm", None)
    if fm is None:
        report.add(
            Finding("fm", "present", Result.NOTE, detail="no fluorescence microscope")
        )
    else:
        check_fm(fm, report, report.level, on_frame=on_frame)
    report.finished_at = now()
    return report


# -- steps every device type shares ------------------------------------------------


def check_readings(device: Any, report: Report, skip: tuple = ()) -> None:
    """Read each of *device*'s parameters, and check it against its own limits and
    choices."""
    for name, parameter in device.parameters.items():
        if name in skip:
            continue
        with report.step(device.name, f"read {name}") as finding:
            value = parameter.get_value()
            finding.read = with_unit(value, parameter.unit)
            problem = out_of_range(value, parameter.limits, parameter.choices)
            if problem:
                finding.result = Result.FAIL
                finding.detail = problem


def check_round_trip(
    device: Any, name: str, value: Any, report: Report, close: float = 1e-6
) -> None:
    """Set *name* to *value*, read it back, then put back what it was and read that
    back too. Two findings: the change and the restore."""
    parameter = device.parameters[name]
    original = parameter.get_value()
    try:
        with report.step(
            device.name, f"set {name}", with_unit(value, parameter.unit)
        ) as finding:
            parameter.set_value(value)
            read = parameter.get_value()
            finding.read = with_unit(read, parameter.unit)
            if not same(read, value, close):
                finding.result = Result.FAIL
                finding.detail = "read back a different value"
    finally:
        with report.step(
            device.name, f"restore {name}", with_unit(original, parameter.unit)
        ) as finding:
            parameter.set_value(original)
            read = parameter.get_value()
            finding.read = with_unit(read, parameter.unit)
            if not same(read, original, close):
                finding.result = Result.FAIL
                finding.detail = "not put back"


def other_choice(choices: Any, current: Any) -> Any:
    """A choice that differs from *current*, or None when there is no other."""
    for choice in choices or ():
        if not same(choice, current):
            return choice
    return None


def same(a: Any, b: Any, close: float = 1e-6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # relative only: values in SI units are as small as 1e-9 (a pixel in metres)
        return math.isclose(a, b, rel_tol=close, abs_tol=1e-15)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y, close) for x, y in zip(a, b))
    return a == b


def out_of_range(value: Any, limits: Any, choices: Any) -> str:
    """Why *value* is outside its parameter's limits or choices, or ''."""
    if (
        isinstance(limits, RangeLimit)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not limits.min <= value <= limits.max
    ):
        return f"outside {limits.min:g} to {limits.max:g}"
    if choices and value is not None and not any(same(value, c) for c in choices):
        return "not one of its choices"
    return ""


# -- formatting ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Quantity:
    value: Any
    unit: Optional[str]

    def __str__(self) -> str:
        if self.unit == "m":
            return _length(self.value)
        return f"{_text(self.value)} {self.unit}" if self.unit else _text(self.value)


def _length(value: Any) -> str:
    """Metres in the unit that reads best: 1.690 mm, 119 nm."""
    values = value if isinstance(value, (tuple, list)) else (value,)
    if not all(isinstance(v, (int, float)) for v in values):
        return f"{_text(value)} m"
    largest = max(abs(v) for v in values)
    scale, unit = next(
        ((s, u) for s, u in ((1e-3, "mm"), (1e-6, "µm")) if largest >= s),
        (1e-9, "nm") if largest else (1e-3, "mm"),  # zero reads as a position: 0 mm
    )
    return f"{' × '.join(f'{v / scale:.4g}' for v in values)} {unit}"


def with_unit(value: Any, unit: Optional[str]) -> Any:
    return _Quantity(value, unit) if unit else value


def _text(value: Any) -> str:
    if isinstance(value, _Quantity):
        return str(value)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (tuple, list)):
        return " × ".join(_text(v) for v in value)
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _arrow(asked: Any, read: Any) -> str:
    if asked is None and read is None:
        return ""
    if asked is None:
        return _text(read)
    if read is None:
        return f"{_text(asked)} → —"
    return f"{_text(asked)} → {_text(read)}"


def _plain(value: Any) -> Any:
    """A value as JSON can hold it."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, _Quantity):
        return {"value": _plain(value.value), "unit": value.unit}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return _text(value)


def _instrument(microscope: Any) -> str:
    info = getattr(getattr(microscope, "system", None), "info", None)
    if info is None:
        return type(microscope).__name__
    parts = [getattr(info, "manufacturer", None), getattr(info, "name", None)]
    return " · ".join(str(p) for p in parts if p)
