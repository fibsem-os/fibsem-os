"""A matrix run: the same cases, on one commit, under every combination of
model, mode (files or code) and playbook (the base one or each case's own),
each combination ("cell") in its own folder. Repeat 1 of every cell runs
before repeat 2 of any, so a matrix stopped part way is still even; run it
again into the same folder to carry on."""

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from fibsem.ai.evals import trace, visual
from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.runner import CASE_PLAYBOOK, RunConfig, run_cases

MATRIX = "matrix.json"
MODES = ("files", "code")
PLAYBOOKS = ("base", CASE_PLAYBOOK)
# The skill each task is decided by: a repeat that never read it decided
# without it.
TASK_SKILLS = {
    "site_selection": "screen-grids",
    "square_finding": "screen-grids",
    "square_quality": "screen-grids",
    "polishing_stop": "autolamella-polishing",
    "target_present": "autolamella-coincidence-milling",
    "fm_target": "autolamella-correlative",
    "correlation": "autolamella-correlative",
    "setup_placement": "autolamella",
}


@dataclass(frozen=True)
class Cell:
    model: str
    mode: str
    playbook: str

    @property
    def name(self) -> str:
        return f"{self.model}-{self.mode}-{self.playbook}"

    def config(self, base: RunConfig, out: Path) -> RunConfig:
        return replace(
            base,
            out=Path(out) / self.name,
            model=self.model,
            code=self.mode == "code",
            playbook=CASE_PLAYBOOK if self.playbook == CASE_PLAYBOOK else None,
        )


def cells(
    models: Iterable[str], modes: Iterable[str], playbooks: Iterable[str]
) -> List[Cell]:
    return [Cell(*combo) for combo in itertools.product(models, modes, playbooks)]


class MatrixError(RuntimeError):
    pass


def start(out: Path, grid: List[Cell], version: Dict[str, Any]) -> None:
    """Write the matrix's record, or check that a matrix already in *out*
    ran the same code: cells on another commit or skills don't compare."""
    out = Path(out)
    record = out / MATRIX
    if record.is_file():
        before = json.loads(record.read_text())
        old = before.get("version") or {}
        for key in ("commit", "skills", "knowledge"):
            if old.get(key) != version.get(key):
                raise MatrixError(
                    f"{out} ran with {key} {old.get(key)}, this checkout has "
                    f"{version.get(key)}: give a new --out, or check out "
                    "the matrix's commit."
                )
    out.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps({"cells": [c.name for c in grid], "version": version}, indent=2)
    )


def run_matrix(
    cases: List[Case],
    out: Path,
    grid: List[Cell],
    base: RunConfig,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Every cell, repeat by repeat; the results in the order they finish."""
    results = []
    for repeat in range(1, base.repeats + 1):
        for cell in grid:
            config = replace(cell.config(base, out), repeats=repeat)
            results += run_cases(cases, config, on_result)
    return results


def _task_skills(case_id: str, items) -> List[str]:
    tasks = set()
    for _, (result, _) in items:
        tasks |= {s.get("task") for s in result.get("scores") or []}
        tasks.add(result.get("task"))
    return sorted({TASK_SKILLS[t] for t in tasks if t in TASK_SKILLS})


def skills_table(folders: List[Path]) -> str:
    """For each case and cell, the repeats that read every skill the case's
    task is decided by."""
    cells_: Dict[str, Dict[int, str]] = defaultdict(dict)
    for n, folder in enumerate(folders):
        for case_id, items in visual.load_runs(folder).items():
            needed = _task_skills(case_id, items)
            read = [
                set(needed) <= set(trace.read_run(run_folder).skills)
                for run_folder, _ in items
            ]
            cells_[case_id][n] = f"{sum(read)}/{len(read)}"
    names = [Path(f).name for f in folders]
    lines = [
        "| Case: task skill read | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for case_id in sorted(cells_):
        row = [cells_[case_id].get(n, "-") for n in range(len(folders))]
        lines.append(f"| {case_id} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def matrix_markdown(out: Path, cases_root: Optional[Path] = None) -> str:
    """The matrix's results: hits and cost per case and cell, then how often
    each cell read the task's skill."""
    out = Path(out)
    record = json.loads((out / MATRIX).read_text())
    folders = [out / name for name in record["cells"] if (out / name).is_dir()]
    if not folders:
        return f"# Matrix: {out.name}\n\nNo runs yet."
    version = record.get("version") or {}
    return "\n\n".join(
        [
            f"# Matrix: {out.name}",
            f"Commit {version.get('commit')}, skills {version.get('skills')}, "
            f"knowledge {version.get('knowledge')}.",
            visual.compare_table(folders, cases_root),
            skills_table(folders),
        ]
    )
