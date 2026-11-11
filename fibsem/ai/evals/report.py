"""A run's results as one table per task: hits, unsafe answers, how often
every repeat of a case was right, and what it cost."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fibsem.ai.evals.runner import RESULT


def load_results(out: Path) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(Path(out).rglob(RESULT))]


def summarise(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per task (and part): runs, hits, unsafe, invalid or missing
    answers, cases right in every repeat, mean cost and time."""
    rows: Dict[tuple, Dict[str, Any]] = {}
    per_case: Dict[tuple, Dict[str, List[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        for score in result.get("scores", []):
            key = (score["task"], score.get("part") or "", score.get("headline", True))
            row = rows.setdefault(
                key,
                {
                    "task": score["task"],
                    "part": score.get("part"),
                    "headline": score.get("headline", True),
                    "runs": 0,
                    "hit": 0,
                    "unsafe": 0,
                    "invalid": 0,
                    "cost_usd": 0.0,
                    "seconds": 0.0,
                },
            )
            row["runs"] += 1
            row["hit"] += score["result"] == "hit"
            row["unsafe"] += score["result"] == "unsafe"
            row["invalid"] += score["result"] == "invalid"
            row["cost_usd"] += float(result.get("spent_usd") or 0.0)
            row["seconds"] += float(result.get("seconds") or 0.0)
            per_case[key][result["case"]].append(score["result"] == "hit")
    for key, row in rows.items():
        cases = per_case[key]
        row["cases"] = len(cases)
        row["all_repeats_right"] = sum(all(hits) for hits in cases.values())
        row["mean_cost_usd"] = round(row.pop("cost_usd") / row["runs"], 3)
        row["mean_seconds"] = round(row.pop("seconds") / row["runs"], 1)
    return list(rows.values())


def table(rows: List[Dict[str, Any]]) -> str:
    """The rows as a Markdown table."""
    head = "| Task | Headline | Runs | Hits | Unsafe | Invalid | Cases right every repeat | Mean cost | Mean time |"
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        name = row["task"] + (f" ({row['part']})" if row["part"] else "")
        lines.append(
            f"| {name} | {'yes' if row['headline'] else 'no'} | {row['runs']} "
            f"| {row['hit']} | {row['unsafe']} | {row['invalid']} "
            f"| {row['all_repeats_right']} of {row['cases']} "
            f"| ${row['mean_cost_usd']:.2f} | {row['mean_seconds']:.0f} s |"
        )
    return "\n".join(lines) + "\n"
