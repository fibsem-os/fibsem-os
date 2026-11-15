"""A run's results as one HTML page: the table, then per case the key and
every repeat's answer drawn on the case's own images, with each repeat's
result, cost, reason and logs.

Images are embedded, so the page opens anywhere. A case whose files can't be
read gets its text without a figure.
"""

import base64
import html
import io
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from fibsem.ai.evals import report, tasks
from fibsem.ai.evals.cases import (
    Case,
    find_cases,
    load_case,
)
from fibsem.ai.evals.runner import ANSWER, RESULT, run_answer
from fibsem.ai.evals.scoring import (
    score_case,
    score_live,
)

log = logging.getLogger(__name__)

PAGE = "report.html"
KEY = "#2ca02c"
RESULT_COLOURS = {
    "hit": "#2ca02c",
    "miss": "#d62728",
    "early": "#ff7f0e",
    "late": "#ff7f0e",
    "invalid": "#7f7f7f",
}

# One run: its result.json and its answer.json (None when it wrote none).
Run = Tuple[Dict[str, Any], Optional[Dict[str, Any]]]


def _colour(n: int) -> str:
    import matplotlib

    # tab10 without its green, which is the key's.
    picks = (0, 1, 3, 4, 5, 6, 8, 9)
    return matplotlib.colormaps["tab10"](picks[n % len(picks)])


# --- images -----------------------------------------------------------------


def _stretch(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, (1, 99.5))
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def _read_image(path: Path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    while image.ndim > 2:
        image = image[0]
    return image


def _shrink(image: np.ndarray, most: int = 2000) -> Tuple[np.ndarray, int]:
    """The image cut down to at most *most* px wide, and the step taken."""
    step = max(1, int(np.ceil(max(image.shape) / most)))
    return image[::step, ::step], step


def _stack_crop(case: Case, x: float, y: float, half: int) -> Tuple[np.ndarray, Tuple]:
    """The max projection over z of the target channel, around (x, y); and
    its extent in stack px."""
    import tifffile

    stack = case.data["stack"]
    data = tifffile.memmap(case.path / stack["file"], mode="r")
    shape = stack.get("shape")
    if shape:
        data = data.reshape(shape)
    channels = stack.get("channels") or []
    if data.ndim == 4:
        index = channels.index(stack.get("target_channel")) if channels else 0
        data = data[index]
    h, w = data.shape[-2:]
    x0, x1 = max(0, int(x) - half), min(w, int(x) + half)
    y0, y1 = max(0, int(y) - half), min(h, int(y) + half)
    crop = np.asarray(data[..., y0:y1, x0:x1]).max(axis=0)
    return crop, (x0, x1, y1, y0)


def _figure(width: float = 10, height: float = 6, **kw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(width, height), **kw)


def _png(fig) -> str:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _show(ax, image: np.ndarray, extent=None) -> None:
    ax.imshow(_stretch(image), cmap="gray", extent=extent, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])


def _legend(ax, runs: List[Run], key_label: str = "key") -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=KEY, marker="o", ls="", label=key_label)]
    for n, (result, _) in enumerate(runs):
        label = f"r{result['repeat']}: " + ", ".join(
            s["result"] for s in result.get("scores", [])
        )
        handles.append(Line2D([], [], color=_colour(n), marker="x", ls="", label=label))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)


def _input(case: Case, name: str) -> Path:
    return case.path / case.data["inputs"][name]


# --- one figure per task ----------------------------------------------------


def _fm_overview(path: Path, step: int = 4) -> np.ndarray:
    """An FM stack as an RGB max projection over z, shrunk by *step*: the
    first channel (reflection) in magenta, the last (the label) in green."""
    import tifffile

    data = tifffile.memmap(path, mode="r")
    if data.ndim == 3:
        data = data[None]
    while data.ndim > 4:
        data = data[0]
    first = _stretch(np.asarray(data[0, :, ::step, ::step]).max(axis=0))
    last = _stretch(np.asarray(data[-1, :, ::step, ::step]).max(axis=0))
    return np.dstack([first, last, first])


def _mark_points(ax, points, colour, marker, size, numbered=True):
    for i, (x, y) in enumerate(points, 1):
        ax.plot(x, y, marker, mfc="none", mec=colour, color=colour, ms=size, mew=2)
        if numbered:
            ax.annotate(
                str(i),
                (x, y),
                (5, 5),
                textcoords="offset points",
                color=colour,
                fontsize=8,
                fontweight="bold",
            )


def _stops(answer: Optional[Dict[str, Any]], names: List[str]) -> Optional[int]:
    stops = [
        names.index(p.get("name"))
        for p in (answer or {}).get("passes") or []
        if str(p.get("decision", "")).lower().replace(" ", "_") == "stop"
        and p.get("name") in names
    ]
    return min(stops) if stops else None


def _pass_strip(case: Case, axes) -> None:
    for ax, entry in zip(axes, case.data.get("passes") or []):
        ax.axis("off")
        path = entry.get("fib")
        if path and (case.path / path).is_file():
            image, _ = _shrink(_read_image(case.path / path), 300)
            _show(ax, image)
        ax.set_title(entry["name"], fontsize=6)


def case_figures(case: Case, runs: List[Run]) -> List[Tuple[str, str]]:
    """(caption, base64 png) per figure of *case*: one, or one per part."""
    jobs = []
    if case.parts:
        for name, part in case.parts.items():
            key = part.get("key") if isinstance(part.get("key"), dict) else case.key
            jobs.append((f"part {name}: {part['task']}", part["task"], key, name))
    else:
        jobs.append((case.task, case.task, None, None))
    figures = []
    for caption, task, key, part in jobs:
        if tasks.get(task) is None:
            continue
        try:
            kw = {"key": key, "part": part} if part is not None else {}
            fig = tasks.get(task).draw(case, runs, **kw)
        except Exception as e:  # a figure is a help, never a reason to fail
            log.warning("No figure for %s (%s): %s", case.id, caption, e)
            figures.append((f"{caption}: no figure ({e})", ""))
            continue
        if fig is not None:
            figures.append((caption, _png(fig)))
    return figures


# --- the page ---------------------------------------------------------------


def load_runs(out: Path) -> Dict[str, List[Tuple[Path, Run]]]:
    """Each case's runs, as (run folder, (result, answer)), repeat order."""
    runs: Dict[str, List[Tuple[Path, Run]]] = defaultdict(list)
    for path in sorted(Path(out).rglob(RESULT)):
        result = json.loads(path.read_text())
        folder = path.parent
        answer = run_answer(folder)
        runs[result["case"]].append((folder, (result, answer)))
    for items in runs.values():
        items.sort(key=lambda item: item[1][0].get("repeat", 0))
    return runs


def _find_case(case_id: str, results: List[Run], cases: Dict[str, Case]):
    if case_id in cases:
        return cases[case_id]
    for result, _ in results:
        path = result.get("case_path")
        if path and Path(path, "case.yaml").is_file():
            return load_case(Path(path))
    return None


def _why(answer: Dict[str, Any]) -> str:
    """The answer's reason; or, for answers that give one per pick or per
    pass, those in order."""
    if answer.get("reason"):
        return str(answer["reason"])
    lines = []
    for n, pick in enumerate(answer.get("picks") or [], 1):
        if isinstance(pick, dict) and pick.get("reason"):
            lines.append(f"{n}. {pick['reason']}")
    for entry in answer.get("passes") or []:
        if isinstance(entry, dict) and entry.get("reason"):
            lines.append(
                f"{entry.get('name')} {entry.get('decision')}: {entry['reason']}"
            )
    return " / ".join(lines)


def _reason(answer: Optional[Dict[str, Any]]) -> str:
    if not answer:
        return "(no answer)"
    if "parts" in answer:
        return " | ".join(
            f"{name}: {_why(part or {})}"
            for name, part in (answer.get("parts") or {}).items()
        )
    return _why(answer)


def _table_html(rows: List[Dict[str, Any]]) -> str:
    head = (
        "Tier",
        "Task",
        "Headline",
        "Runs",
        "Hits",
        "Costly",
        "Invalid",
        "Cases right every repeat",
        "Mean cost",
        "Mean time",
    )
    lines = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for row in rows:
        name = row["task"] + (f" ({row['part']})" if row["part"] else "")
        cells = (
            row.get("tier", "real"),
            name,
            "yes" if row["headline"] else "no",
            row["runs"],
            row["hit"],
            row["costly"],
            row["invalid"],
            f"{row['all_repeats_right']} of {row['cases']}",
            f"${row['mean_cost_usd']:.2f}",
            f"{row['mean_seconds']:.0f} s",
        )
        lines.append(
            "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"
        )
    return "\n".join(lines + ["</table>"])


STYLE = """
body { font: 14px/1.4 system-ui, sans-serif; margin: 24px auto; max-width: 1300px;
       padding: 0 16px; color: #222; background: #fff; }
table { border-collapse: collapse; margin: 8px 0 16px; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f3f3f3; }
section { border-top: 2px solid #ddd; margin-top: 28px; padding-top: 8px; }
img { max-width: 100%; }
.result { font-weight: 600; }
.brief { color: #555; }
"""


def _result_html(result: str) -> str:
    colour = RESULT_COLOURS.get(result, "#000")
    return f"<span class='result' style='color:{colour}'>{html.escape(result)}</span>"


def live_rows(case: Case, runs: List[Run]) -> List[Dict[str, Any]]:
    """Per part: what the real run's decision scores, and how many repeats
    (scored with the case's key as it is now) got that same result."""
    live = score_live(case)
    if live is None:
        return []
    rows = []
    for score in live:
        same, results = 0, []
        for _, answer in runs:
            for s in score_case(case, answer or {}):
                if s.part == score.part:
                    results.append(s.result)
                    same += s.result == score.result
        rows.append(
            {
                "part": score.part,
                "live": score.result,
                "eval": results,
                "same": same,
            }
        )
    return rows


def _live_html(case: Case, rows: List[Dict[str, Any]]) -> str:
    source = html.escape(str((case.data.get("live") or {}).get("source", "")))
    lines = []
    for row in rows:
        part = f"part {row['part']}: " if row["part"] else ""
        evals = ", ".join(_result_html(r) for r in row["eval"])
        lines.append(
            f"<p>{part}<b>Real run:</b> {_result_html(row['live'])} &nbsp; "
            f"<b>Eval repeats now:</b> {evals} &nbsp; "
            f"(same as the real run: {row['same']} of {len(row['eval'])})</p>"
        )
    if source:
        lines.append(f"<p class='brief'>Real run: {source}</p>")
    return "\n".join(lines)


def compare_table(outs: List[Path], cases_root: Optional[Path] = None) -> str:
    """A markdown table of each case (and part) against each run folder:
    hits of repeats, scored with the case's key as it is now, and the mean
    cost of a run."""
    cases = {c.id: c for c in find_cases(cases_root)} if cases_root else {}
    columns = [load_runs(out) for out in outs]
    cells: Dict[str, Dict[int, str]] = defaultdict(dict)
    for n, runs in enumerate(columns):
        for case_id, items in runs.items():
            pairs = [run for _, run in items]
            case = _find_case(case_id, pairs, cases)
            if case is None:
                continue
            hits: Dict[Optional[str], List[bool]] = defaultdict(list)
            for _, answer in pairs:
                for score in score_case(case, answer or {}):
                    hits[score.part].append(score.result == "hit")
            cost = sum(float(r.get("spent_usd") or 0) for r, _ in pairs) / len(pairs)
            for part, got in hits.items():
                name = case_id + (f" ({part})" if part else "")
                cells[name][n] = f"{sum(got)}/{len(got)} (${cost:.2f})"
    names = [Path(out).name for out in outs]
    lines = [
        "| Case | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for name in sorted(cells):
        row = [cells[name].get(n, "-") for n in range(len(outs))]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    lines += ["", "Ran with:"]
    for name, runs in zip(names, columns):
        lines.append(
            f"- {name}: {ran_with(r for items in runs.values() for _, (r, _) in items)}"
        )
    return "\n".join(lines)


def ran_with(results: Iterable[Dict[str, Any]]) -> str:
    """The models, modes, commits and skills a run folder's runs used; more
    than one of a kind means its runs are not alike."""
    seen: Dict[str, set] = defaultdict(set)
    for result in results:
        version = result.get("version") or {}
        seen["model"].add(result.get("model") or "?")
        seen["mode"].add("code" if result.get("code") else "files")
        seen["commit"].add(
            (version.get("commit") or "?")
            + ("+changes" if version.get("changed") else "")
        )
        seen["skills"].add(version.get("skills") or "?")
        seen["knowledge"].add(version.get("knowledge") or "?")
    said = []
    for kind, values in seen.items():
        text = " / ".join(sorted(values))
        said.append(f"{kind} {text}" + (" (MIXED)" if len(values) > 1 else ""))
    return ", ".join(said) or "no runs"


def write_html(out: Path, cases_root: Optional[Path] = None) -> Path:
    """The page for the run folder *out*, written into it; its path."""
    out = Path(out)
    cases = {c.id: c for c in find_cases(cases_root)} if cases_root else {}
    runs = load_runs(out)
    rows = report.summarise(r for items in runs.values() for _, (r, _) in items)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Eval report</title><style>{STYLE}</style></head><body>",
        f"<h1>Eval report: {html.escape(out.name)}</h1>",
        _table_html(rows),
    ]
    compare = []
    for case_id, items in sorted(runs.items()):
        case = _find_case(case_id, [run for _, run in items], cases)
        if case is not None:
            for row in live_rows(case, [run for _, run in items]):
                compare.append((case_id, row))
    if compare:
        parts.append(
            "<h2>Eval against the real run</h2><table><tr><th>Case</th>"
            "<th>Real run</th><th>Eval repeats now</th><th>Same as real</th></tr>"
        )
        for case_id, row in compare:
            name = case_id + (f" ({row['part']})" if row["part"] else "")
            parts.append(
                f"<tr><td>{html.escape(name)}</td><td>{_result_html(row['live'])}</td>"
                f"<td>{', '.join(_result_html(r) for r in row['eval'])}</td>"
                f"<td>{row['same']} of {len(row['eval'])}</td></tr>"
            )
        parts.append("</table>")
    for case_id, items in sorted(runs.items()):
        pairs = [run for _, run in items]
        case = _find_case(case_id, pairs, cases)
        parts.append(f"<section><h2>{html.escape(case_id)}</h2>")
        if case is None:
            parts.append("<p>Case not found: pass --cases to draw it.</p>")
        else:
            parts.append(
                f"<p class='brief'>{html.escape(str(case.data.get('brief', '')))}</p>"
            )
            rows_live = live_rows(case, pairs)
            if rows_live:
                parts.append(_live_html(case, rows_live))
            for caption, png in case_figures(case, pairs):
                parts.append(f"<h3>{html.escape(caption)}</h3>")
                if png:
                    parts.append(
                        f"<img src='data:image/png;base64,{png}' alt='{html.escape(caption)}'>"
                    )
        parts.append(
            "<table><tr><th>Repeat</th><th>Result</th><th>Cost</th>"
            "<th>Time</th><th>Reason</th><th>Logs</th></tr>"
        )
        for folder, (result, answer) in items:
            results = ", ".join(
                f"<span class='result' style='color:{RESULT_COLOURS.get(s['result'], '#000')}'>"
                f"{html.escape((s.get('part') or '') + ' ' + s['result']).strip()}</span>"
                for s in result.get("scores", [])
            )
            rel = folder.relative_to(out).as_posix()
            logs = " ".join(
                f"<a href='{rel}/{name}'>{label}</a>"
                for name, label in (
                    ("agent.log", "agent"),
                    ("session/fibsem-agent.log", "session"),
                    (f"work/{ANSWER}", "answer"),
                    (RESULT, "result"),
                )
                if (folder / name).is_file()
            )
            parts.append(
                f"<tr><td>r{result.get('repeat')}</td><td>{results} ({html.escape(str(result.get('status')))})</td>"
                f"<td>${float(result.get('spent_usd') or 0):.2f}</td>"
                f"<td>{float(result.get('seconds') or 0):.0f} s</td>"
                f"<td>{html.escape(_reason(answer))}</td><td>{logs}</td></tr>"
            )
        parts.append("</table></section>")
    parts.append("</body></html>")
    page = out / PAGE
    page.write_text("\n".join(parts), encoding="utf-8")
    return page
