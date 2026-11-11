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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.evals import report
from fibsem.ai.evals.cases import Case, find_cases, load_case, pass_names
from fibsem.ai.evals.runner import ANSWER, RESULT
from fibsem.ai.evals.scoring import InvalidAnswer, match_squares, yes_no

log = logging.getLogger(__name__)

PAGE = "report.html"
KEY = "#2ca02c"
RESULT_COLOURS = {
    "hit": "#2ca02c",
    "miss": "#d62728",
    "early": "#ff7f0e",
    "late": "#ff7f0e",
    "unsafe": "#9400d3",
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


def draw_site_selection(case: Case, runs: List[Run]):
    image, step = _shrink(_read_image(_input(case, "overview")))
    fig, ax = _figure(12, 5.5)
    h, w = image.shape
    _show(ax, image, extent=(0, w * step, h * step, 0))
    radius = float(case.key.get("hit_radius_px", 100))
    from matplotlib.patches import Circle

    for square in case.key.get("good") or []:
        ax.add_patch(
            Circle(
                (square["x_px"], square["y_px"]), radius, fill=False, color=KEY, lw=1.5
            )
        )
    for square in case.key.get("bad") or []:
        ax.add_patch(
            Circle((square["x_px"], square["y_px"]), radius, fill=False, color="red")
        )
    for n, (_, answer) in enumerate(runs):
        for i, pick in enumerate((answer or {}).get("picks") or []):
            try:
                x, y = float(pick["x_px"]), float(pick["y_px"])
            except (KeyError, TypeError, ValueError):
                continue
            ax.plot(x, y, "x", color=_colour(n), ms=7, mew=2)
            ax.annotate(
                str(i + 1),
                (x, y),
                (4, 4),
                textcoords="offset points",
                color=_colour(n),
                fontsize=7,
            )
    _legend(ax, runs, "keyed good square")
    return fig


def draw_square_finding(case: Case, runs: List[Run]):
    image, step = _shrink(_read_image(_input(case, "overview")))
    h, w = image.shape
    fig, axes = _figure(
        12, 4.8 * len(runs) or 4.8, nrows=max(1, len(runs)), squeeze=False
    )
    squares = case.key["squares"]
    radius = float(case.scoring.get("match_radius_px", float(case.key["pitch_px"]) / 3))
    for ax, (result, answer) in zip(axes[:, 0], runs):
        _show(ax, image, extent=(0, w * step, h * step, 0))
        points = []
        for p in (answer or {}).get("squares") or []:
            try:
                points.append((float(p["x_px"]), float(p["y_px"])))
            except (KeyError, TypeError, ValueError):
                pass
        matched = match_squares(points, squares, radius)
        found = set(matched.values())
        for j, sq in enumerate(squares):
            colour = KEY if j in found else ("#999999" if sq.get("optional") else "red")
            ax.plot(sq["x_px"], sq["y_px"], "o", mfc="none", mec=colour, ms=9, mew=1.5)
        for i, (x, y) in enumerate(points):
            ax.plot(
                x, y, "+", color="#1f77b4" if i in matched else "orange", ms=8, mew=2
            )
        score = (result.get("scores") or [{}])[0]
        d = score.get("details", {})
        ax.set_title(
            f"r{result['repeat']}: {score.get('result')}  recall {d.get('recall')}  "
            f"precision {d.get('precision')}  pitch {(answer or {}).get('pitch_px')} "
            f"(key {case.key['pitch_px']})  angle {(answer or {}).get('angle_deg')} "
            f"(key {case.key['angle_deg']})",
            fontsize=9,
        )
    axes[0, 0].text(
        0.01,
        0.01,
        "circles: key (green found, red missed, grey optional); "
        "+: agent (blue matched, orange extra)",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6),
    )
    return fig


def draw_square_quality(case: Case, runs: List[Run]):
    image = _read_image(_input(case, "overview"))
    squares = case.data.get("squares") or []
    labels = case.key.get("labels") or {}
    cols = 6
    rows = int(np.ceil(len(squares) / cols)) or 1
    fig, axes = _figure(cols * 2.2, rows * 2.6, nrows=rows, ncols=cols, squeeze=False)
    half = 200
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sq in zip(axes.ravel(), squares):
        x, y = int(sq["x_px"]), int(sq["y_px"])
        crop = image[max(0, y - half) : y + half, max(0, x - half) : x + half]
        _show(ax, crop)
        name = str(sq["square"])
        want = labels.get(name, "?")
        said = []
        for result, answer in runs:
            value = ((answer or {}).get("labels") or {}).get(name)
            value = value.get("label") if isinstance(value, dict) else value
            value = str(value or "-").replace(" ", "_")
            said.append(value)
        wrong = any(s != want for s in said)
        ax.set_title(
            f"{name}: {want}\n" + " ".join(said),
            fontsize=7,
            color="red" if wrong else "black",
        )
    return fig


def draw_fm_target(case: Case, runs: List[Run]):
    target = case.key.get("target") or {}
    x, y = target.get("x_px"), target.get("y_px")
    if x is None:
        pts = [a for _, a in runs if a and "x_px" in a]
        if not pts:
            return None
        x, y = pts[0]["x_px"], pts[0]["y_px"]
    crop, extent = _stack_crop(case, float(x), float(y), 400)
    fig, ax = _figure(7, 7)
    _show(ax, crop, extent=extent)
    if "x_px" in target:
        ax.plot(target["x_px"], target["y_px"], "o", mfc="none", mec=KEY, ms=14, mew=2)
    for n, (_, answer) in enumerate(runs):
        try:
            ax.plot(
                float(answer["x_px"]),
                float(answer["y_px"]),
                "x",
                color=_colour(n),
                ms=10,
                mew=2,
            )
        except (KeyError, TypeError, ValueError):
            pass
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"max projection, key usable: {case.key.get('usable')}", fontsize=9)
    _legend(ax, runs)
    return fig


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


def _pairs(answer: Dict[str, Any], side: str) -> List[Tuple[float, float]]:
    points = []
    for f in answer.get("fiducials") or []:
        try:
            points.append((float(f[side][0]), float(f[side][1])))
        except (KeyError, IndexError, TypeError, ValueError):
            points.append((np.nan, np.nan))
    return points


def draw_correlation(case: Case, runs: List[Run]):
    """The FM stack (when it can be read) beside the FIB image: the key's
    target and fiducials in green, each repeat's in its colour, fiducial
    pairs numbered alike on both sides."""
    fib = _read_image(_input(case, "fib_image"))
    key = case.key
    fm = None
    if case.data.get("inputs", {}).get("fm_stack"):
        try:
            step = 4
            fm = (_fm_overview(_input(case, "fm_stack"), step), step)
        except Exception as e:  # the FIB side still says most of it
            log.warning("No FM panel for %s: %s", case.id, e)
    fig, axes = _figure(16 if fm else 10, 7, ncols=2 if fm else 1, squeeze=False)
    axes = axes[0]
    if fm:
        image, step = fm
        ax = axes[0]
        h, w = image.shape[:2]
        ax.imshow(image, extent=(0, w * step, h * step, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "FM max projection (green: label, magenta: reflection)", fontsize=9
        )
        _mark_points(ax, [p[:2] for p in key.get("fiducials_fm_px") or []], KEY, "s", 9)
        target = key.get("target_fm") or {}
        if "x_px" in target:
            ax.plot(
                target["x_px"], target["y_px"], "o", mfc="none", mec=KEY, ms=16, mew=2.5
            )
        for n, (_, answer) in enumerate(runs):
            _mark_points(ax, _pairs(answer or {}, "fm_px"), _colour(n), "s", 7)
    ax = axes[-1]
    _show(ax, fib, extent=(0, fib.shape[1], fib.shape[0], 0))
    _mark_points(ax, key.get("fiducials_fib_px") or [], KEY, "s", 9)
    target = key["target_fib_px"]
    ax.plot(target["x"], target["y"], "o", mfc="none", mec=KEY, ms=16, mew=2.5)
    for n, (_, answer) in enumerate(runs):
        answer = answer or {}
        _mark_points(ax, _pairs(answer, "fib_px"), _colour(n), "s", 7)
        try:
            ax.plot(
                float(answer["x_px"]),
                float(answer["y_px"]),
                "x",
                color=_colour(n),
                ms=14,
                mew=3,
            )
        except (KeyError, TypeError, ValueError):
            pass
    ax.set_title(
        "FIB image. circle/x: target; squares: fiducial pairs, numbered", fontsize=9
    )
    _legend(ax, runs)
    return fig


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


def draw_polishing_stop(case: Case, runs: List[Run], key=None, part=None):
    key = key or case.key
    names = pass_names(case)
    passes = case.data.get("passes") or []
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(max(8, 1.1 * len(names)), 6))
    grid = fig.add_gridspec(2, len(names), height_ratios=(1, 2))
    _pass_strip(case, [fig.add_subplot(grid[0, i]) for i in range(len(names))])
    ax = fig.add_subplot(grid[1, :])
    xs = np.arange(len(names))
    med = [p.get("measured", {}).get("median_nm", np.nan) for p in passes]
    lo = [p.get("measured", {}).get("min_nm", np.nan) for p in passes]
    hi = [p.get("measured", {}).get("max_nm", np.nan) for p in passes]
    ax.fill_between(xs, lo, hi, color="#cccccc", label="min-max")
    ax.plot(xs, med, "k.-", label="median")
    want = names.index(key["stop_after"])
    ax.axvline(
        want, color=KEY, lw=3, alpha=0.6, label=f"key: stop after {key['stop_after']}"
    )
    for n, (result, answer) in enumerate(runs):
        stop = _stops(
            answer if part is None else ((answer or {}).get("parts") or {}).get(part),
            names,
        )
        if stop is not None:
            ax.axvline(
                stop + 0.04 * (n + 1),
                color=_colour(n),
                ls="--",
                label=f"r{result['repeat']} stops after {names[stop]}",
            )
        else:
            ax.text(
                len(names) - 1,
                np.nanmax(hi),
                f"r{result['repeat']}: never stops",
                color=_colour(n),
                ha="right",
                va="top",
                fontsize=8,
            )
    ax.set_xticks(xs, names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("thickness, nm")
    ax.legend(fontsize=7)
    return fig


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


DRAW: Dict[str, Callable] = {
    "site_selection": draw_site_selection,
    "square_finding": draw_square_finding,
    "square_quality": draw_square_quality,
    "fm_target": draw_fm_target,
    "correlation": draw_correlation,
    "polishing_stop": draw_polishing_stop,
    "target_present": draw_target_present,
}


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
        draw = DRAW.get(task)
        if draw is None:
            continue
        try:
            kw = {"key": key, "part": part} if part is not None else {}
            fig = draw(case, runs, **kw)
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
        try:
            answer = json.loads((folder / "work" / ANSWER).read_text())
        except (OSError, ValueError):
            answer = None
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
        "Task",
        "Headline",
        "Runs",
        "Hits",
        "Unsafe",
        "Invalid",
        "Cases right every repeat",
        "Mean cost",
        "Mean time",
    )
    lines = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for row in rows:
        name = row["task"] + (f" ({row['part']})" if row["part"] else "")
        cells = (
            name,
            "yes" if row["headline"] else "no",
            row["runs"],
            row["hit"],
            row["unsafe"],
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
