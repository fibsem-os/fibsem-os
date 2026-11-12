"""``python -m fibsem.ai.evals serve``: the eval review page (``/ui/evals``)
and the routes it reads, on this computer only. No App Server or microscope.

It reads run folders (what ``run --out`` wrote) under one folder, and the
cases those runs used, each scored with its key as it is now. Settling a
disagreement writes the case's key (``fibsem.ai.evals.review``).
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from fibsem.ai.evals import review, visual
from fibsem.ai.evals.cases import Case, find_cases, stop_passes
from fibsem.ai.evals.runner import RESULT
from fibsem.ai.evals.scoring import score_case
from fibsem.server.web_app import web_app_router

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8765


class Settle(BaseModel):
    item: str
    action: str


class Reviewer:
    """The run folders under *runs* and the cases under *cases*."""

    def __init__(self, runs: Path, cases: Optional[Path] = None):
        self.runs = Path(runs).expanduser().resolve()
        self.cases_root = Path(cases).expanduser().resolve() if cases else None

    # --- finding things ---------------------------------------------------

    def run_folders(self) -> List[Path]:
        """Folders holding results: *runs* itself when it is one run, else
        each folder under it that is one. Newest first."""
        if any(self.runs.glob(f"*/r*/{RESULT}")):
            return [self.runs]
        found = [
            p
            for p in self.runs.iterdir()
            if p.is_dir() and any(p.glob(f"*/r*/{RESULT}"))
        ]
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

    def run_folder(self, name: str) -> Path:
        for folder in self.run_folders():
            if folder.name == name:
                return folder
        raise HTTPException(status_code=404, detail=f"no run {name!r}")

    def _cases(self) -> Dict[str, Case]:
        return {c.id: c for c in find_cases(self.cases_root)} if self.cases_root else {}

    def case_runs(
        self, run: str, case_id: str
    ) -> Tuple[Case, List[Tuple[Path, visual.Run]]]:
        items = visual.load_runs(self.run_folder(run)).get(case_id)
        if not items:
            raise HTTPException(status_code=404, detail=f"no case {case_id!r} in {run}")
        case = visual._find_case(case_id, [r for _, r in items], self._cases())
        if case is None:
            raise HTTPException(
                status_code=404,
                detail=f"case {case_id!r} not found: start serve with --cases",
            )
        return case, items

    # --- answers ------------------------------------------------------------

    def runs_list(self) -> List[Dict[str, Any]]:
        out = []
        for folder in self.run_folders():
            results = list(folder.glob(f"*/r*/{RESULT}"))
            out.append(
                {
                    "name": folder.name,
                    "results": len(results),
                    "cases": len({p.parent.parent.name for p in results}),
                    "modified": folder.stat().st_mtime,
                }
            )
        return out

    def run_summary(self, run: str) -> Dict[str, Any]:
        folder = self.run_folder(run)
        cases = self._cases()
        rows = []
        all_results = []
        for case_id, items in sorted(visual.load_runs(folder).items()):
            pairs = [r for _, r in items]
            all_results += [r for r, _ in pairs]
            case = visual._find_case(case_id, pairs, cases)
            row: Dict[str, Any] = {"id": case_id, "repeats": len(pairs)}
            if case is None:
                row.update(task="?", now=[], open=0, found=False)
            else:
                row.update(
                    task=case.task,
                    found=True,
                    now=[_now(case, answer) for _, answer in pairs],
                    open=len(review.disagreements(case, pairs)),
                )
            rows.append(row)
        return {
            "name": folder.name,
            "ran_with": visual.ran_with(all_results),
            "cases": rows,
        }

    def case_detail(self, run: str, case_id: str) -> Dict[str, Any]:
        case, items = self.case_runs(run, case_id)
        pairs = [r for _, r in items]
        captions = []
        if case.parts:
            captions = [f"part {n}: {p.get('task')}" for n, p in case.parts.items()]
        else:
            captions = [case.task]
        repeats = []
        for folder, (result, answer) in items:
            scores = _now_scores(case, answer)
            repeats.append(
                {
                    "repeat": result.get("repeat"),
                    "did": "; ".join(
                        ((s["part"] + ": ") if s.get("part") else "") + did(s)
                        for s in scores
                        if did(s)
                    ),
                    "then": [s["result"] for s in result.get("scores", [])],
                    "now": _now(case, answer),
                    "status": result.get("status"),
                    "cost_usd": float(result.get("spent_usd") or 0),
                    "seconds": float(result.get("seconds") or 0),
                    "reason": visual._reason(answer),
                }
            )
        return {
            "id": case.id,
            "task": case.task,
            "task_text": TASK_TEXT.get(case.task, case.task),
            "key_text": _key_text(case),
            "brief": str(case.data.get("brief", "")),
            "figures": captions,
            "repeats": repeats,
            "live": visual.live_rows(case, pairs),
            "items": review.disagreements(case, pairs),
            "settled": [
                {k: e.get(k) for k in ("at", "run", "id", "action")}
                for e in review.read_log(case)
            ],
            "can_settle": case.task in review.SETTLED_TASKS and not case.parts,
        }

    def figure(self, run: str, case_id: str, n: int) -> bytes:
        case, items = self.case_runs(run, case_id)
        figures = visual.case_figures(case, [r for _, r in items])
        if n >= len(figures) or not figures[n][1]:
            raise HTTPException(status_code=404, detail="no figure")
        return base64.b64decode(figures[n][1])

    def item_image(self, run: str, case_id: str, item_id: str) -> bytes:
        case, items = self.case_runs(run, case_id)
        item = next(
            (
                i
                for i in review.disagreements(case, [r for _, r in items])
                if i["id"] == item_id
            ),
            None,
        )
        if item is None:
            raise HTTPException(status_code=404, detail=f"no open {item_id!r}")
        png = _item_png(case, item, [r for _, r in items])
        if png is None:
            raise HTTPException(status_code=404, detail="no picture")
        return png

    def settle(self, run: str, case_id: str, body: Settle) -> Dict[str, Any]:
        case, items = self.case_runs(run, case_id)
        try:
            review.settle(case, [r for _, r in items], body.item, body.action, run=run)
        except review.SettleError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return self.case_detail(run, case_id)


TASK_TEXT = {
    "site_selection": "Pick sites for lamellae on a grid overview",
    "square_finding": "Find every grid square on an overview",
    "square_quality": "Label each numbered grid square",
    "fm_target": "Find the target in an FM stack",
    "correlation": "Place the FM target on the FIB image",
    "polishing_stop": "Say when to stop polishing, pass by pass",
    "target_present": "Say if the target is still in the lamella, pass by pass",
}


def did(score: Dict[str, Any]) -> str:
    """What one repeat did, in a few words, from its score's details."""
    d, task = score.get("details") or {}, score.get("task")
    if task == "site_selection" and "picks" in d:
        on_good = sum(1 for p in d.get("per_pick", []) if "good" in p)
        text = f"{on_good} of {d['picks']} picks on good squares"
        if d.get("unsafe_picks"):
            text += f", {d['unsafe_picks']} on bad"
        return text
    if task == "square_quality" and "total" in d:
        text = f"{d['correct']} of {d['total']} labels right"
        if d.get("unsafe"):
            text += f", damaged called good: {', '.join(d['unsafe'])}"
        return text
    if task == "polishing_stop" and "key" in d:
        stop = d.get("stopped_after")
        return f"stopped after {stop}" if stop else "never stopped"
    if task == "square_finding" and "recall" in d:
        return f"found {d['recall']:.0%} of squares, {d.get('extra', 0)} extra"
    if task == "fm_target" and "usable" in d:
        if "xy_error_um" in d:
            return f"{d['xy_error_um']} um from the target"
        return f"usable: {d['usable']}"
    if task == "correlation":
        if "across_um" in d:
            return f"{d['across_um']} um across, {d['along_um']} um along"
        if "error_um" in d:
            return f"{d['error_um']} um off"
    if task == "target_present" and "total" in d:
        return f"{d['correct']} of {d['total']} passes right"
    return ""


def _now_scores(case: Case, answer: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s.to_dict() for s in score_case(case, answer or {})]


def _now(case: Case, answer: Optional[Dict[str, Any]]) -> List[str]:
    """A repeat's results, scored with the case's key as it is now."""
    return [s.result for s in score_case(case, answer or {})]


def _key_text(case: Case) -> str:
    key = case.key
    if case.task == "site_selection":
        return f"{len(key.get('good') or [])} good squares, {len(key.get('bad') or [])} bad"
    if case.task == "square_quality":
        return f"{len(key.get('labels') or {})} squares labelled"
    if case.task == "polishing_stop":
        return "stop after " + " or ".join(stop_passes(key))
    return ""


def _item_png(
    case: Case, item: Dict[str, Any], runs: List[visual.Run]
) -> Optional[bytes]:
    """A close look at one disagreement: the site or square with the key and
    the agent's picks drawn, or the pass the agent stopped after."""
    if item["kind"] in ("site", "square"):
        image = visual._read_image(visual._input(case, "overview"))
        if item["kind"] == "site":
            x, y = item["x_px"], item["y_px"]
        else:
            square = next(
                (
                    s
                    for s in case.data.get("squares") or []
                    if str(s.get("square")) == item["square"]
                ),
                None,
            )
            if square is None:
                return None
            x, y = square["x_px"], square["y_px"]
        half = int(max(case.key.get("hit_radius_px", 100) * 3, 250))
        h, w = image.shape
        x0, x1 = max(0, int(x) - half), min(w, int(x) + half)
        y0, y1 = max(0, int(y) - half), min(h, int(y) + half)
        fig, ax = visual._figure(6, 6)
        visual._show(ax, image[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
        if item["kind"] == "site":
            from matplotlib.patches import Circle

            radius = float(case.key.get("hit_radius_px", 100))
            for kind, colour in (("good", visual.KEY), ("bad", "red")):
                for s in case.key.get(kind) or []:
                    ax.add_patch(
                        Circle(
                            (s["x_px"], s["y_px"]),
                            radius,
                            fill=False,
                            color=colour,
                            lw=1.5,
                        )
                    )
            for result, answer in runs:
                for pick in (answer or {}).get("picks") or []:
                    try:
                        ax.plot(
                            float(pick["x_px"]),
                            float(pick["y_px"]),
                            "x",
                            color="#b07cff",
                            ms=9,
                            mew=2,
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        return base64.b64decode(visual._png(fig))
    if item["kind"] == "stop":
        entry = next(
            (p for p in case.data.get("passes") or [] if p.get("name") == item["pass"]),
            None,
        )
        path = entry and (entry.get("sem") or entry.get("fib"))
        if not path or not (case.path / path).is_file():
            return None
        image, _ = visual._shrink(visual._read_image(case.path / path), 800)
        fig, ax = visual._figure(5, 4)
        visual._show(ax, image)
        ax.set_title(f"after {item['pass']}", fontsize=9)
        return base64.b64decode(visual._png(fig))
    return None


def build_app(runs: Path, cases: Optional[Path] = None) -> FastAPI:
    reviewer = Reviewer(runs, cases)
    app = FastAPI(title="fibsem evals review")
    api = APIRouter(prefix="/evals")

    @api.get("/runs")
    def runs_list():
        return reviewer.runs_list()

    @api.get("/runs/{run}")
    def run_summary(run: str):
        return reviewer.run_summary(run)

    @api.get("/runs/{run}/cases/{case_id}")
    def case_detail(run: str, case_id: str):
        return reviewer.case_detail(run, case_id)

    @api.get("/runs/{run}/cases/{case_id}/figures/{n}.png")
    def figure(run: str, case_id: str, n: int):
        return Response(reviewer.figure(run, case_id, n), media_type="image/png")

    @api.get("/runs/{run}/cases/{case_id}/items/{item_id}.png")
    def item_image(run: str, case_id: str, item_id: str):
        return Response(
            reviewer.item_image(run, case_id, item_id), media_type="image/png"
        )

    @api.post("/runs/{run}/cases/{case_id}/settle")
    def settle(run: str, case_id: str, body: Settle):
        return reviewer.settle(run, case_id, body)

    app.include_router(api)

    app.include_router(web_app_router())

    @app.get("/")
    def home():
        return RedirectResponse("/ui/evals")

    return app


def serve(runs: Path, cases: Optional[Path] = None, port: int = PORT) -> None:
    import uvicorn

    app = build_app(runs, cases)
    print(f"Eval review: http://{HOST}:{port}/ui/evals  (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
