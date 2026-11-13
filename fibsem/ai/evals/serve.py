"""``python -m fibsem.ai.evals serve``: the eval review page (``/ui/evals``)
and the routes it reads, on this computer only. No App Server or microscope.

It reads run folders (what ``run --out`` wrote) under one folder, and the
cases those runs used, each scored with its key as it is now. Settling a
disagreement writes the case's key (``fibsem.ai.evals.review``).
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from PIL import Image
from pydantic import BaseModel

from fibsem.ai.evals import review, tasks, trace, visual
from fibsem.ai.evals.cases import Case, find_cases, pixel_size_um
from fibsem.ai.evals.runner import RESULT
from fibsem.ai.evals.scoring import score_case
from fibsem.server.web_app import web_app_router

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8765


class Settle(BaseModel):
    item: str
    action: str
    # For "place": the point clicked, in the image's px.
    x_px: Optional[float] = None
    y_px: Optional[float] = None


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
                detail=f"case {case_id!r} is not in {self.cases_root}"
                if self.cases_root
                else f"case {case_id!r} not found: start serve with --cases",
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
            "cases_root": str(self.cases_root or ""),
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
            "task_text": getattr(tasks.get(case.task), "title", case.task),
            "key_text": _key_text(case),
            "brief": str(case.data.get("brief", "")),
            "figures": captions,
            "repeats": repeats,
            "live": visual.live_rows(case, pairs),
            "items": [
                dict(i, view=_view(case, i, pairs))
                for i in review.disagreements(case, pairs)
            ],
            "settled": [
                {k: e.get(k) for k in ("at", "run", "id", "action")}
                for e in review.read_log(case)
            ],
            "last": _last(case),
            "can_settle": review.can_settle(case),
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
            at = None if body.x_px is None else (body.x_px, body.y_px)
            review.settle(
                case, [r for _, r in items], body.item, body.action, run=run, at=at
            )
        except review.SettleError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return self.case_detail(run, case_id)

    def traces(self, run: str, case_id: str) -> List[Dict[str, Any]]:
        """What each repeat did, from its agent log: skills read, what it
        looked at, tool calls, what it said and what was refused."""
        _, items = self.case_runs(run, case_id)
        out = []
        for folder, (result, _) in items:
            got = trace.read_run(folder)
            out.append(
                {
                    "repeat": result.get("repeat"),
                    "skills": got.skills,
                    "looked_at": got.looked_at,
                    "tools": got.tools.most_common(),
                    "said": got.said,
                    "failed": got.failed,
                    "has_log": (folder / trace.AGENT_LOG).is_file(),
                }
            )
        return out

    def undo(self, run: str, case_id: str) -> Dict[str, Any]:
        case, _ = self.case_runs(run, case_id)
        try:
            review.undo(case, run=run)
        except review.SettleError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return self.case_detail(run, case_id)


def _last(case: Case) -> Optional[Dict[str, Any]]:
    last = review.last_settled(case)
    return {k: last.get(k) for k in ("id", "action")} if last else None


def did(score: Dict[str, Any]) -> str:
    """What one repeat did, in a few words, from its score's details."""
    task = tasks.get(score.get("task"))
    return task.summary(score.get("details") or {}) if task else ""


def _now_scores(case: Case, answer: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s.to_dict() for s in score_case(case, answer or {})]


def _now(case: Case, answer: Optional[Dict[str, Any]]) -> List[str]:
    """A repeat's results, scored with the case's key as it is now."""
    return [s.result for s in score_case(case, answer or {})]


def _key_text(case: Case) -> str:
    task = tasks.get(case.task)
    return task.key_text(case.key) if task and task.key_text else ""


# Disagreements drawn in the page: a plain crop, the marks laid over it in
# the browser, and a click on it places the key.
POINT_KINDS = ("point", "usable", "extra", "missed")


def _view(
    case: Case, item: Dict[str, Any], runs: List[visual.Run]
) -> Optional[Dict[str, Any]]:
    """The crop (image px) the item's picture shows and the marks on it."""
    if item["kind"] not in POINT_KINDS:
        return None
    marks: List[Dict[str, Any]] = []
    if case.task == "square_finding":
        pitch = float(case.key.get("pitch_px") or 100)
        half, radius = pitch * 1.5, pitch / 3
        for sq in case.key.get("squares") or []:
            marks.append(
                {
                    "x": sq["x_px"],
                    "y": sq["y_px"],
                    "kind": "optional" if sq.get("optional") else "key",
                    "r": radius,
                }
            )
        for result, answer in runs:
            for x, y in review._squares_found(answer):
                marks.append({"x": x, "y": y, "kind": "agent"})
    else:
        size = float(pixel_size_um(case) or 1)
        tol = float(
            case.scoring.get("xy_tolerance_um")
            or case.scoring.get("across_tolerance_um")
            or 1.0
        )
        radius = max(tol / size, 5)
        half = max(radius * 6, 80 if case.task == "fm_target" else 150)
        key_xy = review.key_point(case)
        if key_xy:
            marks.append({"x": key_xy[0], "y": key_xy[1], "kind": "key", "r": radius})
        for result, answer in runs:
            xy = review._xy(answer or {})
            if xy:
                marks.append(
                    {
                        "x": xy[0],
                        "y": xy[1],
                        "kind": "agent",
                        "repeat": result.get("repeat"),
                    }
                )
    centre = (item.get("x_px"), item.get("y_px"))
    if centre[0] is None:
        centre = review.key_point(case) or next(((m["x"], m["y"]) for m in marks), None)
    if centre is None:
        return None
    x0, y0 = int(centre[0] - half), int(centre[1] - half)
    side = int(2 * half)
    return {"x0": x0, "y0": y0, "x1": x0 + side, "y1": y0 + side, "marks": marks}


def _crop_png(case: Case, view: Dict[str, Any]) -> Optional[bytes]:
    """The view's crop as a grey PNG, padded where it runs off the image."""
    x0, y0, x1, y1 = view["x0"], view["y0"], view["x1"], view["y1"]
    if case.task == "fm_target":
        mid = ((x0 + x1) / 2, (y0 + y1) / 2)
        crop, extent = visual._stack_crop(case, mid[0], mid[1], (x1 - x0) // 2)
        cx0, cy0 = extent[0], extent[3]
    else:
        name = "fib_image" if case.task == "correlation" else "overview"
        image = visual._read_image(visual._input(case, name))
        h, w = image.shape
        cx0, cy0 = max(0, x0), max(0, y0)
        crop = image[cy0 : min(h, y1), cx0 : min(w, x1)]
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    if crop.size:
        out[
            cy0 - y0 : cy0 - y0 + crop.shape[0], cx0 - x0 : cx0 - x0 + crop.shape[1]
        ] = visual._stretch(crop)
    picture = Image.fromarray((out * 255).astype(np.uint8))
    if picture.width > 800:
        picture = picture.resize((800, 800))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _item_png(
    case: Case, item: Dict[str, Any], runs: List[visual.Run]
) -> Optional[bytes]:
    """A close look at one disagreement: the site or square with the key and
    the agent's picks drawn, the pass the agent stopped after, or the plain
    crop of a point the page draws on."""
    if item["kind"] in POINT_KINDS:
        view = _view(case, item, runs)
        return _crop_png(case, view) if view else None
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

    @api.get("/runs/{run}/cases/{case_id}/traces")
    def traces(run: str, case_id: str):
        return reviewer.traces(run, case_id)

    @api.post("/runs/{run}/cases/{case_id}/undo")
    def undo(run: str, case_id: str):
        return reviewer.undo(run, case_id)

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
