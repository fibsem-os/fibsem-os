"""The app-level router: observation of a running AutoLamella session.

Dependency direction is one-way — autolamella imports fibsem.server, never the
reverse — so this module never imports the application. The application's
``AutoLamellaAppContext`` (fibsem.applications.autolamella.server) is passed into
``build_server(app_context=...)`` by whoever hosts the server inside the app.

Every route is a thin pass-through: the context returns plain JSON-able
snapshots and owns all app knowledge; nothing here touches domain objects.
The level each router needs is applied where it is included, so auth stays in
one place (build_server). A route of an optional part (fibsem.server.parts) is
tagged with it and served only when the context implements that part.
"""

import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse

from fibsem.ai.agent.calls import current_call
from fibsem.server import files
from fibsem.server.auth import refuse_dashboard, take_control
from fibsem.server.parts import (
    GRIDS,
    ITEMS,
    LAMELLAE,
    MILLING_STAGES,
    POSES,
    served_parts,
)
from fibsem.server.responses import (
    AppStatus,
    DisplayImages,
    ExperimentSummary,
    FolderListing,
    ItemDetail,
    Queue,
    SavedImageMetadata,
    SavedImagePixels,
    SavedImagePosition,
    TaskHistory,
    TaskOutputs,
    answer,
)
from fibsem.util.timestamps import to_datetime
from fibsem.workflows import evidence

__all__ = [
    "build_app_config_router",
    "build_app_control_router",
    "build_app_router",
    "build_app_stop_router",
    "build_app_suggest_router",
]


def _served(router: APIRouter, context: Any) -> APIRouter:
    """*router* without the routes of parts *context* doesn't implement (each
    part route is tagged with its part)."""
    served = set(served_parts(context))
    router.routes = [
        route for route in router.routes if not set(getattr(route, "tags", ())) - served
    ]
    return router


def _grid_run_body(body: Dict[str, Any]):
    """Validate the shared body of the grid plan/start verbs, or raise 422."""
    task_names = body.get("task_names")
    grid_names = body.get("grid_names")
    screen_all = body.get("screen_all", False)
    ok = (
        isinstance(task_names, list)
        and bool(task_names)
        and all(isinstance(t, str) for t in task_names)
        and (
            grid_names is None
            or (
                isinstance(grid_names, list)
                and all(isinstance(g, str) for g in grid_names)
            )
        )
        and isinstance(screen_all, bool)
    )
    if not ok:
        raise _missing_field(
            "Pass task_names (non-empty list of grid task names); "
            "grid_names (list of strings) is optional — omitted means every "
            "recorded grid; screen_all (bool) inventories first and runs over "
            "every present grid, and takes no grid_names."
        )
    return task_names, grid_names, screen_all


# The image's acquisition metadata (JSON) rides on the JPEG answer, so one
# request gives the agent both the picture and what it was taken with.
IMAGE_METADATA_HEADER = "X-Image-Metadata"


def _jpeg_or_404(result: Dict[str, Any]) -> Response:
    """An image context answer as a real image/jpeg response (so a browser
    ``<img>`` can consume it), or a structured 404 carrying the valid names."""
    jpeg = result.get("jpeg")
    if jpeg is None:
        detail: Dict[str, Any] = {
            "error_type": "not_found",
            "message": result.get("error", "No such output image."),
        }
        for key in ("filenames", "grid_names"):
            if key in result:
                detail[key] = result[key]
        raise HTTPException(status_code=404, detail=detail)
    headers = {}
    if result.get("metadata") is not None:
        # The image's scale and state (images.image_metadata), for the agent's
        # image tools; a browser <img> ignores it.
        headers[IMAGE_METADATA_HEADER] = json.dumps(result["metadata"])
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)


_REFUSAL_STATUS = {
    "not_found": 404,
    "invalid": 422,
    "task_running": 409,
    "not_pending": 409,
    "not_allowed": 403,
    "no_experiment": 409,
}


def _missing_field(message: str) -> HTTPException:
    """The 422 for a request missing what a route needs, *message* saying what
    to pass."""
    return HTTPException(
        status_code=422, detail={"error_type": "missing_field", "message": message}
    )


def _recorded_or_not_found(result: Dict[str, Any]) -> Dict[str, Any]:
    """*result* of putting something on the record, or the 404 for the item it
    named that is not there."""
    if not result.get("recorded") and result.get("error"):
        raise HTTPException(
            status_code=404,
            detail={
                "error_type": "not_found",
                "message": result["error"],
                "item_names": result.get("item_names", []),
            },
        )
    return result


def _required_text(body: Dict[str, Any], key: str, what: str) -> str:
    """*body*'s non-empty string *key*, or the 422 saying to pass it (*what*)."""
    value = body.get(key)
    if not isinstance(value, str) or not value:
        raise _missing_field(f"Pass {key}: {what}.")
    return value


def _names(body: Dict[str, Any], key: str, what: str) -> List[str]:
    """*body*'s list of strings *key*, or the 422 saying to pass it (*what*)."""
    value = body.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _missing_field(f"Pass {key}: {what} (a list of strings).")
    return value


def _numbers(text: Optional[str]) -> Optional[List[float]]:
    """A comma-separated list of numbers from a query, or the 422 saying so."""
    if text is None:
        return None
    try:
        return [float(part) for part in text.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "invalid",
                "message": f"A region is four numbers, x,y,width,height; not {text!r}.",
            },
        )


def _answer_or_refusal(result: Dict[str, Any]) -> Dict[str, Any]:
    """A context answer, or its refusal as an HTTP error: ``error_type`` picks the
    status, ``error`` is the message, and any lists of valid names ride along."""
    error_type = result.get("error_type")
    if error_type is None:
        return result
    detail = {
        "error_type": error_type,
        "message": result.get("error", result.get("message", "")),
    }
    for key, value in result.items():
        if key.endswith(("_names", "_paths")) or key == "filenames":
            detail[key] = value
    raise HTTPException(status_code=_REFUSAL_STATUS.get(error_type, 422), detail=detail)


def _typed(result: Any) -> Response:
    """A context answer: a refusal raises (`_answer_or_refusal`), a structure
    answers as the codec writes it, and a dict as it is."""
    if isinstance(result, dict):
        return JSONResponse(jsonable_encoder(_answer_or_refusal(result)))
    return answer(result)


def build_app_router(context: Any) -> APIRouter:
    router = APIRouter(prefix="/app")

    @router.get("/workflow/status", response_model=AppStatus)
    def app_status():
        return _typed(context.status())

    @router.get("/workflow/queue", response_model=Queue)
    def app_queue():
        return _typed(context.queue())

    @router.get(
        "/experiment",
        response_model=getattr(context, "summary_type", ExperimentSummary),
    )
    def experiment_summary():
        return _typed(context.experiment_summary())

    @router.get("/experiment/task_history", response_model=TaskHistory)
    def task_history():
        return _typed(context.task_history())

    @router.get("/workflow/summary")
    def run_summary():
        return context.run_summary()

    @router.get("/experiment/protocol")
    def protocol():
        return context.protocol()

    @router.get("/log")
    def log(
        since: Optional[str] = None,
        until: Optional[str] = None,
        task_id: Optional[str] = None,
        level: Optional[str] = None,
        lines: int = 200,
    ):
        # The record cap is the transport's, as the events long-poll's
        # timeout is: it bounds what one answer can carry.
        result = context.log(
            since=since,
            until=until,
            task_id=task_id,
            level=level,
            lines=min(max(lines, 1), 2000),
        )
        if "error" in result:
            status = 404 if result.get("error_type") == "not_found" else 422
            detail = {
                "error_type": result["error_type"],
                "message": result["error"],
            }
            if "levels" in result:
                detail["levels"] = result["levels"]
            raise HTTPException(status_code=status, detail=detail)
        return result

    @router.get(
        "/experiment/items/{item_name}/outputs",
        tags=[ITEMS],
        response_model=getattr(context, "task_outputs_type", TaskOutputs),
    )
    def task_outputs(item_name: str):
        return _typed(context.task_outputs(item_name))

    @router.get(
        "/experiment/items/{item_name}",
        tags=[ITEMS],
        response_model=getattr(context, "item_detail_type", ItemDetail),
    )
    def item_detail(item_name: str):
        return _typed(context.item_detail(item_name))

    @router.get("/experiment/protocol/task_config/{task_name}")
    def protocol_task_config(task_name: str):
        return context.protocol_task_config(task_name)

    @router.get("/experiment/items/{item_name}/task_config/{task_name}", tags=[ITEMS])
    def item_task_config(item_name: str, task_name: str):
        return context.item_task_config(item_name, task_name)

    @router.get("/experiment/items/{item_name}/outputs/{filename}", tags=[ITEMS])
    def output_image(item_name: str, filename: str):
        # Serves actual image/jpeg (not a JSON payload) so a browser <img>
        # can consume it; filename must be a basename the item's own
        # outputs listing names — the context refuses anything else.
        return _jpeg_or_404(context.output_image(item_name, filename))

    @router.get("/experiment/items/{item_name}/history", tags=[LAMELLAE])
    def item_history(
        item_name: str,
        show: Optional[str] = None,
        status: Optional[str] = None,
        task: Optional[str] = None,
        decided_by: Optional[str] = None,
        order: Optional[str] = None,
    ):
        return context.item_history(
            item_name,
            {
                "show": show,
                "status": status,
                "task": task,
                "decided_by": decided_by,
                "order": order,
            },
        )

    @router.post("/experiment/items/{item_name}/thickness", tags=[LAMELLAE])
    def measure_thickness(item_name: str, body: Dict[str, Any]):
        # Computed over a saved file, nothing moves; a POST because the result
        # is saved beside the image, as an autofocus run is.
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise _missing_field("path is required")
        window = {}
        for key in ("width_um", "height_um"):
            value = body.get(key)
            if value is None:
                continue
            try:
                window[key] = float(value)
            except (TypeError, ValueError):
                raise _missing_field(f"{key} is a number of microns.") from None
        return _answer_or_refusal(context.measure_thickness(item_name, path, **window))

    @router.get("/experiment/grids", tags=[GRIDS])
    def grids():
        return context.grids()

    @router.get("/experiment/grids/{grid_name}", tags=[GRIDS])
    def grid_detail(grid_name: str):
        return context.grid_detail(grid_name)

    @router.get("/experiment/grids/{grid_name}/outputs/{filename}", tags=[GRIDS])
    def grid_output_image(grid_name: str, filename: str):
        return _jpeg_or_404(context.grid_output_image(grid_name, filename))

    # {filename:path}: any saved image's path in the experiment, slashes and all.
    @router.get(
        "/experiment/grids/{grid_name}/outputs/{filename:path}/stage_position",
        tags=[GRIDS],
    )
    def overview_stage_position(
        grid_name: str,
        filename: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ):
        # The inverse of the markers: a point on the overview as a stage position.
        px = None if x is None or y is None else [x, y]
        fraction = None if fx is None or fy is None else [fx, fy]
        return _answer_or_refusal(
            context.overview_stage_position(grid_name, filename, px, fraction)
        )

    @router.get(
        "/experiment/grids/{grid_name}/outputs/{filename:path}/markers", tags=[GRIDS]
    )
    def grid_markers(grid_name: str, filename: str):
        return context.grid_markers(grid_name, filename)

    @router.get("/experiment/grid_protocol/task_config/{task_name}", tags=[GRIDS])
    def grid_task_config(task_name: str):
        return _answer_or_refusal(context.grid_task_config(task_name))

    @router.get("/experiment/protocols")
    def protocols():
        return context.protocols()

    @router.get("/experiment/recent")
    def recent_experiments():
        return {"items": context.recent_experiments()}

    @router.get("/experiment/all")
    def saved_experiments(
        instrument: Optional[str] = None,
        operator: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ):
        # instrument: serial or model; since/until: an ISO date or time.
        bounds = {}
        for name, value in (("since", since), ("until", until)):
            if value is not None:
                bounds[name] = to_datetime(value)
                if bounds[name] is None:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error_type": "invalid",
                            "error": f"{name} is an ISO date or time, not {value!r}.",
                        },
                    )
        return context.saved_experiments(
            instrument=instrument, operator=operator, **bounds
        )

    @router.get("/experiment/images", response_model=DisplayImages)
    def display_images():
        return _typed(context.display_images())

    @router.get("/experiment/images/metadata", response_model=SavedImageMetadata)
    def image_metadata(path: str):
        found = context.image_metadata(path)
        error = found.get("error_type") if isinstance(found, dict) else None
        if error is not None:
            status = 403 if error == "not_allowed" else 404
            raise HTTPException(status_code=status, detail=found)
        return _typed(found)

    @router.get("/experiment/images/stage_position", response_model=SavedImagePosition)
    def image_stage_position(
        path: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ):
        # Any saved image in the experiment, beam or FM: a point on it as a
        # stage position, from the image's own metadata.
        px = None if x is None or y is None else [x, y]
        fraction = None if fx is None or fy is None else [fx, fy]
        return _typed(context.image_stage_position(path, px, fraction))

    @router.post("/experiment/images/markers", tags=[LAMELLAE])
    def render_markers(body: Dict[str, Any]):
        # A POST because the picture is saved beside the image, as a thickness
        # result is.
        path, grid_name = body.get("path"), body.get("grid_name")
        if not isinstance(path, str) or not path:
            raise HTTPException(
                status_code=422,
                detail={"error_type": "missing_field", "message": "path is required"},
            )
        if grid_name is not None and not isinstance(grid_name, str):
            raise HTTPException(
                status_code=422,
                detail={"error_type": "invalid", "message": "grid_name is a string"},
            )
        return _answer_or_refusal(context.render_markers(path, grid_name))

    @router.post("/experiment/images/pixels", response_model=SavedImagePixels)
    def image_pixels(body: Dict[str, Any]):
        # A POST on the read router because it computes, never acts: stage
        # positions as points on a saved image, a list a GET could not carry.
        path = _required_text(body, "path", "a saved image inside the experiment")
        positions = body.get("stage_positions")
        if (
            not isinstance(positions, list)
            or not positions
            or not all(isinstance(p, dict) for p in positions)
        ):
            raise _missing_field(
                "Pass stage_positions: a non-empty list of stage positions "
                "({x, y, z, r, t, coordinate_system})."
            )
        return _typed(context.image_pixels(path, positions))

    # -- an experiment's files: the open one's, or a past one's, read only ------

    def _files_answer(experiment, read, *args, **kwargs):
        try:
            return read(context.experiment_root(experiment), *args, **kwargs)
        except files.FileRefusal as refusal:
            raise HTTPException(status_code=refusal.status, detail=refusal.detail)

    def _record_read(experiment, path, raw):
        # A past experiment's every read, and every raw file, go on the record.
        if experiment is not None or raw:
            payload = {"path": path, "raw": raw}
            if experiment is not None:
                payload["experiment"] = experiment
            context.record_event("experiment_file_read", payload)

    @router.get("/experiment/files", response_model=FolderListing)
    def list_files(dir: str = "", experiment: Optional[str] = None):
        listing = _files_answer(experiment, files.list_folder, dir)
        _record_read(experiment, dir, False)
        # As listed: a file has no images count, rather than a null one.
        return JSONResponse(listing)

    @router.get("/experiment/files/{path:path}")
    def read_file(
        request: Request,
        path: str,
        box: Optional[str] = None,
        fraction: Optional[str] = None,
        width: Optional[int] = None,
        raw: bool = False,
        experiment: Optional[str] = None,
    ):
        if raw:
            refuse_dashboard(request)
            # The file as it is, its experiment's folder name and its path in
            # it in the headers: where a fetch keeps it.
            root = _files_answer(experiment, lambda root: root)
            target, relative = _files_answer(experiment, files.raw_file, path)
            _record_read(experiment, relative, True)
            return FileResponse(
                target,
                headers={
                    "X-Experiment": quote(os.path.basename(os.path.normpath(root))),
                    "X-Experiment-Path": quote(relative),
                },
            )
        # A region is "x,y,width,height": source pixels (box) or 0-1 (fraction).
        # width: the preview's at most, for a thumbnail or a full-size view.
        answer = _files_answer(
            experiment,
            files.read,
            path,
            box=_numbers(box),
            fraction=_numbers(fraction),
            max_width=files.preview_width(width),
        )
        _record_read(experiment, path, False)
        return answer

    @router.get("/workflow/proposals")
    def proposals():
        return context.proposals()

    @router.post("/workflow/grids/plan", tags=[GRIDS])
    def grid_workflow_plan(body: Dict[str, Any]):
        # A POST on the read router because it computes, never acts: the
        # preflight dialog's content for a grid run, from a body a GET could
        # not carry cleanly. No hardware is read.
        task_names, grid_names, screen_all = _grid_run_body(body)
        return context.grid_workflow_plan(task_names, grid_names, screen_all)

    @router.post("/experiment/notes")
    def add_note(body: Dict[str, Any]):
        # Observations for the record — event stream + experiment log. On the
        # read router, a POST the dashboard token is refused: a watching agent
        # notes what it sees (FIB-1035), and a note changes nothing.
        text = body.get("text")
        item_name = body.get("item_name")
        reply_to = body.get("reply_to")
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > 4000
            or (item_name is not None and not isinstance(item_name, str))
            or (reply_to is not None and not isinstance(reply_to, str))
        ):
            raise _missing_field(
                "Pass text (a non-empty string, at most 4000 characters); "
                "item_name and reply_to (strings) are optional."
            )
        cited = cited_evidence(context)
        result = context.add_note(
            text,
            item_name,
            reply_to,
            **({"evidence": cited["evidence_id"]} if cited else {}),
        )
        return _recorded_or_not_found(result)

    @router.post("/experiment/evidence")
    def add_evidence(body: Dict[str, Any]):
        # What the agent judged from, kept in the experiment: an image of the
        # experiment's or one it made (uploaded, saved here), with marks on it.
        # Changes nothing, like a note.
        item_name = body.get("item_name")
        if item_name is not None and not isinstance(item_name, str):
            raise _missing_field("item_name, when given, is a string.")

        def save() -> Dict[str, Any]:
            return evidence.save(
                lambda: context.experiment_path,
                image=body.get("image"),
                upload=body.get("upload"),
                marks=body.get("marks"),
                source=body.get("source"),
                item_name=item_name,
            )

        try:
            result = context.add_evidence(item_name, save)
        except evidence.EvidenceRefused as refusal:
            raise _evidence_refused(refusal)
        return _recorded_or_not_found(result)

    @router.get("/events")
    def events(since: int = 0, timeout: float = 0.0):
        # A long-poll: parks up to `timeout` seconds waiting for news. Capped
        # here because the park occupies a threadpool worker — a transport
        # concern, so the transport owns the ceiling.
        return context.events(since=since, timeout=min(max(timeout, 0.0), 30.0))

    @router.get("/events/history")
    def event_history(before: Optional[int] = None, limit: int = 500):
        # The run before the live buffer, from the experiment's events file:
        # what a page opened in the morning needs to show the night.
        return context.event_history(before=before, limit=limit)

    return _served(router, context)


def _evidence_refused(refusal: evidence.EvidenceRefused) -> HTTPException:
    return HTTPException(status_code=refusal.status_code, detail=refusal.detail)


def cited_evidence(context: Any) -> Optional[Dict[str, Any]]:
    """The record of the evidence the agent's call cites
    (`calls.EVIDENCE_HEADER`); None when it cites none."""
    call = current_call()
    if call is None or not call.evidence:
        return None
    try:
        return evidence.load(
            lambda: getattr(context, "experiment_path", None), call.evidence
        )
    except evidence.EvidenceRefused as refusal:
        raise _evidence_refused(refusal)


def build_app_control_router(context: Any) -> APIRouter:
    """Routes that act on the session rather than observe it.

    Mounted (by build_server) at Drive: the operator raises the level from the
    Agent chip (FIB-1035).
    """
    router = APIRouter(prefix="/app")

    @router.post("/workflow/decide")
    def decide(body: Dict[str, Any]):
        # The same function the operator's decision goes through, on the main thread, blocking
        # until applied. Item by id, never by name: names change.
        item_id = body.get("item_id")
        task_name = body.get("task_name")
        outcome = body.get("outcome")
        # What the decision is on: the proposal's own id, or -- from a caller
        # that has not learned it -- the run it came from. The id is the one to
        # pass: a task may ask several proposals in one run.
        proposal_id = body.get("proposal_id") or ""
        task_id = body.get("task_id") or ""
        named = [v for v in (proposal_id, task_id) if v]
        if not all(isinstance(v, str) and v for v in (item_id, task_name, outcome)) or (
            not named or not all(isinstance(v, str) for v in named)
        ):
            raise _missing_field(
                "Pass item_id, task_name, proposal_id and outcome "
                "(Confirmed or Rejected) from GET /app/workflow/proposals."
            )
        values = body.get("values")
        if values is not None and not isinstance(values, dict):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_value",
                    "message": "values, when given, must be a JSON object keyed by "
                    "value name, e.g. {'poi': {'x': ..., 'y': ...}} in metres.",
                },
            )
        patch = body.get("patch")
        if patch is not None and not isinstance(patch, dict):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_value",
                    "message": 'patch, when given, is {"<dotted path>": value}.',
                },
            )
        action = body.get("action") or ""
        if not isinstance(action, str):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_value",
                    "message": "action, when given, is one of the proposal's actions.",
                },
            )
        if outcome == "Rejected" and not str(body.get("reason", "")).strip():
            raise _missing_field("A reject needs a reason; it is recorded on the item.")
        cited = cited_evidence(context)
        result = context.decide(
            item_id,
            task_name,
            outcome,
            values=values,
            reason=str(body.get("reason", "")),
            actor=str(body.get("actor", "")),
            task_id=task_id,
            proposal_id=proposal_id,
            action=action,
            patch=patch,
            **({"evidence": cited["evidence_id"]} if cited else {}),
        )
        if result.get("invalid_value"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_value",
                    "message": result["invalid_value"],
                },
            )
        if not result.get("applied", False) and result.get("available", True):
            error_type = result.get("error_type") or (
                "running" if result.get("running") else "not_pending"
            )
            raise HTTPException(
                # a request that cannot be decided as given is 422; one that
                # the state refuses (running, re-run since, nothing pending) 409
                status_code=422
                if error_type in ("missing_field", "invalid_value")
                else 409,
                detail={
                    "error_type": error_type,
                    "message": result.get("reason", "Not applied."),
                },
            )
        return result

    @router.post("/workflow/start")
    def start_workflow(body: Dict[str, Any]):
        task_names = body.get("task_names")
        item_names = body.get("item_names")
        if (
            not isinstance(task_names, list)
            or not task_names
            or not all(isinstance(t, str) for t in task_names)
            or (
                item_names is not None
                and not (
                    isinstance(item_names, list)
                    and all(isinstance(i, str) for i in item_names)
                )
            )
        ):
            raise _missing_field(
                "Pass task_names (non-empty list of strings); "
                "item_names (list of strings) is optional — omitted "
                "means every item."
            )
        return context.start_workflow(task_names, item_names)

    @router.post("/workflow/grids/start", tags=[GRIDS])
    def start_grid_workflow(body: Dict[str, Any]):
        # The Grids view's Run / Screen all grids, remotely. Shares the lamella
        # run's worker slot, so the context refuses it while any run is live.
        task_names, grid_names, screen_all = _grid_run_body(body)
        unload_at_end = body.get("unload_at_end", False)
        report_at_end = body.get("report_at_end", False)
        if not isinstance(unload_at_end, bool) or not isinstance(report_at_end, bool):
            raise _missing_field("unload_at_end and report_at_end are booleans.")
        return context.start_grid_workflow(
            task_names,
            grid_names,
            screen_all,
            unload_at_end=unload_at_end,
            report_at_end=report_at_end,
        )

    @router.post("/experiment/protocol/supervision")
    def set_supervision(body: Dict[str, Any]):
        task_name = body.get("task_name")
        if not isinstance(task_name, str) or "supervise" not in body:
            raise _missing_field("Pass task_name (string) and supervise (boolean).")
        return context.set_supervision(task_name, bool(body["supervise"]))

    @router.post("/workflow/queue/requeue")
    def requeue_task(body: Dict[str, Any]):
        item_name = body.get("item_name")
        task_name = body.get("task_name")
        if not isinstance(item_name, str) or not isinstance(task_name, str):
            raise _missing_field(
                "Pass item_name and task_name (strings); front (boolean) is optional."
            )
        result = context.requeue_task(
            item_name, task_name, front=bool(body.get("front", False))
        )
        if result.get("error_type") == "already_queued":
            raise HTTPException(
                status_code=409,
                detail={"error_type": "already_queued", "message": result["reason"]},
            )
        return result

    @router.post("/workflow/queue/edit")
    def edit_queue(body: Dict[str, Any]):
        item_name = _required_text(body, "item_name", "the item whose task to edit")
        task_name = _required_text(body, "task_name", "the waiting task to edit")
        action = _required_text(
            body, "action", "remove, move_up, move_down or run_next"
        )
        return _answer_or_refusal(context.edit_queue(item_name, task_name, action))

    @router.post("/workflow/queue/grids", tags=[GRIDS])
    def queue_grid_tasks(body: Dict[str, Any]):
        # The Grids view's Add to queue on a running grid run: at the end.
        task_names = _names(body, "task_names", "the grid tasks to add")
        grid_names = _names(body, "grid_names", "the grids to add them for")
        return context.queue_grid_tasks(task_names, grid_names)

    return _served(router, context)


def build_app_suggest_router(context: Any) -> APIRouter:
    """Suggesting on a pending proposal. Mounted (by build_server) for Assist
    and Drive: a suggestion decides nothing, so it needs no request."""
    router = APIRouter(prefix="/app")

    @router.post("/workflow/proposals/suggest")
    def suggest(body: Dict[str, Any]):
        item_id = body.get("item_id")
        task_name = body.get("task_name")
        proposal_id = body.get("proposal_id")
        if not all(isinstance(v, str) and v for v in (item_id, task_name, proposal_id)):
            raise _missing_field(
                "Pass item_id, task_name and proposal_id from "
                "GET /app/workflow/proposals."
            )
        reason = body.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise _missing_field(
                "Pass a reason: the operator reads it when deciding whether to "
                "approve the suggestion."
            )
        for name, shape in (
            ("values", "keyed by value name"),
            ("patch", "{path: value}"),
        ):
            given = body.get(name)
            if given is not None and not isinstance(given, dict):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error_type": "invalid_value",
                        "message": f"{name}, when given, is a JSON object {shape}.",
                    },
                )
        cited = cited_evidence(context)
        result = context.suggest(
            item_id,
            task_name,
            proposal_id,
            reason=reason.strip(),
            outcome=str(body.get("outcome") or "Confirmed"),
            values=body.get("values"),
            patch=body.get("patch"),
            action=str(body.get("action") or ""),
            evidence=cited,
            actor=str(body.get("actor", "")),
        )
        if not result.get("applied", False) and result.get("available", True):
            error_type = result.get("error_type") or "invalid_value"
            raise HTTPException(
                status_code=422
                if error_type in ("missing_field", "invalid_value")
                else 409,
                detail={
                    "error_type": error_type,
                    "message": result.get("reason", "Not suggested."),
                },
            )
        return result

    return _served(router, context)


def _stop_reason(body: Optional[Dict[str, Any]]) -> str:
    """Why a stop was asked for, for the stopped task's record."""
    reason = body.get("reason") if isinstance(body, dict) else None
    return reason if isinstance(reason, str) else ""


def build_app_stop_router(context: Any) -> APIRouter:
    """Stopping the run. Mounted (by build_server) at Drive, with a reason: the
    workflow is never the agent's own, so at Assist the stop goes in a request
    the operator approves (FIB-1283); a watching agent stops nothing (FIB-1035)."""
    router = APIRouter(prefix="/app")

    @router.post("/workflow/stop")
    def stop_workflow(body: Optional[Dict[str, Any]] = None):
        # The same path as the GUI's Stop button, and never waits on the
        # command lock. The operator's own Stop is never gated.
        return context.stop_workflow(_stop_reason(body))

    @router.post("/workflow/stop_task")
    def stop_task(body: Dict[str, Any]):
        # The timeline's Stop Task: the task running, and the run carries on.
        # It names the task meant, so a stop landing late stops nothing else.
        item_name = _required_text(body, "item_name", "the item of the task running")
        task_name = _required_text(body, "task_name", "the task running")
        return context.stop_task(item_name, task_name, _stop_reason(body))

    return _served(router, context)


def build_app_config_router(context: Any) -> APIRouter:
    """Routes that edit configuration (protocols, item configs, schedules,
    run options). Mounted at Drive, like the control router (FIB-864)."""
    router = APIRouter(prefix="/app")

    def _validated(body: Dict[str, Any]):
        patch = body.get("patch")
        version = body.get("version")
        if (
            not isinstance(patch, dict)
            or not patch
            or not all(isinstance(k, str) for k in patch)
            or not isinstance(version, str)
        ):
            raise _missing_field(
                "Pass patch (a non-empty object of dotted-path: "
                "value entries) and version (from the config read it was "
                "written against)."
            )
        return patch, version

    @router.post("/experiment/protocol/task_config/{task_name}")
    def patch_protocol_task_config(task_name: str, body: Dict[str, Any]):
        patch, version = _validated(body)
        return _refused_or(
            context.apply_protocol_task_config_patch(task_name, patch, version)
        )

    @router.post("/experiment/items/{item_name}/task_config/{task_name}", tags=[ITEMS])
    def patch_item_task_config(item_name: str, task_name: str, body: Dict[str, Any]):
        patch, version = _validated(body)
        return _refused_or(
            context.apply_item_task_config_patch(item_name, task_name, patch, version)
        )

    @router.post("/workflow/schedule")
    def set_task_schedule(body: Dict[str, Any]):
        # Workflow structure, not a config document: a verb like
        # /app/experiment/protocol/supervision. scheduled_at is ISO-8601 or null
        # to clear.
        task_name = body.get("task_name")
        scheduled_at = body.get("scheduled_at")
        if not isinstance(task_name, str) or (
            scheduled_at is not None and not isinstance(scheduled_at, str)
        ):
            raise _missing_field(
                "Pass task_name (string) and scheduled_at "
                "(ISO-8601 string, or null to clear)."
            )
        result = context.set_task_schedule(task_name, scheduled_at)
        if result.get("invalid_value"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_value",
                    "message": result["invalid_value"],
                },
            )
        if not result.get("applied") and result.get("error"):
            raise HTTPException(
                status_code=404,
                detail={
                    "error_type": "not_found",
                    "message": result["error"],
                    "task_names": result.get("task_names", []),
                },
            )
        return result

    @router.post("/workflow/options")
    def set_workflow_options(body: Dict[str, Any]):
        # The workflow's run options, beside the schedule: whether the run
        # summary dialog opens, and whether the beams go off at the end.
        options: Dict[str, Any] = {}
        for key in ("show_run_summary", "turn_beams_off"):
            if key in body:
                if not isinstance(body[key], bool):
                    raise _missing_field(f"{key} is a boolean.")
                options[key] = body[key]
        if not options:
            raise _missing_field(
                "Pass any of show_run_summary (boolean), turn_beams_off (boolean)."
            )
        return context.set_workflow_options(**options)

    # -- the workflow's tasks: the protocol editor and the Workflow tab ---------

    @router.post("/experiment/protocol/tasks")
    def add_task(body: Dict[str, Any]):
        task_type = _required_text(body, "task_type", "a registered task type")
        task_name = _required_text(body, "task_name", "the new task's name")
        return _answer_or_refusal(context.add_task(task_type, task_name))

    @router.delete("/experiment/protocol/tasks/{task_name}")
    def remove_task(task_name: str):
        return _answer_or_refusal(context.remove_task(task_name))

    @router.post("/experiment/protocol/tasks/reorder")
    def reorder_tasks(body: Dict[str, Any]):
        order = _names(body, "order", "every workflow task, in the new order")
        return _answer_or_refusal(context.reorder_tasks(order))

    @router.post("/experiment/protocol/tasks/{task_name}/requirements")
    def set_task_requirements(task_name: str, body: Dict[str, Any]):
        requires = body.get("requires")
        required = body.get("required")
        if (requires is None and required is None) or (
            required is not None and not isinstance(required, bool)
        ):
            raise _missing_field(
                "Pass requires (the earlier tasks it needs, a list) and/or "
                "required (boolean)."
            )
        if requires is not None:
            requires = _names(body, "requires", "the earlier tasks it needs")
        return _answer_or_refusal(
            context.set_task_requirements(task_name, requires, required)
        )

    @router.post("/experiment/protocol/lamella_defaults", tags=[LAMELLAE])
    def set_lamella_defaults(body: Dict[str, Any]):
        if not body:
            raise _missing_field(
                "Pass any of use_petname, name_prefix, poi, alignment_area."
            )
        return _answer_or_refusal(context.set_lamella_defaults(body))

    @router.post("/experiment/protocol/save")
    def save_protocol_as(body: Dict[str, Any]):
        name = _required_text(body, "name", "a file name for the protocol folder")
        overwrite = body.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise _missing_field("overwrite is a boolean.")
        return _answer_or_refusal(context.save_protocol_as(name, overwrite))

    # -- the grid protocol: grid task settings, order and requirements ---------

    @router.post("/experiment/grid_protocol/task_config/{task_name}", tags=[GRIDS])
    def patch_grid_task_config(task_name: str, body: Dict[str, Any]):
        patch, version = _validated(body)
        result = context.apply_grid_task_config_patch(task_name, patch, version)
        return _refused_or(_answer_or_refusal(result))

    @router.post("/experiment/grid_protocol/tasks/reorder", tags=[GRIDS])
    def reorder_grid_tasks(body: Dict[str, Any]):
        order = _names(body, "order", "every grid task, in the new order")
        return _answer_or_refusal(context.reorder_grid_tasks(order))

    @router.post(
        "/experiment/grid_protocol/tasks/{task_name}/requirements", tags=[GRIDS]
    )
    def set_grid_task_requirements(task_name: str, body: Dict[str, Any]):
        requires = _names(body, "requires", "the grid tasks it needs first")
        return _answer_or_refusal(
            context.set_grid_task_requirements(task_name, requires)
        )

    def _validated_reorder(body: Dict[str, Any]):
        milling_key = body.get("milling_key")
        order = body.get("order")
        version = body.get("version")
        if (
            not isinstance(milling_key, str)
            or not isinstance(order, list)
            or not order
            or not all(isinstance(n, str) for n in order)
            or not isinstance(version, str)
        ):
            raise _missing_field(
                "Pass milling_key (string), order (the current "
                "stage names in their new sequence), and version (from "
                "the config read)."
            )
        return milling_key, order, version

    @router.post(
        "/experiment/items/{item_name}/task_config/{task_name}/stages/reorder",
        tags=[MILLING_STAGES],
    )
    def reorder_item_stages(item_name: str, task_name: str, body: Dict[str, Any]):
        milling_key, order, version = _validated_reorder(body)
        return _refused_or(
            context.reorder_milling_stages(
                "item", item_name, task_name, milling_key, order, version
            )
        )

    @router.post(
        "/experiment/protocol/task_config/{task_name}/stages/reorder",
        tags=[MILLING_STAGES],
    )
    def reorder_protocol_stages(task_name: str, body: Dict[str, Any]):
        milling_key, order, version = _validated_reorder(body)
        return _refused_or(
            context.reorder_milling_stages(
                "protocol", "", task_name, milling_key, order, version
            )
        )

    @router.post("/experiment/items/{item_name}/apply_protocol", tags=[ITEMS])
    def apply_protocol_to_item(item_name: str, body: Dict[str, Any]):
        # Wholesale by design — the verb IS "replace with the protocol
        # defaults" — so no version field; refusals ride _refused_or.
        task_names = body.get("task_names")
        if task_names is not None and not (
            isinstance(task_names, list) and all(isinstance(t, str) for t in task_names)
        ):
            raise _missing_field(
                "task_names, when given, is a list of task "
                "name strings; omitted means every task the protocol "
                "defines."
            )
        return _refused_or(context.apply_protocol_to_item(item_name, task_names))

    # -- experiments and protocols: kept to known places -----------------------

    @router.post("/experiment/create")
    def create_experiment(body: Dict[str, Any]):
        name = _required_text(
            body, "name", "the new experiment's folder, under the experiments folder"
        )
        return _answer_or_refusal(
            context.create_experiment(
                name, protocol=body.get("protocol"), description=body.get("description")
            )
        )

    @router.post("/experiment/load")
    def load_experiment(body: Dict[str, Any]):
        path = _required_text(
            body, "path", "a path from the recent experiments or the experiments folder"
        )
        return _answer_or_refusal(context.load_experiment(path))

    @router.post("/experiment/protocol/load")
    def load_protocol(body: Dict[str, Any]):
        name = _required_text(body, "protocol", "a protocol name from the list")
        return _answer_or_refusal(context.load_protocol(name))

    # -- grid records: the Grids tab's inventory read and Rename ----------------

    @router.post("/experiment/grids", tags=[GRIDS])
    def add_grids():
        # A record for every grid the inventory holds and the experiment lacks.
        return _answer_or_refusal(context.add_grids())

    @router.post("/experiment/grids/{grid_name}/rename", tags=[GRIDS])
    def rename_grid(grid_name: str, body: Dict[str, Any]):
        name = _required_text(body, "name", "the grid's new name")
        return _answer_or_refusal(context.rename_grid(grid_name, name))

    @router.post("/experiment/grids/{grid_name}/quality", tags=[GRIDS])
    def set_grid_quality(grid_name: str, body: Dict[str, Any]):
        verdict = _required_text(
            body, "verdict", "GOOD, REWORK, FAILED, or UNASSESSED to clear"
        )
        reason = body.get("reason", "")
        if not isinstance(reason, str):
            raise _missing_field("reason is text.")
        return _answer_or_refusal(context.set_grid_quality(grid_name, verdict, reason))

    @router.post("/experiment/grids/{grid_name}/note", tags=[GRIDS])
    def set_grid_note(grid_name: str, body: Dict[str, Any]):
        note = body.get("note")
        if not isinstance(note, str) or len(note) > 4000:
            raise _missing_field(
                "Pass note: text, at most 4000 characters ('' clears)."
            )
        cited = cited_evidence(context)
        result = context.set_grid_note(
            grid_name, note, **({"evidence": cited["evidence_id"]} if cited else {})
        )
        call = current_call()
        if call is not None and note and result.get("applied"):
            call.recorded = True  # its note is the record of it
        return _answer_or_refusal(result)

    @router.delete("/experiment/grids/{grid_name}", tags=[GRIDS])
    def remove_grid(grid_name: str):
        return _answer_or_refusal(context.remove_grid(grid_name))

    # -- lamella positions: the Lamella tab's buttons ---------------------------

    @router.post("/experiment/items", tags=[ITEMS])
    def add_item(body: Dict[str, Any]):
        # Where: the stage now (neither field), a stage position, or a point on a
        # grid overview ({grid_name, filename, px | fraction}). The app names it
        # unless a name is given.
        stage_position = body.get("stage_position")
        overview = body.get("overview")
        if stage_position is not None and overview is not None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid",
                    "message": "Give stage_position or overview, not both.",
                },
            )
        return _answer_or_refusal(
            context.add_item(
                name=body.get("name"),
                stage_position=stage_position,
                overview=overview,
            )
        )

    @router.delete("/experiment/items/{item_name}", tags=[ITEMS])
    def remove_item(item_name: str):
        return _answer_or_refusal(context.remove_item(item_name))

    @router.post(
        "/experiment/items/{item_name}/move_to",
        tags=[POSES],
        dependencies=[Depends(take_control, scope="function")],
    )
    def move_to_item_pose(item_name: str, body: Dict[str, Any]):
        pose = _required_text(body, "pose", "a pose name from the item (MILLING, ...)")
        return _answer_or_refusal(context.move_to_item_pose(item_name, pose))

    @router.post(
        "/experiment/items/{item_name}/objective/move_to",
        tags=[POSES],
        dependencies=[Depends(take_control, scope="function")],
    )
    def move_objective_to_item(item_name: str):
        # The tab's objective Move to: the FM objective to the stored position.
        return _answer_or_refusal(context.move_objective_to_item(item_name))

    @router.post("/experiment/items/{item_name}/poses/{pose_name}", tags=[POSES])
    def set_item_pose(item_name: str, pose_name: str):
        # From the stage position now: the tab's Set pose.
        return _answer_or_refusal(context.set_item_pose(item_name, pose_name))

    @router.post("/experiment/items/{item_name}/objective_position", tags=[POSES])
    def set_item_objective_position(item_name: str):
        # From the FM objective now: the tab's Use current.
        return _answer_or_refusal(context.set_item_objective_position(item_name))

    @router.post(
        "/experiment/items/{item_name}/objective_position/apply_to_all", tags=[POSES]
    )
    def set_objective_position_on_all(item_name: str):
        # The item's stored position on every lamella: Apply to All Lamella.
        return _answer_or_refusal(context.set_objective_position_on_all(item_name))

    @router.post("/experiment/items/{item_name}/position", tags=[POSES])
    def move_item(item_name: str, body: Dict[str, Any]):
        # The lamella moved to a point, as on an overview; the stage stays.
        return _answer_or_refusal(
            context.move_item(
                item_name,
                stage_position=body.get("stage_position"),
                overview=body.get("overview"),
            )
        )

    @router.post("/experiment/items/{item_name}", tags=[ITEMS])
    def patch_item(item_name: str, body: Dict[str, Any]):
        # The item's own document (geometry, verdict, notes) — the write-side
        # mirror of GET /app/experiment/items/{item_name}, whose payload carries the
        # version this patch must echo.
        patch, version = _validated(body)
        return _refused_or(context.apply_item_patch(item_name, patch, version))

    @router.post("/experiment/items/{item_name}/correlation", tags=[LAMELLAE])
    def apply_item_correlation(item_name: str, body: Dict[str, Any]):
        # The correlation dialog's Continue: the item's POI to a saved run's
        # target, recorded as a correlation.
        run_path = body.get("run_path")
        if not isinstance(run_path, str) or not run_path:
            raise _missing_field(
                "Pass run_path, a run folder from correlation_correlate."
            )
        result = context.apply_item_correlation(item_name, run_path)
        if "error_type" in result:
            return _answer_or_refusal(result)
        return _refused_or(result)

    def _refused_or(result: Dict[str, Any]):
        if result.get("stale"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error_type": "stale_config",
                    "message": "The config changed since your read; re-read "
                    "it and write the patch against the current version.",
                },
            )
        if result.get("error_type") == "task_running":
            raise HTTPException(
                status_code=409,
                detail={
                    "error_type": "task_running",
                    "message": result.get("error", "That task is running."),
                },
            )
        if result.get("invalid_patch"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_patch",
                    "message": result["invalid_patch"],
                    "path": result.get("path"),
                },
            )
        if not result.get("applied") and result.get("error"):
            detail: Dict[str, Any] = {
                "error_type": "not_found",
                "message": result["error"],
            }
            for key in ("item_names", "task_names"):
                if key in result:
                    detail[key] = result[key]
            raise HTTPException(status_code=404, detail=detail)
        return result

    return _served(router, context)
