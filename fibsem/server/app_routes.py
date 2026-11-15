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

import functools
import gzip
import inspect
import json
import os
import tempfile
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute

from fibsem.ai.agent.calls import current_call
from fibsem.server import exports, files, planes
from fibsem.server.auth import (
    refusal_error,
    refuse_dashboard,
    require_operator,
    take_control,
)
from fibsem.server.bodies import (
    AnimationExportBody,
    ApplyProtocolBody,
    CorrelationRunBody,
    DecideBody,
    DrawBody,
    DrawOptions,
    EvidenceBody,
    ExperimentPathBody,
    GridNoteBody,
    GridPlanBody,
    GridQualityBody,
    GridRequirementsBody,
    GridRunBody,
    ImageExportBody,
    ImagePairBody,
    IntensityBody,
    MarkersBody,
    MillingDrawnBody,
    NameBody,
    NewExperimentBody,
    NewItemBody,
    NewTaskBody,
    NoteBody,
    OperatorMessageBody,
    PatchBody,
    PixelsBody,
    PlaceBody,
    PoseBody,
    ProfileBody,
    ProtocolNameBody,
    QueueEditBody,
    QueueGridsBody,
    RequeueBody,
    RequirementsBody,
    RunBody,
    RunPlanBody,
    SaveProtocolBody,
    ScheduleBody,
    StageOrderBody,
    StopBody,
    StopTaskBody,
    SuggestBody,
    SupervisionBody,
    TaskOrderBody,
    ThicknessBody,
    WorkflowOptionsBody,
)
from fibsem.server.parts import (
    GRIDS,
    ITEMS,
    LAMELLAE,
    MILLING_STAGES,
    POSES,
    REPLAY,
    served_parts,
)
from fibsem.server.responses import (
    AnimationFrames,
    AppStatus,
    ConfigEdit,
    Decided,
    DisplayImages,
    EventHistory,
    Events,
    ExperimentOpened,
    ExperimentSummary,
    FileEntry,
    FileRead,
    FolderListing,
    GridDetail,
    GridEdit,
    GridMarkers,
    GridPlan,
    Grids,
    GridTasksQueued,
    ImageDrawn,
    ImageExportForm,
    ImageMeasured,
    ImagePairDrawn,
    ItemDetail,
    ItemEdit,
    ItemHistory,
    ItemMillingDrawn,
    ItemStack,
    LamellaDefaultsSet,
    LogRead,
    MarkersDrawn,
    OptionsSet,
    OverviewStagePosition,
    Proposals,
    ProtocolFiles,
    ProtocolLoaded,
    ProtocolSaved,
    ProtocolView,
    Queue,
    QueueEdited,
    RecentExperiments,
    Recorded,
    Refusal,
    ReplayTimeline,
    ReplayView,
    Requeued,
    RunEstimateView,
    RunStarted,
    RunSummary,
    SavedExperiments,
    SavedImageMetadata,
    SavedImagePixels,
    SavedImagePosition,
    Stopped,
    Suggested,
    TaskConfigRead,
    TaskEdit,
    TaskHistory,
    TaskOutputs,
    Thickness,
    answer,
)
from fibsem.tools import bug_report
from fibsem.util.timestamps import to_datetime
from fibsem.workflows import evidence
from fibsem.workflows.events.text import note_asked, with_text
from fibsem.workflows.history_rows import (
    HistoryDecidedBy,
    HistoryOrder,
    HistoryShow,
    HistoryStatus,
)

if TYPE_CHECKING:
    from fibsem.server.app_context import AppContext

__all__ = [
    "build_app_config_router",
    "build_app_control_router",
    "build_app_router",
    "build_app_stop_router",
    "build_app_suggest_router",
]


def _served(router: APIRouter, context: "AppContext") -> APIRouter:
    """*router* without the routes of parts *context* doesn't implement (each
    part route is tagged with its part)."""
    served = set(served_parts(context))
    router.routes = [
        route for route in router.routes if not set(getattr(route, "tags", ())) - served
    ]
    return router


def _grid_run_body(
    body: GridPlanBody,
) -> Tuple[List[str], Optional[List[str]], bool]:
    """The grid plan/start verbs' tasks, grids and screen_all, or the 422 when
    no task is named."""
    if not body.task_names:
        raise _missing_field(
            "Pass task_names (non-empty list of grid task names); "
            "grid_names (list of strings) is optional — omitted means every "
            "recorded grid; screen_all (bool) inventories first and runs over "
            "every present grid, and takes no grid_names."
        )
    return body.task_names, body.grid_names, body.screen_all


# The image's acquisition metadata (JSON) rides on the JPEG answer, so one
# request gives the agent both the picture and what it was taken with.
IMAGE_METADATA_HEADER = "X-Image-Metadata"


def _jpeg(result: Any) -> Response:
    """An image context answer as a real image/jpeg response (so a browser
    ``<img>`` can consume it), or its refusal carrying the valid names."""
    jpeg = _answer_or_refusal(result)["jpeg"]
    headers = {}
    if result.get("metadata") is not None:
        # The image's scale and state (images.image_metadata), for the agent's
        # image tools; a browser <img> ignores it.
        headers[IMAGE_METADATA_HEADER] = json.dumps(result["metadata"])
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)


def _refusal_error(refusal: Refusal) -> HTTPException:
    """*refusal* as the HTTP error a route answers it with."""
    return refusal_error(
        refusal.status, refusal.error_type, refusal.message, **refusal.names
    )


def _missing_field(message: str) -> HTTPException:
    """The 422 for a request missing what a route needs, *message* saying what
    to pass."""
    return refusal_error(422, "missing_field", message)


def _note_body(body: NoteBody) -> Tuple[str, Optional[str]]:
    """A note's or a message's text and item name, or the 422 when the text
    is blank or too long."""
    if not body.text.strip() or len(body.text) > 4000:
        raise _missing_field("Pass text (a non-empty string, at most 4000 characters).")
    return body.text, body.item_name


def _required_text(body: Any, key: str, what: str) -> str:
    """*body*'s non-empty string *key* (a dict's key, or a typed body's
    field), or the 422 saying to pass it (*what*)."""
    value = body.get(key) if isinstance(body, dict) else getattr(body, key)
    if not isinstance(value, str) or not value:
        raise _missing_field(f"Pass {key}: {what}.")
    return value


def _given(body: Any, *keys: str) -> Dict[str, Any]:
    """A typed body's *keys* that were given (not None), as keywords."""
    return {key: getattr(body, key) for key in keys if getattr(body, key) is not None}


def _numbers(text: Optional[str]) -> Optional[List[float]]:
    """A comma-separated list of numbers from a query, or the 422 saying so."""
    if text is None:
        return None
    try:
        return [float(part) for part in text.split(",")]
    except ValueError:
        raise refusal_error(
            422, "invalid", f"A region is four numbers, x,y,width,height; not {text!r}."
        )


def _run_body(body: RunBody) -> Tuple[List[str], Optional[List[str]]]:
    """A lamella run's task_names and item_names, or the 422 when no task is
    named."""
    if not body.task_names:
        raise _missing_field(
            "Pass task_names (non-empty list of strings); "
            "item_names (list of strings) is optional — omitted "
            "means every item."
        )
    return body.task_names, body.item_names


def _answer_or_refusal(result: Any) -> Any:
    """A context answer, or its `Refusal` as an HTTP error."""
    if isinstance(result, Refusal):
        raise _refusal_error(result)
    return result


def _typed(result: Any) -> Response:
    """A context answer: a refusal raises (`_answer_or_refusal`), a structure
    answers as the codec writes it, and a dict as it is."""
    result = _answer_or_refusal(result)
    if isinstance(result, dict):
        return JSONResponse(jsonable_encoder(result))
    return answer(result)


def _as_sent(endpoint):
    """*endpoint* answering its dict as it is: its route's ``response_model``
    describes the answer without FastAPI filtering it or filling in nulls."""

    def sent(result):
        if isinstance(result, Refusal):
            raise _refusal_error(result)
        if isinstance(result, Response):
            return result
        return JSONResponse(jsonable_encoder(result))

    if inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        async def awaited(*args, **kwargs):
            return sent(await endpoint(*args, **kwargs))

        return awaited

    @functools.wraps(endpoint)
    def route(*args, **kwargs):
        return sent(endpoint(*args, **kwargs))

    return route


class _AsSentRoute(APIRoute):
    def __init__(self, path: str, endpoint, **kwargs):
        super().__init__(path, _as_sent(endpoint), **kwargs)


def _router() -> APIRouter:
    return APIRouter(prefix="/app", route_class=_AsSentRoute)


def _app_type(context: "AppContext", base: type) -> type:
    """The class *context*'s app answers for *base* (`AppContext.answer_type`)."""
    return context.answer_type(base)


# A route that answers an image, for /openapi.json.
_JPEG = {200: {"content": {"image/jpeg": {}}, "description": "A JPEG image."}}
# A file read raw (``raw=true``) answers the file as it is.
_RAW_FILE = {200: {"content": {"application/octet-stream": {}}}}
_PLANES = {
    200: {
        "content": {"application/octet-stream": {}},
        "description": "The planes' values; the X-Planes header says their shape.",
    }
}

# An export answers the file it makes, to download.
_EXPORT = {
    200: {
        "content": {exports.MEDIA_TYPES[ext]: {} for ext in exports.MEDIA_TYPES},
        "description": "The exported file.",
    }
}


def _download(content: bytes, name: str) -> Response:
    """*content* as the file *name*, for a browser to save."""
    return Response(
        content,
        media_type=exports.MEDIA_TYPES[os.path.splitext(name)[1]],
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


def build_app_router(context: "AppContext") -> APIRouter:
    router = _router()

    @router.get("/workflow/status", response_model=AppStatus)
    def app_status():
        return _typed(context.status())

    @router.get("/workflow/queue", response_model=Queue)
    def app_queue():
        return _typed(context.queue())

    @router.get(
        "/experiment",
        response_model=_app_type(context, ExperimentSummary),
    )
    def experiment_summary():
        return _typed(context.experiment_summary())

    @router.get("/experiment/task_history", response_model=TaskHistory)
    def task_history():
        return _typed(context.task_history())

    @router.get("/workflow/summary", response_model=RunSummary)
    def run_summary():
        return context.run_summary()

    @router.get("/experiment/protocol", response_model=_app_type(context, ProtocolView))
    def protocol():
        return context.protocol()

    @router.get("/log", response_model=LogRead)
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
        return _answer_or_refusal(result)

    @router.get(
        "/experiment/items/{item_name}/outputs",
        tags=[ITEMS],
        response_model=_app_type(context, TaskOutputs),
    )
    def task_outputs(item_name: str):
        return _typed(context.task_outputs(item_name))

    @router.get(
        "/experiment/items/{item_name}/stack", tags=[ITEMS], response_model=ItemStack
    )
    def item_stack(item_name: str, role: str):
        return _typed(context.item_stack(item_name, role))

    @router.get("/experiment/replay", tags=[REPLAY], response_model=ReplayTimeline)
    def replay():
        return _typed(context.replay())

    @router.get("/experiment/replay/{index}", tags=[REPLAY], response_model=ReplayView)
    def replay_scene(index: int, item: Optional[str] = None):
        return _typed(context.replay_scene(index, item))

    @router.get(
        "/experiment/items/{item_name}",
        tags=[ITEMS],
        response_model=_app_type(context, ItemDetail),
    )
    def item_detail(item_name: str):
        return _typed(context.item_detail(item_name))

    @router.get(
        "/experiment/protocol/task_config/{task_name}", response_model=TaskConfigRead
    )
    def protocol_task_config(task_name: str):
        return context.protocol_task_config(task_name)

    @router.get(
        "/experiment/items/{item_name}/task_config/{task_name}",
        response_model=TaskConfigRead,
        tags=[ITEMS],
    )
    def item_task_config(item_name: str, task_name: str):
        return context.item_task_config(item_name, task_name)

    @router.get(
        "/experiment/items/{item_name}/outputs/{filename}",
        response_class=Response,
        responses=_JPEG,
        tags=[ITEMS],
    )
    def output_image(item_name: str, filename: str, width: Optional[int] = None):
        # Serves actual image/jpeg (not a JSON payload) so a browser <img>
        # can consume it; filename must be a basename the item's own
        # outputs listing names — the context refuses anything else.
        return _jpeg(
            context.output_image(item_name, filename, files.preview_width(width))
        )

    @router.get(
        "/experiment/items/{item_name}/history",
        response_model=ItemHistory,
        tags=[ITEMS],
    )
    def item_history(
        item_name: str,
        show: Optional[HistoryShow] = None,
        status: Optional[HistoryStatus] = None,
        task: Optional[str] = None,
        decided_by: Optional[HistoryDecidedBy] = None,
        order: Optional[HistoryOrder] = None,
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

    @router.post(
        "/experiment/items/{item_name}/thickness",
        response_model=Thickness,
        tags=[LAMELLAE],
    )
    def measure_thickness(item_name: str, body: ThicknessBody):
        # Computed over a saved file, nothing moves; a POST because the result
        # is saved beside the image, as an autofocus run is.
        path = body.path
        if not path:
            raise _missing_field("path is required")
        window = _given(body, "width_um", "height_um")
        if body.centre is not None:
            window["centre"] = list(body.centre)
        return _answer_or_refusal(context.measure_thickness(item_name, path, **window))

    @router.get("/experiment/grids", response_model=Grids, tags=[GRIDS])
    def grids():
        return context.grids()

    @router.get(
        "/experiment/grids/{grid_name}", response_model=GridDetail, tags=[GRIDS]
    )
    def grid_detail(grid_name: str):
        return context.grid_detail(grid_name)

    @router.get(
        "/experiment/grids/{grid_name}/outputs/{filename}",
        response_class=Response,
        responses=_JPEG,
        tags=[GRIDS],
    )
    def grid_output_image(grid_name: str, filename: str, cells: bool = False):
        # cells: the overview's named cells drawn on it, listed in the
        # metadata header.
        return _jpeg(context.grid_output_image(grid_name, filename, cells=cells))

    # {filename:path}: any saved image's path in the experiment, slashes and all.
    @router.get(
        "/experiment/grids/{grid_name}/outputs/{filename:path}/stage_position",
        response_model=OverviewStagePosition,
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
        "/experiment/grids/{grid_name}/outputs/{filename:path}/markers",
        response_model=GridMarkers,
        tags=[GRIDS],
    )
    def grid_markers(grid_name: str, filename: str):
        return context.grid_markers(grid_name, filename)

    @router.get(
        "/experiment/grid_protocol/task_config/{task_name}",
        response_model=TaskConfigRead,
        tags=[GRIDS],
    )
    def grid_task_config(task_name: str):
        return _answer_or_refusal(context.grid_task_config(task_name))

    @router.get("/experiment/protocols", response_model=ProtocolFiles)
    def protocols():
        return context.protocols()

    @router.get("/experiment/recent", response_model=RecentExperiments)
    def recent_experiments():
        return {"items": context.recent_experiments()}

    @router.get("/experiment/all", response_model=SavedExperiments)
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
                    raise refusal_error(
                        422, "invalid", f"{name} is an ISO date or time, not {value!r}."
                    )
        return context.saved_experiments(
            instrument=instrument, operator=operator, **bounds
        )

    @router.get("/experiment/images", response_model=DisplayImages)
    def display_images():
        return _typed(context.display_images())

    @router.get("/experiment/images/metadata", response_model=SavedImageMetadata)
    def image_metadata(path: str):
        return _typed(context.image_metadata(path))

    @router.get("/experiment/images/stage_position", response_model=SavedImagePosition)
    def image_stage_position(
        path: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        crop: Optional[str] = None,
    ):
        # Any saved image in the experiment, beam or FM: a point on it as a
        # stage position, from the image's own metadata. crop: the point is
        # read off a crop's picture, a cell's name (B3) or "x,y,width,height".
        px = None if x is None or y is None else [x, y]
        fraction = None if fx is None or fy is None else [fx, fy]
        if crop is not None and "," in crop:
            crop = _numbers(crop)
        return _typed(context.image_stage_position(path, px, fraction, crop))

    @router.post(
        "/experiment/images/markers", response_model=MarkersDrawn, tags=[LAMELLAE]
    )
    def render_markers(body: MarkersBody):
        # A POST because the picture is saved beside the image, as a thickness
        # result is.
        if not body.path:
            raise refusal_error(422, "missing_field", "path is required")
        return _answer_or_refusal(
            context.render_markers(
                body.path, body.grid_name, **_given(body, "fov", "current")
            )
        )

    def _draw_options(body: DrawOptions, *planes: str) -> Dict[str, Any]:
        options = _given(
            body, "crosshair", "metadata_bar", "channels", "looks", *planes
        )
        if body.box is not None:
            if body.box[2] <= 0 or body.box[3] <= 0:
                raise _missing_field("box is [x, y, width, height] in pixels.")
            options["box"] = list(body.box)
        return options

    @router.post("/experiment/images/drawn", response_model=ImageDrawn)
    def draw_image(body: DrawBody):
        # A POST because the picture is saved beside the image, as markers are.
        path = _required_text(body, "path", "a saved image inside the experiment")
        return _typed(
            context.draw_image(path, body.marks, **_draw_options(body, "plane"))
        )

    @router.post("/experiment/images/pair", response_model=ImagePairDrawn)
    def draw_pair(body: ImagePairBody):
        before = _required_text(body, "before", "a saved image inside the experiment")
        after = _required_text(body, "after", "a saved image inside the experiment")
        options = _draw_options(body, "before_plane", "after_plane")
        options.pop("crosshair", None)
        options.update(_given(body, "share_looks"))
        return _typed(context.draw_pair(before, after, body.marks, **options))

    @router.post("/experiment/analysis/intensity", response_model=ImageMeasured)
    def measure_intensity(body: IntensityBody):
        # A POST because its figure is saved beside the image, as a drawing is.
        path = _required_text(body, "path", "a saved image inside the experiment")
        if not body.points:
            raise _missing_field("Pass points: a list of [x, y] in the image's pixels.")
        points = [list(p) for p in body.points]
        options = _given(body, "radius_px", "channel", "reference", "planes")
        return _typed(context.measure_intensity(path, points, **options))

    @router.post("/experiment/analysis/profile", response_model=ImageMeasured)
    def line_profile(body: ProfileBody):
        path = _required_text(body, "path", "a saved image inside the experiment")
        options = _given(body, "width_px", "channel", "plane", "dip")
        return _typed(
            context.line_profile(path, list(body.start), list(body.end), **options)
        )

    @router.get("/experiment/analysis/quality", response_model=ImageMeasured)
    def image_quality(
        path: str, channel: Optional[str] = None, plane: Optional[int] = None
    ):
        return _typed(context.image_quality(path, channel, plane))

    @router.post(
        "/experiment/items/{item_name}/milling/drawn",
        response_model=ItemMillingDrawn,
        tags=[LAMELLAE],
    )
    def draw_item_milling(item_name: str, body: MillingDrawnBody):
        # A POST because the picture is saved beside the image, as markers are.
        tasks = body.task_names
        task_names = [tasks] if isinstance(tasks, str) else tasks
        if not task_names:
            raise _missing_field(
                "Pass task_names: the milling tasks to draw, the first in colour."
            )
        if body.background not in ("related", "none"):
            raise refusal_error(422, "invalid", "background is related or none")
        options = _given(body, "scan_direction", "alignment_area", "metadata_bar")
        return _typed(
            context.draw_item_milling(
                item_name, task_names, body.image_path, body.background, **options
            )
        )

    @router.post("/experiment/images/pixels", response_model=SavedImagePixels)
    def image_pixels(body: PixelsBody):
        # A POST on the read router because it computes, never acts: stage
        # positions as points on a saved image, a list a GET could not carry.
        path = _required_text(body, "path", "a saved image inside the experiment")
        positions = body.stage_positions
        if not positions:
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

    @router.get(
        "/experiment/files/{path:path}", response_model=FileRead, responses=_RAW_FILE
    )
    def read_file(
        request: Request,
        path: str,
        box: Optional[str] = None,
        fraction: Optional[str] = None,
        width: Optional[int] = None,
        z: Optional[int] = None,
        channels: Optional[str] = None,
        cells: Optional[str] = None,
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
        # A fluorescence stack: plane z (0 the first; none, the max projection)
        # in channels, comma-separated names (none, all).
        # cells: "true" draws the image's named cells on the preview; names,
        # comma-separated ("B3,C4"), crop those cells.
        show_cells = (cells or "").strip().lower() == "true"
        crops = None if show_cells or not cells else [c for c in cells.split(",") if c]
        tag = _files_answer(
            experiment, files.read_etag, path, box, fraction, width, z, channels, cells
        )
        # kept by the browser, but asked about each time: an unchanged file answers 304
        cache = {"ETag": tag, "Cache-Control": "no-cache"} if tag else {}
        if tag is not None and request.headers.get("if-none-match") == tag:
            _record_read(experiment, path, False)
            return Response(status_code=304, headers=cache)
        answer = _files_answer(
            experiment,
            files.read,
            path,
            box=_numbers(box),
            fraction=_numbers(fraction),
            max_width=files.preview_width(width),
            z=z,
            channels=[c for c in (channels or "").split(",") if c] or None,
            cells=show_cells,
            crops=crops,
        )
        _record_read(experiment, path, False)
        return JSONResponse(jsonable_encoder(answer), headers=cache)

    @router.post(
        "/report",
        responses={200: {"content": {"application/zip": {}}}},
        dependencies=[Depends(require_operator())],
    )
    def bug_report_bundle(body: Optional[Dict[str, Any]] = None):
        # The operator's Report an Issue from the web page: the bundle is
        # theirs to save and email. Nothing is sent, and no image data goes in.
        body = body or {}
        content = bug_report.BugReportContent(
            title=str(body.get("title") or ""),
            description=str(body.get("description") or ""),
            include_screenshots=bool(body.get("screenshots")),
            include_agent_log=bool(body.get("agent_log")),
            system_context=bug_report.collect_system_context(),
        )
        experiment = context.experiment_path
        with tempfile.TemporaryDirectory() as folder:
            path = bug_report.build_bug_report_bundle(
                content, str(experiment) if experiment else None, folder
            )
            name = os.path.basename(path)
            with open(path, "rb") as f:
                data = f.read()
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @router.post(
        "/experiment/upload",
        response_model=FileEntry,
        dependencies=[Depends(require_operator())],
    )
    async def upload_file(request: Request, name: str, dir: str = files.UPLOAD_DIR):
        # The operator's file, added from the web page (the Agent page's
        # attach, Experiment Files): the body is the file as it is.
        upload = _files_answer(None, lambda root: files.Upload(root, name, dir))
        try:
            async for chunk in request.stream():
                upload.add(chunk)
        except files.FileRefusal as refusal:
            raise HTTPException(status_code=refusal.status, detail=refusal.detail)
        except BaseException:
            upload.cancel()
            raise
        entry = upload.finish()
        context.record_event(
            "experiment_file_added", {"path": entry["path"], "size": entry["size"]}
        )
        return JSONResponse(entry)

    @router.get("/experiment/planes/{path:path}", responses=_PLANES)
    def read_planes(
        request: Request,
        path: str,
        factor: int = 1,
        z: Optional[str] = None,
        first: Optional[int] = None,
        last: Optional[int] = None,
        channels: Optional[str] = None,
        box: Optional[str] = None,
        experiment: Optional[str] = None,
    ):
        # An image's raw planes for the web viewer, which draws them itself: the
        # values as stored, gzipped, C-order, little-endian; X-Planes says their
        # (C, Z, Y, X) shape, dtype and channels. factor 1, 4 or 8; z a plane
        # (0 the first), "all", or max/mean/min over first..last; box x,y,w,h
        # a region at full size.
        picked = [c for c in (channels or "").split(",") if c] or None
        plane = int(z) if z is not None and z.lstrip("-").isdigit() else z
        region = None
        if box is not None:
            try:
                region = [int(n) for n in box.split(",")]
            except ValueError:
                raise _missing_field("box is x,y,width,height in pixels.") from None
        query = ("planes", factor, z, first, last, channels, box)
        tag = _files_answer(experiment, files.read_etag, path, *query)
        cache = {"ETag": tag, "Cache-Control": "no-cache"} if tag else {}
        if tag is not None and request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers=cache)
        header, data = _files_answer(
            experiment, planes.read, path, factor, plane, first, last, picked, region
        )
        _record_read(experiment, path, False)
        return Response(
            gzip.compress(data.tobytes(), compresslevel=1),
            media_type="application/octet-stream",
            headers={
                **cache,
                "Content-Encoding": "gzip",
                "X-Planes": json.dumps(header),
            },
        )

    # -- exports: the Qt export dialogs' files, for a person to download ---------
    # Each renders, so a POST computes and changes nothing; FastAPI runs these
    # plain functions in its threadpool, off the event loop.

    @router.get("/experiment/export/image/options", response_model=ImageExportForm)
    def image_export_form(path: str, experiment: Optional[str] = None):
        return answer(_files_answer(experiment, exports.image_form, path))

    @router.post("/experiment/export/image", responses=_EXPORT)
    def export_image(body: ImageExportBody, experiment: Optional[str] = None):
        path = _required_text(body, "path", "an image inside the experiment")
        content, name = _files_answer(
            experiment, exports.image_file, path, body.options, body.format
        )
        return _download(content, name)

    def _animation_steps(item_name: str) -> Dict[str, Any]:
        return _answer_or_refusal(context.animation_steps(item_name))

    @router.get(
        "/experiment/export/animation/frames",
        response_model=AnimationFrames,
        tags=[LAMELLAE],
    )
    def animation_frames(item_name: str):
        found = _animation_steps(item_name)
        return answer(
            exports.animation_frames(found["root"], found["item_name"], found["steps"])
        )

    @router.post("/experiment/export/animation", responses=_EXPORT, tags=[LAMELLAE])
    def export_animation(body: AnimationExportBody):
        item_name = _required_text(body, "item_name", "the lamella to animate")
        found = _animation_steps(item_name)
        try:
            content, name = exports.animation_file(
                found["item_name"], found["steps"], body.options, body.format
            )
        except files.FileRefusal as refusal:
            raise HTTPException(status_code=refusal.status, detail=refusal.detail)
        return _download(content, name)

    @router.get("/workflow/proposals", response_model=Proposals)
    def proposals():
        return context.proposals()

    @router.post("/workflow/plan", response_model=RunEstimateView, tags=[LAMELLAE])
    def lamella_run_plan(body: RunPlanBody):
        # The Workflow tab's confirmation for a run, or for adding to the
        # running queue (add): computes, never acts.
        task_names, item_names = _run_body(body)
        return context.lamella_run_plan(task_names, item_names, body.run_next, body.add)

    @router.post("/workflow/grids/plan", response_model=GridPlan, tags=[GRIDS])
    def grid_workflow_plan(body: GridPlanBody):
        # A POST on the read router because it computes, never acts: the
        # preflight dialog's content for a grid run, from a body a GET could
        # not carry cleanly. No hardware is read.
        task_names, grid_names, screen_all = _grid_run_body(body)
        return context.grid_workflow_plan(task_names, grid_names, screen_all)

    @router.post("/experiment/notes", response_model=Recorded)
    def add_note(body: NoteBody):
        # Observations for the record — event stream + experiment log. On the
        # read router, a POST the dashboard token is refused: a watching agent
        # notes what it sees, and a note changes nothing.
        text, item_name = _note_body(body)
        cited = cited_evidence(context)
        result = context.add_note(
            text,
            item_name,
            **({"evidence": cited["evidence_id"]} if cited else {}),
        )
        return _answer_or_refusal(result)

    @router.post("/experiment/messages", response_model=Recorded)
    def add_message(body: OperatorMessageBody):
        # The agent's message to the operator, on the record as a note to them.
        # Like a note, it changes nothing.
        text, item_name = _note_body(body)
        reply_to = body.reply_to
        cited = cited_evidence(context)
        result = context.add_message(
            text,
            item_name,
            reply_to,
            **({"evidence": cited["evidence_id"]} if cited else {}),
        )
        return _answer_or_refusal(result)

    @router.post("/experiment/evidence", response_model=Recorded)
    def add_evidence(body: EvidenceBody):
        # What the agent judged from, kept in the experiment: an image of the
        # experiment's or one it made (uploaded, saved here), with marks on it.
        # Changes nothing, like a note.
        item_name = body.item_name

        def save() -> Dict[str, Any]:
            return evidence.save(
                lambda: context.experiment_path,
                image=body.image,
                upload=body.upload,
                marks=body.marks,
                source=body.source,
                item_name=item_name,
            )

        try:
            result = context.add_evidence(item_name, save)
        except evidence.EvidenceRefused as refusal:
            raise _evidence_refused(refusal)
        return _answer_or_refusal(result)

    @router.get("/events", response_model=Events)
    def events(
        since: int = 0,
        timeout: float = 0.0,
        after: Optional[str] = None,
        text: bool = False,
    ):
        # A long-poll: parks up to `timeout` seconds waiting for news. Capped
        # here because the park occupies a threadpool worker — a transport
        # concern, so the transport owns the ceiling. `after` is an event id,
        # in place of `since`, and reaches back past a restart. `text` adds
        # what each record says, in words, for a page to show.
        timeout = min(max(timeout, 0.0), 30.0)
        if after:
            answer = context.events(since=since, timeout=timeout, after=after)
        else:
            answer = context.events(since=since, timeout=timeout)
        return _with_text(answer) if text else answer

    @router.get("/events/history", response_model=EventHistory)
    def event_history(
        before: Optional[int] = None, limit: int = 500, text: bool = False
    ):
        # The run before the live buffer, from the experiment's events file:
        # what a page opened in the morning needs to show the night.
        answer = context.event_history(before=before, limit=limit)
        return _with_text(answer) if text else answer

    return _served(router, context)


def _with_text(answer: Dict[str, Any]) -> Dict[str, Any]:
    """*answer* with each record's ``text`` and ``group`` (`events.text`)."""
    events = answer.get("events")
    if not isinstance(events, list):
        return answer
    asked: Dict[str, Any] = {}
    described = []
    for record in events:
        note_asked(asked, record)
        described.append(with_text(record, asked))
    return {**answer, "events": described}


def _evidence_refused(refusal: evidence.EvidenceRefused) -> HTTPException:
    return HTTPException(status_code=refusal.status_code, detail=refusal.detail)


def cited_evidence(context: "AppContext") -> Optional[Dict[str, Any]]:
    """The record of the evidence the agent's call cites
    (`calls.EVIDENCE_HEADER`); None when it cites none."""
    call = current_call()
    if call is None or not call.evidence:
        return None
    try:
        return evidence.load(lambda: context.experiment_path, call.evidence)
    except evidence.EvidenceRefused as refusal:
        raise _evidence_refused(refusal)


def build_app_control_router(context: "AppContext") -> APIRouter:
    """Routes that act on the session rather than observe it.

    Mounted (by build_server) at Drive: the operator raises the level from the
    Agent chip.
    """
    router = _router()

    @router.post("/workflow/decide", response_model=Decided)
    def decide(body: DecideBody):
        # The same function the operator's decision goes through, on the main thread, blocking
        # until applied. Item by id, never by name: names change.
        # What the decision is on: the proposal's own id, or -- from a caller
        # that has not learned it -- the run it came from. The id is the one to
        # pass: a task may ask several proposals in one run.
        proposal_id = body.proposal_id or ""
        task_id = body.task_id or ""
        if not all((body.item_id, body.task_name, body.outcome)) or not (
            proposal_id or task_id
        ):
            raise _missing_field(
                "Pass item_id, task_name, proposal_id and outcome "
                "(Confirmed or Rejected) from GET /app/workflow/proposals."
            )
        if body.outcome == "Rejected" and not body.reason.strip():
            raise _missing_field("A reject needs a reason; it is recorded on the item.")
        cited = cited_evidence(context)
        result = context.decide(
            body.item_id,
            body.task_name,
            body.outcome,
            values=body.values,
            reason=body.reason,
            actor=body.actor,
            task_id=task_id,
            proposal_id=proposal_id,
            action=body.action or "",
            patch=body.patch,
            **({"evidence": cited["evidence_id"]} if cited else {}),
        )
        return _answer_or_refusal(result)

    @router.post("/workflow/start", response_model=RunStarted)
    def start_workflow(body: RunBody):
        return context.start_workflow(*_run_body(body))

    @router.post("/workflow/grids/start", response_model=RunStarted, tags=[GRIDS])
    def start_grid_workflow(body: GridRunBody):
        # The Grids view's Run / Screen all grids, remotely. Shares the lamella
        # run's worker slot, so the context refuses it while any run is live.
        task_names, grid_names, screen_all = _grid_run_body(body)
        return context.start_grid_workflow(
            task_names,
            grid_names,
            screen_all,
            unload_at_end=body.unload_at_end,
            report_at_end=body.report_at_end,
        )

    @router.post("/experiment/protocol/supervision", response_model=TaskEdit)
    def set_supervision(body: SupervisionBody):
        return context.set_supervision(body.task_name, body.supervise)

    @router.post("/workflow/queue/requeue", response_model=Requeued)
    def requeue_task(body: RequeueBody):
        result = context.requeue_task(body.item_name, body.task_name, front=body.front)
        return _answer_or_refusal(result)

    @router.post("/workflow/queue/edit", response_model=QueueEdited)
    def edit_queue(body: QueueEditBody):
        item_name = _required_text(body, "item_name", "the item whose task to edit")
        task_name = _required_text(body, "task_name", "the waiting task to edit")
        action = _required_text(
            body, "action", "remove, move_up, move_down or run_next"
        )
        return _answer_or_refusal(context.edit_queue(item_name, task_name, action))

    @router.post("/workflow/queue/grids", response_model=GridTasksQueued, tags=[GRIDS])
    def queue_grid_tasks(body: QueueGridsBody):
        # The Grids view's Add to queue on a running grid run: at the end.
        return context.queue_grid_tasks(body.task_names, body.grid_names)

    return _served(router, context)


def build_app_suggest_router(context: "AppContext") -> APIRouter:
    """Suggesting on a pending proposal. Mounted (by build_server) for Assist
    and Drive: a suggestion decides nothing, so it needs no request."""
    router = _router()

    @router.post("/workflow/proposals/suggest", response_model=Suggested)
    def suggest(body: SuggestBody):
        if not all((body.item_id, body.task_name, body.proposal_id)):
            raise _missing_field(
                "Pass item_id, task_name and proposal_id from "
                "GET /app/workflow/proposals."
            )
        if not body.reason.strip():
            raise _missing_field(
                "Pass a reason: the operator reads it when deciding whether to "
                "approve the suggestion."
            )
        cited = cited_evidence(context)
        result = context.suggest(
            body.item_id,
            body.task_name,
            body.proposal_id,
            reason=body.reason.strip(),
            outcome=body.outcome or "Confirmed",
            values=body.values,
            patch=body.patch,
            action=body.action or "",
            evidence=cited,
            actor=body.actor,
        )
        return _answer_or_refusal(result)

    return _served(router, context)


def build_app_stop_router(context: "AppContext") -> APIRouter:
    """Stopping the run. Mounted (by build_server) at Drive, with a reason: the
    workflow is never the agent's own, so at Assist the stop goes in a request
    the operator approves; a watching agent stops nothing."""
    router = _router()

    @router.post("/workflow/stop", response_model=Stopped)
    def stop_workflow(body: Optional[StopBody] = None):
        # The same path as the GUI's Stop button, and never waits on the
        # command lock. The operator's own Stop is never gated.
        return context.stop_workflow(body.reason if body else "")

    @router.post("/workflow/stop_task", response_model=Stopped)
    def stop_task(body: StopTaskBody):
        # The timeline's Stop Task: the task running, and the run carries on.
        # It names the task meant, so a stop landing late stops nothing else.
        item_name = _required_text(body, "item_name", "the item of the task running")
        task_name = _required_text(body, "task_name", "the task running")
        return context.stop_task(item_name, task_name, body.reason)

    return _served(router, context)


def build_app_config_router(context: "AppContext") -> APIRouter:
    """Routes that edit configuration (protocols, item configs, schedules,
    run options). Mounted at Drive, like the control router."""
    router = _router()
    item_edit = ItemEdit[_app_type(context, ItemDetail)]

    def _validated(body: PatchBody):
        patch, version = body.patch, body.version
        if not patch:
            raise _missing_field(
                "Pass patch (a non-empty object of dotted-path: "
                "value entries) and version (from the config read it was "
                "written against)."
            )
        return patch, version

    @router.post(
        "/experiment/protocol/task_config/{task_name}", response_model=ConfigEdit
    )
    def patch_protocol_task_config(task_name: str, body: PatchBody):
        patch, version = _validated(body)
        return _answer_or_refusal(
            context.apply_protocol_task_config_patch(task_name, patch, version)
        )

    @router.post(
        "/experiment/items/{item_name}/task_config/{task_name}",
        response_model=ConfigEdit,
        tags=[ITEMS],
    )
    def patch_item_task_config(item_name: str, task_name: str, body: PatchBody):
        patch, version = _validated(body)
        return _answer_or_refusal(
            context.apply_item_task_config_patch(item_name, task_name, patch, version)
        )

    @router.post("/workflow/schedule", response_model=TaskEdit)
    def set_task_schedule(body: ScheduleBody):
        # Workflow structure, not a config document: a verb like
        # /app/experiment/protocol/supervision. scheduled_at is ISO-8601 or null
        # to clear.
        return _answer_or_refusal(
            context.set_task_schedule(body.task_name, body.scheduled_at)
        )

    @router.post("/workflow/options", response_model=OptionsSet)
    def set_workflow_options(body: WorkflowOptionsBody):
        # The workflow's run options, beside the schedule: whether the run
        # summary dialog opens, and whether the beams go off at the end.
        options = {k: v for k, v in asdict(body).items() if v is not None}
        if not options:
            raise _missing_field(
                "Pass any of show_run_summary (boolean), turn_beams_off (boolean)."
            )
        return context.set_workflow_options(**options)

    # -- the workflow's tasks: the protocol editor and the Workflow tab ---------

    @router.post("/experiment/protocol/tasks", response_model=TaskEdit)
    def add_task(body: NewTaskBody):
        task_type = _required_text(body, "task_type", "a registered task type")
        task_name = _required_text(body, "task_name", "the new task's name")
        return _answer_or_refusal(context.add_task(task_type, task_name))

    @router.delete("/experiment/protocol/tasks/{task_name}", response_model=TaskEdit)
    def remove_task(task_name: str):
        return _answer_or_refusal(context.remove_task(task_name))

    @router.post("/experiment/protocol/tasks/reorder", response_model=TaskEdit)
    def reorder_tasks(body: TaskOrderBody):
        return _answer_or_refusal(context.reorder_tasks(body.order))

    @router.post(
        "/experiment/protocol/tasks/{task_name}/requirements", response_model=TaskEdit
    )
    def set_task_requirements(task_name: str, body: RequirementsBody):
        if body.requires is None and body.required is None:
            raise _missing_field(
                "Pass requires (the earlier tasks it needs, a list) and/or "
                "required (boolean)."
            )
        return _answer_or_refusal(
            context.set_task_requirements(task_name, body.requires, body.required)
        )

    @router.post(
        "/experiment/protocol/lamella_defaults",
        response_model=LamellaDefaultsSet,
        tags=[LAMELLAE],
    )
    def set_lamella_defaults(body: Dict[str, Any]):
        # Left a dict: a key given as null clears it, so which keys came
        # matters, and the app checks them.
        if not body:
            raise _missing_field(
                "Pass any of use_petname, name_prefix, poi, alignment_area."
            )
        return _answer_or_refusal(context.set_lamella_defaults(body))

    @router.post("/experiment/protocol/save", response_model=ProtocolSaved)
    def save_protocol_as(body: SaveProtocolBody):
        name = _required_text(body, "name", "a file name for the protocol folder")
        return _answer_or_refusal(context.save_protocol_as(name, body.overwrite))

    # -- the grid protocol: grid task settings, order and requirements ---------

    @router.post(
        "/experiment/grid_protocol/task_config/{task_name}",
        response_model=ConfigEdit,
        tags=[GRIDS],
    )
    def patch_grid_task_config(task_name: str, body: PatchBody):
        patch, version = _validated(body)
        result = context.apply_grid_task_config_patch(task_name, patch, version)
        return _answer_or_refusal(result)

    @router.post(
        "/experiment/grid_protocol/tasks/reorder", response_model=TaskEdit, tags=[GRIDS]
    )
    def reorder_grid_tasks(body: TaskOrderBody):
        return _answer_or_refusal(context.reorder_grid_tasks(body.order))

    @router.post(
        "/experiment/grid_protocol/tasks/{task_name}/requirements",
        response_model=TaskEdit,
        tags=[GRIDS],
    )
    def set_grid_task_requirements(task_name: str, body: GridRequirementsBody):
        return _answer_or_refusal(
            context.set_grid_task_requirements(task_name, body.requires)
        )

    def _validated_reorder(body: StageOrderBody):
        milling_key, order, version = body.milling_key, body.order, body.version
        if not order:
            raise _missing_field(
                "Pass milling_key (string), order (the current "
                "stage names in their new sequence), and version (from "
                "the config read)."
            )
        return milling_key, order, version

    @router.post(
        "/experiment/items/{item_name}/task_config/{task_name}/stages/reorder",
        response_model=ConfigEdit,
        tags=[MILLING_STAGES],
    )
    def reorder_item_stages(item_name: str, task_name: str, body: StageOrderBody):
        milling_key, order, version = _validated_reorder(body)
        return _answer_or_refusal(
            context.reorder_milling_stages(
                "item", item_name, task_name, milling_key, order, version
            )
        )

    @router.post(
        "/experiment/protocol/task_config/{task_name}/stages/reorder",
        response_model=ConfigEdit,
        tags=[MILLING_STAGES],
    )
    def reorder_protocol_stages(task_name: str, body: StageOrderBody):
        milling_key, order, version = _validated_reorder(body)
        return _answer_or_refusal(
            context.reorder_milling_stages(
                "protocol", "", task_name, milling_key, order, version
            )
        )

    @router.post(
        "/experiment/items/{item_name}/apply_protocol",
        response_model=ConfigEdit,
        tags=[ITEMS],
    )
    def apply_protocol_to_item(item_name: str, body: ApplyProtocolBody):
        # Wholesale by design — the verb IS "replace with the protocol
        # defaults" — so no version field; a refusal is the context's.
        return _answer_or_refusal(
            context.apply_protocol_to_item(item_name, body.task_names)
        )

    # -- experiments and protocols: kept to known places -----------------------

    @router.post(
        "/experiment/create",
        response_model=ExperimentOpened[_app_type(context, ExperimentSummary)],
    )
    def create_experiment(body: NewExperimentBody):
        name = _required_text(
            body, "name", "the new experiment's folder, under the experiments folder"
        )
        return _answer_or_refusal(
            context.create_experiment(
                name,
                protocol=body.protocol,
                description=body.description,
                sample=body.sample,
            )
        )

    @router.post(
        "/experiment/load",
        response_model=ExperimentOpened[_app_type(context, ExperimentSummary)],
    )
    def load_experiment(body: ExperimentPathBody):
        path = _required_text(
            body, "path", "a path from the recent experiments or the experiments folder"
        )
        return _answer_or_refusal(context.load_experiment(path))

    @router.post(
        "/experiment/protocol/load",
        response_model=ProtocolLoaded[_app_type(context, ProtocolView)],
    )
    def load_protocol(body: ProtocolNameBody):
        name = _required_text(body, "protocol", "a protocol name from the list")
        return _answer_or_refusal(context.load_protocol(name))

    # -- grid records: the Grids tab's inventory read and Rename ----------------

    @router.post("/experiment/grids", response_model=GridEdit, tags=[GRIDS])
    def add_grids():
        # A record for every grid the inventory holds and the experiment lacks.
        return _answer_or_refusal(context.add_grids())

    @router.post(
        "/experiment/grids/{grid_name}/rename", response_model=GridEdit, tags=[GRIDS]
    )
    def rename_grid(grid_name: str, body: NameBody):
        name = _required_text(body, "name", "the grid's new name")
        return _answer_or_refusal(context.rename_grid(grid_name, name))

    @router.post(
        "/experiment/grids/{grid_name}/quality", response_model=GridEdit, tags=[GRIDS]
    )
    def set_grid_quality(grid_name: str, body: GridQualityBody):
        verdict = _required_text(
            body, "verdict", "GOOD, REWORK, FAILED, or UNASSESSED to clear"
        )
        return _answer_or_refusal(
            context.set_grid_quality(grid_name, verdict, body.reason)
        )

    @router.post(
        "/experiment/grids/{grid_name}/note", response_model=GridEdit, tags=[GRIDS]
    )
    def set_grid_note(grid_name: str, body: GridNoteBody):
        note = body.note
        if len(note) > 4000:
            raise _missing_field(
                "Pass note: text, at most 4000 characters ('' clears)."
            )
        cited = cited_evidence(context)
        result = _answer_or_refusal(
            context.set_grid_note(
                grid_name, note, **({"evidence": cited["evidence_id"]} if cited else {})
            )
        )
        call = current_call()
        if call is not None and note and result.get("applied"):
            call.recorded = True  # its note is the record of it
        return _answer_or_refusal(result)

    @router.delete(
        "/experiment/grids/{grid_name}", response_model=GridEdit, tags=[GRIDS]
    )
    def remove_grid(grid_name: str):
        return _answer_or_refusal(context.remove_grid(grid_name))

    # -- lamella positions: the Lamella tab's buttons ---------------------------

    @router.post("/experiment/items", response_model=item_edit, tags=[ITEMS])
    def add_item(body: NewItemBody):
        # Where: the stage now (neither field), a stage position, or a point on a
        # grid overview ({grid_name, filename, px | fraction}). The app names it
        # unless a name is given.
        if body.stage_position is not None and body.overview is not None:
            raise refusal_error(
                422, "invalid", "Give stage_position or overview, not both."
            )
        return _answer_or_refusal(
            context.add_item(
                name=body.name,
                stage_position=body.stage_position,
                overview=body.overview,
            )
        )

    @router.delete(
        "/experiment/items/{item_name}", response_model=item_edit, tags=[ITEMS]
    )
    def remove_item(item_name: str):
        return _answer_or_refusal(context.remove_item(item_name))

    @router.post(
        "/experiment/items/{item_name}/move_to",
        response_model=item_edit,
        tags=[POSES],
        dependencies=[Depends(take_control, scope="function")],
    )
    def move_to_item_pose(item_name: str, body: PoseBody):
        pose = _required_text(body, "pose", "a pose name from the item (MILLING, ...)")
        return _answer_or_refusal(context.move_to_item_pose(item_name, pose))

    @router.post(
        "/experiment/items/{item_name}/objective/move_to",
        response_model=item_edit,
        tags=[POSES],
        dependencies=[Depends(take_control, scope="function")],
    )
    def move_objective_to_item(item_name: str):
        # The tab's objective Move to: the FM objective to the stored position.
        return _answer_or_refusal(context.move_objective_to_item(item_name))

    @router.post(
        "/experiment/items/{item_name}/poses/{pose_name}",
        response_model=item_edit,
        tags=[POSES],
    )
    def set_item_pose(item_name: str, pose_name: str):
        # From the stage position now: the tab's Set pose.
        return _answer_or_refusal(context.set_item_pose(item_name, pose_name))

    @router.post(
        "/experiment/items/{item_name}/objective_position",
        response_model=item_edit,
        tags=[POSES],
    )
    def set_item_objective_position(item_name: str):
        # From the FM objective now: the tab's Use current.
        return _answer_or_refusal(context.set_item_objective_position(item_name))

    @router.post(
        "/experiment/items/{item_name}/objective_position/apply_to_all",
        response_model=item_edit,
        tags=[POSES],
    )
    def set_objective_position_on_all(item_name: str):
        # The item's stored position on every lamella: Apply to All Lamella.
        return _answer_or_refusal(context.set_objective_position_on_all(item_name))

    @router.post(
        "/experiment/items/{item_name}/position", response_model=item_edit, tags=[POSES]
    )
    def move_item(item_name: str, body: PlaceBody):
        # The lamella moved to a point, as on an overview; the stage stays.
        return _answer_or_refusal(
            context.move_item(
                item_name,
                stage_position=body.stage_position,
                overview=body.overview,
            )
        )

    @router.post(
        "/experiment/items/{item_name}", response_model=item_edit, tags=[ITEMS]
    )
    def patch_item(item_name: str, body: PatchBody):
        # The item's own document (geometry, verdict, notes) — the write-side
        # mirror of GET /app/experiment/items/{item_name}, whose payload carries the
        # version this patch must echo.
        patch, version = _validated(body)
        return _answer_or_refusal(context.apply_item_patch(item_name, patch, version))

    @router.post(
        "/experiment/items/{item_name}/correlation",
        response_model=ConfigEdit,
        tags=[LAMELLAE],
    )
    def apply_item_correlation(item_name: str, body: CorrelationRunBody):
        # The correlation dialog's Continue: the item's POI to a saved run's
        # target, recorded as a correlation.
        run_path = body.run_path
        if not run_path:
            raise _missing_field(
                "Pass run_path, a run folder from correlation_correlate."
            )
        return _answer_or_refusal(context.apply_item_correlation(item_name, run_path))

    return _served(router, context)
