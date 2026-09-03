"""The app-level router: observation of a running AutoLamella session.

Dependency direction is one-way — autolamella imports fibsem.server, never the
reverse — so this module never imports the application. It defines the
:class:`AppContext` protocol structurally; the application's ``AgentContext``
(fibsem.applications.autolamella.server) satisfies it and is passed into
``build_server(app_context=...)`` by whoever hosts the server inside the app.

Every route is a thin pass-through: the context returns plain JSON-able
snapshots and owns all app knowledge; nothing here touches domain objects.
The read-scope requirement is applied where the router is included, so auth
stays in one place (build_server).
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Response

try:  # Protocol is typing-only; keep import robust on the 3.8 floor
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

__all__ = [
    "AppContext",
    "build_app_config_router",
    "build_app_control_router",
    "build_app_router",
]


class AppContext(Protocol):
    """What a hosted application must answer. Matched structurally."""

    def status(self) -> Dict[str, Any]: ...

    def queue(self) -> Dict[str, Any]: ...

    def experiment_summary(self) -> Dict[str, Any]: ...

    def task_history(self) -> Dict[str, Any]: ...

    def run_summary(self) -> Dict[str, Any]: ...

    def protocol(self) -> Dict[str, Any]: ...

    def task_outputs(self, item_name: str) -> Dict[str, Any]: ...

    def item_detail(self, item_name: str) -> Dict[str, Any]: ...

    def output_image(self, item_name: str, filename: str) -> Dict[str, Any]: ...

    def grids(self) -> Dict[str, Any]: ...

    def grid_detail(self, grid_name: str) -> Dict[str, Any]: ...

    def grid_output_image(self, grid_name: str, filename: str) -> Dict[str, Any]: ...

    def grid_markers(self, grid_name: str, filename: str) -> Dict[str, Any]: ...

    def protocol_task_config(self, task_name: str) -> Dict[str, Any]: ...

    def item_task_config(self, item_name: str, task_name: str) -> Dict[str, Any]: ...

    def apply_item_task_config_patch(
        self, item_name: str, task_name: str, patch: Dict[str, Any], version: str
    ) -> Dict[str, Any]: ...

    def apply_protocol_task_config_patch(
        self, task_name: str, patch: Dict[str, Any], version: str
    ) -> Dict[str, Any]: ...

    def apply_item_patch(
        self, item_name: str, patch: Dict[str, Any], version: str
    ) -> Dict[str, Any]: ...

    def set_task_schedule(
        self, task_name: str, scheduled_at: Optional[str]
    ) -> Dict[str, Any]: ...

    def apply_protocol_to_item(
        self, item_name: str, task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]: ...

    def reorder_milling_stages(
        self,
        level: str,
        item_name: str,
        task_name: str,
        milling_key: str,
        order: List[str],
        version: str,
    ) -> Dict[str, Any]: ...

    def recent_experiments(self) -> List[Dict[str, Any]]: ...

    def events(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]: ...

    def display_images(self) -> Dict[str, Any]: ...

    def pending_prompt(self) -> Dict[str, Any]: ...

    def answer_prompt(
        self, response: bool, nonce: int, value: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...

    def add_note(
        self, text: str, item_name: Optional[str] = None
    ) -> Dict[str, Any]: ...

    def stop_workflow(self) -> Dict[str, Any]: ...

    def start_workflow(
        self, task_names: List[str], item_names: Optional[List[str]] = None
    ) -> Dict[str, Any]: ...

    def set_supervision(
        self,
        task_name: str,
        supervise: bool,
        supervisor: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def requeue_task(
        self, item_name: str, task_name: str, front: bool = False
    ) -> Dict[str, Any]: ...


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
    return Response(content=jpeg, media_type="image/jpeg")


def build_app_router(context: AppContext) -> APIRouter:
    router = APIRouter(prefix="/app")

    @router.get("/status")
    def app_status():
        return context.status()

    @router.get("/queue")
    def app_queue():
        return context.queue()

    @router.get("/experiment_summary")
    def experiment_summary():
        return context.experiment_summary()

    @router.get("/task_history")
    def task_history():
        return context.task_history()

    @router.get("/run_summary")
    def run_summary():
        return context.run_summary()

    @router.get("/protocol")
    def protocol():
        return context.protocol()

    @router.get("/task_outputs/{item_name}")
    def task_outputs(item_name: str):
        return context.task_outputs(item_name)

    @router.get("/items/{item_name}")
    def item_detail(item_name: str):
        return context.item_detail(item_name)

    @router.get("/protocol/task_config/{task_name}")
    def protocol_task_config(task_name: str):
        return context.protocol_task_config(task_name)

    @router.get("/items/{item_name}/task_config/{task_name}")
    def item_task_config(item_name: str, task_name: str):
        return context.item_task_config(item_name, task_name)

    @router.get("/items/{item_name}/outputs/{filename}")
    def output_image(item_name: str, filename: str):
        # Serves actual image/jpeg (not a JSON payload) so a browser <img>
        # can consume it; filename must be a basename the item's own
        # task_outputs listing names — the context refuses anything else.
        return _jpeg_or_404(context.output_image(item_name, filename))

    @router.get("/grids")
    def grids():
        return context.grids()

    @router.get("/grids/{grid_name}")
    def grid_detail(grid_name: str):
        return context.grid_detail(grid_name)

    @router.get("/grids/{grid_name}/outputs/{filename}")
    def grid_output_image(grid_name: str, filename: str):
        return _jpeg_or_404(context.grid_output_image(grid_name, filename))

    @router.get("/grids/{grid_name}/outputs/{filename}/markers")
    def grid_markers(grid_name: str, filename: str):
        return context.grid_markers(grid_name, filename)

    @router.get("/recent_experiments")
    def recent_experiments():
        return {"items": context.recent_experiments()}

    @router.get("/images")
    def display_images():
        return context.display_images()

    @router.get("/prompt")
    def pending_prompt():
        return context.pending_prompt()

    @router.post("/workflow/stop")
    def stop_workflow():
        # Deliberately on the read router, like the core server's stop_milling:
        # stopping is the safety action, and must never be gated behind the
        # arming the dangerous actions need, nor wait on the command lock.
        return context.stop_workflow()

    @router.get("/events")
    def events(since: int = 0, timeout: float = 0.0):
        # A long-poll: parks up to `timeout` seconds waiting for news. Capped
        # here because the park occupies a threadpool worker — a transport
        # concern, so the transport owns the ceiling.
        return context.events(since=since, timeout=min(max(timeout, 0.0), 30.0))

    return router


def build_app_control_router(context: AppContext) -> APIRouter:
    """Routes that act on the session rather than observe it.

    Mounted (by build_server) behind the ``control`` scope, which embedded
    hosting does not arm yet — so today these exist only where a hosting
    explicitly arms control (tests; later, the arming dialog).
    """
    router = APIRouter(prefix="/app")

    @router.post("/prompt/answer")
    def answer_prompt(body: Dict[str, Any]):
        # The nonce is mandatory: an answer must name the question it answers
        # (from GET /app/prompt), never "whatever is pending by the time this
        # lands" — the prompt can change between the read and the answer.
        try:
            nonce = int(body["nonce"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_nonce",
                    "message": "Echo the integer nonce from GET /app/prompt; "
                    "an answer must name the question it answers.",
                },
            )
        value = body.get("value")
        if value is not None and not isinstance(value, dict):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_value",
                    "message": "value, when given, must be a JSON object: "
                    "{left, top, width, height} for EditAlignmentArea, "
                    "{x, y} for PickPOI.",
                },
            )
        result = context.answer_prompt(
            bool(body.get("response", False)), nonce, value=value
        )
        if result.get("stale"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error_type": "stale_prompt",
                    "message": "That question is no longer pending; "
                    "re-read /app/prompt and answer the current one.",
                },
            )
        if result.get("invalid_value"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "invalid_value",
                    "message": result["invalid_value"],
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
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "Pass task_names (non-empty list of strings); "
                    "item_names (list of strings) is optional — omitted "
                    "means every item.",
                },
            )
        return context.start_workflow(task_names, item_names)

    @router.post("/agent/notes")
    def add_note(body: Dict[str, Any]):
        # Observations for the record — event stream + experiment log. On the
        # control scope: an agent trusted to answer is trusted to annotate.
        text = body.get("text")
        item_name = body.get("item_name")
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > 4000
            or (item_name is not None and not isinstance(item_name, str))
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "Pass text (a non-empty string, at most 4000 "
                    "characters); item_name (string) is optional.",
                },
            )
        result = context.add_note(text, item_name)
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

    @router.post("/supervision")
    def set_supervision(body: Dict[str, Any]):
        task_name = body.get("task_name")
        supervisor = body.get("supervisor")
        if (
            not isinstance(task_name, str)
            or "supervise" not in body
            or supervisor not in (None, "human", "agent")
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "Pass task_name (string) and supervise (boolean); "
                    "supervisor is optional, 'human' or 'agent'.",
                },
            )
        return context.set_supervision(task_name, bool(body["supervise"]), supervisor)

    @router.post("/queue/requeue")
    def requeue_task(body: Dict[str, Any]):
        item_name = body.get("item_name")
        task_name = body.get("task_name")
        if not isinstance(item_name, str) or not isinstance(task_name, str):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "Pass item_name and task_name (strings); "
                    "front (boolean) is optional.",
                },
            )
        return context.requeue_task(
            item_name, task_name, front=bool(body.get("front", False))
        )

    return router


def build_app_config_router(context: AppContext) -> APIRouter:
    """Routes that edit configuration — the ``configure`` scope's own router.

    Deliberately not on the control router: answering a question with
    geometry (control) and editing protocols (configure) are different
    grants, and the arming dialog shows them as separate rungs (FIB-864).
    """
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
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "Pass patch (a non-empty object of dotted-path: "
                    "value entries) and version (from the config read it was "
                    "written against).",
                },
            )
        return patch, version

    @router.post("/protocol/task_config/{task_name}")
    def patch_protocol_task_config(task_name: str, body: Dict[str, Any]):
        patch, version = _validated(body)
        return _refused_or(
            context.apply_protocol_task_config_patch(task_name, patch, version)
        )

    @router.post("/items/{item_name}/task_config/{task_name}")
    def patch_item_task_config(item_name: str, task_name: str, body: Dict[str, Any]):
        patch, version = _validated(body)
        return _refused_or(
            context.apply_item_task_config_patch(item_name, task_name, patch, version)
        )

    @router.post("/workflow/schedule")
    def set_task_schedule(body: Dict[str, Any]):
        # Workflow structure, not a config document: a verb like /app/supervision,
        # on the configure scope. scheduled_at is ISO-8601 or null to clear.
        task_name = body.get("task_name")
        scheduled_at = body.get("scheduled_at")
        if not isinstance(task_name, str) or (
            scheduled_at is not None and not isinstance(scheduled_at, str)
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "Pass task_name (string) and scheduled_at "
                    "(ISO-8601 string, or null to clear).",
                },
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
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "Pass milling_key (string), order (the current "
                    "stage names in their new sequence), and version (from "
                    "the config read).",
                },
            )
        return milling_key, order, version

    @router.post("/items/{item_name}/task_config/{task_name}/stages/reorder")
    def reorder_item_stages(item_name: str, task_name: str, body: Dict[str, Any]):
        milling_key, order, version = _validated_reorder(body)
        return _refused_or(
            context.reorder_milling_stages(
                "item", item_name, task_name, milling_key, order, version
            )
        )

    @router.post("/protocol/task_config/{task_name}/stages/reorder")
    def reorder_protocol_stages(task_name: str, body: Dict[str, Any]):
        milling_key, order, version = _validated_reorder(body)
        return _refused_or(
            context.reorder_milling_stages(
                "protocol", "", task_name, milling_key, order, version
            )
        )

    @router.post("/items/{item_name}/apply_protocol")
    def apply_protocol_to_item(item_name: str, body: Dict[str, Any]):
        # Wholesale by design — the verb IS "replace with the protocol
        # defaults" — so no version field; refusals ride _refused_or.
        task_names = body.get("task_names")
        if task_names is not None and not (
            isinstance(task_names, list) and all(isinstance(t, str) for t in task_names)
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error_type": "missing_field",
                    "message": "task_names, when given, is a list of task "
                    "name strings; omitted means every task the protocol "
                    "defines.",
                },
            )
        return _refused_or(context.apply_protocol_to_item(item_name, task_names))

    @router.post("/items/{item_name}")
    def patch_item(item_name: str, body: Dict[str, Any]):
        # The item's own document (geometry, verdict, notes) — the write-side
        # mirror of GET /app/items/{item_name}, whose payload carries the
        # version this patch must echo.
        patch, version = _validated(body)
        return _refused_or(context.apply_item_patch(item_name, patch, version))

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

    return router
