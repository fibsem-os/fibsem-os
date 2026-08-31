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

from fastapi import APIRouter, HTTPException

try:  # Protocol is typing-only; keep import robust on the 3.8 floor
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

__all__ = ["AppContext", "build_app_control_router", "build_app_router"]


class AppContext(Protocol):
    """What a hosted application must answer. Matched structurally."""

    def status(self) -> Dict[str, Any]: ...

    def queue(self) -> Dict[str, Any]: ...

    def experiment_summary(self) -> Dict[str, Any]: ...

    def task_history(self) -> Dict[str, Any]: ...

    def run_summary(self) -> Dict[str, Any]: ...

    def protocol(self) -> Dict[str, Any]: ...

    def task_outputs(self, item_name: str) -> Dict[str, Any]: ...

    def recent_experiments(self) -> List[Dict[str, Any]]: ...

    def events(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]: ...

    def display_images(self) -> Dict[str, Any]: ...

    def pending_prompt(self) -> Dict[str, Any]: ...

    def answer_prompt(self, response: bool, nonce: int) -> Dict[str, Any]: ...

    def stop_workflow(self) -> Dict[str, Any]: ...

    def set_supervision(
        self,
        task_name: str,
        supervise: bool,
        supervisor: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def requeue_task(
        self, item_name: str, task_name: str, front: bool = False
    ) -> Dict[str, Any]: ...


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
        result = context.answer_prompt(bool(body.get("response", False)), nonce)
        if result.get("stale"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error_type": "stale_prompt",
                    "message": "That question is no longer pending; "
                    "re-read /app/prompt and answer the current one.",
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
