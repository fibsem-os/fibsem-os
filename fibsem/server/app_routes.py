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

from typing import Any, Dict, List

from fastapi import APIRouter

try:  # Protocol is typing-only; keep import robust on the 3.8 floor
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

__all__ = ["AppContext", "build_app_router"]


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

    return router
