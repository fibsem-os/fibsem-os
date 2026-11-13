"""Mount the agent's routes on the App Server (`fibsem.server.server`)."""

from typing import TYPE_CHECKING, Any, Callable, FrozenSet, Mapping, Optional

from fastapi import FastAPI

from fibsem.server.agent.files import AgentFiles
from fibsem.server.agent.library import build_library_router
from fibsem.server.agent.plan import PlanStore, build_agent_plan_router
from fibsem.server.agent.requests import RequestStore, build_request_router
from fibsem.server.agent.session import build_agent_session_router
from fibsem.server.agent.transcript import build_agent_transcript_router

if TYPE_CHECKING:
    from fibsem.server.app_context import AppContext


def mount_agent_routes(
    app: FastAPI,
    app_context: "AppContext",
    auth,
    mounted: FrozenSet[str],
    refused_path: Callable[[Mapping[str, Any]], Optional[str]],
) -> None:
    """The agent's requests, session, plan, transcript and library, over
    *app_context*. *mounted* are the server's parts a request may call;
    *refused_path* says why a request's path argument is refused."""

    def experiment_path() -> Optional[str]:
        return app_context.experiment_path

    emit = app_context.record_event
    # The agent's requests: actions the operator approves in the app, run
    # through the server's own routes. The app decides them through
    # app.state.requests.
    app.state.requests = RequestStore(
        app,
        auth,
        mounted,
        emit=emit,
        describe=app_context.describe_action,
        check=app_context.check_action,
        not_allowed=app.state.blocked_route,
        experiment_path=experiment_path,
        experiment=app_context.recording_experiment,
        path_refused=refused_path,
    )
    app.include_router(build_request_router(app.state.requests))
    # The agent's own report of its session: model, goal, budget, spend.
    app.include_router(build_agent_session_router(emit))
    # Where each session's plan and transcript go: its experiment, or its own
    # folder until one is open.
    app.state.agent_files = AgentFiles(experiment_path)
    app.state.agent_plan = PlanStore(
        experiment_path,
        lambda: app_context.experiment_sample,
        app_context.set_experiment_sample,
        files=app.state.agent_files,
    )
    app.include_router(build_agent_plan_router(experiment_path, emit))
    # What fibsem-agent printed.
    app.include_router(build_agent_transcript_router(experiment_path))
    # The agent's playbooks, skills and subagents: shipped and the lab's.
    app.include_router(build_library_router())
