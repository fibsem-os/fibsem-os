"""The microscope the App Server serves, and the operator's routes that connect it.

The server runs over the session whether or not a microscope is connected:
each ``/microscope`` route reads the session's microscope when it is called
(:class:`ServedMicroscope`), and answers 409 ``not_connected`` without one.
Connecting and disconnecting are the session's (``AppSession.connect``), the
same calls the window's connection dialog makes; the routes here are the
operator's only, never the agent's.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from fibsem import config
from fibsem.devices.runs import RUNNING
from fibsem.server.auth import require_operator
from fibsem.server.leases import Leases
from fibsem.server.responses import (
    ConfigurationOption,
    ConnectedConfiguration,
    ConnectionView,
    RunAnswer,
    answer,
)

__all__ = ["NOT_CONNECTED_MESSAGE", "ServedMicroscope", "build_connection_router"]

NOT_CONNECTED_MESSAGE = (
    "No microscope is connected: the operator connects one in the app."
)


def not_connected() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"error_type": "not_connected", "message": NOT_CONNECTED_MESSAGE},
    )


class ServedMicroscope:
    """The microscope a server serves: a fixed one, or the session's at each
    call. Its control's leases are made for each microscope it connects."""

    def __init__(self, microscope: Any = None, session: Any = None) -> None:
        self._microscope = microscope
        self._session = session
        self._leases: Optional[Leases] = None
        self._lock = threading.Lock()

    @property
    def current(self) -> Any:
        """The microscope now, or None while none is connected."""
        if self._session is not None:
            return self._session.microscope
        return self._microscope

    def require(self) -> Any:
        """The microscope now, or 409 ``not_connected``."""
        microscope = self.current
        if microscope is None:
            raise not_connected()
        return microscope

    @property
    def leases(self) -> Leases:
        """The leases on the connected microscope's control, or 409."""
        control = self.require().control
        with self._lock:
            if self._leases is None or self._leases.control is not control:
                self._leases = Leases(control)
            return self._leases

    def record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Put an event on the record: the microscope's while one is
        connected, else the session's own stream."""
        microscope = self.current
        if microscope is not None:
            microscope.record_event(kind, payload)
        elif self._session is not None:
            self._session.record_event(kind, payload)


def connection_view(session: Any) -> ConnectionView:
    state = session.connection
    configuration = state.get("configuration")
    run = state.get("run")
    if run is not None:
        run = {k: v for k, v in run.items() if k != "result"}
    return ConnectionView(
        connected=state["connected"],
        connecting=state.get("connecting"),
        error=state.get("error"),
        configuration=(
            ConnectedConfiguration(**configuration)
            if configuration is not None
            else None
        ),
        configurations=[ConfigurationOption(**c) for c in state["configurations"]],
        run=RunAnswer(**run) if run is not None else None,
    )


def _refused(status_code: int, error_type: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error_type": error_type, "message": message},
    )


def build_connection_router(session: Any) -> APIRouter:
    """``/microscope/connection``, ``/microscope/connect`` and
    ``/microscope/disconnect``, behind the operator token."""
    router = APIRouter(prefix="/microscope", dependencies=[Depends(require_operator())])
    starting = threading.Lock()

    @router.get("/connection", response_model=ConnectionView)
    def get_connection():
        return answer(connection_view(session))

    @router.post("/connect", response_model=ConnectionView, status_code=202)
    def connect(body: Optional[Dict[str, Any]] = None):
        """Connect with the named configuration (the default one without a
        name), as a long command: the answer is at once, and the page reads
        ``/microscope/connection`` for the steps and the end."""
        name = (body or {}).get("configuration") or config.DEFAULT_CONFIGURATION_NAME
        entry = config.USER_CONFIGURATIONS.get(name)
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path:
            raise _refused(
                404, "not_found", f"No configuration named {name!r} is registered."
            )
        with starting:
            if session.microscope is not None:
                raise _refused(
                    409, "connected", "A microscope is connected; disconnect it first."
                )
            run = session.connect_runs.latest
            if session.connecting is not None or (
                run is not None and run.state == RUNNING
            ):
                raise _refused(409, "connecting", "A connect is already under way.")
            session.connect_runs.start("connect", lambda: _connect(session, path))
        return answer(connection_view(session), status_code=202)

    @router.post("/disconnect", response_model=ConnectionView)
    def disconnect():
        try:
            session.disconnect()
        except RuntimeError as refused:
            raise _refused(409, "run_on", str(refused)) from None
        return answer(connection_view(session))

    return router


def _connect(session: Any, path: str) -> Optional[str]:
    session.connect(path)
    return None
