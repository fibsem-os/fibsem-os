"""What the App Server serves, and the operator's routes that connect a microscope.

The server runs over the session whether or not a microscope is connected:
each ``/microscope`` route reads what is served when it is called
(:class:`ServedInstrument`): the session's microscope, plus any devices built
from a Device Server file with no microscope. A route answers 409
``not_connected`` when nothing is served, and the state routes when no
microscope is.
Connecting and disconnecting are the session's (``AppSession.connect``), the
same calls the window's connection dialog makes; the routes here are the
operator's only, never the agent's.
"""

from __future__ import annotations

import threading
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException

from fibsem import config
from fibsem.devices.runs import RUNNING
from fibsem.instrument import CONNECTED_MESSAGE, Instrument, needs_microscope
from fibsem.server.auth import require_operator
from fibsem.server.leases import Leases
from fibsem.server.responses import (
    ConfigurationOption,
    ConnectedConfiguration,
    ConnectionView,
    RunAnswer,
    answer,
)

__all__ = [
    "NOT_CONNECTED_MESSAGE",
    "ServedInstrument",
    "build_connection_router",
]

NOT_CONNECTED_MESSAGE = (
    "No microscope is connected: the operator connects one in the app."
)


def not_connected() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"error_type": "not_connected", "message": NOT_CONNECTED_MESSAGE},
    )


class ServedInstrument:
    """What a server serves: a microscope (a fixed one, or the session's at each
    call) and any *instruments* with no microscope, such as a `DeviceInstrument`.

    Devices and services are the union, the microscope's first; a name the
    microscope has wins. Control is the microscope's while one is connected,
    else the first other instrument's; its leases are made for each control."""

    def __init__(
        self,
        microscope: Any = None,
        session: Any = None,
        instruments: Sequence[Instrument] = (),
    ) -> None:
        self._microscope = microscope
        self._session = session
        self._instruments = list(instruments)
        self._leases: Optional[Leases] = None
        self._lock = threading.Lock()

    @property
    def current(self) -> Any:
        """The microscope now, or None while none is connected."""
        if self._session is not None:
            return self._session.microscope
        return self._microscope

    def require(self) -> Any:
        """The microscope now, or 409 ``not_connected`` (saying what is
        connected instead, while only devices are)."""
        microscope = self.current
        if microscope is None:
            others = self.instruments
            if not others:
                raise not_connected()
            raise HTTPException(
                status_code=409,
                detail={
                    "error_type": "not_connected",
                    "message": needs_microscope(others[0]),
                },
            )
        return microscope

    @property
    def instruments(self) -> List[Instrument]:
        """Everything served now: the microscope first, while one is connected,
        then the session's devices while a ``devices`` configuration is."""
        microscope = self.current
        served = [microscope] if microscope is not None else []
        connected = getattr(self._session, "instrument", None)
        if connected is not None and connected is not microscope:
            served.append(connected)
        return served + self._instruments

    @property
    def connected(self) -> bool:
        """Whether anything is served now."""
        return bool(self.instruments)

    def _union(self, part: str) -> Mapping[str, Any]:
        merged: Dict[str, Any] = {}
        for instrument in reversed(self.instruments):
            merged.update(getattr(instrument, part))
        return MappingProxyType(merged)

    @property
    def devices(self) -> Mapping[str, Any]:
        """Every served device by name, or 409 ``not_connected`` with nothing served."""
        if not self.connected:
            raise not_connected()
        return self._union("devices")

    @property
    def services(self) -> Mapping[str, Any]:
        """Every served service by name, or 409 ``not_connected`` with nothing served."""
        if not self.connected:
            raise not_connected()
        return self._union("services")

    def find(self, part: str, name: str) -> Any:
        """The served device (*part* ``devices``) or service named *name*, or None."""
        for instrument in self.instruments:
            found = getattr(instrument, part).get(name)
            if found is not None:
                return found
        return None

    @property
    def control_now(self) -> Any:
        """Who is in control, or None while nothing is served."""
        instruments = self.instruments
        return instruments[0].control if instruments else None

    @property
    def control(self) -> Any:
        """Who is in control, or 409 ``not_connected`` while nothing is served."""
        control = self.control_now
        if control is None:
            raise not_connected()
        return control

    @property
    def leases(self) -> Leases:
        """The leases on the served control, or 409."""
        control = self.control
        with self._lock:
            if self._leases is None or self._leases.control is not control:
                self._leases = Leases(control)
            return self._leases

    @property
    def kinds(self) -> List[str]:
        """The kind of each served instrument, the microscope's first."""
        return [instrument.kind for instrument in self.instruments]

    def record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Put an event on the record: the microscope's while one is
        connected, else the session's own stream, else the first instrument's."""
        microscope = self.current
        if microscope is not None:
            microscope.record_event(kind, payload)
        elif self._session is not None:
            self._session.record_event(kind, payload)
        elif self._instruments:
            self._instruments[0].record_event(kind, payload)


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
            if session.instrument is not None:
                raise _refused(
                    409, "connected", f"{CONNECTED_MESSAGE}; disconnect it first."
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
