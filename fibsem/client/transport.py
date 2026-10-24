"""The one connection to an App Server that every client of it shares.

It finds the server, sends the token, follows a restarted server, reads an answer
as the data it is (JSON, an image, an array) and raises a refusal as the exception
it names.
"""

import io
import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import httpx
import numpy as np

from fibsem import __version__
from fibsem.client.pairing import (
    DEFAULT_URL,
    agent_token,
    ask_to_connect,
    forget_agent_token,
)
from fibsem.devices.errors import error_from
from fibsem.devices.wire import (
    FRAME_METADATA_HEADER,
    FRAMES_MEDIA_TYPE,
    NPY_MEDIA_TYPE,
    Frame,
    frames_from_bytes,
)
from fibsem.server.discovery import read_discovery_file
from fibsem.server.images import TIFF_MEDIA_TYPE
from fibsem.server.leases import LEASE_HEADER
from fibsem.server.operator_token import read_operator_token
from fibsem.structures.image import FibsemImage

TIMEOUT = httpx.Timeout(120.0, connect=5.0)  # moves and acquisitions are slow

# httpx logs every request: a remote device's reads would fill the log
for _logger in ("httpx", "httpcore"):
    logging.getLogger(_logger).setLevel(logging.WARNING)

NO_SERVER = (
    "No App Server found. Pass url and token, set FIBSEM_SERVER_URL and "
    "FIBSEM_SERVER_TOKEN, or start the App Server on this machine (it writes "
    "~/.fibsem/app-server.json)."
)


class ServerUnreachable(ConnectionError):
    """No server was found, or it did not answer: ``timed_out`` when a request
    may have reached it."""

    def __init__(self, message: str, timed_out: bool = False):
        super().__init__(message)
        self.timed_out = timed_out


class ServerError(RuntimeError):
    """The server refused or failed with an error that has no local exception.

    ``status`` is the HTTP status (None for a long command's run), ``detail`` what
    the server sent: every App Server refusal has ``error_type`` and ``message``.
    """

    def __init__(self, what: str, status: Optional[int], detail: Any):
        self.status, self.detail = status, detail
        said = detail
        if isinstance(detail, dict) and "message" in detail:
            said = f"{detail.get('error_type')}: {detail['message']}"
        super().__init__(f"{what}: {status} {said}" if status else f"{what}: {said}")

    @property
    def error_type(self) -> Optional[str]:
        return self.detail.get("error_type") if isinstance(self.detail, dict) else None


# The App Server a process is pointed at, set by whoever launches it.
SERVER_URL_ENV = "FIBSEM_SERVER_URL"
SERVER_TOKEN_ENV = "FIBSEM_SERVER_TOKEN"
OPERATOR_TOKEN_ENV = "FIBSEM_OPERATOR_TOKEN"


def server_env() -> Tuple[Optional[str], Optional[str]]:
    """The server url and agent token this process's environment names, each
    None when unset. Read on each call: launchers and tests set them per process."""
    return (
        os.environ.get(SERVER_URL_ENV) or None,
        os.environ.get(SERVER_TOKEN_ENV) or None,
    )


def resolve(
    url: Optional[str] = None, token: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """The server's url and token: as given, else from the environment, else the
    agent token this computer was paired with there (`fibsem.client.pairing`),
    else from the discovery file the App Server on this machine writes (the
    agent token this agent was allowed, else the file's connect token, which
    asks to be allowed), else the agent token paired at the SSH tunnel's end.
    None when none is known (yet). Never an operator's paired token."""
    env_url, env_token = server_env()
    url = url or env_url
    token = token or env_token
    if url and not token:
        token = agent_token(url)
    if url and token:
        return url, token
    found = read_discovery_file()
    if found is not None:
        url = url or found["url"]
        token = (
            token
            or agent_token(url)
            or found.get("token")
            or found.get("connect_token")
        )
        return url, token
    if not url and (paired := agent_token(DEFAULT_URL)) is not None:
        return DEFAULT_URL, token or paired
    return None


# The control lease this thread of control holds (`fibsem.client.control`): its
# requests carry it, so they run inside it.
_lease: ContextVar[Optional[str]] = ContextVar("fibsem_lease", default=None)


@contextmanager
def in_lease(lease: Optional[str]) -> Iterator[None]:
    """Send this thread of control's requests inside *lease*, for the block."""
    token = _lease.set(lease)
    try:
        yield
    finally:
        _lease.reset(token)


def resolve_operator(
    url: Optional[str] = None, token: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """The server's url and the operator's token, for a client that acts as the
    operator: url as given, else ``FIBSEM_SERVER_URL``, else the discovery file;
    token as given, else ``FIBSEM_OPERATOR_TOKEN``, else this install's operator
    token file. Never the agent's token. None when either is unknown."""
    url = url or server_env()[0]
    if not url:
        found = read_discovery_file()
        url = found["url"] if found is not None else None
    token = token or os.environ.get(OPERATOR_TOKEN_ENV) or read_operator_token()
    return (url, token) if url and token else None


def error_for(what: str, status: Optional[int], detail: Any) -> Exception:
    """The exception a server's error answer names: the local one when it has
    one (`error_from`), else `ServerError`."""
    error = error_from(detail, status) if isinstance(detail, dict) else None
    return error if error is not None else ServerError(what, status, detail)


class Transport:
    """One App Server, by url and token (`resolve`).

    A restarted server can come back with a new address or token. On a 401, or a
    connection refused, the transport resolves again and, when that finds another
    server, retries there once. A request that may have reached the server (one
    that timed out) is never retried.

    With ``discover=False`` it is the server at *url* and nothing else, with
    *token* or none: a Device Server, which has no token and no discovery file.

    Holding the discovery file's connect token, its first request asks the
    operator to allow this agent (`ask_to_connect`, up to ``connect_wait_s``)
    and carries on with the token an Allow answers. ``asks = False`` answers
    the refusal instead (a probe that mustn't wait).
    """

    asks = True

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: httpx.Timeout = TIMEOUT,
        discover: bool = True,
        connect_wait_s: float = 120.0,
    ):
        self._asked = (url, token)
        self._discover = discover
        resolved = resolve(url, token) if discover else (url, token)
        if resolved is None:
            raise ServerUnreachable(NO_SERVER)
        self._resolved = resolved
        self._user_agent = user_agent or f"fibsem-client/{__version__}"
        self._timeout = timeout
        self._connect_wait_s = connect_wait_s
        self._http = self._connect()

    @property
    def url(self) -> str:
        return self._resolved[0]

    @property
    def bearer(self) -> Optional[str]:
        """The ``Authorization`` header's value; None with no token."""
        token = self._resolved[1]
        return None if token is None else f"Bearer {token}"

    @property
    def user_agent(self) -> str:
        return self._user_agent

    @user_agent.setter
    def user_agent(self, user_agent: str) -> None:
        """Sent from the next request on, through a reconnect too."""
        self._user_agent = user_agent
        self._http.headers["User-Agent"] = user_agent

    def _connect(self) -> httpx.Client:
        headers = {"User-Agent": self._user_agent}
        if self.bearer is not None:
            headers["Authorization"] = self.bearer
        return self._client(self.url, headers)

    def _client(self, url: str, headers: Dict[str, str]) -> httpx.Client:
        """The HTTP client for *url*; a test's sends to its app in process."""
        return httpx.Client(base_url=url, headers=headers, timeout=self._timeout)

    def _reresolve(self) -> bool:
        if not self._discover:
            return False
        fresh = resolve(*self._asked)
        if fresh is None or fresh == self._resolved:
            return False
        self._http.close()
        self._resolved = fresh
        self._http = self._connect()
        return True

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """The server's answer whatever its status, as httpx's ``request``
        answers it; `fetch` reads it as data."""
        try:
            return self._attempt(method, path, kwargs)
        except httpx.TimeoutException:
            raise ServerUnreachable(
                f"{method} {self.url}{path}: no answer in time", timed_out=True
            ) from None
        except httpx.TransportError as error:
            raise ServerUnreachable(
                f"{method} {self.url}{path}: {type(error).__name__}"
            ) from None

    def _attempt(
        self, method: str, path: str, kwargs: Dict[str, Any]
    ) -> httpx.Response:
        lease = _lease.get()
        if lease is not None:
            kwargs = {
                **kwargs,
                "headers": {**(kwargs.get("headers") or {}), LEASE_HEADER: lease},
            }
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError:  # nothing was sent
            if not self._reresolve():
                raise
            response = self._http.request(method, path, **kwargs)
        if response.status_code == 401:
            self._forget_unpaired()
            if self._reresolve():
                response = self._http.request(method, path, **kwargs)
        if self.asks and _error_type(response) == "connect_first":
            refused = self._ask_to_connect(response)
            if refused is not None:
                return refused
            return self._http.request(method, path, **kwargs)
        return response

    def _forget_unpaired(self) -> None:
        """A kept agent token the server refuses was unpaired: drop it, so
        the next resolve asks to connect again."""
        if self._discover and self._resolved[1] == agent_token(self.url):
            forget_agent_token(self.url)

    def _ask_to_connect(self, response: httpx.Response) -> Optional[httpx.Response]:
        """Ask the operator to allow this agent; on Allow, carry on with its
        token and answer None. Otherwise the refusal to answer: still waiting,
        or denied."""
        try:
            token = ask_to_connect(
                self.url,
                self._resolved[1],
                self._user_agent,
                wait_s=self._connect_wait_s,
            )
        except PermissionError as denied:
            return _refusal(response, "connect_denied", str(denied))
        if token is None:
            return _refusal(
                response,
                "connect_waiting",
                "Waiting for the operator to allow this agent: they click "
                "Allow on the agent card. Try again once they have.",
            )
        self._http.close()
        self._resolved = (self.url, token)
        self._http = self._connect()
        return None

    def fetch(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Any = None,
    ) -> Any:
        """The answer as data: JSON, a `FibsemImage` (a TIFF), an array (npy, a
        `Frame` with its metadata) or a list of frames. An error answer raises
        (`error_for`)."""
        extra: Dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        response = self.request(
            method, path, json=body, params=params, headers=headers, **extra
        )
        return answer_of(response, f"{method} {path}")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def answer_of(response: httpx.Response, what: str) -> Any:
    """*response* as data (`read`), or the error it answers raised (`error_for`),
    *what* naming the call."""
    if response.is_success:
        return read(response)
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):  # not JSON: an unhandled server error
        detail = response.text
    raise error_for(what, response.status_code, detail)


def _detail(response: httpx.Response) -> dict:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return {}
    return detail if isinstance(detail, dict) else {}


def _error_type(response: httpx.Response) -> Optional[str]:
    if response.status_code != 403:
        return None
    return _detail(response).get("error_type")


def _refusal(response: httpx.Response, error_type: str, message: str) -> httpx.Response:
    """A 403 in the server's shape, in place of *response*."""
    return httpx.Response(
        403,
        json={"detail": {"error_type": error_type, "message": message}},
        request=response.request,
    )


def read(response: httpx.Response) -> Any:
    """A successful answer as the data it carries."""
    content_type = response.headers.get("content-type", "")
    if content_type.startswith(TIFF_MEDIA_TYPE):
        return FibsemImage.load(io.BytesIO(response.content))
    if content_type.startswith(FRAMES_MEDIA_TYPE):
        return frames_from_bytes(response.content)
    if content_type.startswith(NPY_MEDIA_TYPE):
        data = np.load(io.BytesIO(response.content), allow_pickle=False)
        metadata = response.headers.get(FRAME_METADATA_HEADER)
        return data if metadata is None else Frame(data, json.loads(metadata))
    return response.json()
