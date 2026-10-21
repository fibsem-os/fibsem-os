"""The one connection to an App Server that every client of it shares.

It finds the server, sends the token, follows a restarted server, reads an answer
as the data it is (JSON, an image, an array) and raises a refusal as the exception
it names.
"""

import io
import json
import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx
import numpy as np

from fibsem import __version__
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
from fibsem.server.operator_token import read_operator_token
from fibsem.structures import FibsemImage

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


def resolve(
    url: Optional[str] = None, token: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """The server's url and token: as given, else from the environment, else from
    the discovery file the App Server on this machine writes. None when none is
    known (yet)."""
    url = url or os.environ.get("FIBSEM_SERVER_URL")
    token = token or os.environ.get("FIBSEM_SERVER_TOKEN")
    if url and token:
        return url, token
    found = read_discovery_file()
    if found is not None:
        return url or found["url"], token or found["token"]
    return None


def resolve_operator(
    url: Optional[str] = None, token: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """The server's url and the operator's token, for a client that acts as the
    operator: url as given, else ``FIBSEM_SERVER_URL``, else the discovery file;
    token as given, else ``FIBSEM_OPERATOR_TOKEN``, else this install's operator
    token file. Never the agent's token. None when either is unknown."""
    url = url or os.environ.get("FIBSEM_SERVER_URL")
    if not url:
        found = read_discovery_file()
        url = found["url"] if found is not None else None
    token = token or os.environ.get("FIBSEM_OPERATOR_TOKEN") or read_operator_token()
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
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: httpx.Timeout = TIMEOUT,
        discover: bool = True,
    ):
        self._asked = (url, token)
        self._discover = discover
        resolved = resolve(url, token) if discover else (url, token)
        if resolved is None:
            raise ServerUnreachable(NO_SERVER)
        self._resolved = resolved
        self._user_agent = user_agent or f"fibsem-client/{__version__}"
        self._timeout = timeout
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
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError:  # nothing was sent
            if not self._reresolve():
                raise
            return self._http.request(method, path, **kwargs)
        if response.status_code == 401 and self._reresolve():
            return self._http.request(method, path, **kwargs)
        return response

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
        if response.is_success:
            return read(response)
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):  # not JSON: an unhandled server error
            detail = response.text
        raise error_for(f"{method} {path}", response.status_code, detail)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


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
