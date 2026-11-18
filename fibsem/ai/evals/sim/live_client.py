"""Tool calls to a live run's App Server, as an agent or the operator makes
them, waiting for a long command's run to end."""

import json
import time
from pathlib import Path
from typing import Any, Dict

import httpx

from fibsem.client.pairing import ask_to_connect
from fibsem.client.tools import send
from fibsem.client.transport import Transport
from fibsem.server.operator_token import read_operator_token

RUN_WAIT_S = 30.0
RUN_TIMEOUT_S = 1800.0


class LiveError(RuntimeError):
    pass


class LiveClient:
    """Calls the catalog's tools on one App Server with one token."""

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.transport = Transport(url, token)

    @classmethod
    def as_agent(cls, found_at: Path, name: str, wait_s: float = 30) -> "LiveClient":
        """Connect to the server *found_at* names as the agent *name*, which
        the server lets in without a click (``allow_local_agents``)."""
        found = json.loads(Path(found_at).read_text())
        token = ask_to_connect(found["url"], found["connect_token"], name, wait_s)
        if not token:
            raise LiveError(f"The App Server did not let {name} connect.")
        return cls(found["url"], token)

    @classmethod
    def as_operator(cls, found_at: Path, home: Path) -> "LiveClient":
        """The operator, with the token the server wrote in its state root."""
        found = json.loads(Path(found_at).read_text())
        return cls(
            found["url"], read_operator_token(Path(home) / "operator-token.json")
        )

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "LiveClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def set_level(self, level: str) -> None:
        """The agent's level, as the operator sets it."""
        answer = httpx.put(
            f"{self.url}/app/agent/level",
            json={"level": level},
            headers={"Authorization": f"Bearer {self.token}"},
        )
        if not answer.is_success:
            raise LiveError(f"Could not set the agent's level: {answer.text[:300]}")

    def call(self, tool: str, **arguments: Any) -> Any:
        """A tool call's result; a long command's once its run has ended."""
        answer = send(self.transport, tool, arguments)
        if not answer.is_success:
            raise LiveError(f"{tool}: {answer.status_code} {answer.text[:500]}")
        body = answer.json()
        result = body.get("result", body) if isinstance(body, dict) else body
        if (
            isinstance(result, dict)
            and isinstance(result.get("run_id"), str)
            and result.get("state") == "running"
        ):
            return self.wait_for_run(result["service"], result["run_id"])
        if isinstance(result, dict) and result.get("state") in ("failed", "cancelled"):
            raise LiveError(f"{tool} {result['state']}: {result.get('error')}")
        if (
            isinstance(result, dict)
            and result.get("state") == "done"
            and "run_id" in result
        ):
            return result.get("result")
        return result

    def wait_for_run(
        self, service: str, run_id: str, timeout_s: float = RUN_TIMEOUT_S
    ) -> Any:
        """The result of the run *run_id* on *service* once it is done;
        raises when it fails, is cancelled or goes past *timeout_s*."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            answer = send(
                self.transport,
                "service_wait_for_run",
                {"service": service, "run_id": run_id, "wait": RUN_WAIT_S},
            )
            if not answer.is_success:
                raise LiveError(f"waiting on {service}: {answer.text[:500]}")
            body = answer.json()
            run: Dict[str, Any] = body.get("run") or body.get("result") or {}
            if run.get("run_id") not in (None, run_id):
                raise LiveError(f"{service} run {run_id} was replaced by another")
            if run.get("state") == "running":
                continue
            if run.get("state") != "done":
                raise LiveError(
                    f"{service} {run.get('command')} {run.get('state')}: "
                    f"{run.get('error')}"
                )
            return run.get("result")
        raise LiveError(f"{service} run {run_id} did not end in {timeout_s:.0f} s")
