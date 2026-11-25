"""The gateway page's Settings and Customise: this PC's keys and its
connectors (``mcp.yaml``), and (to come) its install preferences and
playbooks.

Keys are write-only here. The page sets or removes one and reads whether it
is set and where from, never its value: no route answers with a key, and
none is logged. Only the operator at this PC (this install's own operator
token, the gateway's window or the page it opens) changes a key; a computer
paired as operator, a hub or an agent never does.

The names a key may be stored under are `keystore.KEYS` and the token name
of each connector in ``mcp.yaml`` (`keystore.check_name`). A connector's
token is stored with where it may go (`extra_mcp.pin_token`), so editing the
connector's address can't send it somewhere new.

Connectors are edited by any operator, at this PC or paired, and checked as
the agent reads them, so a connector that wouldn't load can't be saved. A
command connector runs a program on this PC at each agent start: it is
added, changed, removed and tested only at this PC, and a new or changed
command is saved only once the operator confirms its command line.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fibsem import keystore
from fibsem.ai.agent import extra_mcp

__all__ = [
    "MAX_KEY_LENGTH",
    "ConnectorRefused",
    "KeyRefused",
    "connectors",
    "key_rows",
    "remove_connector",
    "remove_key",
    "save_connector",
    "set_key",
    "test_connector",
]

# Longer than any API key or token, short enough that a pasted file isn't one.
MAX_KEY_LENGTH = 4096


class KeyRefused(ValueError):
    """Why a key can't be set or removed, in words, with a status."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type


def _connectors() -> Dict[str, extra_mcp.Server]:
    try:
        return extra_mcp.load()
    except extra_mcp.ExtraMcpError:
        return {}


def key_rows(environ: Optional[Dict[str, str]] = None) -> List[dict]:
    """Each key fibsemOS reads, then each connector's token: whether it is
    set and where from, and where a connector's token may go. Never a key."""
    environ = os.environ if environ is None else environ
    store = keystore.store_name()
    connectors = _connectors()
    used_by: Dict[str, List[str]] = {}
    for server in connectors.values():
        if server.token:
            used_by.setdefault(server.token, []).append(server.name)
    rows = []
    names = list(keystore.KEYS) + sorted(set(used_by) - set(keystore.KEYS))
    for name in names:
        if environ.get(name):
            source = "environment"
        elif keystore.stored_secret(name):
            source = store or "key store"
        else:
            source = None
        row = {
            "name": name,
            "about": keystore.KEYS.get(name)
            or f"the token for {', '.join(used_by[name])}",
            "set": source is not None,
            "source": source,
            "connectors": used_by.get(name, []),
            "sent_to": (
                keystore.pinned(name) if source not in (None, "environment") else None
            ),
            "goes_to": extra_mcp.places_for(name, connectors),
        }
        rows.append(row)
    return rows


def _checked(name: str) -> None:
    try:
        keystore.check_name(name, extra_mcp.token_names())
    except keystore.KeyNameRefused as refused:
        raise KeyRefused(422, "bad_key_name", str(refused)) from None


def set_key(name: str, value: object) -> dict:
    """Store *value* under *name*, and where a connector's token may go.
    Raises `KeyRefused`."""
    _checked(name)
    if not isinstance(value, str) or not value.strip():
        raise KeyRefused(422, "no_key", "Paste the key: nothing was given.")
    value = value.strip()
    if len(value) > MAX_KEY_LENGTH or any(c in value for c in "\r\n"):
        raise KeyRefused(
            422, "bad_key", "That isn't a key: it's too long or has lines."
        )
    try:
        keystore.set_secret(name, value)
        extra_mcp.pin_token(name, _connectors())
    except keystore.SecretStoreError as error:
        raise KeyRefused(409, "no_key_store", str(error)) from None
    return _row(name)


def remove_key(name: str) -> dict:
    """Remove the stored key *name*. Raises `KeyRefused`."""
    if name in keystore.RESERVED:
        _checked(name)
    try:
        keystore.delete_secret(name)
    except keystore.SecretStoreError as error:
        raise KeyRefused(409, "no_key_store", str(error)) from None
    return _row(name)


def _row(name: str) -> dict:
    found = [row for row in key_rows() if row["name"] == name]
    return found[0] if found else {"name": name, "set": False, "source": None}


# -- connectors (mcp.yaml) ----------------------------------------------------


class ConnectorRefused(KeyRefused):
    """Why a connector can't be saved, removed or tested."""


def _raw(file: Optional[Path] = None) -> Dict[str, Any]:
    file = file or extra_mcp.path()
    if not file.is_file():
        return {}
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConnectorRefused(
            409, "bad_connectors_file", f"{file} can't be read: {error}"
        ) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConnectorRefused(
            409,
            "bad_connectors_file",
            f"{file}: each connector is a name with its settings under it",
        )
    return data


def connectors(file: Optional[Path] = None) -> dict:
    """``{file, connectors: [{name, entry, problem}], problem}``: each
    connector as the file has it, with why it can't be used, if it can't."""
    file = file or extra_mcp.path()
    try:
        raw = _raw(file)
    except ConnectorRefused as refused:
        return {"file": str(file), "connectors": [], "problem": str(refused)}
    rows = []
    for name, entry in raw.items():
        try:
            extra_mcp._server(name, entry)
            problem = None
        except extra_mcp.ExtraMcpError as error:
            problem = str(error)
        rows.append({"name": str(name), "entry": entry, "problem": problem})
    return {"file": str(file), "connectors": rows, "problem": None}


def command_line(entry: Any) -> Optional[str]:
    """A command connector's command, as one line; None for a URL one."""
    command = entry.get("command") if isinstance(entry, dict) else None
    if not isinstance(command, list):
        return None
    return " ".join(str(part) for part in command)


def _at_this_pc(name: str, local: bool, what: str) -> None:
    if not local:
        raise ConnectorRefused(
            403,
            "at_this_pc_only",
            f"{name} runs a program on this PC: {what} only at this PC, on the "
            "gateway's own page.",
        )


def save_connector(
    name: str,
    entry: Any,
    local: bool,
    confirm: Optional[str] = None,
    file: Optional[Path] = None,
) -> dict:
    """Add or replace connector *name* with *entry*, checked as the agent
    reads it. A command connector is saved only at this PC (*local*), and a
    new or changed command only once *confirm* repeats its command line.
    The file before is kept as ``mcp.yaml.bak``. Raises `ConnectorRefused`."""
    file = file or extra_mcp.path()
    raw = _raw(file)
    try:
        extra_mcp._server(name, entry)
    except extra_mcp.ExtraMcpError as error:
        raise ConnectorRefused(422, "bad_connector", str(error)) from None
    before = raw.get(name)
    line = command_line(entry)
    if line is not None or command_line(before) is not None:
        _at_this_pc(name, local, "it is changed")
    if line is not None and line != command_line(before) and confirm != line:
        raise ConnectorRefused(
            428,
            "confirm_command",
            f"Saving {name} runs this on this PC at each agent start: {line}",
        )
    raw[name] = entry
    _write(file, raw)
    return {"name": name, "entry": entry, "problem": None}


def remove_connector(name: str, local: bool, file: Optional[Path] = None) -> None:
    file = file or extra_mcp.path()
    raw = _raw(file)
    if name not in raw:
        raise ConnectorRefused(404, "no_connector", f"No connector {name}.")
    if command_line(raw[name]) is not None:
        _at_this_pc(name, local, "it is removed")
    del raw[name]
    _write(file, raw)


def _write(file: Path, raw: Dict[str, Any]) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.is_file():
        shutil.copyfile(file, file.with_name(file.name + ".bak"))
    text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    staged = file.with_name(file.name + ".new")
    staged.write_text(text, encoding="utf-8")
    os.replace(staged, file)


TEST_TIMEOUT_S = 20.0


def test_connector(name: str, local: bool, file: Optional[Path] = None) -> dict:
    """``{tools}``: the tools connector *name* offers, asked as the agent
    asks, with its token only where it was stored for. Raises
    `ConnectorRefused`."""
    import asyncio
    import threading

    raw = _raw(file)
    if name not in raw:
        raise ConnectorRefused(404, "no_connector", f"No connector {name}.")
    if command_line(raw[name]) is not None:
        _at_this_pc(name, local, "it is tested")
    try:
        server = extra_mcp._server(name, raw[name])
        secrets = extra_mcp.tokens({name: server})
    except extra_mcp.ExtraMcpError as error:
        raise ConnectorRefused(409, "bad_connector", str(error)) from None
    found: Dict[str, Any] = {}

    def run() -> None:
        try:
            found["tools"] = asyncio.run(
                asyncio.wait_for(extra_mcp._tools_of(server, secrets), TEST_TIMEOUT_S)
            )
        except BaseException as error:  # noqa: BLE001 - said to the caller
            found["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(TEST_TIMEOUT_S + 5)
    if "tools" in found:
        return {"tools": found["tools"]}
    error = found.get("error")
    why = (
        "it didn't answer in time"
        if error is None or isinstance(error, asyncio.TimeoutError)
        else type(error).__name__
    )
    raise ConnectorRefused(
        502, "connector_failed", f"{name} couldn't be listed: {why}."
    )
