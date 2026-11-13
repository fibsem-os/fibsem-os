"""Write web/openapi.json: the classes the App Server's routes take and answer,
from its /openapi.json with every route the web app reads (/app, the agent's and the
operator's) and no microscope connected. `bun run types` in web/ runs it and
writes web/src/lib/schema.ts from it; CI fails when that file is stale."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402

OUT = ROOT / "web" / "openapi.json"


def schema() -> dict:
    """The App Server's /openapi.json, with AutoLamella's /app routes."""
    session = AppSession()
    app = build_server(
        app_context=AutoLamellaAppContext(session),
        session=session,
        # The operator's routes read their agent only when called.
        operator=object(),
        auth=AuthConfig(token="schema"),
    )
    return app.openapi()


def _refs(value) -> set:
    """The classes *value* names, by name."""
    if isinstance(value, dict):
        ref = value.get("$ref", "")
        own = {ref.rsplit("/", 1)[-1]} if ref else set()
        return own.union(*(_refs(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_refs(v) for v in value))
    return set()


def _named(full: dict, part: str) -> set:
    """The classes the routes name in *part* of each operation ("responses" or
    "requestBody"), and the classes in those, by name."""
    schemas = full["components"]["schemas"]
    found = set()
    todo = _refs(
        [
            operation.get(part, {})
            for operations in full["paths"].values()
            for operation in operations.values()
        ]
    )
    while todo:
        name = todo.pop()
        if name not in found and name in schemas:
            found.add(name)
            todo |= _refs(schemas[name])
    return found


def _plain(value, answered: bool):
    """*value* without pydantic's titles, which repeat the names. An answer has
    every field present, since `fibsem.codec.encode` writes each one; a body
    keeps the fields it may leave out."""
    if isinstance(value, list):
        return [_plain(v, answered) for v in value]
    if not isinstance(value, dict):
        return value
    out = {
        k: _plain(v, answered)
        for k, v in value.items()
        if not (k == "title" and isinstance(v, str))
    }
    if answered and isinstance(value.get("properties"), dict):
        out["required"] = sorted(value["properties"])
    return out


def classes() -> dict:
    """The classes the routes take and answer: the web names them by class, not
    by path. A class that is both is written as an answer."""
    full = schema()
    schemas = full["components"]["schemas"]
    answered = _named(full, "responses")
    taken = _named(full, "requestBody") - answered
    return {
        "openapi": full["openapi"],
        "info": full["info"],
        "paths": {},
        "components": {
            "schemas": {
                n: _plain(schemas[n], n in answered) for n in sorted(answered | taken)
            }
        },
    }


if __name__ == "__main__":
    text = json.dumps(classes(), indent=1, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8", newline="\n")
