"""Write web/openapi.json: the classes the App Server's routes answer, from its
/openapi.json with every route the web app reads (/app, the agent's and the
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


def _answered(full: dict) -> list:
    """The classes the routes answer, and the classes in those, by name."""
    schemas = full["components"]["schemas"]
    found = set()
    todo = _refs(
        [
            operation.get("responses", {})
            for operations in full["paths"].values()
            for operation in operations.values()
        ]
    )
    while todo:
        name = todo.pop()
        if name not in found and name in schemas:
            found.add(name)
            todo |= _refs(schemas[name])
    return sorted(found)


def _as_answered(value):
    """*value* as an answer reads: every field present, since `fibsem.codec.encode`
    writes each one, and without pydantic's titles, which repeat the names."""
    if isinstance(value, list):
        return [_as_answered(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {
        k: _as_answered(v)
        for k, v in value.items()
        if not (k == "title" and isinstance(v, str))
    }
    if isinstance(value.get("properties"), dict):
        out["required"] = sorted(value["properties"])
    return out


def classes() -> dict:
    """The classes the routes answer: the web reads answers by class, not by path."""
    full = schema()
    schemas = full["components"]["schemas"]
    return {
        "openapi": full["openapi"],
        "info": full["info"],
        "paths": {},
        "components": {
            "schemas": {n: _as_answered(schemas[n]) for n in _answered(full)}
        },
    }


if __name__ == "__main__":
    text = json.dumps(classes(), indent=1, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8", newline="\n")
