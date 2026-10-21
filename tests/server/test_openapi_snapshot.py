"""The checked-in openapi.json is what the App Server answers."""

import json

import pytest

pytest.importorskip("fastapi")

from fibsem.applications.autolamella.server.openapi import SNAPSHOT, openapi, written


def test_openapi_snapshot_is_current():
    import fastapi

    snapshot = SNAPSHOT.read_text(encoding="utf-8")
    wrote_with = json.loads(snapshot)["info"].get("x-fastapi")
    if wrote_with != fastapi.__version__:
        pytest.skip(f"openapi.json was written with FastAPI {wrote_with}")
    assert written(openapi()) == snapshot, (
        "openapi.json is out of date: run "
        "python -m fibsem.applications.autolamella.server.openapi"
    )
