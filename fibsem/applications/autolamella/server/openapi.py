"""The AutoLamella App Server's API, as its ``/openapi.json`` answers it.

Checked in beside this module (``openapi.json``), so a change to what a route
takes or answers shows in review, and the web app builds its types from it. A
test fails while the file is out of date; regenerate it with::

    python -m fibsem.applications.autolamella.server.openapi
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

SNAPSHOT = Path(__file__).with_name("openapi.json")


def openapi() -> Dict[str, Any]:
    """The schema of the App Server hosting AutoLamella on the Demo, every part
    and the operator's routes mounted."""
    import fastapi

    import fibsem
    from fibsem import utils
    from fibsem.ai.agent.session import AgentSession
    from fibsem.applications.autolamella.server import AutoLamellaAppContext
    from fibsem.server import AuthConfig, build_server

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(None),
        auth=AuthConfig.generate(),
        operator=AgentSession(None),
    )
    schema = app.openapi()
    # The installed fibsem's version is a default; it isn't the API.
    _drop_default(schema, fibsem.__version__)
    schema["info"]["x-fastapi"] = fastapi.__version__
    return schema


def _drop_default(node: Any, value: str) -> None:
    if isinstance(node, dict):
        if node.get("default") == value:
            del node["default"]
        for child in node.values():
            _drop_default(child, value)
    elif isinstance(node, list):
        for child in node:
            _drop_default(child, value)


def written(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, indent=1, sort_keys=True) + "\n"


if __name__ == "__main__":
    SNAPSHOT.write_text(written(openapi()), encoding="utf-8")
    print(f"wrote {SNAPSHOT}")
