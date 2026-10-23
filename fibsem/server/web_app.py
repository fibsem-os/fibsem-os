"""The web app's files (built from web/ into fibsem/server/ui), served at
``/ui`` by the App Server and the gateway. Asset paths are files; every other
path is a page and gets index.html."""

import mimetypes
import pkgutil

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

__all__ = ["page", "web_app_router"]


def page(filename: str) -> HTMLResponse:
    """A packaged page of fibsem.server, as HTML."""
    found = pkgutil.get_data("fibsem.server", filename)
    if found is None:  # pragma: no cover - packaging defect, not runtime state
        raise HTTPException(status_code=404, detail=f"{filename} not packaged")
    return HTMLResponse(found.decode("utf-8"))


def web_app_router() -> APIRouter:
    """Unauthenticated, like /health: the files are static and every call the
    pages make needs a token."""
    router = APIRouter()

    @router.get("/ui")
    @router.get("/ui/{path:path}")
    def web_app(path: str = ""):
        if path.startswith("assets/") and ".." not in path:
            try:
                asset = pkgutil.get_data("fibsem.server", f"ui/{path}")
            except OSError:
                asset = None
            if asset is None:
                raise HTTPException(status_code=404, detail=f"{path} not found")
            kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return Response(asset, media_type=kind)
        return page("ui/index.html")

    return router
