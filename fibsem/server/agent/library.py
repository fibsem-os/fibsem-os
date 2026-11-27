"""``/app/library``: the agent's playbooks, skills and subagents
(`fibsem.ai.agent.library`), what fibsemOS ships beside the lab's own, for the
web page's Knowledge window. Reading is at Watch; saving (an edit to a
shipped one makes the lab's copy) and removing are the operator's only. The
agent never edits them.
"""

from typing import Any, Optional, Sequence

from fastapi import APIRouter, Depends

from fibsem.ai.agent import library
from fibsem.server.auth import refusal_error, require_monitoring, require_operator
from fibsem.server.bodies import LibraryFileBody
from fibsem.server.responses import LibraryIndex, LibraryItem, answer


def _refused(error: library.LibraryError):
    status = {"not_found": 404, "exists": 409}.get(error.error_type, 422)
    return refusal_error(status, error.error_type, error.message)


def build_library_router(
    read: Optional[Sequence[Any]] = None, write: Optional[Sequence[Any]] = None
) -> APIRouter:
    """The library's routes. *read* and *write* are the dependencies that
    let a caller read and change it: the App Server's own (Watch, and the
    operator) unless a server with its own sign-in (the Hub) gives its own."""
    router = APIRouter(
        prefix="/app/library",
        dependencies=list(read)
        if read is not None
        else [Depends(require_monitoring())],
    )
    operator = list(write) if write is not None else [Depends(require_operator())]

    @router.get("/options", response_model=library.Options)
    def library_options():
        return answer(library.options())

    @router.get("/{kind}", response_model=LibraryIndex)
    def library_index(kind: str):
        try:
            return answer(LibraryIndex(kind, library.items(kind)))
        except library.LibraryError as error:
            raise _refused(error)

    @router.get("/{kind}/{name}", response_model=library.File)
    def library_read(kind: str, name: str, source: Optional[str] = None):
        try:
            return answer(library.read(kind, name, source))
        except library.LibraryError as error:
            raise _refused(error)

    @router.put("/{kind}/{name}", response_model=library.File, dependencies=operator)
    def library_save(kind: str, name: str, body: LibraryFileBody):
        try:
            text = body.text
            if body.data is not None:
                text = library.dump(kind, body.data)
            return answer(library.save(kind, name, text))
        except library.LibraryError as error:
            raise _refused(error)

    @router.post("/{kind}/{name}", response_model=library.File, dependencies=operator)
    def library_create(kind: str, name: str):
        try:
            return answer(library.create(kind, name))
        except library.LibraryError as error:
            raise _refused(error)

    @router.delete("/{kind}/{name}", response_model=LibraryItem, dependencies=operator)
    def library_remove(kind: str, name: str):
        try:
            return answer(library.remove(kind, name))
        except library.LibraryError as error:
            raise _refused(error)

    return router
