"""``/app/library``: the agent's profiles, playbooks and skills
(`fibsem.ai.agent.library`), what fibsemOS ships beside the lab's own, for the
web page's Knowledge window. Reading is at Watch; saving (an edit to a
shipped one makes the lab's copy) and removing are the operator's only. The
agent never edits them.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from fibsem.ai.agent import library
from fibsem.server.auth import refusal_error, require_monitoring, require_operator
from fibsem.server.responses import LibraryIndex, LibraryItem, answer


def _refused(error: library.LibraryError):
    status = 404 if error.error_type == "not_found" else 422
    return refusal_error(status, error.error_type, error.message)


def build_library_router() -> APIRouter:
    router = APIRouter(
        prefix="/app/library", dependencies=[Depends(require_monitoring())]
    )
    operator = [Depends(require_operator())]

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
    def library_save(kind: str, name: str, body: Dict[str, Any]):
        try:
            return answer(library.save(kind, name, body.get("text")))
        except library.LibraryError as error:
            raise _refused(error)

    @router.delete("/{kind}/{name}", response_model=LibraryItem, dependencies=operator)
    def library_remove(kind: str, name: str):
        try:
            return answer(library.remove(kind, name))
        except library.LibraryError as error:
            raise _refused(error)

    return router
