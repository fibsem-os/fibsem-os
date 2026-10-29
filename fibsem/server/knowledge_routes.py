"""``/app/knowledge``: the knowledge base (`fibsem.knowledge`) over HTTP.

Reads and writes at Watch, the agent's and the operator's alike; a write says
which of them made it, from the caller's token. Removing an entry is the
operator's only. Images a write cites are paths in the open experiment.
"""

import base64
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image

from fibsem import knowledge
from fibsem.server import files
from fibsem.server.auth import (
    is_operator_token,
    refusal_error,
    require_monitoring,
    require_operator,
)
from fibsem.server.images import preview_jpeg_bytes
from fibsem.server.responses import (
    KnowledgeEntry,
    KnowledgeIndex,
    KnowledgePicture,
    KnowledgeRemoved,
    answer,
)

# The width of a picture knowledge_get answers, as experiment_read_file's.
PICTURE_PX = 768


def _refused(error: knowledge.KnowledgeError) -> HTTPException:
    status = 404 if error.error_type == "not_found" else 422
    return refusal_error(status, error.error_type, error.message)


def _invalid(message: str) -> HTTPException:
    return refusal_error(422, "invalid", message)


def _picture(path: str) -> Dict[str, Any]:
    with Image.open(path) as opened:
        data = np.asarray(opened.convert("RGB"))
    jpeg, width, height = preview_jpeg_bytes(data, max_width=PICTURE_PX)
    return {
        "image_b64_jpeg": base64.b64encode(jpeg).decode("ascii"),
        "width": width,
        "height": height,
    }


def _optional_text(body: Dict[str, Any], key: str) -> Optional[str]:
    value = body.get(key)
    if value is not None and not isinstance(value, str):
        raise _invalid(f"{key} is a string.")
    return value or None


def build_knowledge_router(
    experiment_path: Callable[[], Optional[str]],
) -> APIRouter:
    router = APIRouter(
        prefix="/app/knowledge", dependencies=[Depends(require_monitoring())]
    )

    @router.get("", response_model=KnowledgeIndex)
    def knowledge_index(section: Optional[str] = None, sample: Optional[str] = None):
        try:
            rows = knowledge.entries(section, sample)
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeIndex(rows, knowledge.samples()))

    @router.get("/{section}/{id}", response_model=KnowledgeEntry)
    def knowledge_entry(
        section: str, id: str, sample: Optional[str] = None, pictures: bool = False
    ):
        try:
            entry = knowledge.read(section, id, sample)
            shown: Optional[List[KnowledgePicture]] = None
            if pictures:
                folder = os.path.dirname(entry.path)
                shown = [
                    KnowledgePicture(name, _picture(os.path.join(folder, name)))
                    for name in entry.images
                    if os.path.isfile(os.path.join(folder, name))
                ]
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeEntry(entry, shown))

    @router.get("/{section}/{id}/images/{name}")
    def knowledge_image(section: str, id: str, name: str, sample: Optional[str] = None):
        try:
            path = knowledge.image(section, id, name, sample)
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return FileResponse(str(path), media_type="image/png")

    @router.post("", response_model=KnowledgeEntry)
    def knowledge_write(request: Request, body: Dict[str, Any]):
        by = (
            knowledge.OPERATOR
            if is_operator_token(
                request.app.state.auth, request.headers.get("authorization")
            )
            else knowledge.AGENT
        )
        section = _optional_text(body, "section")
        title, text = body.get("title"), body.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise _invalid("title and text are strings.")
        sample = _optional_text(body, "sample")
        entry_id = _optional_text(body, "id")
        images = body.get("images") or []
        if not isinstance(images, list) or not all(
            isinstance(path, str) for path in images
        ):
            raise _invalid("images is a list of paths in the open experiment.")
        experiment = experiment_path()
        if experiment and not os.path.isdir(experiment):
            experiment = None
        name = os.path.basename(os.path.normpath(experiment)) if experiment else None
        if section == knowledge.EXPERIMENT and entry_id is None:
            if name is None:
                raise _invalid(
                    "No experiment is open: give the experiment's name as id."
                )
            entry_id = name
        if images and experiment is None:
            raise refusal_error(
                409,
                "no_experiment",
                "No experiment is open: images are paths in the open experiment.",
            )
        try:
            sources = [files.resolve(experiment, path) for path in images]
        except files.FileRefusal as refusal:
            raise HTTPException(status_code=refusal.status, detail=refusal.detail)
        try:
            entry = knowledge.write(
                section,
                title,
                text,
                by,
                sample=sample,
                id=entry_id,
                experiment=name,
                images=sources,
            )
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeEntry(entry))

    @router.delete(
        "/{section}/{id}",
        response_model=KnowledgeRemoved,
        dependencies=[Depends(require_operator())],
    )
    def knowledge_delete(section: str, id: str, sample: Optional[str] = None):
        try:
            entry = knowledge.delete(section, id, sample)
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeRemoved(True, entry))

    return router
