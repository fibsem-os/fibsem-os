"""``/app/knowledge``: the knowledge base (`fibsem.knowledge`) over HTTP.

Reads and writes at Watch, the agent's and the operator's alike; a write says
which of them made it, from the caller's token. Keeping, resolving and removing
an entry are the operator's only. Images a write cites are paths in the open
experiment, copied in as it is written; the entry keeps the experiment's name
as its session tag and the microscope's as its instrument tag, never a path.
"""

import base64
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml
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


def _instrument(request: Request) -> Optional[str]:
    """The connected microscope's configured name, if it has one."""
    served = getattr(request.app.state, "served", None)
    microscope = served.current if served is not None else None
    try:
        name = microscope.system.info.name
    except AttributeError:
        return None
    return name if isinstance(name, str) and name not in ("", "Unknown") else None


def _experiment_id(folder: Optional[str]) -> Optional[str]:
    """The open experiment's own id, from its experiment.yaml, if it reads."""
    if not folder:
        return None
    try:
        with open(os.path.join(folder, "experiment.yaml"), encoding="utf-8") as f:
            saved = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    value = saved.get("_id") if isinstance(saved, dict) else None
    return str(value) if value else None


def _source(request: Request, experiment: Optional[str], item: Optional[str]):
    """Where an entry comes from: ids and names, never a path."""
    microscope = getattr(getattr(request.app.state, "served", None), "current", None)
    try:
        serial = microscope.system.info.serial_number
    except AttributeError:
        serial = None
    agent = request.app.state.auth.agent_session or {}
    return {
        "experiment_id": _experiment_id(experiment),
        "instrument_serial": serial if serial not in ("", "Unknown") else None,
        "agent_session": agent.get("session_id"),
        "item": item,
    }


def build_knowledge_router(
    experiment_path: Callable[[], Optional[str]],
) -> APIRouter:
    router = APIRouter(
        prefix="/app/knowledge", dependencies=[Depends(require_monitoring())]
    )
    try:
        knowledge.make_folders()
    except OSError:
        pass  # a read-only state root still serves what it has

    @router.get("", response_model=KnowledgeIndex)
    def knowledge_index(
        section: Optional[str] = None,
        sample: Optional[str] = None,
        playbook: Optional[str] = None,
        status: Optional[str] = None,
        lasts: Optional[str] = None,
        resolved: bool = False,
    ):
        try:
            rows = knowledge.entries(section, sample, playbook, status, lasts, resolved)
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeIndex(rows, knowledge.samples(), knowledge.playbooks()))

    @router.get("/{section}/{id}", response_model=KnowledgeEntry)
    def knowledge_entry(
        section: str,
        id: str,
        sample: Optional[str] = None,
        playbook: Optional[str] = None,
        pictures: bool = False,
    ):
        try:
            entry = knowledge.read(section, id, sample, playbook)
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
    def knowledge_image(
        section: str,
        id: str,
        name: str,
        sample: Optional[str] = None,
        playbook: Optional[str] = None,
    ):
        try:
            path = knowledge.image(section, id, name, sample, playbook)
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
        playbook = _optional_text(body, "playbook")
        lasts = _optional_text(body, "lasts")
        remind = body.get("remind")
        if remind is not None and not isinstance(remind, str):
            raise _invalid("remind is a string.")
        entry_id = _optional_text(body, "id")
        item = _optional_text(body, "item")
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
                playbook=playbook,
                id=entry_id,
                lasts=lasts,
                remind=remind,
                instrument=_instrument(request),
                session=name,
                source=_source(request, experiment, item),
                images=sources,
                image_names=images,
            )
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeEntry(entry))

    @router.post(
        "/{section}/{id}/keep",
        response_model=KnowledgeEntry,
        dependencies=[Depends(require_operator())],
    )
    def knowledge_keep(
        section: str,
        id: str,
        sample: Optional[str] = None,
        playbook: Optional[str] = None,
    ):
        try:
            entry = knowledge.keep(section, id, sample, playbook)
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeEntry(entry))

    @router.post(
        "/{section}/{id}/resolve",
        response_model=KnowledgeEntry,
        dependencies=[Depends(require_operator())],
    )
    def knowledge_resolve(
        section: str,
        id: str,
        sample: Optional[str] = None,
        playbook: Optional[str] = None,
    ):
        try:
            entry = knowledge.resolve(section, id, sample, playbook)
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeEntry(entry))

    @router.delete(
        "/{section}/{id}",
        response_model=KnowledgeRemoved,
        dependencies=[Depends(require_operator())],
    )
    def knowledge_delete(
        section: str,
        id: str,
        sample: Optional[str] = None,
        playbook: Optional[str] = None,
    ):
        try:
            entry = knowledge.delete(section, id, sample, playbook)
        except knowledge.KnowledgeError as error:
            raise _refused(error)
        return answer(KnowledgeRemoved(True, entry))

    return router
