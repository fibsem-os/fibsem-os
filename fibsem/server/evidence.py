"""The agent's evidence: the image it judged from, what it saw in it, and marks
on it, kept in the open experiment.

Any acting call, request, suggested answer or note may carry one. The image is
either one the agent took or read inside the experiment, or one it made (an
overlay its own code drew), uploaded and saved here, under
``<experiment>/agent/evidence/``, so it is kept with the run like every other
image. A call names its evidence by id (`calls.EVIDENCE_HEADER`); the sidecar
saves it first and sends the id, so the agent passes it with the call it makes.

No FastAPI: the app context and the request store read it too.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fibsem.server.allowlist import image_refusal

# Where an experiment keeps the agent's evidence.
FOLDER = os.path.join("agent", "evidence")

# An uploaded image: PNG or JPEG, at most this many bytes, and saved at most
# this many pixels on its longest side (Patrick, 2026-10-16).
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_SIDE = 2048
MAX_TEXT = 2000
MAX_MARKS = 50
MAX_LABEL = 80

_ID = re.compile(r"^[0-9a-f]{8}$")


class EvidenceRefused(Exception):
    """Evidence that can't be kept: (status, error_type, message)."""

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = {"error_type": error_type, "message": message}


def folder(experiment_path: str) -> str:
    return os.path.join(experiment_path, FOLDER)


def _root(experiment_path: Callable[[], Optional[str]]) -> str:
    root = experiment_path()
    if not root:
        raise EvidenceRefused(
            409, "no_experiment", "Evidence is kept in the experiment: open one first."
        )
    return root


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    return float(value)


def _label(mark: Dict[str, Any]) -> str:
    label = mark.get("label", "")
    if not isinstance(label, str):
        raise ValueError
    return label.strip()[:MAX_LABEL]


def _marks(marks: Any) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """*marks* as ``{points: [{x, y, label}], areas: [{x, y, width, height,
    label}]}`` in the image's pixels (origin top left); None for none."""
    if marks is None:
        return None
    try:
        if not isinstance(marks, dict) or set(marks) - {"points", "areas"}:
            raise ValueError
        points = [
            {"x": _number(p["x"]), "y": _number(p["y"]), "label": _label(p)}
            for p in marks.get("points") or []
        ]
        areas = [
            {
                "x": _number(a["x"]),
                "y": _number(a["y"]),
                "width": _number(a["width"]),
                "height": _number(a["height"]),
                "label": _label(a),
            }
            for a in marks.get("areas") or []
        ]
    except (KeyError, TypeError, ValueError, AttributeError):
        raise EvidenceRefused(
            422,
            "invalid",
            "marks is {points: [{x, y, label?}], areas: [{x, y, width, height, "
            "label?}]}, in the image's pixels from its top left.",
        )
    if len(points) + len(areas) > MAX_MARKS:
        raise EvidenceRefused(422, "invalid", f"At most {MAX_MARKS} marks.")
    return {"points": points, "areas": areas} if points or areas else None


def _checked_image(path: Any, experiment_path: Callable[[], Optional[str]]) -> str:
    refusal = image_refusal(path, experiment_path)
    if refusal is not None:
        raise EvidenceRefused(*refusal)
    return os.path.realpath(str(path))


def _save_upload(data: Any, target: str) -> None:
    """Decode the uploaded *data* (base64 PNG or JPEG) and save it, re-encoded
    and at most `MAX_SIDE` on its longest side, as the PNG *target*: what is
    kept is what the server drew, never the bytes as sent."""
    from PIL import Image

    try:
        raw = base64.b64decode(str(data), validate=True)
    except (binascii.Error, ValueError):
        raise EvidenceRefused(422, "invalid", "upload is base64 PNG or JPEG bytes.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise EvidenceRefused(
            413,
            "too_large",
            f"An uploaded image is at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise ValueError(image.format)
            image.load()
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    except Exception:  # noqa: BLE001 - anything PIL can't read is refused
        raise EvidenceRefused(422, "invalid", "upload must be a PNG or JPEG image.")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(target, format="PNG")


def save(
    experiment_path: Callable[[], Optional[str]],
    text: Any,
    image: Any = None,
    upload: Any = None,
    marks: Any = None,
    source: Any = None,
    item_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Keep one piece of evidence in the open experiment; answer its record.

    One of *image* (a path inside the experiment) or *upload* (base64 PNG or
    JPEG, saved into the experiment); *text*, what the agent saw in it;
    *marks*, as `_marks` reads them; *source*, the experiment image an upload
    was drawn from."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise EvidenceRefused(
            422,
            "missing_field",
            f"Pass text (at most {MAX_TEXT} characters): what you saw in the image.",
        )
    if (image is None) == (upload is None):
        raise EvidenceRefused(
            422,
            "missing_field",
            "Pass one of image (a path inside the experiment) or upload (an "
            "image you made).",
        )
    if item_name is not None and not isinstance(item_name, str):
        raise EvidenceRefused(422, "invalid", "item_name is a string.")
    root = _root(experiment_path)
    checked_marks = _marks(marks)
    checked_source = _checked_image(source, experiment_path) if source else None
    evidence_id = secrets.token_hex(4)
    where = folder(root)
    os.makedirs(where, exist_ok=True)
    if upload is not None:
        path = os.path.join(where, f"{evidence_id}.png")
        _save_upload(upload, path)
    else:
        path = _checked_image(image, experiment_path)
    record = {
        "evidence_id": evidence_id,
        "text": text.strip(),
        "image": path,
        "uploaded": upload is not None,
        "source": checked_source,
        "marks": checked_marks,
        "item_name": item_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(where, f"{evidence_id}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record


def load(
    experiment_path: Callable[[], Optional[str]], evidence_id: Any
) -> Dict[str, Any]:
    """The record of the evidence *evidence_id* in the open experiment."""
    root = _root(experiment_path)
    if not isinstance(evidence_id, str) or not _ID.match(evidence_id):
        raise EvidenceRefused(422, "invalid", "evidence is an evidence_id.")
    try:
        with open(os.path.join(folder(root), f"{evidence_id}.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        raise EvidenceRefused(
            404,
            "not_found",
            f"No evidence {evidence_id} in the open experiment.",
        )


def frame_path(evidence: Optional[Dict[str, Any]]) -> Optional[str]:
    """The beam image *evidence* is about, to draw values on: what an upload was
    drawn from, else its own image; None for none."""
    if not evidence:
        return None
    return evidence.get("source") or (
        None if evidence.get("uploaded") else evidence.get("image")
    )
