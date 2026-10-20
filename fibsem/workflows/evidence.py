"""Evidence: the image someone judged from, with marks on it, kept in the open
experiment. The words for it are the reason of what cites it.

An agent's acting call, request, suggestion, decision or note may cite one by
id. The image is either one inside the experiment or one the agent made (an
overlay its own code drew), uploaded and saved under ``<experiment>/evidence/``
so it is kept with the run. Paths inside the experiment are kept relative to
it, so a moved experiment still finds them.
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

FOLDER = "evidence"
# The record's kind for evidence kept.
EVIDENCE = "evidence"

# An uploaded image: PNG or JPEG, at most this many bytes, and saved at most
# this many pixels on its longest side.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_SIDE = 2048
MAX_MARKS = 50
MAX_LABEL = 80

_ID = re.compile(r"^[0-9a-f]{8}$")


class EvidenceRefused(Exception):
    """Evidence that can't be kept or found: (status, error_type, message)."""

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.detail = {"error_type": error_type, "message": message}


def folder(experiment_path: str) -> str:
    return os.path.join(experiment_path, FOLDER)


def _root(experiment_path: Callable[[], Optional[str]]) -> str:
    root = experiment_path()
    if not root:
        raise EvidenceRefused(
            409, "no_experiment", "Evidence is kept in the experiment: open one first."
        )
    return os.path.realpath(str(root))


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


def _found(path: str, root: str, item_name: Optional[str]) -> Optional[str]:
    """The file *path* names: as given when absolute; a relative one (as
    proposal_list gives a reference image) in the item's folder, then the
    experiment's, then the one item folder that holds it (FIB-1494)."""
    if os.path.isabs(path):
        return path if os.path.isfile(path) else None
    folders = [os.path.join(root, item_name)] if item_name else []
    for candidate in [os.path.join(f, path) for f in folders + [root]]:
        if os.path.isfile(candidate):
            return candidate
    if item_name:
        return None
    held = [
        candidate
        for candidate in (
            os.path.join(root, entry, path) for entry in sorted(os.listdir(root))
        )
        if os.path.isfile(candidate)
    ]
    return held[0] if len(held) == 1 else None


def _inside(path: Any, root: str, name: str, item_name: Optional[str] = None) -> str:
    """*path*, an image file inside the experiment at *root*, relative to it."""
    found = _found(path, root, item_name) if isinstance(path, str) and path else None
    if found is None:
        raise EvidenceRefused(
            422,
            "invalid",
            f"{name} is the path of an image file you took or read (as the image "
            "tools answered it), or one proposal_list gives, relative to its item's "
            f"folder (pass item_name) or the experiment's: no file {path!r}.",
        )
    path = found
    target = os.path.realpath(path)
    if os.path.commonpath([root, target]) != root:
        raise EvidenceRefused(
            403,
            "not_allowed",
            f"{name}: give a path inside the open experiment ({root}), not {path}.",
        )
    return os.path.relpath(target, root)


def _save_upload(data: Any, target: str) -> None:
    """Decode the uploaded *data* (base64 PNG or JPEG) and save it, re-encoded
    and at most `MAX_SIDE` on its longest side, as the PNG *target*: what is
    kept is what was drawn, never the bytes as sent."""
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
    image: Any = None,
    upload: Any = None,
    marks: Any = None,
    source: Any = None,
    item_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Keep one piece of evidence in the open experiment; answer its record.

    One of *image* (a path inside the experiment) or *upload* (base64 PNG or
    JPEG, saved into the experiment); *marks*, as `_marks` reads them;
    *source*, the experiment image an upload was drawn from."""
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
    checked_source = _inside(source, root, "source", item_name) if source else None
    evidence_id = secrets.token_hex(4)
    where = folder(root)
    if upload is not None:
        os.makedirs(where, exist_ok=True)
        _save_upload(upload, os.path.join(where, f"{evidence_id}.png"))
        path = os.path.join(FOLDER, f"{evidence_id}.png")
    else:
        path = _inside(image, root, "image", item_name)
    record = {
        "evidence_id": evidence_id,
        "image": path,
        "uploaded": upload is not None,
        "source": checked_source,
        "marks": checked_marks,
        "item_name": item_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(where, exist_ok=True)
    with open(os.path.join(where, f"{evidence_id}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return _absolute(record, root)


def _absolute(record: Dict[str, Any], root: str) -> Dict[str, Any]:
    """*record* with its image paths joined to the experiment at *root*."""
    record = dict(record)
    for name in ("image", "source"):
        if record.get(name):
            record[name] = os.path.join(root, record[name])
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
            return _absolute(json.load(f), root)
    except (OSError, ValueError):
        raise EvidenceRefused(
            404, "not_found", f"No evidence {evidence_id} in the open experiment."
        )


def find(experiment_path: Optional[str], evidence_id: Any) -> Optional[Dict[str, Any]]:
    """The record of *evidence_id* in the experiment at *experiment_path*, or
    None when there is none to show."""
    if not evidence_id or not experiment_path:
        return None
    try:
        return load(lambda: experiment_path, evidence_id)
    except EvidenceRefused:
        return None


def frame_path(evidence: Optional[Dict[str, Any]]) -> Optional[str]:
    """The beam image *evidence* is, to draw values on; None for none or for an
    image the agent made (uploaded, or a picture it rendered such as markers on
    an overview), which is shown as it is."""
    if not evidence or evidence.get("uploaded"):
        return None
    image = evidence.get("image")
    if not image or not str(image).lower().endswith((".tif", ".tiff")):
        return None
    return image
