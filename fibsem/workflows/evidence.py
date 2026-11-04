"""Evidence: the image someone judged from, with marks on it, kept in the open
experiment. The words for it are the reason of what cites it.

An agent's acting call, request, suggestion, decision or note may cite one by
id, and so may an operator's message to the agent (``by`` says whose it is). The image is either one inside the experiment or one the agent made (an
overlay its own code drew), uploaded and saved under ``<experiment>/evidence/``
so it is kept with the run. Paths inside the experiment are kept relative to
it, so a moved experiment still finds them.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
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
MAX_LABEL = 32
# Who marked the image: the agent, or the operator in a message to it.
AGENT, OPERATOR = "agent", "operator"
# A point's mark: a cross, or a ring crosshair.
POINT_SHAPES = ("cross", "target")
# What someone says of a mark, or of a whole picture: right or wrong.
GOOD, BAD = "good", "bad"
VERDICTS = (GOOD, BAD)
# The kinds of marks, in the order they are listed.
KINDS = ("points", "areas", "circles", "lines", "outlines")
# An outline's corners: at least a triangle, at most this many.
MAX_CORNERS = 64

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


class _LongLabel(ValueError):
    pass


def _label(mark: Dict[str, Any]) -> str:
    label = mark.get("label", "")
    if not isinstance(label, str):
        raise ValueError
    label = label.strip()
    if len(label) > MAX_LABEL:
        raise _LongLabel(label)
    return label


def _verdict(mark: Dict[str, Any]) -> Dict[str, str]:
    """A mark's ``{"verdict": good | bad}``, or nothing when it has none."""
    verdict = mark.get("verdict")
    if verdict is None:
        return {}
    if verdict not in VERDICTS:
        raise ValueError
    return {"verdict": verdict}


def _corners(outline: Dict[str, Any]) -> List[List[float]]:
    corners = [[_number(x), _number(y)] for x, y in outline["points"]]
    if not 3 <= len(corners) <= MAX_CORNERS:
        raise ValueError
    return corners


def centroid(corners: List[List[float]]) -> tuple:
    """The centre of an outline's *corners*, the mean of its corners."""
    return (
        sum(x for x, _ in corners) / len(corners),
        sum(y for _, y in corners) / len(corners),
    )


def _flag(mark: Dict[str, Any], name: str) -> bool:
    value = mark.get(name, False)
    if not isinstance(value, bool):
        raise ValueError
    return value


_SHAPE = (
    "marks is {points: [{x, y, shape?, label?}], areas: [{x, y, width, height, "
    "label?}], circles: [{x, y, radius, label?}], lines: [{x1, y1, x2, y2, "
    "arrow?, measure?, label?}], outlines: [{points: [[x, y], ...], label?}], "
    "crop?}, in the image's full-resolution pixels from "
    "its top left, or as read off a crop's picture when crop names it (a cell, B3, "
    "or the box [x, y, width, height]); shape is cross or target; an outline has "
    f"3 to {MAX_CORNERS} corners. Any mark may take verdict, good or bad. On a "
    "fluorescence stack any mark may take z, the plane it is on (0 the first, as "
    "the correlation tools count): a picture of the stack then shows that plane."
)


def _marks(marks: Any) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """*marks* as ``{points, areas, circles, lines}`` (`_SHAPE`) in the
    image's pixels (origin top left), only the kinds given; None for none."""
    if marks is None:
        return None
    try:
        if not isinstance(marks, dict) or set(marks) - set(KINDS):
            raise ValueError
        checked = {
            "points": [
                {
                    "x": _number(p["x"]),
                    "y": _number(p["y"]),
                    "shape": _shape(p),
                    "label": _label(p),
                    **_verdict(p),
                    **_z(p),
                }
                for p in marks.get("points") or []
            ],
            "areas": [
                {
                    "x": _number(a["x"]),
                    "y": _number(a["y"]),
                    "width": _number(a["width"]),
                    "height": _number(a["height"]),
                    "label": _label(a),
                    **_verdict(a),
                    **_z(a),
                }
                for a in marks.get("areas") or []
            ],
            "circles": [
                {
                    "x": _number(c["x"]),
                    "y": _number(c["y"]),
                    "radius": _number(c["radius"]),
                    "label": _label(c),
                    **_verdict(c),
                    **_z(c),
                }
                for c in marks.get("circles") or []
            ],
            "lines": [
                {
                    "x1": _number(line["x1"]),
                    "y1": _number(line["y1"]),
                    "x2": _number(line["x2"]),
                    "y2": _number(line["y2"]),
                    "arrow": _flag(line, "arrow"),
                    "measure": _flag(line, "measure"),
                    "label": _label(line),
                    **_verdict(line),
                    **_z(line),
                }
                for line in marks.get("lines") or []
            ],
            "outlines": [
                {
                    "points": _corners(o),
                    "label": _label(o),
                    **_verdict(o),
                    **_z(o),
                }
                for o in marks.get("outlines") or []
            ],
        }
    except _LongLabel as long:
        raise EvidenceRefused(
            422,
            "invalid",
            f"A mark's label is at most {MAX_LABEL} characters: shorten {str(long)!r}.",
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise EvidenceRefused(422, "invalid", _SHAPE)
    if sum(len(kind) for kind in checked.values()) > MAX_MARKS:
        raise EvidenceRefused(422, "invalid", f"At most {MAX_MARKS} marks.")
    checked = {name: kind for name, kind in checked.items() if kind}
    return checked or None


_NOT_A_STACK = "z is for marks on a fluorescence stack's planes."


def _z(mark: Dict[str, Any]) -> Dict[str, float]:
    """A mark's plane of a stack, ``{"z": n}``, or nothing when it has none: a
    fraction (a fitted z, as correlation_refine_point answers it) is kept, and
    drawn on the nearest plane."""
    z = mark.get("z")
    if z is None:
        return {}
    if _number(z) < 0:
        raise ValueError
    return {"z": z}


def _on_planes(
    marks: Optional[Dict[str, List[Dict[str, Any]]]], image_path: str
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """*marks*, refused when one names a plane the image doesn't have: z is
    for a fluorescence stack's planes."""
    planes = [m["z"] for kind in (marks or {}).values() for m in kind if "z" in m]
    if not planes:
        return marks
    from fibsem.fm.preview import is_fluorescence_image
    from fibsem.fm.stack_file import StackFile

    count = 0
    if is_fluorescence_image(image_path):
        with StackFile(image_path) as stack:
            count = stack.planes
    if not count:
        raise EvidenceRefused(422, "invalid", _NOT_A_STACK)
    if round(max(planes)) >= count:
        raise EvidenceRefused(
            422,
            "invalid",
            f"z is a plane of the stack, 0 to {count - 1}: not {max(planes):g}.",
        )
    return marks


def _shape(point: Dict[str, Any]) -> str:
    shape = point.get("shape", POINT_SHAPES[0])
    if shape not in POINT_SHAPES:
        raise ValueError
    return shape


def _channels(channels: Any) -> Optional[List[str]]:
    if channels is None:
        return None
    if not isinstance(channels, list) or not all(
        isinstance(name, str) and name for name in channels
    ):
        raise EvidenceRefused(
            422, "invalid", "channels is a list of the channel names shown."
        )
    return list(channels)


def _with_lengths(
    marks: Optional[Dict[str, List[Dict[str, Any]]]], image_path: str
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """*marks* with each measured line labelled with its length, from the
    image's pixel size, as the ruler reads it; unlabelled when the image
    doesn't say its pixel size."""
    measured = [line for line in (marks or {}).get("lines", []) if line["measure"]]
    if not measured:
        return marks
    from fibsem.geometry.saved_images import image_pixel_size
    from fibsem.imaging.export import format_si

    try:
        size = image_pixel_size(image_path)
    except Exception:  # noqa: BLE001 - an image that doesn't read stays unmeasured
        size = None
    for line in measured:
        if size:
            pixels = math.hypot(line["x2"] - line["x1"], line["y2"] - line["y1"])
            line["label"] = format_si(pixels * size, "m")
        else:
            line["label"] = ""
    return marks


def _with_stage(
    marks: Optional[Dict[str, List[Dict[str, Any]]]], image_path: str
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """*marks* with the stage position (metres) under each place they name,
    from the image's own metadata (`image_points_to_stage`): a point, an
    area's and a circle's centre, a line's two ends. Left as they are when the
    image has no stage geometry."""
    from fibsem.geometry.saved_images import image_points_to_stage

    if not marks:
        return marks
    places = []
    for point in marks.get("points", []):
        places.append((point, "stage", point["x"], point["y"]))
    for area in marks.get("areas", []):
        centre = (area["x"] + area["width"] / 2, area["y"] + area["height"] / 2)
        places.append((area, "stage", *centre))
    for circle in marks.get("circles", []):
        places.append((circle, "stage", circle["x"], circle["y"]))
    for line in marks.get("lines", []):
        places.append((line, "stage1", line["x1"], line["y1"]))
        places.append((line, "stage2", line["x2"], line["y2"]))
    for outline in marks.get("outlines", []):
        places.append((outline, "stage", *centroid(outline["points"])))
    try:
        positions = image_points_to_stage(image_path, [(x, y) for _, _, x, y in places])
    except Exception:  # noqa: BLE001 - an image that doesn't read keeps pixels only
        return marks
    for (mark, key, _, _), position in zip(places, positions):
        if position is not None:
            mark[key] = {"x": float(position.x), "y": float(position.y)}
    return marks


def _uncropped(marks: Any) -> Any:
    """*marks* without the crop they may be given in."""
    if isinstance(marks, dict) and "crop" in marks:
        return {key: value for key, value in marks.items() if key != "crop"}
    return marks


def _moved(
    marks: Optional[Dict[str, List[Dict[str, Any]]]], crop: Any, image_path: str
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """*marks* read off the picture of *crop* (a cell's name or a box,
    `fibsem.imaging.cells.from_crop`) moved into the image's pixels; as they
    are when no crop is given."""
    if crop is None:
        return marks
    from fibsem.imaging.cells import from_crop
    from fibsem.imaging.marks import scaled_marks

    try:
        scale, x0, y0, _ = from_crop(image_path, crop)
    except ValueError as exc:
        raise EvidenceRefused(422, "invalid", str(exc))
    return scaled_marks(marks, scale, scale, x0, y0)


def _crop(marks: Any) -> Any:
    return marks.get("crop") if isinstance(marks, dict) else None


def checked_marks(
    marks: Any, image_path: str
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """*marks* as `_marks` reads them (refused as it refuses them), moved
    from the crop they name into the image's pixels, each measured line
    labelled with its length and each place given its stage position, from
    the image at *image_path*."""
    placed = _moved(_marks(_uncropped(marks)), _crop(marks), image_path)
    placed = _on_planes(placed, image_path)
    return _with_stage(_with_lengths(placed, image_path), image_path)


def _found(path: str, root: str, item_name: Optional[str]) -> Optional[str]:
    """The file *path* names: as given when absolute; a relative one (as
    proposal_list gives a reference image) in the item's folder, then the
    experiment's, then the one item folder that holds it."""
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
    try:
        inside = os.path.commonpath([root, target]) == root
    except ValueError:  # another drive, on Windows
        inside = False
    if not inside:
        raise EvidenceRefused(
            403,
            "not_allowed",
            f"{name}: give a path inside the open experiment ({root}), not {path}.",
        )
    return os.path.relpath(target, root).replace(os.sep, "/")


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


def _drawn_on(image_path: str, root: str) -> Dict[str, Any]:
    """For a picture the draw tools drew: ``{source, source_marks,
    source_plane}``, the image inside the experiment it was drawn on and the
    marks drawn there; empty for any other image."""
    from fibsem.imaging.marks import drawn_from

    drawn = drawn_from(image_path)
    if drawn is None:
        return {}
    try:
        source = _inside(drawn["source"], root, "source")
    except EvidenceRefused:
        return {}
    return {
        "source": source,
        "source_marks": drawn.get("marks"),
        "source_plane": drawn.get("plane"),
    }


def marked_on(record: Dict[str, Any]) -> tuple:
    """``(image, marks, plane)``: the image to answer the evidence *record* on
    (absolute), and its marks there. The image a drawn picture or an upload
    was drawn on, with the marks drawn (none known for an upload); else the
    record's own image and marks."""
    if record.get("source"):
        return (
            record["source"],
            record.get("source_marks"),
            record.get("source_plane"),
        )
    return record.get("image"), record.get("marks"), None


def _answered(
    experiment_path: Callable[[], Optional[str]], answers: Any
) -> Optional[Dict[str, Any]]:
    """The record of the picture *answers* names, or None for none."""
    if answers is None:
        return None
    try:
        return load(experiment_path, answers)
    except EvidenceRefused as refusal:
        raise EvidenceRefused(
            refusal.status_code,
            refusal.error_type,
            f"answers is the evidence id of the picture: {refusal.message}",
        )


_JUDGED = (
    "judged is [{kind, index, verdict?, moved_to?}]: a mark of the picture "
    "answered, by its kind (points, areas, circles, lines, outlines) and place in "
    "that list (0 the first), good or bad, and where it should be instead, "
    "{x, y} in the image's pixels."
)


def _place(kind: str, mark: Dict[str, Any]) -> tuple:
    """Where a mark of *kind* is, as one point in its image's pixels."""
    if kind == "areas":
        return mark["x"] + mark["width"] / 2, mark["y"] + mark["height"] / 2
    if kind == "lines":
        return mark["x1"], mark["y1"]
    if kind == "outlines":
        return centroid(mark["points"])
    return mark["x"], mark["y"]


def _judged(
    judged: Any, answered: Dict[str, Any], image_path: str, root: str
) -> List[Dict[str, Any]]:
    """*judged* checked against the marks of the picture *answered* on the
    image at *image_path*: each with the mark's label and place, and, when
    moved, how far (pixels, and metres when the image says its pixel size)
    and the stage position it was moved to."""
    from fibsem.geometry.saved_images import image_pixel_size

    on, marks, _ = marked_on(answered)
    if os.path.realpath(str(on or "")) != os.path.realpath(image_path):
        raise EvidenceRefused(
            422,
            "invalid",
            "judged marks are on the image the picture was drawn on: "
            f"{os.path.relpath(str(on), root) if on else 'none'}.",
        )
    try:
        size = image_pixel_size(image_path)
    except Exception:  # noqa: BLE001 - an image that doesn't read is unmeasured
        size = None
    out = []
    try:
        if not isinstance(judged, list) or len(judged) > MAX_MARKS:
            raise ValueError
        for entry in judged:
            kind, index = entry["kind"], entry["index"]
            if kind not in KINDS or isinstance(index, bool):
                raise ValueError
            mark = ((marks or {}).get(kind) or [])[int(index)]
            if int(index) < 0:
                raise IndexError
            given = _verdict(entry)
            moved = entry.get("moved_to")
            if not given and moved is None:
                raise ValueError
            x, y = _place(kind, mark)
            one = {"kind": kind, "index": int(index), "label": mark.get("label", "")}
            one.update(given)
            one["at"] = {"x": x, "y": y}
            if moved is not None:
                to = {"x": _number(moved["x"]), "y": _number(moved["y"])}
                pixels = math.hypot(to["x"] - x, to["y"] - y)
                one["moved_to"] = to
                one["moved_px"] = pixels
                one["moved_m"] = pixels * size if size else None
            out.append(one)
    except (IndexError, TypeError):
        raise EvidenceRefused(
            422, "invalid", f"{_JUDGED} The picture has no such mark."
        )
    except (KeyError, ValueError, AttributeError):
        raise EvidenceRefused(422, "invalid", _JUDGED)
    moved = [one for one in out if "moved_to" in one]
    if moved:
        staged = _with_stage(
            {"points": [dict(m["moved_to"]) for m in moved]}, image_path
        )
        for one, point in zip(moved, staged["points"]):
            if "stage" in point:
                one["moved_to"]["stage"] = point["stage"]
    return out


def save(
    experiment_path: Callable[[], Optional[str]],
    image: Any = None,
    upload: Any = None,
    marks: Any = None,
    source: Any = None,
    item_name: Optional[str] = None,
    by: str = AGENT,
    channels: Any = None,
    answers: Any = None,
    verdict: Any = None,
    judged: Any = None,
) -> Dict[str, Any]:
    """Keep one piece of evidence in the open experiment; answer its record.

    One of *image* (a path inside the experiment) or *upload* (base64 PNG or
    JPEG, saved into the experiment); *marks*, as `_marks` reads them, each
    given its stage position when the image says where it was taken;
    *source*, the experiment image an upload was drawn from; *by*, who marked
    it; *channels*, the fluorescence channels shown when the marks were made
    on a stack's projection.

    Feedback on a picture: *answers*, the evidence id of the picture it is
    about; *verdict* on the whole picture, good or bad; *judged*, a verdict
    on each of that picture's marks (`_judged`). With *answers* and no
    *image*, the image is the one the picture was drawn on (`marked_on`).
    A picture the draw tools drew (`fibsem.imaging.marks.drawn_from`) keeps
    the image it was drawn on as *source*, with the marks drawn there."""
    root = _root(experiment_path)
    answered = _answered(experiment_path, answers)
    if answered is not None and image is None and upload is None:
        image = marked_on(answered)[0]
    if (image is None) == (upload is None):
        raise EvidenceRefused(
            422,
            "missing_field",
            "Pass one of image (a path inside the experiment) or upload (an "
            "image you made).",
        )
    if item_name is not None and not isinstance(item_name, str):
        raise EvidenceRefused(422, "invalid", "item_name is a string.")
    if verdict is not None and verdict not in VERDICTS:
        raise EvidenceRefused(422, "invalid", "verdict is good or bad.")
    if (verdict is not None or judged is not None) and answered is None:
        raise EvidenceRefused(
            422, "invalid", "verdict and judged are about a picture: give answers."
        )
    checked_marks = _marks(_uncropped(marks))
    checked_channels = _channels(channels)
    checked_source = _inside(source, root, "source", item_name) if source else None
    drawn_on: Dict[str, Any] = {}
    evidence_id = secrets.token_hex(4)
    where = folder(root)
    if upload is not None:
        if _crop(marks) is not None:
            raise EvidenceRefused(
                422,
                "invalid",
                "A crop is for marks on an image in the experiment, not an upload.",
            )
        if any("z" in m for kind in (checked_marks or {}).values() for m in kind):
            raise EvidenceRefused(422, "invalid", _NOT_A_STACK)
        os.makedirs(where, exist_ok=True)
        _save_upload(upload, os.path.join(where, f"{evidence_id}.png"))
        path = f"{FOLDER}/{evidence_id}.png"
    else:
        path = _inside(image, root, "image", item_name)
        if checked_channels is not None:
            from fibsem.fm.preview import is_fluorescence_image

            if not is_fluorescence_image(os.path.join(root, path)):
                raise EvidenceRefused(
                    422,
                    "invalid",
                    "channels is for marks on a fluorescence stack's projection; "
                    f"{image} is not one.",
                )
        image_path = os.path.join(root, path)
        checked_marks = _moved(checked_marks, _crop(marks), image_path)
        checked_marks = _on_planes(checked_marks, image_path)
        checked_marks = _with_lengths(checked_marks, image_path)
        checked_marks = _with_stage(checked_marks, image_path)
        if checked_source is None:
            drawn_on = _drawn_on(image_path, root)
            checked_source = drawn_on.pop("source", None)
    checked_judged = (
        _judged(judged, answered, os.path.join(root, path), root)
        if judged is not None
        else None
    )
    record = {
        "evidence_id": evidence_id,
        "by": by,
        "image": path,
        "uploaded": upload is not None,
        "source": checked_source,
        "marks": checked_marks,
        "channels": checked_channels,
        "item_name": item_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **drawn_on,
    }
    if answered is not None:
        record["answers"] = answered["evidence_id"]
        record["verdict"] = verdict
        record["judged"] = checked_judged
    os.makedirs(where, exist_ok=True)
    with open(os.path.join(where, f"{evidence_id}.json"), "w") as f:
        json.dump(record, f, indent=2)
    return _absolute(record, root)


def _absolute(record: Dict[str, Any], root: str) -> Dict[str, Any]:
    """*record* with its image paths joined to the experiment at *root*."""
    record = dict(record)
    for name in ("image", "source"):
        if record.get(name):
            record[name] = os.path.normpath(os.path.join(root, record[name]))
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
