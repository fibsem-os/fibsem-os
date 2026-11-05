"""The operator's feedback on an agent's picture, kept as an example in the
knowledge base: what the agent picked, what was said of it, and what for.

A kept example is a sample entry (``knowledge.write``) holding its own copy of
the picture with the marks drawn, so it reads after the experiment is moved or
gone. The agent's picks are drawn in its purple, the operator's marks in blue,
each with a tick or a cross when one was said; a moved pick has an arrow to
where it should be. Where it came from is metadata, never a path.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fibsem import knowledge
from fibsem.geometry.saved_images import image_pixel_size
from fibsem.imaging.marks import MARK_COLOURS, draw, png_bytes
from fibsem.workflows.evidence import BAD, GOOD, KINDS, marked_on

# A kept example's own copy of its image at full resolution, up to this size.
MAX_SOURCE = 64 * 1024 * 1024

UNRATED = "unrated"

_SAID = {GOOD: "good", BAD: "bad"}


def said(record: Dict[str, Any]) -> bool:
    """Whether *record* says good or bad of anything: the picture, a pick or a
    mark. Read from a body as sent too, so anything else is nothing said."""

    def any_verdict(marks: Any) -> bool:
        return isinstance(marks, list) and any(
            isinstance(m, dict) and m.get("verdict") for m in marks
        )

    marks = record.get("marks")
    return bool(
        record.get("verdict")
        or any_verdict(record.get("judged"))
        or (isinstance(marks, dict) and any(any_verdict(marks.get(k)) for k in KINDS))
    )


def _name(kind: str, index: int, mark: Dict[str, Any]) -> str:
    label = mark.get("label")
    return f'"{label}"' if label else f"{kind.rstrip('s')} {index + 1}"


def _marks_drawn(record: Dict[str, Any], picks: Dict[str, Any]) -> Dict[str, Any]:
    """The operator's marks and the agent's picks on one picture: each pick in
    the agent's colour with what was said of it, a moved one with an arrow."""
    drawn: Dict[str, List[Dict[str, Any]]] = {}
    agent = MARK_COLOURS["agent"]
    judged = {(j["kind"], j["index"]): j for j in record.get("judged") or []}
    for kind in KINDS:
        for index, mark in enumerate(picks.get(kind) or []):
            one = {**mark, "colour": agent}
            one.pop("verdict", None)
            said = judged.get((kind, index))
            if said and said.get("verdict"):
                one["verdict"] = said["verdict"]
            drawn.setdefault(kind, []).append(one)
            moved = (said or {}).get("moved_to")
            if moved:
                x, y = said["at"]["x"], said["at"]["y"]
                drawn.setdefault("lines", []).append(
                    {
                        "x1": x,
                        "y1": y,
                        "x2": moved["x"],
                        "y2": moved["y"],
                        "arrow": True,
                    }
                )
    for kind in KINDS:
        drawn.setdefault(kind, []).extend((record.get("marks") or {}).get(kind) or [])
    return {kind: marks for kind, marks in drawn.items() if marks}


def _text(
    record: Dict[str, Any], picks: Dict[str, Any], message: str, task: Optional[str]
) -> str:
    lines = []
    if task:
        lines.append(f"Task: {task}")
    if message:
        lines.append(f"Operator: {message}")
    if record.get("verdict"):
        lines.append(f"The picture: {_SAID[record['verdict']]}")
    judged = record.get("judged") or []
    if judged:
        lines.append("")
        lines.append("The agent's picks:")
        for j in judged:
            mark = (picks.get(j["kind"]) or [])[j["index"]]
            words = [_SAID[j["verdict"]]] if j.get("verdict") else []
            if j.get("moved_to"):
                far = j.get("moved_m")
                words.append(
                    f"moved {far * 1e6:.1f} µm"
                    if far
                    else f"moved {j.get('moved_px', 0):.0f} px"
                )
            lines.append(f"- {_name(j['kind'], j['index'], mark)}: {', '.join(words)}")
    own = record.get("marks") or {}
    added = [
        (kind, index, mark)
        for kind in KINDS
        for index, mark in enumerate(own.get(kind) or [])
    ]
    if added:
        lines.append("")
        lines.append("The operator's marks:")
        for kind, index, mark in added:
            verdict = f": {_SAID[mark['verdict']]}" if mark.get("verdict") else ""
            lines.append(f"- {kind.rstrip('s')} {_name(kind, index, mark)}{verdict}")
    return "\n".join(lines)


def _title(record: Dict[str, Any], message: str) -> str:
    judged = record.get("judged") or []
    good = sum(1 for j in judged if j.get("verdict") == GOOD)
    bad = sum(1 for j in judged if j.get("verdict") == BAD)
    parts = []
    if record.get("verdict"):
        parts.append(f"picture {_SAID[record['verdict']]}")
    if good or bad:
        parts.append(f"{good} good, {bad} bad of the agent's picks")
    head = (
        "Feedback: " + "; ".join(parts) if parts else "Feedback on the agent's picture"
    )
    first = " ".join(message.split())
    title = f"{head}. {first}" if first else head
    return (
        title
        if len(title) <= knowledge.MAX_TITLE
        else title[: knowledge.MAX_TITLE - 1] + "…"
    )


def keep(
    record: Dict[str, Any],
    answered: Optional[Dict[str, Any]],
    sample: str,
    message: str = "",
    task: Optional[str] = None,
    root: Optional[str] = None,
    instrument: Optional[str] = None,
    source: Optional[Dict[str, Optional[str]]] = None,
) -> knowledge.Entry:
    """Write the operator's feedback *record* (an evidence record, with the
    picture it *answered*) as a kept example of *sample*: the image drawn with
    the agent's picks and the operator's marks, the *message*, the *task* the
    picture was for. *root*, the experiment, names the image and the entry's
    session tag; *instrument* and *source* as `knowledge.write` takes them.
    The image itself is copied in too, when not too big."""
    image = record.get("image")
    picks = (marked_on(answered)[1] if answered else None) or {}
    marks = _marks_drawn(record, picks)
    rgb = draw(
        image, marks, colour=MARK_COLOURS["operator"], channels=record.get("channels")
    ).rgb
    name = os.path.relpath(image, root) if root else os.path.basename(image)
    with tempfile.TemporaryDirectory() as folder:
        picture = os.path.join(folder, os.path.basename(image).split(".")[0] + ".png")
        with open(picture, "wb") as f:
            f.write(png_bytes(rgb))
        entry = knowledge.write(
            knowledge.SAMPLE,
            _title(record, message),
            _text(record, picks, message, task),
            knowledge.OPERATOR,
            sample=sample,
            instrument=instrument,
            session=os.path.basename(os.path.normpath(root)) if root else None,
            source={**(source or {}), "evidence_id": record.get("evidence_id")},
            images=[picture],
            image_names=[name.replace(os.sep, "/")],
        )
    copied = _copy_source(entry, image)
    _save_marks(entry, record, picks, name.replace(os.sep, "/"), task, copied)
    return entry


def _copy_source(entry: knowledge.Entry, image: str) -> Optional[str]:
    """The image at full resolution beside the entry (``source.<ext>``), as
    the agent saw it, for an eval to run on; None when over `MAX_SOURCE`."""
    try:
        if os.path.getsize(image) > MAX_SOURCE:
            return None
        lower = image.lower()
        stack = lower.endswith((".ome.tif", ".ome.tiff"))
        name = "source" + (".ome" if stack else "") + Path(lower).suffix
        shutil.copy2(image, os.path.join(os.path.dirname(entry.path), name))
    except OSError:
        return None
    return name


def _image_facts(image: str) -> Dict[str, Any]:
    """The image's size in pixels and its pixel size, when it says."""
    facts: Dict[str, Any] = {}
    try:
        if image.lower().endswith((".tif", ".tiff")):
            import tifffile

            with tifffile.TiffFile(image) as tif:
                shape = tif.pages[0].shape
        else:
            from PIL import Image

            with Image.open(image) as opened:
                shape = (opened.height, opened.width)
        facts["width"], facts["height"] = int(shape[1]), int(shape[0])
    except Exception:  # noqa: BLE001 - an unread size is left out
        pass
    try:
        size = image_pixel_size(image)
    except Exception:  # noqa: BLE001 - an image that doesn't say is unmeasured
        size = None
    if size:
        facts["pixel_size"] = size
    return facts


def _save_marks(
    entry: knowledge.Entry,
    record: Dict[str, Any],
    picks: Dict[str, Any],
    name: str,
    task: Optional[str],
    copied: Optional[str],
) -> None:
    """``marks.json`` beside the entry: the example as data, for a scorer. The
    agent's picks in the image's pixels with what was said of each and where a
    moved one should be, the operator's marks, and the image's size."""
    judged = {(j["kind"], j["index"]): j for j in record.get("judged") or []}
    rows = []
    for kind in KINDS:
        for index, mark in enumerate(picks.get(kind) or []):
            said = judged.get((kind, index), {})
            # every pick starts unrated; rating them all is never asked for
            row = {"kind": kind, "index": index, "mark": mark, "verdict": UNRATED}
            for key in ("verdict", "moved_to", "moved_px", "moved_m"):
                if said.get(key) is not None:
                    row[key] = said[key]
            rows.append(row)
    data = {
        "image": name,
        # the image as the agent saw it, copied beside this file; None when too big
        "source_file": copied,
        **_image_facts(record.get("image")),
        "task": task,
        "verdict": record.get("verdict"),
        "picks": rows,
        "marks": record.get("marks") or {},
        "channels": record.get("channels"),
    }
    path = os.path.join(os.path.dirname(entry.path), "marks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
