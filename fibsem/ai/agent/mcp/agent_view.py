"""An image tool's answer as the agent reads it, in the sidecar: one path per
image, relative to the open experiment; floats rounded, never coarser than a
nanometre; each fact of an image answer said once; a drawing's computed facts
without the marks it was given. The previews, and the App Server's answer that
the web and Qt views and the Python client read, are untouched."""

import math
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

SIGNIFICANT = 5
# A metre value below 1 keeps at least nanometres: a stage position's resolution.
NM_DECIMALS = 9

_WINDOWS = re.compile(r"^[A-Za-z]:[\\/]")


def rounded(value: float) -> float:
    """*value* to `SIGNIFICANT` figures, at least `NM_DECIMALS` decimals below
    1, and never past the units (a POSIX time keeps its seconds)."""
    if value == 0 or not math.isfinite(value):
        return value
    digits = SIGNIFICANT - 1 - math.floor(math.log10(abs(value)))
    if abs(value) < 1:
        digits = max(digits, NM_DECIMALS)
    return round(value, max(digits, 0))


def _absolute(text: str) -> bool:
    return text.startswith(("/", "\\\\")) or bool(_WINDOWS.match(text))


def _folded(root: str) -> bool:
    """Whether paths under *root* compare without case: a Windows path."""
    return root.startswith("\\\\") or bool(_WINDOWS.match(root))


def relative(text: str, root: Optional[str]) -> str:
    """*text*, a path inside the folder *root*, relative to it with ``/``
    between its parts; anything else as it is. *root* may be a Windows path
    read on another system."""
    if not root or not _absolute(text):
        return text
    base = root.replace("\\", "/").rstrip("/") + "/"
    path = text.replace("\\", "/")
    if _folded(root):
        inside = path.lower().startswith(base.lower())
    else:
        inside = path.startswith(base)
    return path[len(base) :] if inside else text


def absolute_paths(node: Any) -> Iterator[str]:
    """Every string in *node* that is an absolute path."""
    if isinstance(node, str):
        if _absolute(node):
            yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from absolute_paths(value)
    elif isinstance(node, list):
        for value in node:
            yield from absolute_paths(value)


def outside(node: Any, root: Optional[str]) -> bool:
    """Whether *node* names an absolute path that is not inside *root*."""
    return any(relative(p, root) == p for p in absolute_paths(node))


def _given(block: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in block.items() if v is not None}


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path)[-1]


def _preview(preview: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """*preview* without what *summary* already says: the source's size, the
    beam, the field width and pixel size when they are the image's."""
    preview = _given(preview)
    full = [preview.get("full_height"), preview.get("full_width")]
    region = preview.get("region")
    shape = summary.get("shape")
    if (isinstance(shape, list) and full == shape[:2]) or (
        isinstance(region, dict) and full == [region.get("height"), region.get("width")]
    ):
        preview.pop("full_height", None)
        preview.pop("full_width", None)
    for key in ("beam_type", "hfw"):
        if key in preview and _same(preview[key], summary.get(key)):
            del preview[key]
    size = preview.get("pixelsize")
    if isinstance(size, dict) and _same(size.get("x"), size.get("y")):
        if _same(size.get("x"), summary.get("pixel_size")):
            del preview["pixelsize"]
    return preview


def _same(a: Any, b: Any) -> bool:
    """Whether *a* and *b* read the same once rounded."""
    if isinstance(a, float) and isinstance(b, float):
        return rounded(a) == rounded(b)
    return a == b


def _crops(
    crops: List[Any], summary: Dict[str, Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    """Each crop with its own preview facts, and what all their previews share."""
    previews = [
        _preview(c["preview"], summary)
        if isinstance(c, dict) and isinstance(c.get("preview"), dict)
        else None
        for c in crops
    ]
    if not previews or any(p is None for p in previews):
        return crops, {}
    first = previews[0]
    shared = {
        key: value
        for key, value in first.items()
        if key not in ("region", "image_index")
        and all(key in p and p[key] == value for p in previews)
    }
    return [
        {**crop, "preview": {k: v for k, v in p.items() if k not in shared}}
        for crop, p in zip(crops, previews)
    ], shared


def _image_answer(answer: Dict[str, Any]) -> Dict[str, Any]:
    """An image answer (`fibsem.server.images.image_answer`, or a file read's)
    with each fact once: no nulls in the summary, the preview without what the
    summary says, what every crop shares once in ``crop_preview``. The bar's
    words (``info``) are the summary's again."""
    answer = dict(answer)
    summary = _given(answer["summary"])
    if summary.get("kind") == "image":
        del summary["kind"]
    path = answer.get("path")
    if summary.get("path") == path:
        summary.pop("path", None)
    if path is None:
        answer.pop("path", None)
    elif answer.get("name") == _basename(path):
        del answer["name"]
    if "acquired_at" in summary:
        # A file read's modified time: the image says when it was taken.
        answer.pop("modified", None)
    answer.pop("info", None)
    answer["summary"] = summary
    if isinstance(answer.get("preview"), dict):
        answer["preview"] = _preview(answer["preview"], summary)
    if isinstance(answer.get("crops"), list):
        answer["crops"], shared = _crops(answer["crops"], summary)
        if shared:
            answer["crop_preview"] = shared
    return answer


def _image_answers(node: Any) -> Any:
    """*node* with every image answer in it said once (`_image_answer`)."""
    if isinstance(node, dict):
        if isinstance(node.get("summary"), dict) and (
            "preview" in node or "crops" in node
        ):
            node = _image_answer(node)
        return {key: _image_answers(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_image_answers(value) for value in node]
    return node


_STAGES = ("stage", "stage1", "stage2")
# Where an unlabelled mark is, to tell it by: an outline's corners are not kept.
_ANCHORS = ("x", "y", "x1", "y1", "x2", "y2", "width", "height", "radius")


def _mark(mark: Any) -> Any:
    """What a drawing computed of *mark*: its label (a measured line's
    length), plane and stage positions."""
    if not isinstance(mark, dict):
        return mark
    kept = {k: mark[k] for k in ("label", "z") if mark.get(k) not in (None, "")}
    if "label" not in kept:
        kept.update({k: mark[k] for k in _ANCHORS if k in mark})
    kept.update({k: mark[k] for k in _STAGES if k in mark})
    return kept


def _drawn(answer: Any) -> Any:
    """A drawing's answer (`ImageDrawn`, `ImagePairDrawn`) without the images
    and marks it was given: the picture's path, each mark's computed facts and
    the layers."""
    if not isinstance(answer, dict):
        return answer
    answer = {k: v for k, v in answer.items() if k not in ("image", "before", "after")}
    marks = answer.get("marks")
    if marks is None:
        answer.pop("marks", None)
    elif isinstance(marks, dict):
        answer["marks"] = {
            kind: [_mark(m) for m in items] if isinstance(items, list) else items
            for kind, items in marks.items()
        }
    return answer


def _output(answer: Any) -> Any:
    """A saved output image's answer: its metadata block without nulls."""
    if isinstance(answer, dict) and isinstance(answer.get("metadata"), dict):
        return {**answer, "metadata": _given(answer["metadata"])}
    return answer


# Each image tool's own cut, before the cuts every image tool gets.
TOOL_VIEWS: Dict[str, Callable[[Any], Any]] = {
    "analysis_draw_image": _drawn,
    "analysis_draw_pair": _drawn,
    "grid_get_output_image": _output,
    "item_get_output_image": _output,
}

# The tools whose answers the agent reads through `view`: those that answer an
# image. Others keep their answer as the route gives it (a proposal's values,
# passed back in a decision, are not rounded).
VIEWED = frozenset(
    {
        "beam_acquire_image",
        "fm_acquire_image",
        "device_call_command",
        "service_wait_for_run",
        "experiment_read_file",
        "analysis_crop_image",
        "experiment_render_markers",
        "item_draw_milling",
        *TOOL_VIEWS,
    }
)


def _plain(node: Any, root: Optional[str]) -> Any:
    if isinstance(node, float):
        return rounded(node)
    if isinstance(node, str):
        return relative(node, root)
    if isinstance(node, dict):
        return {key: _plain(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [_plain(value, root) for value in node]
    return node


def view(data: Any, tool: Optional[str], root: Optional[str] = None) -> Any:
    """*data*, *tool*'s answer, as the agent reads it; *root* is the open
    experiment's folder (None: paths stay as they are). A tool not in
    `VIEWED` keeps its answer."""
    if tool not in VIEWED:
        return data
    data = TOOL_VIEWS.get(tool, lambda answer: answer)(data)
    return _plain(_image_answers(data), root)


def dumps(data: Any) -> str:
    """*data* as the compact JSON text a viewed answer travels as."""
    import json

    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
