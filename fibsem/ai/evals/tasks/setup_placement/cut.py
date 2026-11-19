"""Setup placement cases cut from a real run (``evals from-run``)."""

from typing import Any, Dict, Optional

BRIEF = (
    "This FIB image at the milling angle shows the site of {item}, where a "
    "lamella is to be made. Place the lamella through the target here: give "
    "its centre as x, y px on the image. The lamella must not sit on a grid "
    "bar, and its trenches must stay inside the image."
)
SIZE = " The lamella is {w:.1f} um wide and {s:.1f} um high between the trenches."


def _image_facts(path) -> Dict[str, Any]:
    """The FIB image's pixel size and size in pixels, from its metadata
    where the file carries it."""
    try:
        from fibsem.structures.image import FibsemImage

        image = FibsemImage.load(str(path))
    except Exception:  # noqa: BLE001 - any unreadable image: no facts
        return {}
    facts: Dict[str, Any] = {"fib_size_px": [image.data.shape[1], image.data.shape[0]]}
    size = getattr(getattr(image.metadata, "pixel_size", None), "x", None)
    if size:
        facts["fib_pixel_size_um"] = round(float(size) * 1e6, 4)
    return facts


def cut(segment: Dict[str, Any], inputs: Dict[str, Any], folder=None) -> Dict[str, Any]:
    facts = segment.get("facts") or {}
    item = segment.get("item") or "this lamella"
    brief = BRIEF.format(item=item)
    lamella: Optional[Dict[str, float]] = facts.get("lamella_um")
    if lamella and {"width", "spacing"} <= set(lamella):
        brief += SIZE.format(w=lamella["width"], s=lamella["spacing"])
    case_inputs: Dict[str, Any] = {"fib_image": inputs["fib_image"]}
    if folder is not None:
        case_inputs.update(_image_facts(folder / inputs["fib_image"]))
    for name in ("milling_angle_deg", "lamella_um"):
        if facts.get(name) is not None:
            case_inputs[name] = facts[name]
    return {"brief": brief, "inputs": case_inputs}
