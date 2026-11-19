"""Polishing stop cases cut from a real run (``evals from-run``)."""

from typing import Any, Dict, List

BRIEF = (
    "This lamella ({item}) was polished in the passes listed below{angle}; "
    "{target} After each pass there are SEM and FIB images, and a measured "
    "thickness where one was taken (which can be off). Say for each pass "
    "whether you would carry on or stop there: stop once the lamella is thin "
    "enough, or before it gets damaged (a ragged or eroding front face, a "
    "crack, a break). Mark only one pass stop."
)


def _measured(entry: Dict[str, Any]) -> Dict[str, Any]:
    value = entry.get("measured") or entry.get("thickness_nm")
    if isinstance(value, dict):
        return value
    if value is not None:
        return {"median_nm": value}
    return {}


def cut(segment: Dict[str, Any], inputs: Dict[str, Any], folder=None) -> Dict[str, Any]:
    facts = segment.get("facts") or {}
    passes: List[Dict[str, Any]] = []
    for n, entry in enumerate(inputs.get("passes") or [], start=1):
        one: Dict[str, Any] = {"name": str(entry.get("name") or f"P{n:02d}")}
        for beam in ("sem", "fib"):
            if entry.get(beam):
                one[beam] = entry[beam]
        measured = _measured(entry)
        if measured:
            one["measured"] = measured
        passes.append(one)
    angle = facts.get("milling_angle_deg")
    target = facts.get("target_nm")
    brief = BRIEF.format(
        item=segment.get("item") or "one lamella",
        angle=f" on a {angle:g} degree milling angle" if angle else "",
        target=f"the target thickness is {target:g} nm."
        if target
        else "the target thickness is the protocol's.",
    )
    return {"brief": brief, "passes": passes}
