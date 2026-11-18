"""The scripted baseline for live polishing: the item's own Polishing task
pass by pass down a fixed schedule, stopping at the target or a broken band."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from fibsem.ai.evals.sim.live_client import LiveClient
from fibsem.ai.evals.sim.scripted import (
    _image_path,
)

log = logging.getLogger(__name__)

POLISH_ITEM = "Lam-01"
POLISH_IMAGE_HFW = 30e-6  # m, the FIB image a pass is judged on
# each pass's gap between the trenches (m), and where to stop (nm measured:
# a FIB band reads a few tens of nm wide of the true gap)
POLISH_SCHEDULE = (1.0e-6, 0.6e-6, 0.4e-6, 0.3e-6, 0.25e-6, 0.22e-6, 0.2e-6)
POLISH_STOP_NM = 250
FULL_FRAME_BELOW = 80e-12  # A: passes under this align on the full frame
BROKEN_SHARE = 0.2  # a band missing in this share of its columns is broken
WIDER = 1.3  # a band this much wider than before a pass is broken
# warnings that make a reading no use to stop on or to compare with
UNSURE = ("several_bands", "low_contrast")


def _measure(client: LiveClient, item: str, n: int) -> Dict[str, Any]:
    """A FIB image of the lamella, saved in its folder, and the band's
    thickness on it: median nm, and whether the band looks broken."""
    shot = client.call(
        "beam_acquire_image",
        beam_type="ION",
        image_settings={"hfw": POLISH_IMAGE_HFW, "resolution": [1536, 1024]},
        path=item,
        filename=f"P{n:02d}",
    )
    path = _image_path(shot)
    found = client.call(
        "item_measure_thickness", item_name=item, path=path[path.index(f"{item}/") :]
    )
    continuity = found.get("continuity") or {}
    dropped = continuity.get("dropped", 0) / max(1, continuity.get("columns", 1))
    codes = [w.get("code") for w in found.get("warnings") or []]
    median = (found.get("thickness_nm") or {}).get("median")
    log.info("P%02d: %s nm, dropped %.0f%%, %s", n, median, dropped * 100, codes)
    return {
        "nm": median,
        "broken": dropped > BROKEN_SHARE or "no_band" in codes,
        "sure": median is not None
        and dropped <= BROKEN_SHARE
        and not any(c in codes for c in UNSURE),
    }


def polishing(client: LiveClient, work: Path) -> Dict[str, Any]:
    """Polish the item pass by pass with its own Polishing task, aligning
    before each pass on the full frame (the passes are at a low current),
    narrowing the gap down a fixed schedule; stop when the band measures at
    the target, or looks broken."""
    item = POLISH_ITEM
    config = client.call("item_get_task_config", item_name=item, task_name="Polishing")
    task = config["config"]["milling"]["mill_polishing"]
    task["alignment"]["full_frame_below"] = FULL_FRAME_BELOW
    # the reading as rough milled is often wrong, with or without a warning
    # (the trenches' far walls read as bands): it is kept for the record, but
    # neither stops the polishing nor is compared with
    measured = _measure(client, item, 0)
    sure = None
    passes, damage = 0, False
    for spacing in POLISH_SCHEDULE:
        if sure is not None and sure["nm"] <= POLISH_STOP_NM:
            break
        this = json.loads(json.dumps(task))
        this["stages"][0]["pattern"]["spacing"] = spacing
        client.call("milling_run_task", config=this, path=item)
        passes += 1
        measured = _measure(client, item, passes)
        # a band that reads much wider than the last sure reading has come apart
        wider = (
            sure is not None
            and measured["nm"] is not None
            and measured["nm"] > sure["nm"] * WIDER
        )
        if measured["broken"] or wider:
            damage = True
            break
        if measured["sure"]:
            sure = measured
    return {
        "item": item,
        "thickness_nm": round(
            (measured if damage or sure is None else sure)["nm"] or 0
        ),
        "passes": passes,
        "damage_seen": damage,
        "reason": (
            "the band looked broken after the last pass"
            if damage
            else f"the band measured {measured['nm']} nm"
        ),
    }
