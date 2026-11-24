"""The scripted baseline for live correlation: no correlation, the FIB
image's centre."""

from pathlib import Path
from typing import Any, Dict

from fibsem.ai.evals.sim.live_client import LiveClient
from fibsem.ai.evals.sim.scripted import _image_path

ITEM = "Lam-01"  # the lamella the case set up
CORRELATION_HFW = 100e-6  # m, the field width the burns were placed at


def correlation(client: LiveClient, work: Path) -> Dict[str, Any]:
    """A baseline with no correlation: it takes the FIB image the burns were
    placed on and answers the image's centre, where an operator placing by
    eye puts a lamella. The target is off the centre, so it should miss."""
    shot = client.call(
        "beam_acquire_image",
        beam_type="ION",
        image_settings={"hfw": CORRELATION_HFW, "resolution": [1536, 1024]},
        path=ITEM,
        filename="burns",
    )
    return {
        "item": ITEM,
        "fib_image": _image_path(shot),
        "target_px": [768.0, 512.0],
        "target_m": [0.0, 0.0],
        "poi_set": False,
        "reason": "no correlation in this baseline: the image's centre",
    }
