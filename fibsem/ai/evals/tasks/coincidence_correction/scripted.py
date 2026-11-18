"""The scripted baseline for live coincidence correction: check and move
vertically at a site's field width, then save the site."""

import logging
from pathlib import Path
from typing import Any, Dict

from fibsem.ai.evals.sim.live_client import LiveClient
from fibsem.ai.evals.sim.scripted import (
    SITE_HFW,
    make_coincident,
)

log = logging.getLogger(__name__)

CORRECTION_TRIES = 6  # the coincidence case starts far off: more moves
CORRECTION_SITE = "Site-01"
CLAIM_M = 1e-6  # a last check this near the centre: say it is coincident


def coincidence(client: LiveClient, work: Path) -> Dict[str, Any]:
    """Make the site under the stage coincident, check it once more, and save
    it: coincident only if that last check says so."""
    make_coincident(client, tries=CORRECTION_TRIES)
    check = client.call("alignment_check_coincidence", hfw=SITE_HFW)
    found = (check.get("summary") or {}).get("metadata") or check
    dy = float(found["dy"])
    client.call("item_add", name=CORRECTION_SITE)
    # a check repeats to a few tenths of a micron: coincident within a micron
    done = abs(dy) <= CLAIM_M
    return {
        "site": CORRECTION_SITE,
        "coincident": done,
        "how": (
            f"a last coincidence check measured the FIB {dy * 1e6:.2f} um off "
            f"the SEM's point (score {float(found.get('score', 0)):.3f})"
        ),
    }
