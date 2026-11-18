"""The scripted baseline for live coincidence milling: the polishing
baseline, with no FM."""

from pathlib import Path
from typing import Any, Dict

from fibsem.ai.evals.sim.live_client import LiveClient
from fibsem.ai.evals.tasks.polishing.scripted import polishing


def coincidence_milling(client: LiveClient, work: Path) -> Dict[str, Any]:
    """The polishing baseline with no FM: it polishes the lamella where it
    was placed and never looks for the cilium, so it says the cilium is kept
    on trust. On a lamella placed off the cilium it should lose it."""
    answer = polishing(client, work)
    answer["cilium_kept"] = True
    answer["reason"] += "; the cilium was not checked (no FM in this baseline)"
    return answer
