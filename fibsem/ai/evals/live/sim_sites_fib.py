"""Live site selection from a FIB overview, at the SEM orientation
(``make-sim --task live-site-selection-fib``)."""

from pathlib import Path
from typing import Any, Dict, Optional

from fibsem.ai.evals.live import sim_sites


def write_case(
    seed: int, out: Path, overrides: Optional[Dict[str, Any]] = None, **_: Any
) -> Path:
    return sim_sites.write_case(seed, out, overrides, beam="FIB")
