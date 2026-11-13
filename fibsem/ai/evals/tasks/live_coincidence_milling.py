"""Live coincidence milling: on the simulator, polish a lamella rough milled
through a cell's primary cilium to the target, keeping the cilium in it
(``fibsem.ai.evals.live.sim_coincidence_milling``). Scored as live polishing
(``fibsem.ai.evals.tasks.live_polishing``), and on the run's own sample
record of whether the cilium's midpoint was milled away: losing it is a
costly miss, and so is saying it is still there when it is not.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks import live_polishing
from fibsem.ai.evals.tasks.base import EvalTask, Sim

# how close (m) a record's target midpoint must be to the key's cilium
SAME = 50e-9


def cilium_of(answer: Dict[str, Any], key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The run's record of the key's cilium, matched by its midpoint."""
    sim = (answer.get("record") or {}).get("sim") or {}
    want = np.asarray(key.get("cilium") or [], dtype=float)
    if want.size != 3:
        return None
    best = None
    for target in sim.get("targets") or []:
        d = float(np.linalg.norm(np.asarray(target["midpoint"], dtype=float) - want))
        if d <= SAME and (best is None or d < best[0]):
            best = (d, target)
    return None if best is None else best[1]


def score_live_coincidence_milling(
    case: Case, answer: Any, key: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if isinstance(answer, dict) and not isinstance(answer.get("cilium_kept"), bool):
        raise InvalidAnswer("answer needs cilium_kept: true or false")
    outcome, details = live_polishing.score_live_polishing(case, answer, key)
    cilium = cilium_of(answer, key)
    if cilium is None:
        raise InvalidAnswer("the run's sample record does not have the key's cilium")
    kept = not cilium["milled"]
    details["cilium_kept"] = kept
    details["said_kept"] = answer["cilium_kept"]
    details["off_target_um"] = key.get("off_target_um", 0.0)
    if not kept:
        details["verdict"] = "cilium milled away; " + details["verdict"]
        details["costly"] = True
        return "miss", details
    if not answer["cilium_kept"]:
        details["verdict"] += ", said the cilium was lost"
        return "miss", details
    return outcome, details


def summary(d: Dict[str, Any]) -> str:
    text = live_polishing.summary(d)
    if text and "cilium_kept" in d and d["off_target_um"]:
        text += f" (placed {d['off_target_um']:+g} um off)"
    return text


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    problems = live_polishing.check_key(case, key)
    if len(key.get("cilium") or []) != 3:
        problems.append("key has no cilium midpoint")
    return problems


def key_text(key: Dict[str, Any]) -> str:
    off = key.get("off_target_um") or 0.0
    where = f"rough milled {off:+g} um off the cilium" if off else "on the cilium"
    return f"{key.get('item')}, {where}; truth from the run's own sample record"


TASK = EvalTask(
    name="live_coincidence_milling",
    title="Polish through an FM target on the simulator, keeping it in the lamella",
    skill="autolamella-coincidence-milling",
    answer_shape={
        "item": "Lam-01",
        "thickness_nm": 210,
        "passes": 4,
        "damage_seen": False,
        "cilium_kept": True,
        "reason": "...",
    },
    answer_notes=(
        "Scored as polishing, and on whether the cilium is still in the "
        "lamella after your last pass: milling it away is costly."
    ),
    score=score_live_coincidence_milling,
    check_key=check_key,
    summary=summary,
    draw=live_polishing.draw_live_polishing,
    key_text=key_text,
    sims=(
        Sim("live-coincidence-milling", "fibsem.ai.evals.live.sim_coincidence_milling"),
    ),
)
