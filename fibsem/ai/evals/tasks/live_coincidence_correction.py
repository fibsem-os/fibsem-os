"""Live coincidence correction: on the simulator, the stage starts over a
good cell at the wrong height; the agent makes the site coincident, checks
it, and saves it. Scored from the height the site was saved at, against
where the simulator's SEM and FIB view centres meet at that cell
(``fibsem.ai.evals.live.sim_correction``).

The site's pose is the experiment's saved item, which the runner puts in
the answer as ``record`` after the run: what the agent saved, not what it
says it saved.
"""

import math
from typing import Any, Dict, List, Tuple

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.scoring import InvalidAnswer
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim
from fibsem.ai.evals.tasks.live_site_selection import COINCIDENT, _positions
from fibsem.ai.evals.visual import KEY, _colour, _figure, _legend

# Saying it is coincident when the beams are further apart than this (m) is
# costly: the next step mills where the FIB looks, not at the cell
CLAIM_SLACK = 2e-6


def _separation(optics: Dict[str, Any], z: float) -> float:
    """How far apart the SEM's and FIB's view centres are at stage z (m)."""
    return optics["coincidence_per_z"] * abs(z - optics["coincident_z"])


def score_live_coincidence_correction(
    case: Case, answer: Any, key: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(answer, dict) or not isinstance(answer.get("coincident"), bool):
        raise InvalidAnswer("answer needs coincident: true or false")
    scoring = case.scoring
    tolerance = float(scoring.get("tolerance_um", COINCIDENT * 1e6)) * 1e-6
    optics, start, cell = key["optics"], key["start"], key["cell"]
    name = str(answer.get("site") or key.get("site") or "")
    positions = _positions(answer)
    at = positions.get(name)
    if at is None and len(positions) == 1:
        # saved under another name: still the one site
        at = next(iter(positions.values()))
    said = answer["coincident"]
    details: Dict[str, Any] = {
        "start_off_um": round(_separation(optics, start["z"]) * 1e6, 2),
        "said_coincident": said,
        "saved": at is not None,
    }
    if at is None:
        details["costly"] = False
        return "miss", details
    off = _separation(optics, float(at.get("z") or 0.0))
    moved = math.hypot(at["x"] - start["x"], at["y"] - start["y"])
    details.update(
        {
            "final_off_um": round(off * 1e6, 2),
            "coincident": off <= tolerance,
            "moved_um": round(moved * 1e6, 2),
            "on_cell": moved <= cell["radius_um"] * 1e-6,
            "costly": said and off > CLAIM_SLACK,
        }
    )
    hit = details["coincident"] and details["on_cell"] and said
    return ("hit" if hit else "miss"), details


def draw_live_coincidence_correction(case: Case, runs: List[Run]):
    """Each repeat's beam separation (um), where it started and where its
    site was saved, against the tolerance."""
    key = case.key
    fig, ax = _figure(8, 5)
    start = key.get("start_off_um")
    tolerance = float(case.scoring.get("tolerance_um", COINCIDENT * 1e6))
    ax.axhspan(0, tolerance, color=KEY, alpha=0.15, lw=0)
    ax.axhline(tolerance, color=KEY, lw=1)
    for n, (result, _) in enumerate(runs):
        details = (result.get("scores") or [{}])[0].get("details") or {}
        final = details.get("final_off_um")
        if final is None:
            ax.plot([0], [start], "o", color=_colour(n))
            ax.annotate(
                "not saved", (0, start), xytext=(6, 0), textcoords="offset points"
            )
            continue
        ax.plot([0, 1], [start, final], "-o", color=_colour(n), lw=2)
        if details.get("said_coincident"):
            ax.annotate(
                "said coincident",
                (1, final),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xticks([0, 1], ["start", "saved"])
    ax.set_xlim(-0.3, 1.6)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("SEM to FIB view centre (um)")
    _legend(ax, runs, f"coincident: within {tolerance:g} um")
    return fig


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    problems = []
    for field in ("optics", "start", "cell"):
        if not isinstance(key.get(field), dict):
            problems.append(f"key has no {field}")
    return problems


def summary(d: Dict[str, Any]) -> str:
    if "start_off_um" not in d:
        return ""
    if not d.get("saved"):
        return f"started {d['start_off_um']:g} um apart; no site saved"
    text = f"{d['start_off_um']:g} um apart to {d['final_off_um']:g} um"
    if not d.get("on_cell"):
        text += f", moved {d['moved_um']:g} um off the cell"
    if d.get("costly"):
        text += ", said coincident when it was not"
    return text


def key_text(key: Dict[str, Any]) -> str:
    return f"starts {key.get('start_off_um', '?')} um off coincidence"


TASK = EvalTask(
    name="live_coincidence_correction",
    title="Make a site coincident on the simulator, and check it",
    skill="fibsem-microscope",
    answer_shape={"site": "Site-01", "coincident": True, "how": "..."},
    answer_notes=(
        "The site is the item you saved (item_add), scored at the height it "
        "was saved at; coincident is whether you checked it is."
    ),
    score=score_live_coincidence_correction,
    check_key=check_key,
    summary=summary,
    draw=draw_live_coincidence_correction,
    key_text=key_text,
    sims=(Sim("live-coincidence-correction", "fibsem.ai.evals.live.sim_correction"),),
)
