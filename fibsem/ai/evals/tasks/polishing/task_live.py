"""Live polishing: on the simulator, polish a rough-milled lamella to the
target pass by pass and stop there, or at the first sign of damage. Scored
on the sample's own record from the run (each pass's true gap between the
trenches, and where damage started), which the runner puts in the answer as
``record.sim`` (``fibsem.ai.evals.tasks.polishing.make_live``).

- no damage: stopping within ``reached_nm`` above the target or ``over_nm``
  below it is right; thicker is early (a miss), thinner is too far (costly)
- damage: the right stop is the pass it started on, said so; milling on after
  it is costly
"""

from typing import Any, Dict, List, Optional, Tuple

from fibsem.ai.evals.core.cases import Case
from fibsem.ai.evals.core.scoring import InvalidAnswer
from fibsem.ai.evals.core.visual import KEY, _colour, _figure, _legend
from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim


def _nm(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value) * 1e9, 1)


def passes_of(answer: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, Any]:
    """The milling passes in the run's sample record (those that cut a
    gap), how many of them the set-up milled, and where damage started."""
    record = answer.get("record") or {}
    sim = record.get("sim") or {}
    every = sim.get("passes") or []
    setup = int((record.get("prepared") or {}).get("setup_passes") or 0)
    cut = [p for p in every if p.get("gap") is not None]
    before = sum(1 for p in every[:setup] if p.get("gap") is not None)
    return cut, before, sim.get("damage_started")


def score_live_polishing(
    case: Case, answer: Any, key: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(answer, dict):
        raise InvalidAnswer("answer is not a dict")
    said = answer.get("thickness_nm")
    if not isinstance(said, (int, float)) or isinstance(said, bool):
        raise InvalidAnswer("answer needs thickness_nm, the thickness it stopped at")
    if not isinstance(answer.get("damage_seen"), bool):
        raise InvalidAnswer("answer needs damage_seen: true or false")
    scoring = case.scoring
    target = float(scoring.get("target_nm", 200))
    reached = float(scoring.get("reached_nm", 50))
    over = float(scoring.get("over_nm", 60))
    cut, before, damage = passes_of(answer)
    if not cut:
        raise InvalidAnswer("the run has no sample record: was it a live run?")
    last = cut[-1]
    final = _nm(last.get("gap_mean") or last.get("gap"))
    start = _nm(
        (cut[before - 1].get("gap_mean") or cut[before - 1].get("gap"))
        if before
        else None
    )
    details: Dict[str, Any] = {
        "start_nm": start,
        "final_nm": final,
        "passes": len(cut) - before,
        "said_nm": float(said),
        "said_off_nm": round(float(said) - final, 1),
        "damage_seen": answer["damage_seen"],
        "gaps_nm": [_nm(p.get("gap_mean") or p.get("gap")) for p in cut[before:]],
    }
    if damage is not None:
        # the pass damage started on, among the passes that cut
        started = damage.get("pass")
        after = [p for p in cut if (p.get("pass") or 0) > (started or 0)]
        details["damage"] = {"pass": started, "kind": damage.get("kind")}
        details["passes_after_damage"] = len(after)
        if after:
            details["verdict"] = "milled on after damage"
            details["costly"] = True
            return "miss", details
        details["costly"] = False
        details["verdict"] = "stopped at damage" + (
            "" if answer["damage_seen"] else ", unsaid"
        )
        return ("hit" if answer["damage_seen"] else "miss"), details
    if final > target + reached:
        details.update(verdict="early", costly=False)
        return "miss", details
    if final < target - over:
        details.update(verdict="too thin", costly=True)
        return "miss", details
    details.update(verdict="right", costly=False)
    return "hit", details


def draw_live_polishing(case: Case, runs: List[Run]):
    """Each repeat's true thickness after every pass, against the target's
    band; a cross where damage started."""
    scoring = case.scoring
    target = float(scoring.get("target_nm", 200))
    fig, ax = _figure(9, 5)
    ax.axhspan(
        target - float(scoring.get("over_nm", 60)),
        target + float(scoring.get("reached_nm", 50)),
        color=KEY,
        alpha=0.15,
        lw=0,
    )
    ax.axhline(target, color=KEY, lw=1)
    for n, (result, answer) in enumerate(runs):
        if not answer:
            continue
        cut, before, damage = passes_of(answer)
        gaps = [
            _nm(p.get("gap_mean") or p.get("gap")) for p in cut[max(0, before - 1) :]
        ]
        xs = list(range(len(gaps)))
        ax.plot(xs, gaps, "-o", color=_colour(n), lw=1.5, ms=4)
        if damage is not None:
            for x, p in zip(xs, cut[max(0, before - 1) :]):
                if p.get("pass") == damage.get("pass"):
                    ax.plot(x, gaps[x], "x", color="#d62728", ms=12, mew=3)
        said = answer.get("thickness_nm")
        if isinstance(said, (int, float)) and gaps:
            ax.plot(xs[-1], said, "_", color=_colour(n), ms=18, mew=2)
    ax.set_yscale("log")
    ax.set_xlabel("pass (0: as rough milled)")
    ax.set_ylabel("true thickness (nm); dash: what the run said")
    _legend(ax, runs, f"target {target:g} nm and its band")
    return fig


def check_key(case: Case, key: Dict[str, Any]) -> List[str]:
    return [] if key.get("item") else ["key has no item"]


def summary(d: Dict[str, Any]) -> str:
    if "final_nm" not in d:
        return ""
    text = f"{d['passes']} passes, {d['start_nm']:g} to {d['final_nm']:g} nm: {d['verdict']}"
    if abs(d.get("said_off_nm") or 0) > 50:
        text += f" (said {d['said_nm']:g} nm)"
    return text


def key_text(key: Dict[str, Any]) -> str:
    return f"{key.get('item')}, truth from the run's own sample record"


TASK = EvalTask(
    name="live_polishing",
    title="Polish a lamella to the target on the simulator, and stop",
    skill="autolamella-polishing",
    answer_shape={
        "item": "Lam-01",
        "thickness_nm": 210,
        "passes": 4,
        "damage_seen": False,
        "reason": "...",
    },
    answer_notes=(
        "Scored on the lamella's true thickness after your last pass, and on "
        "whether you milled on after damage started."
    ),
    score=score_live_polishing,
    check_key=check_key,
    summary=summary,
    draw=draw_live_polishing,
    key_text=key_text,
    sims=(Sim("live-polishing", "fibsem.ai.evals.tasks.polishing.make_live"),),
    scripted="fibsem.ai.evals.tasks.polishing.scripted:polishing",
)
