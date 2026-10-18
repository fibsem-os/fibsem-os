"""Stage positions on an overview: a proposal any app's grid tasks can make.

An overview task proposes where items go on its grid; a decision confirms the
positions, and the app makes its items there -- AutoLamella its lamellae, a
volume app its own kind of site. Each position is carried as the pose pair the
instrument's geometry produced for it (``ItemPoses``).

Not a point on an image and not a bare stage position: either would have made
a decision need a microscope to turn into poses, and there is no instrument
where a decision is made -- the Review tab days later, or an agent over the
server. The geometry is read where the position is placed instead, in the
renderer or in a proposer running at the beam.

Importing this module registers the kind, its value's codec and its writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from fibsem.structures import MicroscopeState
from fibsem.workflows.proposals import (
    PreparedWrite,
    ProposalKind,
    ValueRefused,
    register_proposal_kind,
    register_value_codec,
    register_value_writer,
)

# Where items go on a grid: the positions a decision creates them at. The only
# kind whose confirmed values make items rather than edit the one the proposal
# is on.
OVERVIEW_POSITIONS = "overview_positions"


@dataclass(frozen=True)
class ItemPoses:
    """Both poses for an item, whichever side it was marked from: where it is
    milled, and where it is looked at under fluorescence.

    `fluorescence` is None when the microscope has no fluorescence detector --
    there is nothing to derive it from, and a pose invented for an instrument
    that does not exist would be worse than its absence.
    """

    milling: MicroscopeState
    fluorescence: Optional[MicroscopeState] = None


register_proposal_kind(
    ProposalKind(name=OVERVIEW_POSITIONS, values=("positions",), label="Positions")
)


def _positions_to_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    return [
        {
            "milling": p.milling.to_dict(),
            "fluorescence": (
                p.fluorescence.to_dict() if p.fluorescence is not None else None
            ),
        }
        if isinstance(p, ItemPoses)
        else p
        for p in value
    ]


def _positions_from_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    out: List[Any] = []
    for p in value:
        if not isinstance(p, dict) or "milling" not in p:
            out.append(p)
            continue
        fm = p.get("fluorescence")
        out.append(
            ItemPoses(
                milling=MicroscopeState.from_dict(p["milling"]),
                fluorescence=MicroscopeState.from_dict(fm) if fm else None,
            )
        )
    return out


register_value_codec("positions", _positions_to_dict, _positions_from_dict)


def _prepare_positions(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Check the positions, then hand them to the experiment's app, which plans
    the items it makes at them (``experiment.prepare_items_at``) without
    touching anything yet.

    Planned in full and undone together, like every other write: a
    half-created set is worse than none, and the decision is only appended
    once the whole set is on the experiment. The poses come from the value
    rather than the instrument, so nothing here needs a microscope.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueRefused(f"positions must be a list, not {type(value).__name__}.")
    placed: List[ItemPoses] = []
    for entry in value:
        if not isinstance(entry, ItemPoses):
            raise ValueRefused(
                f"every position must be an ItemPoses, not {type(entry).__name__}."
            )
        if entry.milling is None:
            raise ValueRefused("every position needs a milling pose.")
        placed.append(entry)
    prepare = getattr(experiment, "prepare_items_at", None)
    if prepare is None:
        raise ValueRefused("positions can only be written to an experiment.")
    return prepare(item, placed)


# The experiment is there for this one write: it makes items rather than
# editing the one the proposal is on.
register_value_writer("positions", _prepare_positions)
