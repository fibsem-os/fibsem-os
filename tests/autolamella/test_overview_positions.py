"""Confirming a decision can create items, not only edit the one it is on.

The one write in the table that makes lamellae: a decision on a grid's
overview-positions proposal creates one lamella per accepted position. It is
planned like every other write -- checked in full before anything happens,
undone together on any failure -- and needs no microscope, because the poses
it creates from are in the value rather than read off an instrument.

Real Experiment and real GridRecord against the Demo microscope; the poses are
built the way the app builds them, through `build_lamella_poses`.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import LamellaPoses, build_lamella_poses
from fibsem.applications.autolamella.proposals import (
    OVERVIEW_POSITIONS,
    PROPOSAL_KINDS,
    Decision,
    DecisionOutcome,
    Proposal,
    ValueRefused,
    prepare_values,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures import FibsemStagePosition

# The Arctis simulator, because this needs a holder with grids in it.
CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
OVERVIEW = "SEM Overview"


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(tmp_path, microscope):
    exp = Experiment(path=tmp_path, name="positions-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.sync_grids_from_inventory(microscope._stage)
    assert exp.grids, "the Demo holder has grids to put lamellae on"
    return exp


def _placed(microscope, x: float, y: float) -> LamellaPoses:
    """A position as the renderer would hand it over: marked on the overview,
    turned into both poses by the instrument's geometry there and then."""
    return build_lamella_poses(
        microscope=microscope,
        position=FibsemStagePosition(x=x, y=y, z=0, r=0, t=0),
    )


def _proposal(values=None) -> Proposal:
    """As the overview task would record it: the run it came from is stamped on
    the provenance, which is what a decision names."""
    return Proposal(
        kind=OVERVIEW_POSITIONS,
        values=values or {},
        provenance={"task_id": "run-1", "proposer": OVERVIEW},
    )


def _decide(experiment, grid, proposal, values, task_name=OVERVIEW):
    grid.proposals[task_name] = proposal
    return experiment.decide(
        grid.id,
        task_name,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values=values,
            task_id=proposal.task_id,
        ),
    )


# ---------------------------------------------------------------------------
# The kind and its value
# ---------------------------------------------------------------------------


def test_the_kind_carries_positions_and_nothing_else():
    assert PROPOSAL_KINDS[OVERVIEW_POSITIONS].values == ("positions",)


def test_a_position_round_trips_through_the_record(microscope):
    """The value is written to YAML and read back as both poses, which is what
    a lamella is made from."""
    placed = _placed(microscope, 1e-4, 2e-4)
    proposal = _proposal({"positions": [placed]})

    back = Proposal.from_dict(proposal.to_dict())

    assert len(back.values["positions"]) == 1
    read = back.values["positions"][0]
    assert isinstance(read, LamellaPoses)
    assert read.milling.stage_position.x == pytest.approx(
        placed.milling.stage_position.x
    )
    assert (read.fluorescence is None) == (placed.fluorescence is None)


# ---------------------------------------------------------------------------
# What a confirm does
# ---------------------------------------------------------------------------


def test_confirming_creates_one_lamella_for_each_accepted_position(
    microscope, experiment
):
    grid = experiment.grids[0]
    before = len(experiment.positions)
    placed = [_placed(microscope, 1e-4, 0), _placed(microscope, -1e-4, 5e-5)]

    result = _decide(experiment, grid, _proposal(), {"positions": placed})

    assert result.applied, result.reason
    assert len(experiment.positions) == before + 2
    made = experiment.positions[-2:]
    assert all(lamella.grid_id == grid.id for lamella in made), "stamped with the grid"
    assert {lamella.name for lamella in made} == {lamella.name for lamella in made}, (
        "each has its own name"
    )


def test_confirming_with_nothing_placed_is_an_answer(microscope, experiment):
    """A grid the reviewer looked at and left empty is decided, not rejected."""
    grid = experiment.grids[0]
    before = len(experiment.positions)

    result = _decide(experiment, grid, _proposal(), {"positions": []})

    assert result.applied
    assert len(experiment.positions) == before
    assert grid.proposals[OVERVIEW].current.outcome is DecisionOutcome.Confirmed


def test_the_lamellae_are_created_at_the_poses_in_the_value(microscope, experiment):
    """The poses come from the value, not from wherever the stage happens to
    be: that is what lets a decision be made with no instrument to read."""
    grid = experiment.grids[0]
    placed = _placed(microscope, 3e-4, -2e-4)

    _decide(experiment, grid, _proposal(), {"positions": [placed]})

    made = experiment.positions[-1]
    assert made.milling_pose.stage_position.x == pytest.approx(
        placed.milling.stage_position.x
    )
    assert made.milling_pose.stage_position.y == pytest.approx(
        placed.milling.stage_position.y
    )


# ---------------------------------------------------------------------------
# All or nothing
# ---------------------------------------------------------------------------


def test_a_value_that_is_not_a_position_is_refused_before_anything_is_made(
    microscope, experiment
):
    grid = experiment.grids[0]
    before = len(experiment.positions)

    result = _decide(
        experiment, grid, _proposal(), {"positions": [_placed(microscope, 0, 0), "x"]}
    )

    assert not result.applied and result.error_type == "invalid_value"
    assert len(experiment.positions) == before, "the good one was not made either"
    assert grid.proposals[OVERVIEW].pending, "and nothing was decided"


def test_positions_must_be_a_list(microscope, experiment):
    grid = experiment.grids[0]
    result = _decide(
        experiment, grid, _proposal(), {"positions": _placed(microscope, 0, 0)}
    )
    assert not result.applied and "must be a list" in result.reason


def test_a_failure_part_way_through_leaves_no_lamellae_and_no_decision(
    microscope, experiment, monkeypatch
):
    """The second creation raises. The first must not survive it, and the
    decision must not be on the record."""
    grid = experiment.grids[0]
    before = len(experiment.positions)
    real = Experiment.add_new_lamella
    calls = {"n": 0}

    def explode(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the second one failed")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Experiment, "add_new_lamella", explode)
    placed = [_placed(microscope, 1e-4, 0), _placed(microscope, -1e-4, 0)]

    result = _decide(experiment, grid, _proposal(), {"positions": placed})

    assert not result.applied
    assert len(experiment.positions) == before, "the first was undone"
    assert grid.proposals[OVERVIEW].pending, "nothing was decided"


def test_a_confirm_without_a_protocol_is_refused_rather_than_crashing(
    microscope, experiment
):
    """A lamella is built from the protocol's defaults. An experiment with
    none loaded is refused before anything is created, with something a user
    can act on -- found by confirming in the app against an experiment whose
    protocol had not been loaded."""
    grid = experiment.grids[0]
    experiment.task_protocol = None
    before = len(experiment.positions)

    result = _decide(
        experiment, grid, _proposal(), {"positions": [_placed(microscope, 1e-4, 0)]}
    )

    assert not result.applied and result.error_type == "invalid_value"
    assert "No protocol is loaded" in result.reason
    assert len(experiment.positions) == before


def test_the_writer_needs_an_experiment_and_an_item_with_an_id(microscope):
    """Refused, rather than half-written: prepare_values checks before it
    plans."""
    placed = [_placed(microscope, 0, 0)]
    with pytest.raises(ValueRefused):
        prepare_values(None, object(), OVERVIEW_POSITIONS, {"positions": placed})
