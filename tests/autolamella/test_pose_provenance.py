"""Where a lamella's poses came from, and whether they can still be trusted (FIB-831).

A lamella's milling and fluorescence poses are two observations of one piece of sample
from two instruments; the transform between them is a first guess. So each pose says
whether a person put it there, or it was worked out from the other one, or it was
observed and the other one has moved since. The rest of the application reads this to
decide defaults and to show which lamellae still need centring by hand.
"""

from fibsem.applications.autolamella.structures import Lamella, PoseProvenance
from fibsem.structures import FibsemStagePosition, MicroscopeState


def _lamella():
    return Lamella(petname="Lamella-01", path="/nowhere/Lamella-01", number=1)


def _state(x=0.0):
    return MicroscopeState(stage_position=FibsemStagePosition(x=x, y=0.0, z=0.0))


def test_a_pose_that_never_said_is_observed():
    """Every pose saved before provenance existed was somebody's decision."""
    lamella = _lamella()
    lamella.poses["MILLING"] = _state()

    assert lamella.provenance_of("MILLING") is PoseProvenance.OBSERVED


def test_the_setters_record_an_observation():
    """The plain assignment every workflow task uses: it recorded where it was."""
    lamella = _lamella()
    lamella.milling_pose = _state()
    lamella.fluorescence_pose = _state()

    assert lamella.provenance_of("MILLING") is PoseProvenance.OBSERVED
    assert lamella.provenance_of("FLUORESCENCE") is PoseProvenance.OBSERVED


def test_a_setter_does_not_doubt_the_other_pose():
    """A task re-recording its milling pose is not a reason to call the fluorescence
    pose stale. Only a caller that *moved* the lamella says so, explicitly."""
    lamella = _lamella()
    lamella.fluorescence_pose = _state()
    lamella.milling_pose = _state(1e-6)

    assert lamella.provenance_of("FLUORESCENCE") is PoseProvenance.OBSERVED


def test_set_pose_records_what_it_is_told():
    lamella = _lamella()
    lamella.set_pose("FLUORESCENCE", _state(), PoseProvenance.DERIVED)

    assert lamella.provenance_of("FLUORESCENCE") is PoseProvenance.DERIVED


def test_moving_a_pose_keeps_what_it_carries():
    """The objective position most of all: someone focused there by hand."""
    lamella = _lamella()
    pose = _state()
    pose.objective_position = 7.7e-3
    lamella.set_pose("FLUORESCENCE", pose, PoseProvenance.DERIVED)

    lamella.set_pose_position("FLUORESCENCE", FibsemStagePosition(x=5e-6, y=0, z=0))

    assert lamella.fluorescence_pose.objective_position == 7.7e-3
    assert lamella.fluorescence_pose.stage_position.x == 5e-6
    assert lamella.provenance_of("FLUORESCENCE") is PoseProvenance.OBSERVED


def test_only_an_observed_pose_can_go_stale():
    """A derived one was a guess already; a missing one has nothing to be stale
    about."""
    lamella = _lamella()
    lamella.set_pose("FLUORESCENCE", _state(), PoseProvenance.DERIVED)
    lamella.mark_pose_stale("FLUORESCENCE")
    assert lamella.provenance_of("FLUORESCENCE") is PoseProvenance.DERIVED

    lamella.mark_pose_stale("MILLING")  # no such pose: nothing happens
    assert "MILLING" not in lamella.pose_provenance

    lamella.set_pose("FLUORESCENCE", _state(), PoseProvenance.OBSERVED)
    lamella.mark_pose_stale("FLUORESCENCE")
    assert lamella.provenance_of("FLUORESCENCE") is PoseProvenance.STALE


def test_provenance_round_trips_through_the_experiment_file():
    lamella = _lamella()
    lamella.set_pose("MILLING", _state(), PoseProvenance.OBSERVED)
    lamella.set_pose("FLUORESCENCE", _state(), PoseProvenance.DERIVED)

    loaded = Lamella.from_dict(lamella.to_dict())

    assert loaded.provenance_of("MILLING") is PoseProvenance.OBSERVED
    assert loaded.provenance_of("FLUORESCENCE") is PoseProvenance.DERIVED
    assert loaded.to_dict()["pose_provenance"] == {
        "MILLING": "observed",
        "FLUORESCENCE": "derived",
    }


def test_a_file_cannot_make_a_pose_less_trustworthy_than_a_person_did():
    """An unknown value, or an entry for a pose that is not there, reads as observed
    -- the same as a file from before provenance existed."""
    lamella = _lamella()
    lamella.set_pose("MILLING", _state(), PoseProvenance.DERIVED)
    data = lamella.to_dict()
    data["pose_provenance"] = {"MILLING": "guessed?", "FLUORESCENCE": "derived"}

    loaded = Lamella.from_dict(data)

    assert loaded.provenance_of("MILLING") is PoseProvenance.OBSERVED
    assert "FLUORESCENCE" not in loaded.pose_provenance


def test_an_old_file_loads_with_every_pose_observed():
    lamella = _lamella()
    lamella.milling_pose = _state()
    data = lamella.to_dict()
    del data["pose_provenance"]

    loaded = Lamella.from_dict(data)

    assert loaded.provenance_of("MILLING") is PoseProvenance.OBSERVED
