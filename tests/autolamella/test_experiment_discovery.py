"""Finding experiments on disk, and narrowing them to a question (FIB-457).

"What did this microscope do this week?" needs something to group by -- the
session record (FIB-451) -- and a way to find the experiments. This is the
second half.

There is no global registry: the default root is inside the installed package,
users create experiments anywhere, and the recents list holds ten. So the
contract is "complete for the roots you gave me, silent about everywhere else",
and the parts that must not break are the ones that make a partial answer safe:
never raising on a bad file, never quietly folding unattributable runs in with a
real instrument's totals.
"""

import os
import time

import pytest
import yaml

from fibsem.applications.autolamella.tools.experiments import (
    discover_experiments,
    filter_experiments,
    group_by_instrument,
)
from fibsem.config import ExperimentSummary, peek_experiment

DAY = 86400.0


def _write(root, name, *, serial=None, model=None, operator=None, age_days=0,
           lamellae=0, subdir=""):
    """An experiment.yaml as the app writes one."""
    directory = os.path.join(root, subdir, name) if subdir else os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    record = {
        "name": name,
        "_id": f"id-{name}",
        "path": directory,
        "created_at": time.time() - age_days * DAY,
        "positions": [{} for _ in range(lamellae)],
        "metadata": {},
        "session": None,
    }
    if serial is not None:
        record["session"] = {
            "recorded_at": record["created_at"],
            "system": {
                "serial_number": serial,
                "model": model,
                "fibsem_version": "0.5.2",
                "application": "autolamella",
            },
            "user": {"name": operator, "hostname": "a-machine"},
            "plugins": {},
        }
    path = os.path.join(directory, "experiment.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(record, f)
    return path


@pytest.fixture
def facility(tmp_path):
    """Two instruments, five runs, one of them predating the session record."""
    root = str(tmp_path)
    _write(root, "arctis-monday", serial="SN-4471", model="Arctis",
           operator="Dr Someone", age_days=6, lamellae=4)
    _write(root, "arctis-friday", serial="SN-4471", model="Arctis",
           operator="Dr Someone", age_days=1, lamellae=2)
    _write(root, "aquilos-run", serial="SN-1102", model="Aquilos",
           operator="A Colleague", age_days=3, lamellae=7)
    _write(root, "old-run", serial="SN-1102", model="Aquilos",
           operator="A Colleague", age_days=40, lamellae=1)
    _write(root, "before-sessions", age_days=2)  # no session key
    return root


# ---------------------------------------------------------------------------
# Reading one
# ---------------------------------------------------------------------------


def test_a_summary_carries_the_instrument_and_operator(tmp_path) -> None:
    path = _write(str(tmp_path), "a-run", serial="SN-4471", model="Arctis",
                  operator="Dr Someone", lamellae=3)

    summary = peek_experiment(path)

    assert summary.instrument_serial == "SN-4471"
    assert summary.instrument_model == "Arctis"
    assert summary.operator == "Dr Someone"
    assert summary.software_version == "0.5.2"
    assert summary.num_lamella == 3


@pytest.mark.parametrize("session", [None, "absent"])
def test_an_experiment_with_no_session_reports_none(tmp_path, session) -> None:
    """Written before FIB-451, or adopted by nothing yet. A key present but null
    has to read the same as the key being missing -- that is how an experiment
    that no session has touched actually serialises.
    """
    path = _write(str(tmp_path), "a-run")
    if session == "absent":
        with open(path) as f:
            record = yaml.safe_load(f)
        del record["session"]
        with open(path, "w") as f:
            yaml.safe_dump(record, f)

    summary = peek_experiment(path)

    assert summary.available is True
    assert summary.instrument_serial is None
    assert summary.operator is None


def test_an_unreadable_file_is_flagged_not_dropped(tmp_path) -> None:
    """Keeping it is the point: a listing that silently omits what it could not
    parse reads as "this never happened"."""
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "experiment.yaml").write_text("{{{ not yaml")

    summary = peek_experiment(str(directory / "experiment.yaml"))

    assert summary.exists is True
    assert summary.available is False
    assert summary.name == "broken"  # falls back to the directory


# ---------------------------------------------------------------------------
# Finding them
# ---------------------------------------------------------------------------


def test_it_finds_every_experiment_under_a_root(facility) -> None:
    assert {e.name for e in discover_experiments(facility)} == {
        "arctis-monday", "arctis-friday", "aquilos-run", "old-run", "before-sessions",
    }


def test_the_listing_is_newest_first(facility) -> None:
    found = discover_experiments(facility)

    assert [e.name for e in found[:2]] == ["arctis-friday", "before-sessions"]
    assert found[-1].name == "old-run"


def test_it_does_not_descend_into_an_experiment(tmp_path) -> None:
    """Everything below an experiment.yaml belongs to that experiment. Without
    pruning, a nested run folder would be reported as a second experiment -- and
    the walk would cover every image directory on the way."""
    root = str(tmp_path)
    _write(root, "outer", serial="SN-1", model="Arctis")
    _write(os.path.join(root, "outer"), "inner-run", serial="SN-1", model="Arctis")

    assert [e.name for e in discover_experiments(root)] == ["outer"]


def test_it_finds_experiments_one_level_down(tmp_path) -> None:
    """A per-project or per-user folder between the root and the experiments is
    the normal facility layout."""
    root = str(tmp_path)
    _write(root, "a-run", serial="SN-1", model="Arctis", subdir="project-x")

    assert [e.name for e in discover_experiments(root)] == ["a-run"]


def test_it_stops_at_the_depth_limit(tmp_path) -> None:
    """Unbounded descent over a data drive is slow enough to look like a hang."""
    root = str(tmp_path)
    _write(root, "deep", serial="SN-1", model="Arctis", subdir="a/b/c/d")

    assert discover_experiments(root, max_depth=2) == []
    assert len(discover_experiments(root, max_depth=6)) == 1


def test_a_root_that_is_not_there_returns_nothing(tmp_path) -> None:
    """Pointing this at an unmounted drive is an ordinary mistake, and an empty
    answer beats a traceback."""
    assert discover_experiments(str(tmp_path / "not-mounted")) == []


def test_one_bad_file_does_not_take_out_the_walk(tmp_path) -> None:
    root = str(tmp_path)
    _write(root, "fine", serial="SN-1", model="Arctis")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "experiment.yaml").write_text("{{{ not yaml")

    found = discover_experiments(root)

    assert len(found) == 2
    assert [e.name for e in found if not e.available] == ["broken"]


# ---------------------------------------------------------------------------
# Narrowing them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("needle", ["SN-4471", "sn-4471", "Arctis", "arctis"])
def test_instrument_matches_serial_or_model_either_case(facility, needle) -> None:
    """A facility manager knows their instrument by whichever they use; making
    them guess which one the flag wants is not worth a second flag."""
    found = filter_experiments(discover_experiments(facility), instrument=needle)

    assert {e.name for e in found} == {"arctis-monday", "arctis-friday"}


def test_an_experiment_with_no_session_matches_no_instrument(facility) -> None:
    """It is not evidence of the instrument you asked about. Folding unknowns in
    is how a per-instrument total comes out wrong."""
    found = filter_experiments(discover_experiments(facility), instrument="Arctis")

    assert "before-sessions" not in {e.name for e in found}


def test_it_filters_by_operator(facility) -> None:
    found = filter_experiments(discover_experiments(facility), operator="colleague")

    assert {e.name for e in found} == {"aquilos-run", "old-run"}


def test_it_filters_to_a_time_window(facility) -> None:
    """The question the issue is named for: what happened this week."""
    week = time.time() - 7 * DAY
    found = filter_experiments(discover_experiments(facility), since=week)

    assert "old-run" not in {e.name for e in found}
    assert len(found) == 4


def test_since_and_until_bound_both_ends(facility) -> None:
    found = filter_experiments(
        discover_experiments(facility),
        since=time.time() - 5 * DAY,
        until=time.time() - 2 * DAY,
    )

    assert {e.name for e in found} == {"aquilos-run", "before-sessions"}


def test_no_filters_changes_nothing(facility) -> None:
    found = discover_experiments(facility)

    assert filter_experiments(found) == found


def test_the_instrument_and_the_week_compose(facility) -> None:
    found = filter_experiments(
        discover_experiments(facility),
        instrument="SN-1102",
        since=time.time() - 7 * DAY,
    )

    assert {e.name for e in found} == {"aquilos-run"}


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_it_groups_by_serial_not_model(facility) -> None:
    """Serial is what distinguishes two of the same model in one facility, which
    is the case grouping exists for."""
    grouped = group_by_instrument(discover_experiments(facility))

    assert {e.name for e in grouped["SN-4471"]} == {"arctis-monday", "arctis-friday"}
    assert {e.name for e in grouped["SN-1102"]} == {"aquilos-run", "old-run"}


def test_unattributable_runs_get_their_own_bucket(facility) -> None:
    """Worth seeing rather than hiding: it is how many runs cannot be attributed."""
    grouped = group_by_instrument(discover_experiments(facility))

    assert [e.name for e in grouped[None]] == ["before-sessions"]


def test_grouping_nothing_gives_nothing() -> None:
    assert group_by_instrument([]) == {}


# ---------------------------------------------------------------------------
# The CLI's own argument handling
# ---------------------------------------------------------------------------


def test_days_and_since_together_is_rejected() -> None:
    """Both set a lower bound. Silently letting one win would answer a different
    question than the one asked."""
    from argparse import Namespace

    from fibsem.cli import _since_timestamp

    with pytest.raises(ValueError, match="one or the other"):
        _since_timestamp(Namespace(days=7, since="2026-08-01"))


def test_neither_days_nor_since_means_no_lower_bound() -> None:
    from argparse import Namespace

    from fibsem.cli import _since_timestamp

    assert _since_timestamp(Namespace(days=None, since=None)) is None


def test_days_resolves_to_a_timestamp_in_the_past() -> None:
    from argparse import Namespace

    from fibsem.cli import _since_timestamp

    since = _since_timestamp(Namespace(days=7, since=None))

    assert time.time() - 8 * DAY < since < time.time() - 6 * DAY


def test_the_summary_type_is_the_one_the_recents_list_uses() -> None:
    """One type for both, so a facility listing and the load dialog cannot drift
    into disagreeing about what an experiment says about itself."""
    from typing import List, get_type_hints

    from fibsem.config import get_recent_experiments

    # get_type_hints, not raw __annotations__: one module uses postponed
    # evaluation and the other does not, so the raw values are a string and a
    # type and would never compare equal however aligned the code is.
    expected = List[ExperimentSummary]
    assert get_type_hints(get_recent_experiments)["return"] == expected
    assert get_type_hints(discover_experiments)["return"] == expected
    assert peek_experiment("/nowhere/experiment.yaml").__class__ is ExperimentSummary
