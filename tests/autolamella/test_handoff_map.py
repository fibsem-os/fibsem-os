"""What the handoff document is allowed to say about a lamella.

The document travels with the grid to a TEM, is read by someone who was not there, and
is the only thing they have. So the tests that matter are the ones about *restraint*:
what it must not claim, and where it must print a dash instead of a number.

Two defects motivated most of this, both found by looking at real output:

- reading "the last completed milling task" for the final geometry picks up
  `Mill Fiducial` on a lamella whose milling had barely started, and reported the
  fiducial's width as the lamella's;
- there is no field in the record that says a lamella is *ready*, so the document must
  not compute one.
"""

from pathlib import Path

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    DefectType,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.tools.handoff_map import (
    HandoffOptions,
    final_geometry,
    lamella_row,
    provenance_line,
    summary_line,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern, TrenchPattern


@pytest.fixture
def lamella(tmp_path: Path) -> Lamella:
    lam = Lamella(path=tmp_path / "lam", number=1, petname="01-test-lamella")
    lam.milling_angle = 15.0
    return lam


def _completed(name: str) -> AutoLamellaTaskState:
    return AutoLamellaTaskState(name=name, status=AutoLamellaTaskStatus.Completed)


def _milling_config(stage: FibsemMillingStage) -> AutoLamellaTaskConfig:
    """A task config whose one milling task holds *stage*."""
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    config = AutoLamellaTaskConfig()
    config.milling = {"task": FibsemMillingTaskConfig(stages=[stage])}
    return config


def _trench(spacing: float, width: float) -> FibsemMillingStage:
    return FibsemMillingStage(
        name="Polish", pattern=TrenchPattern(width=width, spacing=spacing)
    )


def _fiducial(width: float) -> FibsemMillingStage:
    """A rectangle -- like a fiducial, it has a width but no spacing."""
    return FibsemMillingStage(name="Fiducial", pattern=RectanglePattern(width=width))


class TestFinalGeometry:
    def test_a_trench_gives_the_thickness_and_width(self, lamella):
        lamella.task_config["Polishing"] = _milling_config(_trench(300e-9, 9e-6))
        lamella.task_history.append(_completed("Polishing"))

        geometry = final_geometry(lamella)
        assert geometry["thickness"] == pytest.approx(300e-9)
        assert geometry["width"] == pytest.approx(9e-6)

    def test_a_fiducial_is_not_read_as_a_lamella(self, lamella):
        """The defect this file exists for.

        A fiducial is milled early and has a width, so "the last completed milling task"
        reported it as the lamella's own -- a straight-faced "width: 1.0 um" for a
        lamella that had not been cut yet.
        """
        lamella.task_config["Mill Fiducial"] = _milling_config(_fiducial(1e-6))
        lamella.task_history.append(_completed("Mill Fiducial"))

        geometry = final_geometry(lamella)
        assert geometry == {"thickness": None, "width": None}

    def test_the_trench_wins_over_a_later_fiducial(self, lamella):
        """Order in the history does not override what the pattern actually describes."""
        lamella.task_config["Rough Milling"] = _milling_config(_trench(4e-6, 10e-6))
        lamella.task_config["Mill Fiducial"] = _milling_config(_fiducial(1e-6))
        lamella.task_history.append(_completed("Rough Milling"))
        lamella.task_history.append(_completed("Mill Fiducial"))

        assert final_geometry(lamella)["thickness"] == pytest.approx(4e-6)

    def test_the_most_recent_trench_wins(self, lamella):
        lamella.task_config["Rough Milling"] = _milling_config(_trench(4e-6, 10e-6))
        lamella.task_config["Polishing"] = _milling_config(_trench(300e-9, 9e-6))
        lamella.task_history.append(_completed("Rough Milling"))
        lamella.task_history.append(_completed("Polishing"))

        assert final_geometry(lamella)["thickness"] == pytest.approx(300e-9)

    def test_an_unmilled_lamella_reports_nothing(self, lamella):
        assert final_geometry(lamella) == {"thickness": None, "width": None}

    def test_a_task_that_never_completed_is_not_read(self, lamella):
        """Configured is not the same as done."""
        lamella.task_config["Polishing"] = _milling_config(_trench(300e-9, 9e-6))
        assert final_geometry(lamella) == {"thickness": None, "width": None}


class TestTheRow:
    def test_unknown_values_are_dashes_not_zeroes(self, lamella):
        row = lamella_row(lamella)
        assert row["Thickness"] == "-"
        assert row["Width"] == "-"
        assert row["Last task"] == "-"
        assert row["Finished"] == "-"

    def test_a_flagged_lamella_says_so(self, lamella):
        lamella.defect.set_defect("lost the fiducial", DefectType.FAILURE)
        row = lamella_row(lamella)
        assert row["Defect"] == "failed"
        assert row["Note"] == "lost the fiducial"

    def test_rework_is_distinct_from_failed(self, lamella):
        lamella.defect.set_defect("too thick", DefectType.REWORK)
        assert lamella_row(lamella)["Defect"] == "rework"

    def test_an_unflagged_lamella_is_not_called_anything(self, lamella):
        """Not "ok", not "ready" -- the record holds no such judgement to report."""
        row = lamella_row(lamella)
        assert row["Defect"] == "-"

    def test_the_description_is_preferred_over_the_defect_note(self, lamella):
        lamella.description = "mitochondria cluster"
        lamella.defect.set_defect("too thick", DefectType.REWORK)
        assert lamella_row(lamella)["Note"] == "mitochondria cluster"


class TestTheSummaryLine:
    @pytest.fixture
    def experiment(self, tmp_path):
        exp = Experiment(path=tmp_path, name="handoff")
        for i, state in enumerate(
            (DefectType.NONE, DefectType.NONE, DefectType.REWORK, DefectType.FAILURE)
        ):
            lam = Lamella(path=Path(exp.path) / f"L{i}", number=i, petname=f"L{i}")
            if state is not DefectType.NONE:
                lam.defect.set_defect("", state)
            exp.positions.append(lam)
        return exp

    def test_it_counts_only_what_a_human_flagged(self, experiment):
        line = summary_line(experiment, HandoffOptions())
        assert "4 lamellae" in line
        assert "1 rework" in line
        assert "1 failed" in line

    def test_it_never_claims_a_lamella_is_ready(self, experiment):
        """There is no readiness field, so there is no readiness claim."""
        line = summary_line(experiment, HandoffOptions()).lower()
        for word in ("ready", "good", "usable", "ok"):
            assert word not in line

    def test_the_grid_and_slot_appear_when_given(self, experiment):
        line = summary_line(experiment, HandoffOptions(grid="A", slot="3"))
        assert "Grid A" in line and "slot 3" in line

    def test_a_selection_narrows_the_counts(self, experiment):
        options = HandoffOptions(lamella_names=["L0", "L1"])
        line = summary_line(experiment, options)
        assert "2 lamellae" in line
        assert "failed" not in line


class TestProvenance:
    def test_it_is_empty_without_a_session(self, tmp_path):
        """Better to say nothing than to print a line of "unknown"s.

        Every experiment written before the session record existed has none, and a
        header full of blanks reads as a measurement that came back empty.
        """
        exp = Experiment(path=tmp_path, name="no-session")
        exp.created_at = None
        exp.session = None
        assert provenance_line(exp) == ""

    def test_it_names_the_instrument_and_operator(self, tmp_path):
        from fibsem.structures import FibsemUser, SessionInfo, SystemInfo

        exp = Experiment(path=tmp_path, name="with-session")
        exp.session = SessionInfo(
            system=SystemInfo(
                name="config",
                ip_address="0.0.0.0",
                manufacturer="Thermo",
                model="Aquilos 2",
                serial_number="9876",
                hardware_version="1",
                software_version="1",
                fibsem_revision="abc",
            ),
            user=FibsemUser(name="operator"),
        )
        line = provenance_line(exp)
        assert "Aquilos 2" in line and "9876" in line and "operator" in line


class TestSelection:
    def test_none_means_every_lamella(self, tmp_path):
        exp = Experiment(path=tmp_path, name="sel")
        for i in range(3):
            exp.positions.append(
                Lamella(path=Path(exp.path) / f"L{i}", number=i, petname=f"L{i}")
            )
        assert len(HandoffOptions().selected(exp)) == 3

    def test_names_select_in_experiment_order(self, tmp_path):
        exp = Experiment(path=tmp_path, name="sel")
        for i in range(3):
            exp.positions.append(
                Lamella(path=Path(exp.path) / f"L{i}", number=i, petname=f"L{i}")
            )
        chosen = HandoffOptions(lamella_names=["L2", "L0"]).selected(exp)
        assert [lam.name for lam in chosen] == ["L0", "L2"]


class TestTheFlagIsOffByDefault:
    def test_handoff_map_is_not_enabled_for_anyone_yet(self):
        """It ships beside the overview plot, not in place of it, until it has been used."""
        from fibsem.config import FeatureFlags

        assert FeatureFlags().handoff_map is False
