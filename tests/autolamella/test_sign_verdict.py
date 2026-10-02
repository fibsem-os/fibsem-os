"""A verdict set from the app names the operator who set it.

The lamella menu and the grid card write a verdict with no author: neither widget
holds the experiment, which is what knows who is at the instrument. The code that
saves the change does, and calls `Experiment.sign_verdict` first.
"""

from fibsem.applications.autolamella.structures import (
    Experiment,
    GridRecord,
    Lamella,
    QualityRecord,
    Verdict,
)


def _experiment(tmp_path) -> Experiment:
    experiment = Experiment(path=tmp_path, name="sign-verdict-test")
    experiment.metadata["user"] = "Ada"  # typed into the create dialog
    return experiment


def _lamella(tmp_path, verdict: Verdict, author: str = "") -> Lamella:
    lamella = Lamella(path=tmp_path / "01-a", number=1, petname="01-a")
    lamella.defect = QualityRecord(verdict=verdict, author=author)
    return lamella


def test_an_unsigned_verdict_is_signed_by_the_operator(tmp_path):
    lamella = _lamella(tmp_path, Verdict.GOOD)

    _experiment(tmp_path).sign_verdict(lamella)

    assert lamella.defect.author == "human:Ada"


def test_a_verdict_that_already_names_someone_keeps_them(tmp_path):
    """An agent's verdict, or one set by a review decision, is not re-attributed."""
    lamella = _lamella(tmp_path, Verdict.FAILED, author="agent:some-model")

    _experiment(tmp_path).sign_verdict(lamella)

    assert lamella.defect.author == "agent:some-model"


def test_no_verdict_is_signed_by_nobody(tmp_path):
    """Taking a judgement back is not a judgement to put a name to."""
    lamella = _lamella(tmp_path, Verdict.UNASSESSED)

    _experiment(tmp_path).sign_verdict(lamella)

    assert lamella.defect.author == ""


def test_a_grid_s_verdict_is_signed_too(tmp_path):
    grid = GridRecord(name="grid-a")
    grid.quality.set_defect(state=Verdict.GOOD)

    _experiment(tmp_path).sign_verdict(grid)

    assert grid.quality.author == "human:Ada"
