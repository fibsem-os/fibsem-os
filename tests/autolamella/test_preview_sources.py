"""The agent's previews read the experiment with no window (FIB-1459)."""

from pathlib import Path

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.applications.autolamella.task_outputs import PreviewSources


def _experiment(tmp_path: Path) -> Experiment:
    experiment = Experiment(path=tmp_path, name="exp")
    experiment.positions.append(Lamella(path=tmp_path / "L1", number=1, petname="L1"))
    return experiment


def test_previews_read_the_experiment(tmp_path):
    experiment = _experiment(tmp_path)
    sources = PreviewSources(lambda: experiment)

    assert sources.task_config("L1", "NoSuchTask") is None
    assert sources.item_pose("L1", "no-such-pose") is None
    assert sources.item_image("L1") is None
    assert sources.review_proposal({"proposal_id": "nope"}) is None


def test_nothing_to_show_with_no_experiment_or_item(tmp_path):
    experiment = _experiment(tmp_path)
    for sources in (PreviewSources(lambda: None), PreviewSources(lambda: experiment)):
        assert sources.task_config("missing", "Trench") is None
        assert sources.item_pose("missing", "MILLING") is None
        assert sources.item_image("missing") is None
        assert sources.overview_image("missing", "overview.tif") is None
        assert sources.item_overview("missing") is None
        assert sources.review_image({"proposal_id": "nope"}) is None
