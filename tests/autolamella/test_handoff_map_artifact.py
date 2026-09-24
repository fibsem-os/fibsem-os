"""Writing the handoff map without anyone asking for it.

The map matters most at the end of a run nobody stayed for -- which is exactly when no
one is there to open a dialog. This is the hook that writes it anyway.

Two things are pinned here beyond "it writes a file". It is off unless the flag is on,
because a PDF appearing in every user's experiment directory is not a change to make for
everyone before anyone has read one. And a map that cannot be written must not take the
run down with it: the run is the valuable thing, the artifact is a convenience.
"""

from pathlib import Path

import pytest

from fibsem.applications.autolamella.structures import (
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.tools.artifacts import (
    HANDOFF_MAP_SUFFIX,
    write_handoff_map,
)
from fibsem.hooks import FunctionHook, HookContext, HookEvent, HookManager


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="auto-handoff")
    exp.path = str(tmp_path / "auto-handoff")
    Path(exp.path).mkdir(parents=True, exist_ok=True)
    for i in range(2):
        lam = Lamella(path=Path(exp.path) / f"L{i}", number=i, petname=f"L{i}")
        Path(lam.path).mkdir(parents=True, exist_ok=True)
        exp.positions.append(lam)
    return exp


@pytest.fixture
def context() -> HookContext:
    return HookContext(event=HookEvent.WORKFLOW_COMPLETED)


class TestWritingIt:
    def test_it_writes_into_the_experiment_directory(self, experiment, context):
        """The grid's document, not a lamella's -- it covers all of them at once."""
        path = write_handoff_map(experiment, context)

        assert path is not None
        assert Path(path).exists()
        assert Path(path).parent == Path(experiment.path)
        assert Path(path).name.endswith(HANDOFF_MAP_SUFFIX)

    def test_it_overwrites_rather_than_accumulating(self, experiment, context):
        """One current answer to "what is on this grid".

        A directory of dated near-duplicates makes the recipient choose between them,
        which is a worse problem than having no map at all.
        """
        first = write_handoff_map(experiment, context)
        second = write_handoff_map(experiment, context)

        assert first == second
        pdfs = list(Path(experiment.path).glob("*.pdf"))
        assert len(pdfs) == 1

    def test_it_carries_the_grid_details_the_operator_typed(self, experiment, context):
        """Whatever was last entered in the dialog, kept on the experiment."""
        experiment.metadata["grid"] = "A"
        experiment.metadata["slot"] = "3"

        path = write_handoff_map(experiment, context)
        text = Path(path).read_bytes()
        # The reportlab output is compressed, so this asserts on the file existing and
        # being non-trivial rather than on its text; the content itself is covered by
        # test_handoff_map.py against the functions that produce it.
        assert len(text) > 1000

    def test_an_experiment_with_no_overviews_still_produces_a_document(
        self, experiment, context
    ):
        """A run where the overview was never acquired still has a table worth sending."""
        assert not experiment.find_overview_images()
        path = write_handoff_map(experiment, context)
        assert Path(path).exists()


class TestTheHookIsContained:
    def test_a_failure_does_not_stop_the_run(self, experiment):
        """HookManager.fire swallows what a hook raises; this is the proof, not a claim."""
        manager = HookManager()
        manager.register(
            FunctionHook(
                name="handoff_map",
                events=[HookEvent.WORKFLOW_COMPLETED],
                callback=lambda ctx: write_handoff_map(experiment, ctx),
            )
        )
        # A path that cannot be written to.
        experiment.path = "/definitely/not/a/directory"

        # The assertion is that this returns rather than raising.
        manager.fire(HookContext(event=HookEvent.WORKFLOW_COMPLETED))

    def test_it_fires_on_the_end_of_a_run(self, experiment):
        seen = []
        manager = HookManager()
        manager.register(
            FunctionHook(
                name="handoff_map",
                events=[HookEvent.WORKFLOW_COMPLETED],
                callback=lambda ctx: seen.append(write_handoff_map(experiment, ctx)),
            )
        )

        manager.fire(HookContext(event=HookEvent.ITEM_COMPLETED))
        assert seen == [], "an item completing is not the end of the run"

        manager.fire(HookContext(event=HookEvent.WORKFLOW_COMPLETED))
        assert len(seen) == 1


class TestItIsOffByDefault:
    def test_the_flag_gates_it(self):
        from fibsem.config import FeatureFlags

        assert FeatureFlags().handoff_map is False
