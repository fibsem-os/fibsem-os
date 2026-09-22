"""The Review inbox's filter row (FIB-994): a field matching name or task, a
Decided chip, and a menu for which kind of item and whether a later task waits.

An inbox of sixty rows is the case these exist for, so the fixture builds
several lamellae and two grids rather than one of each.
"""

import os

import pytest

pytest.importorskip("PyQt5")

import numpy as np  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem.applications.autolamella.proposals import (  # noqa: E402
    POINT_OF_INTEREST,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    Proposal,
    auto_author,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui import review_tab_widget as R  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.grid import (  # noqa: E402
    BeamOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (  # noqa: E402
    MillRoughTaskConfig,
)
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"
SEM = "SEM Overview"
FIB = "FIB Overview"
RUN = "run-1"
PIXELSIZE = 100e-9


def _image(beam: BeamType = BeamType.ION) -> FibsemImage:
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=beam, hfw=512 * PIXELSIZE),
        pixel_size=Point(PIXELSIZE, PIXELSIZE),
        microscope_state=MicroscopeState(stage_position=FibsemStagePosition()),
    )
    return FibsemImage(data=np.zeros((512, 512), dtype=np.uint8), metadata=metadata)


def _poi(task_id: str = RUN) -> Proposal:
    return Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={"task_id": task_id, "proposer": "centre-of-image"},
    )


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    """Three lamellae and two grids. Rough Milling waits on two of the
    lamellae; Setup is to check on the third. Grid-01's SEM overview waits and
    holds its FIB overview; Grid-02's FIB overview waits and holds nothing."""
    exp = Experiment(path=tmp_path, name="filter-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SETUP, required=True, attention=Attention.review_later
                ),
                AutoLamellaTaskDescription(name=ROUGH, required=True, requires=[SETUP]),
            ]
        )
    )
    os.makedirs(exp.path, exist_ok=True)
    for _ in range(3):
        exp.add_new_lamella(
            MicroscopeState(stage_position=FibsemStagePosition()),
            EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
        )
    for lamella in exp.positions:
        lamella.path.mkdir(parents=True, exist_ok=True)
    # named for what the filter tests type, not petnames: the fixture has to
    # be readable from the assertions
    exp.positions[0].name = "01-alpha-whale"
    exp.positions[1].name = "02-beta-whale"
    exp.positions[2].name = "03-gamma-otter"

    for lamella in exp.positions[:2]:
        lamella.task_history.append(
            AutoLamellaTaskState(
                name=ROUGH, status=AutoLamellaTaskStatus.AwaitingDecision
            )
        )
        lamella.proposals[ROUGH] = [
            Proposal(kind=TASK_RESULT, provenance={"task_id": RUN})
        ]
    # the third is a look, not a decision: it lands in "To check"
    third = exp.positions[2]
    third.task_history.append(
        AutoLamellaTaskState(name=SETUP, status=AutoLamellaTaskStatus.Completed)
    )
    proposal = _poi()
    proposal.decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author=auto_author("centre-of-image"),
            values={"poi": Point(0.0, 0.0)},
            via="workflow",
        )
    )
    third.proposals[SETUP] = [proposal]

    protocol = exp.grid_protocol
    protocol.add(
        BeamOverviewGridTaskConfig(task_name=SEM, attention=Attention.review_later)
    )
    protocol.add(
        BeamOverviewGridTaskConfig(
            task_name=FIB,
            orientation="FIB",
            requires=[SEM],
            attention=Attention.review_later,
        )
    )
    for name, task in (("Grid-01", SEM), ("Grid-02", FIB)):
        grid = exp.add_grid(GridRecord(name=name))
        directory = exp.grid_path(grid) / task
        directory.mkdir(parents=True)
        _image(BeamType.ELECTRON).save(str(directory / "overview"))
        grid.task_history.append(
            AutoLamellaTaskState(
                name=task, status=AutoLamellaTaskStatus.AwaitingDecision
            )
        )
        grid.proposals[task] = [
            Proposal(
                kind=TASK_RESULT,
                provenance={"task_id": RUN, "reference_image": f"{task}/overview.tif"},
            )
        ]
    return exp


@pytest.fixture
def warnings(monkeypatch) -> list:
    shown: list = []
    monkeypatch.setattr(
        R.QMessageBox,
        "warning",
        staticmethod(lambda _parent, _title, text: shown.append(text)),
    )
    return shown


@pytest.fixture
def tab(qapp, experiment, warnings) -> R.ReviewTabWidget:
    widget = R.ReviewTabWidget()
    widget.set_experiment(experiment)
    return widget


def _rows(tab) -> list:
    """The listed rows as text, without the group headers. Taken from the
    entries rather than by parsing summaries, which a header can look like."""
    listed = len(tab._entries)
    summaries = [t for t in tab.row_summaries() if t.count(" · ") >= 2]
    assert len(summaries) == listed, "one summary per entry"
    return summaries


def _names(tab) -> list:
    return [item.name for item, _task, _proposal, _state in tab._entries]


def test_the_unfiltered_inbox_lists_every_group(tab):
    texts = tab.row_summaries()
    assert texts[0] == "Waiting · 4", "two lamellae and two grids"
    assert len([t for t in texts if "To check" in t]) == 1
    assert sorted(_names(tab)) == [
        "01-alpha-whale",
        "02-beta-whale",
        "03-gamma-otter",
        "Grid-01",
        "Grid-02",
    ]


def test_typing_a_name_narrows_the_rows_and_the_count_says_so(tab):
    tab.filter_text.setText("whale")

    texts = tab.row_summaries()
    assert texts[0] == "Waiting · 2 of 4", "the count is of the match"
    assert _names(tab) == ["01-alpha-whale", "02-beta-whale"]
    assert "To check" not in " ".join(texts), "the otter's group is empty, so unlisted"


def test_typing_a_task_matches_the_task_not_only_the_name(tab):
    tab.filter_text.setText("rough")

    assert _names(tab) == ["01-alpha-whale", "02-beta-whale"]
    assert all("Rough Milling" in row for row in _rows(tab))


def test_every_word_has_to_appear_somewhere(tab):
    tab.filter_text.setText("beta rough")
    assert _names(tab) == ["02-beta-whale"]

    tab.filter_text.setText("beta setup")
    assert _names(tab) == [], "beta has no setup proposal, so no row matches both"


def test_the_filter_is_case_insensitive(tab):
    tab.filter_text.setText("GRID-01")
    assert _names(tab) == ["Grid-01"]


def test_grids_only_and_lamellae_only(tab):
    tab.filters.set_kind(R.KIND_GRIDS)
    assert _names(tab) == ["Grid-01", "Grid-02"]

    tab.filters.set_kind(R.KIND_LAMELLAE)
    assert sorted(_names(tab)) == ["01-alpha-whale", "02-beta-whale", "03-gamma-otter"]

    tab.filters.set_kind(R.KIND_ALL)
    assert len(_names(tab)) == 5


def test_held_only_keeps_what_a_later_task_waits_on(tab, experiment):
    """Grid-01's SEM overview holds its FIB overview; Grid-02's FIB overview
    holds nothing, and neither does a lamella's Rough Milling."""
    assert R.waiting_on(experiment, SEM, experiment.grids[0]) == [FIB]
    assert R.waiting_on(experiment, FIB, experiment.grids[1]) == []

    tab.filters.held_only.setChecked(True)
    tab.refresh()

    assert _names(tab) == ["Grid-01"], "only the decision something waits on"


def test_held_only_drops_a_result_that_was_already_applied(tab, experiment):
    """Rough Milling requires the otter's Setup, so a later task does require
    it -- but the producer applied that result itself and the run moved on, so
    nothing is held. Requiring alone is not holding."""
    otter = experiment.positions[2]
    assert R.waiting_on(experiment, SETUP, otter) == [ROUGH], "a later task requires it"
    assert otter.proposal(SETUP).to_check and not otter.proposal(SETUP).pending

    tab.filters.held_only.setChecked(True)
    tab.refresh()

    assert otter.name not in _names(tab)


def test_the_filters_stack(tab):
    tab.filters.set_kind(R.KIND_LAMELLAE)
    tab.filter_text.setText("whale")
    assert _names(tab) == ["01-alpha-whale", "02-beta-whale"]

    tab.filter_text.setText("otter")
    assert _names(tab) == ["03-gamma-otter"], "and the group follows the match"


def test_the_decided_chip_lists_decided_rows_and_the_filter_narrows_them(
    tab, experiment
):
    lamella = experiment.positions[0]
    experiment.decide(
        lamella.id,
        ROUGH,
        Decision(task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:op"),
    )
    tab.refresh()
    assert "Decided" not in " ".join(tab.row_summaries()), "hidden by default"

    tab.show_decided.setChecked(True)
    assert any("Decided · 1" in t for t in tab.row_summaries())

    tab.filter_text.setText("grid")
    assert "Decided" not in " ".join(tab.row_summaries()), (
        "the decided group is filtered like the others"
    )


def test_filtering_does_not_change_what_is_pending(tab):
    """The tab badge and the stall check read these counts. A narrowed list is
    a view; it must not make the run look like it has less to answer."""
    heard = []
    tab.pending_changed.connect(heard.append)
    assert tab.pending_count == 4 and tab.check_count == 1

    tab.filter_text.setText("grid")

    assert len(tab._entries) == 2, "the list is narrowed"
    assert tab.pending_count == 4, "but four proposals are still waiting"
    assert tab.check_count == 1
    assert heard == [4], "and the badge was not told otherwise"

    tab.filters.set_kind(R.KIND_GRIDS)
    tab.filter_text.setText("zzz")
    assert tab._entries == [] and tab.pending_count == 4


def test_a_filter_that_matches_nothing_says_so_rather_than_nothing_waiting(tab):
    tab.filter_text.setText("zzz")

    assert tab.row_summaries() == ["Nothing matches · 5 hidden"]
    assert tab.stack.currentWidget() is tab.empty
    assert tab.empty.text() == "No proposals match the filter."

    tab.filter_text.setText("")
    assert len(_names(tab)) == 5, "and clearing it brings them back"


def test_the_selection_is_kept_while_it_still_matches(tab, experiment):
    grid = experiment.grids[0]
    (index,) = [i for i, entry in enumerate(tab._entries) if entry[0] is grid]
    tab._select_entry(index)

    tab.filter_text.setText("grid")

    item, task_name, _proposal, _state = tab._entries[tab._current_index()]
    assert item is grid and task_name == SEM, "still on the row it was on"


def test_typing_does_not_confirm_or_reject_the_current_proposal(tab, experiment, qapp):
    """R and Return are shortcuts on this tab, and Qt dispatches a shortcut
    before the focused widget sees the key. Typing a filter term must not
    decide anything."""
    lamella = experiment.positions[0]
    tab.show()  # focus needs a shown widget, even offscreen
    qapp.processEvents()
    tab.filter_text.setFocus()
    qapp.processEvents()
    assert tab._typing(), "the field has the keyboard"

    tab._on_reject_shortcut()
    tab._on_confirm_shortcut()

    assert lamella.proposal(ROUGH).pending, "no decision was recorded"
    assert lamella.proposal(ROUGH).decisions == []

    # and the shortcuts still work when the field does not have the keys
    tab.list.setFocus()
    qapp.processEvents()
    assert not tab._typing()
    tab.close()


def test_the_filter_icon_takes_the_accent_while_it_narrows(tab):
    assert not tab.filters.narrowing
    assert tab.filters.toolTip() == "Filter by item or hold"

    tab.filters.set_kind(R.KIND_GRIDS)
    assert tab.filters.narrowing
    assert tab.filters.toolTip() == "Showing grids only"

    tab.filters.held_only.setChecked(True)
    tab.filters._paint()
    assert tab.filters.toolTip() == "Showing grids only, what holds a task"


def test_the_row_name_is_not_bold(tab):
    """A bold 13px name read as a heading; the dot carries the state."""
    row = None
    for i in range(tab.list.count()):
        widget = tab.list.itemWidget(tab.list.item(i))
        if isinstance(widget, R._InboxRow):
            row = widget
            break
    assert row is not None
    assert not row.name.font().bold()
    assert row.name.font().pixelSize() == 12
