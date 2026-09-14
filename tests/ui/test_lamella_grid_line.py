"""A lamella's row names the grid it is on, and withholds the stage actions while
that grid is off the stage.

The card, the Workflow row and the Experiment-list row put the grid's name ahead
of the status, accent while that grid is on the stage and muted while it is not,
and only when the experiment has more than one grid. The context comes from the
window, built off the experiment's records and the stage's last inventory read.
Move-to and Update are withheld while the grid is off the stage: they would act
on whatever grid *is* there. The defect icon is drawn only once there is a defect.
"""

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    DefectState,
    DefectType,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.lamella_card_widget import (
    LamellaCardContainer,
)
from fibsem.applications.autolamella.ui.lamella_list_widget import (
    LamellaListWidget,
    grid_of,
    grids_named,
    not_on_stage_reason,
    short_status,
)
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
)
from fibsem.structures import FibsemStagePosition, MicroscopeState
from fibsem.ui.tokens import ACCENT_COLOR, NEUTRAL_550

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def tidy(destroy_widgets_after_test):
    """The bare widgets are torn down properly at the end of their test. Not
    for the window test, whose fixture closes the window itself."""
    yield


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="rows")
    (tmp_path / "rows").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    return exp


@pytest.fixture
def lamellae(experiment):
    oak = experiment.add_grid(GridRecord(name="grid-oak"))
    elm = experiment.add_grid(GridRecord(name="grid-elm"))
    config = experiment.task_protocol.task_config
    for name, grid in (("on-oak", oak), ("on-elm", elm), ("unlinked", None)):
        experiment.add_new_lamella(
            microscope_state=MicroscopeState(),
            task_config=config,
            name=name,
            grid_id=grid.id if grid else None,
        )
    return {p.name: p for p in experiment.positions}


def _context(experiment, loaded):
    return {g.id: (g.name, g.name in loaded) for g in experiment.grids}


def _colour(label) -> str:
    return (
        "accent"
        if ACCENT_COLOR in label.styleSheet()
        else ("muted" if NEUTRAL_550 in label.styleSheet() else "?")
    )


class TestTheHelpers:
    def test_grid_of(self, experiment, lamellae):
        context = _context(experiment, {"grid-oak"})
        assert grid_of(lamellae["on-oak"], context) == ("grid-oak", True)
        assert grid_of(lamellae["on-elm"], context) == ("grid-elm", False)
        assert grid_of(lamellae["unlinked"], context) is None
        assert grid_of(lamellae["on-oak"], None) is None

    def test_grids_are_named_only_when_there_is_more_than_one(self):
        assert not grids_named(None)
        assert not grids_named({"a": ("grid-oak", True)})
        assert grids_named({"a": ("grid-oak", True), "b": ("grid-elm", False)})

    def test_reason(self):
        assert not_on_stage_reason(None) == ""
        assert not_on_stage_reason(("grid-oak", True)) == ""
        assert (
            not_on_stage_reason(("grid-elm", False)) == "grid-elm is not on the stage"
        )

    def test_short_status_drops_the_time(self):
        assert short_status("Rough Milling (02:13PM)") == "Rough Milling"
        assert short_status("Rough Milling") == "Rough Milling"
        assert short_status("") == ""


class TestTheCard:
    def test_grid_and_stage_actions_follow_the_context(
        self, experiment, lamellae, tidy
    ):
        container = LamellaCardContainer(columns=1)
        cards = {n: container.add_lamella(p) for n, p in lamellae.items()}
        # No context yet: no grid named, nothing withheld.
        assert all(not c._grid_label.isVisibleTo(c) for c in cards.values())
        assert cards["on-oak"]._action_move.isEnabled()

        container.set_grid_context(_context(experiment, {"grid-oak"}))
        oak, elm, none = cards["on-oak"], cards["on-elm"], cards["unlinked"]
        assert (
            oak._grid_label.text() == "grid-oak"
            and _colour(oak._grid_label) == "accent"
        )
        assert oak._action_move.isEnabled()
        assert oak._action_move.text() == "Move to Position"
        assert (
            elm._grid_label.text() == "grid-elm" and _colour(elm._grid_label) == "muted"
        )
        assert elm._grid_label.toolTip() == "grid-elm is not on the stage"
        assert not elm._action_move.isEnabled()
        assert not elm._action_update.isEnabled()
        assert elm._action_move.text() == (
            "Move to Position (grid-elm is not on the stage)"
        )
        assert not none._grid_label.isVisibleTo(none)
        assert none._action_move.isEnabled()

        # The exchange: elm onto the stage, oak back to the magazine.
        container.set_grid_context(_context(experiment, {"grid-elm"}))
        assert elm._action_move.isEnabled() and not oak._action_move.isEnabled()
        assert _colour(oak._grid_label) == "muted"
        # And a card added afterwards gets the context it missed.
        late = container.add_lamella(lamellae["on-oak"])
        assert late._grid_label.text() == "grid-oak"

    def test_one_grid_is_not_named(self, experiment, lamellae, tidy):
        container = LamellaCardContainer(columns=1)
        card = container.add_lamella(lamellae["on-elm"])
        elm = experiment.get_grid_by_name("grid-elm")
        container.set_grid_context({elm.id: ("grid-elm", False)})
        assert not card._grid_label.isVisibleTo(card)
        # ... but the stage actions are still withheld while it is off the stage.
        assert not card._action_move.isEnabled()

    def test_the_defect_icon_only_once_there_is_a_defect(
        self, experiment, lamellae, tidy
    ):
        container = LamellaCardContainer(columns=1, mode="standard")
        card = container.add_lamella(lamellae["on-oak"])
        assert not card._btn_defect.isVisibleTo(card)
        lamellae["on-oak"].defect = DefectState(state=DefectType.REWORK)
        assert card._btn_defect.isVisibleTo(card)
        assert card._btn_defect.toolTip().startswith("Rework")
        # Set back from the actions menu's Defect submenu.
        none = next(a for a in card._defect_menu.actions() if a.text() == "No defect")
        none.trigger()
        assert lamellae["on-oak"].defect.state is DefectType.NONE
        assert not card._btn_defect.isVisibleTo(card)

    def test_compact_drops_the_time_and_the_grid_survives_a_mode_switch(
        self, experiment, lamellae, tidy
    ):
        container = LamellaCardContainer(columns=1, mode="cozy")
        lamella = lamellae["on-oak"]
        card = container.add_lamella(lamella)
        container.set_grid_context(_context(experiment, {"grid-oak"}))
        assert card._status_label.text() == ""
        for mode in ("standard", "compact", "cozy"):
            container.set_mode(mode)
            assert card._grid_label.text() == "grid-oak"
            assert card._grid_label.parent() is not None


class TestTheRows:
    def test_the_workflow_row(self, experiment, lamellae, tidy):
        widget = LamellaListWidget()
        widget.set_grid_context(_context(experiment, {"grid-oak"}))
        widget.set_lamellae(list(lamellae.values()))
        rows = {widget._row(i).lamella.name: widget._row(i) for i in range(3)}
        assert rows["on-oak"].grid_label.text() == "grid-oak"
        assert _colour(rows["on-oak"].grid_label) == "accent"
        assert rows["on-elm"].grid_label.toolTip() == "grid-elm is not on the stage"
        assert not rows["unlinked"].grid_label.isVisibleTo(rows["unlinked"])
        assert not rows["on-oak"].btn_defect.isVisibleTo(rows["on-oak"])
        assert [a.text() for a in rows["on-oak"].btn_actions.menu().actions()] == [
            "Edit",
            "Remove",
            "Defect",
        ]
        widget.set_grid_context(None)
        assert not rows["on-oak"].grid_label.isVisibleTo(rows["on-oak"])

    def test_the_experiment_list_row(self, experiment, lamellae, tidy):
        widget = LamellaNameListWidget()
        widget.enable_move_to_action(True)
        widget.enable_update_action(True)
        widget.enable_remove_button(True)
        widget.set_grid_context(_context(experiment, {"grid-oak"}))
        widget.set_lamella(list(lamellae.values()))
        rows = {r.lamella.name: r for r in widget._rows()}
        oak, elm = rows["on-oak"], rows["on-elm"]
        assert oak.grid_label.text() == "grid-oak"
        assert oak.action_move_to.isEnabled()
        assert not elm.action_move_to.isEnabled()
        assert not elm.action_update.isEnabled()
        assert elm.action_move_to.text() == (
            "Move to Position (grid-elm is not on the stage)"
        )
        # Four inline buttons became one menu; only what the host enabled shows.
        assert oak.btn_actions.isVisibleTo(oak)
        assert not oak.action_edit.isVisible()
        assert oak.action_remove.isVisible()
        widget.set_grid_context(_context(experiment, {"grid-elm"}))
        assert elm.action_move_to.isEnabled() and not oak.action_move_to.isEnabled()


class TestFromTheWindow:
    @pytest.fixture
    def main_ui(self, qapp, monkeypatch, tmp_path):
        import fibsem.config as fibsem_config
        from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

        window = module.AutoLamellaSingleWindowUI()
        ui = window.autolamella_ui
        config = os.path.join(
            os.path.dirname(fibsem_config.__file__),
            "config",
            "sim-arctis-configuration.yaml",
        )
        monkeypatch.setattr(
            ui.system_widget,
            "load_configuration",
            lambda configuration_name=None: config,
        )
        ui.system_widget.connect_to_microscope()
        yield window
        ui.microscope.disconnect()
        original_quit = qapp.quit
        qapp.quit = lambda: None
        try:
            window.close()
        finally:
            qapp.quit = original_quit
        # Destroyed here, on purpose, rather than left to the collector: a
        # window this size collected during a later test's `processEvents`
        # took that test down with a segfault.
        from PyQt5.QtCore import QEvent

        window.deleteLater()
        qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        del window
        gc.collect()

    def test_a_load_from_the_grids_tab_reaches_every_lamella_display(
        self, main_ui, tmp_path
    ):
        ui = main_ui.autolamella_ui
        stage = ui.microscope._stage
        exp = Experiment(path=tmp_path, name="exp")
        (tmp_path / "exp").mkdir()
        exp.task_protocol = AutoLamellaTaskProtocol()
        ui.experiment = exp
        exp.sync_grids_from_inventory(stage)  # several grids: names are shown
        grid = exp.get_grid_by_name("Grid-02")
        ui.add_new_lamella(
            stage_position=FibsemStagePosition(x=0, y=0, z=0, r=0, t=0),
            name="on-two",
            grid_id=grid.id,
        )
        main_ui._on_experiment_update()
        main_ui.grids_tab.set_experiment(exp)

        card = main_ui.lamella_card_container._cards[exp.positions[0].id]
        row = main_ui.lamella_list_widget._row(0)
        assert card._grid_label.text() == "Grid-02"
        assert _colour(card._grid_label) == "muted"
        assert not card._action_move.isEnabled()
        assert row.grid_label.toolTip() == "Grid-02 is not on the stage"

        # The load, the way a card's Load action does it: synchronous here.
        main_ui.grids_tab._synchronous = True
        main_ui.grids_tab._on_load(grid)
        assert stage.loaded_grids[0].name == "Grid-02"
        assert card._action_move.isEnabled()
        assert _colour(card._grid_label) == "accent"
        assert main_ui.lamella_list_widget._row(0).grid_label.toolTip() == (
            "Grid-02 is on the stage"
        )
