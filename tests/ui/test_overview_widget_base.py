"""What both overview widgets do the same way, because they do it through one base:
picking a marked position, the right-click menu, the move gate, and letting go of the
microscope on close. Run against each widget."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem import config as fibsem_config
from fibsem import utils
from fibsem.imaging.overview_session import FLUORESCENCE, OverviewSession
from fibsem.structures.settings import FibsemStagePosition


def _beam(qapp):
    from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return FibsemOverviewWidget(OverviewSession(microscope))


def _fluorescence(qapp):
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget
    from tests.fixtures.fm_microscope import build_microscope

    microscope = build_microscope()
    microscope.fm.objective.insert()
    return FMOverviewWidget(OverviewSession(microscope, FLUORESCENCE))


BUILDERS = {"beam": _beam, "fluorescence": _fluorescence}


@pytest.fixture(params=sorted(BUILDERS))
def overview(request, qapp):
    widget = BUILDERS[request.param](qapp)
    widget.resize(1200, 800)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def toasts(monkeypatch):
    shown = []
    monkeypatch.setattr(
        "fibsem.ui.notification_service.show_toast",
        lambda message, *_, **__: shown.append(message),
    )
    return shown


def _mark_the_stage(qapp, widget, name="L1"):
    """Mark a position where the stage is, and return its canvas point."""
    here = widget.session.stage_position
    widget.set_positions(
        [
            FibsemStagePosition(
                name=name, x=here.x, y=here.y, z=here.z, r=here.r, t=here.t
            )
        ]
    )
    qapp.processEvents()
    return widget._frame().to_canvas(here)


def test_a_click_on_a_marked_position_selects_it(qapp, overview):
    x, y = _mark_the_stage(qapp, overview)
    heard = []
    overview.position_selected.connect(heard.append)

    overview._on_canvas_clicked(x, y)

    assert heard == ["L1"]
    assert overview.selected_position == "L1"


def test_the_menu_offers_a_new_position_and_moving_the_selected_one(qapp, overview):
    x, y = _mark_the_stage(qapp, overview)
    overview.set_selected_position("L1")
    moved = []
    overview.position_move_requested.connect(lambda name, _: moved.append(name))

    menu = overview._position_menu(x + 20.0, y)

    assert [action.label for action in menu.actions] == [
        overview.ADD_POSITION_LABEL,
        "Move Selected Position Here (L1)",
    ]
    menu.actions[1].callback()
    assert moved == ["L1"]


def test_nothing_moves_or_is_marked_during_a_run(qapp, overview, toasts, monkeypatch):
    x, y = _mark_the_stage(qapp, overview)
    started = []
    monkeypatch.setattr(
        "fibsem.ui.widgets.overview_base.FunctionWorker",
        lambda *args, **kwargs: started.append(args),
    )
    monkeypatch.setattr(type(overview), "is_acquiring", property(lambda _: True))

    overview.move_to(overview.session.stage_position)
    overview._on_canvas_double_clicked(x, y)

    assert started == []
    assert toasts == ["Cannot move the stage during an acquisition."] * 2
    assert overview._position_menu(x, y) is None


def test_a_move_outside_the_limits_is_refused(qapp, overview, toasts, monkeypatch):
    started = []
    monkeypatch.setattr(
        "fibsem.ui.widgets.overview_base.FunctionWorker",
        lambda *args, **kwargs: started.append(args),
    )
    here = overview.session.stage_position
    far = FibsemStagePosition(x=1.0, y=1.0, z=here.z, r=here.r, t=here.t)

    overview.move_to(far)

    assert started == []
    assert toasts == ["That position is outside the stage limits."]


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_closing_lets_go_of_the_microscope(qapp, kind):
    widget = BUILDERS[kind](qapp)
    microscope = widget.microscope
    signals = [
        widget.session.changed,
        microscope.stage.position.changed,
        microscope.sample_handling.holder_changed,
        microscope.imaging.tiled_progress,
    ]
    connected = [len(signal) for signal in signals]

    widget.close()

    assert [len(signal) for signal in signals] == [n - 1 for n in connected]
    widget.deleteLater()
    qapp.processEvents()
