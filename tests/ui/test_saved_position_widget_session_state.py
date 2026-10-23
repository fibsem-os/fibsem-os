"""The Saved Positions panel reads and writes the configuration's session state."""

import os

import pytest

pytest.importorskip("PyQt5")

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.saved_positions import load_saved_positions  # noqa: E402
from fibsem.session_state import session_state_for  # noqa: E402
from fibsem.structures import FibsemStagePosition  # noqa: E402

SHIPPED = os.path.join(cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=SHIPPED, manufacturer="Demo", setup_logging=False
    )
    yield microscope
    microscope.disconnect()


def _panel(microscope=None):
    from fibsem.ui.widgets.saved_position_widget import SavedPositionListWidget

    return SavedPositionListWidget(microscope=microscope)


def test_adding_a_position_saves_it_with_the_configuration(qapp, microscope):
    panel = _panel(microscope)

    panel.add_position(FibsemStagePosition(name="cryo", x=1e-3, y=0, z=0, r=0, t=0))

    saved = load_saved_positions(session_state_for(microscope))
    assert [p.name for p in saved] == ["cryo"]


def test_a_new_panel_shows_what_was_saved(qapp, microscope):
    _panel(microscope).add_position(FibsemStagePosition(name="cryo"))

    assert [p.name for p in _panel(microscope).get_positions()] == ["cryo"]


def test_without_a_microscope_nothing_is_written(qapp):
    """The movement widget builds the panel before it is connected."""
    import fibsem.session_state as session_state

    panel = _panel()

    panel.add_position(FibsemStagePosition(name="cryo"))

    assert panel.get_positions()[0].name == "cryo"  # still shown
    directory = session_state.SESSION_STATE_DIRECTORY
    assert not os.path.exists(directory) or os.listdir(directory) == []


def test_the_panel_s_edits_go_through_the_stage_movement_service(qapp, microscope):
    panel = _panel(microscope)
    service = microscope.stage_movement

    panel._on_add_clicked()
    panel._on_add_clicked()
    first, second = panel.get_positions()
    panel._on_row_name_changed(first, "cryo")
    panel._on_row_name_changed(second, "cryo")  # refused: taken

    assert [p.name for p in service.saved_positions()] == ["cryo", "Position 01"]
    assert [p.name for p in panel.get_positions()] == ["cryo", "Position 01"]
    panel.remove_position(panel.get_positions()[1])
    assert [p.name for p in service.saved_positions()] == ["cryo"]
