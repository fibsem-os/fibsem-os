"""The coating panels in Microscope tab > Sample, on the Arctis simulator."""

import os
import time

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

import fibsem.config as cfg
from fibsem import utils
from fibsem.services.coating import CoatingProgress, CoatingStatus
from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget
from fibsem.ui.widgets.coating_widget import CoatingWidget, progress_update


def _wait_until(app, condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        ),
    )
    return microscope


def test_the_sample_tab_shows_both_panels(qapp, arctis):
    widget = FibsemSampleWidget(microscope=arctis)
    coating = widget.coating_widget
    assert isinstance(coating, CoatingWidget)
    assert coating.sputter_panel is not None and coating.gis_panel is not None
    widget._teardown_connections()


def test_the_arctis_position_is_shown_not_edited(qapp, arctis):
    widget = CoatingWidget(microscope=arctis)
    assert widget.deposit_position.text() == "x 0 µm, y 0 µm, z −990 µm, tilt 10°"
    assert widget.gis_state.text() == "Retracted · Heater off · Closed"
    widget._teardown_connections()


def test_deposit_runs_the_service_command(qapp, arctis):
    widget = CoatingWidget(microscope=arctis)
    widget.deposit_seconds.setValue(2)
    widget.deposit_button.click()
    assert widget.deposit_button.text() == "Cancel"
    assert not widget.sputter_button.isEnabled()
    _wait_until(qapp, lambda: widget._running is None)
    assert widget.deposit_button.text() == "Deposit"
    assert arctis.coating.progress.get_value().status is CoatingStatus.FINISHED
    widget._teardown_connections()


def test_sputter_runs_the_service_command(qapp, arctis):
    widget = CoatingWidget(microscope=arctis)
    widget.sputter_seconds.setValue(4)
    widget.sputter_button.click()
    _wait_until(qapp, lambda: widget._running is None)
    assert arctis.devices["sputter_coater"].sim_runs[-1][0] == 4.0
    widget._teardown_connections()


def test_a_report_reads_as_its_step():
    running = progress_update(
        CoatingProgress(CoatingStatus.RUNNING, "Depositing", 12.0, 30.0)
    )
    assert running.message == "Depositing" and running.remaining_seconds == 12.0
    assert progress_update(CoatingProgress(CoatingStatus.FINISHED)).finished
    failed = progress_update(CoatingProgress(CoatingStatus.FAILED, error="stuck"))
    assert failed.is_failed and failed.message == "stuck"
