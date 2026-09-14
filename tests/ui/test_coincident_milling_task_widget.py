"""The Coincident Milling task widget (FIB-985): a write-through projection.

One monitoring control lands on every coincidence stage; a stage added in the
list gets the strategy and the current values; the config round-trips to the
same yaml shape the generic form produced.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy

import pytest
import yaml

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
    MILL_COINCIDENT_KEY,
    MillCoincidentTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
    SetupCoincidenceMillingTaskConfig,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategy
from fibsem.structures import FibsemRectangle, Point


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def microscope():
    from fibsem import config as cfg
    from fibsem import utils

    m, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    yield m
    m.disconnect()


@pytest.fixture()
def widget(qapp, microscope):
    from fibsem.applications.autolamella.ui.autolamella_coincident_milling_task_config_widget import (
        AutoLamellaCoincidentMillingTaskConfigWidget,
    )

    sources = [
        ChannelSettings(name="Reflection", excitation_wavelength=550),
        ChannelSettings(
            name="Red", excitation_wavelength=550, emission_wavelength="Fluorescence"
        ),
    ]
    w = AutoLamellaCoincidentMillingTaskConfigWidget(
        microscope=microscope, channel_sources=lambda: sources
    )
    w._test_sources = sources
    yield w
    w.close()


def _two_stage_config() -> MillCoincidentTaskConfig:
    config = MillCoincidentTaskConfig(task_name="Coincidence Milling")
    milling = config.milling[MILL_COINCIDENT_KEY]
    second = deepcopy(milling.stages[0])
    second.name = "Coincident Milling 02"
    second.pattern.scan_direction = "BottomToTop"
    milling.stages.append(second)
    for stage in milling.stages:
        stage.strategy.config.intensity_drop_fraction = 0.4
        stage.strategy.config.warmup_duration = 30.0
    return config


def _emitted(widget):
    received = []
    widget.settings_changed.connect(received.append)
    return received


def test_shows_the_monitoring_values_of_the_mill(widget, qapp):
    config = _two_stage_config()
    widget.set_task_config(config)
    qapp.processEvents()

    assert widget.spin_drop.value() == 40
    assert widget.spin_warmup.value() == 30.0
    assert widget.spin_confirm.value() == 10
    assert widget.spin_timeout.value() == pytest.approx(30.0)
    assert widget.channel_widget._channel.name == "Monitoring"
    # the stage list carries both stages; the strategy panel is hidden
    stages = widget._stages_widget().get_stages()
    assert [s.name for s in stages] == [
        "Coincident Milling 01",
        "Coincident Milling 02",
    ]
    assert not widget._stages_widget()._strategy_panel.isVisible()


def test_one_monitoring_value_writes_through_to_every_stage(widget, qapp):
    widget.set_task_config(_two_stage_config())
    received = _emitted(widget)

    widget.spin_drop.setValue(55)
    widget.spin_confirm.setValue(7)
    widget.spin_timeout.setValue(12.0)
    widget.spin_timelapse.setValue(3.0)
    qapp.processEvents()

    assert received
    config = received[-1]
    stages = config.milling[MILL_COINCIDENT_KEY].enabled_stages
    assert len(stages) == 2
    for stage in stages:
        assert isinstance(stage.strategy, CoincidenceMillingStrategy)
        assert stage.strategy.config.intensity_drop_fraction == pytest.approx(0.55)
        assert stage.strategy.config.consecutive_triggers == 7
        assert stage.strategy.config.timeout == 720
        assert stage.strategy.config.save_rate_limit == pytest.approx(3.0)


def test_a_stage_added_in_the_list_gets_the_strategy_and_the_values(widget, qapp):
    widget.set_task_config(_two_stage_config())
    widget.spin_drop.setValue(33)
    qapp.processEvents()

    # the list's own "+" adds a stage with the default (Standard) strategy
    stages_widget = widget._stages_widget()
    stages_widget._list._on_add_stage()
    qapp.processEvents()

    config = widget.get_task_config()
    stages = config.milling[MILL_COINCIDENT_KEY].stages
    assert len(stages) == 3
    for stage in stages:
        assert isinstance(stage.strategy, CoincidenceMillingStrategy)
        assert stage.strategy.config.intensity_drop_fraction == pytest.approx(0.33)


def test_roundtrip_keeps_the_yaml_shape(widget, qapp):
    original = _two_stage_config()
    before = yaml.safe_load(yaml.safe_dump(original.to_dict()))

    widget.set_task_config(original)
    after = yaml.safe_load(yaml.safe_dump(widget.get_task_config().to_dict()))

    assert set(after) == set(before)
    assert after["parameters"] == before["parameters"]
    assert after["monitoring_channel"] == before["monitoring_channel"]
    assert len(after["milling"][MILL_COINCIDENT_KEY]["stages"]) == 2
    for stage in after["milling"][MILL_COINCIDENT_KEY]["stages"]:
        assert stage["strategy"]["name"] == "CoincidenceMilling"


def test_copy_from_takes_the_filters_not_the_exposure(widget, qapp):
    widget.set_task_config(_two_stage_config())
    widget.spin_drop.setValue(40)
    before_exposure = widget.config.monitoring_channel.exposure_time
    received = _emitted(widget)

    widget._copy_channel(widget._test_sources[1])

    channel = widget.get_task_config().monitoring_channel
    assert channel.name == "Red"
    assert channel.emission_wavelength == "Fluorescence"
    assert channel.exposure_time == before_exposure
    assert received


def test_setup_record_is_shown_read_only(widget, qapp):
    widget.set_setup_record(None)
    assert "per site" in widget.label_setup.text()

    record = SetupCoincidenceMillingTaskConfig(task_name="Setup Coincidence Milling")
    widget.set_setup_record(record, "lamella-03")
    assert "Not set up" in widget.label_setup.text()

    record.objective_position = 2.418e-3
    record.fm_roi = FibsemRectangle(0.44, 0.38, 0.16, 0.24)
    record.pattern_offset = Point(1.2e-6, -0.4e-6)
    widget.set_setup_record(record, "lamella-03")
    text = widget.label_setup.text()
    assert "2.418 mm" in text
    assert "x 0.44 y 0.38" in text
    assert "+1.2 µm, -0.4 µm" in text


def test_advanced_writes_field_of_view_and_the_zstack_flag(widget, qapp):
    widget.set_task_config(_two_stage_config())
    widget.spin_fov.setValue(120.0)
    widget.chk_zstack.setChecked(False)

    config = widget.get_task_config()
    assert config.milling[MILL_COINCIDENT_KEY].field_of_view == pytest.approx(120e-6)
    assert config.acquire_fluorescence_images is False
