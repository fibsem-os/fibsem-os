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


def test_the_stage_table_writes_onto_the_live_stages(widget, qapp):
    widget.set_task_config(_two_stage_config())
    qapp.processEvents()
    table = widget.stage_table
    assert len(table._rows) == 2
    # the sections read as the rest of the app does
    assert widget.channel_panel is not None and widget.stop_panel is not None

    # edit the second row's width and depth through the table
    index, direction, current, width, height, depth, _ = table._rows[1]
    assert index.text() == "2"
    assert direction.isEnabled()  # a Rectangle has a scan direction
    assert direction.currentText() == "BottomToTop"
    width.setValue(12.0)
    depth.setValue(0.6)
    qapp.processEvents()

    stages = widget.get_task_config().milling[MILL_COINCIDENT_KEY].stages
    assert stages[1].pattern.width == pytest.approx(12e-6)
    assert stages[1].pattern.depth == pytest.approx(0.6e-6)
    assert stages[0].pattern.width == pytest.approx(9e-6)  # untouched
    # the full editor's list holds the same values
    live = widget._stages_widget().get_stages()[1]
    assert live.pattern.width == pytest.approx(12e-6)
    assert live.pattern.depth == pytest.approx(0.6e-6)


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
    assert widget._setup_fields["objective"].text() == "per site"
    assert "run it to set" in widget.label_setup.text()

    record = SetupCoincidenceMillingTaskConfig(task_name="Setup Coincidence Milling")
    widget.set_setup_record(record, "lamella-03")
    assert widget._setup_fields["objective"].text() == "—"
    assert "Not set up for lamella-03" in widget.label_setup.text()

    record.objective_position = 2.418e-3
    record.fm_roi = FibsemRectangle(0.44, 0.38, 0.16, 0.24)
    record.pattern_offset = Point(1.2e-6, -0.4e-6)
    widget.set_setup_record(record, "lamella-03")
    fields = widget._setup_fields
    assert fields["objective"].text() == "2.418 mm"
    assert "x 0.44  y 0.38" in fields["fm_roi"].text()
    assert "+1.2 µm, -0.4 µm" in fields["pattern_offset"].text()
    assert fields["drop"].text() == "40 % drop"
    assert all(f.isReadOnly() for f in fields.values())
    assert "run that task again" in widget.label_setup.text()


def test_advanced_writes_field_of_view_and_the_zstack_flag(widget, qapp):
    widget.set_task_config(_two_stage_config())
    widget.spin_fov.setValue(120.0)
    widget.chk_zstack.setChecked(False)

    config = widget.get_task_config()
    assert config.milling[MILL_COINCIDENT_KEY].field_of_view == pytest.approx(120e-6)
    assert config.acquire_fluorescence_images is False


def test_every_task_of_the_coincidence_protocol_builds_a_form(qapp):
    """Selecting a task in the protocol editor builds the generic parameter form
    for it. The Setup task's per-site fields (objective height None until the
    task records one) crashed that form; they are hidden from it now."""
    import os

    from PyQt5.QtWidgets import QGridLayout, QWidget

    from fibsem.applications.autolamella import config as autolamella_config
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
    from fibsem.applications.autolamella.ui.autolamella_task_config_widget import (
        build_parameter_rows,
    )
    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTaskConfig,
    )

    path = os.path.join(
        autolamella_config.BASE_PATH,
        "protocol",
        "development",
        "task-protocol-coincidence.yaml",
    )
    protocol = AutoLamellaTaskProtocol.load(path)
    host = QWidget()
    for name, config in protocol.task_config.items():
        grid = QGridLayout(QWidget(host))
        rows = build_parameter_rows(config, grid)
        fields = {row.field for row in rows}
        if isinstance(config, SetupCoincidenceMillingTaskConfig):
            assert "objective_position" not in fields, name
            assert "field_of_view" in fields and "intensity_drop_fraction" in fields
    host.close()


def test_loading_the_coincidence_tasks_does_not_warn_about_their_own_keys(caplog):
    import logging

    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTaskConfig,
    )

    setup = SetupCoincidenceMillingTaskConfig(task_name="Setup Coincidence Milling")
    mill = MillCoincidentTaskConfig(task_name="Coincidence Milling")
    with caplog.at_level(logging.WARNING):
        SetupCoincidenceMillingTaskConfig.from_dict(setup.to_dict())
        MillCoincidentTaskConfig.from_dict(mill.to_dict())
    assert "Unknown parameter" not in caplog.text
