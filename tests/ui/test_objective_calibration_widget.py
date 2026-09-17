"""The FM widget defers to the microscope configuration for the objective.

`fm-configuration.yaml` used to be applied to the objective after connect and written
back by autosave, so whatever the microscope configuration said was overwritten every
launch. Now the working file's positions apply only while the configuration is silent,
and stop being written once it is not.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.fm.structures import (  # noqa: E402
    ChannelSettings,
    FluorescenceConfiguration,
    OverviewParameters,
    ZParameters,
)
from fibsem.ui.widgets.fluorescence_control_widget import (  # noqa: E402
    FMControlWidget,
    _DemoHost,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def toasts(monkeypatch):
    """Toasts go to a list, not to the notification service.

    The service is a module-level QObject owned by whichever QApplication first
    created it; a later test module's teardown can delete it, and the next toast
    raises "wrapped C/C++ object has been deleted" from inside the widget under
    test. The widgets' own behaviour is what these tests are about.
    """
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda message, notification_type="info": shown.append(
            (message, notification_type)
        ),
    )
    return shown


@pytest.fixture()
def widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    host = _DemoHost()
    w = FMControlWidget(microscope=build_microscope(), parent=host)
    yield w
    w.close()
    w.deleteLater()


def _working_file(focus, limit) -> FluorescenceConfiguration:
    return FluorescenceConfiguration(
        channel_settings=[ChannelSettings(name="one", excitation_wavelength=365.0)],
        z_parameters=ZParameters(),
        overview_parameters=OverviewParameters(),
        focus_position=focus,
        limit_position=limit,
    )


def test_the_working_file_applies_while_the_configuration_is_silent(widget):
    assert widget.microscope.system.fm.focus_position is None
    widget._apply_fluorescence_configuration(_working_file(6.6e-3, 9.1e-3))
    assert widget.fm.objective.focus_position == pytest.approx(6.6e-3)
    assert widget.fm.objective.limit_position == pytest.approx(9.1e-3)


def test_the_working_file_cannot_override_a_configured_calibration(widget):
    """The defect the first attempt at this shipped with: the microscope-level test
    passed, and the app applied the working file a moment later."""
    widget.microscope.system.fm.focus_position = 7.0e-3
    widget.microscope.system.fm.limit_position = 8.0e-3
    widget.fm.objective.focus_position = 7.0e-3
    widget.fm.objective.limit_position = 8.0e-3

    widget._apply_fluorescence_configuration(_working_file(6.6e-3, 9.1e-3))

    assert widget.fm.objective.focus_position == pytest.approx(7.0e-3)
    assert widget.fm.objective.limit_position == pytest.approx(8.0e-3)


def test_the_working_file_stops_carrying_a_configured_calibration(widget):
    widget.fm.objective.focus_position = 6.6e-3
    widget.fm.objective.limit_position = 9.1e-3

    built = widget._build_fluorescence_configuration()
    assert built.focus_position == pytest.approx(6.6e-3)
    assert built.limit_position == pytest.approx(9.1e-3)

    widget.microscope.system.fm.focus_position = 7.0e-3
    widget.microscope.system.fm.limit_position = 8.0e-3
    built = widget._build_fluorescence_configuration()
    assert built.focus_position is None
    assert built.limit_position is None


def test_saving_the_calibration_writes_the_configuration(widget, tmp_path, monkeypatch):
    import yaml

    import fibsem.config as cfg
    from fibsem import utils

    path = tmp_path / "site.yaml"
    path.write_text(
        yaml.safe_dump(
            utils.load_yaml(
                os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
            )
        )
    )
    widget.microscope.configuration_path = str(path)
    widget.fm.objective.focus_position = 7.25e-3
    widget.fm.objective.limit_position = 8.75e-3

    widget.objectiveControlWidget.save_calibration()

    assert widget.microscope.system.fm.focus_position == pytest.approx(7.25e-3)
    written = utils.load_yaml(str(path))
    assert written["calibration"]["objective"]["focus_position"] == pytest.approx(
        7.25e-3
    )
    assert written["calibration"]["objective"]["limit_position"] == pytest.approx(
        8.75e-3
    )
