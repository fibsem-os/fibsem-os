"""The hardware check against the Demo microscope's FM: every step is recorded, a failed
step does not stop the run, and the FM is left as it was found.

On the odemis METEOR simulator the same check ran through ``fibsem-cli check``; that is
recorded on the pull request, not here.
"""

import json
import os

import pytest

from fibsem import cli, utils
from fibsem import config as cfg
from fibsem.structures import RangeLimit
from fibsem.tools.hardware_check import (
    Level,
    Result,
    out_of_range,
    run_hardware_check,
    same,
)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml"),
        manufacturer="Demo",
        setup_logging=False,
    )
    yield microscope
    microscope.disconnect()


def _state(microscope):
    return {
        name: {key: p.get_value() for key, p in device.parameters.items()}
        for name, device in microscope.fm.devices.items()
    }


def _steps(report):
    return [(f.device, f.step) for f in report.findings]


def test_read_reads_every_part_and_changes_nothing(microscope):
    before = _state(microscope)

    report = run_hardware_check(microscope, Level.READ)

    assert report.passed, report.format()
    for name in ("camera", "light_source", "filter_set", "objective"):
        for key in microscope.fm.devices[name].parameters:
            assert (name, f"read {key}") in _steps(report)
    assert not any(step.startswith(("set ", "restore ")) for _, step in _steps(report))
    assert _state(microscope) == before


def test_settings_changes_each_setting_and_puts_it_back(microscope):
    before = _state(microscope)

    report = run_hardware_check(microscope, Level.SETTINGS)

    assert report.passed, report.format()
    for device, key in [
        ("camera", "exposure_time"),
        ("camera", "binning"),
        ("light_source", "power"),
        ("filter_set", "excitation_wavelength"),
        ("filter_set", "emission_filter"),
    ]:
        assert (device, f"set {key}") in _steps(report)
        assert (device, f"restore {key}") in _steps(report)
    assert ("camera", "resolution follows binning") in _steps(report)
    assert _state(microscope) == before


def test_no_level_moves_the_objective(microscope, monkeypatch):
    objective = microscope.fm.devices["objective"]
    moved = []
    for hook in ("_insert", "_retract", "_move_absolute", "_move_relative"):
        monkeypatch.setattr(objective, hook, lambda *a, h=hook: moved.append(h))

    run_hardware_check(microscope, Level.ACQUIRE)

    assert moved == []


def test_acquire_takes_one_frame_at_the_cameras_resolution(microscope):
    frames = []

    report = run_hardware_check(microscope, Level.ACQUIRE, on_frame=frames.append)

    assert report.passed, report.format()
    (frame,) = frames
    width, height = microscope.fm.devices["camera"].resolution.get_value()
    assert frame.data.shape[-2:] == (height, width)


def test_a_failed_read_is_recorded_and_the_run_goes_on(microscope, monkeypatch):
    power = microscope.fm.devices["light_source"].parameters["power"]

    def broken():
        raise TimeoutError("no answer from the light source")

    monkeypatch.setattr(power, "_read", broken)

    report = run_hardware_check(microscope, Level.READ)

    (failed,) = [f for f in report.findings if f.result is Result.FAIL]
    assert (failed.device, failed.step) == ("light_source", "read power")
    assert "no answer" in failed.detail
    assert ("objective", "read position") in _steps(report)  # after it
    assert not report.passed


def test_a_write_the_driver_ignores_fails_the_set_and_is_still_restored(
    microscope, monkeypatch
):
    exposure = microscope.fm.devices["camera"].parameters["exposure_time"]
    original = exposure.get_value()
    monkeypatch.setattr(exposure, "_write", lambda value: None)

    report = run_hardware_check(microscope, Level.SETTINGS)

    found = {f.step: f for f in report.findings if f.device == "camera"}
    assert found["set exposure_time"].result is Result.FAIL
    assert found["set exposure_time"].detail == "read back a different value"
    assert found["restore exposure_time"].result is Result.PASS
    assert exposure.get_value() == original


def test_without_an_fm_it_says_so_and_passes(microscope, monkeypatch):
    monkeypatch.setattr(microscope, "fm", None)

    report = run_hardware_check(microscope, Level.READ)

    assert report.passed
    assert [(f.device, f.result) for f in report.findings if f.device == "fm"] == [
        ("fm", Result.NOTE)
    ]


def test_every_other_device_has_its_readings_checked(microscope):
    report = run_hardware_check(microscope, Level.READ)

    assert report.passed, report.format()
    for name, device in microscope.devices.items():
        if name in microscope.fm.devices:
            continue
        for key in device.parameters:
            assert (name, f"read {key}") in _steps(report)


def test_a_device_adds_its_own_steps_through_its_hook(microscope, monkeypatch):
    stage = microscope.devices["stage"]
    levels = []

    def hardware_check(report, level):
        levels.append(level)
        with report.step("stage", "its own step"):
            pass

    monkeypatch.setattr(stage, "hardware_check", hardware_check)
    report = run_hardware_check(microscope, Level.SETTINGS)

    assert levels == [Level.SETTINGS]
    assert ("stage", "its own step") in _steps(report)
    assert ("stage", "read position") not in _steps(report)  # it replaced the default


def test_a_hook_that_raises_fails_and_the_run_goes_on(microscope, monkeypatch):
    def broken(report, level):
        raise RuntimeError("the driver's check is broken")

    monkeypatch.setattr(microscope.devices["stage"], "hardware_check", broken)
    report = run_hardware_check(microscope, Level.READ)

    (failed,) = [f for f in report.findings if f.result is Result.FAIL]
    assert (failed.device, failed.step) == ("stage", "check")
    assert any(f.device == "chamber" for f in report.findings)  # still checked


def test_the_report_records_when_it_ran_with_the_offset(microscope):
    report = run_hardware_check(microscope, Level.READ)

    saved = report.to_dict()
    assert saved["started_at"][-6] in "+-"  # ISO 8601 with its UTC offset
    assert saved["counts"]["pass"] == len(report.findings)
    json.dumps(saved)  # every value is plain


def test_out_of_range():
    assert out_of_range(0.5, RangeLimit(0.0, 1.0), None) == ""
    assert out_of_range(1.5, RangeLimit(0.0, 1.0), None) == "outside 0 to 1"
    assert out_of_range(450.00000000000006, None, [450.0, 550.0]) == ""
    assert out_of_range(500.0, None, [450.0, 550.0]) == "not one of its choices"


def test_acquire_needs_yes_from_the_command_line(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["hardware-check", "--level", "acquire"])
    assert exit.value.code == 2
    assert "--yes" in capsys.readouterr().err


def test_the_command_checks_through_the_app_server_and_fails_on_a_failed_step(
    microscope, monkeypatch, tmp_path, capsys
):
    """``fibsem-cli hardware-check`` checks the server's devices, as the operator."""
    pytest.importorskip("uvicorn")
    from fibsem.server import AuthConfig, build_server
    from tests.fixtures.app_server import serving

    auth = AuthConfig.generate(level="watch", token="agent")
    auth.operator_token = "operator"
    argv = [
        "hardware-check",
        "--level",
        "settings",
        "--json",
        str(tmp_path / "report.json"),
    ]
    with serving(build_server(microscope, auth=auth)) as url:
        monkeypatch.setenv("FIBSEM_SERVER_URL", url)
        monkeypatch.setenv("FIBSEM_OPERATOR_TOKEN", "operator")
        with pytest.raises(SystemExit) as exit:
            cli.main(argv)
        assert exit.value.code == 0, capsys.readouterr()
        assert "0 failed" in capsys.readouterr().out
        assert json.loads((tmp_path / "report.json").read_text())["passed"] is True

        power = microscope.fm.devices["light_source"].parameters["power"]
        monkeypatch.setattr(power, "_write", lambda value: None)
        with pytest.raises(SystemExit) as exit:
            cli.main(argv)
        assert exit.value.code == 1


def test_same_compares_small_si_values_relatively():
    """A pixel size in metres is ~1e-7: an absolute tolerance would call 119 nm and
    238 nm the same, which a report on the odemis sim did."""
    assert not same(119e-9, 238e-9)
    assert not same((119e-9, 119e-9), (238e-9, 238e-9))
    assert same(1.19e-7, 1.19e-7 * (1 + 1e-9))
    assert same(0.0, 0.0)


def _lagging_pixel_size(microscope, monkeypatch, stale_reads):
    """The camera's pixel size answers with the old value for *stale_reads* reads
    after a binning change, as odemis does for ~2 ms while it republishes it."""
    camera = microscope.fm.devices["camera"]
    pixel_size, binning = camera.parameters["pixel_size"], camera.parameters["binning"]
    read_pixel_size, write_binning = pixel_size._read, binning._write
    state = {"stale": None, "left": 0}

    def write(value):
        state["stale"], state["left"] = read_pixel_size(), stale_reads
        write_binning(value)

    def read():
        if state["left"] > 0:
            state["left"] -= 1
            return state["stale"]
        return read_pixel_size()

    monkeypatch.setattr(binning, "_write", write)
    monkeypatch.setattr(pixel_size, "_read", read)


def _finding(report, step):
    (found,) = [f for f in report.findings if f.step == step]
    return found


def test_a_pixel_size_that_catches_up_is_a_note_with_how_long(microscope, monkeypatch):
    _lagging_pixel_size(microscope, monkeypatch, stale_reads=2)

    report = run_hardware_check(microscope, Level.SETTINGS)

    found = _finding(report, "pixel size follows binning")
    assert found.result is Result.NOTE
    assert "followed after" in found.detail
    assert report.passed


def test_a_pixel_size_that_never_follows_fails(microscope, monkeypatch):
    from fibsem.tools.hardware_check import fm as fm_check

    monkeypatch.setattr(fm_check, "FOLLOW_WAIT", 0.05)
    _lagging_pixel_size(microscope, monkeypatch, stale_reads=10**6)

    report = run_hardware_check(microscope, Level.SETTINGS)

    assert _finding(report, "pixel size follows binning").result is Result.FAIL
    assert not report.passed


def test_lengths_read_in_the_unit_that_suits_them():
    from fibsem.tools.hardware_check import with_unit

    assert str(with_unit(1.69e-3, "m")) == "1.69 mm"
    assert str(with_unit((1.19e-7, 1.19e-7), "m")) == "119 × 119 nm"
    assert str(with_unit(0.0, "m")) == "0 mm"
