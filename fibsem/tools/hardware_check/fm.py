"""The fluorescence microscope's steps in the hardware check
(:mod:`fibsem.tools.hardware_check`).

What each level does here:

- ``READ``: whether the FM answers, then every parameter of the camera, light source,
  filter set and objective, against its own limits and choices. The objective's
  position, state and limits are read and never changed.
- ``SETTINGS``: exposure, binning (and that the resolution and pixel size follow it),
  gain where the camera has one, power, excitation and emission, each changed, read
  back and put back. These are settings for the next exposure, so nothing lights or
  moves.
- ``ACQUIRE``: one frame with the settings as they are, its size checked against the
  camera's resolution.

Nothing here moves the objective: insert, retract and focus moves need a decision
about when that is safe with a sample under it.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from fibsem.tools.hardware_check import (
    Finding,
    Level,
    Report,
    Result,
    check_readings,
    check_round_trip,
    other_choice,
    same,
    with_unit,
)

PARTS = ("camera", "light_source", "filter_set", "objective")


def check_fm(
    fm: Any,
    report: Report,
    level: Level,
    on_frame: Optional[Callable[[Any], None]] = None,
) -> None:
    devices = fm.devices
    group = devices.get("fm")
    if group is not None and hasattr(group, "check_health"):
        with report.step("fm", "answers") as finding:
            problem = group.check_health()
            if problem:
                finding.result = Result.FAIL
                finding.detail = str(problem)

    for name in PARTS:
        device = devices.get(name)
        if device is None:
            report.add(_note(name, "present", "this FM has none"))
            continue
        check_readings(device, report)

    if level.includes(Level.SETTINGS):
        _check_settings(devices, report)
    if level.includes(Level.ACQUIRE):
        _check_frame(fm, report, on_frame)


def _check_settings(devices: Any, report: Report) -> None:
    camera = devices.get("camera")
    if camera is not None:
        exposure = camera.parameters["exposure_time"]
        current = exposure.get_value()
        doubled = current * 2
        limits = exposure.limits
        if limits is not None and doubled > limits.max:
            doubled = current / 2
        check_round_trip(camera, "exposure_time", doubled, report)
        _check_binning(camera, report)
        if "gain" in camera.parameters and camera.parameters["gain"].settable:
            gain = camera.parameters["gain"].get_value()
            check_round_trip(camera, "gain", 0.25 if gain > 0.5 else 0.75, report)

    light = devices.get("light_source")
    if light is not None:
        power = light.parameters["power"].get_value()
        check_round_trip(light, "power", 0.25 if power > 0.5 else 0.75, report)

    filters = devices.get("filter_set")
    if filters is not None:
        for name in ("excitation_wavelength", "emission_filter"):
            parameter = filters.parameters[name]
            value = other_choice(parameter.choices, parameter.get_value())
            if value is None:
                report.add(_note(filters.name, f"set {name}", "only one choice"))
                continue
            check_round_trip(filters, name, value, report)


def _check_binning(camera: Any, report: Report) -> None:
    """Change the binning, and check the resolution and pixel size follow it."""
    binning = camera.parameters["binning"]
    original = binning.get_value()
    other = other_choice(binning.choices, original)
    if other is None:
        report.add(_note(camera.name, "set binning", "only one choice"))
        return
    resolution = camera.parameters["resolution"].get_value()
    pixel_size = camera.parameters["pixel_size"].get_value()
    check_round_trip(camera, "binning", other, report)
    # the round trip has put it back: change it again to look at what follows it
    try:
        binning.set_value(other)
        ratio = original / other
        _check_follows(
            camera,
            "pixel_size",
            with_unit(tuple(p / ratio for p in pixel_size), "m"),
            report,
        )
        _check_follows(
            camera, "resolution", tuple(round(r * ratio) for r in resolution), report
        )
    finally:
        binning.set_value(original)


FOLLOW_WAIT = 1.0  # seconds a dependent value gets to catch up


def _check_follows(camera: Any, name: str, expected: Any, report: Report) -> None:
    """Whether *name* follows the binning just set. Read at once, as code that sets
    the binning and then uses it would; a value that catches up later is a note
    saying how long it took, one that never does in FOLLOW_WAIT a failure."""
    want = expected.value if hasattr(expected, "value") else expected
    unit = getattr(expected, "unit", None)
    parameter = camera.parameters[name]
    with report.step(
        camera.name, f"{name.replace('_', ' ')} follows binning", expected
    ) as finding:
        start = time.perf_counter()
        found = parameter.get_value()
        if same(found, want, 1e-3):
            finding.read = with_unit(found, unit)
            return
        while time.perf_counter() - start < FOLLOW_WAIT:
            time.sleep(0.001)
            found = parameter.get_value()
            if same(found, want, 1e-3):
                waited = (time.perf_counter() - start) * 1e3
                finding.read = with_unit(found, unit)
                finding.result = Result.NOTE
                finding.detail = (
                    f"stale when read at once; followed after {waited:.0f} ms"
                )
                return
        finding.read = with_unit(found, unit)
        finding.result = Result.FAIL
        finding.detail = f"not followed after {FOLLOW_WAIT:g} s"


def _check_frame(fm: Any, report: Report, on_frame: Optional[Callable]) -> None:
    camera = fm.devices.get("camera")
    expected = None
    if camera is not None:
        width, height = camera.parameters["resolution"].get_value()
        expected = (height, width)
    with report.step("camera", "acquire a frame", expected) as finding:
        image = fm.acquire_image()
        data = image.data
        finding.read = tuple(data.shape[-2:])
        finding.detail = f"{data.dtype}, mean {float(data.mean()):.1f}"
        if expected is not None and tuple(data.shape[-2:]) != expected:
            finding.result = Result.FAIL
            finding.detail = f"not the camera's resolution; {finding.detail}"
        if on_frame is not None:
            on_frame(image)


def _note(device: str, step: str, detail: str) -> Finding:
    return Finding(device, step, Result.NOTE, detail=detail)
