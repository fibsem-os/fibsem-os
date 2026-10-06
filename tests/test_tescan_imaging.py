"""Tescan imaging and the autofunctions as beam commands make the SDK calls they make today.

``TescanBeam``'s ``acquire``, ``last_image``, ``autocontrast``, ``auto_focus`` and live
view are ``TescanMicroscope``'s ``acquire_image``, ``last_image``, ``autocontrast``,
``auto_focus`` and acquisition worker moved onto the beam. Each case runs an old call on
a microscope connected as the app connects it, over a recording fake of the SDK
(``tests/fixtures/tescan_sdk.py``), and requires the result, the SDK calls in order and
the messages that the same call made on a microscope without beam devices, recorded in
``tests/fixtures/tescan_imaging_calls.json`` before the ``_get``/``_set`` branches that
path read through were deleted (FIB-1161). Nothing here has run on an instrument.
"""

import json
import logging
import os
import threading

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes import tescan as tescan_module
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    Point,
)
from tests.fixtures.tescan_sdk import connect

E, I = BeamType.ELECTRON, BeamType.ION
RECORDED = os.path.join(
    os.path.dirname(__file__), "fixtures", "tescan_imaging_calls.json"
)
AREA = FibsemRectangle(0.25, 0.25, 0.5, 0.5)


def _system():
    return utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system


@pytest.fixture
def connected(monkeypatch):
    """Connected as the app connects it, with its beam devices."""
    monkeypatch.setattr(tescan_module, "TESCAN_ELECTRON_TO_ION_SETTLE_TIME", 0)
    microscope, fake = connect(monkeypatch, _system())
    assert set(microscope.beams) == {E, I}
    # what the last image reported: the API cannot read these
    for beam_type in (E, I):
        cache = microscope._beam_parameters[beam_type]
        cache.resolution = [1536, 1024]
        cache.dwell_time = 1e-6
        cache.stigmation = Point(0.1, -0.2)
        cache.preset = "30 keV; 1 nA" if beam_type is I else None
    return microscope, fake


class _Messages(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append([record.levelname, record.getMessage()])


def _system_info(info):
    """The system info without the commit it ran at, which the recording can't match."""
    if info is None:
        return None
    ddict = info.to_dict()
    ddict.pop("fibsem_revision", None)
    return ddict


def _plain(value):
    if isinstance(value, FibsemImage):
        md = value.metadata
        return {
            "shape": list(value.data.shape),
            "settings": md.image_settings.to_dict(),
            "pixel_size": [md.pixel_size.x, md.pixel_size.y],
            "stage": md.microscope_state.stage_position.to_dict(),
            "system": _system_info(md.system_info),
        }
    if isinstance(value, ImageSettings):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _run(microscope, fake, call):
    handler = _Messages()
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    fake.log.clear()
    try:
        try:
            result = _plain(call(microscope))
        except Exception as e:  # a refusal is behaviour too
            result = f"raises {type(e).__name__}: {e}"
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    cache = {
        bt.name: [
            list(microscope._beam_parameters[bt].resolution or []),
            microscope._beam_parameters[bt].dwell_time,
        ]
        for bt in (E, I)
    }
    ran = {
        "result": result,
        "sdk": list(fake.log),
        "log": handler.messages,
        "cache": cache,
        "last_settings": _plain(microscope._last_imaging_settings),
    }
    # as the recording stored it: JSON, with anything else as its repr
    return json.loads(json.dumps(ran, default=repr))


def _settings(beam_type, reduced=False, hfw=80e-6):
    return ImageSettings(
        beam_type=beam_type,
        resolution=(768, 512),
        dwell_time=2e-7,
        hfw=hfw,
        reduced_area=FibsemRectangle(0.1, 0.2, 0.3, 0.4) if reduced else None,
        path="/data",
        filename="img",
    )


def _cases():
    for b in (E, I):
        n = b.name
        yield f"{n} acquire settings", lambda m, b=b: m.acquire_image(_settings(b))
        yield (
            f"{n} acquire same hfw",
            lambda m, b=b: m.acquire_image(
                _settings(b, hfw=m.get_field_of_view(beam_type=b))
            ),
        )
        yield (
            f"{n} acquire reduced",
            lambda m, b=b: m.acquire_image(_settings(b, reduced=True)),
        )
        yield f"{n} acquire current", lambda m, b=b: m.acquire_image(beam_type=b)
        other = I if b is E else E
        yield (
            f"{n} acquire both",  # settings win, as before
            lambda m, b=b, o=other: m.acquire_image(_settings(b), beam_type=o),
        )
        yield (
            f"{n} last image",
            lambda m, b=b: (m.acquire_image(_settings(b)), m.last_image(b))[1],
        )
        yield f"{n} last image none", lambda m, b=b: m.last_image(b)
        yield f"{n} autocontrast", lambda m, b=b: m.autocontrast(b)
        yield f"{n} autocontrast area", lambda m, b=b: m.autocontrast(b, AREA)
        yield f"{n} auto focus", lambda m, b=b: m.auto_focus(b)
    yield (
        "electron then ion",  # the settle before an ion image
        lambda m: [m.acquire_image(_settings(E)), m.acquire_image(_settings(I))],
    )
    yield "acquire nothing", lambda m: m.acquire_image()


CASES = dict(_cases())


with open(RECORDED) as f:
    EXPECTED = json.load(f)


def test_every_case_was_recorded():
    assert sorted(CASES) == sorted(EXPECTED)


@pytest.mark.parametrize("case", list(CASES))
def test_imaging_makes_the_same_calls_logs_and_result(connected, case):
    microscope, fake = connected
    assert _run(microscope, fake, CASES[case]) == EXPECTED[case]


def test_the_cases_make_sdk_calls():
    calls = [case["sdk"] for case in EXPECTED.values()]
    assert sum(1 for c in calls if c) >= len(CASES) * 0.7


def test_imaging_goes_through_the_beam_commands(connected):
    microscope, _ = connected
    used = []
    for beam in microscope.beams.values():
        for name in ("acquire", "last_image", "autocontrast", "auto_focus"):
            original = getattr(beam, name)

            def wrapper(*args, _name=f"{beam.name}.{name}", _f=original, **kw):
                used.append(_name)
                return _f(*args, **kw)

            setattr(beam, name, wrapper)
    sem, fib = microscope.beams[E], microscope.beams[I]
    microscope.acquire_image(_settings(E))
    microscope.acquire_image(beam_type=I)
    microscope.last_image(I)
    microscope.autocontrast(E)
    microscope.auto_focus(E)
    microscope.auto_focus(I)  # no ion focus: the old warning, not the beam
    assert used == [
        f"{sem.name}.acquire",
        f"{fib.name}.acquire",
        f"{fib.name}.last_image",
        f"{sem.name}.autocontrast",
        f"{sem.name}.auto_focus",
    ]


def test_the_ion_beam_has_no_auto_focus(connected):
    microscope, _ = connected
    sem, fib = microscope.beams[E], microscope.beams[I]
    for beam in (sem, fib):
        for name in ("acquire", "last_image", "autocontrast", "start_live"):
            assert beam.commands[name].available, (beam.name, name)
    assert sem.commands["auto_focus"].available
    assert not fib.commands["auto_focus"].available


@pytest.mark.parametrize("beam_type", [E, I])
def test_live_view_runs_on_the_beam_and_reaches_the_old_signal(connected, beam_type):
    microscope, fake = connected
    beam = microscope.beams[beam_type]
    seen, done = [], threading.Event()

    def on_frame(image):
        seen.append(image)
        if len(seen) == 3:
            done.set()

    signal = (
        microscope.sem_acquisition_signal
        if beam_type is E
        else microscope.fib_acquisition_signal
    )
    signal.connect(on_frame)
    try:
        microscope.start_acquisition(beam_type)
        assert beam.is_live and microscope.is_acquiring
        assert microscope._acquisition_thread is None  # the beam's, not the old
        assert done.wait(10)
    finally:
        microscope.stop_acquisition()
        signal.disconnect(on_frame)
    assert not beam.is_live and not microscope.is_acquiring
    column = "SEM" if beam_type is E else "FIB"
    assert all(isinstance(image, FibsemImage) for image in seen)
    assert len(fake.calls(f"{column}.Scan.AcquireImage")) >= 3
    # every SDK call of the loop held the connection lock
    assert not [p for p in fake.unlocked if p.endswith("AcquireImage")]
