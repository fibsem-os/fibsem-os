"""The device-built Demo images through its beams' commands.

``DemoMicroscope``'s ``acquire_image``, ``autocontrast``,
``auto_focus`` and live view go through the beam devices, whose driver runs the
Demo's own imaging code; ``test_microscope_contract.py`` pins what it images.
"""

import threading

import pytest

from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, ImageSettings
from tests.fixtures.microscopes import demo_microscope

AREA = FibsemRectangle(0.25, 0.25, 0.5, 0.5)


microscope = demo_microscope(scope="function")


@pytest.fixture(autouse=True)
def _stop_acquisition(microscope):
    yield
    microscope.stop_acquisition()


def _record(microscope, used):
    for beam in microscope.beams.values():
        for name in ("acquire", "autocontrast", "auto_focus"):
            original = getattr(beam, name)

            def wrapper(*args, _name=f"{beam.name}.{name}", _f=original, **kwargs):
                used.append(_name)
                return _f(*args, **kwargs)

            setattr(beam, name, wrapper)


def test_every_imaging_command_is_available(microscope):
    for beam in microscope.beams.values():
        for name in (
            "acquire",
            "autocontrast",
            "auto_focus",
            "start_live",
            "stop_live",
        ):
            assert beam.commands[name].available, (beam.name, name)


def test_the_old_imaging_methods_go_through_the_beams(microscope):
    used = []
    _record(microscope, used)
    sem, fib = microscope.beams[BeamType.ELECTRON], microscope.beams[BeamType.ION]

    settings = ImageSettings(
        beam_type=BeamType.ELECTRON, resolution=(384, 256), hfw=80e-6
    )
    image = microscope.acquire_image(settings)
    assert image.data.shape == (256, 384)
    assert image.metadata.image_settings.beam_type is BeamType.ELECTRON
    assert isinstance(microscope.acquire_image(beam_type=BeamType.ION), FibsemImage)
    microscope.autocontrast(BeamType.ELECTRON, AREA)
    microscope.auto_focus(BeamType.ION)
    assert used == [
        f"{sem.name}.acquire",
        f"{fib.name}.acquire",
        f"{sem.name}.autocontrast",
        f"{fib.name}.auto_focus",
    ]
    # each routine scans the full frame after, as before
    assert sem.scanning_mode.get_value().name.lower() == "full_frame"


def test_the_beam_acquires_with_its_current_settings(microscope):
    fib = microscope.beams[BeamType.ION]
    fib.resolution.set_value((768, 512))
    image = fib.acquire()
    assert image.data.shape == (512, 768)
    assert image.metadata.image_settings.beam_type is BeamType.ION


def test_live_view_runs_on_the_beam_and_its_frames_stay_there(microscope):
    """Live frames come on the beam's ``live_frame`` (and ``image_acquired``), and
    no longer on the microscope's SEM signal: the views take them from the beam."""
    sem = microscope.beams[BeamType.ELECTRON]
    pushed, acquired, relayed, done = [], [], [], threading.Event()

    def on_frame(image):
        pushed.append(image)
        if len(pushed) == 2:
            done.set()

    sem.live_frame.connect(on_frame)
    sem.image_acquired.connect(acquired.append)
    microscope.imaging.image_shown.connect(relayed.append)
    try:
        microscope.start_acquisition(BeamType.ELECTRON)
        assert sem.is_live and microscope.is_acquiring
        assert done.wait(10)
    finally:
        microscope.stop_acquisition()
        microscope.imaging.image_shown.disconnect(relayed.append)
    assert not sem.is_live and not microscope.is_acquiring
    assert len(pushed) >= 2 and all(isinstance(i, FibsemImage) for i in pushed)
    assert acquired[: len(pushed)] == pushed[: len(acquired)]
    assert relayed == []


def test_every_acquire_is_an_image_acquired_once(microscope):
    sem = microscope.beams[BeamType.ELECTRON]
    acquired = []
    sem.image_acquired.connect(acquired.append)
    image = sem.acquire()
    assert acquired == [image]


def test_a_disabled_beam_does_not_image(microscope):
    """A column switched off has no device, so it images nothing: it raises, as on
    the other backends, rather than running the Demo's imaging code without one."""
    from copy import deepcopy

    from fibsem.drivers.demo.microscope import DemoMicroscope

    system = deepcopy(microscope.system)
    system.ion.enabled = False
    electron_only = DemoMicroscope(system)
    assert BeamType.ION not in electron_only.beams
    for call in (
        lambda: electron_only.acquire_image(beam_type=BeamType.ION),
        lambda: electron_only.autocontrast(BeamType.ION),
        lambda: electron_only.auto_focus(BeamType.ION),
    ):
        with pytest.raises(ValueError, match="ION beam is not enabled"):
            call()
    assert isinstance(
        electron_only.acquire_image(beam_type=BeamType.ELECTRON), FibsemImage
    )


def test_a_live_view_that_acquires_each_frame_is_one_image_per_frame(microscope):
    """Thermo's live view takes each frame through ``acquire`` and then pushes it on
    ``live_frame``: that is one image, so ``image_acquired`` sees it once."""
    sem = microscope.beams[BeamType.ELECTRON]
    pushed, acquired, done = [], [], threading.Event()

    def live(stop):
        while not stop.is_set() and len(pushed) < 3:
            image = sem.acquire()
            pushed.append(image)
            sem.live_frame.emit(image)
        done.set()

    sem._live = live
    sem.image_acquired.connect(acquired.append)
    try:
        sem.start_live()
        assert done.wait(10)
    finally:
        sem.stop_live()
    assert acquired == pushed
