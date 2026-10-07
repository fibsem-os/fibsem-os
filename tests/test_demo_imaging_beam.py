"""The device-built Demo images through its beams' commands.

``DemoMicroscope``'s ``acquire_image``, ``last_image``, ``autocontrast``,
``auto_focus`` and live view go through the beam devices, whose driver runs the
Demo's own imaging code; ``test_microscope_contract.py`` pins what it images.
"""

import threading

import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, ImageSettings

AREA = FibsemRectangle(0.25, 0.25, 0.5, 0.5)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    yield microscope
    microscope.stop_acquisition()


def _record(microscope, used):
    for beam in microscope.beams.values():
        for name in ("acquire", "last_image", "autocontrast", "auto_focus"):
            original = getattr(beam, name)

            def wrapper(*args, _name=f"{beam.name}.{name}", _f=original, **kwargs):
                used.append(_name)
                return _f(*args, **kwargs)

            setattr(beam, name, wrapper)


def test_every_imaging_command_is_available(microscope):
    for beam in microscope.beams.values():
        for name in (
            "acquire",
            "last_image",
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
    assert isinstance(microscope.last_image(BeamType.ION), FibsemImage)
    microscope.autocontrast(BeamType.ELECTRON, AREA)
    microscope.auto_focus(BeamType.ION)
    assert used == [
        f"{sem.name}.acquire",
        f"{fib.name}.acquire",
        f"{fib.name}.last_image",
        f"{sem.name}.autocontrast",
        f"{fib.name}.auto_focus",
    ]
    # each routine scans the full frame after, as before
    assert sem.scanning_mode.get_value().name.lower() == "full_frame"


def test_the_beam_acquires_with_its_current_settings(microscope):
    fib = microscope.beams[BeamType.ION]
    fib.resolution.set_value((512, 256))
    image = fib.acquire()
    assert image.data.shape == (256, 512)
    assert image.metadata.image_settings.beam_type is BeamType.ION


def test_live_view_runs_on_the_beam_and_reaches_the_old_signal(microscope):
    sem = microscope.beams[BeamType.ELECTRON]
    pushed, seen, done = [], [], threading.Event()
    sem.live_frame.connect(lambda image: pushed.append(image))

    def on_frame(image):
        seen.append(image)
        if len(seen) == 2:
            done.set()

    microscope.sem_acquisition_signal.connect(on_frame)
    microscope.start_acquisition(BeamType.ELECTRON)
    assert sem.is_live and microscope.is_acquiring
    assert done.wait(10)
    microscope.stop_acquisition()
    assert not sem.is_live and not microscope.is_acquiring
    assert len(pushed) >= 2 and all(isinstance(i, FibsemImage) for i in pushed)


def test_a_disabled_beam_does_not_image(microscope):
    """A column switched off has no device, so it images nothing: it raises, as on
    the other backends, rather than running the Demo's imaging code without one."""
    from copy import deepcopy

    from fibsem.microscopes.device_demo import DemoMicroscope

    system = deepcopy(microscope.system)
    system.ion.enabled = False
    electron_only = DemoMicroscope(system)
    assert BeamType.ION not in electron_only.beams
    for call in (
        lambda: electron_only.acquire_image(beam_type=BeamType.ION),
        lambda: electron_only.last_image(BeamType.ION),
        lambda: electron_only.autocontrast(BeamType.ION),
        lambda: electron_only.auto_focus(BeamType.ION),
    ):
        with pytest.raises(ValueError, match="ION beam is not enabled"):
            call()
    assert isinstance(
        electron_only.acquire_image(beam_type=BeamType.ELECTRON), FibsemImage
    )
