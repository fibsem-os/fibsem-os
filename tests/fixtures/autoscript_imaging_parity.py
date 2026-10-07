"""Record the AutoScript calls of Thermo's imaging methods, through the beams.

Run as a script, in its own interpreter, for the same reason as
``autoscript_beam_parity.py``, whose fake SDK, microscope and recorder it reuses. It
writes JSON to the path it is given: ``cases``, each holding what an imaging method
returned, the SDK calls and writes it made (the grab, the image read and the
autofunctions included) and the messages it logged, on the microscope with its beams
built as connect builds them (the old methods' are in ``autoscript_old_calls.json``,
recorded over this fake before they were deleted); and ``facts``, what the new API
makes of the commands.

``get_microscope_state`` and ``_set_additional_metadata`` are recorded rather than
run: both sides call the same ones, and what they read is the microscope's, not the
imaging method's.
"""

import copy
import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autoscript_beam_parity as B  # noqa: E402  (installs the fake SDK)
import numpy as np  # noqa: E402

import fibsem.utils  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemRectangle,
    ImageSettings,
    MicroscopeState,
)

S, LOG = B.S, B.LOG

# last_image names its settings by the time; one name, so both sides agree
fibsem.utils.current_timestamp = lambda: "now"


class FakeAdornedImage:
    """The SDK's AdornedImage, as much of it as the conversion reads."""

    def __init__(self, data=None, metadata=None):
        self.data = data
        self.metadata = metadata

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]


S.STRUCTS.AdornedImage = S.A.AdornedImage = FakeAdornedImage


def _adorned(shape, dtype):
    metadata = types.SimpleNamespace(
        acquisition=types.SimpleNamespace(acquisition_datetime="2026-10-05 06:00:00"),
        binary_result=types.SimpleNamespace(
            pixel_size=types.SimpleNamespace(x=1e-8, y=1e-8)
        ),
        scan_settings=types.SimpleNamespace(dwell_time=3e-7),
    )
    return FakeAdornedImage(np.full(shape, 7, dtype=dtype), metadata)


class FakeImaging(S.Node):
    """``connection.imaging``: records calls, and the grab and the read return frames."""

    def grab_frame(self, settings=None):
        LOG.append(["call", f"{self._path}.grab_frame", [S._plain(settings)], "{}"])
        return _adorned((4, 6), np.uint8)

    def get_image(self, settings=None):
        LOG.append(["call", f"{self._path}.get_image", [S._plain(settings)], "{}"])
        return _adorned((4, 6), np.uint16)

    @property
    def state(self):
        """Acquiring for as many reads as ``_acquiring`` says, then idle."""
        left = self.__dict__.get("_acquiring", 0)
        object.__setattr__(self, "_acquiring", max(left - 1, 0))
        LOG.append(["get", f"{self._path}.state"])
        return "ACQUIRING" if left > 0 else "Idle"


def make():
    microscope = B.make(plasma=False)
    object.__setattr__(
        microscope.connection, "imaging", FakeImaging("connection.imaging")
    )
    microscope._last_imaging_settings = ImageSettings(path="/data", filename="last")

    def get_microscope_state(beam_type=None):
        LOG.append(["state", beam_type.name if beam_type else None])
        return MicroscopeState(timestamp=0.0)

    def set_additional_metadata(image):
        LOG.append(["metadata", list(image.data.shape)])

    microscope.get_microscope_state = get_microscope_state
    microscope._set_additional_metadata = set_additional_metadata
    microscope._build_beams()
    return microscope


def _image(image):
    """An image as plain data: its pixels, settings and state."""
    metadata = image.metadata
    return {
        "shape": list(image.data.shape),
        "dtype": str(image.data.dtype),
        "settings": metadata.image_settings.to_dict(),
        "pixel_size": [metadata.pixel_size.x, metadata.pixel_size.y],
        "timestamp": metadata.microscope_state.timestamp,
    }


def _settings(beam_type, reduced=False):
    return ImageSettings(
        beam_type=beam_type,
        resolution=(768, 512),
        dwell_time=2e-7,
        hfw=80e-6,
        reduced_area=FibsemRectangle(0.1, 0.2, 0.3, 0.4) if reduced else None,
        line_integration=2,
        frame_integration=None,
        path="/data",
        filename="img",
    )


AREA = FibsemRectangle(0.25, 0.25, 0.5, 0.5)


def _calls(beam_type):
    return (
        ("acquire settings", lambda m: _image(m.acquire_image(_settings(beam_type)))),
        (
            "acquire reduced",
            lambda m: _image(m.acquire_image(_settings(beam_type, reduced=True))),
        ),
        ("acquire current", lambda m: _image(m.acquire_image(beam_type=beam_type))),
        (
            "acquire both",  # a beam type wins: current settings, as before
            lambda m: _image(
                m.acquire_image(_settings(beam_type), beam_type=beam_type)
            ),
        ),
        (
            "acquire last settings",
            lambda m: (
                m.acquire_image(_settings(beam_type)),
                m._last_imaging_settings.filename,
            )[1],
        ),
        ("last image", lambda m: _image(m.last_image(beam_type))),
        ("autocontrast", lambda m: m.autocontrast(beam_type)),
        ("autocontrast area", lambda m: m.autocontrast(beam_type, AREA)),
        ("auto focus", lambda m: m.auto_focus(beam_type)),
        ("auto focus area", lambda m: m.auto_focus(beam_type, reduced_area=AREA)),
    )


def cases():
    out = []
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        for name, call in _calls(beam_type):
            new = make()
            out.append(
                {"key": f"{beam_type.name} {name}", "new": B.run(lambda: call(new))}
            )
    # neither settings nor a beam type: the old error
    new = make()
    out.append({"key": "acquire nothing", "new": B.run(lambda: new.acquire_image())})
    return out


def _live(microscope, beam_type, frames=4):
    """Live view on *beam_type* until the old signal has had *frames* images: two
    from the fast path, then one per pass of the loop (a fast path that finds the
    imaging idle, then a grab)."""
    object.__setattr__(microscope.connection.imaging, "_acquiring", 2)
    seen, done = [], threading.Event()
    beam = microscope.beams.get(beam_type)
    stop = microscope._stop_acquisition_event if beam is None else beam._live_stop

    def on_frame(image):
        seen.append([list(image.data.shape), str(image.data.dtype)])
        if len(seen) == frames:
            stop.set()
            done.set()

    signal = (
        microscope.sem_acquisition_signal
        if beam_type is BeamType.ELECTRON
        else microscope.fib_acquisition_signal
    )
    signal.connect(on_frame)

    def call():
        microscope.start_acquisition(beam_type)
        assert done.wait(10), "live view never reached its frames"
        threads = [microscope._acquisition_thread] + [
            b._live_thread for b in microscope.beams.values()
        ]
        for thread in threads:
            if thread is not None:
                thread.join(10)
        return [seen, bool(microscope.is_acquiring)]

    return call


def live_cases():
    out = []
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        new = make()
        out.append(
            {"key": f"{beam_type.name} live", "new": B.run(_live(new, beam_type))}
        )
    return out


def facts():
    out = {}
    microscope = make()
    out["commands"] = {
        bt.name: sorted(name for name, info in beam.commands.items() if info.available)
        for bt, beam in microscope.beams.items()
    }

    # the old methods go through the beam's commands
    microscope = make()
    used = []
    for beam in microscope.beams.values():
        for command in ("acquire", "last_image", "autocontrast", "auto_focus"):
            original = getattr(beam, command)

            def wrapper(*args, _name=f"{beam.name}.{command}", _f=original, **kw):
                used.append(_name)
                return _f(*args, **kw)

            setattr(beam, command, wrapper)
    microscope.acquire_image(_settings(BeamType.ELECTRON))
    microscope.acquire_image(beam_type=BeamType.ION)
    microscope.last_image(BeamType.ION)
    microscope.autocontrast(BeamType.ELECTRON)
    microscope.auto_focus(BeamType.ION)
    out["used"] = list(used)
    microscope = make()

    # each vendor call runs with the beam's channel selected under the lock
    held = []
    imaging = microscope.connection.imaging
    auto = microscope.connection.auto_functions

    def recording(path):
        def call(*args, **kwargs):
            held.append([path, microscope._threading_lock._is_owned()])
            return _adorned((2, 2), np.uint8)

        return call

    object.__setattr__(imaging, "grab_frame", recording("grab_frame"))
    object.__setattr__(imaging, "get_image", recording("get_image"))
    object.__setattr__(auto, "run_auto_cb", recording("run_auto_cb"))
    object.__setattr__(auto, "run_auto_focus", recording("run_auto_focus"))
    beam = microscope.beams[BeamType.ION]
    beam.acquire()
    beam.acquire(_settings(BeamType.ION))
    beam.last_image()
    beam.autocontrast()
    beam.auto_focus(AREA)
    out["held"] = held

    # live view through the microscope: the beam is live, and stop stops it
    microscope = make()
    object.__setattr__(microscope.connection.imaging, "_acquiring", 10**9)
    sem = microscope.beams[BeamType.ELECTRON]
    microscope.start_acquisition(BeamType.ELECTRON)
    live = {"live": sem.is_live, "acquiring": bool(microscope.is_acquiring)}
    microscope.start_acquisition(BeamType.ION)  # already acquiring: warns, no-op
    live["ion"] = microscope.beams[BeamType.ION].is_live
    microscope.stop_acquisition()
    live["stopped"] = [sem.is_live, bool(microscope.is_acquiring)]
    live["thread"] = microscope._acquisition_thread is None
    out["live"] = live

    # the new API refuses settings for the other beam
    try:
        beam.acquire(_settings(BeamType.ELECTRON))
        out["other_beam"] = None
    except ValueError as e:
        out["other_beam"] = str(e)

    # a disabled column has no beam, so its imaging stays with the old code
    microscope = B.make(plasma=False, ion=False)
    microscope._build_beams()
    out["ion_disabled"] = sorted(bt.name for bt in microscope.beams)
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(
            {"cases": cases() + live_cases(), "facts": copy.deepcopy(facts())},
            f,
            default=str,
        )
