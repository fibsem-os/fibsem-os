"""Nothing fibsem sends odemis carries a numpy value (FIB-1367).

The odemis back end unpickles every request with its own numpy, 1.x on Ubuntu 22.04,
which cannot read a numpy 2 pickle. On the odemis METEOR simulator a VA write of
``np.float64(0.1)`` raised ``ModuleNotFoundError: No module named 'numpy._core'`` and a
``moveRel`` with one never completed, while plain floats went through.
"""

from collections import namedtuple

import numpy as np

from fibsem.drivers.odemis.connection import OdemisConnection, PlainCalls, plain, raw


def _has_numpy(value):
    if isinstance(value, (np.generic, np.ndarray)):
        return True
    if isinstance(value, dict):
        return any(_has_numpy(k) or _has_numpy(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_numpy(v) for v in value)
    return False


class _VA:
    def __init__(self, value):
        self.value = value

    def subscribe(self, listener, init=False):
        self.listener = listener


class _Component:
    """Records what reaches it, as the back end would receive it."""

    def __init__(self):
        self.calls = []
        self.exposureTime = _VA(0.1)

    def moveRel(self, shift):
        self.calls.append(("moveRel", shift))

    def set_beam_shift(self, x, y, channel=None):
        self.calls.append(("set_beam_shift", x, y, channel))


def test_plain_turns_numpy_into_python_all_the_way_down():
    value = {
        "z": np.float64(1e-3),
        "binning": (np.int64(2), np.int64(2)),
        "bands": [np.array([4.14e-7, 4.5e-7])],
    }

    out = plain(value)

    assert out == {"z": 1e-3, "binning": (2, 2), "bands": [[4.14e-7, 4.5e-7]]}
    assert not _has_numpy(out)
    assert type(out["z"]) is float and type(out["binning"]) is tuple


def test_plain_leaves_other_values_alone():
    Point = namedtuple("Point", "x y")
    point, marker = Point(1, 2), object()

    assert plain(point) is point
    assert plain(marker) is marker
    assert plain("SecondaryElectrons") == "SecondaryElectrons"


def test_calls_and_va_writes_go_out_plain():
    component = _Component()
    wrapped = PlainCalls(component)

    wrapped.moveRel({"z": np.float64(1e-6)})
    wrapped.set_beam_shift(np.float64(1e-6), np.float32(-2e-6), channel=np.int64(1))
    wrapped.exposureTime.value = np.float64(0.2)

    assert not _has_numpy(component.calls)
    assert type(component.exposureTime.value) is float
    assert component.calls[0] == ("moveRel", {"z": 1e-6})


def test_reads_and_other_attributes_pass_through():
    component = _Component()
    wrapped = PlainCalls(component)
    frame = np.zeros((2, 2))

    wrapped.last_image = frame  # a test configuring a stub, not a VA write
    assert component.last_image is frame
    assert wrapped.exposureTime.value == 0.1
    assert raw(wrapped) is component
    assert raw(component) is component


def test_the_connection_wraps_the_client_and_the_stage():
    client, stage = _Component(), _Component()
    connection = OdemisConnection(client, stage, lock=None)

    connection.vendor_stage.moveRel({"x": np.float64(5e-5)})

    assert raw(connection.sdk) is client
    assert stage.calls == [("moveRel", {"x": 5e-5})]
    assert not _has_numpy(stage.calls)


class _RemoteMethod:
    """As Pyro's: every attribute, ``__name__`` included, is another remote method."""

    def __init__(self, sent):
        self.sent = sent

    def __getattr__(self, name):
        return _RemoteMethod(self.sent)

    def __call__(self, *args):
        self.sent.append(args)


class _Proxy:
    def __init__(self):
        self.sent = []
        self.set_resolution = _RemoteMethod(self.sent)


def test_a_pyro_remote_method_is_called_through():
    proxy = _Proxy()

    PlainCalls(proxy).set_resolution((np.int64(1536), np.int64(1024)), "electron")

    assert proxy.sent == [((1536, 1024), "electron")]
    assert not _has_numpy(proxy.sent)
