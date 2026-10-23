"""A remote beam's live view: it runs on the server, and its frames come on the
beam's live socket (``WS /devices/{beam}/live``), the latest only.

Skipped when the [server] extra is not installed.
"""

import asyncio
import os
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

from websockets.exceptions import InvalidStatus  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from fibsem.client.remote_microscope import RemoteMicroscope  # noqa: E402
from fibsem.drivers.remote.devices import connect_remote_beams  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.devices import (  # noqa: E402
    WS_PROTOCOL,
    DeviceServer,
    demo_devices,
    stream_live,
)
from fibsem.structures import BeamType, FibsemImage, ImageSettings  # noqa: E402

TOKEN = "test-token"


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture(scope="module", autouse=True)
def no_delay():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    yield
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


def _frames(beam):
    frames = []
    beam.live_frame.connect(frames.append)
    return frames


def test_a_remote_beam_goes_live_on_a_device_server():
    server = DeviceServer(demo_devices()).start()
    try:
        beams = connect_remote_beams("127.0.0.1", server.port)
        electron = beams[BeamType.ELECTRON]
        frames = _frames(electron)
        try:
            electron.start_live()
            assert wait_for(lambda: len(frames) >= 2)
            assert electron.live.get_value()  # live on the server
        finally:
            electron.stop_live()
            electron.client.close()
        image = frames[0]
        assert isinstance(image, FibsemImage) and image.data.ndim == 2
        assert image.metadata.image_settings.beam_type is BeamType.ELECTRON
    finally:
        server.stop()


@pytest.fixture(scope="module")
def auth():
    return AuthConfig.generate(level="drive", token=TOKEN)


@pytest.fixture(scope="module")
def url(microscope, auth):
    app = build_server(microscope, auth=auth)
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning", ws=WS_PROTOCOL
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "App Server failed to start"
        threading.Event().wait(0.01)
    yield f"127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def remote(url):
    remote = RemoteMicroscope.connect(f"http://{url}", TOKEN)
    yield remote
    remote.disconnect()


def test_live_view_off_the_pc_runs_on_the_microscope(remote, microscope):
    ion = remote.beams[BeamType.ION]
    frames = _frames(ion)
    try:
        ion.start_live()
        assert wait_for(lambda: len(frames) >= 2)
        assert microscope.beams[BeamType.ION].is_live
    finally:
        ion.stop_live()
    assert wait_for(lambda: not microscope.beams[BeamType.ION].is_live)


def test_a_live_view_already_running_is_followed_and_left_running(remote, microscope):
    local = microscope.beams[BeamType.ELECTRON]
    electron = remote.beams[BeamType.ELECTRON]
    local.start_live()  # the operator's
    try:
        frames = _frames(electron)
        electron.start_live()
        assert wait_for(lambda: len(frames) >= 2)
        electron.stop_live()
        assert local.is_live
    finally:
        local.stop_live()


def test_the_live_socket_needs_the_token(url):
    with pytest.raises(InvalidStatus) as refused:
        connect(f"ws://{url}/microscope/devices/electron/live", open_timeout=5)
    assert refused.value.response.status_code == 401


def test_the_live_socket_refuses_a_browser_page(url):
    with pytest.raises(InvalidStatus) as refused:
        connect(
            f"ws://{url}/microscope/devices/electron/live",
            additional_headers={"Authorization": f"Bearer {TOKEN}"},
            origin="http://example.com",
            open_timeout=5,
        )
    assert refused.value.response.status_code == 403


def test_only_a_beam_has_a_live_socket(url):
    with pytest.raises(InvalidStatus) as refused:
        connect(
            f"ws://{url}/microscope/devices/stage/live",
            additional_headers={"Authorization": f"Bearer {TOKEN}"},
            open_timeout=5,
        )
    assert refused.value.response.status_code == 404


def test_a_slow_client_gets_the_latest_frame_not_a_queue(microscope):
    """Frames that arrive while one is sending replace each other: the next sent
    is the newest."""
    from fibsem.drivers.remote.devices import _live_image

    beam = microscope.beams[BeamType.ELECTRON]
    images = [
        beam.acquire(
            ImageSettings(beam_type=BeamType.ELECTRON, resolution=(32, 24), hfw=hfw)
        )
        for hfw in (100e-6, 110e-6, 120e-6, 130e-6)
    ]

    class Slow:
        """A websocket whose first send waits until released."""

        def __init__(self):
            self.sent = []
            self.release = asyncio.Event()

        async def accept(self):
            pass

        async def receive_text(self):
            await asyncio.Event().wait()

        async def send_bytes(self, data):
            if not self.sent:
                self.sent.append(data)
                await self.release.wait()
            else:
                self.sent.append(data)

        async def close(self, code):
            pass

    slow = Slow()

    async def run():
        streaming = asyncio.ensure_future(stream_live(slow, beam))
        await asyncio.sleep(0.05)
        beam.live_frame.emit(images[0])
        await asyncio.sleep(0.05)  # now sending the first
        for image in images[1:]:
            beam.live_frame.emit(image)
        await asyncio.sleep(0.05)
        slow.release.set()
        await asyncio.sleep(0.1)
        streaming.cancel()

    asyncio.run(run())
    hfws = [_live_image(data).metadata.image_settings.hfw for data in slow.sent]
    assert hfws == [pytest.approx(100e-6), pytest.approx(130e-6)]
