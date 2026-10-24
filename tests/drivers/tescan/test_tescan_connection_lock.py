"""Every SharkSEM call in the Tescan driver is serialised on the connection's lock (FIB-786).

SharkSEM is one socket. A long call (a frame transfer) running while another thread
issues any other call interleaves request/response bytes on the shared stream, and
whichever side reads next gets a torn buffer -- observed on the simulator as
``struct.error: unpack requires a buffer of 4 bytes`` inside ``Recv`` when a position
was added while reference images were being acquired.

Two guards here:

* an AST rule: every SDK call site in the driver's connection, devices and services must
  sit lexically inside a ``with`` block on the connection's lock, so a new unlocked call
  site fails this test rather than shipping as a latent race. There are no exceptions.
* a behavioural check: real driver methods hammered from two threads against a fake
  connection that detects overlapping SDK entry.
"""

import ast
import inspect
import threading
import time

import pytest

from fibsem.drivers.tescan import connection as tescan_connection
from fibsem.drivers.tescan import devices as tescan_devices
from fibsem.drivers.tescan import microscope as tescan_microscope
from fibsem.drivers.tescan import sdk
from fibsem.drivers.tescan import services as tescan_services
from fibsem.drivers.tescan.connection import TescanConnection
from fibsem.drivers.tescan.microscope import TescanMicroscope
from fibsem.drivers.tescan.services import TescanMilling
from fibsem.structures.settings import MillingState

# ---------------------------------------------------------------------------
# AST rule
# ---------------------------------------------------------------------------


def _chain_of(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return node, list(reversed(parts))


def _is_sdk_call(func_node) -> bool:
    """A Call whose func is an attribute chain reaching the SharkSEM connection."""
    root, chain = _chain_of(func_node)
    if isinstance(root, ast.Name):
        # connection.sdk.X(...), self.connection.sdk.X(...), or the connection's own
        # self.sdk.X(...)
        if "sdk" in chain[:2]:
            return True
        # a device's vendor beam: self._beam.X.Y(...)
        if root.id == "self" and chain[:1] == ["_beam"] and len(chain) > 1:
            return True
        # beam = connection.beam(...); beam.X.Y(...)
        if root.id in ("beam", "sem", "fib") and chain:
            return True
    return False


def _is_lock(node) -> bool:
    """``self.lock``, ``self._lock``, ``connection.lock``, ``self.connection.lock``."""
    root, chain = _chain_of(node)
    return isinstance(root, ast.Name) and chain[-1:] in (["lock"], ["_lock"])


class _LockAudit(ast.NodeVisitor):
    def __init__(self):
        self.violations = []
        self._func_stack = []
        self._lock_depth = 0

    def visit_FunctionDef(self, node):
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_With(self, node):
        is_lock = any(_is_lock(item.context_expr) for item in node.items)
        if is_lock:
            self._lock_depth += 1
        self.generic_visit(node)
        if is_lock:
            self._lock_depth -= 1

    def visit_Call(self, node):
        if _is_sdk_call(node.func) and self._lock_depth == 0:
            func = self._func_stack[-1] if self._func_stack else "<module>"
            _, chain = _chain_of(node.func)
            self.violations.append(f"{func}:{node.lineno} {'.'.join(chain)}")
        self.generic_visit(node)


@pytest.mark.parametrize(
    "module",
    [tescan_microscope, tescan_connection, tescan_devices, tescan_services],
    ids=lambda m: m.__name__,
)
def test_every_sdk_call_site_is_under_the_connection_lock(module):
    tree = ast.parse(inspect.getsource(module))
    audit = _LockAudit()
    audit.visit(tree)
    assert audit.violations == [], (
        "SharkSEM call sites outside `with connection.lock:` "
        "(one socket -- unlocked calls tear the protocol):\n  "
        + "\n  ".join(audit.violations)
    )


# ---------------------------------------------------------------------------
# behavioural: two threads, one fake connection, no overlapping SDK entry
# ---------------------------------------------------------------------------


class OverlapDetector:
    def __init__(self):
        self._busy = threading.Lock()
        self.overlaps = 0

    def __enter__(self):
        if not self._busy.acquire(blocking=False):
            self.overlaps += 1
            self._acquired = False
        else:
            self._acquired = True
        time.sleep(0.005)  # widen the race window: a real SDK call is not instant
        return self

    def __exit__(self, *exc):
        if self._acquired:
            self._busy.release()


class RacingDrawBeam:
    """Every method body runs inside the shared overlap detector."""

    def __init__(self, detector: OverlapDetector):
        self._detector = detector

    def GetStatus(self):
        with self._detector:
            return ("IDLE", 1.0, 0.0)

    def UnloadLayer(self):
        with self._detector:
            return None

    def EstimateTime(self):
        with self._detector:
            return 1.0


def make_microscope(monkeypatch):
    m = object.__new__(TescanMicroscope)
    detector = OverlapDetector()

    class Conn:
        DrawBeam = RacingDrawBeam(detector)

    m._connection = TescanConnection(Conn(), "localhost", 8300)
    # the status mapping is defined inside the SDK-gated import block, absent
    # without tescanautomation installed (same pattern as the spot-burn tests)
    monkeypatch.setattr(
        sdk,
        "DrawBeamStatusToPatterningState",
        {"IDLE": MillingState.IDLE},
        raising=False,
    )
    m.milling = TescanMilling(parent=m, connection=m._connection)
    return m, detector


def test_concurrent_driver_calls_never_overlap_on_the_connection(monkeypatch):
    m, detector = make_microscope(monkeypatch)
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            m.milling.read_state()
            m.milling._clear()
            m.milling._estimate()

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        assert m.milling.read_state() is MillingState.IDLE
        m.milling._estimate()
    stop.set()
    thread.join(timeout=5)

    assert detector.overlaps == 0, (
        f"{detector.overlaps} overlapping SDK calls -- the connection lock is not "
        "covering every call site"
    )


def test_the_detector_itself_catches_overlap():
    """Prove the harness can see the failure it guards against: two bare threads
    hitting the fake connection without the driver's lock must overlap."""
    detector = OverlapDetector()
    draw_beam = RacingDrawBeam(detector)
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            draw_beam.GetStatus()

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and detector.overlaps == 0:
        draw_beam.GetStatus()
    stop.set()
    thread.join(timeout=5)

    assert detector.overlaps > 0
