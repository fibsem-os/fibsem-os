"""Every SharkSEM call in the Tescan driver is serialised on _connection_lock (FIB-786).

SharkSEM is one socket. A long call (a frame transfer) running while another thread
issues any other call interleaves request/response bytes on the shared stream, and
whichever side reads next gets a torn buffer -- observed on the simulator as
``struct.error: unpack requires a buffer of 4 bytes`` inside ``Recv`` when a position
was added while reference images were being acquired.

Two guards here:

* an AST rule: every SDK call site in ``tescan.py`` must sit lexically inside a
  ``with self._connection_lock:`` block, so a new unlocked call site fails this test
  rather than shipping as a latent race. ``_get_impl``/``_set_impl`` are the one
  allowed exception -- their public wrappers ``_get``/``_set`` hold the lock.
* a behavioural check: real driver methods hammered from two threads against a fake
  connection that detects overlapping SDK entry.
"""

import ast
import inspect
import threading
import time

from fibsem.microscopes import tescan as tescan_module
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, MillingState

# Their callers (_get/_set) take the lock around the whole dispatch, so these two are
# exempt from the lexical rule. Nothing else may join this list without the same
# always-locked-caller property.
LOCKED_BY_CALLER = {"_get_impl", "_set_impl"}


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
        # self.connection.X.Y(...) has root `self` and chain starting "connection"
        if root.id == "self" and chain and chain[0] == "connection":
            return True
        # beam = self._get_beam(...); beam.X.Y(...)
        if root.id in ("beam", "sem", "fib") and chain:
            return True
    if isinstance(root, ast.Call):
        # inline chains: self._get_beam(...).Preset.Enum(...)
        inner_root, inner_chain = _chain_of(root.func)
        if (
            isinstance(inner_root, ast.Name)
            and inner_root.id == "self"
            and inner_chain == ["_get_beam"]
        ):
            return True
    return False


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
        is_lock = any(
            "_connection_lock" in ast.dump(item.context_expr) for item in node.items
        )
        if is_lock:
            self._lock_depth += 1
        self.generic_visit(node)
        if is_lock:
            self._lock_depth -= 1

    def visit_Call(self, node):
        if _is_sdk_call(node.func) and self._lock_depth == 0:
            func = self._func_stack[-1] if self._func_stack else "<module>"
            if func not in LOCKED_BY_CALLER:
                _, chain = _chain_of(node.func)
                self.violations.append(f"{func}:{node.lineno} {'.'.join(chain)}")
        self.generic_visit(node)


def test_every_sdk_call_site_is_under_the_connection_lock():
    tree = ast.parse(inspect.getsource(tescan_module))
    audit = _LockAudit()
    audit.visit(tree)
    assert audit.violations == [], (
        "SharkSEM call sites outside `with self._connection_lock:` "
        "(one socket -- unlocked calls tear the protocol):\n  "
        + "\n  ".join(audit.violations)
    )


def test_the_allowlist_is_only_the_impl_pair():
    """The exemption exists for _get_impl/_set_impl alone; growing it silently would
    reopen the hole the lexical rule closes."""
    assert LOCKED_BY_CALLER == {"_get_impl", "_set_impl"}


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
    m._connection_lock = threading.RLock()
    detector = OverlapDetector()

    class Conn:
        DrawBeam = RacingDrawBeam(detector)

    m.connection = Conn()
    # the status mapping is defined inside the SDK-gated import block, absent
    # without tescanautomation installed (same pattern as the spot-burn tests)
    monkeypatch.setattr(
        tescan_module,
        "DrawBeamStatusToPatterningState",
        {"IDLE": MillingState.IDLE},
        raising=False,
    )
    return m, detector


def test_concurrent_driver_calls_never_overlap_on_the_connection(monkeypatch):
    m, detector = make_microscope(monkeypatch)
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            m.get_milling_state()
            m.clear_patterns()
            m.estimate_milling_time()

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        assert m.get_milling_state() is MillingState.IDLE
        m.estimate_milling_time()
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
