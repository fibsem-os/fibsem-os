"""The FM and the beams share one connection, so an FM read must hand it back (FIB-517).

`ThermoFisherFluorescenceMicroscope.set_active_channel()` points the shared connection at
the FM. A property getter that called it and walked away left the microscope there --
and since the FM overview's info bar read the objective on every stage poll, a beam
acquisition that had set its own channel found the FM instead. It stopped a workflow task.

`autoscript.py` cannot be imported without the AutoScript SDK, which is not installed off
the microscope, so the context manager is exercised against a stub connection through a
stub of the class rather than the real import. What is under test is the contract --
capture, set, restore, restore-on-failure -- not the SDK.
"""
from contextlib import contextmanager

import pytest


class _Imaging:
    """The half of the connection that owns the active view."""

    def __init__(self, view: int = 1):
        self.view = view
        self.history: list = []

    def get_active_view(self) -> int:
        return self.view

    def set_active_view(self, view: int) -> None:
        self.view = view
        self.history.append(("view", view))

    def set_active_device(self, device) -> None:
        self.history.append(("device", device))


class _Connection:
    def __init__(self, view: int = 1):
        self.imaging = _Imaging(view)


class _FM:
    """`ThermoFisherFluorescenceMicroscope`'s channel handling, without the SDK import.

    Copied rather than imported: `fibsem/fm/autoscript.py` imports
    `autoscript_sdb_microscope_client` at module scope, which is absent off the
    microscope. The structural test below pins this against the real source so the two
    cannot drift.
    """

    FM_VIEW = 3
    FM_DEVICE = "FLM"

    def __init__(self, view: int = 1):
        import threading

        self.connection = _Connection(view)
        self._active_view = self.FM_VIEW
        self._active_device = self.FM_DEVICE
        self._channel_lock = threading.RLock()
        self._channel_depth = 0
        self._restore_view = None

    def set_active_channel(self):
        self.connection.imaging.set_active_view(self._active_view)
        self.connection.imaging.set_active_device(self._active_device)

    @contextmanager
    def active_channel(self):
        with self._channel_lock:
            if self._channel_depth == 0:
                self._restore_view = self.connection.imaging.get_active_view()
            self.set_active_channel()
            self._channel_depth += 1
        try:
            yield
        finally:
            with self._channel_lock:
                self._channel_depth -= 1
                if self._channel_depth == 0:
                    self.connection.imaging.set_active_view(self._restore_view)


class TestTheContract:
    def test_the_block_runs_on_the_fm(self, ):
        fm = _FM(view=1)  # the beam side owns it

        with fm.active_channel():
            assert fm.connection.imaging.view == _FM.FM_VIEW

    def test_it_hands_the_view_back(self):
        """The whole point. Leaving it on the FM is what broke a running beam task."""
        fm = _FM(view=1)

        with fm.active_channel():
            pass

        assert fm.connection.imaging.view == 1

    def test_it_hands_it_back_when_the_read_fails(self):
        """A getter that raises must not leave the microscope pointed elsewhere -- an
        FM that is off, or answering slowly, would otherwise strand the beam side."""
        fm = _FM(view=1)

        with pytest.raises(RuntimeError):
            with fm.active_channel():
                raise RuntimeError("detector did not answer")

        assert fm.connection.imaging.view == 1

    def test_it_restores_whatever_was_there(self):
        """Not a hardcoded beam view: it puts back what it found."""
        for view in (0, 1, 2, 7):
            fm = _FM(view=view)
            with fm.active_channel():
                pass
            assert fm.connection.imaging.view == view

    def test_only_the_outermost_scope_restores(self):
        """A tileset holds the channel for the whole run and each tile opens a scope
        inside it. An inner scope that restored would put the beam view back between
        every pair of tiles -- the flicker the run-level scope exists to avoid."""
        fm = _FM(view=1)

        with fm.active_channel():
            with fm.active_channel():
                assert fm.connection.imaging.view == _FM.FM_VIEW
            assert fm.connection.imaging.view == _FM.FM_VIEW, "inner scope restored"

        assert fm.connection.imaging.view == 1

    def test_the_lock_is_not_held_across_the_body(self):
        """The scope can span a whole tileset, and the lock it takes is
        `FibsemMicroscope._threading_lock` -- a class attribute, shared by every caller
        in the process. Holding it for minutes would block them all; the known ones are
        a Pause/Resume click and the milling monitor loop, neither of which should
        overlap an FM run, but the point is that a shared lock held that long makes any
        future caller a hostage.
        """
        import threading

        fm = _FM(view=1)
        acquired = threading.Event()

        def take_the_lock():
            with fm._channel_lock:
                acquired.set()

        with fm.active_channel():
            other = threading.Thread(target=take_the_lock)
            other.start()
            other.join(timeout=2)

        assert acquired.is_set(), "another thread could not take the lock mid-scope"

    def test_a_failed_entry_does_not_poison_every_later_scope(self):
        """Entering is two RPCs, and a dropped connection can fail the second.

        That raises out of `__enter__`, so the caller's block never runs and the
        `finally` never runs either. Counting the scope before the channel was actually
        taken left the depth permanently too high -- every later scope then looked
        nested, nothing ever restored the view again, and FIB-517 was back for the rest
        of the session with nothing on screen to say so.
        """
        fm = _FM(view=1)

        def connection_reset(device):
            raise RuntimeError("connection reset")

        fm.connection.imaging.set_active_device = connection_reset
        with pytest.raises(RuntimeError):
            with fm.active_channel():
                pass

        assert fm._channel_depth == 0, "a failed entry left a scope counted as open"

        # And the next healthy scope still hands the view back.
        fm.connection.imaging.set_active_device = lambda device: None
        fm.connection.imaging.view = 1
        with fm.active_channel():
            pass

        assert fm.connection.imaging.view == 1

    def test_an_inner_scope_that_fails_leaves_the_outer_one_intact(self):
        """The same failure one level down must not decrement the run's own scope: a
        tileset still owns the channel, and its restore is still the outermost one."""
        fm = _FM(view=1)

        with fm.active_channel():
            fm.connection.imaging.set_active_device = self._raises
            with pytest.raises(RuntimeError):
                with fm.active_channel():
                    pass
            assert fm._channel_depth == 1
            fm.connection.imaging.set_active_device = lambda device: None

        assert fm.connection.imaging.view == 1

    @staticmethod
    def _raises(device):
        raise RuntimeError("connection reset")

    def test_set_active_channel_alone_does_not_restore(self):
        """The unscoped call still exists, for the live stream that owns the channel for
        its duration. This is what every other caller must not use."""
        fm = _FM(view=1)

        fm.set_active_channel()

        assert fm.connection.imaging.view == _FM.FM_VIEW


class TestEveryMetadataReadIsInsideTheScope:
    """`acquire_image` must cover `_construct_image` too, not just the exposure.

    `_construct_image` reads like formatting, but it calls `get_metadata`, which reads
    the exposure, gain, binning, power, emission wavelength and objective position back
    off the device. On the TFS driver each of those takes the shared channel on its own
    account. Left outside the scope they are 14 separate capture-set-restore cycles per
    image -- 56 round trips, and the microscope's active view changing 28 times -- and
    the metadata would describe the state after the channel went back rather than the
    state the frame was taken under.

    Behavioural rather than structural: `fibsem.fm.microscope` imports without the SDK,
    so the real `acquire_image` can be run against the demo FM and asked what it did.
    """

    @staticmethod
    def _watch(fm, monkeypatch) -> list:
        """Record the scope depth at each device read. Returns the log."""
        from contextlib import contextmanager

        depth = [0]
        reads: list = []

        @contextmanager
        def counting_scope():
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1

        monkeypatch.setattr(fm, "active_channel", counting_scope)

        # binning stands in for the whole set: `get_metadata` reaches it more than any
        # other, directly and through pixel_size and resolution.
        camera = type(fm.camera)
        original = camera.binning

        def fget(self):
            reads.append(depth[0])
            return original.fget(self)

        monkeypatch.setattr(camera, "binning", property(fget, original.fset))
        return reads

    @pytest.fixture
    def fm(self):
        from fibsem import utils

        microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
        return microscope.fm

    def test_the_metadata_is_read_with_the_channel_still_held(self, fm, monkeypatch):
        reads = self._watch(fm, monkeypatch)

        fm.acquire_image()

        assert reads, "no device read was observed -- the probe missed"
        assert all(d > 0 for d in reads), (
            f"{sum(1 for d in reads if d == 0)} of {len(reads)} device reads happened "
            f"outside the channel scope"
        )


class TestTheRealDriverMatches:
    """Structural, since the SDK import cannot be satisfied here.

    Pins that every place in the driver that needs the FM channel goes through the
    scoped form -- with the live stream, which owns it deliberately, as the exception.
    """

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        import fibsem.fm as fm_package

        return (Path(fm_package.__file__).parent / "autoscript.py").read_text(encoding="utf-8")

    @staticmethod
    def _functions(source: str):
        import ast

        return {
            node.name: node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        }

    def test_the_context_manager_captures_and_restores_the_view(self):
        import ast

        node = self._functions(self._source())["active_channel"]
        body = ast.dump(node)
        assert "get_active_view" in body, "does not capture the view"
        assert "set_active_view" in body, "does not restore the view"
        assert any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(node)), (
            "restores outside a finally, so a failing read strands the connection"
        )

    def test_only_the_live_stream_sets_the_channel_unscoped(self):
        """Everything discrete hands the connection back. `_start_fast_acquisition`,
        `start_acquisition` and `stop_acquisition` are the live stream, which holds it
        for as long as it runs and re-forces it per frame."""
        import ast

        source = self._source()
        allowed = {
            "set_active_channel",  # the primitive itself
            "active_channel",  # the scoped form, which calls it
            "fm_settings",  # every caller of this is inside a scope
            "_start_fast_acquisition",
            "start_acquisition",
            "stop_acquisition",
            # every path here comes through `FluorescenceMicroscope.acquire_image`,
            # which scopes, and a tileset scopes the whole run
            "acquire_image",
        }
        offenders = sorted(
            name
            for name, node in self._functions(source).items()
            if name not in allowed
            and "set_active_channel"
            in {
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
            }
        )
        assert offenders == [], (
            f"these take the shared channel without handing it back: {offenders}"
        )

    def test_the_bookkeeping_is_locked_but_the_body_is_not(self):
        """Both halves matter. Unlocked, two scopes interleave and leave the view on the
        FM. Locked across the body, a run-length scope holds a process-wide lock for
        minutes and blocks everything else that takes it."""
        import ast

        node = self._functions(self._source())["active_channel"]
        locked = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.With) and "channel_lock" in ast.dump(child)
        ]
        assert locked, "the bookkeeping is not locked"
        assert not any(
            isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Yield)
            for block in locked
            for inner in ast.walk(block)
        ), "the lock is held across the body"

    def test_the_scope_is_counted_only_after_the_channel_is_taken(self):
        """Order matters, and only in one direction.

        Entering is two RPCs. If the depth is incremented first and the second RPC then
        fails, the exception comes out of `__enter__`, no `finally` ever runs, and the
        count stays high forever -- after which every scope looks nested and nothing
        restores the view again. Incrementing last costs nothing and cannot strand it.
        """
        import ast

        node = self._functions(self._source())["active_channel"]
        entry = next(
            child
            for child in ast.walk(node)
            if isinstance(child, ast.With) and "set_active_channel" in ast.dump(child)
        )

        def index_of(predicate) -> int:
            return next(i for i, stmt in enumerate(entry.body) if predicate(stmt))

        took_it = index_of(
            lambda stmt: any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "set_active_channel"
                for n in ast.walk(stmt)
            )
        )
        counted_it = index_of(
            lambda stmt: isinstance(stmt, ast.AugAssign)
            and "_channel_depth" in ast.dump(stmt.target)
        )
        assert took_it < counted_it, (
            "the scope is counted before the channel is taken, so a connection that "
            "drops mid-entry leaves the depth permanently too high"
        )

    def test_the_objective_state_getter_is_scoped(self):
        """The one that stopped a workflow task, named so a regression is legible."""
        import ast

        node = self._functions(self._source())["state"]
        assert any(
            isinstance(child, ast.With)
            and "active_channel" in ast.dump(child.items[0].context_expr)
            for child in ast.walk(node)
        )
